# kvswitch core: dense-model (Qwen3 / Llama-3.2) forward over a preallocated external KV cache.
# Native B and every translation method share THIS code path; only the cache contents differ.
import os, time, json, math, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.expanduser("~/kvswitch")
_t0 = time.time()


def log(*a):
    print(f"[{time.time() - _t0:7.0f}s]", *a, flush=True)


def rot_half(x):
    x1, x2 = x.chunk(2, -1)
    return torch.cat((-x2, x1), -1)


def rope(x, cos, sin):  # x [..,T,D] fp32, cos/sin [T,D]
    return x * cos + rot_half(x) * sin


def derope(y, cos, sin):  # exact inverse rotation
    return y * cos - rot_half(y) * sin


class Model:
    def __init__(self, name, dev="cuda"):
        self.name = name
        p = f"{ROOT}/models/{name}"
        self.tok = AutoTokenizer.from_pretrained(p)
        m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16).to(dev).eval()
        for q in m.parameters():
            q.requires_grad_(False)
        self.m, self.dev = m, dev
        c = m.config
        self.L, self.Hq, self.Hk = c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads
        self.D = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        self.layers = m.model.layers
        self.qknorm = hasattr(self.layers[0].self_attn, "q_norm")
        self.family = "qwen3" if "Qwen3" in name else "llama"
        self._cs = None

    # ---- RoPE (uses the model's own rotary_emb, so llama3 frequency scaling is honoured) ----
    def cossin(self, n0, n1):
        if self._cs is None or self._cs[0].shape[0] < n1:
            nmax = max(n1, 65536)
            pos = torch.arange(nmax, device=self.dev)
            cos, sin = self.m.model.rotary_emb(torch.zeros(1, device=self.dev, dtype=torch.float32), pos[None])
            self._cs = (cos[0].float(), sin[0].float())
        return self._cs[0][n0:n1], self._cs[1][n0:n1]

    def qkv(self, l, x):  # x [1,T,H] -> pre-RoPE q,k (post q/k-norm), v  [1,H,T,D]
        a = self.layers[l].self_attn
        T = x.shape[1]
        q = a.q_proj(x).view(1, T, self.Hq, self.D)
        k = a.k_proj(x).view(1, T, self.Hk, self.D)
        if self.qknorm:
            q, k = a.q_norm(q), a.k_norm(k)
        return q.transpose(1, 2), k.transpose(1, 2), a.v_proj(x).view(1, T, self.Hk, self.D).transpose(1, 2)

    def new_cache(self, nmax):
        return Cache(self, nmax)

    @torch.no_grad()
    def forward(self, ids, cache, n0=None, want="last", pre_layer=None, collect=None, layers=None):
        """Process ids [1,T] at positions n0..n0+T-1, attending to cache[:n0] (+ causal self).
        Writes the new K(post-RoPE)/V into cache. want: 'last' -> logits of last pos, 'all' -> all logits,
        'hidden' -> final hidden. pre_layer(l): called before layer l touches the cache (C6 overlap hook).
        collect: dict to receive pre-RoPE k, v (and q post-RoPE) per layer (for fitting)."""
        n0 = cache.n if n0 is None else n0
        T = ids.shape[1]
        CH = 4096
        if T > CH and collect is None:  # chunked prefill (bounded activation / mask memory)
            for s in range(0, T - CH, CH):
                self.forward(ids[:, s:s + CH], cache, n0 + s, want="hidden", layers=layers)
            s = ((T - 1) // CH) * CH
            return self.forward(ids[:, s:], cache, n0 + s, want=want, pre_layer=pre_layer, layers=layers)
        cos, sin = self.cossin(n0, n0 + T)
        h = self.m.model.embed_tokens(ids)
        mask = None
        causal = n0 == 0 and T > 1
        if T > 1 and n0 > 0:
            j = torch.arange(n0 + T, device=self.dev)
            mask = j[None, :] <= (n0 + torch.arange(T, device=self.dev))[:, None]
        for l in range(self.L) if layers is None else layers:
            layer = self.layers[l]
            x = layer.input_layernorm(h)
            q, k, v = self.qkv(l, x)
            qp = rope(q.float(), cos, sin).to(q.dtype)
            kp = rope(k.float(), cos, sin).to(k.dtype)
            if collect is not None:
                collect.setdefault("k", []).append(k[0]); collect.setdefault("v", []).append(v[0])
                if collect.get("q"):
                    collect.setdefault("qp", []).append(qp[0])
            if pre_layer is not None:
                pre_layer(l)
            cache.k[l][:, n0:n0 + T] = kp[0]
            cache.v[l][:, n0:n0 + T] = v[0]
            K, V = cache.k[l][None, :, :n0 + T], cache.v[l][None, :, :n0 + T]
            if mask is not None:  # mem-efficient kernel with mask; expand GQA explicitly
                G = self.Hq // self.Hk
                o = F.scaled_dot_product_attention(qp, K.repeat_interleave(G, 1), V.repeat_interleave(G, 1), attn_mask=mask)
            else:
                o = F.scaled_dot_product_attention(qp, K, V, is_causal=causal, enable_gqa=True)
            h = h + layer.self_attn.o_proj(o.transpose(1, 2).reshape(1, T, -1))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        cache.n = max(cache.n, n0 + T)
        if want == "hidden":
            return h
        if want == "last":
            h = h[:, -1:]
        return self.m.lm_head(self.m.model.norm(h)).float()

    @torch.no_grad()
    def generate(self, first_logits, cache, max_new=64, stop=None):
        stop = set(stop or self.stop_ids())
        out = []
        nxt = int(first_logits[0, -1].argmax())
        for _ in range(max_new):
            if nxt in stop:
                break
            out.append(nxt)
            lg = self.forward(torch.tensor([[nxt]], device=self.dev), cache)
            nxt = int(lg[0, -1].argmax())
        return out

    def stop_ids(self):
        s = {self.tok.eos_token_id}
        for t in ["<|im_end|>", "<|eot_id|>", "<|endoftext|>", "<|end_of_text|>"]:
            i = self.tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0 and i != self.tok.unk_token_id:
                s.add(i)
        return [x for x in s if x is not None]

    # ---- chat template (hand-built so that history is an exact token prefix of later prompts) ----
    def turn(self, role, text):
        if self.family == "qwen3":
            s = f"<|im_start|>{role}\n{text}<|im_end|>\n"
        else:
            s = f"<|start_header_id|>{role}<|end_header_id|>\n\n{text}<|eot_id|>"
        return self.tok(s, add_special_tokens=False).input_ids

    def gen_prompt(self):
        s = "<|im_start|>assistant\n<think>\n\n</think>\n\n" if self.family == "qwen3" else \
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return self.tok(s, add_special_tokens=False).input_ids

    def bos(self):
        return [] if self.family == "qwen3" else [self.tok.bos_token_id]

    def conv_ids(self, msgs):
        """msgs: list of (role,text). Returns ids and per-message (start,end) spans."""
        ids, spans = list(self.bos()), []
        for r, t in msgs:
            s = len(ids); ids += self.turn(r, t); spans.append((s, len(ids)))
        return ids, spans


class Cache:
    """Preallocated per-layer K (post-RoPE, in the owner model's RoPE) and V: [Hk, Nmax, D] bf16."""

    def __init__(self, model, nmax, lazy=False):
        self.L, self.n, self.nmax = model.L, 0, nmax
        self.shape, self.dev = (model.Hk, nmax, model.D), model.dev
        self.k, self.v = [None] * model.L, [None] * model.L
        if not lazy:
            for l in range(model.L):
                self.alloc(l)

    def alloc(self, l):
        self.k[l] = torch.empty(self.shape, dtype=torch.bfloat16, device=self.dev)
        self.v[l] = torch.empty(self.shape, dtype=torch.bfloat16, device=self.dev)

    def free_layer(self, l):
        # tell the caching allocator the current (possibly side) stream still has queued reads of these blocks
        for t in (self.k[l], self.v[l]):
            if t is not None:
                t.record_stream(torch.cuda.current_stream())
        self.k[l] = None; self.v[l] = None

    def bytes(self):
        return sum(t.numel() * 2 for t in self.k + self.v if t is not None)

    def truncate(self, n):
        self.n = n


def kv_bytes(model, n):
    return 2 * model.L * model.Hk * model.D * 2 * n


class Timer:
    def __init__(self):
        self.ev = []

    def mark(self, name, stream=None):
        e = torch.cuda.Event(enable_timing=True)
        e.record(stream)
        self.ev.append((name, e))

    def result(self):
        torch.cuda.synchronize()
        out = {}
        for (a, ea), (b, eb) in zip(self.ev, self.ev[1:]):
            out[b] = out.get(b, 0) + ea.elapsed_time(eb)
        out["total"] = self.ev[0][1].elapsed_time(self.ev[-1][1])
        return out
