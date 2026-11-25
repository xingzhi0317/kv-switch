# KL-trained translator, retrained with the published recipe (KV-Lingo arXiv 2609.32610 §3.2/§4.1/App. E; Hybrid-KV
# arXiv 2611.12062 Part 1 §3.3/§3.6/§5).  Replaces train_dense_kl.py, which differed from the papers in: 1 conversation
# per step (papers 8), a random cut with 128 continuation tokens (papers: prompt -> full answer, <= 2048), stage-1 fit on
# FineWeb (papers: chat prefixes), bias not trained (Hybrid-KV: y = W x + b), no LR sweep (KV-Lingo: 5 LRs half a decade
# apart, picked by final val KL on 64 held-out conversations).
#   map     : per target layer, K and V separately, y = (x - mu_x) W + b over the concatenated K (resp. V) of nu = 3
#             source layers, all heads; keys in content space (source RoPE stripped, target RoPE applied); W and b
#             trained (b initialised to the closed-form mu_y), fp32; both models frozen, bf16
#   init    : fit_kl_init.py (chat prefixes, closed form) -> mappers/kvbridge3c_{DIR}
#   loss    : KL(target reading its own prompt cache || target reading the translated prompt cache), per answer token,
#             averaged over the answer, then over the 8 conversations of a step
#   optim   : AdamW, no weight decay, grad clip 1.0, cosine schedule with 5% warmup, 2000 steps (Hybrid-KV Part 1)
#   data    : kl_data.py (Nemotron chat, prompt <= 8K tokens, answer <= 2048 tokens; 64 validation conversations)
# env: DIR LR STEPS BATCH VAL_EVERY TAG     output: mappers/kvbridge3ckl{TAG}_{DIR}, results/train_kl2_{DIR}_{TAG}.json
import os, sys, json, math, time, random
import torch
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
HOME = os.path.expanduser("~")
sys.path.insert(0, f"{HOME}/kvbase/kvbridge")
sys.path.insert(0, f"{HOME}/kvcodesign")
from kvbridge import Mapper
import kl_data
E = os.environ.get
DIR = E("DIR", "17to4")
SRC_M, TGT_M = {"17to4": ("Qwen3-1.7B", "Qwen3-4B"), "4to17": ("Qwen3-4B", "Qwen3-1.7B")}[DIR]
LR, STEPS, BATCH, VAL_EVERY = float(E("LR", 1e-4)), int(E("STEPS", 2000)), int(E("BATCH", 8)), int(E("VAL_EVERY", 250))
TAG = E("TAG", f"lr{E('LR', '1e-4')}")
INIT = E("INIT", f"{HOME}/kvbase/mappers/kvbridge3c_{DIR}")
OUT = f"{HOME}/kvbase/mappers/kvbridge3ckl{TAG}_{DIR}"
RES = f"{HOME}/kvcodesign/results/train_kl2_{DIR}_{TAG}.json"
dev, D, CH = "cuda", 128, 512
t0 = time.time()
log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{(time.time() - t0) / 60:.1f} min]", *a, flush=True)
M = f"{HOME}/kvswitch/models"
tok = AutoTokenizer.from_pretrained(f"{M}/Qwen3-4B")
S = AutoModelForCausalLM.from_pretrained(f"{M}/{SRC_M}", torch_dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)
T = AutoModelForCausalLM.from_pretrained(f"{M}/{TGT_M}", torch_dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)
mp = Mapper.load(INIT, device=dev, dtype=torch.float32)
P = {}
for l, e in enumerate(mp.layers):
    for k in ("W_k", "W_v", "mu_yk", "mu_yv"):
        P[f"{k}{l}"] = torch.nn.Parameter(e[k].float().clone())
inv = 1.0 / (1e6 ** (torch.arange(0, D, 2, device=dev, dtype=torch.float32) / D))
rot = lambda x: torch.cat([-x[..., D // 2:], x[..., :D // 2]], -1)


def translate(sc, n):   # same arithmetic as the official kvbridge.transfer, differentiable in W and b
    ang = torch.arange(n, device=dev, dtype=torch.float32)[:, None] * inv[None]
    c, s_ = torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)
    sk = [(lambda k: (k * c - rot(k) * s_).transpose(0, 1).reshape(n, -1))(sc.layers[l].keys[0].float()) for l in range(len(sc.layers))]
    sv = [sc.layers[l].values[0].float().transpose(0, 1).reshape(n, -1) for l in range(len(sc.layers))]
    out = DynamicCache()
    for t, e in enumerate(mp.layers):   # per target layer under checkpointing: the [n, 3*1024] fp32 inputs are not kept
        ch = e["source_layers"]
        yk, yv = checkpoint(_map_layer, t, n, c, s_, *[sk[s] for s in ch], *[sv[s] for s in ch], use_reentrant=False)             if torch.is_grad_enabled() else _map_layer(t, n, c, s_, *[sk[s] for s in ch], *[sv[s] for s in ch])
        out.update(yk, yv, t)
    return out


def _map_layer(t, n, c, s_, *xs):
    e, k = mp.layers[t], len(xs) // 2
    yk = (torch.cat(xs[:k], -1) - e["mu_xk"]) @ P[f"W_k{t}"] + P[f"mu_yk{t}"]
    yv = (torch.cat(xs[k:], -1) - e["mu_xv"]) @ P[f"W_v{t}"] + P[f"mu_yv{t}"]
    yk = yk.view(n, -1, D).transpose(0, 1); yv = yv.view(n, -1, D).transpose(0, 1)
    yk = yk * c + rot(yk) * s_
    return yk[None].to(torch.bfloat16), yv[None].to(torch.bfloat16)


def chunk_kl(h, ref_h):   # KL summed over a chunk of answer positions, logits never stored for backward
    lp = T.lm_head(h).float().log_softmax(-1)
    ref = T.lm_head(ref_h).float().log_softmax(-1)
    return (ref.exp() * (ref - lp)).sum()


class _Prefix:   # stateless stand-in for a cache: one layer's translated prompt K/V, prepended to the answer's own K/V
    def __init__(self, k, v):
        self.k, self.v = k, v

    def update(self, k, v, layer_idx, cache_kwargs=None):
        return torch.cat([self.k, k], 2), torch.cat([self.v, v], 2)


def _layer(i, h, k, v, mask, pos, cos, sin):
    return T.model.layers[i](h, attention_mask=mask, position_ids=pos, past_key_value=_Prefix(k, v), use_cache=True,
                             cache_position=pos[0], position_embeddings=(cos, sin))


def student_hidden(tc, n, ci, grad):
    """Target decoder over the answer tokens on the translated prompt cache tc (n positions).  Each layer runs under
    activation checkpointing; the cache is passed as a stateless prefix so recomputation in backward is exact.
    Same arithmetic as T.model(ci, past_key_values=tc), with ~25x less activation memory."""
    m = ci.shape[1]
    h = T.model.embed_tokens(ci)
    pos = torch.arange(n, n + m, device=dev)[None]
    cos, sin = T.model.rotary_emb(h, pos)
    mask = (torch.arange(n + m, device=dev)[None] <= pos[0][:, None])[None, None]   # [1, 1, m, n + m]
    for i in range(len(T.model.layers)):
        k, v = tc.layers[i].keys, tc.layers[i].values
        h = checkpoint(_layer, i, h, k, v, mask, pos, cos, sin, use_reentrant=False) if grad else _layer(i, h, k, v, mask, pos, cos, sin)
    return T.model.norm(h)


def kl_loss(p, c, grad=True):
    pi, ci = torch.tensor([p], device=dev), torch.tensor([c], device=dev)
    with torch.no_grad():
        tc = DynamicCache(); T.model(pi, past_key_values=tc, use_cache=True)
        ref_h = T.model(ci, past_key_values=tc, use_cache=True).last_hidden_state[0]
        del tc
        sc = S.model(pi, use_cache=True).past_key_values
    with torch.set_grad_enabled(grad):
        h = student_hidden(translate(sc, len(p)), len(p), ci, grad)[0]
        tot = 0
        for a in range(0, len(c), CH):
            tot = tot + (checkpoint(chunk_kl, h[a:a + CH], ref_h[a:a + CH], use_reentrant=False) if grad else chunk_kl(h[a:a + CH], ref_h[a:a + CH]))
        return tot / len(c)


tok_ = tok
train, val = kl_data.load(tok_)
log("conversations", len(train), "train,", len(val), "val; mean prompt", sum(len(x["p"]) for x in train) // len(train),
    "mean answer", sum(len(x["c"]) for x in train) // len(train))


def validate():
    with torch.no_grad():
        return round(sum(float(kl_loss(x["p"], x["c"], grad=False)) for x in val) / len(val), 4)


def save(step):
    for l, e in enumerate(mp.layers):
        for k in ("W_k", "W_v", "mu_yk", "mu_yv"):
            e[k] = P[f"{k}{l}"].detach()
    mp.meta.update(method="kvbridge-format map, closed form on chat prefixes + KL distillation through the frozen target "
                          "(KV-Lingo / Hybrid-KV recipe, train_kl2.py)", kl_steps=step, lr=LR, batch=BATCH)
    mp.save(OUT)


opt = torch.optim.AdamW(P.values(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
hist = dict(config=dict(dir=DIR, steps=STEPS, lr=LR, batch=BATCH, init=INIT), val=[(0, validate())], train=[])
log("step 0 (closed form) val KL", hist["val"][-1][1])
rng, run, oom, warm = random.Random(0), [], 0, max(1, int(0.05 * STEPS))
order = list(range(len(train))); rng.shuffle(order); ptr = 0
for step in range(1, STEPS + 1):
    f = min(1.0, step / warm) * 0.5 * (1 + math.cos(math.pi * max(0, step - warm) / max(1, STEPS - warm)))
    for g in opt.param_groups:
        g["lr"] = LR * f
    tot = 0.0
    for _ in range(BATCH):
        x = train[order[ptr % len(order)]]; ptr += 1
        try:
            loss = kl_loss(x["p"], x["c"]) / BATCH
            loss.backward(); tot += float(loss)
        except torch.OutOfMemoryError:
            oom += 1; log("OOM prompt", len(x["p"]), "answer", len(x["c"]), "- skipped"); torch.cuda.empty_cache()
    torch.nn.utils.clip_grad_norm_(P.values(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    run.append(tot)
    if step % 25 == 0:
        hist["train"].append((step, sum(run) / len(run)))
        log("step", step, "train KL", round(sum(run) / len(run), 4), "s/step", round((time.time() - t0) / step, 1), "oom", oom); run = []
    if step % VAL_EVERY == 0 or step == STEPS:
        hist["val"].append((step, validate())); log("step", step, "val KL", hist["val"][-1][1])
        save(step); hist["oom"] = oom; json.dump(hist, open(RES, "w"), indent=1)
log("KL2DONE", OUT)
