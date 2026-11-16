# [kvcodesign] copy of kvbase/eval_quality.py + low-rank shared-latent translators (method rrr_k{rk}_v{rv}[_bf16]).
# Quality under the CacheBridge handoff protocol: source (Qwen3-1.7B) prefills history H; target (Qwen3-4B) gets
# H's KV via a translator, processes the query Q and answers.  Thinking disabled.
# methods: native | src_native (1.7B reads and answers) | cb_official | cb_fused | kvbridge | dense_fused | dense_fp8
#          rrr_k{rk}_v{rv}[_bf16] low-rank | lrf_bf16 / lrf_fp8 fused kernels | tid_k{rk}_v{rv} low-rank with token
#          identity (fit_tid.py) | <kvbridge or rrr_...>_l0 = same translator with target layer 0 computed exactly
#          kl_k{rk}_v{rv}_{tag}[_bf16] low-rank map fine-tuned by KL through the frozen target (train_kl.py)
# tasks  : see tasks.py (quality | hotpotqa_e | gsm8k | gsm8k_h | squad | kvret | passkey_{2k,8k,16k})
# every generated answer records n_out (tokens generated) and trunc (hit the cap); gsm8k_h also records score_256
# env: TASK N METHODS MAPPER_CB OUT [MX]
import os, sys, json, numpy as np
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # before importing transformers / huggingface_hub
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

E = os.environ.get
ROOT = os.path.expanduser("~/kvbase")
sys.path.insert(0, ROOT)
MODELS = os.path.expanduser("~/kvswitch/models")
TASK, NS = E("TASK", "quality"), int(E("N", 200))
METHODS = E("METHODS", "native,cb_official,cb_fused,kvbridge,dense_fused,dense_fp8").split(",")
CB = E("MAPPER_CB", f"{ROOT}/mappers/cachebridge_17to4")
OUT = E("OUT", os.path.expanduser(f"~/kvcodesign/results/quality_{TASK}.json"))
dev = "cuda"
LS, LT, H, D = 28, 36, 8, 128

tok = AutoTokenizer.from_pretrained(f"{MODELS}/Qwen3-4B")
S = AutoModelForCausalLM.from_pretrained(f"{MODELS}/Qwen3-1.7B", torch_dtype=torch.bfloat16).to(dev).eval()
T = AutoModelForCausalLM.from_pretrained(f"{MODELS}/Qwen3-4B", torch_dtype=torch.bfloat16).to(dev).eval()
TR = {}
if {"cb_official"} & set(METHODS):
    from cachebridge.mapper import PageMapper
    TR["cb_official"] = PageMapper(CB, method="cachebridge", source_rope_theta=1e6, target_rope_theta=1e6, device=dev)
if {"cb_fused"} & set(METHODS):
    from fastmap import FastHeadLocal
    TR["cb_fused"] = FastHeadLocal(CB)
if {"cb_cublas32"} & set(METHODS):
    from fastmap import CublasHeadLocalF32
    TR["cb_cublas32"] = CublasHeadLocalF32(CB)
if {"cb_tf32x3"} & set(METHODS):
    from fastmap import FastHeadLocalF32
    TR["cb_tf32x3"] = FastHeadLocalF32(CB, prec="tf32x3", BM=16, num_warps=4)
if {"kvbridge", "kvbridge_l0", "dense_fused", "dense_fp8"} & set(METHODS):
    from kvbridge import Mapper as KBMapper, transfer as kb_transfer
    kb = KBMapper.load(f"{ROOT}/mappers/kvbridge_17to4", device=dev)
    from fastmap import FastDense
    if "dense_fused" in METHODS:
        TR["dense_fused"] = FastDense(kb)
    if "dense_fp8" in METHODS:
        TR["dense_fp8"] = FastDense(kb, fp8=True)


class LowRank:
    """Shared-latent reduced-rank ridge: strip source RoPE -> z = (x - mu_x) P -> y = z D + mu_y -> target RoPE.
    K and V use separate latents (ranks rk, rv). Weights from screen2.py (metric 'var', alpha 0.01)."""
    def __init__(self, rk, rv, dtype=torch.float32, path=None):
        if path:   # one file holding all four matrices (train_kl.py: behaviourally fine-tuned low-rank map)
            a = b = torch.load(path)
        else:
            a = torch.load(os.path.expanduser(f"~/kvcodesign/mappers/rrr_var_r{rk}.pt"))
            b = torch.load(os.path.expanduser(f"~/kvcodesign/mappers/rrr_var_r{rv}.pt"))
        PK = LS * H * D
        g = lambda t: t.to(dev, dtype)
        self.Pk, self.Dk, self.Pv, self.Dv = g(a["Pk"]), g(a["Dk"]), g(b["Pv"]), g(b["Dv"])
        self.mxk, self.mxv = g(a["mx"][:PK]), g(a["mx"][PK:])
        self.myk, self.myv = g(a["my"][:LT * H * D]), g(a["my"][LT * H * D:])
        self.dt = dtype
        inv = 1.0 / (1e6 ** (torch.arange(0, D, 2, device=dev, dtype=torch.float32) / D))
        self.inv = inv

    def cs(self, n):
        ang = torch.arange(n, device=dev, dtype=torch.float32)[:, None] * self.inv[None]
        return torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)  # [n, D]

    @staticmethod
    def rot(x):
        return torch.cat([-x[..., D // 2:], x[..., :D // 2]], -1)

    @torch.no_grad()
    def __call__(self, sk, sv, chunk=2048):  # sk, sv: [LS, n, H, D] (keys rotated) -> [2, LT, n, H, D] bf16
        n = sk.shape[1]
        out = torch.empty(2, LT, n, H, D, device=dev, dtype=torch.bfloat16)
        cos, sin = self.cs(n)
        for p in range(0, n, chunk):
            q = min(n, p + chunk); m = q - p
            c, s_ = cos[None, p:q, None], sin[None, p:q, None]
            k = sk[:, p:q].float()
            k = k * c - self.rot(k) * s_                                       # strip source RoPE
            xk = k.permute(1, 0, 2, 3).reshape(m, -1).to(self.dt) - self.mxk
            xv = sv[:, p:q].permute(1, 0, 2, 3).reshape(m, -1).to(self.dt) - self.mxv
            yk = ((xk @ self.Pk) @ self.Dk + self.myk).float().view(m, LT, H, D).permute(1, 0, 2, 3)
            out[1, :, p:q] = ((xv @ self.Pv) @ self.Dv + self.myv).view(m, LT, H, D).permute(1, 0, 2, 3).to(torch.bfloat16)
            out[0, :, p:q] = (yk * c + self.rot(yk) * s_).to(torch.bfloat16)  # apply target RoPE
        return out


def rope_keys(k, n):   # k [n, H, D] content space -> rotated (target RoPE, positions 0..n-1), fp32
    inv = 1.0 / (1e6 ** (torch.arange(0, D, 2, device=dev, dtype=torch.float32) / D))
    ang = torch.arange(n, device=dev, dtype=torch.float32)[:, None] * inv[None]
    c, s_ = torch.cat([ang.cos(), ang.cos()], -1)[:, None], torch.cat([ang.sin(), ang.sin()], -1)[:, None]
    return k * c + torch.cat([-k[..., D // 2:], k[..., :D // 2]], -1) * s_


@torch.no_grad()
def tgt_l0(hist):
    """Token identity in the target model: its layer-0 K (content space) and V depend only on the token ids
    (shared tokenizer), so they are computed exactly, without reading the history.  -> k0, v0 [n, H, D] fp32"""
    lay = T.model.layers[0]; a = lay.self_attn
    x = lay.input_layernorm(T.model.embed_tokens(torch.tensor([hist], device=dev)))
    n = len(hist)
    return a.k_norm(a.k_proj(x).view(1, n, H, D))[0].float(), a.v_proj(x).view(n, H, D).float()


class LowRankTID:
    """Low-rank map with token identity (fit_tid.py): target layer 0 exact; layers 1..35 from
    y = ([x - mx, s (e - me)] P) D + my  with e = exact target layer-0 K,V of the same token."""
    def __init__(self, rk, rv, dtype=torch.float32):
        a = torch.load(os.path.expanduser(f"~/kvcodesign/mappers/tid_r{rk}.pt"))
        b = torch.load(os.path.expanduser(f"~/kvcodesign/mappers/tid_r{rv}.pt"))
        PK, QK, HD = LS * H * D, LT * H * D, H * D
        g = lambda t: t.to(dev, dtype)
        self.Pk, self.Dk, self.Pv, self.Dv = g(a["Pk"]), g(a["Dk"]), g(b["Pv"]), g(b["Dv"])
        self.mxk, self.mxv, self.me = g(a["mx"][:PK]), g(a["mx"][PK:]), g(a["me"])
        self.myk, self.myv = g(a["my"][HD:QK]), g(a["my"][QK + HD:])
        self.sk_, self.sv_, self.dt = a["s_k"], b["s_v"], dtype
        # The first token is the attention sink; identity features wreck its V (relative error 0.44 -> 1.99 offline,
        # diag_tid.py), so position 0 is translated by the plain low-rank map of the same ranks.
        self.lr = LowRank(rk, rv, dtype)

    @torch.no_grad()
    def __call__(self, sk, sv, k0, v0, chunk=2048):
        n = sk.shape[1]
        out = torch.empty(2, LT, n, H, D, device=dev, dtype=torch.bfloat16)
        cos, sin = self.lr.cs(n)
        e = torch.cat([k0.reshape(n, -1), v0.reshape(n, -1)], 1).to(self.dt) - self.me
        out[0, 0], out[1, 0] = rope_keys(k0, n).to(torch.bfloat16), v0.to(torch.bfloat16)
        for p in range(0, n, chunk):
            q = min(n, p + chunk); m = q - p
            c, s_ = cos[None, p:q, None], sin[None, p:q, None]
            k = sk[:, p:q].float()
            k = k * c - LowRank.rot(k) * s_
            xk = torch.cat([k.permute(1, 0, 2, 3).reshape(m, -1).to(self.dt) - self.mxk, e[p:q] * self.sk_], 1)
            xv = torch.cat([sv[:, p:q].permute(1, 0, 2, 3).reshape(m, -1).to(self.dt) - self.mxv, e[p:q] * self.sv_], 1)
            yk = ((xk @ self.Pk) @ self.Dk + self.myk).float().view(m, LT - 1, H, D).permute(1, 0, 2, 3)
            out[1, 1:, p:q] = ((xv @ self.Pv) @ self.Dv + self.myv).view(m, LT - 1, H, D).permute(1, 0, 2, 3).to(torch.bfloat16)
            out[0, 1:, p:q] = (yk * c + LowRank.rot(yk) * s_).to(torch.bfloat16)
        out[:, 1:, :1] = self.lr(sk[:, :1], sv[:, :1])[:, 1:]
        return out


# method suffix "_l0": any translator, with target layer 0 replaced by the exact token-identity K,V
for m in METHODS:
    if m.startswith("kl_"):    # kl_k{rk}_v{rv}_{tag}[_bf16]: low-rank map fine-tuned by KL through the frozen target
        base = m[:-5] if m.endswith("_bf16") else m
        TR[m] = LowRank(0, 0, torch.bfloat16 if m.endswith("_bf16") else torch.float32, path=os.path.expanduser(f"~/kvcodesign/mappers/{base}.pt"))
        continue
    if m.startswith("tid_"):   # tid_k{rk}_v{rv}[_bf16]
        parts = m.split("_")
        TR[m] = LowRankTID(int(parts[1][1:]), int(parts[2][1:]), torch.bfloat16 if m.endswith("_bf16") else torch.float32)
        continue
    if m.endswith("_l0") and m.startswith("rrr_"):
        parts = m.split("_")
        TR[m] = LowRank(int(parts[1][1:]), int(parts[2][1:]), torch.bfloat16 if "_bf16" in m else torch.float32)
        continue
    if m in ("lrf_bf16", "lrf_fp8"):   # fused Triton kernels (fastlr.py), K 768 + V 1536
        sys.path.insert(0, os.path.expanduser("~/kvcodesign"))
        from fastlr import FastLowRank
        sc = json.load(open(os.path.expanduser("~/kvcodesign/data/fp8_scales_k768_v1536.json")))
        TR[m] = FastLowRank(768, 1536, fp8=(m == "lrf_fp8"), scales=sc)
    if m.startswith("rrr_"):
        parts = m.split("_")
        rk, rv = int(parts[1][1:]), int(parts[2][1:])
        TR[m] = LowRank(rk, rv, torch.bfloat16 if m.endswith("_bf16") else torch.float32)


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasks import build_samples, score   # task construction and format-insensitive scoring (tasks.py)

samples = build_samples(TASK, NS, lambda t: tok(t, add_special_tokens=False).input_ids)
if E("MX"):   # optional override of the generation cap
    for s_ in samples:
        if "mx" in s_:
            s_["mx"] = int(E("MX"))


# ---------------------------------------------------------------- handoff
@torch.no_grad()
def target_cache(method, hist):
    if method == "native":
        c = DynamicCache(); T(torch.tensor([hist], device=dev), past_key_values=c, use_cache=True); return c
    if method == "src_native":   # source model (1.7B) reads the history and answers itself
        c = DynamicCache(); S(torch.tensor([hist], device=dev), past_key_values=c, use_cache=True); return c
    sc = S(torch.tensor([hist], device=dev), use_cache=True).past_key_values
    if method == "kvbridge_l0":   # official dense translation, layer 0 replaced by the exact token-identity K,V
        c = target_cache("kvbridge", hist)
        k0, v0 = tgt_l0(hist)
        c.layers[0].keys = rope_keys(k0, len(hist)).transpose(0, 1)[None].to(c.layers[0].keys.dtype)
        c.layers[0].values = v0.transpose(0, 1)[None].to(c.layers[0].values.dtype)
        return c
    if method == "kvbridge":
        n = sc.layers[0].keys.shape[2]
        if n <= 8192:
            return kb_transfer(sc, kb)
        # long histories: the official transfer() run on 4096-token slices with explicit positions, then concatenated
        # (identical arithmetic per token; avoids the out-of-memory failure at 16K tokens on a 24 GB card)
        parts = []
        for p in range(0, n, 4096):
            cc = DynamicCache()
            for l in range(LS):
                cc.update(sc.layers[l].keys[:, :, p:p + 4096], sc.layers[l].values[:, :, p:p + 4096], l)
            parts.append(kb_transfer(cc, kb, positions=torch.arange(p, min(n, p + 4096), device=dev)))
            del cc
        del sc
        c = DynamicCache()
        for l in range(LT):
            c.update(torch.cat([x.layers[l].keys for x in parts], 2), torch.cat([x.layers[l].values for x in parts], 2), l)
        return c
    sk = torch.stack([sc.layers[l].keys[0].transpose(0, 1) for l in range(LS)]).contiguous()  # [L,T,H,D]
    sv = torch.stack([sc.layers[l].values[0].transpose(0, 1) for l in range(LS)]).contiguous()
    n = sk.shape[1]
    del sc   # free the source cache before translating (matters at 16K tokens on a 24 GB card)
    if method == "cb_official":
        out = torch.cat([TR[method].map_page(torch.stack([sk[:, p:p + 2048], sv[:, p:p + 2048]]), position_start=p, output_device=dev)
                         for p in range(0, n, 2048)], 2)
    elif method.startswith("tid_"):
        out = TR[method](sk, sv, *tgt_l0(hist))
    else:
        out = TR[method](sk, sv)
        if method.endswith("_l0"):
            k0, v0 = tgt_l0(hist)
            out[0, 0], out[1, 0] = rope_keys(k0, n).to(out.dtype), v0.to(out.dtype)
    del sk, sv
    c = DynamicCache()
    for l in range(LT):
        k, v = out[0, l].transpose(0, 1)[None], out[1, l].transpose(0, 1)[None]
        if n <= 8192:   # long histories keep views of `out` instead of a second 2.4 GB copy (same values)
            k, v = k.contiguous(), v.contiguous()
        c.update(k, v, l)
    return c


LETTERS = [tok.encode(x, add_special_tokens=False)[0] for x in "ABCD"]
STOP = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}


@torch.no_grad()
def answer(c, s, M=None):   # -> (prediction, generated token ids or None, truncated at the cap?)
    M = M or T
    lg = M(torch.tensor([s["q"]], device=dev), past_key_values=c, use_cache=True).logits[0, -1]
    if s["kind"] == "mc":
        return int(lg[LETTERS].argmax()), None, False
    outt, nxt, trunc = [], int(lg.argmax()), True
    for _ in range(s["mx"]):
        if nxt in STOP:
            trunc = False
            break
        outt.append(nxt)
        nxt = int(M(torch.tensor([[nxt]], device=dev), past_key_values=c, use_cache=True).logits[0, -1].argmax())
    if trunc and nxt in STOP:   # the stop token arrived exactly at the cap
        trunc = False
    return tok.decode(outt, skip_special_tokens=True).strip(), outt, trunc


res = json.load(open(OUT)) if os.path.exists(OUT) else dict(task=TASK, samples=[])
done = len(res["samples"])
for i, s in enumerate(samples):
    if i < done:
        continue
    rec = dict(n_hist=len(s["h"]), m={}, **s.get("meta", {}))
    for m in METHODS:
        c = target_cache(m, s["h"])
        p, ids, trunc = answer(c, s, S if m == "src_native" else T)
        r = dict(pred=p, **score(s, p))
        if ids is not None:
            r.update(n_out=len(ids), trunc=trunc)
            if s["kind"] == "gsm" and s["mx"] > 256:   # same greedy output cut at 256 tokens (truncation control)
                r["score_256"] = score(s, tok.decode(ids[:256], skip_special_tokens=True).strip())["score"]
        rec["m"][m] = r
        del c
    res["samples"].append(rec)
    if (i + 1) % 20 == 0 or i == len(samples) - 1:
        json.dump(res, open(OUT, "w"))
        print(TASK, i + 1, {m: round(float(np.mean([x["m"][m]["score"] for x in res["samples"]])), 4) for m in METHODS}, flush=True)
    torch.cuda.empty_cache()
json.dump(res, open(OUT, "w"))
print("QUALDONE")
