# [kvcodesign] fit_cb2.py with the official recipe's key ridge: K initial fit uses ridge KRIDGE (official
# qwen3_14b_32b_trajectory recipe: K_ridge=1.0), V initial fit keeps ridge 0.01, then the same V calibration.
# Same support (top-8 source layers by head-local R^2) as fit_cb2. Purpose: check whether our ridge choice hurt CacheBridge.
# Fit a CacheBridge (head-local + attention-readout-calibrated) mapper for Qwen3-1.7B -> Qwen3-4B.
# Official code used where it exists:
#   - kvbridge.fit.default_calibration_texts / collect_kv : FineWeb-Edu 500 x 1024 tok, stride 4, content-space KV
#   - cachebridge.offline.initialization.fit_layer        : head-local centered ridge (lambda=0.01) init
#   - cachebridge.offline.calibration.algebra             : probabilities / gram / solve (ridge .01 x mean diag)
# Not released by CacheBridge, re-implemented from the paper/docs:
#   - support selection: per target layer, mean R^2 over K,V,heads of single-source-layer head-local regressions; top-k
#   - sequential V readout calibration driver on 24 QuALITY docs x 4 questions (layer order, propagate corrections)
# env: K(8) NQ_DOCS(24) QPOS(8) OUT
import os, json, math, random, collections, torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

E = os.environ.get
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from kvbridge.fit import default_calibration_texts, collect_kv
from kvbridge.rope import strip_rope, apply_rope
from cachebridge.offline.initialization import fit_layer
from cachebridge.offline.calibration import algebra

K, NQ, QPOS = int(E("K", 8)), int(E("NQ_DOCS", 24)), int(E("QPOS", 8))
OUT = E("OUT", os.path.expanduser("~/kvbase/mappers/cachebridge_17to4_kr1"))
KRIDGE = float(E("KRIDGE", 1.0))
MODELS = os.path.expanduser("~/kvswitch/models")
dev = "cuda"
THETA = 1e6
log = lambda *a: print(*a, flush=True)

tok = AutoTokenizer.from_pretrained(f"{MODELS}/Qwen3-1.7B")
S = AutoModelForCausalLM.from_pretrained(f"{MODELS}/Qwen3-1.7B", torch_dtype=torch.bfloat16).to(dev).eval()
T = AutoModelForCausalLM.from_pretrained(f"{MODELS}/Qwen3-4B", torch_dtype=torch.bfloat16).to(dev).eval()
S.requires_grad_(False); T.requires_grad_(False)
torch.set_grad_enabled(False)  # fix: no autograd graph anywhere in the calibration driver
LS, LT = S.config.num_hidden_layers, T.config.num_hidden_layers
H, D = T.config.num_key_value_heads, 128

# ---------------- 1. calibration traces (official kvbridge collector) ----------------
cache_path = os.path.expanduser("~/kvbase/mappers/calib_17to4.pt")
if os.path.exists(os.path.expanduser(f"~/kvbase/mappers/cb_init_17to4_kr{float(E('KRIDGE', 1.0)):g}.pt")):
    sk = sv = tk = tv = [torch.zeros(1)]
elif os.path.exists(cache_path):
    sk, sv, tk, tv = torch.load(cache_path)
else:
    texts = default_calibration_texts(500)
    sk, sv = collect_kv(S, tok, texts, seq_len=1024, stride=4, device=dev, desc="src")
    tk, tv = collect_kv(T, tok, texts, seq_len=1024, stride=4, device=dev, desc="tgt")
    torch.save((sk, sv, tk, tv), cache_path)
N = sk[0].shape[0]
log("observations", N)

OLD = os.path.expanduser("~/kvbase/mappers/cb_init_17to4.pt")
INIT = os.path.expanduser(f"~/kvbase/mappers/cb_init_17to4_kr{KRIDGE:g}.pt")
SKIP_INIT = os.path.exists(INIT)
if SKIP_INIT:
    support, layers, R2 = torch.load(INIT); log("loaded init")
else:
    support, old_layers, R2 = torch.load(OLD)   # reuse V init (ridge .01) from the old initial fit
    log("reusing support from", OLD)
SKIP_SUPPORT = True
# ---------------- 2. support selection by head-local R^2 (paper Sec. "Source layers and initial fit") ----------------
sub = torch.randperm(N, generator=torch.Generator().manual_seed(0))[:32768]


def r2_headlocal(xs, ys):  # xs, ys: [n, H, d] -> mean R^2 over heads (centered ridge, in-sample)
    x = xs.to(dev).float(); y = ys.to(dev).float()
    x = x - x.mean(0); y = y - y.mean(0)
    xh, yh = x.transpose(0, 1), y.transpose(0, 1)  # [H,n,d]
    G = xh.transpose(1, 2) @ xh + 0.01 * torch.eye(D, device=dev)
    W = torch.linalg.solve(G, xh.transpose(1, 2) @ yh)
    res = yh - xh @ W
    return (1 - res.square().sum((1, 2)) / yh.square().sum((1, 2))).mean().item()


for s in []:
    xk, xv = sk[s][sub], sv[s][sub]
    for t in range(LT):
        R2[t, s] = 0.5 * (r2_headlocal(xk, tk[t][sub]) + r2_headlocal(xv, tv[t][sub]))
    log("R2 source layer", s, "done")
log("support", support)

# ---------------- 3. head-local initial fit (official fit_layer) ----------------
os.makedirs(OUT, exist_ok=True)
if not SKIP_INIT:
    layers, diag = [], []
    src = torch.stack([torch.stack(sk), torch.stack(sv)])  # [2, LS, N, H, D] fp16 CPU
for t in ([] if SKIP_INIT else range(LT)):
    tgt = torch.stack([tk[t], tv[t]])
    tensors = {k: v.clone() for k, v in old_layers[t].items()}; dg = None
    tk1, _ = fit_layer(src, tgt, selected=support[t], ridge=KRIDGE, device=dev, dtype=torch.float32)  # official default FP32
    n0 = tensors["k_weight"].norm().item()
    tensors["k_weight"], tensors["k_bias"] = tk1["k_weight"], tk1["k_bias"]
    log("layer", t, "K weight norm ridge .01 -> %g: %.1f -> %.1f" % (KRIDGE, n0, tk1["k_weight"].norm().item()))
    layers.append(tensors); diag.append(dg)
    log("init layer", t)
if not SKIP_INIT:
    del src
    torch.save((support, layers, R2), INIT)
del sk, sv, tk, tv

# ---------------- 4. sequential attention-readout V calibration (paper Sec. "Attention-readout calibration") ----------------
from datasets import load_dataset
q = load_dataset("emozilla/quality", split="validation")
arts = {}
for r in q:
    arts.setdefault(r["article"], []).append(r)
docs = [v for v in arts.values() if len(v) >= 4]
random.Random(0).shuffle(docs)
docs = docs[:NQ]
pairs = []
for d in docs:
    art = d[0]["article"]
    hist = tok(f"<|im_start|>user\nRead the following article.\n\n{art}", add_special_tokens=False).input_ids[:6000]
    for r in d[:4]:
        opts = "\n".join(f"{c}. {o}" for c, o in zip("ABCD", r["options"]))
        qids = tok(f"\n\nQuestion: {r['question']}\n{opts}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:",
                   add_special_tokens=False).input_ids
        pairs.append((hist, qids))
log("calibration pairs", len(pairs))


def cos_sin(pos):
    from kvbridge.rope import build_cos_sin
    return build_cos_sin(pos, D, THETA, dev, torch.float32)


@torch.no_grad()
def source_kv(hist):
    out = S(torch.tensor([hist], device=dev), use_cache=True)
    pos = torch.arange(len(hist), device=dev)
    ks = [strip_rope(out.past_key_values.layers[l].keys[0].float(), pos, THETA).to(torch.bfloat16).cpu().pin_memory() for l in range(LS)]
    vs = [out.past_key_values.layers[l].values[0].cpu().pin_memory() for l in range(LS)]
    return ks, vs


def map_layer(t, ks, vs, pos):
    L = layers[t]
    sel = L["selected"].tolist()
    xk = torch.cat([ks[s].to(dev, non_blocking=True).float() for s in sel], -1)  # [H,n,K*D]
    xv = torch.cat([vs[s].to(dev, non_blocking=True).float() for s in sel], -1)
    k = torch.einsum("hnf,hfd->hnd", xk, L["k_weight"].to(dev)) + L["k_bias"].to(dev)[:, None]
    v = torch.einsum("hnf,hfd->hnd", xv, L["v_weight"].to(dev)) + L["v_bias"].to(dev)[:, None]
    return apply_rope(k, pos, THETA), v, xv


cap = {}


def hook(l):
    def f(mod, args, kwargs):
        cap[l] = kwargs  # q/k/v are recomputed below from hidden states
    return f


# per-pair precompute: native target history KV (for the shadow readout) and source KV
prep = []
for hist, qids in pairs:
    with torch.no_grad():
        nat = T(torch.tensor([hist], device=dev), use_cache=True).past_key_values
    ks, vs = source_kv(hist)
    prep.append(dict(hist=hist, qids=qids, nat=[(nat.layers[l].keys.cpu().pin_memory(), nat.layers[l].values.cpu().pin_memory()) for l in range(LT)],
                     ks=ks, vs=vs))
    del nat
log("prepared", "mem GB", torch.cuda.memory_allocated() / 2**30)

attn_in = {}


def capture_qkv(l):
    def f(mod, args, kwargs):
        attn_in[l] = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
    return f


hs = [T.model.layers[l].self_attn.register_forward_pre_hook(capture_qkv(l), with_kwargs=True) for l in range(LT)]


def q_states(l, hidden, n0, nq):
    a = T.model.layers[l].self_attn
    hsh = (1, nq, -1, D)
    qq = a.q_norm(a.q_proj(hidden).view(hsh)).transpose(1, 2)
    kk = a.k_norm(a.k_proj(hidden).view(hsh)).transpose(1, 2)
    vv = a.v_proj(hidden).view(hsh).transpose(1, 2)
    cos, sin = cos_sin(torch.arange(n0, n0 + nq, device=dev))
    rot = lambda x: (x.float() * cos + torch.cat([-x[..., D // 2:], x[..., :D // 2]], -1).float() * sin)
    return rot(qq), rot(kk), vv.float()


report = []
for t in range(LT):
    G = torch.zeros(H, K * D + 1, K * D + 1, device=dev, dtype=torch.float64)
    C = torch.zeros(H, K * D + 1, D, device=dev, dtype=torch.float64)
    e0 = 0.0
    for p in prep:
        n0, nq = len(p["hist"]), len(p["qids"])
        pos = torch.arange(n0, device=dev)
        cache = DynamicCache()
        for l in range(LT):  # history cache from the CURRENT mapper (earlier layers already corrected)
            k, v, _ = map_layer(l, p["ks"], p["vs"], pos)
            cache.update(k[None].to(torch.bfloat16), v[None].to(torch.bfloat16), l)
            del k, v, _
        with torch.no_grad():
            T(torch.tensor([p["qids"]], device=dev), past_key_values=cache,
              position_ids=torch.arange(n0, n0 + nq, device=dev)[None])
        qq, kq, vq = q_states(t, attn_in[t], n0, nq)
        kmap, vmap, xv = map_layer(t, p["ks"], p["vs"], pos)
        positions = list(range(nq - QPOS, nq))
        kn, vn = p["nat"][t][0][0].to(dev).float(), p["nat"][t][1][0].to(dev).float()
        Pm = algebra.probabilities(qq, torch.cat([kmap, kq[0]], 1)[None], n0, positions)  # [H, G*Q, n0+nq]
        Pn = algebra.probabilities(qq, torch.cat([kn, kq[0]], 1)[None], n0, positions)
        Om = algebra.pool(Pm, torch.cat([vmap, vq[0]], 1))
        On = algebra.pool(Pn, torch.cat([vn, vq[0]], 1))
        R = (On - Om).double()  # readout residual [H, G*Q, D]
        Xb = torch.cat([xv, torch.ones_like(xv[..., :1])], -1)  # source V features + intercept [H, n0, K*D+1]
        Dq = algebra.pool(Pm[..., :n0], Xb).double()  # [H, G*Q, K*D+1]
        g, c = algebra.gram(Dq, R)
        G += g; C += c
        e0 += R.square().sum().item()
        if t == 0 and len(report) == 0 and p is prep[0]:
            log("mem after first pair GB", torch.cuda.memory_allocated() / 2**30, torch.cuda.max_memory_allocated() / 2**30)
        del cache, kmap, vmap, xv, Pm, Pn, Om, On, R, Xb, Dq, kn, vn, qq, kq, vq
        attn_in.clear()
        import gc; gc.collect()
        torch.cuda.empty_cache()
        assert not G.requires_grad
    dTheta, lam, resid = algebra.solve(G / len(prep), C / len(prep), fraction=0.01)
    dTheta = dTheta.float().cpu()
    layers[t]["v_weight"] = layers[t]["v_weight"] + dTheta[:, :-1]
    layers[t]["v_bias"] = layers[t]["v_bias"] + dTheta[:, -1]
    report.append(dict(layer=t, readout_residual_energy=e0 / len(prep)))
    log("calibrated layer", t, "residual energy", round(e0 / len(prep), 4), "mem GB", round(torch.cuda.memory_allocated() / 2**30, 2), round(torch.cuda.max_memory_allocated() / 2**30, 2))
for h in hs:
    h.remove()

for t, L in enumerate(layers):
    save_file({k: v.contiguous() for k, v in L.items()}, f"{OUT}/layer_{t:03d}.safetensors")
json.dump(dict(k_ridge=KRIDGE, support=support, R2=R2.tolist(), K=K, calib=report, n_obs=N, n_pairs=len(pairs)),
          open(f"{OUT}/fit_report.json", "w"), indent=1)
log("FITCBDONE")
