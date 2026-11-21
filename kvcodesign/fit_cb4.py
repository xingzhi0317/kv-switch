# CacheBridge mapper with the paper's ENERGY-CONSTRAINED value calibration (arXiv 2609.00891, Appendix B).
# fit_cb4.py = fit_cb2_dir.py (same official initial fit, same support selection, same sequential driver) plus:
#   1. value correction per target KV head solved as  dTheta(mu) = (G + lam*I + mu*M)^-1 C,  lam = max(1e-8, .01 tr(G)/p),
#      subject to  tr(dTheta^T M dTheta) <= 0.01 * E0  (paper: "at most 1% of the initial residual energy"), where
#        M  = second moment of the source V features (+ intercept) over history tokens, + 1e-4 * mean(diag) stabilisation
#        E0 = the initial mapper's value residual energy vs re-prefill, mean over 4096 sampled history states
#      mu = 0 if the unconstrained solution meets the budget, else a scalar (bisection) search puts it on the boundary.
#      The official public solver (cachebridge.offline.calibration.algebra.solve) is a plain ridge without this
#      constraint; the CacheBridge docs state that energy-constrained calibration is not released.  Re-implemented.
#   2. readout matched at ALL query positions (fit_cb2: the last 8), 24 QuALITY documents x 4 questions as in the paper.
#   3. the initial (uncalibrated) mapper is saved too: {OUT}_init  (to measure what calibration adds).
# DIR: 17to4 | 4to17 | same17 (Qwen3-1.7B -> itself: sanity check, the fitted mapper must reproduce native)
# env: DIR K(8) NQ_DOCS(24) FRAC(0.01) OUT
import os, json, random, torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

E = os.environ.get
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from kvbridge.rope import strip_rope, apply_rope
from cachebridge.offline.initialization import fit_layer
from cachebridge.offline.calibration import algebra

K, NQ, FRAC = int(E("K", 8)), int(E("NQ_DOCS", 24)), float(E("FRAC", 0.01))
DIR = E("DIR", "17to4")
SRC_M, TGT_M = {"17to4": ("Qwen3-1.7B", "Qwen3-4B"), "4to17": ("Qwen3-4B", "Qwen3-1.7B"),
                "same17": ("Qwen3-1.7B", "Qwen3-1.7B")}[DIR]
OUT = E("OUT", os.path.expanduser(f"~/kvbase/mappers/cachebridge_{DIR}_ec"))
MODELS = os.path.expanduser("~/kvswitch/models")
dev, THETA, NSAMP = "cuda", 1e6, 4096
log = lambda *a: print(*a, flush=True)

tok = AutoTokenizer.from_pretrained(f"{MODELS}/Qwen3-1.7B")
S = AutoModelForCausalLM.from_pretrained(f"{MODELS}/{SRC_M}", torch_dtype=torch.bfloat16).to(dev).eval()
T = S if SRC_M == TGT_M else AutoModelForCausalLM.from_pretrained(f"{MODELS}/{TGT_M}", torch_dtype=torch.bfloat16).to(dev).eval()
S.requires_grad_(False); T.requires_grad_(False)
torch.set_grad_enabled(False)
LS, LT = S.config.num_hidden_layers, T.config.num_hidden_layers
H, D = T.config.num_key_value_heads, 128

# ---------------- 1-3. support selection + official head-local initial fit (as fit_cb2_dir.py; cached) ----------------
INIT = os.path.expanduser(f"~/kvbase/mappers/cb_init_{DIR}.pt")
if os.path.exists(INIT):
    support, layers, R2 = torch.load(INIT)
    log("loaded init", INIT)
else:
    sk, sv, tk, tv = torch.load(os.path.expanduser("~/kvbase/mappers/calib_17to4.pt"))   # (1.7B, 4B) on the same tokens
    if DIR == "4to17":
        sk, sv, tk, tv = tk, tv, sk, sv
    elif DIR == "same17":
        tk, tv = sk, sv
    N = sk[0].shape[0]
    sub = torch.randperm(N, generator=torch.Generator().manual_seed(0))[:32768]

    def r2_headlocal(xs, ys):
        x = xs.to(dev).float(); y = ys.to(dev).float()
        x = x - x.mean(0); y = y - y.mean(0)
        xh, yh = x.transpose(0, 1), y.transpose(0, 1)
        G = xh.transpose(1, 2) @ xh + 0.01 * torch.eye(D, device=dev)
        W = torch.linalg.solve(G, xh.transpose(1, 2) @ yh)
        return (1 - (yh - xh @ W).square().sum((1, 2)) / yh.square().sum((1, 2))).mean().item()

    R2 = torch.zeros(LT, LS)
    for s in range(LS):
        for t in range(LT):
            R2[t, s] = 0.5 * (r2_headlocal(sk[s][sub], tk[t][sub]) + r2_headlocal(sv[s][sub], tv[t][sub]))
        log("R2 source layer", s)
    support = [sorted(R2[t].topk(K).indices.tolist()) for t in range(LT)]
    src = torch.stack([torch.stack(sk), torch.stack(sv)])
    layers = []
    for t in range(LT):
        tensors, _ = fit_layer(src, torch.stack([tk[t], tv[t]]), selected=support[t], ridge=0.01, device=dev, dtype=torch.float64)
        layers.append(tensors)
        log("init layer", t)
    del src, sk, sv, tk, tv
    torch.save((support, layers, R2), INIT)
log("support", support)
CORRUPT = float(E("CORRUPT", 0))   # test of the calibration itself: perturb the initial V maps, then calibrate
if CORRUPT > 0:
    g = torch.Generator().manual_seed(3)
    for L in layers:
        w = L["v_weight"]
        L["v_weight"] = w + CORRUPT * w.float().std() * torch.randn(w.shape, generator=g).to(w.dtype)
    log("corrupted initial V maps, relative noise", CORRUPT)


def save_mapper(path, ls, extra):
    os.makedirs(path, exist_ok=True)
    for t, L in enumerate(ls):
        save_file({k: v.contiguous() for k, v in L.items()}, f"{path}/layer_{t:03d}.safetensors")
    json.dump(dict(support=support, K=K, **extra), open(f"{path}/fit_report.json", "w"), indent=1)


save_mapper(OUT + "_init", layers, dict(note="official head-local initial fit only, no value calibration"))
log("saved", OUT + "_init")

# ---------------- 4. calibration data: 24 QuALITY documents x 4 questions (paper), target reads the appended question --
from datasets import load_dataset
arts = {}
for r in load_dataset("emozilla/quality", split="validation"):
    arts.setdefault(r["article"], []).append(r)
docs = [v for v in arts.values() if len(v) >= 4]
random.Random(0).shuffle(docs)
pairs = []
for d in docs[:NQ]:
    hist = tok(f"<|im_start|>user\nRead the following article.\n\n{d[0]['article']}", add_special_tokens=False).input_ids[:6000]
    for r in d[:4]:
        opts = "\n".join(f"{c}. {o}" for c, o in zip("ABCD", r["options"]))
        pairs.append((hist, tok(f"\n\nQuestion: {r['question']}\n{opts}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:",
                                add_special_tokens=False).input_ids))
log("calibration pairs", len(pairs))
rng = torch.Generator().manual_seed(1)


def source_kv(hist):
    out = S(torch.tensor([hist], device=dev), use_cache=True)
    pos = torch.arange(len(hist), device=dev)
    ks = [strip_rope(out.past_key_values.layers[l].keys[0].float(), pos, THETA).to(torch.bfloat16).cpu().pin_memory() for l in range(LS)]
    vs = [out.past_key_values.layers[l].values[0].cpu().pin_memory() for l in range(LS)]
    return ks, vs


def map_layer(t, ks, vs, pos):
    L = layers[t]
    sel = L["selected"].tolist()
    xk = torch.cat([ks[s].to(dev, non_blocking=True).float() for s in sel], -1)
    xv = torch.cat([vs[s].to(dev, non_blocking=True).float() for s in sel], -1)
    k = torch.einsum("hnf,hfd->hnd", xk, L["k_weight"].to(dev)) + L["k_bias"].to(dev)[:, None]
    v = torch.einsum("hnf,hfd->hnd", xv, L["v_weight"].to(dev)) + L["v_bias"].to(dev)[:, None]
    return apply_rope(k, pos, THETA), v, xv


prep = []
for hist, qids in pairs:
    nat = T(torch.tensor([hist], device=dev), use_cache=True).past_key_values
    ks, vs = source_kv(hist)
    samp = torch.randperm(len(hist), generator=rng)[:NSAMP // len(pairs) + 1]       # E0 states: ~4096 in total
    prep.append(dict(hist=hist, qids=qids, ks=ks, vs=vs, samp=samp,
                     nat=[(nat.layers[l].keys.cpu().pin_memory(), nat.layers[l].values.cpu().pin_memory()) for l in range(LT)]))
    del nat
log("prepared")

attn_in = {}
hooks = [T.model.layers[l].self_attn.register_forward_pre_hook(
    (lambda l: lambda mod, args, kwargs: attn_in.__setitem__(l, kwargs.get("hidden_states", args[0] if args else None)))(l),
    with_kwargs=True) for l in range(LT)]


def q_states(l, hidden, n0, nq):
    a = T.model.layers[l].self_attn
    hsh = (1, nq, -1, D)
    qq = a.q_norm(a.q_proj(hidden).view(hsh)).transpose(1, 2)
    kk = a.k_norm(a.k_proj(hidden).view(hsh)).transpose(1, 2)
    vv = a.v_proj(hidden).view(hsh).transpose(1, 2)
    from kvbridge.rope import build_cos_sin
    cos, sin = build_cos_sin(torch.arange(n0, n0 + nq, device=dev), D, THETA, dev, torch.float32)
    rot = lambda x: (x.float() * cos + torch.cat([-x[..., D // 2:], x[..., :D // 2]], -1).float() * sin)
    return rot(qq), rot(kk), vv.float()


def constrained_solve(G, C, M, budget):
    """per head: dTheta(mu) = (G + lam I + mu M)^-1 C with tr(dTheta^T M dTheta) <= budget (mu >= 0, bisection)."""
    p = G.shape[-1]
    I = torch.eye(p, device=G.device, dtype=G.dtype)
    out, mus, energies = [], [], []
    for h in range(G.shape[0]):
        g = 0.5 * (G[h] + G[h].T); m = 0.5 * (M[h] + M[h].T)
        lam = max(1e-8, 0.01 * float(torch.trace(g)) / p)
        solve = lambda mu: torch.linalg.solve(g + lam * I + mu * m, C[h])
        energy = lambda d: float(torch.einsum("pd,pq,qd->", d, m, d))
        d, mu = solve(0.0), 0.0
        if energy(d) > budget[h]:
            lo, hi = 0.0, float(torch.trace(g)) / max(float(torch.trace(m)), 1e-30) + 1.0
            while energy(solve(hi)) > budget[h]:
                hi *= 10
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if energy(solve(mid)) > budget[h] else (lo, mid)
            mu, d = hi, solve(hi)
        out.append(d); mus.append(mu); energies.append(energy(d))
    return torch.stack(out), mus, energies


report = []
for t in range(LT):
    G = torch.zeros(H, K * D + 1, K * D + 1, device=dev, dtype=torch.float64)
    C = torch.zeros(H, K * D + 1, D, device=dev, dtype=torch.float64)
    M = torch.zeros_like(G)
    e_init = torch.zeros(H, device=dev, dtype=torch.float64); ns = 0
    r0 = 0.0
    for p in prep:
        n0, nq = len(p["hist"]), len(p["qids"])
        pos = torch.arange(n0, device=dev)
        cache = DynamicCache()
        for l in range(LT):   # history cache from the CURRENT mapper (earlier layers already corrected)
            k, v, _ = map_layer(l, p["ks"], p["vs"], pos)
            cache.update(k[None].to(torch.bfloat16), v[None].to(torch.bfloat16), l)
        T(torch.tensor([p["qids"]], device=dev), past_key_values=cache, position_ids=torch.arange(n0, n0 + nq, device=dev)[None])
        qq, kq, vq = q_states(t, attn_in[t], n0, nq)
        kmap, vmap, xv = map_layer(t, p["ks"], p["vs"], pos)
        positions = list(range(nq))   # all query positions
        kn, vn = p["nat"][t][0][0].to(dev).float(), p["nat"][t][1][0].to(dev).float()
        Pm = algebra.probabilities(qq, torch.cat([kmap, kq[0]], 1)[None], n0, positions)
        Pn = algebra.probabilities(qq, torch.cat([kn, kq[0]], 1)[None], n0, positions)
        R = (algebra.pool(Pn, torch.cat([vn, vq[0]], 1)) - algebra.pool(Pm, torch.cat([vmap, vq[0]], 1))).double()
        Xb = torch.cat([xv, torch.ones_like(xv[..., :1])], -1)
        g, c = algebra.gram(algebra.pool(Pm[..., :n0], Xb).double(), R)
        G += g; C += c
        M += (Xb.double().transpose(1, 2) @ Xb.double()) / n0          # source-feature second moment
        s = p["samp"].to(dev)
        e_init += (vmap[:, s] - vn[:, s]).double().square().sum((1, 2)); ns += len(s)   # current (= initial) V map of layer t
        r0 += R.square().sum().item()
        del cache, kmap, vmap, xv, Pm, Pn, R, Xb, kn, vn, qq, kq, vq
        attn_in.clear(); torch.cuda.empty_cache()
    G /= len(prep); C /= len(prep); M /= len(prep)
    M += 1e-4 * M.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None] * torch.eye(M.shape[-1], device=dev, dtype=M.dtype)
    E0 = e_init / ns
    dTheta, mus, energies = constrained_solve(G, C, M, (FRAC * E0).tolist())
    dTheta = dTheta.float().cpu()
    layers[t]["v_weight"] = layers[t]["v_weight"] + dTheta[:, :-1]
    layers[t]["v_bias"] = layers[t]["v_bias"] + dTheta[:, -1]
    report.append(dict(layer=t, readout_residual=r0 / len(prep), E0=E0.tolist(), correction_energy=energies, mu=mus,
                       constrained_heads=sum(m > 0 for m in mus)))
    log("layer", t, "readout residual", round(r0 / len(prep), 3), "E0", round(float(E0.mean()), 4),
        "correction energy / E0", round(sum(energies) / max(float(E0.sum()), 1e-30), 5), "constrained heads", sum(m > 0 for m in mus))
for h in hooks:
    h.remove()
save_mapper(OUT, layers, dict(note="initial fit + energy-constrained readout V calibration (Appendix B)", frac=FRAC,
                              n_pairs=len(pairs), calib=report))
log("FITCB4DONE", OUT)
