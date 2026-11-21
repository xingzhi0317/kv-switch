# VARIANT of CacheBridge (not the paper's method; labelled as a variant everywhere):
#   initial map  = ALL-HEAD per-layer linear map (official kvbridge fit, top-3 source layers: mappers/kvbridge3_{DIR},
#                  the same initialiser as the KL-trained translator) instead of CacheBridge's head-local map, because
#                  for Qwen3-1.7B <-> 4B same-index KV heads do not correspond (REPORT sec. 12);
#   calibration  = CacheBridge's energy-constrained attention-readout value calibration (paper Appendix B, as in
#                  fit_cb4.py), each target KV head now corrected from ALL source V features of its 3 source layers
#                  (3072 + intercept) instead of its own head's.
# The correction is folded into the kvbridge-format map (W_v, mu_yv), so the online cost equals the 3-layer map (kb3).
# Output: ~/kvbase/mappers/kvbridge3cb_{DIR}/ (kvbridge format; eval_quality.py method "kb3cb")
# env: DIR (17to4|4to17) NQ_DOCS(24) FRAC(0.01)
import os, sys, json, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

E = os.environ.get
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
HOME = os.path.expanduser("~")
sys.path.insert(0, f"{HOME}/kvbase/kvbridge")
from kvbridge import Mapper, transfer
from kvbridge.rope import strip_rope, apply_rope, build_cos_sin
from cachebridge.offline.calibration import algebra

NQ, FRAC = int(E("NQ_DOCS", 24)), float(E("FRAC", 0.01))
DIR = E("DIR", "17to4")
SRC_M, TGT_M = {"17to4": ("Qwen3-1.7B", "Qwen3-4B"), "4to17": ("Qwen3-4B", "Qwen3-1.7B")}[DIR]
# INIT: any kvbridge-format map, e.g. kvbridge_{DIR} (official 12 source layers) -> OUT kvbridge12cb_{DIR}
INIT = E("INIT", f"{HOME}/kvbase/mappers/kvbridge3_{DIR}")
OUT = E("OUT", f"{HOME}/kvbase/mappers/kvbridge3cb_{DIR}")
MODELS = f"{HOME}/kvswitch/models"
dev, THETA, NSAMP = "cuda", 1e6, 4096
log = lambda *a: print(*a, flush=True)

tok = AutoTokenizer.from_pretrained(f"{MODELS}/Qwen3-1.7B")
S = AutoModelForCausalLM.from_pretrained(f"{MODELS}/{SRC_M}", torch_dtype=torch.bfloat16).to(dev).eval()
T = AutoModelForCausalLM.from_pretrained(f"{MODELS}/{TGT_M}", torch_dtype=torch.bfloat16).to(dev).eval()
S.requires_grad_(False); T.requires_grad_(False)
torch.set_grad_enabled(False)
LS, LT = S.config.num_hidden_layers, T.config.num_hidden_layers
H, D = T.config.num_key_value_heads, 128
mp = Mapper.load(INIT, device=dev, dtype=torch.float32)
for e in mp.layers:   # K maps are never modified: keep them in bf16 (the official weights are stored in bf16 anyway);
    e["W_k"] = e["W_k"].to(torch.bfloat16)   # only W_v (calibrated) stays float32.  Saves 1.8 GB at 12 source layers.
ref_mp = Mapper.load(INIT, device=dev)          # untouched copy for the parity check (freed right after it)
assert len(mp.layers) == LT

# ---------------- calibration data: 24 QuALITY documents x 4 questions (as fit_cb4.py / the paper) ----------------
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


def source_kv(hist):   # content-space source K (RoPE stripped) and V, per layer [n, H*D] (head-major), bf16 on CPU
    out = S(torch.tensor([hist], device=dev), use_cache=True)
    pos = torch.arange(len(hist), device=dev)
    n = len(hist)
    ks = [strip_rope(out.past_key_values.layers[l].keys[0].float(), pos, THETA).transpose(0, 1).reshape(n, -1)
          .to(torch.bfloat16).cpu().pin_memory() for l in range(LS)]
    vs = [out.past_key_values.layers[l].values[0].transpose(0, 1).reshape(n, -1).cpu().pin_memory() for l in range(LS)]
    return ks, vs, out.past_key_values


def map_layer(t, ks, vs, pos):   # kvbridge arithmetic: y = (x - mu_x) W + mu_y; keys get target RoPE
    e = mp.layers[t]
    xk = torch.cat([ks[s].to(dev, non_blocking=True).float() for s in e["source_layers"]], -1)
    xv = torch.cat([vs[s].to(dev, non_blocking=True).float() for s in e["source_layers"]], -1)   # [n, 3*H*D]
    n = xk.shape[0]
    k = ((xk - e["mu_xk"]) @ e["W_k"].float() + e["mu_yk"]).view(n, H, D).transpose(0, 1)
    v = ((xv - e["mu_xv"]) @ e["W_v"] + e["mu_yv"]).view(n, H, D).transpose(0, 1)
    return apply_rope(k, pos, THETA), v, xv


prep = []
for i, (hist, qids) in enumerate(pairs):
    nat = T(torch.tensor([hist], device=dev), use_cache=True).past_key_values
    ks, vs, sc = source_kv(hist)
    if i == 0:   # parity: our re-implementation of the map == the official kvbridge transfer
        off = transfer(sc, ref_mp)
        pos = torch.arange(len(hist), device=dev)
        for t in (0, LT // 2, LT - 1):
            k, v, _ = map_layer(t, ks, vs, pos)
            dk = (k - off.layers[t].keys[0].float()).norm() / off.layers[t].keys[0].float().norm()
            dv = (v - off.layers[t].values[0].float()).norm() / off.layers[t].values[0].float().norm()
            log("parity layer", t, "rel diff K", round(float(dk), 5), "V", round(float(dv), 5))
            assert dk < 0.02 and dv < 0.02, "re-implemented map differs from the official transfer"
        del off, ref_mp
        torch.cuda.empty_cache()
    del sc
    samp = torch.randperm(len(hist), generator=rng)[:NSAMP // len(pairs) + 1]
    prep.append(dict(hist=hist, qids=qids, ks=ks, vs=vs, samp=samp,
                     nat=[(nat.layers[l].keys.cpu().pin_memory(), nat.layers[l].values.cpu().pin_memory()) for l in range(LT)]))
    del nat
log("prepared")
del S; torch.cuda.empty_cache()   # the source model is not needed after its caches are taken (frees ~3.4 GB)

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
    cos, sin = build_cos_sin(torch.arange(n0, n0 + nq, device=dev), D, THETA, dev, torch.float32)
    rot = lambda x: (x.float() * cos + torch.cat([-x[..., D // 2:], x[..., :D // 2]], -1).float() * sin)
    return rot(qq), rot(kk), vv.float()


def constrained_solve(G, C, M, budget):
    """Same problem as fit_cb4.py -- per head dTheta(mu) = (G + lam I + mu M)^-1 C with tr(dTheta^T M dTheta) <= budget --
    solved exactly via one generalised eigendecomposition per head (M = L L^T, L^-1 (G + lam I) L^-T = U diag(s) U^T):
    dTheta(mu) = L^-T U diag(1/(s+mu)) U^T L^-1 C  and  energy(mu) = sum_i |(U^T L^-1 C)_i|^2 / (s_i+mu)^2, so the mu
    search costs nothing.  G (float32, possibly CPU) and M are moved to the GPU in float64 one head at a time."""
    p = G.shape[-1]
    m = M.to(dev, torch.float64); m = 0.5 * (m + m.T)
    L = torch.linalg.cholesky(m)
    out, mus, energies = [], [], []
    for h in range(G.shape[0]):
        g = G[h].to(dev, torch.float64); g = 0.5 * (g + g.T)
        lam = max(1e-8, 0.01 * float(torch.trace(g)) / p)
        g.diagonal().add_(lam)
        A = torch.linalg.solve_triangular(L, torch.linalg.solve_triangular(L, g, upper=False).T, upper=False)   # L^-1 g L^-T
        del g
        s, U = torch.linalg.eigh(0.5 * (A + A.T)); del A
        Bt = U.T @ torch.linalg.solve_triangular(L, C[h].to(dev, torch.float64), upper=False)            # U^T L^-1 C  [p, D]
        w = Bt.square().sum(1)
        energy = lambda mu: float((w / (s + mu) ** 2).sum())
        mu = 0.0
        if energy(0.0) > budget[h]:
            lo, hi = 0.0, 1.0
            while energy(hi) > budget[h]:
                hi *= 10
            for _ in range(100):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if energy(mid) > budget[h] else (lo, mid)
            mu = hi
        d = torch.linalg.solve_triangular(L.T, U @ (Bt / (s + mu)[:, None]), upper=True)
        if h == 0:   # check against the direct solve of the same system
            ref = torch.linalg.solve(G[0].to(dev, torch.float64) * 0.5 + G[0].to(dev, torch.float64).T * 0.5
                                     + lam * torch.eye(p, device=dev, dtype=torch.float64) + mu * m, C[0].to(dev, torch.float64))
            log("  solver check head 0: rel diff", float((d - ref).norm() / ref.norm().clamp_min(1e-30)))
            del ref
        out.append(d.cpu()); mus.append(mu); energies.append(energy(mu))
        del U, s, Bt, d; torch.cuda.empty_cache()
    return torch.stack(out).to(dev), mus, energies


report = []
for t in range(LT):
    P = len(mp.layers[t]["source_layers"]) * H * D + 1
    G = torch.zeros(H, P, P, device=dev, dtype=torch.float32)    # float32 accumulation (12 source layers: 4.8 GB)
    C = torch.zeros(H, P, D, device=dev, dtype=torch.float64)
    M = torch.zeros(P, P, device=dev, dtype=torch.float32)       # source features are shared by all target heads
    e_init = torch.zeros(H, device=dev, dtype=torch.float64); ns = 0; r0 = 0.0
    for p in prep:
        n0, nq = len(p["hist"]), len(p["qids"])
        pos = torch.arange(n0, device=dev)
        cache = DynamicCache()
        for l in range(LT):   # history cache from the CURRENT map (earlier layers already corrected)
            k, v, _ = map_layer(l, p["ks"], p["vs"], pos)
            cache.update(k[None].to(torch.bfloat16), v[None].to(torch.bfloat16), l)
        T(torch.tensor([p["qids"]], device=dev), past_key_values=cache, position_ids=torch.arange(n0, n0 + nq, device=dev)[None])
        qq, kq, vq = q_states(t, attn_in[t], n0, nq)
        kmap, vmap, xv = map_layer(t, p["ks"], p["vs"], pos)
        positions = list(range(nq))
        kn, vn = p["nat"][t][0][0].to(dev).float(), p["nat"][t][1][0].to(dev).float()
        Pm = algebra.probabilities(qq, torch.cat([kmap, kq[0]], 1)[None], n0, positions)
        Pn = algebra.probabilities(qq, torch.cat([kn, kq[0]], 1)[None], n0, positions)
        R = (algebra.pool(Pn, torch.cat([vn, vq[0]], 1)) - algebra.pool(Pm, torch.cat([vmap, vq[0]], 1))).double()
        Xb = torch.cat([xv, torch.ones_like(xv[:, :1])], -1)                          # [n0, P] all source heads + 1
        A = torch.einsum("hqn,np->hqp", Pm[..., :n0], Xb)                             # attention-pooled features
        G.baddbmm_(A.transpose(1, 2), A, alpha=1.0 / A.shape[1])                      # = algebra.gram(A, R)[0], in place
        C += (A.double().transpose(1, 2) @ R) / A.shape[1]
        M.addmm_(Xb.T, Xb, alpha=1.0 / n0)
        s = p["samp"].to(dev)
        e_init += (vmap[:, s] - vn[:, s]).double().square().sum((1, 2)); ns += len(s)
        r0 += R.square().sum().item()
        del cache, kmap, vmap, xv, Pm, Pn, R, Xb, A, kn, vn, qq, kq, vq
        attn_in.clear(); torch.cuda.empty_cache()
    G /= len(prep); C /= len(prep); M /= len(prep)
    M += 1e-4 * M.diagonal().mean() * torch.eye(P, device=dev, dtype=M.dtype)
    E0 = e_init / ns
    G = G.cpu(); T.to("cpu"); torch.cuda.empty_cache()   # room for the fp64 eigendecompositions (~9.3 GB at 12 layers)
    dTheta, mus, energies = constrained_solve(G, C, M, (FRAC * E0).tolist())                     # [H, P, D]
    del G
    T.to(dev); torch.cuda.empty_cache()
    e = mp.layers[t]
    dW = dTheta[:, :-1].permute(1, 0, 2).reshape(P - 1, H * D).float()                         # columns head-major
    db = dTheta[:, -1].reshape(H * D).float()
    e["mu_yv"] = e["mu_yv"] + e["mu_xv"] @ dW + db    # y = (x - mu_x)(W + dW) + mu_y'  ==  old + x dW + db
    e["W_v"] = e["W_v"] + dW
    report.append(dict(layer=t, readout_residual=r0 / len(prep), E0=E0.tolist(), correction_energy=energies, mu=mus))
    log("layer", t, "readout residual", round(r0 / len(prep), 3), "E0", round(float(E0.mean()), 4),
        "correction energy / E0", round(sum(energies) / max(float(E0.sum()), 1e-30), 5), "constrained heads", sum(m > 0 for m in mus))
for h in hooks:
    h.remove()
mp.meta["method"] = ("VARIANT: all-head per-layer map (official kvbridge fit, top-3) + CacheBridge energy-constrained "
                     "attention-readout value calibration (arXiv 2609.00891 App. B, re-implemented)")
mp.meta["calibration_frac"] = FRAC
mp.save(OUT)
json.dump(report, open(f"{OUT}/calibration_report.json", "w"))
log("FITCB5DONE", OUT)
