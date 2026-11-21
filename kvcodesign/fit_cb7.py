# VARIANT of CacheBridge (all-head map + CacheBridge's energy-constrained value calibration), fast: the calibration of
# fit_cb5.py computed layer by layer as in fit_cb6.py, with a cheaper constrained solver.
#   initial map : kvbridge-format all-head map (INIT, e.g. kvbridge8_8to14 = official CFM functions, 8 source layers);
#                 every target KV head is corrected from ALL source V features of its layers (K*H*D + 1 = 8193 for k=8).
#   order       : question hidden states kept at layer t's input; after layer t is solved they are advanced through it with
#                 the CALIBRATED map (each layer runs once; fit_cb5.py re-ran the whole target per layer).  A built-in check
#                 compares this path with the full-model path on two questions.
#   solver      : per head, dTheta(mu) = (G + lam I + mu M)^-1 C, tr(dTheta^T M dTheta) <= FRAC * E0, as fit_cb5.py, but
#                 each solve = fp32 Cholesky + fp64 iterative refinement (no 8193-dim fp64 eigendecomposition) and mu found by
#                 bisection on log(mu); head 0 is checked against a direct fp64 solve.
#   budget stats: STATS=fineweb -> M and E0 on the 4096 sampled fitting states (paper App. B; cb_fast.py budget_samples.pt);
#                 otherwise on the QuALITY history tokens (= fit_cb5.py, used to check this code against fit_cb5.py).
# The correction is folded into the kvbridge-format map (W_v, mu_yv): online cost = the initial map's.
# env: DIR (17to4|8to14) INIT OUT FRAC(0.01) PROMPT(old|cb) TRUNC(6000; 0 = complete articles) STATS(""|fineweb) BS
import os, sys, json, time, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

E = os.environ.get
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
HOME = os.path.expanduser("~")
from kvbridge import Mapper
from kvbridge.rope import strip_rope, apply_rope, build_cos_sin
from cachebridge.offline.calibration import algebra
sys.path.insert(0, f"{HOME}/kvcodesign")
from tasks import quality_cb_text, chat_split

DIR = E("DIR", "17to4")
NQ, FRAC = int(E("NQ_DOCS", 24)), float(E("FRAC", 0.01))
SRC_M, TGT_M = {"17to4": ("Qwen3-1.7B", "Qwen3-4B"), "8to14": ("Qwen3-8B-ms", "Qwen3-14B-ms")}[DIR]
INIT, OUT = os.path.expanduser(E("INIT")), os.path.expanduser(E("OUT"))
PROMPT, TRUNC, STATS = E("PROMPT", "old"), int(E("TRUNC", 6000)), E("STATS", "")
BS = os.path.expanduser(E("BS", f"~/kvbase/stats/{DIR}_full500/budget_samples.pt"))
MODELS = f"{HOME}/kvswitch/models"
dev, THETA, NSAMP = "cuda:0", 1e6, 4096
t_start = time.time()
log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{(time.time() - t_start) / 60:.1f} min]", *a, flush=True)
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False

tok = AutoTokenizer.from_pretrained(f"{MODELS}/{SRC_M}")
S = AutoModelForCausalLM.from_pretrained(f"{MODELS}/{SRC_M}", torch_dtype=torch.bfloat16).to(dev).eval()
if DIR == "8to14":
    T = AutoModelForCausalLM.from_pretrained(f"{MODELS}/{TGT_M}", torch_dtype=torch.bfloat16, device_map="auto",
                                             max_memory={0: "0GiB", 1: "20GiB", 2: "20GiB"}).eval()
else:
    T = AutoModelForCausalLM.from_pretrained(f"{MODELS}/{TGT_M}", torch_dtype=torch.bfloat16).to(dev).eval()
LS, LT = S.config.num_hidden_layers, T.config.num_hidden_layers
H, D = T.config.num_key_value_heads, 128
TDEV = [next(T.model.layers[l].parameters()).device for l in range(LT)]
EDEV, RDEV = T.model.embed_tokens.weight.device, T.model.norm.weight.device
mp = Mapper.load(INIT, device="cpu", dtype=torch.float32)
log("initial map", INIT, "source layers per target layer", len(mp.layers[0]["source_layers"]))
enc = lambda x: tok(x, add_special_tokens=False).input_ids

# ---------------- calibration data: 24 QuALITY documents x 4 questions (as fit_cb4/5/6.py) ---------------------------
from datasets import load_dataset
arts = {}
for r in load_dataset("emozilla/quality", split="validation"):
    arts.setdefault(r["article"], []).append(r)
docs = [v for v in arts.values() if len(v) >= 4]
random.Random(0).shuffle(docs)
rng = torch.Generator().manual_seed(1)
DOCS, PAIRS = [], []
for di, d in enumerate(docs[:NQ]):
    if PROMPT == "cb":
        hist = chat_split(enc, *quality_cb_text(d[0]["article"], d[0]))[0]
    else:
        hist = tok(f"<|im_start|>user\nRead the following article.\n\n{d[0]['article']}", add_special_tokens=False).input_ids
    hist = hist[:TRUNC] if TRUNC else hist
    DOCS.append(dict(hist=hist))
    for r in d[:4]:
        if PROMPT == "cb":
            q = chat_split(enc, *quality_cb_text(d[0]["article"], r))[1]
        else:
            opts = "\n".join(f"{c}. {o}" for c, o in zip("ABCD", r["options"]))
            q = tok(f"\n\nQuestion: {r['question']}\n{opts}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:",
                    add_special_tokens=False).input_ids
        PAIRS.append(dict(doc=di, qids=q, samp=torch.randperm(len(hist), generator=rng)[:NSAMP // (4 * NQ) + 1]))
log("prompt", PROMPT, "truncation", TRUNC or "none", "budget statistics", STATS or "QuALITY history", "pairs", len(PAIRS))

for d in DOCS:   # history-side tensors once per document; source K content-space, all heads flattened per layer
    hist = d["hist"]; n0 = len(hist)
    out = S(torch.tensor([hist], device=dev), use_cache=True)
    pos = torch.arange(n0, device=dev)
    d["ks"] = [strip_rope(out.past_key_values.layers[l].keys[0].float(), pos, THETA).transpose(0, 1).reshape(n0, H * D).to(torch.bfloat16).cpu().pin_memory() for l in range(LS)]
    d["vs"] = [out.past_key_values.layers[l].values[0].transpose(0, 1).reshape(n0, H * D).cpu().pin_memory() for l in range(LS)]
    del out
    nat = T.model(torch.tensor([hist], device=EDEV), use_cache=True).past_key_values
    d["nat"] = [(nat.layers[l].keys[0].cpu().pin_memory(), nat.layers[l].values[0].cpu().pin_memory()) for l in range(LT)]
    del nat
log("prepared source and re-prefill KV")
BSD = torch.load(BS) if STATS == "fineweb" else None


def map_layer(t, d, dv):   # kvbridge arithmetic y = (x - mu_x) W + mu_y -> rotated K [H,n,D], V [H,n,D], V features [n, P-1]
    e = mp.layers[t]
    xk = torch.cat([d["ks"][s].to(dv, non_blocking=True).float() for s in e["source_layers"]], -1)
    xv = torch.cat([d["vs"][s].to(dv, non_blocking=True).float() for s in e["source_layers"]], -1)
    n = xk.shape[0]
    k = ((xk - e["mu_xk"].to(dv)) @ e["W_k"].to(dv) + e["mu_yk"].to(dv)).view(n, H, D).transpose(0, 1)
    v = ((xv - e["mu_xv"].to(dv)) @ e["W_v"].to(dv) + e["mu_yv"].to(dv)).view(n, H, D).transpose(0, 1)
    return apply_rope(k, torch.arange(n, device=dv), THETA), v, xv


def q_states(t, hidden, n0, nq, dv):
    a = T.model.layers[t].self_attn
    hsh = (1, nq, -1, D)
    qq = a.q_norm(a.q_proj(hidden).view(hsh)).transpose(1, 2)
    kk = a.k_norm(a.k_proj(hidden).view(hsh)).transpose(1, 2)
    vv = a.v_proj(hidden).view(hsh).transpose(1, 2)
    cos, sin = build_cos_sin(torch.arange(n0, n0 + nq, device=dv), D, THETA, dv, torch.float32)
    rot = lambda x: (x.float() * cos + torch.cat([-x[..., D // 2:], x[..., :D // 2]], -1).float() * sin)
    return rot(qq), rot(kk), vv.float()


class OneLayerCache:
    def __init__(self, k, v):
        self.k, self.v = k, v

    def update(self, k, v, layer_idx, cache_kwargs=None):
        return torch.cat([self.k, k], 2), torch.cat([self.v, v], 2)


def advance(t, h, kmap, vmap, n0):
    dv = TDEV[t]; nq = h.shape[1]
    torch.cuda.set_device(dv)
    h = h.to(dv)
    pid = torch.arange(n0, n0 + nq, device=dv)[None]
    rd = T.model.rotary_emb.inv_freq.device
    pe = tuple(x.to(dv) for x in T.model.rotary_emb(h.to(rd), pid.to(rd)))
    mask = torch.zeros(1, 1, nq, n0 + nq, device=dv, dtype=h.dtype)
    mask[..., n0:] = torch.triu(torch.full((nq, nq), torch.finfo(h.dtype).min, device=dv, dtype=h.dtype), 1)
    out = T.model.layers[t](h, attention_mask=mask, position_ids=pid, past_key_value=OneLayerCache(kmap[None].to(torch.bfloat16), vmap[None].to(torch.bfloat16)),
                            use_cache=True, cache_position=pid[0], position_embeddings=pe)
    return out[0] if isinstance(out, tuple) else out


for p in PAIRS[:2]:   # check: layer-by-layer path == full-model path (initial map)
    d = DOCS[p["doc"]]; n0, nq = len(d["hist"]), len(p["qids"])
    cache = DynamicCache(); kv = []
    for l in range(LT):
        torch.cuda.set_device(TDEV[l])
        k, v, _ = map_layer(l, d, TDEV[l]); kv.append((k, v))
        cache.update(k[None].to(torch.bfloat16), v[None].to(torch.bfloat16), l)
    full = T(torch.tensor([p["qids"]], device=EDEV), past_key_values=cache, position_ids=torch.arange(n0, n0 + nq, device=EDEV)[None]).logits[0, -1].float()
    h = T.model.embed_tokens(torch.tensor([p["qids"]], device=EDEV))
    for l in range(LT):
        h = advance(l, h, *kv[l], n0)
    seq = T.lm_head(T.model.norm(h.to(RDEV)))[0, -1].float()
    full, seq = full.log_softmax(-1), seq.to(full.device).log_softmax(-1)
    log("check layer-by-layer vs full model: KL", float((full.exp() * (full - seq)).sum()), "same argmax", bool(full.argmax() == seq.argmax()))
    del cache, kv


def constrained_solve(G, C, M, budget, dv):
    """per head h: d(mu) = (G_h + lam_h I + mu M)^-1 C_h, smallest mu >= 0 with tr(d^T M d) <= budget_h.
    Each solve: fp32 Cholesky + 4 steps of fp64 iterative refinement; mu by bisection on log(mu)."""
    p = G.shape[-1]
    m = (0.5 * (M + M.T)).to(dv, torch.float64)
    out, mus, energies = [], [], []
    for h in range(G.shape[0]):
        g = G[h].to(dv, torch.float64); g = 0.5 * (g + g.T)
        lam = max(1e-8, 0.01 * float(torch.trace(g)) / p)
        c = C[h].to(dv, torch.float64)

        def system(mu):   # G + lam I + mu M, no identity matrix materialised
            A = g + mu * m
            A.diagonal().add_(lam)
            return A

        def solve(mu, refine=True):   # the mu search uses fp32 solves only; the final mu is refined in fp64
            A = system(mu)
            L, info = torch.linalg.cholesky_ex(A.float())
            if int(info) != 0:   # fp32 factorisation lost definiteness (large mu * M): factorise in fp64 instead
                return torch.cholesky_solve(c, torch.linalg.cholesky(A))
            d = torch.cholesky_solve(c.float(), L).double()
            for _ in range(4 if refine else 0):
                d += torch.cholesky_solve((c - A @ d).float(), L).double()
            return d
        energy = lambda d: float((d * (m @ d)).sum())
        d = solve(0.0); mu = 0.0
        if energy(d) > budget[h]:
            hi = float(torch.trace(g)) / float(torch.trace(m))
            while energy(solve(hi, False)) > budget[h]:
                hi *= 10
            lo = hi / 10
            while energy(solve(lo, False)) <= budget[h] and lo > 1e-30:
                lo /= 10
            for _ in range(22):
                mid = (lo * hi) ** 0.5
                lo, hi = (mid, hi) if energy(solve(mid, False)) > budget[h] else (lo, mid)
            mu = hi; d = solve(mu)
        if h == 0:
            ref = torch.linalg.solve(system(mu), c)
            rel = float((d - ref).norm() / ref.norm().clamp_min(1e-30))
        out.append(d.cpu()); mus.append(mu); energies.append(energy(d))
        del g, c
    return torch.stack(out), mus, energies, rel


hs = [T.model.embed_tokens(torch.tensor([p["qids"]], device=EDEV)) for p in PAIRS]
report = []
for t in range(LT):
    dv = TDEV[t]; torch.cuda.set_device(dv)
    e = mp.layers[t]
    P = len(e["source_layers"]) * H * D + 1
    G = torch.zeros(H, P, P, device=dv, dtype=torch.float32)
    C = torch.zeros(H, P, D, device=dv, dtype=torch.float64)
    M = torch.zeros(P, P, device=dv, dtype=torch.float64)   # fp64 accumulation (12-layer maps: 12289 features)
    e_init = torch.zeros(H, device=dv, dtype=torch.float64); ns = 0; r0 = 0.0
    for di, d in enumerate(DOCS):
        n0 = len(d["hist"])
        kmap, vmap, xv = map_layer(t, d, dv)
        kn, vn = d["nat"][t][0].to(dv).float(), d["nat"][t][1].to(dv).float()
        Xb = torch.cat([xv, torch.ones_like(xv[:, :1])], -1)
        Mdoc = (Xb.T @ Xb).double() / n0 if BSD is None else None   # same for the 4 questions of a document
        for pi, p in enumerate(PAIRS):
            if p["doc"] != di:
                continue
            nq = len(p["qids"])
            hidden = T.model.layers[t].input_layernorm(hs[pi].to(dv))
            qq, kq, vq = q_states(t, hidden, n0, nq, dv)
            positions = list(range(nq))
            Pm = algebra.probabilities(qq, torch.cat([kmap, kq[0]], 1)[None], n0, positions)
            Pn = algebra.probabilities(qq, torch.cat([kn, kq[0]], 1)[None], n0, positions)
            R = (algebra.pool(Pn, torch.cat([vn, vq[0]], 1)) - algebra.pool(Pm, torch.cat([vmap, vq[0]], 1))).double()
            A = torch.einsum("hqn,np->hqp", Pm[..., :n0], Xb)
            G.baddbmm_(A.transpose(1, 2), A, alpha=1.0 / A.shape[1])
            C += (A.double().transpose(1, 2) @ R) / A.shape[1]
            if BSD is None:
                M += Mdoc
                s = p["samp"].to(dv)
                e_init += (vmap[:, s] - vn[:, s]).double().square().sum((1, 2)); ns += len(s)
            r0 += R.square().sum().item()
            del Pm, Pn, R, A, qq, kq, vq, hidden
        del kmap, vmap, xv, kn, vn, Xb, Mdoc
    G /= len(PAIRS); C /= len(PAIRS)
    if BSD is None:
        M /= len(PAIRS); E0 = e_init / ns
    else:   # paper: token averages over the 4096 sampled fitting states, per layer and KV head
        xs = torch.cat([BSD["src_v"][s].reshape(-1, H * D).to(dv).float() for s in e["source_layers"]], -1)
        vm = ((xs - e["mu_xv"].to(dv)) @ e["W_v"].to(dv) + e["mu_yv"].to(dv)).view(-1, H, D)
        E0 = (vm - BSD["tgt_v"][t].to(dv).float()).double().square().sum(-1).mean(0)
        Xs = torch.cat([xs, torch.ones_like(xs[:, :1])], -1).double()
        M = (Xs.T @ Xs) / Xs.shape[0]   # fp64: 4096 states < 8193 features, so M is PD only through the 1e-4 ridge,
                                        # which fp32 rounding of the large eigenvalues (~1e-7 * max) would swamp
        del xs, vm, Xs
    M.diagonal().add_(1e-4 * M.diagonal().mean())
    G = G.cpu(); torch.cuda.empty_cache()   # heads are moved back one at a time (room next to the 14B layers)
    dTheta, mus, energies, rel = constrained_solve(G, C, M, (FRAC * E0).tolist(), dv)
    del G, C, M; torch.cuda.empty_cache()
    dW = dTheta[:, :-1].permute(1, 0, 2).reshape(P - 1, H * D).float()
    db = dTheta[:, -1].reshape(H * D).float()
    e["mu_yv"] = e["mu_yv"] + e["mu_xv"] @ dW + db
    e["W_v"] = e["W_v"] + dW
    for di, d in enumerate(DOCS):
        kmap, vmap, _ = map_layer(t, d, dv)
        for pi, p in enumerate(PAIRS):
            if p["doc"] == di:
                hs[pi] = advance(t, hs[pi], kmap, vmap, len(d["hist"]))
        del kmap, vmap
    torch.cuda.empty_cache()
    report.append(dict(layer=t, readout_residual=r0 / len(PAIRS), E0=E0.tolist(), correction_energy=energies, mu=mus, solver_rel_diff_head0=rel))
    log("layer", t, "readout residual", round(r0 / len(PAIRS), 3), "E0", round(float(E0.mean()), 4),
        "correction energy / E0", round(sum(energies) / max(float(E0.sum()), 1e-30), 5),
        "constrained heads", sum(x > 0 for x in mus), "solver check", f"{rel:.1e}")
mp.meta["method"] = "VARIANT: all-head map + CacheBridge energy-constrained readout V calibration (fit_cb7.py, layer by layer)"
mp.meta["calibration_frac"] = FRAC
mp.save(OUT)
json.dump(dict(init=INIT, frac=FRAC, prompt=PROMPT, trunc=TRUNC, stats=STATS or "quality", minutes=(time.time() - t_start) / 60,
               calib=report), open(f"{OUT}/calibration_report.json", "w"), indent=1)
log("FITCB7DONE", OUT)
