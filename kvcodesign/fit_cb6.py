# CacheBridge energy-constrained value calibration (arXiv 2609.00891, App. B) -- same objective and data as fit_cb4.py,
# computed layer by layer instead of re-running the whole target model once per calibrated layer:
#   fit_cb4.py: for every target layer t, rebuild the mapped history cache of ALL layers and run the full target on the
#               question -> O(L^2) layer evaluations, ~1 TB of host->GPU copies per layer (1.7B->4B: ~2 h).
#   here      : the question tokens' hidden states h_p are kept at the input of layer t.  Layer t's statistics come from
#               h_p; after its V correction is solved, h_p is advanced through layer t with the CALIBRATED mapped history
#               KV of layer t.  Layer t only ever sees layers < t, so this is the same computation as fit_cb4.py
#               (each layer runs once; history-side tensors are stored once per document, not once per question).
#   solver    : exact per-head generalised eigendecomposition + scalar search on mu (fit_cb5.py), same problem as the
#               per-head bisection with dense solves in fit_cb4.py.
# A built-in check compares the layer-by-layer path with the full-model path on two questions before calibrating.
# DIR: 17to4 (Qwen3-1.7B -> 4B, one GPU) | 8to14 (Qwen3-8B -> 14B: source on cuda:0, target on cuda:1,2)
# Alignment options (paper App. A/B):  PROMPT=cb  -> CacheBridge's QuALITY layout (tasks.quality_cb_text, the same text the
#   quality_cb evaluation uses);  TRUNC=0 -> complete articles (default 6000 = fit_cb4.py);  STATS=<file> -> M and E0 from
#   the initialization data (fit_cb_init_big.py *_stats.pt) instead of the QuALITY history tokens.
# env: DIR K(8) NQ_DOCS(24) FRAC(0.01) INIT (initial mapper dir) OUT PROMPT TRUNC STATS
import os, gc, json, time, random, torch
from safetensors.torch import save_file, load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

E = os.environ.get
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from kvbridge.rope import strip_rope, apply_rope, build_cos_sin
from cachebridge.offline.calibration import algebra

DIR = E("DIR", "17to4")
NQ, FRAC = int(E("NQ_DOCS", 24)), float(E("FRAC", 0.01))
SRC_M, TGT_M = {"17to4": ("Qwen3-1.7B", "Qwen3-4B"), "8to14": ("Qwen3-8B-ms", "Qwen3-14B-ms")}[DIR]
INIT = E("INIT", os.path.expanduser({"17to4": "~/kvbase/mappers/cachebridge_17to4_ec_init",
                                     "8to14": "~/kvbase/mappers/cachebridge_8to14_init"}[DIR]))
OUT = E("OUT", os.path.expanduser(f"~/kvbase/mappers/cachebridge_{DIR}_ec6"))
MODELS = os.path.expanduser("~/kvswitch/models")
dev, THETA, NSAMP = "cuda:0", 1e6, 4096
t_start = time.time()
log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{(time.time() - t_start) / 60:.1f} min]", *a, flush=True)
torch.set_grad_enabled(False)

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
EDEV = T.model.embed_tokens.weight.device
RDEV = T.model.norm.weight.device

layers = [load_file(f"{INIT}/layer_{t:03d}.safetensors") for t in range(LT)]
K = len(layers[0]["selected"])
support = [layers[t]["selected"].tolist() for t in range(LT)]
log("loaded initial mapper", INIT, "K", K, "layers", LT)

PROMPT, TRUNC, STATS = E("PROMPT", "old"), int(E("TRUNC", 6000)), E("STATS", "")
import sys
sys.path.insert(0, os.path.expanduser("~/kvcodesign"))
from tasks import quality_cb_text, chat_split
enc = lambda x: tok(x, add_special_tokens=False).input_ids
ST = torch.load(STATS) if STATS else None
log("prompt", PROMPT, "history truncation", TRUNC or "none", "budget statistics", STATS or "QuALITY history tokens")
# ---------------- calibration data: 24 QuALITY documents x 4 questions (paper), identical to fit_cb4.py -------------
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
log("calibration pairs", len(PAIRS), "documents", len(DOCS))

for d in DOCS:   # history-side tensors, once per document
    hist = d["hist"]; n0 = len(hist)
    out = S(torch.tensor([hist], device=dev), use_cache=True)
    pos = torch.arange(n0, device=dev)
    d["ks"] = [strip_rope(out.past_key_values.layers[l].keys[0].float(), pos, THETA).to(torch.bfloat16).cpu().pin_memory() for l in range(LS)]
    d["vs"] = [out.past_key_values.layers[l].values[0].cpu().pin_memory() for l in range(LS)]
    del out
    nat = T.model(torch.tensor([hist], device=EDEV), use_cache=True).past_key_values
    d["nat"] = [(nat.layers[l].keys[0].cpu().pin_memory(), nat.layers[l].values[0].cpu().pin_memory()) for l in range(LT)]
    del nat
log("prepared source and re-prefill KV")


def map_layer(t, d, dv):   # -> rotated K [H, n, D], V [H, n, D] (fp32), V features xv [H, n, K*D]
    L = layers[t]
    sel = L["selected"].tolist()
    xk = torch.cat([d["ks"][s].to(dv, non_blocking=True).float() for s in sel], -1)
    xv = torch.cat([d["vs"][s].to(dv, non_blocking=True).float() for s in sel], -1)
    k = torch.einsum("hnf,hfd->hnd", xk, L["k_weight"].to(dv)) + L["k_bias"].to(dv)[:, None]
    v = torch.einsum("hnf,hfd->hnd", xv, L["v_weight"].to(dv)) + L["v_bias"].to(dv)[:, None]
    return apply_rope(k, torch.arange(k.shape[1], device=dv), THETA), v, xv


def q_states(t, hidden, n0, nq, dv):   # as fit_cb4.py: rotated q, k and v of the question tokens at layer t
    a = T.model.layers[t].self_attn
    hsh = (1, nq, -1, D)
    qq = a.q_norm(a.q_proj(hidden).view(hsh)).transpose(1, 2)
    kk = a.k_norm(a.k_proj(hidden).view(hsh)).transpose(1, 2)
    vv = a.v_proj(hidden).view(hsh).transpose(1, 2)
    cos, sin = build_cos_sin(torch.arange(n0, n0 + nq, device=dv), D, THETA, dv, torch.float32)
    rot = lambda x: (x.float() * cos + torch.cat([-x[..., D // 2:], x[..., :D // 2]], -1).float() * sin)
    return rot(qq), rot(kk), vv.float()


class OneLayerCache:   # what Qwen3Attention needs from a cache: prepend the mapped history K, V of this layer
    def __init__(self, k, v):
        self.k, self.v = k, v

    def update(self, k, v, layer_idx, cache_kwargs=None):
        return torch.cat([self.k, k], 2), torch.cat([self.v, v], 2)


def advance(t, h, kmap, vmap, n0):   # question hidden states through layer t, history = given mapped K, V
    dv = TDEV[t]; nq = h.shape[1]
    torch.cuda.set_device(dv)
    h = h.to(dv)
    pid = torch.arange(n0, n0 + nq, device=dv)[None]
    rd = T.model.rotary_emb.inv_freq.device
    pe = tuple(x.to(dv) for x in T.model.rotary_emb(h.to(rd), pid.to(rd)))
    mask = torch.zeros(1, 1, nq, n0 + nq, device=dv, dtype=h.dtype)
    mask[..., n0:] = torch.triu(torch.full((nq, nq), torch.finfo(h.dtype).min, device=dv, dtype=h.dtype), 1)
    cache = OneLayerCache(kmap[None].to(torch.bfloat16), vmap[None].to(torch.bfloat16))
    out = T.model.layers[t](h, attention_mask=mask, position_ids=pid, past_key_value=cache, use_cache=True,
                            cache_position=pid[0], position_embeddings=pe)
    return out[0] if isinstance(out, tuple) else out


# ---------------- check: layer-by-layer path == full-model path (initial mapper, 2 questions) -----------------------
for p in PAIRS[:2]:
    d = DOCS[p["doc"]]; n0, nq = len(d["hist"]), len(p["qids"])
    cache = DynamicCache(); kv = []
    for l in range(LT):
        k, v, _ = map_layer(l, d, TDEV[l]); kv.append((k, v))
        cache.update(k[None].to(torch.bfloat16), v[None].to(torch.bfloat16), l)
    full = T(torch.tensor([p["qids"]], device=EDEV), past_key_values=cache,
             position_ids=torch.arange(n0, n0 + nq, device=EDEV)[None]).logits[0, -1].float()
    h = T.model.embed_tokens(torch.tensor([p["qids"]], device=EDEV))
    for l in range(LT):
        h = advance(l, h, *kv[l], n0)
    seq = T.lm_head(T.model.norm(h.to(RDEV)))[0, -1].float()
    full, seq = full.log_softmax(-1), seq.to(full.device).log_softmax(-1)
    log("check layer-by-layer vs full model: KL", float((full.exp() * (full - seq)).sum()),
        "max |dlogp| top-10", float((full - seq)[full.topk(10).indices].abs().max()), "same argmax", bool(full.argmax() == seq.argmax()))
    del cache, kv


def constrained_solve(G, C, M, budget, dv):
    """per head: dTheta(mu) = (G + lam I + mu M)^-1 C,  tr(dTheta^T M dTheta) <= budget  (fit_cb4.py's problem),
    exact via M = L L^T, L^-1 (G + lam I) L^-T = U diag(s) U^T  ->  energy(mu) = sum_i w_i / (s_i + mu)^2."""
    p = G.shape[-1]
    out, mus, energies = [], [], []
    for h in range(G.shape[0]):
        g = 0.5 * (G[h] + G[h].T); m = 0.5 * (M[h] + M[h].T)
        lam = max(1e-8, 0.01 * float(torch.trace(g)) / p)
        L = torch.linalg.cholesky(m)
        gl = g + lam * torch.eye(p, device=dv, dtype=g.dtype)
        A = torch.linalg.solve_triangular(L, torch.linalg.solve_triangular(L, gl, upper=False).T, upper=False)
        s, U = torch.linalg.eigh(0.5 * (A + A.T))
        Bt = U.T @ torch.linalg.solve_triangular(L, C[h], upper=False)
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
        dd = torch.linalg.solve_triangular(L.T, U @ (Bt / (s + mu)[:, None]), upper=True)
        if h == 0:
            ref = torch.linalg.solve(gl + mu * m, C[h])
            rel = float((dd - ref).norm() / ref.norm().clamp_min(1e-30))
        out.append(dd); mus.append(mu); energies.append(energy(mu))
    return torch.stack(out), mus, energies, rel


# ---------------- layer-by-layer calibration --------------------------------------------------------------------------
hs = [T.model.embed_tokens(torch.tensor([p["qids"]], device=EDEV)) for p in PAIRS]   # question hidden states, layer 0 input
report = []
for t in range(LT):
    dv = TDEV[t]
    torch.cuda.set_device(dv)   # cuSOLVER/cuBLAS handles follow the current device; the 14B target spans two GPUs
    G = torch.zeros(H, K * D + 1, K * D + 1, device=dv, dtype=torch.float64)
    C = torch.zeros(H, K * D + 1, D, device=dv, dtype=torch.float64)
    M = torch.zeros_like(G)
    e_init = torch.zeros(H, device=dv, dtype=torch.float64); ns = 0; r0 = 0.0
    for di, d in enumerate(DOCS):
        n0 = len(d["hist"])
        kmap, vmap, xv = map_layer(t, d, dv)
        kn, vn = d["nat"][t][0].to(dv).float(), d["nat"][t][1].to(dv).float()
        Xb = torch.cat([xv, torch.ones_like(xv[..., :1])], -1)
        Mdoc = (Xb.double().transpose(1, 2) @ Xb.double()) / n0
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
            g, c = algebra.gram(algebra.pool(Pm[..., :n0], Xb).double(), R)
            G += g; C += c; M += Mdoc
            s = p["samp"].to(dv)
            e_init += (vmap[:, s] - vn[:, s]).double().square().sum((1, 2)); ns += len(s)
            r0 += R.square().sum().item()
            del Pm, Pn, R, qq, kq, vq, hidden
        del kmap, vmap, xv, kn, vn, Xb, Mdoc
    G /= len(PAIRS); C /= len(PAIRS); M /= len(PAIRS)
    if ST is not None:   # paper: M and E0 from the initialization states
        M = ST["M"][t].to(dv, torch.float64); e_init = ST["E0"][t].to(dv, torch.float64); ns = 1
    M += 1e-4 * M.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None] * torch.eye(M.shape[-1], device=dv, dtype=M.dtype)
    E0 = e_init / ns
    dTheta, mus, energies, rel = constrained_solve(G, C, M, (FRAC * E0).tolist(), dv)
    dTheta = dTheta.float().cpu()
    layers[t]["v_weight"] = layers[t]["v_weight"] + dTheta[:, :-1]
    layers[t]["v_bias"] = layers[t]["v_bias"] + dTheta[:, -1]
    for di, d in enumerate(DOCS):   # advance the question states through layer t with the CALIBRATED map
        kmap, vmap, _ = map_layer(t, d, dv)
        for pi, p in enumerate(PAIRS):
            if p["doc"] == di:
                hs[pi] = advance(t, hs[pi], kmap, vmap, len(d["hist"]))
        del kmap, vmap
    torch.cuda.empty_cache()
    report.append(dict(layer=t, readout_residual=r0 / len(PAIRS), E0=E0.tolist(), correction_energy=energies, mu=mus,
                       constrained_heads=sum(m > 0 for m in mus), solver_rel_diff_head0=rel))
    log("layer", t, "readout residual", round(r0 / len(PAIRS), 3), "E0", round(float(E0.mean()), 4),
        "correction energy / E0", round(sum(energies) / max(float(E0.sum()), 1e-30), 5),
        "constrained heads", sum(m > 0 for m in mus), "solver check", f"{rel:.1e}")

os.makedirs(OUT, exist_ok=True)
for t, L in enumerate(layers):
    save_file({k: v.contiguous() for k, v in L.items()}, f"{OUT}/layer_{t:03d}.safetensors")
json.dump(dict(support=support, K=K, note="initial fit + energy-constrained readout V calibration (App. B), layer by layer",
               frac=FRAC, n_pairs=len(PAIRS), init=INIT, minutes=(time.time() - t_start) / 60, calib=report),
          open(f"{OUT}/fit_report.json", "w"), indent=1)
log("FITCB6DONE", OUT)
