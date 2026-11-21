# CacheBridge-style value calibration applied to OUR low-rank map (Qwen3-1.7B -> 4B).
# Low-rank map (screen2.py): z_v = (x_v - mx_v) P_v  (shared latent over all source layers and heads, rank RV),
#                            V_t = z_v D_v[:, cols(t)] + my_v[cols(t)]  for every target layer t.
# Calibration (paper App. B objective, as fit_cb6/7.py): layer by layer, each target KV head's V decoder columns are corrected
# from the latent (RV + intercept features, shared by all heads) to match the target's attention readouts at the question
# tokens of 24 QuALITY documents x 4 questions; energy budget FRAC * E0 (FRAC=1e12 = no budget).  The correction is folded
# into D_v / my_v, so the fused kernel and its speed are unchanged.  K maps are not touched.
# Settings match the plotted all-head 12-layer variant (fit_cb5.py): PROMPT old, TRUNC 6000, M/E0 on QuALITY history tokens.
# Checks: the map reimplemented here == eval_quality.LowRank arithmetic on one document; layer-by-layer == full model.
# Output: ~/kvcodesign/mappers/lrc_k{RK}_v{RV}_{TAG}.pt (keys of eval_quality LowRank(path=...)): eval method lrc_k{RK}_v{RV}_{TAG}[_bf16]
# env: RK RV FRAC TAG
import os, sys, json, time, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

E = os.environ.get
HOME = os.path.expanduser("~")
from kvbridge.rope import strip_rope, apply_rope, build_cos_sin
from cachebridge.offline.calibration import algebra

RK, RV, FRAC, TAG = int(E("RK", 2048)), int(E("RV", 2048)), float(E("FRAC", 0.01)), E("TAG", "f1")
NQ, TRUNC = 24, int(E("TRUNC", 6000))
MODELS, LRDIR = f"{HOME}/kvswitch/models", f"{HOME}/kvcodesign/mappers"
OUT = f"{LRDIR}/lrc_k{RK}_v{RV}_{TAG}.pt"
dev, THETA, NSAMP = "cuda:0", 1e6, 4096
t_start = time.time()
log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{(time.time() - t_start) / 60:.1f} min]", *a, flush=True)
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False

tok = AutoTokenizer.from_pretrained(f"{MODELS}/Qwen3-1.7B")
S = AutoModelForCausalLM.from_pretrained(f"{MODELS}/Qwen3-1.7B", torch_dtype=torch.bfloat16).to(dev).eval()
T = AutoModelForCausalLM.from_pretrained(f"{MODELS}/Qwen3-4B", torch_dtype=torch.bfloat16).to(dev).eval()
LS, LT = S.config.num_hidden_layers, T.config.num_hidden_layers
H, D = T.config.num_key_value_heads, 128
PK, QK = LS * H * D, LT * H * D
a, b = torch.load(f"{LRDIR}/rrr_var_r{RK}.pt"), torch.load(f"{LRDIR}/rrr_var_r{RV}.pt")
Pk, Dk, Pv, Dv = (t.to(dev, torch.float32) for t in (a["Pk"], a["Dk"], b["Pv"], b["Dv"]))
mx, my = a["mx"].to(dev, torch.float32), a["my"].to(dev, torch.float32).clone()
mxk, mxv = mx[:PK], mx[PK:]
myk, myv = my[:QK], my[QK:]          # myv is a view: corrections to it land in `my`
log("low-rank map K rank", RK, "V rank", RV, "FRAC", FRAC)

from datasets import load_dataset
arts = {}
for r in load_dataset("emozilla/quality", split="validation"):
    arts.setdefault(r["article"], []).append(r)
docs = [v for v in arts.values() if len(v) >= 4]
random.Random(0).shuffle(docs)
rng = torch.Generator().manual_seed(1)
DOCS, PAIRS = [], []
for di, d in enumerate(docs[:NQ]):   # = fit_cb4/5/6/7.py PROMPT old
    hist = tok(f"<|im_start|>user\nRead the following article.\n\n{d[0]['article']}", add_special_tokens=False).input_ids
    hist = hist[:TRUNC] if TRUNC else hist
    DOCS.append(dict(hist=hist))
    for r in d[:4]:
        opts = "\n".join(f"{c}. {o}" for c, o in zip("ABCD", r["options"]))
        q = tok(f"\n\nQuestion: {r['question']}\n{opts}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:", add_special_tokens=False).input_ids
        PAIRS.append(dict(doc=di, qids=q, samp=torch.randperm(len(hist), generator=rng)[:NSAMP // (4 * NQ) + 1]))

for di, d in enumerate(DOCS):   # latents once per document; target re-prefill KV once per document
    hist = d["hist"]; n0 = len(hist)
    sc = S.model(torch.tensor([hist], device=dev), use_cache=True).past_key_values
    pos = torch.arange(n0, device=dev)
    xk = torch.cat([strip_rope(sc.layers[l].keys[0].float(), pos, THETA).transpose(0, 1).reshape(n0, H * D) for l in range(LS)], 1)
    xv = torch.cat([sc.layers[l].values[0].float().transpose(0, 1).reshape(n0, H * D) for l in range(LS)], 1)
    d["zk"], d["zv"] = (xk - mxk) @ Pk, (xv - mxv) @ Pv                           # [n0, RK], [n0, RV]
    if di == 0:   # check: eval_quality.LowRank arithmetic (rotated source keys in, rotated target keys out)
        sk = torch.stack([sc.layers[l].keys[0].transpose(0, 1) for l in range(LS)]).float()      # [LS, n, H, D] rotated
        inv = 1.0 / (THETA ** (torch.arange(0, D, 2, device=dev, dtype=torch.float32) / D))
        ang = pos.float()[:, None] * inv[None]
        c_, s_ = torch.cat([ang.cos(), ang.cos()], -1)[None, :, None], torch.cat([ang.sin(), ang.sin()], -1)[None, :, None]
        rot = lambda x: torch.cat([-x[..., D // 2:], x[..., :D // 2]], -1)
        kk = (sk * c_ - rot(sk) * s_).permute(1, 0, 2, 3).reshape(n0, -1) - mxk
        yk = ((kk @ Pk) @ Dk + myk).view(n0, LT, H, D).permute(1, 0, 2, 3)
        yk = yk * c_ + rot(yk) * s_
        mine = apply_rope((d["zk"] @ Dk[:, :H * D] + myk[:H * D]).view(n0, H, D).transpose(0, 1), pos, THETA)
        log("check map vs eval_quality.LowRank, target layer 0 K: max abs diff", float((mine.transpose(0, 1) - yk[0]).abs().max()),
            "scale", float(yk[0].abs().max()))
        del sk, kk, yk, mine
    del xk, xv, sc
    nat = T.model(torch.tensor([hist], device=dev), use_cache=True).past_key_values
    d["nat"] = [(nat.layers[l].keys[0].cpu().pin_memory(), nat.layers[l].values[0].cpu().pin_memory()) for l in range(LT)]
    del nat
log("prepared", len(DOCS), "documents,", len(PAIRS), "questions")


def map_layer(t, d):   # -> rotated K [H,n,D], V [H,n,D], V features z_v [n, RV]
    cols = slice(t * H * D, (t + 1) * H * D); n = d["zk"].shape[0]
    k = (d["zk"] @ Dk[:, cols] + myk[cols]).view(n, H, D).transpose(0, 1)
    v = (d["zv"] @ Dv[:, cols] + myv[cols]).view(n, H, D).transpose(0, 1)
    return apply_rope(k, torch.arange(n, device=dev), THETA), v, d["zv"]


def q_states(t, hidden, n0, nq):
    at = T.model.layers[t].self_attn
    hsh = (1, nq, -1, D)
    qq = at.q_norm(at.q_proj(hidden).view(hsh)).transpose(1, 2)
    kk = at.k_norm(at.k_proj(hidden).view(hsh)).transpose(1, 2)
    vv = at.v_proj(hidden).view(hsh).transpose(1, 2)
    cos, sin = build_cos_sin(torch.arange(n0, n0 + nq, device=dev), D, THETA, dev, torch.float32)
    rot = lambda x: (x.float() * cos + torch.cat([-x[..., D // 2:], x[..., :D // 2]], -1).float() * sin)
    return rot(qq), rot(kk), vv.float()


class OneLayerCache:
    def __init__(self, k, v):
        self.k, self.v = k, v

    def update(self, k, v, layer_idx, cache_kwargs=None):
        return torch.cat([self.k, k], 2), torch.cat([self.v, v], 2)


def advance(t, h, kmap, vmap, n0):
    nq = h.shape[1]
    pid = torch.arange(n0, n0 + nq, device=dev)[None]
    pe = T.model.rotary_emb(h, pid)
    mask = torch.zeros(1, 1, nq, n0 + nq, device=dev, dtype=h.dtype)
    mask[..., n0:] = torch.triu(torch.full((nq, nq), torch.finfo(h.dtype).min, device=dev, dtype=h.dtype), 1)
    out = T.model.layers[t](h, attention_mask=mask, position_ids=pid, past_key_value=OneLayerCache(kmap[None].to(torch.bfloat16), vmap[None].to(torch.bfloat16)),
                            use_cache=True, cache_position=pid[0], position_embeddings=pe)
    return out[0] if isinstance(out, tuple) else out


for p in PAIRS[:2]:   # check: layer-by-layer path == full-model path
    d = DOCS[p["doc"]]; n0, nq = len(d["hist"]), len(p["qids"])
    cache = DynamicCache(); kv = []
    for l in range(LT):
        k, v, _ = map_layer(l, d); kv.append((k, v))
        cache.update(k[None].to(torch.bfloat16), v[None].to(torch.bfloat16), l)
    full = T(torch.tensor([p["qids"]], device=dev), past_key_values=cache, position_ids=torch.arange(n0, n0 + nq, device=dev)[None]).logits[0, -1].float()
    h = T.model.embed_tokens(torch.tensor([p["qids"]], device=dev))
    for l in range(LT):
        h = advance(l, h, *kv[l], n0)
    seq = T.lm_head(T.model.norm(h))[0, -1].float()
    full, seq = full.log_softmax(-1), seq.log_softmax(-1)
    log("check layer-by-layer vs full model: KL", float((full.exp() * (full - seq)).sum()), "same argmax", bool(full.argmax() == seq.argmax()))
    del cache, kv


def constrained_solve(G, C, M, budget):   # as fit_cb7.py (fp32 Cholesky + fp64 refinement, fp64 fallback)
    p = G.shape[-1]
    m = (0.5 * (M + M.T)).double()
    out, mus, energies = [], [], []
    for h in range(G.shape[0]):
        g = G[h].double(); g = 0.5 * (g + g.T)
        lam = max(1e-8, 0.01 * float(torch.trace(g)) / p)
        c = C[h].double()

        def system(mu):
            A = g + mu * m
            A.diagonal().add_(lam)
            return A

        def solve(mu, refine=True):   # the mu search uses fp32 solves only; the final mu is refined in fp64
            A = system(mu)
            L, info = torch.linalg.cholesky_ex(A.float())
            if int(info) != 0:
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
        out.append(d); mus.append(mu); energies.append(energy(d))
    return torch.stack(out), mus, energies, rel


hs = [T.model.embed_tokens(torch.tensor([p["qids"]], device=dev)) for p in PAIRS]
report = []
P = RV + 1
for t in range(LT):
    G = torch.zeros(H, P, P, device=dev, dtype=torch.float32)
    C = torch.zeros(H, P, D, device=dev, dtype=torch.float64)
    M = torch.zeros(P, P, device=dev, dtype=torch.float32)
    e_init = torch.zeros(H, device=dev, dtype=torch.float64); ns = 0; r0 = 0.0
    for di, d in enumerate(DOCS):
        n0 = len(d["hist"])
        kmap, vmap, z = map_layer(t, d)
        kn, vn = d["nat"][t][0].to(dev).float(), d["nat"][t][1].to(dev).float()
        Xb = torch.cat([z, torch.ones_like(z[:, :1])], -1)
        for pi, p in enumerate(PAIRS):
            if p["doc"] != di:
                continue
            nq = len(p["qids"])
            hidden = T.model.layers[t].input_layernorm(hs[pi])
            qq, kq, vq = q_states(t, hidden, n0, nq)
            positions = list(range(nq))
            Pm = algebra.probabilities(qq, torch.cat([kmap, kq[0]], 1)[None], n0, positions)
            Pn = algebra.probabilities(qq, torch.cat([kn, kq[0]], 1)[None], n0, positions)
            R = (algebra.pool(Pn, torch.cat([vn, vq[0]], 1)) - algebra.pool(Pm, torch.cat([vmap, vq[0]], 1))).double()
            A = torch.einsum("hqn,np->hqp", Pm[..., :n0], Xb)
            G.baddbmm_(A.transpose(1, 2), A, alpha=1.0 / A.shape[1])
            C += (A.double().transpose(1, 2) @ R) / A.shape[1]
            M.addmm_(Xb.T, Xb, alpha=1.0 / n0)
            s = p["samp"].to(dev)
            e_init += (vmap[:, s] - vn[:, s]).double().square().sum((1, 2)); ns += len(s)
            r0 += R.square().sum().item()
            del Pm, Pn, R, A, qq, kq, vq, hidden
        del kmap, vmap, kn, vn, Xb
    G /= len(PAIRS); C /= len(PAIRS); M /= len(PAIRS)
    M.diagonal().add_(1e-4 * M.diagonal().mean())
    E0 = e_init / ns
    dTheta, mus, energies, rel = constrained_solve(G, C, M, (FRAC * E0).tolist())          # [H, P, D]
    cols = slice(t * H * D, (t + 1) * H * D)
    Dv[:, cols] += dTheta[:, :-1].permute(1, 0, 2).reshape(RV, H * D).float()            # columns head-major
    myv[cols] += dTheta[:, -1].reshape(H * D).float()
    for di, d in enumerate(DOCS):
        kmap, vmap, _ = map_layer(t, d)
        for pi, p in enumerate(PAIRS):
            if p["doc"] == di:
                hs[pi] = advance(t, hs[pi], kmap, vmap, len(d["hist"]))
        del kmap, vmap
    torch.cuda.empty_cache()
    report.append(dict(layer=t, readout_residual=r0 / len(PAIRS), E0=E0.tolist(), correction_energy=energies, mu=mus, solver_rel_diff_head0=rel))
    log("layer", t, "readout residual", round(r0 / len(PAIRS), 3), "E0", round(float(E0.mean()), 4),
        "correction energy / E0", round(sum(energies) / max(float(E0.sum()), 1e-30), 4), "solver check", f"{rel:.1e}")
torch.save(dict(Pk=Pk.cpu(), Dk=Dk.cpu(), Pv=Pv.cpu(), Dv=Dv.cpu(), mx=mx.cpu(), my=my.cpu()), OUT)
json.dump(dict(rk=RK, rv=RV, frac=FRAC, trunc=TRUNC, minutes=(time.time() - t_start) / 60, calib=report),
          open(OUT.replace(".pt", "_report.json"), "w"), indent=1)
log("LRCDONE", OUT)
