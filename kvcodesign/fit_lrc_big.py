# CacheBridge-style value calibration of OUR low-rank map for a large pair (Qwen3-8B -> 14B); fit_lrc.py (1.7B -> 4B,
# verified) with the multi-GPU handling of fit_cb7.py and the paper-aligned data of the 8B -> 14B CacheBridge runs:
#   map         : z_v = (x_v - mx_v) P_v (rank RV, all source layers and heads), V_t = z_v D_v[:, cols(t)] + my_v[cols(t)]
#   calibration : layer by layer, per target KV head, the V decoder columns of layer t are corrected from [z_v, 1] to match
#                 the target's attention readouts at the question tokens of 24 QuALITY documents x 4 questions
#                 (CacheBridge Table 6 layout, complete articles); energy budget FRAC * E0, with M and E0 on the 4096
#                 sampled FineWeb fitting states (cb_fast.py budget_samples.pt).  M depends on z only -> one M for all layers.
#   speed       : latents computed once per document on GPU 0 and copied once to each target GPU; per-layer decoder slices;
#                 the mu search uses fp32 Cholesky solves, only the final mu is refined in fp64 (2049-dim systems).
# Checks: map == eval_quality.LowRank arithmetic (layer 0 K); layer-by-layer == full model on two questions.
# Output: ~/kvcodesign/mappers/8to14/lrc_k{RK}_v{RV}_{TAG}.pt  (eval_quality method lrc_k{RK}_v{RV}_{TAG}, DIR=8to14)
# env: RK RV FRAC TAG
import os, sys, json, time, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

E = os.environ.get
HOME = os.path.expanduser("~")
from kvbridge.rope import strip_rope, apply_rope, build_cos_sin
from cachebridge.offline.calibration import algebra
sys.path.insert(0, f"{HOME}/kvcodesign")
from tasks import quality_cb_text, chat_split

RK, RV, FRAC, TAG = int(E("RK", 2048)), int(E("RV", 2048)), float(E("FRAC", 0.01)), E("TAG", "f1")
NQ, NSAMP = 24, 4096
MODELS, LRDIR = f"{HOME}/kvswitch/models", f"{HOME}/kvcodesign/mappers/8to14"
BS = f"{HOME}/kvbase/stats/8to14_full500/budget_samples.pt"
OUT = f"{LRDIR}/lrc_k{RK}_v{RV}_{TAG}.pt"
d0, THETA = "cuda:0", 1e6
t_start = time.time()
log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{(time.time() - t_start) / 60:.1f} min]", *a, flush=True)
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False

tok = AutoTokenizer.from_pretrained(f"{MODELS}/Qwen3-8B-ms")
enc = lambda x: tok(x, add_special_tokens=False).input_ids
S = AutoModelForCausalLM.from_pretrained(f"{MODELS}/Qwen3-8B-ms", torch_dtype=torch.bfloat16).to(d0).eval()
T = AutoModelForCausalLM.from_pretrained(f"{MODELS}/Qwen3-14B-ms", torch_dtype=torch.bfloat16, device_map="auto",
                                         max_memory={0: "0GiB", 1: "20GiB", 2: "20GiB"}).eval()
LS, LT = S.config.num_hidden_layers, T.config.num_hidden_layers
H, D = T.config.num_key_value_heads, 128
PK, QK = LS * H * D, LT * H * D
TDEV = [next(T.model.layers[l].parameters()).device for l in range(LT)]
DEVS = sorted(set(TDEV), key=str)
EDEV, RDEV = T.model.embed_tokens.weight.device, T.model.norm.weight.device
a, b = torch.load(f"{LRDIR}/rrr_var_r{RK}.pt"), torch.load(f"{LRDIR}/rrr_var_r{RV}.pt")
Pk, Pv = a["Pk"].to(d0, torch.float32), b["Pv"].to(d0, torch.float32)
Dk, Dv = a["Dk"].float(), b["Dv"].float().clone()            # masters on the CPU; per-layer slices go to the layer's GPU
mx, my = a["mx"].float(), a["my"].float().clone()
mxk, mxv = mx[:PK].to(d0), mx[PK:].to(d0)
myk, myv = my[:QK], my[QK:]                                    # views: corrections to myv land in `my`
log("low-rank map K rank", RK, "V rank", RV, "FRAC", FRAC, "target GPUs", [str(x) for x in DEVS])

from datasets import load_dataset
arts = {}
for r in load_dataset("emozilla/quality", split="validation"):
    arts.setdefault(r["article"], []).append(r)
docs = [v for v in arts.values() if len(v) >= 4]
random.Random(0).shuffle(docs)
DOCS, PAIRS = [], []
for di, d in enumerate(docs[:NQ]):   # = fit_cb6/7.py PROMPT=cb TRUNC=0
    DOCS.append(dict(hist=chat_split(enc, *quality_cb_text(d[0]["article"], d[0]))[0]))
    for r in d[:4]:
        PAIRS.append(dict(doc=di, qids=chat_split(enc, *quality_cb_text(d[0]["article"], r))[1]))

for di, d in enumerate(DOCS):
    hist = d["hist"]; n0 = len(hist)
    sc = S.model(torch.tensor([hist], device=d0), use_cache=True).past_key_values
    pos = torch.arange(n0, device=d0)
    xk = torch.cat([strip_rope(sc.layers[l].keys[0].float(), pos, THETA).transpose(0, 1).reshape(n0, H * D) for l in range(LS)], 1)
    xv = torch.cat([sc.layers[l].values[0].float().transpose(0, 1).reshape(n0, H * D) for l in range(LS)], 1)
    zk, zv = (xk - mxk) @ Pk, (xv - mxv) @ Pv
    if di == 0:   # check: eval_quality.LowRank arithmetic for target layer 0 K (rotated in, rotated out)
        sk = torch.stack([sc.layers[l].keys[0].transpose(0, 1) for l in range(LS)]).float()
        inv = 1.0 / (THETA ** (torch.arange(0, D, 2, device=d0, dtype=torch.float32) / D))
        ang = pos.float()[:, None] * inv[None]
        c_, s_ = torch.cat([ang.cos(), ang.cos()], -1)[None, :, None], torch.cat([ang.sin(), ang.sin()], -1)[None, :, None]
        rot = lambda x: torch.cat([-x[..., D // 2:], x[..., :D // 2]], -1)
        kk = (sk * c_ - rot(sk) * s_).permute(1, 0, 2, 3).reshape(n0, -1) - mxk
        y0 = ((kk @ Pk) @ Dk[:, :H * D].to(d0) + myk[:H * D].to(d0)).view(n0, H, D)
        y0 = y0 * c_[0] + rot(y0) * s_[0]
        mine = apply_rope((zk @ Dk[:, :H * D].to(d0) + myk[:H * D].to(d0)).view(n0, H, D).transpose(0, 1), pos, THETA).transpose(0, 1)
        log("check map vs LowRank arithmetic, layer 0 K: max abs diff", float((mine - y0).abs().max()), "scale", float(y0.abs().max()))
        del sk, kk, y0, mine
    d["z"] = {dv: (zk.to(dv), zv.to(dv)) for dv in DEVS}
    del xk, xv, zk, zv, sc
    nat = T.model(torch.tensor([hist], device=EDEV), use_cache=True).past_key_values
    d["nat"] = [(nat.layers[l].keys[0].cpu().pin_memory(), nat.layers[l].values[0].cpu().pin_memory()) for l in range(LT)]
    del nat
log("prepared", len(DOCS), "documents,", len(PAIRS), "questions")

# energy-budget statistics on the FineWeb fitting states: M = second moment of [z_v, 1] (same for every layer), E0 per layer
BSD = torch.load(BS)
zs = (BSD["src_v"].permute(1, 0, 2, 3).reshape(BSD["src_v"].shape[1], -1).to(d0).float() - mxv) @ Pv    # [NS, RV]
Zs = torch.cat([zs, torch.ones_like(zs[:, :1])], -1).double()
M0 = (Zs.T @ Zs / Zs.shape[0]); M0.diagonal().add_(1e-4 * M0.diagonal().mean())
M0 = M0.cpu(); zs = zs.cpu()
log("budget statistics: M", tuple(M0.shape), "from", zs.shape[0], "FineWeb states")
SL = {}   # per-layer decoder slices on the layer's GPU


def slices(t):
    if t not in SL:
        cols, dv = slice(t * H * D, (t + 1) * H * D), TDEV[t]
        SL[t] = (Dk[:, cols].to(dv), myk[cols].to(dv), Dv[:, cols].to(dv), myv[cols].to(dv))
    return SL[t]


def map_layer(t, d):
    dv = TDEV[t]; dk, mk, dvv, mv = slices(t); zk, zv = d["z"][dv]; n = zk.shape[0]
    k = (zk @ dk + mk).view(n, H, D).transpose(0, 1)
    v = (zv @ dvv + mv).view(n, H, D).transpose(0, 1)
    return apply_rope(k, torch.arange(n, device=dv), THETA), v, zv


def q_states(t, hidden, n0, nq, dv):
    at = T.model.layers[t].self_attn
    hsh = (1, nq, -1, D)
    qq = at.q_norm(at.q_proj(hidden).view(hsh)).transpose(1, 2)
    kk = at.k_norm(at.k_proj(hidden).view(hsh)).transpose(1, 2)
    vv = at.v_proj(hidden).view(hsh).transpose(1, 2)
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


for p in PAIRS[:2]:   # check: layer-by-layer path == full-model path
    d = DOCS[p["doc"]]; n0, nq = len(d["hist"]), len(p["qids"])
    cache = DynamicCache(); kv = []
    for l in range(LT):
        torch.cuda.set_device(TDEV[l])
        k, v, _ = map_layer(l, d); kv.append((k, v))
        cache.update(k[None].to(torch.bfloat16), v[None].to(torch.bfloat16), l)
    full = T(torch.tensor([p["qids"]], device=EDEV), past_key_values=cache, position_ids=torch.arange(n0, n0 + nq, device=EDEV)[None]).logits[0, -1].float()
    h = T.model.embed_tokens(torch.tensor([p["qids"]], device=EDEV))
    for l in range(LT):
        h = advance(l, h, *kv[l], n0)
    seq = T.lm_head(T.model.norm(h.to(RDEV)))[0, -1].float()
    full, seq = full.log_softmax(-1), seq.to(full.device).log_softmax(-1)
    log("check layer-by-layer vs full model: KL", float((full.exp() * (full - seq)).sum()), "same argmax", bool(full.argmax() == seq.argmax()))
    del cache, kv


def constrained_solve(G, C, m, budget):   # as fit_lrc.py (fp32 search, fp64-refined final solve, fp64 fallback)
    p = G.shape[-1]
    out, mus, energies = [], [], []
    for h in range(G.shape[0]):
        g = G[h].double(); g = 0.5 * (g + g.T)
        lam = max(1e-8, 0.01 * float(torch.trace(g)) / p)
        c = C[h].double()

        def system(mu):
            A = g + mu * m
            A.diagonal().add_(lam)
            return A

        def solve(mu, refine=True):
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


hs = [T.model.embed_tokens(torch.tensor([p["qids"]], device=EDEV)) for p in PAIRS]
report = []
P = RV + 1
for t in range(LT):
    dv = TDEV[t]; torch.cuda.set_device(dv)
    G = torch.zeros(H, P, P, device=dv, dtype=torch.float32)
    C = torch.zeros(H, P, D, device=dv, dtype=torch.float64)
    r0 = 0.0
    for di, d in enumerate(DOCS):
        n0 = len(d["hist"])
        kmap, vmap, z = map_layer(t, d)
        kn, vn = d["nat"][t][0].to(dv).float(), d["nat"][t][1].to(dv).float()
        Xb = torch.cat([z, torch.ones_like(z[:, :1])], -1)
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
            r0 += R.square().sum().item()
            del Pm, Pn, R, A, qq, kq, vq, hidden
        del kmap, vmap, kn, vn, Xb
    G /= len(PAIRS); C /= len(PAIRS)
    _, _, dvv, mv = slices(t)                                                     # E0: the current (= initial) map of layer t
    vm = (zs.to(dv) @ dvv + mv).view(-1, H, D)
    E0 = (vm - BSD["tgt_v"][t].to(dv).float()).double().square().sum(-1).mean(0)
    del vm
    dTheta, mus, energies, rel = constrained_solve(G, C, M0.to(dv), (FRAC * E0).tolist())   # [H, P, D]
    cols = slice(t * H * D, (t + 1) * H * D)
    Dv[:, cols] += dTheta[:, :-1].permute(1, 0, 2).reshape(RV, H * D).float().cpu()
    myv[cols] += dTheta[:, -1].reshape(H * D).float().cpu()
    SL.pop(t, None)                                                                # re-slice the calibrated columns
    for di, d in enumerate(DOCS):
        kmap, vmap, _ = map_layer(t, d)
        for pi, p in enumerate(PAIRS):
            if p["doc"] == di:
                hs[pi] = advance(t, hs[pi], kmap, vmap, len(d["hist"]))
        del kmap, vmap
    SL.pop(t, None); del G, C; torch.cuda.empty_cache()
    report.append(dict(layer=t, readout_residual=r0 / len(PAIRS), E0=E0.tolist(), correction_energy=energies, mu=mus, solver_rel_diff_head0=rel))
    log("layer", t, "readout residual", round(r0 / len(PAIRS), 3), "E0", round(float(E0.mean()), 4),
        "correction energy / E0", round(sum(energies) / max(float(E0.sum()), 1e-30), 4), "solver check", f"{rel:.1e}")
torch.save(dict(Pk=a["Pk"], Dk=Dk, Pv=b["Pv"], Dv=Dv, mx=mx, my=my, meta=dict(dir="8to14", rk=RK, rv=RV, frac=FRAC)), OUT)
json.dump(dict(rk=RK, rv=RV, frac=FRAC, minutes=(time.time() - t_start) / 60, calib=report), open(OUT.replace(".pt", "_report.json"), "w"), indent=1)
log("LRCBIGDONE", OUT)
