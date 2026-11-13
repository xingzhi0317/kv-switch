# Closed-form translators (Case 1).
#  B2 (NVIDIA-style): multi-source-layer affine ridge on de-RoPE'd K and on V.
#  B1 (CacheBridge-style): same K map; V map calibrated on B's attention outputs: min ||P X W - P V_B||.
# Stats are accumulated once for a KMAX-layer source window; k=1 maps are a sub-block of the same stats.
# env: A B TAG NSEQ(64) NVAL(8) SEQ(2048) KMAX(3) TQ(256) GROUP(7)
import os, json, torch, random
from ks import *

E = os.environ.get
A_, B_, TAG = E("A"), E("B"), E("TAG")
NSEQ, NVAL, SEQ = int(E("NSEQ", 64)), int(E("NVAL", 8)), int(E("SEQ", 2048))
KMAX, TQ, GROUP = int(E("KMAX", 3)), int(E("TQ", 256)), int(E("GROUP", 7))
A, B = Model(A_), Model(B_)
dev = "cuda"


def calib_seqs(M, n, seed, split):
    """Half chat (UltraChat, model's template), half c4 text; packed to SEQ tokens."""
    rng = random.Random(seed)
    uc = json.load(open(f"{ROOT}/data/ultrachat_train.json" if split == "train" else f"{ROOT}/data/ultrachat.json"))
    c4 = torch.load(os.path.expanduser(f"~/kvmap/reuse/c4_{'train' if split == 'train' else 'test'}.pt"))
    out = []
    for i in range(n):
        if i % 2 == 0:
            ids = []
            while len(ids) < SEQ:
                conv = uc[rng.randrange(len(uc))]
                ids += M.conv_ids([(m["role"], m["content"]) for m in conv])[0]
            out.append(torch.tensor(ids[:SEQ]))
        else:
            o = rng.randrange(len(c4) - SEQ)
            out.append(c4[o:o + SEQ].long())
    return out


def window(c, k, L):
    s = min(max(c - k // 2, 0), L - k)
    return list(range(s, s + k))


centers = [round(l * (A.L - 1) / (B.L - 1)) for l in range(B.L)]
src = [window(c, KMAX, A.L) for c in centers]
BLK = A.Hk * A.D
F_ = KMAX * BLK + 1
Fo = B.Hk * B.D
G = B.Hq // B.Hk


def feats(kv, l):  # kv: per-A-layer [Hk,N,D] -> [N,F] fp32 (+bias column)
    X = torch.cat([kv[s].transpose(0, 1).reshape(kv[s].shape[1], -1) for s in src[l]], -1).float()
    return torch.cat([X, torch.ones_like(X[:, :1])], -1)


def collect(M, ids, q=False):
    c = M.new_cache(SEQ)
    col = {"q": q}
    M.forward(ids[None].to(dev), c, collect=col, want="hidden")
    del c
    return col


def accumulate(seqs, lays, plain):
    S = dict(n=0)
    if plain:
        for nm in ("AK", "AV"):
            S[nm] = [torch.zeros(F_, F_, device=dev) for _ in range(B.L)]
        for nm in ("BK", "BV"):
            S[nm] = [torch.zeros(F_, Fo, device=dev) for _ in range(B.L)]
        S["YK"], S["YV"] = torch.zeros(B.L, device=dev), torch.zeros(B.L, device=dev)
    S["AVa"] = {l: torch.zeros(B.Hk, F_, F_, device=dev) for l in lays}
    S["BVa"] = {l: torch.zeros(B.Hk, F_, B.D, device=dev) for l in lays}
    S["YVa"] = {l: torch.zeros(B.Hk, device=dev) for l in lays}
    g = torch.Generator().manual_seed(0)
    for i, ids in enumerate(seqs):
        ca, cb = collect(A, ids), collect(B, ids, q=bool(lays))
        N = ids.shape[0]
        qpos = (torch.randperm(N - 256, generator=g)[:TQ].sort().values + 256).to(dev)
        cos, sin = B.cossin(0, N)
        for l in range(B.L):
            if not plain and l not in lays:
                continue
            Xv = feats(ca["v"], l)
            if plain:
                Xk = feats(ca["k"], l)
                Yk = cb["k"][l].transpose(0, 1).reshape(N, -1).float()
                Yv = cb["v"][l].transpose(0, 1).reshape(N, -1).float()
                S["AK"][l] += Xk.T @ Xk; S["BK"][l] += Xk.T @ Yk; S["YK"][l] += (Yk * Yk).sum()
                S["AV"][l] += Xv.T @ Xv; S["BV"][l] += Xv.T @ Yv; S["YV"][l] += (Yv * Yv).sum()
                del Xk, Yk, Yv
            if l in lays:
                kpB = rope(cb["k"][l].float(), cos, sin)
                qB = cb["qp"][l][:, qpos].float()
                s = torch.einsum("htd,hnd->htn", qB, kpB.repeat_interleave(G, 0)) * B.D ** -0.5
                s = s.masked_fill(torch.arange(N, device=dev)[None, None] > qpos[None, :, None], float("-inf"))
                P = s.softmax(-1)
                for h in range(B.Hq):
                    gh = h // G
                    Xt = P[h] @ Xv
                    Ot = P[h] @ cb["v"][l][gh].float()
                    S["AVa"][l][gh] += Xt.T @ Xt; S["BVa"][l][gh] += Xt.T @ Ot; S["YVa"][l][gh] += (Ot * Ot).sum()
                del P, s
        S["n"] += N
        del ca, cb
        if i % 16 == 0:
            log(f"  seq {i} mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G")
    return S


def sub_idx(l, k):
    w = window(centers[l], k, A.L)
    return torch.cat([torch.arange(src[l].index(s) * BLK, (src[l].index(s) + 1) * BLK) for s in w] + [torch.tensor([F_ - 1])]).to(dev)


def ridge(Aa, Bb, lam):
    R = torch.eye(Aa.shape[0], device=dev) * lam * Aa.diagonal().mean()
    R[-1, -1] = 0
    return torch.linalg.solve(Aa + R, Bb)


def rel_err(W, Aa, Bb, yy):
    return (((W * (Aa @ W)).sum() - 2 * (W * Bb).sum() + yy) / yy).item()


LAMS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
KS = sorted({1, KMAX})
out = {k: dict(WK=[], WV=[], WVa=[None] * B.L) for k in KS}
rep = {k: dict(lamK=None, lamV=None, lamVa=None, errK=[], errV=[], errVa=[None] * B.L) for k in KS}

# ---- pass 1: plain K/V stats (train + val) ----
tr = calib_seqs(A, NSEQ, 0, "train")
va = calib_seqs(A, NVAL, 1, "val")
log("pass plain")
St, Sv = accumulate(tr, [], True), accumulate(va, [], True)
for k in KS:
    for kind in ("K", "V"):
        best = None
        for lam in LAMS:
            errs = []
            for l in range(B.L):
                ix = sub_idx(l, k)
                W = ridge(St["A" + kind][l][ix][:, ix], St["B" + kind][l][ix], lam)
                errs.append(rel_err(W, Sv["A" + kind][l][ix][:, ix], Sv["B" + kind][l][ix], Sv["Y" + kind][l]))
            m = sum(errs) / len(errs)
            log(f"k={k} {kind} lam={lam:g} val_rel_err={m:.4f}")
            if best is None or m < best[0]:
                best = (m, lam, errs)
        rep[k]["lam" + kind], rep[k]["err" + kind] = best[1], best[2]
        for l in range(B.L):
            ix = sub_idx(l, k)
            out[k]["W" + kind].append(ridge(St["A" + kind][l][ix][:, ix], St["B" + kind][l][ix], best[1]).cpu())
del St, Sv
torch.cuda.empty_cache()

# ---- pass 2..: attention-calibrated V, layer groups to bound memory ----
for g0 in range(0, B.L, GROUP):
    lays = list(range(g0, min(B.L, g0 + GROUP)))
    log("pass vattn", lays)
    St, Sv = accumulate(tr, lays, False), accumulate(va, lays, False)
    for k in KS:
        for l in lays:
            ix = sub_idx(l, k)
            best = None
            for lam in LAMS:
                Ws = [ridge(St["AVa"][l][h][ix][:, ix], St["BVa"][l][h][ix], lam) for h in range(B.Hk)]
                e = sum(rel_err(Ws[h], Sv["AVa"][l][h][ix][:, ix], Sv["BVa"][l][h][ix], Sv["YVa"][l][h]) for h in range(B.Hk)) / B.Hk
                if best is None or e < best[0]:
                    best = (e, lam, Ws)
            out[k]["WVa"][l] = torch.cat(best[2], -1).cpu()  # [F_k, Hk*D], head-major columns
            rep[k]["errVa"][l] = (best[0], best[1])
    del St, Sv
    torch.cuda.empty_cache()

for k in KS:
    out[k] = {nm: torch.stack(v) for nm, v in out[k].items()}
    out[k]["src"] = [window(centers[l], k, A.L) for l in range(B.L)]
    log(f"k={k} mean val rel err K {sum(rep[k]['errK']) / B.L:.4f} V {sum(rep[k]['errV']) / B.L:.4f} "
        f"Va(attn-out) {sum(e for e, _ in rep[k]['errVa']) / B.L:.4f}")
os.makedirs(f"{ROOT}/maps", exist_ok=True)
torch.save(out, f"{ROOT}/maps/{TAG}.pt")
json.dump(rep, open(f"{ROOT}/results/fit_{TAG}.json", "w"), indent=1)
log("FITDONE")
