# Case 2 closed-form init (KV-Lingo-style init / NVIDIA-style ridge) on character-aligned token pairs.
# env: A B TAG NSEQ(64) NVAL(8) SEQ(2048) KMAX(3)
import os, json, random, torch
from ks import *
from xalign import conv_aligned

E = os.environ.get
A_, B_, TAG = E("A"), E("B"), E("TAG")
NSEQ, NVAL, SEQ, KMAX = int(E("NSEQ", 64)), int(E("NVAL", 8)), int(E("SEQ", 2048)), int(E("KMAX", 3))
A, B = Model(A_), Model(B_)
dev = "cuda"


def seqs(n, seed, split):
    rng = random.Random(seed)
    uc = json.load(open(f"{ROOT}/data/ultrachat_train.json" if split == "train" else f"{ROOT}/data/ultrachat.json"))
    out = []
    for i in range(n):
        msgs = []
        while True:
            conv = uc[rng.randrange(len(uc))]
            msgs += [(m["role"], m["content"]) for m in conv]
            ia, ib, mp = conv_aligned(A, B, msgs)
            if len(ib) >= SEQ and len(ia) >= 16:
                break
        keep = mp[:SEQ]
        na = int(keep[keep >= 0].max()) + 1
        out.append((torch.tensor(ia[:max(na, 1)]), torch.tensor(ib[:SEQ]), keep))
    return out


def window(c, k, L):
    s = min(max(c - k // 2, 0), L - k)
    return list(range(s, s + k))


centers = [round(l * (A.L - 1) / (B.L - 1)) for l in range(B.L)]
src = [window(c, KMAX, A.L) for c in centers]
BLK = A.Hk * A.D
F_ = KMAX * BLK + 1
Fo = B.Hk * B.D


def collect(M, ids):
    c = M.new_cache(len(ids))
    col = {"q": False}
    M.forward(ids[None].to(dev), c, collect=col, want="hidden")
    return col


def acc(data):
    S = dict(AK=[torch.zeros(F_, F_, device=dev) for _ in range(B.L)], BK=[torch.zeros(F_, Fo, device=dev) for _ in range(B.L)],
             AV=[torch.zeros(F_, F_, device=dev) for _ in range(B.L)], BV=[torch.zeros(F_, Fo, device=dev) for _ in range(B.L)],
             YK=torch.zeros(B.L, device=dev), YV=torch.zeros(B.L, device=dev))
    for ia, ib, mp in data:
        ca, cb = collect(A, ia), collect(B, ib)
        ok = (mp >= 0).nonzero().flatten().to(dev)
        pa = mp.to(dev)[ok]
        for l in range(B.L):
            for kind, ka, kb in (("K", ca["k"], cb["k"]), ("V", ca["v"], cb["v"])):
                X = torch.cat([ka[s][:, pa].transpose(0, 1).reshape(len(pa), -1) for s in src[l]], -1).float()
                X = torch.cat([X, torch.ones_like(X[:, :1])], -1)
                Y = kb[l][:, ok].transpose(0, 1).reshape(len(ok), -1).float()
                S["A" + kind][l] += X.T @ X; S["B" + kind][l] += X.T @ Y; S["Y" + kind][l] += (Y * Y).sum()
    return S


def sub_idx(l, k):
    w = window(centers[l], k, A.L)
    return torch.cat([torch.arange(src[l].index(s) * BLK, (src[l].index(s) + 1) * BLK) for s in w] + [torch.tensor([F_ - 1])]).to(dev)


def ridge(Aa, Bb, lam):
    R = torch.eye(Aa.shape[0], device=dev) * lam * Aa.diagonal().mean(); R[-1, -1] = 0
    return torch.linalg.solve(Aa + R, Bb)


def rel_err(W, Aa, Bb, yy):
    return (((W * (Aa @ W)).sum() - 2 * (W * Bb).sum() + yy) / yy).item()


tr, va = seqs(NSEQ, 0, "train"), seqs(NVAL, 1, "val")
log("aligned; frac B tokens with a source:", float(sum((m >= 0).float().mean() for _, _, m in tr) / len(tr)))
St, Sv = acc(tr), acc(va)
out, rep = {}, {}
for k in sorted({1, KMAX}):
    out[k], rep[k] = dict(src=[window(centers[l], k, A.L) for l in range(B.L)]), {}
    for kind in ("K", "V"):
        best = None
        for lam in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            errs = [rel_err(ridge(St["A" + kind][l][sub_idx(l, k)][:, sub_idx(l, k)], St["B" + kind][l][sub_idx(l, k)], lam),
                            Sv["A" + kind][l][sub_idx(l, k)][:, sub_idx(l, k)], Sv["B" + kind][l][sub_idx(l, k)], Sv["Y" + kind][l]) for l in range(B.L)]
            m = sum(errs) / len(errs)
            if best is None or m < best[0]:
                best = (m, lam)
        rep[k][kind] = best
        out[k]["W" + kind] = torch.stack([ridge(St["A" + kind][l][sub_idx(l, k)][:, sub_idx(l, k)], St["B" + kind][l][sub_idx(l, k)], best[1]).cpu() for l in range(B.L)])
    log(f"k={k}", rep[k])
torch.save(out, f"{ROOT}/maps/{TAG}.pt")
json.dump(rep, open(f"{ROOT}/results/fit_{TAG}.json", "w"), indent=1)
log("FITDONE")
