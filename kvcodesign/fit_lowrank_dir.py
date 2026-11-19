# Low-rank shared-latent translator (ours) for either direction -- the fit of screen2.py (metric "var", alpha 0.01),
# written without the 17to4-only constants of common.py.  CPU only.
#   per kind (K, V separately): x = all source layers' content-space K (resp. V) of a token, y = all target layers'.
#   ridge W = (Xc^T Xc + a*mean(diag) I)^-1 Xc^T Yc on the train split (first 90% of calibration tokens);
#   reduced rank: top-r eigenvectors U of the (per target layer variance-equalised) in-sample prediction covariance;
#   P = W T U_r, D = U_r^T T^-1  (T = per (target layer) 1/sqrt(mean variance) column weights)  ->  y = (x - mx) P D + my
# Data: ~/kvbase/mappers/calib_17to4.pt holds both models' KV on the same tokens (sk, sv = 1.7B; tk, tv = 4B); for
#   4to17 the roles are swapped.
# Output: ~/kvcodesign/mappers/{DIR}/rrr_var_r{r}.pt (same format as screen2.py: Pk Dk Pv Dv mx my), r = 768 1536 2048
# env: DIR (4to17) THREADS (64) ALPHA (0.01)
import os, time, torch
E = os.environ.get
torch.set_num_threads(int(E("THREADS", 64)))
DIR, A = E("DIR", "4to17"), float(E("ALPHA", "0.01"))
HOME = os.path.expanduser("~")
OUT = f"{HOME}/kvcodesign/mappers/{DIR}"
os.makedirs(OUT, exist_ok=True)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
RANKS, KTOP = (768, 1536, 2048), 2304


def top_eig(W, Gb, k, iters=5, seed=0):   # as common.py
    g = torch.Generator().manual_seed(seed)
    Q = torch.linalg.qr(torch.randn(W.shape[1], k, generator=g))[0]
    op = lambda M: W.T @ (Gb @ (W @ M))
    for _ in range(iters):
        Q = torch.linalg.qr(op(Q))[0]
    B = Q.T @ op(Q)
    ev, U = torch.linalg.eigh(B.double())
    return Q @ U.flip(1).float()


log("loading calib")
sk, sv, tk, tv = torch.load(f"{HOME}/kvbase/mappers/calib_17to4.pt", mmap=True)
src = {"k": sk, "v": sv} if DIR == "17to4" else {"k": tk, "v": tv}
tgt = {"k": tk, "v": tv} if DIR == "17to4" else {"k": sk, "v": sv}
N = src["k"][0].shape[0]; ntr = int(N * 0.9); CH = 8192
LTn = len(tgt["k"])
log("N", N, "train", ntr, "source layers", len(src["k"]), "target layers", LTn)
res = {}
for kind in ("k", "v"):
    X = torch.stack(list(src[kind]), 1).reshape(N, -1)          # fp16 [N, LS*H*D]
    Y = torch.stack(list(tgt[kind]), 1).reshape(N, -1)          # fp16 [N, LT*H*D]
    mx = sum(X[i:min(ntr, i + CH)].double().sum(0) for i in range(0, ntr, CH)) / ntr
    my = sum(Y[i:min(ntr, i + CH)].double().sum(0) for i in range(0, ntr, CH)) / ntr
    mxf, myf = mx.float(), my.float()
    G = torch.zeros(X.shape[1], X.shape[1]); C = torch.zeros(X.shape[1], Y.shape[1])
    yy = torch.zeros(Y.shape[1], dtype=torch.float64)
    for i in range(0, ntr, CH):
        j = min(ntr, i + CH)
        xc = X[i:j].float() - mxf; yc = Y[i:j].float() - myf
        G.addmm_(xc.T, xc); C.addmm_(xc.T, yc); yy += yc.double().square().sum(0)
        if (i // CH) % 10 == 0:
            log(kind, "stats rows", j)
    Xval, Yval = X[ntr:].float(), Y[ntr:].float()
    del X, Y
    w = (yy / ntr).float().view(LTn, -1).mean(1).rsqrt().repeat_interleave(Y_cols := C.shape[1] // LTn)   # T weights
    Gd = G.double(); Gd.diagonal().add_(A * G.diagonal().mean().item())
    W = torch.cholesky_solve(C.double(), torch.linalg.cholesky(Gd)).float(); del Gd
    log(kind, "ridge done")
    U = top_eig(W * w, G, KTOP)
    del G, C
    for r in RANKS:
        P = (W * w) @ U[:, :r]; Dm = U[:, :r].T.contiguous() / w
        pred = (Xval - mxf) @ P @ Dm + myf
        e = (pred - Yval).square().view(-1, LTn, Y_cols).sum((0, 2)); s = (Yval - Yval.mean(0)).square().view(-1, LTn, Y_cols).sum((0, 2))
        r2 = (1 - e / s).mean().item()
        res[(kind, r)] = (P, Dm, mxf, myf)
        log(kind, "rank", r, "held-out R2 (mean over target layers)", round(r2, 4))
    pred = (Xval - mxf) @ W + myf
    e = (pred - Yval).square().view(-1, LTn, Y_cols).sum((0, 2)); s = (Yval - Yval.mean(0)).square().view(-1, LTn, Y_cols).sum((0, 2))
    log(kind, "full-rank ridge held-out R2", round((1 - e / s).mean().item(), 4))
    del W, U, Xval, Yval
for r in RANKS:
    Pk, Dk, mxk, myk = res[("k", r)]; Pv, Dv, mxv, myv = res[("v", r)]
    torch.save(dict(Pk=Pk, Dk=Dk, Pv=Pv, Dv=Dv, mx=torch.cat([mxk, mxv]), my=torch.cat([myk, myv]),
                    meta=dict(dir=DIR, alpha=A, rank=r, metric="var")), f"{OUT}/rrr_var_r{r}.pt")
log("LOWRANKDONE", OUT)
