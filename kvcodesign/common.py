# Shared helpers (auto-split from screen.py): stats, R^2, attention-readout metric, FP8 sim, ridge/RRR solvers.
# Offline screening of mapping structures (Qwen3-1.7B -> Qwen3-4B), CPU only, no downstream tasks.
# Data: ~/kvbase/mappers/calib_17to4.pt (official kvbridge collector, content-space KV, docs in order).
#   train = first 90% of observations, val = last 10% (contiguous => document-level split, except one boundary doc).
# Metrics: held-out R^2 of K and V (mean over target layers) and attention readout error on held-out FineWeb docs
#   (collect_val.py) using the target model's real queries.
# Structures: head-local (CacheBridge init, existing), kvbridge (released dense, 12 src layers), dense ridge over all
#   source layers, reduced-rank ridge (shared latent: K/V separate or joint, all target layers or grouped),
#   RoPE-equivariant K (complex per-frequency ridge, optionally low-rank), each with a stronger-ridge variant + FP8 sim.
# env: THREADS STAGE(all) ALPHAS
import os, json, math, time, torch
E = os.environ.get
torch.set_num_threads(int(E("THREADS", 96)))
HOME = os.path.expanduser("~")
CALIB = f"{HOME}/kvbase/mappers/calib_17to4.pt"
STATS = f"{HOME}/kvcodesign/data/stats_17to4.pt"
VALD = f"{HOME}/kvcodesign/data/val_docs.pt"
OUTJ = E("OUT", f"{HOME}/kvcodesign/results/screen.json")
LS, LT, H, D, HQ = 28, 36, 8, 128, 32
PK, QK = LS * H * D, LT * H * D          # 28672, 36864 per kind
THETA = 1e6
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
results = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}


def save():
    json.dump(results, open(OUTJ, "w"), indent=1)


# ------------------------------------------------------------------ stage A: sufficient statistics
if not os.path.exists(STATS):
    log("loading calib")
    sk, sv, tk, tv = torch.load(CALIB)
    N = sk[0].shape[0]
    X = torch.cat([torch.stack(sk, 1).reshape(N, -1), torch.stack(sv, 1).reshape(N, -1)], 1)  # fp16 [N, 2*PK]
    del sk, sv
    Y = torch.cat([torch.stack(tk, 1).reshape(N, -1), torch.stack(tv, 1).reshape(N, -1)], 1)  # fp16 [N, 2*QK]
    del tk, tv
    ntr = int(N * 0.9)
    log("N", N, "train", ntr, "X", tuple(X.shape), "Y", tuple(Y.shape))
    CH = 8192
    mx = torch.zeros(X.shape[1], dtype=torch.float64); my = torch.zeros(Y.shape[1], dtype=torch.float64)
    for i in range(0, ntr, CH):
        j = min(ntr, i + CH)
        mx += X[i:j].double().sum(0); my += Y[i:j].double().sum(0)
    mx /= ntr; my /= ntr
    mxf, myf = mx.float(), my.float()
    G = torch.zeros(X.shape[1], X.shape[1]); C = torch.zeros(X.shape[1], Y.shape[1]); yy = torch.zeros(Y.shape[1], dtype=torch.float64)
    for i in range(0, ntr, CH):
        j = min(ntr, i + CH)
        xc = X[i:j].float() - mxf; yc = Y[i:j].float() - myf
        G.addmm_(xc.T, xc); C.addmm_(xc.T, yc); yy += yc.double().square().sum(0)
        log("stats rows", j)
    torch.save(dict(G=G, C=C, yy=yy, mx=mx, my=my, ntr=ntr, Xval=X[ntr:].clone(), Yval=Y[ntr:].clone()), STATS)
    del X, Y, G, C
log("loading stats")
st = torch.load(STATS)
G, C, mx, my, ntr = st["G"], st["C"], st["mx"].float(), st["my"].float(), st["ntr"]
Xval, Yval = st["Xval"].float() - mx, st["Yval"].float()
var_col = (st["yy"] / ntr).float()
# per (kind, target layer) block weight: 1/sqrt(mean variance) => Frobenius objective ~ sum of (1 - R^2) over blocks
blk_var = var_col.view(2 * LT, H * D).mean(1)
w = blk_var.rsqrt().repeat_interleave(H * D)                  # [2*QK]
Cs = C * w                                                     # scaled cross-covariance
SK, SV = slice(0, PK), slice(PK, 2 * PK)
TK, TV = slice(0, QK), slice(QK, 2 * QK)
log("stats loaded; G", tuple(G.shape), "C", tuple(C.shape), "val rows", Xval.shape[0])


def r2_blocks(pred, Y):  # pred, Y: [n, 2*QK] unscaled -> per (kind, layer) R^2
    e = (pred - Y).square().view(-1, 2 * LT, H * D).sum((0, 2))
    s = (Y - Y.mean(0)).square().view(-1, 2 * LT, H * D).sum((0, 2))
    r = 1 - e / s
    return r[:LT], r[LT:]


# held-out docs for the readout metric (wait for collect_val.py if it is still running)
while not os.path.exists(VALD):
    log("waiting for", VALD); time.sleep(120)
time.sleep(5)
DOCS = torch.load(VALD)
log("val docs", len(DOCS))


def rope(x, pos):  # x [..., n, D] content space -> rotated (rotate_half convention)
    inv = 1.0 / (THETA ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    ang = pos.float()[:, None] * inv[None]
    cos, sin = torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)
    return x * cos + torch.cat([-x[..., D // 2:], x[..., :D // 2]], -1) * sin


def attn_out(q, k, v, qpos):  # q [LT, nq, HQ, D] rotated; k, v [LT, n, H, D] -> [LT, HQ, nq, D]
    n = k.shape[1]
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3).repeat_interleave(HQ // H, 1)
    v = v.permute(0, 2, 1, 3).repeat_interleave(HQ // H, 1)
    s = q @ k.transpose(-1, -2) / math.sqrt(D)
    s.masked_fill_(torch.arange(n)[None, :] > qpos[:, None], float("-inf"))
    return s.softmax(-1) @ v


NATIVE_O = []
for d in DOCS:
    n = d["sk"].shape[1]
    q = rope(d["tq"].float().transpose(1, 2), d["qpos"]).transpose(1, 2)
    NATIVE_O.append((q, attn_out(q, rope(d["tk"].float().transpose(1, 2), torch.arange(n)).transpose(1, 2), d["tv"].float(), d["qpos"])))


def readout(pk, pv):
    num = torch.zeros(LT); den = torch.zeros(LT)
    for d, (q, on) in zip(DOCS, NATIVE_O):
        n = d["sk"].shape[1]
        xk = d["sk"].float().permute(1, 0, 2, 3).reshape(n, PK) - mx[SK]
        xv = d["sv"].float().permute(1, 0, 2, 3).reshape(n, PK) - mx[SV]
        yk = (pk(xk, xv) / w[TK] + my[TK]).view(n, LT, H, D).transpose(0, 1)
        yv = (pv(xk, xv) / w[TV] + my[TV]).view(n, LT, H, D).transpose(0, 1)
        o = attn_out(q, rope(yk.transpose(1, 2), torch.arange(n)).transpose(1, 2), yv, d["qpos"])
        num += (o - on).square().sum((1, 2, 3)); den += on.square().sum((1, 2, 3))
    return num / den


def summarize(name, pk, pv, macs, extra=None):
    """pk/pv: functions (centered source K [n,PK], centered source V [n,PK]) -> scaled centered predictions [n, QK]."""
    pred = torch.cat([pk(Xval[:, SK], Xval[:, SV]) / w[TK] + my[TK], pv(Xval[:, SK], Xval[:, SV]) / w[TV] + my[TV]], 1)
    rk, rv = r2_blocks(pred, Yval)
    ro = readout(pk, pv)
    d = dict(macs=macs, r2_k=rk.mean().item(), r2_v=rv.mean().item(), readout_err=ro.mean().item(),
             r2_k_layers=rk.tolist(), r2_v_layers=rv.tolist(), readout_layers=ro.tolist())
    d.update(extra or {})
    results.setdefault("structures", {})[name] = d
    save()
    log(f"{name:40s} MACs {macs / 1e6:8.1f}M  R2 K {d['r2_k']:.4f}  V {d['r2_v']:.4f}  readout {d['readout_err']:.4f}",
        {k: (round(v, 3) if isinstance(v, float) else v) for k, v in (extra or {}).items() if not isinstance(v, list)})
    return d


def q8(x, dim):  # simulated FP8 E4M3 with absmax scaling along `dim`
    s = x.abs().amax(dim, keepdim=True).clamp_min(1e-12) / 448.0
    return (x / s).to(torch.float8_e4m3fn).float() * s


def chol_solve(A, B, lam):
    A = A.double().clone(); A.diagonal().add_(lam)
    L = torch.linalg.cholesky(A)
    return torch.cholesky_solve(B.double(), L).float()


def top_eig(W, Gb, k, iters=5, seed=0):
    """top-k eigvecs of W^T Gb W (= covariance of in-sample ridge predictions) by randomized subspace iteration"""
    g = torch.Generator().manual_seed(seed)
    Q = torch.linalg.qr(torch.randn(W.shape[1], k, generator=g))[0]
    op = lambda M: W.T @ (Gb @ (W @ M))
    for _ in range(iters):
        Q = torch.linalg.qr(op(Q))[0]
    B = Q.T @ op(Q)
    ev, U = torch.linalg.eigh(B.double())
    ev, U = ev.flip(0).float(), U.flip(1).float()
    return Q @ U, ev


def eff_norm_k(Wk_scaled):  # mean over target layers of ||effective unscaled K map||_F
    return (Wk_scaled / w[TK]).square().view(PK, LT, H * D).sum((0, 2)).sqrt().mean().item()


