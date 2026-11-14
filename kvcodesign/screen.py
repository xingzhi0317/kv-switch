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


# ------------------------------------------------------------------ baselines with existing weights
if "headlocal_cb_init" not in results.get("structures", {}):
    support, layers, _ = torch.load(f"{HOME}/kvbase/mappers/cb_init_17to4.pt")

    def hl(kind):
        def f(xk, xv):
            x = (xk if kind == "k" else xv) + mx[SK if kind == "k" else SV]   # un-center
            x = x.view(-1, LS, H, D)
            out = []
            for t in range(LT):
                L = layers[t]; sel = L["selected"].tolist()
                feat = torch.cat([x[:, s] for s in sel], -1).transpose(0, 1)   # [H, n, K*D]
                y = torch.einsum("hnf,hfd->nhd", feat, L[f"{kind}_weight"].float()) + L[f"{kind}_bias"].float()
                out.append(y.reshape(-1, H * D))
            sl = TK if kind == "k" else TV
            return (torch.cat(out, 1) - my[sl]) * w[sl]
        return f
    nk = sum(layers[t]["k_weight"].float().norm().item() for t in range(LT)) / LT
    summarize("headlocal_cb_init", hl("k"), hl("v"), LT * H * 8 * D * D * 2, dict(knorm=nk))
    del layers

if "dense12_kvbridge" not in results.get("structures", {}):
    from safetensors.torch import load_file
    kb = load_file(f"{HOME}/kvbase/mappers/kvbridge_17to4/mapper.safetensors")
    meta = json.load(open(f"{HOME}/kvbase/mappers/kvbridge_17to4/mapper.json"))
    keys = sorted(kb.keys())
    log("kvbridge keys sample", keys[:8])

    def entry(t, name):
        for pat in (f"layer_{t:02d}.{name}", f"layers.{t}.{name}", f"layer_{t}.{name}", f"{t}.{name}", f"layers.{t:03d}.{name}"):
            if pat in kb:
                return kb[pat].float()
        raise KeyError((t, name, keys[:10]))

    def kbf(kind):
        def f(xk, xv):
            x = (xk if kind == "k" else xv) + mx[SK if kind == "k" else SV]
            x = x.view(-1, LS, H * D)
            out = []
            for t in range(LT):
                X = torch.cat([x[:, s] for s in meta["source_layers"][t]], -1)
                out.append((X - entry(t, f"mu_x{kind}")) @ entry(t, f"W_{kind}") + entry(t, f"mu_y{kind}"))
            sl = TK if kind == "k" else TV
            return (torch.cat(out, 1) - my[sl]) * w[sl]
        return f
    nk = sum(entry(t, "W_k").norm().item() for t in range(LT)) / LT
    summarize("dense12_kvbridge", kbf("k"), kbf("v"), LT * 12 * H * D * H * D * 2, dict(knorm=nk))
    del kb

# ------------------------------------------------------------------ dense ridge over all source layers (K->K, V->V)
mean_diag_k = G.diagonal()[SK].mean().item(); mean_diag_v = G.diagonal()[SV].mean().item()
ALPHAS = [float(a) for a in E("ALPHAS", "1e-5,1e-4,1e-3,1e-2,1e-1").split(",")]
W = {}
for a in ALPHAS:
    Wk = chol_solve(G[SK, SK], Cs[SK, TK], a * mean_diag_k)
    Wv = chol_solve(G[SV, SV], Cs[SV, TV], a * mean_diag_v)
    W[a] = (Wk, Wv)
    summarize(f"dense28_sep_a{a:g}", lambda xk, xv: xk @ Wk, lambda xk, xv: xv @ Wv, 2 * PK * QK, dict(alpha=a, knorm=eff_norm_k(Wk)))
    if a in (1e-3,):
        fq = lambda Wx, mxx: (lambda xc: q8(xc + mxx, 1) @ q8(Wx, 0) - mxx @ Wx)
        summarize(f"dense28_sep_a{a:g}_fp8", lambda xk, xv, g=fq(Wk, mx[SK]): g(xk), lambda xk, xv, g=fq(Wv, mx[SV]): g(xv), 2 * PK * QK, dict(alpha=a, fp8=True))
if "mean_only" not in results.get("structures", {}):
    summarize("mean_only", lambda xk, xv: torch.zeros(xk.shape[0], QK), lambda xk, xv: torch.zeros(xk.shape[0], QK), 0)
st_ = results["structures"]
a_best = max(ALPHAS, key=lambda a: st_[f"dense28_sep_a{a:g}"]["r2_k"] + st_[f"dense28_sep_a{a:g}"]["r2_v"])
a_reg = max(ALPHAS)  # strongest ridge = "conditioning-regularized" variant
results["alpha_best"] = a_best; save()
log("best alpha", a_best)
for a in list(W):
    if a not in (a_best, a_reg):
        del W[a]

# ------------------------------------------------------------------ reduced-rank ridge: shared latent
RANKS = [256, 512, 1024, 2048]
KE = 2304
for a in sorted({a_best, a_reg}):
    Wk, Wv = W[a]
    tag = "" if a == a_best else "_reg"
    # (1) K and V separate latents, shared across all target layers: rank r each
    Uk, evk = top_eig(Wk, G[SK, SK], KE); Uv, evv = top_eig(Wv, G[SV, SV], KE)
    for r in RANKS:
        Pk, Pv = Wk @ Uk[:, :r], Wv @ Uv[:, :r]
        torch.save(dict(Pk=Pk, Uk=Uk[:, :r].T.contiguous(), Pv=Pv, Uv=Uv[:, :r].T.contiguous(), mx=mx, my=my, w=w), f"{HOME}/kvcodesign/mappers/rrr_sep_r{r}{tag}.pt")
        summarize(f"rrr_sep_r{r}{tag}", lambda xk, xv: (xk @ Pk) @ Uk[:, :r].T, lambda xk, xv: (xv @ Pv) @ Uv[:, :r].T,
                  2 * (PK * r + r * QK), dict(alpha=a, rank=r, knorm=eff_norm_k(Pk @ Uk[:, :r].T), Pk_norm2=torch.linalg.matrix_norm(Pk, 2).item()))
        if r in (512, 1024):
            def fp8(P, U, mxx):
                bias = mxx @ P
                return lambda xc: q8(q8(xc + mxx, 1) @ q8(P, 0) - bias, 1) @ q8(U, 0)
            summarize(f"rrr_sep_r{r}{tag}_fp8", lambda xk, xv, g=fp8(Pk, Uk[:, :r].T.contiguous(), mx[SK]): g(xk),
                      lambda xk, xv, g=fp8(Pv, Uv[:, :r].T.contiguous(), mx[SV]): g(xv), 2 * (PK * r + r * QK), dict(alpha=a, rank=r, fp8=True))
    results.setdefault("spectra", {})[f"sep{tag}"] = dict(k=evk[:KE].tolist()[::8], v=evv[:KE].tolist()[::8]); save()
    # (2) grouped: 4 groups of 9 consecutive target layers, separate K/V latent per group
    for rg in (128, 256, 512):
        P, U = {}, {}
        for kind, Wx, Gb in (("k", Wk, G[SK, SK]), ("v", Wv, G[SV, SV])):
            P[kind], U[kind] = [], []
            for g in range(4):
                cols = slice(g * 9 * H * D, (g + 1) * 9 * H * D)
                Ug, _ = top_eig(Wx[:, cols], Gb, rg + 128, iters=4, seed=g)
                P[kind].append(Wx[:, cols] @ Ug[:, :rg]); U[kind].append(Ug[:, :rg].T)
        f = lambda kind: (lambda xk, xv: torch.cat([((xk if kind == "k" else xv) @ P[kind][g]) @ U[kind][g] for g in range(4)], 1))
        summarize(f"rrr_sep_group4_r{rg}{tag}", f("k"), f("v"), 2 * (4 * PK * rg + QK * rg), dict(alpha=a, rank_per_group=rg))
    del Uk, Uv

# (3) joint K+V latent (needs the full 57344-dim solve): only at the best alpha
if E("JOINT", "1") == "1":
    lamv = torch.cat([torch.full((PK,), a_best * mean_diag_k), torch.full((PK,), a_best * mean_diag_v)])
    A = G.double(); A.diagonal().add_(lamv.double())
    L = torch.linalg.cholesky(A); del A
    Wj = torch.cholesky_solve(Cs.double(), L).float(); del L
    log("joint solve done")
    summarize("dense28_joint", lambda xk, xv: torch.cat([xk, xv], 1) @ Wj[:, TK], lambda xk, xv: torch.cat([xk, xv], 1) @ Wj[:, TV],
              (2 * PK) * (2 * QK), dict(alpha=a_best))
    Uj, evj = top_eig(Wj, G, KE)
    for r in RANKS:
        Pj = Wj @ Uj[:, :r]
        Z = lambda xk, xv: torch.cat([xk, xv], 1) @ Pj
        summarize(f"rrr_joint_r{r}", lambda xk, xv: Z(xk, xv) @ Uj[TK, :r].T, lambda xk, xv: Z(xk, xv) @ Uj[TV, :r].T,
                  2 * PK * r + r * 2 * QK, dict(alpha=a_best, rank=r))
        torch.save(dict(Pj=Pj, Uj=Uj[:, :r].T.contiguous(), mx=mx, my=my, w=w), f"{HOME}/kvcodesign/mappers/rrr_joint_r{r}.pt")
    results.setdefault("spectra", {})["joint"] = evj.tolist()[::8]; save()
    del Wj, Uj

# ------------------------------------------------------------------ RoPE-equivariant K: complex linear per frequency
# rotate_half convention: pair (i, i+64) is the complex number a + j b, rotated by e^{j pos theta_i}. A linear map commutes
# with every such rotation iff it only mixes equal frequencies and is complex-linear => per frequency f a complex matrix
# A_f [224 src units, 288 tgt units]. Its normal equations are sub-blocks of the real G and C (no extra data pass).
Gk, Ck = G[SK, SK], Cs[SK, TK]
src_u = torch.arange(LS * H) * D; tgt_u = torch.arange(LT * H) * D
for a in sorted({a_best, a_reg}):
    tag = "" if a == a_best else "_reg"
    Af, Mf = [], []
    for f in range(D // 2):
        ia, ib = src_u + f, src_u + f + D // 2
        oc, od = tgt_u + f, tgt_u + f + D // 2
        ZZ = torch.complex(Gk[ia][:, ia] + Gk[ib][:, ib], Gk[ia][:, ib] - Gk[ib][:, ia]).to(torch.complex128)
        ZY = torch.complex(Ck[ia][:, oc] + Ck[ib][:, od], Ck[ia][:, od] - Ck[ib][:, oc]).to(torch.complex128)
        R = ZZ.clone(); R.diagonal().add_(2 * a * mean_diag_k)
        A_ = torch.linalg.solve(R, ZY)
        Af.append(A_); Mf.append(A_.conj().T @ ZZ @ A_)

    def eqv(rank):
        As = []
        for A_, M in zip(Af, Mf):
            if rank is None:
                As.append(A_)
            else:
                ev, V = torch.linalg.eigh(M)
                V = V[:, -rank:]
                As.append(A_ @ V @ V.conj().T)

        def f(xk, xv):
            out = torch.empty(xk.shape[0], QK)
            for fq, A_ in enumerate(As):
                z = torch.complex(xk[:, src_u + fq], xk[:, src_u + fq + D // 2]).to(torch.complex64)
                y = z @ A_.to(torch.complex64)
                out[:, tgt_u + fq] = y.real; out[:, tgt_u + fq + D // 2] = y.imag
            return out
        return f
    Wv = W[a][1]
    for rank in (None, 64, 32, 16):
        macs_k = 4 * (D // 2) * (LS * H * LT * H if rank is None else (LS * H + LT * H) * rank)
        for rv in (512, 1024):
            Uv, _ = top_eig(Wv, G[SV, SV], rv + 256, iters=4)
            Pv = Wv @ Uv[:, :rv]
            nm = f"equivK_{'full' if rank is None else f'r{rank}'}+rrrV_r{rv}{tag}"
            summarize(nm, eqv(rank), lambda xk, xv: (xv @ Pv) @ Uv[:, :rv].T, macs_k + (PK + QK) * rv,
                      dict(alpha=a, rank_per_freq=rank, rank_v=rv, macs_k=macs_k))

# ------------------------------------------------------------------ attention readout error on held-out docs
log("readout eval")
docs = torch.load(VALD) if os.path.exists(VALD) else None
results["readout_ready"] = docs is not None; save()
log("SCREENDONE")
