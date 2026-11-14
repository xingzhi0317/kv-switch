# Readout-aware reduced-rank ridge + fair dense baseline.
# Full-rank ridge solves each target column independently, so for any target metric T (block-diagonal per
# (layer, head)) the weighted ridge solution is W T: only the rank-r projection depends on the metric.
#   K metric: per-frequency query energy (collect_qstats.py); V metric: sum_j W_o,j^T W_o,j over the GQA group.
# Also: "dense12_refit" = kvbridge's own source-layer choice, refit on our train split only (fair held-out baseline).
# env: THREADS OUT ALPHA
import os
os.environ.setdefault("OUT", os.path.expanduser("~/kvcodesign/results/screen2.json"))
from common import *

A = float(E("ALPHA", "0.01"))
QS = torch.load(f"{HOME}/kvcodesign/data/qstats_4b.pt")
WRAW = f"{HOME}/kvcodesign/data/wraw_a{A:g}.pt"
mdk, mdv = G.diagonal()[SK].mean().item(), G.diagonal()[SV].mean().item()
if os.path.exists(WRAW):
    Wk, Wv = torch.load(WRAW)
else:
    Wk = chol_solve(G[SK, SK], C[SK, TK], A * mdk); Wv = chol_solve(G[SV, SV], C[SV, TV], A * mdv)
    torch.save((Wk, Wv), WRAW)
log("raw ridge maps ready")
ID = lambda x: x
# metrics as (apply T, apply T^-1) on [*, QK] column space
wk = QS["wk"].reshape(-1).clamp_min(1e-8)                     # [QK]
sqk = wk.sqrt()
Mv = QS["Mv"].double()                                        # [LT, H, D, D]
ev, V = torch.linalg.eigh(Mv)
Mh = (V * ev.clamp_min(1e-10).sqrt()[..., None, :]) @ V.transpose(-1, -2)
Mhi = (V * ev.clamp_min(1e-10).rsqrt()[..., None, :]) @ V.transpose(-1, -2)
Mh, Mhi = Mh.float(), Mhi.float()
blk = lambda M: (lambda Y: torch.einsum("nlhd,lhde->nlhe", Y.view(Y.shape[0], LT, H, D), M).reshape(Y.shape[0], QK))


def metric(kind, mode):
    """returns (T, Tinv) acting on row-vectors [n, QK] in RAW target units"""
    base = w[TK] if kind == "k" else w[TV]                     # per-layer variance equalization (as in screen.py)
    if mode == "var":
        return (lambda Y: Y * base), (lambda Y: Y / base)
    if kind == "k":
        # per-layer normalize the weighted metric so each layer's total weighted variance matches 'var'
        tw = sqk.view(LT, -1); vv = var_col[TK].view(LT, -1)
        sc = ((vv * base.view(LT, -1) ** 2).sum(1) / (vv * tw ** 2).sum(1)).sqrt()
        f = (tw * sc[:, None]).reshape(-1)
        return (lambda Y: Y * f), (lambda Y: Y / f)
    # V: block 128x128 per (layer, head), plus per-layer scalar normalization (approx: trace-based)
    vv = var_col[TV].view(LT, H, D)
    num = (vv * base.view(LT, H, D) ** 2).sum((1, 2))
    den = torch.einsum("lhd,lhde->l", vv, Mh.square())
    sc = (num / den).sqrt()
    T_, Ti = Mh * sc[:, None, None, None], Mhi / sc[:, None, None, None]
    return blk(T_), blk(Ti)


def rrr(Wx, Gb, T, Ti, r, k=2304):
    WT = T(Wx)                                                   # [P, QK]: W @ T (T acts on columns)
    U, _ = top_eig(WT, Gb, k)
    return U


# ---- fair dense baseline: kvbridge's 12 source layers, refit on train split, same alpha grid
meta = json.load(open(f"{HOME}/kvbase/mappers/kvbridge_17to4/mapper.json"))
if "dense12_refit" not in results.get("structures", {}):
    Wd = {}
    for kind, S_, T_ in (("k", SK, TK), ("v", SV, TV)):
        md = mdk if kind == "k" else mdv
        Wd[kind] = []
        for t in range(LT):
            idx = torch.cat([torch.arange(s * H * D, (s + 1) * H * D) for s in meta["source_layers"][t]]) + S_.start
            cols = slice(T_.start + t * H * D, T_.start + (t + 1) * H * D)
            Wd[kind].append((idx - S_.start, chol_solve(G[idx][:, idx], C[idx, cols], A * md)))
        log("dense12 refit", kind)
    dk = lambda xk, xv: torch.cat([xk[:, i] @ W_ for i, W_ in Wd["k"]], 1) * w[TK]
    dv = lambda xk, xv: torch.cat([xv[:, i] @ W_ for i, W_ in Wd["v"]], 1) * w[TV]
    summarize("dense12_refit", dk, dv, LT * 12 * H * D * H * D * 2, dict(alpha=A))
    del Wd

for mode in ("var", "readout"):
    Tk, Tki = metric("k", mode); Tv, Tvi = metric("v", mode)
    Uk = rrr(Wk, G[SK, SK], Tk, Tki, None); Uv = rrr(Wv, G[SV, SV], Tv, Tvi, None)
    for r in (256, 512, 768, 1024, 1536, 2048):
        Pk = Tk(Wk) @ Uk[:, :r]; Dk = Tki(Uk[:, :r].T.contiguous())      # y_raw = x P D
        Pv = Tv(Wv) @ Uv[:, :r]; Dv = Tvi(Uv[:, :r].T.contiguous())
        torch.save(dict(Pk=Pk, Dk=Dk, Pv=Pv, Dv=Dv, mx=mx, my=my), f"{HOME}/kvcodesign/mappers/rrr_{mode}_r{r}.pt")
        summarize(f"rrr_{mode}_r{r}", lambda xk, xv: (xk @ Pk) @ Dk * w[TK], lambda xk, xv: (xv @ Pv) @ Dv * w[TV],
                  2 * (PK * r + r * QK), dict(alpha=A, rank=r, metric=mode))
        if mode == "readout" and r in (768, 1024):
            def fp8(P, Dm, mxx):
                b = mxx @ P
                return lambda xc: q8(q8(xc + mxx, 1) @ q8(P, 0) - b, 1) @ q8(Dm, 0)
            summarize(f"rrr_{mode}_r{r}_fp8", lambda xk, xv, g=fp8(Pk, Dk, mx[SK]): g(xk) * w[TK],
                      lambda xk, xv, g=fp8(Pv, Dv, mx[SV]): g(xv) * w[TV], 2 * (PK * r + r * QK), dict(alpha=A, rank=r, metric=mode, fp8=True))
    # full-rank reference under the same alpha (metric-independent)
summarize("dense28_raw", lambda xk, xv: (xk @ Wk) * w[TK], lambda xk, xv: (xv @ Wv) * w[TV], 2 * PK * QK, dict(alpha=A))
log("SCREEN2DONE")
