# Token identity as an extra input of the closed-form map (PLAN_v5 stage 3, step 1).  CPU only, our own code.
# Idea (borrowed from arXiv 2611.12062, which repairs "locates but cannot copy" with target token embeddings):
# both models share the tokenizer, so the token ids of the history are known.  The target model's layer-0 K and V
# depend only on the token id:  e = [k_norm(k_proj(ln(embed(id)))), v_proj(ln(embed(id)))]  (2 x 1024 numbers).
#   (a) layer 0 of the target cache is computed exactly instead of being translated;
#   (b) e is appended to the source features, and the same closed-form reduced-rank ridge is refitted for target
#       layers 1..35:   y = ([x - mx, s (e - me)] P) D + my.      No training.
# In the calibration data e is simply the layer-0 block of the target KV, so the only new statistic is
# Cey = E^T Y (cached in data/stats_tid.pt); G_xe is already inside the stored cross-covariance C.
# Offline evaluation on held-out data, every variant with layer 0 exact and scored on layers 1..35:
#   R^2 of K / V, attention-readout error (target's real queries), K / V error of digit tokens vs ordinary words.
# Output: mappers/tid_r{768,1536,2048}.pt, results/screen_tid.json      env: THREADS ALPHA
import os
os.environ.setdefault("OUT", os.path.expanduser("~/kvcodesign/results/screen_tid.json"))
from common import *
from tokerr_types import token_types

A = float(E("ALPHA", "0.01"))
HD = H * D
EC = torch.cat([torch.arange(0, HD), QK + torch.arange(0, HD)])            # layer-0 K and V columns of Y
NE = 2 * HD
ESTATS = f"{HOME}/kvcodesign/data/stats_tid.pt"
if os.path.exists(ESTATS):
    Cey = torch.load(ESTATS)
else:
    log("loading calib for E^T Y")
    _, _, tk, tv = torch.load(CALIB)
    N = tk[0].shape[0]
    Cey = torch.zeros(NE, 2 * QK)
    for i in range(0, ntr, 8192):
        j = min(ntr, i + 8192)
        yc = torch.cat([torch.stack([t[i:j] for t in tk], 1).reshape(j - i, -1), torch.stack([t[i:j] for t in tv], 1).reshape(j - i, -1)], 1).float() - my
        Cey.addmm_(yc[:, EC].T, yc)
        log("E^T Y rows", j)
    del tk, tv
    torch.save(Cey, ESTATS)
Gee, me = Cey[:, EC], my[EC]
TC = {"k": torch.arange(HD, QK), "v": QK + torch.arange(HD, QK)}             # target columns of layers 1..35


def fit(kind):
    S_ = SK if kind == "k" else SV
    Gxx, Gxe = G[S_, S_], C[S_][:, EC]
    md = Gxx.diagonal().mean()
    s = float((md / Gee.diagonal().mean()).sqrt())                           # put e on the scale of the source features
    Ga = torch.cat([torch.cat([Gxx, Gxe * s], 1), torch.cat([Gxe.T * s, Gee * s * s], 1)], 0)
    W = chol_solve(Ga, torch.cat([C[S_][:, TC[kind]], Cey[:, TC[kind]] * s], 0), A * float(md))
    wt = w[TC[kind]]
    U, _ = top_eig(W * wt, Ga, 2304)
    log("fit", kind, "scale", round(s, 4), "W", tuple(W.shape))
    return W, U, s, wt


Wk, Uk, s_k, wtk = fit("k")
Wv, Uv, s_v, wtv = fit("v")
VARIANTS = {}   # name -> (fk, fv, macs); f(x_centred [n, PK], e_centred [n, NE]) -> centred prediction of layers 1..35
for r in (768, 1536, 2048):
    Pk, Dk = (Wk * wtk) @ Uk[:, :r], Uk[:, :r].T.contiguous() / wtk
    Pv, Dv = (Wv * wtv) @ Uv[:, :r], Uv[:, :r].T.contiguous() / wtv
    torch.save(dict(Pk=Pk, Dk=Dk, Pv=Pv, Dv=Dv, mx=mx, my=my, me=me, s_k=s_k, s_v=s_v), f"{HOME}/kvcodesign/mappers/tid_r{r}.pt")
tid = lambda r: torch.load(f"{HOME}/kvcodesign/mappers/tid_r{r}.pt")
old = lambda r: torch.load(f"{HOME}/kvcodesign/mappers/rrr_var_r{r}.pt")
Wk0, Wv0 = torch.load(f"{HOME}/kvcodesign/data/wraw_a0.01.pt")
aug = lambda x, e, s: torch.cat([x, e * s], 1)
VARIANTS["dense (no identity)"] = (lambda x, e: x @ Wk0[:, HD:], lambda x, e: x @ Wv0[:, HD:], 2 * PK * (QK - HD))
VARIANTS["dense + identity"] = (lambda x, e: aug(x, e, s_k) @ Wk, lambda x, e: aug(x, e, s_v) @ Wv, 2 * (PK + NE) * (QK - HD))
for rk, rv in ((768, 1536), (2048, 2048)):
    a, b = old(rk), old(rv)
    VARIANTS[f"low-rank K{rk}+V{rv} (no identity)"] = (lambda x, e, a=a: ((x @ a["Pk"]) @ a["Dk"])[:, HD:], lambda x, e, b=b: ((x @ b["Pv"]) @ b["Dv"])[:, HD:], (PK + QK) * (rk + rv))
    a, b = tid(rk), tid(rv)
    VARIANTS[f"low-rank K{rk}+V{rv} + identity"] = (lambda x, e, a=a: (aug(x, e, s_k) @ a["Pk"]) @ a["Dk"], lambda x, e, b=b: (aug(x, e, s_v) @ b["Pv"]) @ b["Dv"],
                                                    (PK + NE + QK - HD) * (rk + rv))

TY = [token_types(d["ids"]) for d in DOCS]
res = {}
for name, (fk, fv, macs) in VARIANTS.items():
    # held-out calibration rows: R^2 of layers 1..35 (layer 0 exact for every variant)
    ev_ = Yval[:, EC] - me
    pred = Yval.clone()
    pred[:, TC["k"]] = fk(Xval[:, SK], ev_) + my[TC["k"]]
    pred[:, TC["v"]] = fv(Xval[:, SV], ev_) + my[TC["v"]]
    rk_, rv_ = r2_blocks(pred, Yval)
    # held-out documents: readout error and per-token-type error
    num = torch.zeros(LT); den = torch.zeros(LT); te = {t: {x: [0.0, 0.0] for x in "kv"} for t in ("digit", "word", "cont", "cap")}
    for d, (q, on), ty in zip(DOCS, NATIVE_O, TY):
        n = d["sk"].shape[1]
        xk = d["sk"].float().permute(1, 0, 2, 3).reshape(n, PK) - mx[SK]
        xv = d["sv"].float().permute(1, 0, 2, 3).reshape(n, PK) - mx[SV]
        tk_, tv_ = d["tk"].float().permute(1, 0, 2, 3).reshape(n, QK), d["tv"].float().permute(1, 0, 2, 3).reshape(n, QK)
        e = torch.cat([tk_[:, :HD], tv_[:, :HD]], 1) - me
        yk = torch.cat([tk_[:, :HD], fk(xk, e) + my[TC["k"]]], 1); yv = torch.cat([tv_[:, :HD], fv(xv, e) + my[TC["v"]]], 1)
        o = attn_out(q, rope(yk.view(n, LT, H, D).transpose(0, 1).transpose(1, 2), torch.arange(n)).transpose(1, 2), yv.view(n, LT, H, D).transpose(0, 1), d["qpos"])
        num += (o - on).square().sum((1, 2, 3)); den += on.square().sum((1, 2, 3))
        for x, yh, y, off in (("k", yk, tk_, 0), ("v", yv, tv_, QK)):
            err = (yh - y)[:, HD:].square().view(n, LT - 1, HD).sum(2)                           # [n, 35]
            ref = (y - my[off:off + QK])[:, HD:].square().view(n, LT - 1, HD).sum(2).mean(0)     # per-layer normaliser
            rel = (err / ref).mean(1)
            for t in te:
                sel = torch.tensor([a == t for a in ty])
                te[t][x][0] += float(rel[sel].sum()); te[t][x][1] += int(sel.sum())
    ro = (num / den)[1:]
    res[name] = dict(macs=macs, r2_k=float(rk_[1:].mean()), r2_v=float(rv_[1:].mean()), readout=float(ro.mean()), readout_layers=ro.tolist(),
                     r2_v_layers=rv_.tolist(), r2_k_layers=rk_.tolist(),
                     **{f"{x}_err_{t}": te[t][x][0] / te[t][x][1] for t in te for x in "kv"})
    r = res[name]
    log(f"{name:36s} MACs {macs / 1e6:6.0f}M  R2 K {r['r2_k']:.3f} V {r['r2_v']:.3f}  readout {r['readout']:.3f} | V err digit {r['v_err_digit']:.3f} word {r['v_err_word']:.3f}"
        f" | K err digit {r['k_err_digit']:.3f} word {r['k_err_word']:.3f}")
    json.dump(res, open(OUTJ, "w"), indent=1)
log("FITTIDDONE")
