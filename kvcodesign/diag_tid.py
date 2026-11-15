# Why does the identity-augmented map have a worse attention readout?  Check the first token (attention sink).
import os
os.environ["OUT"] = "/tmp/wxz_diag.json"
from common import *
HD = H * D
a, b = torch.load(f"{HOME}/kvcodesign/mappers/rrr_var_r2048.pt"), torch.load(f"{HOME}/kvcodesign/mappers/tid_r2048.pt")
me = b["me"]
def preds(d):
    n = d["sk"].shape[1]
    xk = d["sk"].float().permute(1, 0, 2, 3).reshape(n, PK) - mx[SK]; xv = d["sv"].float().permute(1, 0, 2, 3).reshape(n, PK) - mx[SV]
    tk_, tv_ = d["tk"].float().permute(1, 0, 2, 3).reshape(n, QK), d["tv"].float().permute(1, 0, 2, 3).reshape(n, QK)
    e = torch.cat([tk_[:, :HD], tv_[:, :HD]], 1) - me
    old = (torch.cat([tk_[:, :HD], ((xk @ a["Pk"]) @ a["Dk"] + my[TK])[:, HD:]], 1), torch.cat([tv_[:, :HD], ((xv @ a["Pv"]) @ a["Dv"] + my[TV])[:, HD:]], 1))
    new = (torch.cat([tk_[:, :HD], (torch.cat([xk, e * b["s_k"]], 1) @ b["Pk"]) @ b["Dk"] + my[TK][HD:]], 1),
           torch.cat([tv_[:, :HD], (torch.cat([xv, e * b["s_v"]], 1) @ b["Pv"]) @ b["Dv"] + my[TV][HD:]], 1))
    return n, tk_, tv_, old, new
def ro(fn):
    num = torch.zeros(LT); den = torch.zeros(LT)
    for d, (q, on) in zip(DOCS, NATIVE_O):
        n, tk_, tv_, old, new = preds(d)
        yk, yv = fn(tk_, tv_, old, new)
        o = attn_out(q, rope(yk.view(n, LT, H, D).transpose(0, 1).transpose(1, 2), torch.arange(n)).transpose(1, 2), yv.view(n, LT, H, D).transpose(0, 1), d["qpos"])
        num += (o - on).square().sum((1, 2, 3)); den += on.square().sum((1, 2, 3))
    return float((num / den)[1:].mean())
def mix(first, rest):   # take row 0 from one prediction and the other rows from another
    def f(tk_, tv_, old, new):
        src = dict(old=old, new=new, true=(tk_, tv_))
        yk, yv = src[rest][0].clone(), src[rest][1].clone()
        yk[0], yv[0] = src[first][0][0], src[first][1][0]
        return yk, yv
    return f
for first, rest in (("old", "old"), ("new", "new"), ("old", "new"), ("true", "new"), ("true", "old"), ("new", "old")):
    print(f"first token from {first:4s}, other tokens from {rest:4s}: readout", round(ro(mix(first, rest)), 4), flush=True)
e0 = []
for d in DOCS:
    n, tk_, tv_, old, new = preds(d)
    for nm, p in (("old", old), ("new", new)):
        e0.append((nm, float((p[0][0] - tk_[0]).norm() / tk_[0].norm()), float((p[1][0] - tv_[0]).norm() / tv_[0].norm()),
                   float((p[0][1:] - tk_[1:]).norm() / tk_[1:].norm()), float((p[1][1:] - tv_[1:]).norm() / tv_[1:].norm())))
for nm in ("old", "new"):
    r = torch.tensor([x[1:] for x in e0 if x[0] == nm]).mean(0)
    print(nm, "first-token rel err K", round(float(r[0]), 3), "V", round(float(r[1]), 3), "| other tokens K", round(float(r[2]), 3), "V", round(float(r[3]), 3))
