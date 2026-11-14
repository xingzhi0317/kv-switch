# Check eval_quality.LowRank (rotated-key input path) against the offline content-space mapping on a held-out doc.
import os, torch
torch.set_num_threads(32)
src = open(os.path.expanduser("~/kvcodesign/eval_quality.py")).read()
code = src[src.index("class LowRank"):src.index("\nfor m in METHODS:\n    if m.startswith")]
LS, LT, H, D, dev = 28, 36, 8, 128, "cpu"
exec(code)
d = torch.load(os.path.expanduser("~/kvcodesign/data/val_docs.pt"))[0]
m = LowRank(768, 1536)
n = d["sk"].shape[1]
cos, sin = m.cs(n)
c, s = cos[None, :, None], sin[None, :, None]
k = d["sk"].float()
k_rot = k * c + m.rot(k) * s                         # what the source cache holds
out = m(k_rot.to(torch.bfloat16), d["sv"].to(torch.bfloat16)).float()
# offline reference (content space, fp32, no bf16 round trip)
a = torch.load(os.path.expanduser("~/kvcodesign/mappers/rrr_var_r768.pt")); b = torch.load(os.path.expanduser("~/kvcodesign/mappers/rrr_var_r1536.pt"))
PK, QK = LS * H * D, LT * H * D
xk = k.permute(1, 0, 2, 3).reshape(n, -1) - a["mx"][:PK]
xv = d["sv"].float().permute(1, 0, 2, 3).reshape(n, -1) - a["mx"][PK:]
yk = ((xk @ a["Pk"]) @ a["Dk"] + a["my"][:QK]).view(n, LT, H, D).permute(1, 0, 2, 3)
yv = ((xv @ b["Pv"]) @ b["Dv"] + b["my"][QK:]).view(n, LT, H, D).permute(1, 0, 2, 3)
yk_rot = yk * c + m.rot(yk) * s
rel = lambda p, q: ((p - q).norm() / q.norm()).item()
print("K rel err", rel(out[0], yk_rot), "V rel err", rel(out[1], yv))
nat = d["tk"].float(); nat_rot = nat * c + m.rot(nat) * s
print("sanity: rel err vs native K", rel(out[0], nat_rot), "V", rel(out[1], d["tv"].float()))
