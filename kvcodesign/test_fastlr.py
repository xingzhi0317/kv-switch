# Correctness of fastlr.FastLowRank (bf16 / FP8) vs an fp32 reference, plus the attention-readout error on the
# held-out docs. FP8 static scales come from the calibration validation rows (not from the held-out docs).
import os, json, math, torch
from fastlr import FastLowRank, TorchLowRank, rope_table, LS, LT, H, D
HOME = os.path.expanduser("~")
dev = "cuda"
RK, RV = int(os.environ.get("RK", 768)), int(os.environ.get("RV", 1536))
SC = f"{HOME}/kvcodesign/data/fp8_scales_k{RK}_v{RV}.json"

if not os.path.exists(SC):
    st = torch.load(f"{HOME}/kvcodesign/data/stats_17to4.pt", mmap=True)
    X = st["Xval"].to(dev).float()                                     # raw content-space source rows [n, 2*PK]
    PK = LS * H * D
    a = torch.load(f"{HOME}/kvcodesign/mappers/rrr_var_r{RK}.pt"); b = torch.load(f"{HOME}/kvcodesign/mappers/rrr_var_r{RV}.pt")
    xk, xv = X[:, :PK], X[:, PK:]
    zk = (xk - a["mx"][:PK].to(dev)) @ a["Pk"].to(dev)
    zv = (xv - b["mx"][PK:].to(dev)) @ b["Pv"].to(dev)
    sc = dict(k_layer=xk.view(-1, LS, H * D).abs().amax((0, 2)).tolist(), v_layer=xv.view(-1, LS, H * D).abs().amax((0, 2)).tolist(),
              zk=zk.abs().amax().item(), zv=zv.abs().amax().item(), source="calibration validation rows, amax")
    json.dump(sc, open(SC, "w"), indent=1)
    del X, st
scales = json.load(open(SC))
print("scales: zk %.1f zv %.1f  k_layer max %.1f min %.1f" % (scales["zk"], scales["zv"], max(scales["k_layer"]), min(scales["k_layer"])))

docs = torch.load(f"{HOME}/kvcodesign/data/val_docs.pt")
M = dict(ref_fp32=TorchLowRank(RK, RV, dtype=torch.float32), torch_bf16=TorchLowRank(RK, RV),
         fused_bf16=FastLowRank(RK, RV), fused_fp8=FastLowRank(RK, RV, fp8=True, scales=scales))


def rot(x):
    return torch.cat([-x[..., D // 2:], x[..., :D // 2]], -1)


def rope(x, pos):  # x [..., n, H, D] content -> rotated
    c, s = rope_table(pos)
    c, s = torch.cat([c, c], -1)[:, None], torch.cat([s, s], -1)[:, None]
    return x * c + rot(x) * s


def attn(q, k, v, qpos):  # q [LT, nq, HQ, D]; k, v [LT, n, H, D]
    n = k.shape[1]
    q = q.permute(0, 2, 1, 3); k = k.permute(0, 2, 1, 3).repeat_interleave(4, 1); v = v.permute(0, 2, 1, 3).repeat_interleave(4, 1)
    s = q @ k.transpose(-1, -2) / math.sqrt(D)
    s.masked_fill_(torch.arange(n, device=dev)[None, :] > qpos[:, None], float("-inf"))
    return s.softmax(-1) @ v


err = {m: [0.0, 0.0] for m in M}
ro = {m: [torch.zeros(LT, device=dev), torch.zeros(LT, device=dev)] for m in M}
for d in docs:
    n = d["sk"].shape[1]
    pos = torch.arange(n, device=dev)
    sk = rope(d["sk"].to(dev).float(), pos).to(torch.bfloat16).contiguous()      # what the source cache holds
    sv = d["sv"].to(dev).to(torch.bfloat16).contiguous()
    outs = {m: f(sk, sv).float() for m, f in M.items()}
    ref = outs["ref_fp32"]
    q = rope(d["tq"].to(dev).float(), d["qpos"].to(dev))
    on = attn(q, rope(d["tk"].to(dev).float(), pos), d["tv"].to(dev).float(), d["qpos"].to(dev))
    for m, o in outs.items():
        err[m][0] += (o - ref).square().sum().item(); err[m][1] += ref.square().sum().item()
        oo = attn(q, o[0], o[1], d["qpos"].to(dev))
        ro[m][0] += (oo - on).square().sum((1, 2, 3)); ro[m][1] += on.square().sum((1, 2, 3))
res = {}
for m in M:
    res[m] = dict(rel_err_vs_fp32=math.sqrt(err[m][0] / err[m][1]), readout_err=(ro[m][0] / ro[m][1]).mean().item())
    print(f"{m:12s} rel err vs fp32 {res[m]['rel_err_vs_fp32']:.2e}   readout err {res[m]['readout_err']:.4f}")
json.dump(res, open(f"{HOME}/kvcodesign/results/fastlr_correctness_k{RK}_v{RV}.json", "w"), indent=1)
