# Same-precision comparison: CacheBridge official (FP32) vs fused FP32-ieee / FP32-tf32x3 / BF16 kernels.
import glob, os, json, numpy as np, torch
from fastmap import FastHeadLocal, FastHeadLocalF32, CublasHeadLocalF32
from cachebridge.mapper import PageMapper

torch.backends.cuda.matmul.allow_tf32 = False  # PyTorch default; CacheBridge runs true FP32
CB = os.environ.get("MAPPER_CB") or os.path.expanduser("~/kvbase/mappers/cachebridge_17to4_init")
dev = "cuda"
LS, H, D = 28, 8, 128
pm = PageMapper(CB, method="cachebridge", source_rope_theta=1e6, target_rope_theta=1e6, device=dev)
V = {"bf16": FastHeadLocal(CB), "cublas_fp32": CublasHeadLocalF32(CB)}
for prec in ("tf32x3",):
    for BM, nw in ((16, 4), (16, 8), (16, 2)):
        try:
            f = FastHeadLocalF32(CB, prec=prec, BM=BM, num_warps=nw)
            x = torch.randn(LS, 256, H, D, device=dev).to(torch.bfloat16)
            f(x, x); torch.cuda.synchronize()
            V[f"f32_{prec}_BM{BM}_w{nw}"] = f
        except Exception as e:
            print("skip", prec, BM, nw, type(e).__name__, str(e)[:100])
res = {}
torch.manual_seed(0)
T0 = 4096
sk = (torch.randn(LS, T0, H, D, device=dev) * 4).to(torch.bfloat16)
sv = torch.randn(LS, T0, H, D, device=dev).to(torch.bfloat16)
ref = torch.cat([pm.map_page(torch.stack([sk[:, p:p + 2048], sv[:, p:p + 2048]]), position_start=p, output_device=dev)
                 for p in range(0, T0, 2048)], 2)
for k, f in V.items():
    o = f(sk, sv)
    d = o.float() - ref.float()
    res[k] = dict(relerr=(d.norm() / ref.float().norm()).item(), exact_bf16_match=(o == ref).float().mean().item())
    print(k, res[k], flush=True)


def tm(fn, reps=8):
    ts = []
    for r in range(reps + 2):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        if r >= 2:
            ts.append(a.elapsed_time(b))
    return float(np.median(ts))


for N in (8192, 32768):
    sk = (torch.randn(LS, N, H, D, device=dev) * 4).to(torch.bfloat16)
    sv = torch.randn(LS, N, H, D, device=dev).to(torch.bfloat16)
    out = torch.empty(2, 36, N, H, D, device=dev, dtype=torch.bfloat16)
    ob = torch.empty(2, 36, 2048, H, D, device=dev, dtype=torch.bfloat16)

    def cb():
        for p in range(0, N, 2048):
            pm.map_page(torch.stack([sk[:, p:p + 2048], sv[:, p:p + 2048]]), position_start=p, output_device=ob.device, output=ob)

    row = {"cachebridge_official": tm(cb)}
    for k, f in V.items():
        row[k] = tm(lambda: f(sk, sv, out=out))
    res[str(N)] = row
    print(N, {k: round(v, 1) for k, v in row.items()}, flush=True)
    del sk, sv, out
    torch.cuda.empty_cache()
json.dump(res, open(os.path.expanduser("~/kvbase/results/f32_compare.json"), "w"), indent=1)
print("F32DONE")
