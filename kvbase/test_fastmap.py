# Correctness (vs official implementations) + latency of the fused kernels.
import os, json, glob, numpy as np, torch
from fastmap import FastHeadLocal, FastDense
from cachebridge.mapper import PageMapper
from kvbridge import Mapper as KBMapper, transfer as kb_transfer
from transformers import DynamicCache

E = os.environ.get
ROOT = os.path.expanduser("~/kvbase")
dev = "cuda"
LS, H, D = 28, 8, 128
CB = E("MAPPER_CB") or sorted(glob.glob(f"{ROOT}/mappers/cbhl_*"))[-1]
LENS = [int(x) for x in E("LENS", "2048,4096,8192,16384,32768").split(",")]
REPS, WARM = 10, 3
OUT = f"{ROOT}/results/fastmap.json"


def ev():
    e = torch.cuda.Event(enable_timing=True); e.record(); return e


def timeit(fn):
    xs, peaks = [], []
    for r in range(WARM + REPS):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
        a = ev(); o = fn(); b = ev(); torch.cuda.synchronize()
        if r >= WARM:
            xs.append(a.elapsed_time(b)); peaks.append((torch.cuda.max_memory_allocated() - base) / 2**30)
        del o
    return dict(p50=float(np.median(xs)), p95=float(np.percentile(xs, 95)), extra_peak_GB=float(np.median(peaks)))


def relerr(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


res = dict(cfg=dict(cb_mapper=CB, gpu=torch.cuda.get_device_name()))
# ---------------- head-local: vs CacheBridge PageMapper (FP32 reference) ----------------
fh = FastHeadLocal(CB)
pm = PageMapper(CB, method="cachebridge", source_rope_theta=1e6, target_rope_theta=1e6, device=dev)
torch.manual_seed(0)
T0 = 4096
sk = (torch.randn(LS, T0, H, D, device=dev) * 4).to(torch.bfloat16)
sv = torch.randn(LS, T0, H, D, device=dev).to(torch.bfloat16)
ref = torch.cat([pm.map_page(torch.stack([sk[:, p:p + 2048], sv[:, p:p + 2048]]), position_start=p, output_device=dev)
                 for p in range(0, T0, 2048)], 2)
got = fh(sk, sv)
res["headlocal_check"] = dict(K_relerr=relerr(got[0], ref[0]), V_relerr=relerr(got[1], ref[1]),
                              K_maxabs=(got[0].float() - ref[0].float()).abs().max().item())
print("headlocal check", res["headlocal_check"], flush=True)

# ---------------- dense: vs kvbridge.transfer ----------------
kb = KBMapper.load(f"{ROOT}/mappers/kvbridge_17to4", device=dev)
fd = FastDense(kb)
fd8 = FastDense(kb, fp8=True)
c = DynamicCache()
for l in range(LS):
    c.update(sk[l].permute(1, 0, 2)[None].contiguous(), sv[l].permute(1, 0, 2)[None].contiguous(), l)
refc = kb_transfer(c, kb)
refk = torch.stack([refc.layers[t].keys[0].permute(1, 0, 2) for t in range(fd.LT)])
refv = torch.stack([refc.layers[t].values[0].permute(1, 0, 2) for t in range(fd.LT)])
for name, f in (("dense_bf16", fd), ("dense_fp8", fd8)):
    o = f(sk, sv)
    res[f"{name}_check"] = dict(K_relerr=relerr(o[0], refk), V_relerr=relerr(o[1], refv))
    print(name, "check", res[f"{name}_check"], flush=True)
del sk, sv, ref, got, refc, c
torch.cuda.empty_cache()

# ---------------- latency ----------------
for N in LENS:
    torch.manual_seed(N)
    sk = (torch.randn(LS, N, H, D, device=dev) * 4).to(torch.bfloat16)
    sv = torch.randn(LS, N, H, D, device=dev).to(torch.bfloat16)
    out = torch.empty(2, fh.LT, N, H, D, device=dev, dtype=torch.bfloat16)
    row = {}
    row["fast_headlocal"] = timeit(lambda: fh(sk, sv, out=out))
    row["fast_dense_bf16"] = timeit(lambda: fd(sk, sv, out=out))
    row["fast_dense_fp8"] = timeit(lambda: fd8(sk, sv, out=out))
    res[str(N)] = row
    print(N, {k: round(v["p50"], 2) for k, v in row.items()}, flush=True)
    del sk, sv, out; torch.cuda.empty_cache()
json.dump(res, open(OUT, "w"), indent=1)
print("FASTDONE")
