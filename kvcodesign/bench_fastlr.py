# Translation latency on one RTX 4090 (same protocol as kvbase/bench_translate.py: only the translation call is timed,
# CUDA events, warmup then median). Tiles of the fused low-rank kernels are tuned at 32K first.
# Baselines: unfused torch low-rank (cuBLAS), previous-stage fused head-local (CacheBridge weights) and dense
# (kvbridge weights, bf16 / FP8) kernels. Official CacheBridge / kvbridge numbers: kvbase/results/translate_latency.json.
import os, sys, json, itertools, numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/kvbase"))
from fastlr import FastLowRank, TorchLowRank, LS, LT, H, D
HOME = os.path.expanduser("~")
dev = "cuda"
LENS = [int(x) for x in os.environ.get("LENS", "2048,8192,32768").split(",")]
REPS, WARM = 20, 5
OUT = f"{HOME}/kvcodesign/results/fastlr_latency.json"
RK, RV = 768, 1536
scales = json.load(open(f"{HOME}/kvcodesign/data/fp8_scales_k{RK}_v{RV}.json"))


def timeit(fn, *a):
    xs = []
    for r in range(WARM + REPS):
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); fn(*a); e1.record(); torch.cuda.synchronize()
        if r >= WARM:
            xs.append(e0.elapsed_time(e1))
    return float(np.median(xs))


def inputs(T):
    g = torch.Generator(device=dev).manual_seed(0)
    return (torch.randn(LS, T, H, D, device=dev, generator=g).to(torch.bfloat16),
            torch.randn(LS, T, H, D, device=dev, generator=g).to(torch.bfloat16))


import fastlr
res = dict(cfg=dict(gpu=torch.cuda.get_device_name(), rk=RK, rv=RV, macs_per_token=(LS * H * D + LT * H * D) * (RK + RV),
                    tiles_bf16=str(fastlr.CFG_BF16), tiles_fp8=str(fastlr.CFG_FP8)))
# ---- methods
from fastmap import FastHeadLocal, FastDense
from kvbridge import Mapper as KBMapper
kb = KBMapper.load(f"{HOME}/kvbase/mappers/kvbridge_17to4", device=dev)
M = {
    "lowrank_fused_bf16": FastLowRank(RK, RV),
    "lowrank_fused_fp8": FastLowRank(RK, RV, fp8=True, scales=scales),
    "lowrank_torch_bf16": TorchLowRank(RK, RV),
    "headlocal_fused_bf16 (prev stage)": FastHeadLocal(f"{HOME}/kvbase/mappers/cachebridge_17to4"),
    "dense_fused_bf16 (prev stage)": FastDense(kb),
    "dense_fused_fp8 (prev stage)": FastDense(kb, fp8=True),
}
macs = {"lowrank": (LS * H * D + LT * H * D) * (RK + RV), "headlocal": LT * H * 8 * D * D * 2, "dense": LT * 12 * H * D * H * D * 2}
res["latency_ms"] = {}
for T in LENS:
    sk, sv = inputs(T)
    row = {}
    for name, f in M.items():
        ms = timeit(f, sk, sv)
        mac = macs[name.split("_")[0]]
        row[name] = dict(ms=ms, tflops=2 * mac * T / ms / 1e9)
        print(f"T={T:6d} {name:36s} {ms:8.2f} ms  {row[name]['tflops']:6.1f} TFLOPS", flush=True)
    res["latency_ms"][T] = row
    del sk, sv
    torch.cuda.empty_cache()
json.dump(res, open(OUT, "w"), indent=1)
print("BENCHDONE")
