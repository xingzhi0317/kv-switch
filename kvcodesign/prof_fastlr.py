# Per-kernel timing of the fused low-rank translator at 32K vs cuBLAS GEMMs of the same shapes.
import os, json, itertools, numpy as np, torch, triton
import fastlr
from fastlr import FastLowRank, LS, LT, H, D, rope_table
dev = "cuda"
T = 32768
sc = json.load(open(os.path.expanduser("~/kvcodesign/data/fp8_scales_k768_v1536.json")))
g = torch.Generator(device=dev).manual_seed(0)
sk = torch.randn(LS, T, H, D, device=dev, generator=g).to(torch.bfloat16); sv = torch.randn_like(sk)
cos, sin = rope_table(torch.arange(T, device=dev))
out = torch.empty(2, LT, T, H, D, device=dev, dtype=torch.bfloat16)


def tm(fn, n=10):
    for _ in range(3): fn()
    torch.cuda.synchronize(); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / n


for fp8 in (False, True):
    m = FastLowRank(fp8=fp8, scales=sc)
    zdt = torch.float8_e4m3fn if fp8 else torch.bfloat16
    for kind, x, i in (("k", sk, 0), ("v", sv, 1)):
        p = m.parts[kind]; r = p["r"]; z = torch.empty(T, r, device=dev, dtype=zdt)
        enc = {}
        for (BM, BN, w, s) in [(64, 256, 8, 2), (128, 256, 8, 1), (128, 128, 8, 2), (128, 128, 4, 2), (256, 128, 8, 1), (128, 192, 8, 1)]:
            if r % BN: continue
            try:
                enc[(BM, BN, w, s)] = tm(lambda: fastlr._encode[(triton.cdiv(T, BM) * triton.cdiv(r, BN),)](
                    x, p["P"], p["zb"], p["wce"], p["xs"], z, cos, sin, T, r, x.stride(0), x.stride(1), z.stride(0),
                    NU=LS * H, H=H, D=D, ROPE=(kind == "k"), FP8=fp8, BM=BM, BN=BN, ZFP8=fp8, num_warps=w, num_stages=s))
            except Exception as e:
                pass
        dec = {}
        for (BM, BK, w, s, G) in [(BM, BK, w, s, G) for (BM, BK, w, s) in [(128, 64, 8, 3), (128, 32, 4, 4), (64, 64, 4, 3), (256, 32, 8, 3)] for G in (8, 16, 32)]:
            try:
                dec[(BM, BK, w, s, G)] = tm(lambda: fastlr._decode[(triton.cdiv(T, BM) * LT * H,)](
                    z, p["D"], p["yb"], p["wcd"], out[i], cos, sin, T, r, z.stride(0), out.stride(1), out.stride(2),
                    H=H, D=D, ROPE=(kind == "k"), FP8=fp8, BM=BM, BK=BK, GROUP_M=G, NU=LT * H, num_warps=w, num_stages=s))
            except Exception as e:
                pass
        be = min(enc, key=enc.get); bd = min(dec, key=dec.get)
        fe = 2 * T * LS * H * D * r; fd = 2 * T * r * LT * H * D
        print(f"{'fp8' if fp8 else 'bf16'} {kind} r={r}: encode best {be} {enc[be]:.2f} ms ({fe/enc[be]/1e9:.0f} TF) | decode best {bd} {dec[bd]:.2f} ms ({fd/dec[bd]/1e9:.0f} TF)", flush=True)
        print("   enc all", {k: round(v, 2) for k, v in enc.items()}, flush=True)
        print("   dec top", sorted(((round(v, 2), k) for k, v in dec.items()))[:4], flush=True)
