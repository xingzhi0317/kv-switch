# (1) several requests concatenated along T with their own positions == translating each separately;
# (2) throughput of batched translation; (3) CUDA Graph capture at small lengths.
import os, json, numpy as np, torch
from fastlr import FastLowRank, LS, LT, H, D
dev = "cuda"
sc = json.load(open(os.path.expanduser("~/kvcodesign/data/fp8_scales_k768_v1536.json")))
res = {}
g = torch.Generator(device=dev).manual_seed(0)
rnd = lambda T: torch.randn(LS, T, H, D, device=dev, generator=g).to(torch.bfloat16)


def tm(fn, n=20):
    for _ in range(5): fn()
    torch.cuda.synchronize(); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / n


for fp8 in (False, True):
    tag = "fp8" if fp8 else "bf16"
    m = FastLowRank(fp8=fp8, scales=sc)
    lens = [1000, 2048, 777, 4096]
    ks, vs = [rnd(n) for n in lens], [rnd(n) for n in lens]
    sep = [m(k, v).clone() for k, v in zip(ks, vs)]
    pos = torch.cat([torch.arange(n, device=dev) for n in lens])
    cat = m(torch.cat(ks, 1), torch.cat(vs, 1), positions=pos)
    ref = torch.cat(sep, 2)
    diff = (cat.float() - ref.float()).abs().max().item()
    res[f"concat_maxabs_diff_{tag}"] = diff
    print(tag, "concat vs separate max abs diff", diff, flush=True)
    for T, B in ((2048, 1), (2048, 4), (2048, 16), (8192, 4)):
        k, v = rnd(T * B), rnd(T * B)
        p = torch.arange(T, device=dev).repeat(B)
        ms = tm(lambda: m(k, v, positions=p))
        res[f"batch_{tag}_{T}x{B}"] = dict(ms=ms, Mtok_per_s=T * B / ms / 1e3)
        print(f"{tag} {T}x{B}: {ms:.2f} ms  {T * B / ms / 1e3:.2f} M tok/s", flush=True)
    for T in (512, 2048):
        k, v = rnd(T), rnd(T)
        out = torch.empty(2, LT, T, H, D, device=dev, dtype=torch.bfloat16)
        eager = tm(lambda: m(k, v, out=out))
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): m(k, v, out=out)
        torch.cuda.current_stream().wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            m(k, v, out=out)
        ref = out.clone(); gr.replay(); torch.cuda.synchronize()
        same = torch.equal(ref, out)
        graph = tm(lambda: gr.replay())
        res[f"graph_{tag}_{T}"] = dict(eager_ms=eager, graph_ms=graph, identical=same)
        print(f"{tag} T={T}: eager {eager:.3f} ms, graph {graph:.3f} ms, identical {same}", flush=True)
json.dump(res, open(os.path.expanduser("~/kvcodesign/results/fastlr_batch_graph.json"), "w"), indent=1)
