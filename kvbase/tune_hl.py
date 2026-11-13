import glob, os, numpy as np, torch
from fastmap import FastHeadLocal
CB = sorted(glob.glob(os.path.expanduser("~/kvbase/mappers/cbhl_*")))[-1]
N, LS, H, D = 32768, 28, 8, 128
sk = (torch.randn(LS, N, H, D, device="cuda") * 4).to(torch.bfloat16); sv = torch.randn(LS, N, H, D, device="cuda").to(torch.bfloat16)
out = torch.empty(2, 36, N, H, D, device="cuda", dtype=torch.bfloat16)
ref = None
for BM in (32, 64, 128):
    for nw in (2, 4, 8):
        for ns in (1, 2):
            try:
                f = FastHeadLocal(CB, BM=BM, num_warps=nw, num_stages=ns)
                o = f(sk, sv, out=out); torch.cuda.synchronize()
                if ref is None: ref = o[:, :, :2048].clone()
                err = ((o[:, :, :2048].float() - ref.float()).norm() / ref.float().norm()).item()
                ts = []
                for r in range(8):
                    a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
                    a.record(); f(sk, sv, out=out); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
                print(BM, nw, ns, round(float(np.median(ts[2:])), 2), "err", round(err, 5), flush=True)
            except Exception as e:
                print(BM, nw, ns, "fail", type(e).__name__, str(e)[:160], flush=True); torch.cuda.empty_cache()
