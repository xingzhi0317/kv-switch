# RaReCache selector statistics (arXiv 2610.11358, re-implemented): for every target layer and K/V, the top-r
# eigenvectors of the covariance of the map's centred predictions on calibration data.  CPU.  Our code.
# Map = kvbridge mapper (mappers/kvbridge_{DIR}, official code).  Calibration KV = the official collector's
# FineWeb-Edu 500 x 1024 tokens (mappers/calib_17to4.pt, content-space keys; swapped for 4to17); 32768 tokens.
# Output: mappers/rare_vr_{DIR}.pt = dict(Vk [LT, H*D, r], Vv, myk [LT, H*D], myv)     env: DIR R THREADS
import os, sys, torch
torch.set_num_threads(int(os.environ.get("THREADS", 32)))
sys.path.insert(0, os.path.expanduser("~/kvbase/kvbridge"))
from kvbridge import Mapper
DIR, R = os.environ.get("DIR", "17to4"), int(os.environ.get("R", 128))
HOME = os.path.expanduser("~")
sk, sv, tk, tv = torch.load(f"{HOME}/kvbase/mappers/calib_17to4.pt", mmap=True)
if DIR == "4to17":
    sk, sv = tk, tv
mp = Mapper.load(f"{HOME}/kvbase/mappers/kvbridge_{DIR}", dtype=torch.float32)
idx = torch.randperm(sk[0].shape[0], generator=torch.Generator().manual_seed(0))[:32768]
out = {"Vk": [], "Vv": [], "myk": [], "myv": []}
for t, e in enumerate(mp.layers):
    for kind, src in (("k", sk), ("v", sv)):
        X = torch.cat([src[s][idx].float().reshape(len(idx), -1) for s in e["source_layers"]], 1)
        z = (X - e[f"mu_x{kind}"].float()) @ e[f"W_{kind}"].float()                         # centred predictions
        ev, U = torch.linalg.eigh((z.T @ z / len(idx)).double())
        out[f"V{kind}"].append(U[:, -R:].flip(1).float())
        out[f"my{kind}"].append(e[f"mu_y{kind}"].float().reshape(-1))
        if t % 6 == 0:
            print(DIR, "layer", t, kind, "top-r share of prediction variance", round(float(ev[-R:].sum() / ev.sum()), 3), flush=True)
torch.save({k: torch.stack(v) for k, v in out.items()}, f"{HOME}/kvcodesign/mappers/rare_vr_{DIR}.pt")
print("RAREFITDONE", DIR)
