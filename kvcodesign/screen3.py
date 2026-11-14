# Asymmetric K/V latent ranks from the saved reduced-rank ridge mappers (screen2, metric "var").
import os
os.environ.setdefault("OUT", os.path.expanduser("~/kvcodesign/results/screen3.json"))
from common import *
M = lambda r: torch.load(f"{HOME}/kvcodesign/mappers/rrr_var_r{r}.pt")
for rk, rv in ((256, 1024), (512, 1024), (512, 1536), (768, 1536), (512, 2048), (1024, 2048), (2048, 1024), (2048, 512), (2048, 2048)):
    a, b = M(rk), M(rv)
    summarize(f"rrr_k{rk}_v{rv}", lambda xk, xv: (xk @ a["Pk"]) @ a["Dk"] * w[TK], lambda xk, xv: (xv @ b["Pv"]) @ b["Dv"] * w[TV],
              (PK + QK) * (rk + rv), dict(rank_k=rk, rank_v=rv))
# which side dominates: native K + mapped V, mapped K + native V is not available offline for val rows; use dense V / dense K
log("SCREEN3DONE")
