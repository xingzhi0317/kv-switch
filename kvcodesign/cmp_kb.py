# compare two calibrated kvbridge-format mappers made from the same initial map: relative difference of the V correction
# usage: python cmp_kb.py <init dir> <calibrated A> <calibrated B>
import sys, torch
from kvbridge import Mapper
I, A, B = (Mapper.load(p, device="cpu", dtype=torch.float32) for p in sys.argv[1:4])
rows = []
for t, (i, a, b) in enumerate(zip(I.layers, A.layers, B.layers)):
    da = torch.cat([(a["W_v"] - i["W_v"]).flatten(), (a["mu_yv"] - i["mu_yv"]).flatten()]).double()
    db = torch.cat([(b["W_v"] - i["W_v"]).flatten(), (b["mu_yv"] - i["mu_yv"]).flatten()]).double()
    rows.append((t, float((da - db).norm() / da.norm()), float(da.norm() / i["W_v"].double().norm())))
for r in rows[::4] + [rows[-1]]:
    print("layer %2d  correction rel diff %.3f  correction / weights %.3f" % r)
print("median correction rel diff %.3f" % sorted(r[1] for r in rows)[len(rows) // 2])
