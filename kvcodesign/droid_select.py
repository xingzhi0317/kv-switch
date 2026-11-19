# DroidSpeak offline profiling -> chosen recompute group (paper, Sec. on profiling): for every contiguous span [a, b)
# (2-layer granularity) the mean F1 on 50 HotpotQA contexts (questions 200-249, disjoint from the evaluation's 0-199);
# Pareto front = best F1 for each number of recomputed layers; chosen = the fewest recomputed layers whose F1 is within
# 5% (relative) of the receiver's own F1 (native).  Usage: python droid_select.py DIR   (reads the profiling json)
import sys, json, numpy as np
DIR = sys.argv[1]
r = json.load(open(f"results/{DIR}/droid_profile_hotpot.json"))["samples"]
f1 = {m: float(np.mean([s["m"][m]["score"] for s in r])) for m in r[0]["m"]}
nat = f1.pop("native")
print(f"{DIR}: {len(r)} contexts; receiver's own F1 {nat:.3f}; reuse everything (droid0_0) {f1['droid0_0']:.3f}")
best = {}
for m, v in f1.items():
    a, b = map(int, m[5:].split("_"))
    k = b - a
    if k not in best or v > best[k][1]:
        best[k] = (m, v)
front, top = [], -1
for k in sorted(best):
    if best[k][1] > top:
        front.append((k, *best[k])); top = best[k][1]
print("Pareto front (layers recomputed, span, F1, % of native):")
for k, m, v in front:
    print(f"  {k:2d}  {m:12s} {v:.3f}  {100 * v / nat:5.1f}%")
ok = [(k, m, v) for k, m, v in front if v >= 0.95 * nat]
chosen = ok[0] if ok else None
print("chosen (within 5% of native):", chosen)
print("per-start-layer best span:", {a: max(((m, round(v, 3)) for m, v in f1.items() if int(m[5:].split('_')[0]) == a), key=lambda x: x[1]) for a in range(0, 28, 2)})
json.dump(dict(native=nat, f1=f1, front=front, chosen=chosen), open(f"results/{DIR}/droid_select.json", "w"), indent=1)
