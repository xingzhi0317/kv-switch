# Merge downstream results by question index and compare every method with kvbridge (dense) via paired bootstrap.
# Sources: kvcodesign/results/quality_{task}_{rrr,rrr_a,rrr_b,cbfull}.json (this stage) and
#          kvbase/results/quality_{task}_{dense,cbinit}.json (previous stage, same deterministic question order).
import os, json, numpy as np
H = os.path.expanduser("~")
TASKS = ["quality", "hotpotqa_e", "gsm8k", "gsm8k_h"]
FILES = [f"{H}/kvcodesign/results/quality_{{t}}_{x}.json" for x in ("rrr", "rrr_a", "rrr_b", "cbfull", "lrf", "src", "h")] + \
        [f"{H}/kvbase/results/quality_{{t}}_{x}.json" for x in ("dense", "cbinit")]
REF = "kvbridge"
out = {}
for t in TASKS:
    per = {}  # method -> {idx: score}
    nat = {}
    for f in FILES:
        p = f.format(t=t)
        if not os.path.exists(p):
            continue
        src = "prev" if "/kvbase/" in p else "new"
        for i, s in enumerate(json.load(open(p))["samples"]):
            for m, r in s["m"].items():
                key = m if m != "native" else f"native_{src}"
                if m == "cb_official" and "cbinit" in p:
                    key = "cb_init_official"
                if m == "cb_official" and "cbfull" in p:
                    key = "cb_full_official"
                per.setdefault(key, {})[i] = r["score"]
    rows = {}
    rng = np.random.default_rng(0)
    for m, d in per.items():
        idx = sorted(set(d) & set(per.get(REF, {})))
        row = dict(n=len(d), mean=float(np.mean(list(d.values()))))
        if idx and m != REF:
            a = np.array([d[i] for i in idx]); b = np.array([per[REF][i] for i in idx])
            diff = a - b
            bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)]
            row.update(paired_n=len(idx), diff_vs_kvbridge=float(diff.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        rows[m] = row
    out[t] = rows
    print("==", t)
    for m, r in sorted(rows.items(), key=lambda kv: -kv[1]["mean"]):
        extra = f"  vs kvbridge {r['diff_vs_kvbridge']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] (n={r['paired_n']})" if "ci95" in r else ""
        print(f"  {m:24s} n={r['n']:3d} mean={r['mean']:.4f}{extra}")
# noise band: fp32 vs bf16 of the same low-rank mapper
for t in TASKS:
    a = out[t].get("rrr_k768_v1536"); b = out[t].get("rrr_k768_v1536_bf16")
    if a and b:
        print("noise band", t, "fp32-bf16 =", round(a["mean"] - b["mean"], 4))
json.dump(out, open(f"{H}/kvcodesign/results/quality_summary.json", "w"), indent=1)
