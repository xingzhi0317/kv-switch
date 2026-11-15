# Per-question test of "translation inherits what the source model understood":
# compare the source model answering by itself (src_native, Qwen3-1.7B reads and answers) with the target answering
# from a translated cache. If translation only carries the source's understanding, translated correctness should
# track source correctness question by question.  HotpotQA F1 is binarised at >= 0.5 for the conditional rates.
import os, json, numpy as np
H = os.path.expanduser("~")
TASKS = ["quality", "hotpotqa_e", "gsm8k", "gsm8k_h"]
SRC = {t: [f"{H}/kvcodesign/results/quality_{t}_{x}.json" for x in ("src", "rrr", "rrr_a", "rrr_b", "lrf", "h")] +
          [f"{H}/kvbase/results/quality_{t}_dense.json"] for t in TASKS}
TR = ["rrr_k768_v1536", "lrf_bf16", "kvbridge", "rrr_k2048_v2048"]
out = {}
rng = np.random.default_rng(0)
for t in TASKS:
    per = {}
    for f in SRC[t]:
        if os.path.exists(f):
            for i, s in enumerate(json.load(open(f))["samples"]):
                for m, r in s["m"].items():
                    per.setdefault(m, {})[i] = r["score"]
    if "src_native" not in per:
        continue
    print(f"== {t}")
    row = {}
    for m in ["native", "src_native"] + TR:
        if m in per:
            row[m] = dict(n=len(per[m]), mean=float(np.mean(list(per[m].values()))))
            print(f"  {m:18s} n={len(per[m]):3d} mean={row[m]['mean']:.3f}")
    s_ = per["src_native"]
    for m in TR:
        if m not in per:
            continue
        idx = sorted(set(s_) & set(per[m]) & set(per.get("native", per[m])))
        a = np.array([s_[i] for i in idx]); b = np.array([per[m][i] for i in idx])
        ab, bb = a >= 0.5, b >= 0.5
        nat = np.array([per["native"][i] for i in idx]) >= 0.5 if "native" in per else None
        r = dict(n=len(idx), corr=float(np.corrcoef(a, b)[0, 1]),
                 p_tr_right_given_src_right=float(bb[ab].mean()) if ab.any() else None,
                 p_tr_right_given_src_wrong=float(bb[~ab].mean()) if (~ab).any() else None,
                 agreement=float((ab == bb).mean()))
        if nat is not None:   # baseline: how well does the TARGET's own correctness predict translated correctness?
            r.update(corr_with_target_native=float(np.corrcoef(nat.astype(float), b)[0, 1]),
                     p_tr_right_given_tgt_right=float(bb[nat].mean()) if nat.any() else None,
                     p_tr_right_given_tgt_wrong=float(bb[~nat].mean()) if (~nat).any() else None)
        row[m + "_vs_src"] = r
        print(f"  {m:18s} vs src: corr {r['corr']:+.2f} | P(right|src right) {r['p_tr_right_given_src_right']:.2f}  P(right|src wrong) {r['p_tr_right_given_src_wrong']:.2f}"
              + (f" || vs target-native: corr {r['corr_with_target_native']:+.2f} | P(right|tgt right) {r['p_tr_right_given_tgt_right']:.2f}  P(right|tgt wrong) {r['p_tr_right_given_tgt_wrong']:.2f}" if nat is not None else ""))
    out[t] = row
json.dump(out, open(f"{H}/kvcodesign/results/compare_src.json", "w"), indent=1)
