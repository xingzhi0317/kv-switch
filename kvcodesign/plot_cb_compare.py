# Small -> large, CacheBridge protocol: retention per task, the CacheBridge paper (Qwen3 8B->14B, arXiv 2609.00891
# Table 25) vs our runs on Qwen3 1.7B->4B with the same history/query split (paper Table 6):
#   QuALITY (article | question + options, A-D logits), HotpotQA-E (passages | question, F1),
#   SQuAD v1.1 = squad_cb (passage | instruction + question, F1), GSM8K = gsm8k_h (8 shots + test question | answer).
# Ours: retention = method score / re-prefill score on the same questions (results/retention_table.json).
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ours = json.load(open("results/retention_table.json"))["17to4"]
tasks = [("quality", "QuALITY"), ("hotpotqa_e", "HotpotQA-E"), ("squad_cb", "SQuAD"), ("gsm8k_h", "GSM8K")]
paper = [("CacheBridge", [97.09, 88.02, 85.39, 100.00], "#2a6fdb"),
         ("CacheBridge w/o calibration", [62.12, 21.62, 31.63, 6.45], "#9cbcf0"),
         ("CFM (= kvbridge)", [44.27, 19.81, 45.41, 9.72], "#e08a2c")]
mine = [("CacheBridge (our reproduction)", "cb_ec_official", "#2a6fdb"),
        ("CacheBridge variant: all-head map, 3 layers", "kb3cb", "#6a8fd0"),
        ("CacheBridge variant: all-head map, 12 layers", "kb12cb", "#1b3f80"),
        ("kvbridge (official code)", "kvbridge", "#e08a2c"),
        ("KL-trained translator", "kb3kl", "#8c8c8c"),
        ("ours: low-rank (fused)", "lrf_bf16", "#2e9e5b"),
        ("ours: low-rank 2048", "rrr_k2048_v2048", "#7fcf9b")]
fig, ax = plt.subplots(1, 2, figsize=(16, 5), sharey=True, gridspec_kw=dict(width_ratios=[3, 6]))
x = np.arange(len(tasks))


def bars(a, rows, w):
    for i, (lab, vals, col) in enumerate(rows):
        off = (i - (len(rows) - 1) / 2) * w
        for xi, (y, n) in zip(x, vals):
            if y is None:
                a.text(xi + off, 1.5, "–", ha="center", fontsize=7); continue
            a.bar(xi + off, y, w, color=col, label=lab if xi == 0 else None)
            a.text(xi + off, y + 1.5, f"{y:.0f}" + ("" if n is None or n >= 200 else f"\n({n}q)"), ha="center", fontsize=6.5)


bars(ax[0], [(l, [(v, None) for v in vals], c) for l, vals, c in paper], 0.26)
bars(ax[1], [(l, [tuple(ours.get(k, {}).get(t, [None, None])) for t, _ in tasks], c) for l, k, c in mine], 0.118)
for a, title in zip(ax, ["paper: Qwen3 8B -> 14B", "ours: Qwen3 1.7B -> 4B (same protocol)"]):
    a.set_xticks(x, [n for _, n in tasks]); a.set_title(title); a.axhline(100, color="gray", lw=0.6, ls="--")
    a.grid(axis="y", alpha=0.3); a.legend(fontsize=7, loc="upper right")
ax[0].set_ylabel("retention (% of re-prefill)"); ax[0].set_ylim(0, 135)
fig.suptitle("Small -> large switch, CacheBridge evaluation protocol (history/query split as in its Table 6). "
             "Ours: 200 questions/task unless marked", fontsize=10)
plt.tight_layout()
plt.savefig("results/cb_paper_vs_ours.png", dpi=110)
print("saved")
