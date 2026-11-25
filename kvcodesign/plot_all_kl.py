# KL-trained translators evaluated with KV-Lingo / Hybrid-KV style tasks: absolute scores, with both models alone and the papers' values.
# 4B<->8B: results/hkv/res_hkv_*.json (2000 steps, nu=1); 1.7B<->4B: results/hkv/eval2_kvl3_* (long run) with eval2_natives_17to4.
# usage: PYTHONUTF8=1 python plot_all_kl.py
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"], "font.size": 11, "axes.axisbelow": True})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#ececea"
J = lambda f: json.load(open("results/hkv/" + f, encoding="utf8"))
m = lambda d: d["summary"]["mean"] if "summary" in d else d["mean"]
h84, h48, nat = J("res_hkv_8to4.json")["eval"], J("res_hkv_4to8.json")["eval"], J("eval2_natives_17to4.json")
k = {d: J(f"eval2_kvl3_{d}.json") for d in ("17to4", "4to17")}; r17 = J("res_kvl_17to4.json")["eval"]
T = [("arc_c", "ARC-C"), ("mmlu_pro", "MMLU-Pro"), ("coqa", "CoQA F1"), ("needle", "Needle, 2–16K")]
def big(e):      # 4B/8B pair: both models alone come from the 8to4 run
    return {t: dict(m4=m(h84[t]["tgt_native"]), m8=m(h84[t]["src_native"]), cf=m(e[t]["cf"]), sd=m(e[t]["sd"])) for t, _ in T}
def small(d):
    o = {}
    for t, _ in T:
        u = "coqa_multi_sys" if t == "coqa" else t
        if u not in k[d]: continue
        o[t] = dict(m17=m(nat[u]["src_native"]), m4=m(nat[u]["tgt_native"]), cf=m(k[d][u]["cf"]), sd=m(k[d][u]["sd"])) if u in nat else dict(cf=m(k[d][u]["cf"]), sd=m(k[d][u]["sd"]))
    return o
P = [("Qwen3-4B → 8B", big(h48), "m4", "m8", {"coqa": .697}), ("Qwen3-8B → 4B", big(h84), "m4", "m8", {"coqa": .686}),
     ("Qwen3-1.7B → 4B", small("17to4"), "m17", "m4", {"arc_c": .794, "mmlu_pro": .324}), ("Qwen3-4B → 1.7B", small("4to17"), "m17", "m4", {"arc_c": .817, "mmlu_pro": .283})]
fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.6), sharey=True)
for ax, (ttl, d, lo, hi, paper) in zip(axes.flat, P):
    for x, (t, nm) in enumerate(T):
        if t not in d: ax.text(x, .03, "not run", ha="center", fontsize=9.5, color="#8b8a85"); continue
        r = d[t]
        for off, key, c, mk, fc in ((-.24, lo, "#8b8a85", "o", "white"), (-.08, hi, INK, "*", INK), (.08, "cf", "#2a78d6", "s", "#2a78d6"), (.24, "sd", "#eb6834", "D", "#eb6834")):
            if key in r:
                ax.scatter(x + off, r[key], s=150 if mk == "*" else 75, marker=mk, facecolor=fc, edgecolor=c if fc == "white" else "white", linewidth=1.3, zorder=4)
                ax.text(x + off, r[key] + .045, f"{r[key]:.2f}", ha="center", fontsize=8.5, color=INK2)
        if t in paper: ax.scatter(x + .24, paper[t], s=110, marker="_", color=INK, linewidth=2, zorder=5)
    ax.set_xticks(range(len(T))); ax.set_xticklabels([n for _, n in T]); ax.set_ylim(-.04, 1.13); ax.set_xlim(-.5, len(T) - .5)
    ax.set_title(ttl, loc="left", fontsize=13, color=INK); ax.grid(axis="y", color=GRID, lw=.8); ax.tick_params(length=0, colors=INK2)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
h = [Line2D([], [], marker="o", ls="", ms=9, mfc="white", mec="#8b8a85", mew=1.4, label="smaller model alone"), Line2D([], [], marker="*", ls="", ms=13, mfc=INK, mec="white", label="larger model alone"),
     Line2D([], [], marker="s", ls="", ms=9, mfc="#2a78d6", mec="white", label="closed form"), Line2D([], [], marker="D", ls="", ms=9, mfc="#eb6834", mec="white", label="KL-trained"),
     Line2D([], [], marker="_", ls="", ms=12, mec=INK, mew=2, label="paper's KL-trained value")]
fig.legend(handles=h, loc="lower center", ncol=5, frameon=False); fig.tight_layout(rect=(0, .06, 1, 1), h_pad=2)
for e in ("png", "svg"): fig.savefig(f"results/all_kl_pairs.{e}", dpi=170, bbox_inches="tight")
