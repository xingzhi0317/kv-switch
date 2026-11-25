# One simple figure, Qwen3-1.7B -> 4B.  y = mean of three retentions with equal weight (reading QuALITY, long context HotpotQA,
# exact 8K passkey), each = score / re-prefill score on the same questions; x = TTFT speed-up over re-prefill, 8K history.
# CacheLink is not plotted: we have not reproduced it.  usage: PYTHONUTF8=1 python plot_simple.py
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"], "font.size": 13, "axes.axisbelow": True})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#ececea"
S = json.load(open("results/quality_summary.json")); tt = json.load(open("results/ttft_17to4.json"))["lens"]["8192"]; base = tt["native"]["p50"]
def ret(m):
    o = []
    for t in ("quality", "hotpotqa_e", "passkey_8k"):
        r, n = S[t][m], S[t]["native"]
        if r["n"] >= n["n"]: o.append(r["mean"] / n["mean"])
        else: v = r["vs_native"]; o.append((v["ref_mean"] + v["diff"]) / v["ref_mean"])
    return 100 * sum(o) / 3
P = [("Closed-form translation\n(KVBridge)", "#2a78d6", "o", [("kvbridge", "kvbridge", "")], (12, 0), "left"),
     ("Attention calibration\n(CacheBridge, our repro.)", "#a03c78", "s", [("cb_ec_official", "cb_official", "")], (12, 0), "left"),
     ("KL-trained translation\n(KV-Lingo-style, our reimpl.)", "#eb6834", "D", [("kb3cklbest", "kb3kl", "")], (12, 0), "left"),
     ("Translation + recompute (RaReCache)", "#1baf7a", "^", [("rare10", "rare10", "10%"), ("rare30", "rare30", "30%"), ("rare50", "rare50", "50%")], None, None)]
fig, ax = plt.subplots(figsize=(9.2, 6.2))
for nm, c, mk, seq, off, ha in P:
    pts = [(base / tt[k]["p50"], ret(q), g) for q, k, g in seq]
    print(nm.replace("\n", " "), [(round(x, 2), round(y, 1)) for x, y, _ in pts])
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, lw=2, alpha=.5, zorder=2)
    for x, y, g in pts:
        ax.scatter(x, y, s=150, marker=mk, color=c, edgecolor="white", linewidth=1.3, zorder=4)
        if g: ax.annotate(g, (x, y), xytext=(10, -2), textcoords="offset points", fontsize=11, color=INK2, va="center")
    if off: ax.annotate(nm, pts[0][:2], xytext=off, textcoords="offset points", fontsize=12, color=INK, va="center", ha=ha)
    else: ax.annotate("Translation + recompute\n(RaReCache, % recomputed)", pts[0][:2], xytext=(-10, -14), textcoords="offset points", fontsize=12, color=INK, va="top", ha="center")
st = ret("src_native"); print("stay", round(st, 1))
ax.axhline(st, color=INK2, lw=1.4, ls=":", zorder=1)
ax.text(20, st + 1.5, f"stay on 1.7B, no switch: {st:.0f}%", ha="right", fontsize=12, color=INK2)
ax.scatter(1, 100, s=330, marker="*", color=INK, edgecolor="white", zorder=5)
ax.annotate("re-prefill on 4B", (1, 100), xytext=(13, 0), textcoords="offset points", fontsize=12, color=INK, va="center")
ax.set_xscale("log"); ax.set_xlim(.6, 21); ax.set_ylim(0, 108)
ax.xaxis.set_major_locator(FixedLocator([.75, 1, 2, 3, 5, 10, 20])); ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%")); ax.grid(color=GRID, lw=.8); ax.tick_params(length=0, colors=INK2)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color("#b9b8b3")
ax.set_xlabel("TTFT speed-up over re-prefill (8K history)"); ax.set_ylabel("Retention (mean of reading, long context, exact)")
ax.set_title("Qwen3-1.7B → Qwen3-4B", loc="left", fontsize=14)
fig.tight_layout()
for e in ("png", "svg"): fig.savefig(f"results/simple_17to4.{e}", dpi=180, bbox_inches="tight")
