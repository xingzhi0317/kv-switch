# Same simple figure with a second measure: gain over the small model answering alone (points of retention; gain_vs_small.py).  x = TTFT speed-up.
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"], "font.size": 13, "axes.axisbelow": True})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#ececea"
G = json.load(open("results/gain_vs_small.json")); tt = json.load(open("results/ttft_17to4.json"))["lens"]["8192"]; base = tt["native"]["p50"]
P = [("Closed-form translation\n(KVBridge)", "#2a78d6", "o", [("kvbridge", "kvbridge", "")], "up"), ("Attention calibration\n(CacheBridge, our repro., n=50)", "#a03c78", "s", [("cb_ec_official", "cb_official", "")], "r"),
     ("KL-trained translation\n(KV-Lingo-style, our reimpl.)", "#eb6834", "D", [("kb3cklbest", "kb3kl", "")], "r"),
     ("Translation + recompute\n(RaReCache, % recomputed)", "#1baf7a", "^", [("rare10", "rare10", "10%"), ("rare30", "rare30", "30%"), ("rare50", "rare50", "50%")], None)]
fig, ax = plt.subplots(figsize=(9.2, 6.2))
for nm, c, mk, seq, lab in P:
    pts = [(base / tt[k]["p50"], G[q]["gain"], G[q]["ci"], g) for q, k, g in seq]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, lw=2, alpha=.5, zorder=2)
    for x, y, ci, g in pts:
        ax.plot([x, x], ci, color=c, lw=2, alpha=.45, zorder=3); ax.scatter(x, y, s=150, marker=mk, color=c, edgecolor="white", linewidth=1.3, zorder=4)
        ax.annotate(g, (x, y), xytext=(10, 0), textcoords="offset points", fontsize=11, color=INK2, va="center")
    if lab: ax.annotate(nm, pts[0][:2], xytext=(13, 0) if lab == "r" else (0, 34), textcoords="offset points", fontsize=12, color=INK, va="center" if lab == "r" else "bottom", ha="left" if lab == "r" else "center")
    else: ax.annotate(nm, pts[0][:2], xytext=(-12, -40), textcoords="offset points", fontsize=12, color=INK, va="top", ha="center")
ax.axhline(0, color=INK2, lw=1.4, ls=":"); ax.text(20.5, 1.2, "0 = small model (1.7B) answers alone", ha="right", fontsize=12, color=INK2)
y = G["native"]["gain"]; ax.plot([1, 1], G["native"]["ci"], color=INK, lw=2, alpha=.45); ax.scatter(1, y, s=330, marker="*", color=INK, edgecolor="white", zorder=5)
ax.annotate("re-prefill on 4B", (1, y), xytext=(13, 0), textcoords="offset points", fontsize=12, va="center")
ax.set_xscale("log"); ax.set_xlim(.6, 21); ax.set_ylim(-64, 34); ax.xaxis.set_major_locator(FixedLocator([.75, 1, 2, 3, 5, 10, 20])); ax.xaxis.set_minor_locator(NullLocator())
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×")); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0")); ax.grid(color=GRID, lw=.8); ax.tick_params(length=0, colors=INK2)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color("#b9b8b3")
ax.set_xlabel("TTFT speed-up over re-prefill (8K history)"); ax.set_ylabel("Gain over the small model alone (points of retention)"); ax.set_title("Qwen3-1.7B → Qwen3-4B", loc="left", fontsize=14)
fig.tight_layout()
for e in ("png", "svg"): fig.savefig(f"results/simple_gain_17to4.{e}", dpi=180, bbox_inches="tight")
