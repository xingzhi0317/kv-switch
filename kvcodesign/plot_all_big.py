# Everything measured on the larger pairs: retention vs the target's own re-prefill, 95% paired bootstrap over questions.
#   A/B Qwen3-8B -> 14B (QuALITY, CacheBridge prompt; HotpotQA), C Qwen3-0.6B -> 14B (GSM8K); hollow diamond = the paper's value.
# No TTFT was measured on these pairs (14B split over two GPUs).  usage: PYTHONUTF8=1 python plot_all_big.py
import json, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"], "font.size": 11, "axes.linewidth": .8, "axes.axisbelow": True})
INK, INK2, MUTE, GRID = "#0b0b0b", "#52514e", "#8b8a85", "#ececea"
CLOSED, TRAINED, RECOMP, REF = "#2a78d6", "#eb6834", "#1baf7a", "#8b8a85"
random.seed(0)
L = lambda f: json.load(open("results/" + f, encoding="utf8"))["samples"]
sc = lambda f, m: [x["m"][m]["score"] for x in L(f)]
def ret(s, n):
    n = n[:len(s)]; k = len(s); o = []
    for _ in range(2000):
        ix = [random.randrange(k) for _ in range(k)]; o.append(sum(s[i] for i in ix) / max(1e-9, sum(n[i] for i in ix)))
    o.sort(); return 100 * sum(s) / sum(n), 100 * o[50], 100 * o[1949], k
Q = "8to14/quality_quality_cb_"; N = sc(Q + "init.json", "native"); H = "8to14/quality_hotpotqa_e_"; NH = sc(H + "a.json", "native")
R = "06to14/quality_gsm8k_rr.json"; NG = sc(R, "native")
A = [("No history", Q + "init.json", "none", REF, None), ("CFM, 1 source layer", Q + "cfmk_a.json", "kb1", CLOSED, None), ("CFM, 2 layers", Q + "cfmk_a.json", "kb2", CLOSED, None),
     ("CFM, 4 layers", Q + "cfmk_a.json", "kb4", CLOSED, None), ("CFM, 8 layers", Q + "cfmk_b.json", "kb8", CLOSED, None), ("CFM / KVBridge, 12 layers", Q + "cfm.json", "kvbridge", CLOSED, 44),
     ("Low-rank K768/V1536 (ours)", Q + "lowrank.json", "rrr_k768_v1536_bf16", CLOSED, None), ("Low-rank 2048 (ours)", Q + "lowrank.json", "rrr_k2048_v2048", CLOSED, None),
     ("CacheBridge, uncalibrated", Q + "init.json", "cb_official", CLOSED, 62), ("CacheBridge, calibrated 1%", Q + "f1.json", "cb_official", TRAINED, 97), ("CacheBridge, unconstrained", Q + "nocon.json", "cb_official", TRAINED, None),
     ("Low-rank 2048 + calibration 1% (ours)", Q + "lrcq.json", "lrc_k2048_v2048_f1", TRAINED, None), ("All-head 8 layers + calibration 1%", Q + "kb8cb.json", "kb8cb", TRAINED, None), ("All-head 8 layers, unconstrained", Q + "kb8cbn.json", "kb8cbn", TRAINED, None)]
B = [("No history", H + "a.json", "none", REF, None), ("CFM, 8 layers", H + "b_kb8.json", "kb8", CLOSED, None), ("CFM / KVBridge, 12 layers", H + "b_kvbridge.json", "kvbridge", CLOSED, None),
     ("Low-rank 2048 (ours)", H + "a.json", "rrr_k2048_v2048", CLOSED, None), ("CacheBridge, calibrated 1%", H + "a.json", "cb_official", TRAINED, None),
     ("All-head 8 layers + calibration 1%", H + "b_kb8cb.json", "kb8cb", TRAINED, None), ("Low-rank 2048 + calibration 1% (ours)", H + "lrch.json", "lrc_k2048_v2048_f1", TRAINED, None)]
C = [("0.6B alone, no switch", R, "src_native", REF, 100 * 56.1 / 95.1), ("CFM, translation only", R, "kvbridge", CLOSED, 100 * 71.6 / 95.1),
     ("RaReCache, 30% recomputed", R, "rare30", RECOMP, 100 * 90.2 / 95.1), ("RaReCache, 50% recomputed", R, "rare50", RECOMP, 100 * 95.3 / 95.1)]
fig = plt.figure(figsize=(13.2, 8.4)); g = fig.add_gridspec(2, 2, height_ratios=[7, 4], width_ratios=[1, 1])
axs = [fig.add_subplot(g[:, 0]), fig.add_subplot(g[0, 1]), fig.add_subplot(g[1, 1])]
for ax, (rows, nat, ttl) in zip(axs, ((A, N, "Qwen3-8B → 14B, QuALITY"), (B, NH, "Qwen3-8B → 14B, HotpotQA"), (C, NG, "Qwen3-0.6B → 14B, GSM8K"))):
    for i, (nm, f, m, c, paper) in enumerate(rows):
        y = len(rows) - 1 - i; v, lo, hi, k = ret(sc(f, m), nat)
        ax.plot([lo, hi], [y, y], color=c, lw=2, alpha=.45, zorder=2)
        ax.scatter(v, y, s=75, color=c, edgecolor="white", linewidth=1.1, zorder=4)
        if paper is not None:
            ax.scatter(paper, y, s=70, marker="D", facecolor="white", edgecolor=INK, linewidth=1.4, zorder=3)
        ax.text(1.015, y, f"{v:.0f}%  n={k}", transform=ax.get_yaxis_transform(), va="center", fontsize=9.5, color=INK2)
        print(ttl, "|", nm, f"{v:.1f} [{lo:.0f},{hi:.0f}] n={k}", "paper", paper and round(paper))
    ax.axvline(100, color=INK, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1]); ax.set_xlim(20, 106); ax.set_ylim(-.7, len(rows) - .3)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%"); ax.grid(axis="x", color=GRID, lw=.8); ax.tick_params(colors=INK2, length=0)
    ax.set_title(f"{ttl}   (re-prefill = {sum(nat) / len(nat):.2f})", fontsize=12.5, color=INK, loc="left")
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#b9b8b3")
    ax.set_xlabel("retention vs. the 14B's own re-prefill", color=INK)
h = [Line2D([], [], marker="o", ls="", ms=9, mfc=c, mec="white", label=l) for c, l in ((CLOSED, "closed form, no calibration"), (TRAINED, "attention-calibrated"), (RECOMP, "translation + recompute"), (REF, "reference"))]
h.append(Line2D([], [], marker="D", ls="", ms=8, mfc="white", mec=INK, label="paper's reported value"))
fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=10.5); fig.tight_layout(rect=(0, .05, 1, 1), w_pad=3.5, h_pad=2)
for e in ("png", "svg"): fig.savefig(f"results/all_big_pairs.{e}", dpi=180, bbox_inches="tight")
