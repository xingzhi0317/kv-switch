# All baselines measured so far, accuracy against time-to-first-token speed-up, one column per switch type.
#   rows: reading (QuALITY) / long context (HotpotQA, up to 16K) / exact information (8K passkey, question inside the history);
#   y = retention = method score / re-prefill score, averaged over the group's tasks; x = speed-up over re-prefill, 8K history.
# Numbers: results/[DIR/]quality_summary.json, results/ttft_{DIR}.json (no new runs).  usage: python plot_baselines_v6.py
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"], "mathtext.fontset": "stix",
                     "font.size": 11, "axes.linewidth": 0.8, "axes.axisbelow": True})
INK, INK2, MUTE, GRID = "#0b0b0b", "#52514e", "#8b8a85", "#ececea"
CLOSED, TRAINED, RECOMP, TEXT = "#2a78d6", "#eb6834", "#1baf7a", "#a03c78"
ROWS = [(["quality"], "Reading (QuALITY)", (50, 108)), (["hotpotqa_e"], "Long context (HotpotQA)", (-8, 112)), (["passkey_8k"], "Exact (8K passkey)", (-10, 128))]
COLS = [("17to4", "Small → large", "Qwen3-1.7B → Qwen3-4B"), ("4to17", "Large → small", "Qwen3-4B → Qwen3-1.7B"),
        ("b17to17", "Same base model", "Qwen3-1.7B-Base → Qwen3-1.7B")]
# name, colour, marker, [(quality key, timing key, tag)]
GROUPS = [
    ("KVBridge (CFM)", CLOSED, "o", [("kvbridge", "kvbridge", "")]),
    ("CacheBridge (our repro.)", CLOSED, "s", [("cb_ec_official", "cb_official", "")]),
    ("KL-trained, dense (KV-Lingo-style, our reimpl.)", TRAINED, "D", [("kb3cklbest", "kb3kl", "")]),
    ("KL-trained, low-rank (ours)", TRAINED, "h", [("KLLR", "lrf_bf16", "")]),
    ("RaReCache (% recomputed)", RECOMP, "^", [("rare10", "rare10", "10%"), ("rare30", "rare30", "30%"), ("rare50", "rare50", "50%")]),
    ("DroidSpeak (layers recomputed)", RECOMP, "v", [("droid0_0", "droid0_0", "0"), ("droid4_12", "droid4_12", "8"),
                                                     ("droid4_16", "droid4_16", "12"), ("droid4_20", "droid4_20", "16")]),
    ("LLMLingua-2 (% kept)", TEXT, "P", [("lingua25", "lingua25", "25%"), ("lingua50", "lingua50", "50%")]),
    ("Text summary (10%), off-scale", TEXT, "X", [("sum10", "sum10", "")]),
]


NAMES = {"KVBridge (CFM)": "KVBridge", "CacheBridge (our repro.)": "CacheBridge",
         "KL-trained, dense (KV-Lingo-style, our reimpl.)": "KL dense", "KL-trained, low-rank (ours)": "KL low-rank"}


def ratio(rows, m):   # a method run on fewer questions is compared with re-prefill on the same questions
    if rows[m]["n"] >= rows["native"]["n"]:
        return rows[m]["mean"] / rows["native"]["mean"]
    v = rows[m].get("vs_native")
    return (v["ref_mean"] + v["diff"]) / v["ref_mean"] if v and v.get("ref_mean") else None


def retention(S, group, m):
    r = [ratio(S[t], m) if m in S.get(t, {}) else None for t in group]
    return None if None in r else 100 * sum(r) / len(r)


XMIN, XMAX = 0.36, 42
fig, axes = plt.subplots(3, 3, figsize=(13.2, 12.2), sharey="row")
out, jobs = {}, []
for c, (d, head, pair) in enumerate(COLS):
    S = json.load(open("results/" + ("" if d == "17to4" else d + "/") + "quality_summary.json"))
    tt = json.load(open(f"results/ttft_{d}.json"))["lens"]["8192"]
    base = tt["native"]["p50"]
    for r, (grp, name, ylim) in enumerate(ROWS):
        ax = axes[r, c]
        pts, tags = [(1.0, 100.0)], []
        for lab, col, mk, seq in GROUPS:
            seq = [({'17to4': 'kl_k768_v1536_best17t4_bf16', '4to17': 'kl_k768_v1536_best4t17_bf16'}.get(d) if q == 'KLLR' else q, k, g) for q, k, g in seq]
            P = [(base / tt[k]["p50"], retention(S, grp, q), tag) for q, k, tag in seq
                 if k in tt and retention(S, grp, q) is not None]
            if not P:
                continue
            out[(d, name, lab)] = [(round(x, 2), round(y, 1), t) for x, y, t in P]
            pts += [(x, y) for x, y, _ in P]
            tags += [p + (("low",) if lab.startswith("RaRe") else ()) for p in P if p[2]]
            if lab in NAMES:
                tags.append((P[0][0], P[0][1], NAMES[lab]))
            ax.plot([p[0] for p in P], [p[1] for p in P], color=col, lw=1.6, alpha=.5, zorder=2)
            if lab.startswith("Text summary"):   # far below the axis range: pinned to the left edge, value written out
                x, y, _ = P[0]
                ax.scatter(XMIN * 1.07, y, s=95, marker=mk, facecolor=col, edgecolor="white", linewidth=1.1, zorder=4, clip_on=False)
                ax.annotate(f"← {x:.2f}×", (XMIN * 1.07, y), xytext=(-4, -9), textcoords="offset points", va="top", ha="left", fontsize=9.5, color=INK2)
                pts.remove((x, y))
                continue
            for x, y, _ in P:
                ax.scatter(x, y, s=95, marker=mk, facecolor=col, edgecolor="white", linewidth=1.1, zorder=4)
        x0, y0 = base / tt["none"]["p50"], retention(S, grp, "none")
        if y0 is not None:
            out[(d, name, "No history")] = (round(x0, 1), round(y0, 1))
            pts.append((x0, y0))
            ax.scatter(x0, y0, s=70, marker="o", facecolor="white", edgecolor=MUTE, linewidth=1.6, zorder=4)
        tags += [(1, 100, "re-prefill")] + ([(x0, y0, "no history")] if y0 is not None else [])
        ax.scatter(1, 100, s=230, marker="*", facecolor=INK, edgecolor="white", linewidth=1, zorder=5)
        R_ = [p for p in pts if p != (x0, y0)]
        F = sorted(p for p in R_ if not any(q != p and q[0] >= p[0] and q[1] >= p[1] for q in R_))
        out[(d, name, "front")] = [(round(a, 2), round(b, 1)) for a, b in F]
        ax.step([p[0] for p in F], [p[1] for p in F], where="pre", color=MUTE, lw=1.1, ls=(0, (4, 3)), zorder=1)
        if d == "17to4":   # large to small: staying on the 4B is better than re-prefilling the 1.7B (see the caption), not drawn
            st = retention(S, grp, "src_native")
            ax.axhline(st, color=INK2, lw=1, ls=":", zorder=1)
            ax.text(.985, .965, f"dotted: stay on 1.7B, no switch ({st:.0f}%)", transform=ax.transAxes, fontsize=10, color=INK2, va="top", ha="right")
            out[(d, name, "stay")] = round(st, 1)
        ax.set_xscale("log")
        ax.set_xlim(XMIN, XMAX)
        ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 20]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
        ax.set_ylim(*ylim)
        ax.set_yticks([60, 80, 100] if r == 0 else [0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.grid(color=GRID, lw=.8)
        ax.tick_params(colors=INK2, length=3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#b9b8b3")
        if r == 0:
            ax.text(.5, 1.10, head, transform=ax.transAxes, ha="center", fontsize=14.5, fontweight="bold", color=INK)
            ax.text(.5, 1.03, pair, transform=ax.transAxes, ha="center", fontsize=11, color=INK2)
        if r == 2:
            ax.set_xlabel("TTFT speed-up over re-prefill (log scale)", color=INK)
        if c == 0:
            ax.set_ylabel(f"{name}\nretention vs. re-prefill", color=INK, fontsize=12.5)
        jobs.append((ax, pts, tags))

h = [Line2D([], [], marker="*", ls="", ms=14, mfc=INK, mec="white", label="Full re-prefill")]
h += [Line2D([], [], marker=mk, ls="", ms=9.5, mfc=col, mec="white", label=lab) for lab, col, mk, _ in GROUPS]
h += [Line2D([], [], marker="o", ls="", ms=8, mfc="white", mec=MUTE, mew=1.6, label="No history"),
      Line2D([], [], color=MUTE, lw=1.1, ls=(0, (4, 3)), label="Pareto frontier")]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=10.5, columnspacing=1.6, handletextpad=.5, bbox_to_anchor=(.5, .0))
fig.tight_layout(rect=(0, .075, 1, 1), w_pad=1.6, h_pad=1.6)
# tags of swept parameters: first free side, checked against markers and earlier tags
fig.canvas.draw()
rend = fig.canvas.get_renderer()
for ax, pts, tags in jobs:
    marks, placed = [ax.transData.transform(p) for p in pts], []
    for x, y, t, *side in sorted(tags, key=lambda q: -len(q[2])):
        for dx, dy in ((((-8, -6), (-9, 0), (0, -8)) if side else ()) + ((9, 0), (0, 8), (0, -8), (-9, 0), (8, 7), (8, -7), (-8, 7), (-8, -7), (0, 15), (0, -15), (18, 0), (-18, 0), (0, 8))):
            a = ax.annotate(t, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=9, color=INK2, zorder=6,
                            ha="center" if dx == 0 else "left" if dx > 0 else "right",
                            va="center" if dy == 0 else "bottom" if dy > 0 else "top")
            bb = a.get_window_extent(rend)
            if not any(bb.overlaps(b) for b in placed) and not any(bb.padded(4).contains(*m) for m in marks):
                break
            a.remove()
        else:
            a = ax.annotate(t, (x, y), xytext=(0, 8), textcoords="offset points", fontsize=9, color=INK2, ha="center", zorder=6)
            bb = a.get_window_extent(rend)
        placed.append(bb)
for e in ("png", "svg"):  # pdf needs fontTools, blocked by policy on this machine
    fig.savefig(f"results/baselines_v6.{e}", dpi=200, bbox_inches="tight")
for k, v in out.items():
    print(k, v)
