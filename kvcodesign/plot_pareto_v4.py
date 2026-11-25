# Cleaner accuracy x speed plot (Qwen3-1.7B -> 4B): three panels (reading comprehension, exact information, continuation
# KL) against the time-to-first-token speed-up over full re-prefill at an 8K history.  Same numbers as plot_pareto.py
# (results/quality_summary.json, results/ttft_17to4.json); fewer methods, coloured by family, Chinese labels.
#   left out (named in the caption): the 10% summary (0.05x, it would squash the speed axis), RaRe 50%, LLMLingua 25%,
#   the 3-layer linear initialiser.
# v3: adds the KL-trained translator retrained with the published recipe (train_kl2.py, best LR 3e-6), the old
#     under-trained one drawn light with an arrow (same architecture, same online cost).
# v4: adds our fused low-rank map (K768/V1536) after the same KL training recipe (train_kl_lr.py, best of 1e-4/3e-4
#     relative LR), drawn with an arrow from the closed-form fused map (same online cost); both directions.
# usage: DIR=17to4|4to17 python plot_pareto_v4.py  -> results/pareto_{DIR}_v4.png
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False,
                     "font.size": 10})
import os
DIR = os.environ.get("DIR", "17to4")
D4 = {"17to4": "17t4", "4to17": "4t17"}[DIR]
KLLR = f"kl_k768_v1536_best{D4}_bf16"
summ = json.load(open("results/quality_summary.json" if DIR == "17to4" else "results/4to17/quality_summary.json"))
tt = json.load(open(f"results/ttft_{DIR}.json"))["lens"]["8192"]
READ, EXACT = ["quality", "hotpotqa_e"], ["squad", "gsm8k_h", "kvret", "passkey_2k", "passkey_8k"]
TT = {"rrr_k2048_v2048": "rrr_k2048_v2048_bf16", "kb12cb": "kvbridge", "kb12cbe1": "kvbridge", "cb_ec_official": "cb_official",
      "lrc_k2048_v2048_b03": "rrr_k2048_v2048_bf16", "lrc_k768_v1536_f1_bf16": "lrf_bf16", "kb3cklbest": "kb3kl",
      KLLR: "lrf_bf16", "cb_full_official": "cb_official"}   # calibration only changes weights: same online cost
FAM = {"ours": ("#2a78d6", "我们的低秩"), "cb": ("#eb6834", "CacheBridge（复现 / 变体）"),
       "tr": ("#52514e", "其他 KV 翻译"), "rc": ("#1baf7a", "部分重算 / 压缩"), "end": ("#9b9a95", "两端参照")}
METHODS = [  # key, label, family
    ("native", "重读（基准）", "end"), ("none", "不看历史", "end"),
    ("lrf_bf16", "低秩（融合 kernel）", "ours"), ("rrr_k2048_v2048", "低秩（秩 2048）", "ours"),
    ("lrc_k768_v1536_f1_bf16", "低秩（融合）+ 校准", "ours"), ("lrc_k2048_v2048_b03", "低秩 2048 + 校准", "ours"),
    (KLLR, "低秩（融合）+ KL 训练", "ours"),
    ("cb_ec_official" if DIR == "17to4" else "cb_full_official", "CacheBridge 复现", "cb"), ("kb12cbe1", "CacheBridge 变体（kvbridge + 校准）", "cb"),
    ("kvbridge", "kvbridge（CFM）", "tr"), ("kb3kl", "KL 翻译器（旧版）", "tr"), ("kb3cklbest", "KL 训练翻译器（按论文重训）", "tr"),
    ("rare10", "RaRe 10%", "rc"), ("rare30", "RaRe 30%", "rc"), ("lingua50", "LLMLingua 50%", "rc")]
# calibration drawn as an arrow from the same map before calibration (same speed): light, unlabelled start points
ARROWS = [("lrf_bf16", "lrc_k768_v1536_f1_bf16"), ("rrr_k2048_v2048", "lrc_k2048_v2048_b03"), ("kvbridge", "kb12cbe1"), ("kb3kl", "kb3cklbest"), ("lrf_bf16", KLLR)]
BEFORE = {"lrf_bf16", "rrr_k2048_v2048", "kb3kl"}


def ratio(rows, m):   # matched-question retention, as plot_pareto.py
    if m not in rows:
        return None
    if rows[m]["n"] >= rows["native"]["n"]:
        return rows[m]["mean"] / rows["native"]["mean"]
    v = rows[m].get("vs_native")
    return (v["ref_mean"] + v["diff"]) / v["ref_mean"] if v and v.get("ref_mean") else None


def retention(group, m):
    r = [ratio(summ[t], m) for t in group if m in summ.get(t, {})]
    return 100 * sum(r) / len(r) if len(r) == len(group) and None not in r else None


def kl(m):
    import math
    t = summ.get("klchat_2k", {})
    return 100.0 if m == "native" else (100 * math.exp(-t[m]["mean"]) if m in t else None)


base = tt["native"]["p50"]
speed = {m: base / tt[TT.get(m, m)]["p50"] for m, _, _ in METHODS if TT.get(m, m) in tt}
panels = [("阅读理解", "QuALITY + HotpotQA，保留率（%）", lambda m: retention(READ, m)),
          ("精确信息", "SQuAD、GSM8K、键值检索、2K/8K 密码，保留率（%）", lambda m: retention(EXACT, m)),
          ("续写一致性", "多轮对话续写，exp(−KL) × 100", kl)]

TITLE = {"17to4": "Qwen3-1.7B → 4B：融合低秩加 KL 训练后 8.8 倍加速、保留 72–78%；稠密 KL 翻译器 6.0 倍、约 80%。两者都在前沿上",
         "4to17": "Qwen3-4B → 1.7B：融合低秩加 KL 训练后 4.5 倍加速，阅读理解 89% 超过稠密 KL（83%），精确信息和续写略低"}
_COMMON = ("右上更好。单张 RTX 4090，HF transformers。阅读理解和精确信息每任务 200 题（CacheBridge 复现为 50 题），续写 200 段对话。" + chr(10) +
           "两种 KL 训练用同一配方：Nemotron 对话、每步 8 段、学整段回答、2000 步，学习率按 64 段验证对话的 KL 选。"
           "融合低秩 K768/V1536 的初值是 FineWeb 闭式解，稠密 KL 翻译器每层 3 个源层、对话数据闭式初值。箭头：同一结构加校准或 KL 训练（速度不变）。")
NOTE = {"17to4": _COMMON + chr(10) + "稠密 KL 学习率 3e-6，融合低秩相对学习率 3e-4。未画：10% 摘要（0.05×）、RaRe 50%、LLMLingua 25%。",
        "4to17": _COMMON + chr(10) + "稠密 KL 学习率 1e-5 落在扫描上沿，可能没调到最优（KV-Lingo 在这个方向选 3e-4）。融合低秩相对学习率 3e-4。未画：10% 摘要、RaRe 50%、LLMLingua 25%。"}
fig, axes = plt.subplots(1, 3, figsize=(17, 6.2), sharey=True)
fig.patch.set_facecolor("#fcfcfb")
OFFS = {  # (panel, method) -> label offset in points, for the crowded clusters
    (0, "rare10"): (-16, -4), (0, "kvbridge"): (-16, -14), (0, "rare30"): (-18, -14), (0, "lingua50"): (-30, 14),
    (0, "kb12cbe1"): (18, -16), (0, "lrc_k2048_v2048_b03"): (16, -12), 
    (1, "cb_ec_official"): (0, 18), (1, "kb3kl"): (-14, 14), (1, "kvbridge"): (-14, -18), (1, "rare10"): (18, -6),
    (1, "kb12cbe1"): (30, -26), (1, "lrc_k2048_v2048_b03"): (18, -12), 
    (2, "kvbridge"): (-16, -16), (2, "kb12cbe1"): (16, 10), (2, "kb3kl"): (16, 12), (2, "lrc_k2048_v2048_b03"): (-16, 12),
    (2, "rare10"): (16, -10), (2, "lingua50"): (-16, -14), (2, "rare30"): (0, -18),
    (0, "lrc_k768_v1536_f1_bf16"): (-12, 14), (1, "lrc_k768_v1536_f1_bf16"): (-10, 14), (2, "lrc_k768_v1536_f1_bf16"): (-12, 14),
    (2, "cb_ec_official"): (0, -16), (2, "lrc_k2048_v2048_b03"): (-16, 14),
    (0, KLLR): (14, 14), (1, KLLR): (-10, 16), (2, KLLR): (14, 14),
    (0, "kb3cklbest"): (14, -16), (1, "kb3cklbest"): (14, -16), (2, "kb3cklbest"): (14, -16)}
if DIR != "17to4":   # hand offsets were placed for 1.7B -> 4B; elsewhere use the greedy placement
    OFFS = {}
for pi, (ax, (title, xlabel, acc)) in enumerate(zip(axes, panels)):
    pts = {m: (acc(m), speed[m]) for m, _, _ in METHODS if m in speed and acc(m) is not None}
    front = {m for m, (x, y) in pts.items() if not any(x2 >= x and y2 >= y and (x2, y2) != (x, y) for x2, y2 in pts.values())}
    fx = sorted(pts[m] for m in front)
    ax.step([p[0] for p in fx], [p[1] for p in fx], where="post", color="#b9b8b3", lw=1.2, ls="--", zorder=1)
    ax.set_facecolor("#fcfcfb")
    for m, lab, fam in METHODS:
        if m not in pts:
            continue
        x, y = pts[m]
        col = FAM[fam][0]
        hollow = fam == "end"
        if m in BEFORE:   # the low-rank maps before calibration: light, no label
            ax.scatter(x, y, s=45, facecolor=col, edgecolor="none", alpha=0.35, zorder=3)
            continue
        ax.scatter(x, y, s=95 if m in front else 60, facecolor="#fcfcfb" if hollow else col, edgecolor=col,
                   linewidth=2 if hollow else 1.8, zorder=3)
        if m in front and not hollow:
            ax.scatter(x, y, s=230, facecolor="none", edgecolor=col, linewidth=1, alpha=0.45, zorder=2)
    for b0, b1 in ARROWS:
        if b0 in pts and b1 in pts:
            col = FAM[next(f for k, _, f in METHODS if k == b1)][0]
            ax.annotate("", xy=pts[b1], xytext=pts[b0], arrowprops=dict(arrowstyle="-|>", color=col, lw=1.3, alpha=0.8,
                        shrinkA=4, shrinkB=6, mutation_scale=11), zorder=2)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 20]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_ylim(0.55, 25)
    xs = [p[0] for p in pts.values()]
    ax.set_xlim(min(xs) - 8, 112)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b9b8b3")
    ax.tick_params(colors="#52514e")
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color="#0b0b0b")
    ax.set_xlabel(xlabel, color="#52514e")
    # labels: greedy placement, no overlap with earlier labels or with any marker
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    marks = [ax.transData.transform(p) for p in pts.values()]
    placed = []
    order = sorted(pts, key=lambda m: -pts[m][1])
    for m in order:
        if m in BEFORE:
            continue
        lab = next(l for k, l, _ in METHODS if k == m)
        x, y = pts[m]
        if (pi, m) in OFFS:   # crowded points: hand-placed
            dx, dy = OFFS[(pi, m)]
            ha = "left" if dx > 0 else "right" if dx < 0 else "center"
            va = "bottom" if dy > 0 else "top" if dy < 0 else "center"
            t = ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=9.5,
                            color="#0b0b0b" if m in front else "#52514e", zorder=4,
                            arrowprops=dict(arrowstyle="-", color="#b9b8b3", lw=0.7, shrinkA=1, shrinkB=5))
            placed.append(t.get_window_extent(rend))
            continue
        cands = [(r * cx, r * cy, "left" if cx > 0 else "right" if cx < 0 else "center", "bottom" if cy > 0 else "top" if cy < 0 else "center")
                 for r in (16, 28, 42) for cx, cy in ((1, 0.6), (1, -0.6), (-1, 0.6), (-1, -0.6), (0, 1), (0, -1), (1, 0), (-1, 0))]
        for dx, dy, ha, va in cands:
            t = ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=9.5,
                            color="#0b0b0b" if m in front else "#52514e", zorder=4,
                            arrowprops=dict(arrowstyle="-", color="#b9b8b3", lw=0.7, shrinkA=1, shrinkB=5))
            bb = t.get_window_extent(rend).expanded(1.04, 1.15)
            hit = any(bb.overlaps(b) for b in placed) or any(bb.contains(px, py) for px, py in marks)
            inside = ax.bbox.contains(bb.x0, bb.y0) and ax.bbox.contains(bb.x1, bb.y1)
            if not hit and inside:
                placed.append(bb)
                break
            t.remove()
        else:
            t = ax.annotate(lab, (x, y), xytext=(9, 6), textcoords="offset points", fontsize=9.5, color="#52514e", zorder=4)
            placed.append(t.get_window_extent(rend))
axes[0].set_ylabel("首字时延加速（相对完整重读，8K 历史，对数轴）", color="#52514e")
handles = [plt.Line2D([], [], marker="o", ls="", markersize=8, markerfacecolor=("#fcfcfb" if k == "end" else c),
                      markeredgecolor=c, markeredgewidth=1.8, label=l) for k, (c, l) in FAM.items()]
handles.append(plt.Line2D([], [], color="#b9b8b3", ls="--", lw=1.2, label="帕累托前沿（外圈 = 在前沿上）"))
handles.append(plt.Line2D([], [], color="#2a78d6", marker=">", markersize=6, lw=1.3, alpha=0.8, label="箭头：同一结构，加校准或按论文重训（速度不变）"))
fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.005))
fig.suptitle(TITLE[DIR], x=0.01, y=0.995, ha="left", fontsize=15, fontweight="bold", color="#0b0b0b")
fig.text(0.01, 0.935, NOTE[DIR], va="top", ha="left", fontsize=9.5, color="#52514e")
plt.tight_layout(rect=(0, 0.06, 1, 0.84))
plt.savefig(f"results/pareto_{DIR}_v4.png", dpi=150, facecolor=fig.get_facecolor())
print(f"saved results/pareto_{DIR}_v4.png")
for (title, _, acc) in panels:
    print(title, {m: round(acc(m), 1) for m, _, _ in METHODS if acc(m) is not None})
print("speed", {m: round(v, 2) for m, v in speed.items()})
