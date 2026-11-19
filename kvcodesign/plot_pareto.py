# Accuracy x performance plot (PLAN_v6 sec. 1): one point per method, per switching direction.  Run locally on the
# synced results:   python plot_pareto.py [17to4|4to17] [ttft length, default 8192]
# Accuracy (x): retention = method score / re-prefill score per task, averaged over a task group (reading tasks;
#   exact-information tasks), or continuation KL (klchat_2k, lower is better; plotted as exp(-KL) so right = better).
# Performance (y): speed-up of the time to first token over full re-prefill at the given history length (higher =
#   better).  Pareto front: points not dominated in both axes.
# Inputs: results/[DIR/]quality_summary.json (analyze_quality.py), results/ttft_{DIR}.json (eval_quality.py MODE=ttft)
import sys, os, json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = sys.argv[1] if len(sys.argv) > 1 else "17to4"
L = sys.argv[2] if len(sys.argv) > 2 else "8192"
R = "results/" if DIR == "17to4" else f"results/{DIR}/"
summ = json.load(open(R + "quality_summary.json"))
tt = json.load(open(f"results/ttft_{DIR}.json"))["lens"].get(L, {})
READ, EXACT = ["quality", "hotpotqa_e"], ["squad", "gsm8k_h", "kvret", "passkey_2k", "passkey_8k"]
ALIAS = {"cb_full_official": "cb_official", "native_prev": None}       # summary key -> ttft key
LABEL = {"native": "re-prefill", "none": "no history", "kvbridge": "kvbridge", "cb_official": "CacheBridge",
         "lrf_bf16": "ours: low-rank (fused)", "rrr_k2048_v2048": "ours: low-rank 2048",
         "rrr_k2048_v2048_bf16": "ours: low-rank 2048"}
TT_ALIAS = {"rrr_k2048_v2048": "rrr_k2048_v2048_bf16"}               # quality run fp32, timing run bf16 (same map)


KEEP = {"native", "none", "kvbridge", "cb_official", "lrf_bf16", "rrr_k2048_v2048", "rare10", "rare30", "rare50",
        "lingua25", "lingua50", "sum10", "kb3", "kb3kl"}   # the methods of PLAN_v6 sec. 3 (older variants are left out)
LABEL.update({"rare10": "RaRe 10%", "rare30": "RaRe 30%", "rare50": "RaRe 50%", "lingua25": "Lingua 25%",
              "lingua50": "Lingua 50%", "sum10": "summary", "kb3": "linear 3-layer (init)",
              "kb3kl": "KL-trained (KV-Lingo/Hybrid-KV)",
              "droid0_0": "DroidSpeak: reuse all", "droid4_12": "DroidSpeak: recompute 8 layers",
              "droid4_16": "DroidSpeak: recompute 12 layers", "droid4_20": "DroidSpeak: recompute 16 (chosen)"})
KEEP |= {"droid0_0", "droid4_12", "droid4_16", "droid4_20"}
# naive DroidSpeak across sizes (droid.py): sender KV reused as is (depth-proportional layers), layers 0..b-1 recomputed
DSX = {m for t in summ.values() if isinstance(t, dict) for m in t if m.startswith("dsx_direct_")}
# not plotted: across sizes it only matches re-prefill when every layer is recomputed (REPORT sec. 15.4)
LABEL.update({m: "DroidSpeak (naive): " + ("reuse all" if m == "dsx_direct_0" else f"recompute {m.split('_')[2]} layers") + " [50 q/task]" for m in DSX})


def ratio(rows, m):   # method score / re-prefill score; a method run on fewer questions is compared on the same questions
    if rows[m]["n"] >= rows["native"]["n"]:
        return rows[m]["mean"] / rows["native"]["mean"]
    v = rows[m].get("vs_native")
    return (v["ref_mean"] + v["diff"]) / v["ref_mean"] if v and v.get("ref_mean") else None


def retention(group):
    out = {}
    for m in {k for t in group for k in summ.get(t, {})}:
        if ALIAS.get(m, m) not in KEEP:
            continue
        key = ALIAS.get(m, m)
        if key is None or m == "src_native":
            continue
        r = [ratio(summ[t], m) for t in group if m in summ.get(t, {}) and "native" in summ[t] and ratio(summ[t], m) is not None]
        if len(r) == len([t for t in group if t in summ]):
            out[key] = 100 * sum(r) / len(r)
    return out


def kl_axis():
    t = summ.get("klchat_2k", {})
    return {ALIAS.get(m, m): 100 * math.exp(-v["mean"]) for m, v in t.items() if ALIAS.get(m, m) in KEEP} | ({"native": 100.0} if t else {})


def speedup():
    base = tt.get("native", {}).get("p50")
    return {m: base / v["p50"] for m, v in tt.items() if base and "p50" in v}


def front(pts):   # pts: {m: (x, y)} -> methods on the Pareto front (maximise both)
    return {m for m, (x, y) in pts.items() if not any(x2 >= x and y2 >= y and (x2, y2) != (x, y) for x2, y2 in pts.values())}


sp = speedup()
axes = [("reading comprehension: retention (%)\nmean of QuALITY (200 q, multiple choice) and\nLongBench HotpotQA-E (200 q, F1)", retention(READ)),
        ("exact information: retention (%)\nmean of SQuAD v1.1 (200, F1), GSM8K with question in history (200),\n"
         "key-value retrieval (200), passkey in 2K / 8K text (200 / 100)", retention(EXACT)),
        ("continuation KL: exp(-KL) x 100\nNemotron-SFT-Instruction-Following-Chat-v2, 200 conversations,\n"
         "2K-token history, next 128 tokens teacher-forced", kl_axis())]
fig, ax = plt.subplots(1, 3, figsize=(18, 6.3))
for a, (xl, acc) in zip(ax, axes):
    pts = {m: (acc[m], sp[TT_ALIAS.get(m, m)]) for m in acc if TT_ALIAS.get(m, m) in sp}
    f = front(pts)
    for m, (x, y) in pts.items():
        a.scatter(x, y, s=60, c="tab:red" if m in f else "tab:blue", zorder=3)
    a.set(yscale="log")
    fig.canvas.draw()
    placed = []   # label boxes in display pixels; try a few offsets and take the first that overlaps nothing
    for m, (x, y) in sorted(pts.items(), key=lambda kv: -kv[1][1]):
        t = LABEL.get(m, m); w, h = 6.2 * len(t), 11
        px, py = a.transData.transform((x, y))
        for dx, dy, ha in ((6, 3, "left"), (6, -13, "left"), (-6, 3, "right"), (-6, -13, "right"), (6, 14, "left"), (6, -24, "left")):
            x0 = px + dx if ha == "left" else px + dx - w
            box = (x0, py + dy, x0 + w, py + dy + h)
            if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in placed):
                break
        placed.append(box)
        a.annotate(t, (x, y), textcoords="offset pixels", xytext=(dx, dy), ha=ha, fontsize=8)
    fx = sorted((pts[m] for m in f))
    if len(fx) > 1:
        a.plot([p[0] for p in fx], [p[1] for p in fx], "r--", lw=0.8, zorder=2)
    a.set(xlabel=xl, ylabel=f"TTFT speed-up over full re-prefill ({int(L) // 1024}K history)", yscale="log")
    a.grid(alpha=0.3)
names = {"17to4": "Qwen3-1.7B -> Qwen3-4B (weak to strong)", "4to17": "Qwen3-4B -> Qwen3-1.7B (strong to weak)",
         "b17to17": "Qwen3-1.7B-Base -> Qwen3-1.7B (same base model, DroidSpeak setting)"}
fig.suptitle(f"{names[DIR]}; red = Pareto front; single RTX 4090, HF transformers")
plt.tight_layout()
out = f"results/pareto_{DIR}_{L}.png"
plt.savefig(out, dpi=110)
print("saved", out)
for xl, acc in axes:
    print(xl, {m: round(v, 1) for m, v in sorted(acc.items(), key=lambda kv: -kv[1])})
print("speed-up", {m: round(v, 2) for m, v in sp.items()})
