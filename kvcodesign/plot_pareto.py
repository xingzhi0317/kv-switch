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
LABEL = {"native": "full re-prefill", "none": "no history", "kvbridge": "kvbridge (NVIDIA)", "cb_official": "CacheBridge",
         "lrf_bf16": "ours: low-rank K768+V1536 (fused)", "rrr_k2048_v2048": "ours: low-rank 2048",
         "rrr_k2048_v2048_bf16": "ours: low-rank 2048"}
TT_ALIAS = {"rrr_k2048_v2048": "rrr_k2048_v2048_bf16"}               # quality run fp32, timing run bf16 (same map)


KEEP = {"native", "none", "kvbridge", "cb_official", "lrf_bf16", "rrr_k2048_v2048", "rare10", "rare30", "rare50",
        "lingua25", "lingua50", "sum10", "kb3", "kb3kl"}   # the methods of PLAN_v6 sec. 3 (older variants are left out)
LABEL.update({"rare10": "RaReCache 10%", "rare30": "RaReCache 30%", "rare50": "RaReCache 50%", "lingua25": "LLMLingua-2 25%",
              "lingua50": "LLMLingua-2 50%", "sum10": "old model summary", "kb3": "per-layer linear (3 src layers)",
              "kb3kl": "KL-trained translator (KV-Lingo/Hybrid-KV recipe)"})


def retention(group):
    out = {}
    for m in {k for t in group for k in summ.get(t, {})}:
        if ALIAS.get(m, m) not in KEEP:
            continue
        key = ALIAS.get(m, m)
        if key is None or m == "src_native":
            continue
        r = [summ[t][m]["mean"] / summ[t]["native"]["mean"] for t in group if m in summ.get(t, {}) and "native" in summ[t]
             and summ[t][m]["n"] >= summ[t]["native"]["n"]]          # only tasks the method has finished
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
axes = [("reading tasks: retention (%)", retention(READ)), ("exact-information tasks: retention (%)", retention(EXACT)),
        ("continuation KL, 2K history: exp(-KL) x 100", kl_axis())]
fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
for a, (xl, acc) in zip(ax, axes):
    pts = {m: (acc[m], sp[TT_ALIAS.get(m, m)]) for m in acc if TT_ALIAS.get(m, m) in sp}
    f = front(pts)
    for m, (x, y) in pts.items():
        a.scatter(x, y, s=60, c="tab:red" if m in f else "tab:blue", zorder=3)
        a.annotate(LABEL.get(m, m), (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
    fx = sorted((pts[m] for m in f))
    if len(fx) > 1:
        a.plot([p[0] for p in fx], [p[1] for p in fx], "r--", lw=0.8, zorder=2)
    a.set(xlabel=xl, ylabel=f"TTFT speed-up over full re-prefill ({int(L) // 1024}K history)", yscale="log")
    a.grid(alpha=0.3)
names = {"17to4": "Qwen3-1.7B -> Qwen3-4B (weak to strong)", "4to17": "Qwen3-4B -> Qwen3-1.7B (strong to weak)"}
fig.suptitle(f"{names[DIR]}; red = Pareto front; single RTX 4090, HF transformers")
plt.tight_layout()
out = f"results/pareto_{DIR}_{L}.png"
plt.savefig(out, dpi=110)
print("saved", out)
for xl, acc in axes:
    print(xl, {m: round(v, 1) for m, v in sorted(acc.items(), key=lambda kv: -kv[1])})
print("speed-up", {m: round(v, 2) for m, v in sp.items()})
