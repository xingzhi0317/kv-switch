# Aggregate results: retention + bootstrap CI, harmful flips, KL, guard ROC (M3), guarded system, figures.
import os, json, glob, math, collections
import numpy as np
from ks import ROOT

R = f"{ROOT}/results"
FIG = f"{ROOT}/figures"
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(0)
CORRECT = 0.5  # a turn counts as "correct" if score >= 0.5 (F1 tasks) / ==1 (exact tasks)


def load(pair, task):
    out = {}
    for f in sorted(glob.glob(f"{R}/q_{pair}_{task}.json")) + sorted(glob.glob(f"{R}/q_{pair}_{task}_*.json")):
        for s in json.load(open(f))["samples"]:
            d = out.setdefault(s["id"], dict(id=s["id"], N=s["N"], m={}))
            d["m"].update(s["m"])
            if "nohist" in s:
                d["nohist"] = s["nohist"]
    return list(out.values())


def boot_ratio(a, b, n=2000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = rng.integers(0, len(a), (n, len(a)))
    r = a[idx].mean(1) / np.maximum(b[idx].mean(1), 1e-9)
    return float(a.mean() / max(b.mean(), 1e-9)), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def boot_mean(a, n=2000):
    a = np.asarray(a, float)
    idx = rng.integers(0, len(a), (n, len(a)))
    m = a[idx].mean(1)
    return float(a.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def task_table(pair, task, turn=0):
    S = [s for s in load(pair, task) if "native" in s["m"]]
    if not S:
        return None
    methods = sorted({m for s in S for m in s["m"]} - {"native"})
    tab = {}
    if task == "uc":
        for m in methods:
            ss = [s for s in S if m in s["m"]]
            tab[m] = dict(n=len(ss), kl=boot_mean([s["m"][m]["kl"] for s in ss]), top1=boot_mean([s["m"][m]["top1"] for s in ss]))
        return tab
    for m in ["native"] + methods:
        ss = [s for s in S if m in s["m"] and len(s["m"][m]["turns"]) > turn and len(s["m"]["native"]["turns"]) > turn]
        if not ss:
            continue
        a = [s["m"][m]["turns"][turn]["score"] for s in ss]
        b = [s["m"]["native"]["turns"][turn]["score"] for s in ss]
        flips = [float(y >= CORRECT and x < CORRECT) for x, y in zip(a, b)]
        tab[m] = dict(n=len(ss), score=float(np.mean(a)), native=float(np.mean(b)), retention=boot_ratio(a, b),
                      harmful_flip=boot_mean(flips), t_switch_ms=float(np.median([s["m"][m]["t_switch_ms"] for s in ss])),
                      N_median=float(np.median([s["N"] for s in ss])), peak_GB=float(np.median([s["m"][m]["peak_GB"] for s in ss])))
    return tab


TASKS = ["uc", "coqa1", "coqa5", "coqa10", "lb8k", "niah4k", "niah8k", "niah16k", "niah32k", "gsm"]
summary = {}
for pair in ["P1", "P2", "P3", "P4"]:
    summary[pair] = {}
    for t in TASKS:
        tb = task_table(pair, t)
        if tb:
            summary[pair][t] = tb
    for t in ["coqa1", "coqa5", "coqa10"]:  # per-turn retention after the switch (error accumulation)
        for turn in (1, 2):
            tb = task_table(pair, t, turn)
            if tb:
                summary[pair][f"{t}_turn{turn + 1}"] = tb
json.dump(summary, open(f"{R}/summary.json", "w"), indent=1)


# ---------------- M3 guard ----------------
def roc(scores, labels):
    s, y = np.asarray(scores), np.asarray(labels)
    order = np.argsort(-s)
    y = y[order]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    P, Nn = max(y.sum(), 1), max((1 - y).sum(), 1)
    tpr, fpr = np.r_[0, tp / P], np.r_[0, fp / Nn]
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc, s[order]


guard = {}
for pair in ["P1", "P2"]:
    rows = []
    for t in TASKS[1:]:
        for s in load(pair, t):
            if "native" not in s["m"] or "nohist" not in s:
                continue
            nat = s["m"]["native"]["turns"][0]["score"]
            for m, r in s["m"].items():
                if m in ("native", "none"):
                    continue
                y = float(nat >= CORRECT and r["turns"][0]["score"] < CORRECT)
                if "klAB" not in r:
                    continue
                rows.append(dict(task=t, m=m, y=y, s1=r["nll"], s2=r["nll"] - s["nohist"]["nll"], s3=r["ent8"],
                                 s1l=r["nll_last32"], s4=r["klAB"], s5=1 - r["top1AB"], s6=-(r["nll"] - s["nohist"]["nll"]),
                                 ntok=r["ntok"], N=s["N"]))
    if not rows:
        continue
    g = dict(n=len(rows), positives=int(sum(r["y"] for r in rows)))
    for key in ("s1", "s2", "s3", "s1l", "s4", "s5", "s6"):
        fpr, tpr, auc, th = roc([r[key] for r in rows], [r["y"] for r in rows])
        ok = fpr <= 0.10
        g[key] = dict(auc=auc, tpr_at_fpr10=float(tpr[ok].max()), roc=(fpr[::max(1, len(fpr) // 200)].tolist(), tpr[::max(1, len(tpr) // 200)].tolist()))
    g["per_task"] = {}
    for t in sorted({r["task"] for r in rows}):
        rt = [r for r in rows if r["task"] == t]
        if sum(r["y"] for r in rt) < 5 or sum(1 - r["y"] for r in rt) < 5:
            continue
        g["per_task"][t] = {k: dict(auc=roc([r[k] for r in rt], [r["y"] for r in rt])[2],
                                    tpr_at_fpr10=float(roc([r[k] for r in rt], [r["y"] for r in rt])[1][roc([r[k] for r in rt], [r["y"] for r in rt])[0] <= 0.1].max()))
                            for k in ("s1", "s2", "s3", "s4", "s5", "s6")}
        g["per_task"][t]["n"], g["per_task"][t]["pos"] = len(rt), int(sum(r["y"] for r in rt))
    guard[pair] = g
json.dump(guard, open(f"{R}/guard.json", "w"), indent=1)

# ---------------- figures ----------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {"native": "#444444", "none": "#bbbbbb", "B2": "#e69f00", "B1": "#56b4e9", "B3": "#009e73", "ours_r64": "#d55e00",
           "ours_r16": "#cc79a7", "ours_r64_fix2k": "#f0e442", "B1_c3": "#0072b2", "ours_r64_c3": "#8b0000",
           "B4": "#56b4e9", "B4k1": "#e69f00", "B5": "#d55e00", "B5noreplay": "#999999"}
    # multi-turn retention curve
    for pair in summary:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for m in COL:
            for k in ("coqa1", "coqa5", "coqa10"):
                ys = [summary[pair].get(k if i == 0 else f"{k}_turn{i + 1}", {}).get(m, {}).get("retention", [None])[0] for i in range(3)]
                if None not in ys:
                    ax.plot([1, 2, 3], ys, marker="o", color=COL[m], ls={"coqa1": "-", "coqa5": "--", "coqa10": ":"}[k],
                            label=f"{m} k={k[4:]}" if k == "coqa1" else None)
        ax.set_xlabel("turn after switch"); ax.set_ylabel("retention (F1 / native)"); ax.set_title(f"{pair} CoQA multi-turn")
        ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(f"{FIG}/multiturn_{pair}.png", dpi=150); plt.close(fig)
    # guard ROC
    for pair, g in guard.items():
        fig, ax = plt.subplots(figsize=(4, 4))
        for key, lab in [("s1", "nll_new"), ("s2", "nll_new - nll_nohist"), ("s3", "entropy last 8"), ("s4", "KL(A||B) on new input")]:
            ax.plot(*g[key]["roc"], label=f"{lab} AUC={g[key]['auc']:.2f}")
        ax.plot([0, 1], [0, 1], "k:", lw=0.5); ax.axvline(0.1, color="r", lw=0.5)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR (harmful flips caught)"); ax.legend(fontsize=7); ax.set_title(f"{pair} guard ROC")
        fig.tight_layout(); fig.savefig(f"{FIG}/guard_roc_{pair}.png", dpi=150); plt.close(fig)
    # timing: T_switch vs N, peak memory
    for pair in ["P1", "P2"]:
        f = f"{R}/timing_{pair}.json"
        if not os.path.exists(f):
            continue
        T = json.load(open(f))
        Ns = sorted(int(n) for n in T.get("switch", {}))
        if not Ns:
            continue
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
        for name in ["native", "B2", "B1", "B3", "C5", "C5+C6", "C1_aot"]:
            xs = [n for n in Ns if isinstance(T["switch"][str(n)].get(name), dict)]
            axs[0].plot(xs, [T["switch"][str(n)][name]["total"]["p50"] for n in xs], marker="o", label=name)
            axs[1].plot(xs, [T["switch"][str(n)][name]["peak_GB"]["p50"] for n in xs], marker="o", label=name)
        for a in axs:
            a.set_xscale("log", base=2); a.set_xlabel("history tokens")
        axs[0].set_yscale("log"); axs[0].set_ylabel("T_switch p50 (ms)"); axs[1].set_ylabel("peak GPU memory (GB)")
        axs[0].legend(fontsize=7); fig.suptitle(pair); fig.tight_layout(); fig.savefig(f"{FIG}/tswitch_mem_{pair}.png", dpi=150); plt.close(fig)
        if T.get("background"):
            Ns2 = sorted(int(n) for n in T["background"])
            fig, ax = plt.subplots(figsize=(5, 3.5))
            for name, lab in [("C1_translate_turn", "C1 incremental translation"), ("B6_shadow_prefill_turn", "B6 shadow prefill")]:
                ax.plot(Ns2, [T["background"][str(n)][name + "_gpu_ms"] for n in Ns2], marker="o", label=lab + " (GPU ms/turn)")
                ax.plot(Ns2, [T["background"][str(n)][name + "_A_extra_ms_total"] for n in Ns2], marker="x", ls="--", label=lab + " (added A latency, ms)")
            ax.set_xscale("log", base=2); ax.set_xlabel("history tokens"); ax.set_ylabel("ms"); ax.legend(fontsize=6)
            ax.set_title(f"{pair} background cost per {T['cfg']['DELTA']}-token turn"); fig.tight_layout()
            fig.savefig(f"{FIG}/background_{pair}.png", dpi=150); plt.close(fig)
    # pareto: T_switch (timing table at the task's length) vs retention, LongBench 8K
    for pair in summary:
        f = f"{R}/timing_{pair}.json"
        if not os.path.exists(f) or "lb8k" not in summary[pair]:
            continue
        T = json.load(open(f))["switch"].get("8192", {})
        fig, ax = plt.subplots(figsize=(5, 3.5))
        name_t = {"native": "native", "B2": "B2", "B1": "B1", "B3": "B3", "ours_r64": "B1"}
        for m, tm in name_t.items():
            if m in summary[pair]["lb8k"] and isinstance(T.get(tm), dict):
                ax.scatter(T[tm]["total"]["p50"], summary[pair]["lb8k"][m]["retention"][0], color=COL[m], label=m, s=40)
        if isinstance(T.get("C1_aot"), dict) and "ours_r64" in summary[pair]["lb8k"]:
            ax.scatter(T["C1_aot"]["total"]["p50"], summary[pair]["lb8k"]["ours_r64"]["retention"][0], marker="*", s=120, color="k", label="ours (C1 AOT)")
        ax.axhline(0.98, color="r", lw=0.5); ax.set_xscale("log"); ax.set_xlabel("T_switch p50 (ms) @8K"); ax.set_ylabel("retention LongBench")
        ax.legend(fontsize=7); ax.set_title(f"{pair} pareto (LongBench 8K)"); fig.tight_layout(); fig.savefig(f"{FIG}/pareto_{pair}.png", dpi=150); plt.close(fig)
except Exception as e:
    print("figure error", repr(e))


def fmt(x):
    return f"{x[0]:.3f} [{x[1]:.3f},{x[2]:.3f}]" if isinstance(x, (list, tuple)) else f"{x:.3f}"


lines = []
for pair, tb in summary.items():
    lines.append(f"\n### {pair}\n")
    for t, rows in tb.items():
        lines.append(f"\n**{t}**\n")
        if t == "uc":
            lines.append("| method | n | KL(native‖method) | top-1 agree |\n|---|---|---|---|")
            for m, r in rows.items():
                lines.append(f"| {m} | {r['n']} | {fmt(r['kl'])} | {fmt(r['top1'])} |")
        else:
            lines.append("| method | n | score | native | retention [95% CI] | harmful flip [95% CI] | T_switch p50 ms | N p50 |\n|---|---|---|---|---|---|---|---|")
            for m, r in rows.items():
                lines.append(f"| {m} | {r['n']} | {r['score']:.3f} | {r['native']:.3f} | {fmt(r['retention'])} | {fmt(r['harmful_flip'])} | {r['t_switch_ms']:.0f} | {r['N_median']:.0f} |")
for pair, g in guard.items():
    lines.append(f"\n### guard {pair}: n={g['n']} positives={g['positives']}\n")
    for k in ("s1", "s2", "s3", "s1l", "s4", "s5", "s6"):
        lines.append(f"- {k}: AUC={g[k]['auc']:.3f}, TPR@FPR<=10% = {g[k]['tpr_at_fpr10']:.3f}")
    lines.append("\n| task | n | pos | " + " | ".join(f"{k} AUC / TPR@10" for k in ("s1", "s2", "s3", "s4", "s5", "s6")) + " |")
    lines.append("|---" * 9 + "|")
    for t, v in g["per_task"].items():
        lines.append(f"| {t} | {v['n']} | {v['pos']} | " + " | ".join(f"{v[k]['auc']:.2f} / {v[k]['tpr_at_fpr10']:.2f}" for k in ("s1", "s2", "s3", "s4", "s5", "s6")) + " |")
open(f"{R}/tables.md", "w").write("\n".join(lines))
print("\n".join(lines))
