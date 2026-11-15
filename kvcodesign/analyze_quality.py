# Merge downstream results by question index; per-question paired bootstrap (95% CI) of every method against three
# references: kvbridge (dense translation), native (4B re-reads), src_native (1.7B reads and answers by itself).
# Sources: kvcodesign/results/quality_{task}_*.json (this stage) and kvbase/results/quality_{task}_{dense,cbinit}.json
# (previous stage, same deterministic question order).  Also reports truncation and the noise band.
# Root directories can be overridden (KVC, KVB) so the script also runs on a local copy of the results.
import os, json, glob, numpy as np
from tasks import TASKS, EXACT_TASKS
H = os.path.expanduser("~")
KVC = os.environ.get("KVC", f"{H}/kvcodesign/results")
KVB = os.environ.get("KVB", f"{H}/kvbase/results")
REFS = ["kvbridge", "native", "src_native"]
NB = 10000


def key_of(m, tag, prev):
    if m == "native" and prev:
        return "native_prev"
    if m == "cb_official":
        return {"cbinit": "cb_init_official", "cbkr1": "cb_kridge1_official"}.get(tag, "cb_full_official")
    return m


def boot(a, b, rng):
    d = np.asarray(a) - np.asarray(b)
    bs = d[rng.integers(0, len(d), (NB, len(d)))].mean(1)
    return float(d.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


out = {}
for t in TASKS:
    longer = [f"quality_{u}_" for u in TASKS if u != t and u.startswith(t)]   # 'gsm8k' must not pick up 'gsm8k_h_*'
    files = sorted(f for f in glob.glob(f"{KVC}/quality_{t}_*.json") if not f.endswith(".orig.json")
                   and not any(os.path.basename(f).startswith(x) for x in longer))
    files += [f for f in (f"{KVB}/quality_{t}_{x}.json" for x in ("dense", "cbinit")) if os.path.exists(f)]
    per = {}   # method -> {idx: result dict}
    for p in files:
        tag = os.path.basename(p)[len(f"quality_{t}_"):-5]
        for i, s in enumerate(json.load(open(p))["samples"]):
            for m, r in s["m"].items():
                per.setdefault(key_of(m, tag, p.startswith(KVB)), {})[i] = r
    if not per:
        continue
    rng = np.random.default_rng(0)
    rows = {}
    for m, d in per.items():
        row = dict(n=len(d), mean=float(np.mean([r["score"] for r in d.values()])))
        for extra in ("em", "digit_acc", "score_256", "trunc", "n_out"):
            v = [r[extra] for r in d.values() if extra in r]
            if v:
                row[extra] = float(np.mean(v))
        for ref in REFS:
            idx = sorted(set(d) & set(per.get(ref, {})))
            if idx and m != ref:
                diff, ci = boot([d[i]["score"] for i in idx], [per[ref][i]["score"] for i in idx], rng)
                row[f"vs_{ref}"] = dict(n=len(idx), diff=diff, ci95=ci)
        rows[m] = row
    out[t] = rows
    print(f"== {t}" + ("   [exact-information task]" if t in EXACT_TASKS else ""))
    for m, r in sorted(rows.items(), key=lambda kv: -kv[1]["mean"]):
        s = f"  {m:24s} n={r['n']:3d} mean={r['mean']:.4f}"
        for ref in REFS:
            if f"vs_{ref}" in r:
                v = r[f"vs_{ref}"]
                s += f" | vs {ref} {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
        ex = [f"{k}={r[k]:.3f}" for k in ("em", "digit_acc", "score_256", "trunc") if k in r] + ([f"n_out={r['n_out']:.0f}"] if "n_out" in r else [])
        print(s + ("   " + " ".join(ex) if ex else ""))

print("\n== noise band: fp32 minus bf16 of the same low-rank mapper (K768+V1536)")
for t, rows in out.items():
    a, b = rows.get("rrr_k768_v1536"), rows.get("rrr_k768_v1536_bf16")
    if a and b:
        print(f"  {t:12s} {a['mean'] - b['mean']:+.4f}")

print("\n== gate (stage 1): exact-information tasks where every translation method is clearly below native (CI excludes 0)")
TRANS = lambda m: m not in ("native", "native_prev", "src_native")
gate = {}
for t in EXACT_TASKS:
    rows = out.get(t, {})
    tr = {m: r["vs_native"] for m, r in rows.items() if TRANS(m) and "vs_native" in r}
    if not tr:
        continue
    best = max(tr, key=lambda m: tr[m]["diff"])
    gate[t] = dict(best=best, **tr[best], clear_gap=tr[best]["ci95"][1] < 0)
    print(f"  {t:12s} best translation = {best}: {tr[best]['diff']:+.3f} [{tr[best]['ci95'][0]:+.3f}, {tr[best]['ci95'][1]:+.3f}]"
          f"  -> {'clear gap' if gate[t]['clear_gap'] else 'no clear gap'}")
out["_gate"] = gate
json.dump(out, open(os.environ.get("SUMMARY", f"{KVC}/quality_summary.json"), "w"), indent=1)
