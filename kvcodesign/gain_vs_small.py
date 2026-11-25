# Gain over the small model answering alone (src_native), Qwen3-1.7B -> 4B, three tasks with equal weight, paired bootstrap.
#   gain    = mean over tasks of (method - small) / re-prefill score, in points of retention
#   regained= gain / (100 - retention of the small model): share of the small -> large gap recovered (0 = no better than small, 100 = re-prefill)
# Reads per-question results (results/quality_{task}_*.json, results/specprefill/); run where they are: usage: python3 gain_vs_small.py
import json, glob, os, random
R = os.path.expanduser("~/kvcodesign/results/") if os.path.exists(os.path.expanduser("~/kvcodesign/results")) else "results/"
T = ["quality", "hotpotqa_e", "passkey_8k"]; random.seed(0)
get = {t: {} for t in T}
for t in T:
    KB = os.path.expanduser("~/kvbase/results/")
    for f in sorted(glob.glob(R + f"quality_{t}_*.json")) + [KB + f"quality_{t}_dense.json", R + f"specprefill/quality_{t}.json"]:
        if f.endswith(".orig.json"): continue
        try: v = json.load(open(f, encoding="utf8", errors="replace"))["samples"]
        except Exception: continue
        for m in v[0]["m"]:
            tag = os.path.basename(f)[len(f"quality_{t}_"):-5]
            k = "night:" + m if "/specprefill/" in f.replace("\\", "/") else ({"cbinit": "cb_init", "cbec": "cb_ec_official"}.get(tag, "cb_full") if m == "cb_official" else m)
            if m == "native" and f.startswith(KB): k = "native_prev"
            s = [x["m"][m]["score"] for x in v if m in x["m"]]
            if len(s) > len(get[t].get(k, [])): get[t][k] = s
M = [("kvbridge", "closed form (KVBridge)"), ("cb_ec_official", "CacheBridge"), ("kb3cklbest", "KL dense"), ("kl_k768_v1536_best17t4_bf16", "KL low-rank (ours)"),
     ("rare10", "RaReCache 10%"), ("rare30", "RaReCache 30%"), ("rare50", "RaReCache 50%"), ("night:spec20", "SpecPrefill 20%"), ("native", "re-prefill")]
def stat(m, ix=None):
    g, h, per = [], [], []
    for t in T:
        a, s, n = get[t][m], get[t]["src_native" if not m.startswith("night:") else "night:src_native"], get[t]["native" if not m.startswith("night:") else "night:native"]
        k = min(len(a), len(s), len(n)); I = ix[t] if ix else range(k)
        I = [i for i in I if i < k]; A, B, N = (sum(x[i] for i in I) / len(I) for x in (a, s, n))
        g.append(100 * (A - B) / N); h.append(100 * (N - B) / N)
    return sum(g) / 3, 100 * sum(g) / max(1e-9, sum(h)), g
print("method | gain (points) [95%] | share of the small->large gap recovered | per task")
OUT = {}
for m, nm in M:
    n = {t: min(len(get[t][m]), len(get[t]["src_native"])) for t in T}; b = []; pt = []
    for _ in range(2000):
        z = stat(m, {t: [random.randrange(n[t]) for _ in range(n[t])] for t in T}); b.append(z[0]); pt.append(z[2])
    b.sort(); g, r, per = stat(m); ci = [sorted(p[j] for p in pt) for j in range(3)]; OUT[m] = dict(gain=g, ci=[b[50], b[1950]], regained=r, per={t: [per[j], ci[j][50], ci[j][1950]] for j, t in enumerate(T)}); print(f"{nm:24s} {g:+.1f} [{b[50]:+.1f},{b[1950]:+.1f}] | {r:.0f}% | " + " ".join(f"{t[:8]}:{p:+.0f}[{c[50]:+.0f},{c[1950]:+.0f}]" for t, p, c in zip(T, per, ci)), "n", tuple(n.values()))
json.dump(OUT, open(R + "gain_vs_small.json", "w"), indent=1)
