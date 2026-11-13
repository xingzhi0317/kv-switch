import json, sys
S = json.load(open("/home/wxz/kvswitch/results/summary.json"))
for P in sys.argv[1:]:
    for t, rows in S.get(P, {}).items():
        out = []
        for m, r in rows.items():
            if "kl" in r:
                out.append("%s:KL%.2f/top1 %.2f" % (m, r["kl"][0], r["top1"][0]))
            else:
                out.append("%s:%.2f(n%d,flip%.2f,%dms)" % (m, r["retention"][0], r["n"], r["harmful_flip"][0], r["t_switch_ms"]))
        print(P, t, " | ".join(out))
