import json
for P in ["P1", "P2"]:
    T = json.load(open(f"/home/wxz/kvswitch/results/timing_{P}.json"))
    print(P, "unit", T["unit"])
    for N, row in T["switch"].items():
        cells = []
        for k, v in row.items():
            if v == "OOM":
                cells.append(f"{k}:OOM"); continue
            st = " ".join(f"{s}={v[s]['p50']:.0f}" for s in v if s not in ("peak_GB", "peak_over_start_GB", "total"))
            cells.append(f"{k}: tot {v['total']['p50']:.0f}/p95 {v['total']['p95']:.0f} [{st}] +{v['peak_over_start_GB']['p50']:.2f}G")
        print(P, N, "\n   " + "\n   ".join(cells))
    for N, r in T.get("background", {}).items():
        print(P, "bg", N, {k: round(v, 1) for k, v in r.items()})
    for N, r in T.get("c3", {}).items():
        print(P, "c3", N, r if r == "OOM" else {k: round(v, 1) for k, v in r.items()})
