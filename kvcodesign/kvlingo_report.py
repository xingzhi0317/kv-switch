# Side-by-side table: our kvlingo.py runs vs the published numbers of KV-Lingo (arXiv 2609.32610, Tables 9/10, nu=1,
# reasoning off, relative-depth assignment) and Hybrid-KV Part 1 (arXiv 2611.12062 §5).  Reads kvlingo/res_*.json,
# writes kvlingo/REPORT_kvlingo.md.  Paper "RULER" is NIAH over 4-16K; ours is a single 6-digit needle at 2-16K (easier).
import json, os, time

D = os.path.expanduser("~/kvcodesign/kvlingo")
PAPER = {   # (pair, metric) -> {method: value}; native_src / native_tgt are the source / target models alone
    "kvl_17to4": dict(title="KV-Lingo Qwen3-1.7B -> 4B (Table 10, SD depth)", rows={
        "arc_c": dict(native_src=0.770, native_tgt=0.920, cf=0.012, sd=0.794),
        "mmlu_pro": dict(native_src=0.379, native_tgt=0.564, cf=0.016, sd=0.324),
        "needle": dict(native_src=0.980, native_tgt=0.999, cf=0.003, sd=0.645)}),
    "kvl_4to17": dict(title="KV-Lingo Qwen3-4B -> 1.7B (Table 10, SD depth)", rows={
        "arc_c": dict(native_src=0.920, native_tgt=0.770, cf=0.349, sd=0.817),
        "mmlu_pro": dict(native_src=0.564, native_tgt=0.379, cf=0.134, sd=0.283),
        "needle": dict(native_src=0.999, native_tgt=0.980, cf=0.000, sd=0.415)}),
    "hkv_8to4": dict(title="Hybrid-KV Part 1 Qwen3-8B -> 4B (KV-Lingo in brackets)", rows={
        "val_kl": dict(cf="0.322 [0.117]", sd="0.056 [0.023]"),
        "r2_kv": dict(cf="0.802/0.658 [0.815/0.693]"),
        "coqa": dict(native_tgt=0.763, sd="0.686 [0.727]"),
        "coqa_gold_in": dict(native_tgt=0.820, sd=0.807),
        "needle": dict(sd="1.00 (2-8K)")}),
    "hkv_4to8": dict(title="Hybrid-KV Part 1 Qwen3-4B -> 8B (KV-Lingo Table 9 in brackets)", rows={
        "val_kl": dict(cf="0.162 [0.098]", sd="0.039 [0.021]"),
        "r2_kv": dict(cf="0.798/0.646 [0.808/0.676]"),
        "coqa": dict(native_src="[0.721]", native_tgt="[0.712]", cf="[0.394]", sd="0.697 [0.745]"),
        "coqa_gold_in": dict(sd=0.823),
        "arc_c": dict(native_src="[0.920]", native_tgt="[0.937]", cf="[0.905]", sd="[0.91-0.93]"),
        "mmlu_pro": dict(native_src="[0.564]", native_tgt="[0.615]", cf="[0.494]", sd="[0.55-0.57]"),
        "needle": dict(sd="1.00 (2-8K) [RULER ~0.99]")}),
}


def load(tag):
    p = f"{D}/res_{tag}.json"
    return json.load(open(p)) if os.path.exists(p) else {}


def ours(tag, metric):
    r, out = load(tag), {}
    if metric == "val_kl":
        v = r.get("val", [])
        if v:
            out["cf"], out["sd"] = v[0][1], f"{v[-1][1]} (step {v[-1][0]})"
        return out
    if metric == "r2_kv":
        import torch
        p = f"{D}/cf_{tag.split('_')[1]}_nu1_b{1 if tag.startswith('hkv') else 0}.pt"
        if os.path.exists(p):
            r2 = torch.load(p, map_location="cpu")["r2"]
            out["cf"] = f"{sum(r2['k']) / len(r2['k']):.3f}/{sum(r2['v']) / len(r2['v']):.3f}"
        return out
    task = "coqa" if metric == "coqa_gold_in" else metric
    key = "gold_in" if metric == "coqa_gold_in" else ("macro" if metric == "mmlu_pro" else "mean")
    ev = r.get("eval", {}).get(task, {})
    for m in ("cf", "sd"):
        if m in ev:
            out[m] = ev[m].get(key, ev[m]["mean"])
    # natives from the run (same item set) where that model was evaluated alone
    NAT = {"kvl": {"1.7B": ("kvl_4to17", "tgt_native"), "4B": ("kvl_17to4", "tgt_native")},
           "hkv": {"4B": ("hkv_8to4", "tgt_native"), "8B": ("hkv_8to4", "src_native")}}[tag[:3]]
    src, tgt = {"kvl_17to4": ("1.7B", "4B"), "kvl_4to17": ("4B", "1.7B"), "hkv_8to4": ("8B", "4B"), "hkv_4to8": ("4B", "8B")}[tag]
    for col, model in (("native_src", src), ("native_tgt", tgt)):
        if model in NAT:
            t, field = NAT[model]
            e = load(t).get("eval", {}).get(task, {})
            if field in e:
                out[col] = e[field].get(key, e[field]["mean"])
    return out


def fmt(x):
    return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


lines = [f"# KL 翻译器论文对齐（kvlingo.py）", "", f"更新于 {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}。"
         "每格是「我们 / 论文」。论文的 RULER 是 4–16K 的 NIAH，我们的 needle 是 2–16K 单个 6 位口令，更容易。", ""]
for tag, spec in PAPER.items():
    r = load(tag)
    status = "未开始"
    if r.get("val"):
        status = f"训练到第 {r['val'][-1][0]} 步"
    if r.get("eval"):
        status += "，已评测：" + "、".join(r["eval"])
    lines += [f"## {spec['title']}", "", f"状态：{status}", "",
              "| 指标 | 源模型原生 | 目标模型原生 | 闭式解 CF | KL 蒸馏 SD |", "|---|---|---|---|---|"]
    for metric, pap in spec["rows"].items():
        o = ours(tag, metric)
        cells = [f"{fmt(o.get(c))} / {fmt(pap.get(c))}" for c in ("native_src", "native_tgt", "cf", "sd")]
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    ev = r.get("eval", {}).get("needle", {})
    if ev:
        lines.append("")
        lines.append("needle 按长度：" + "；".join(f"{m} {v.get('by_len')}" for m, v in ev.items()))
    lines.append("")
open(f"{D}/REPORT_kvlingo.md", "w").write("\n".join(lines))
print("\n".join(lines))
