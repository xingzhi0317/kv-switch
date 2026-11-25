# Paper-style probes for the KL-trained translators (kvlingo.py / kvlingo_train2.py), aligned with KV-Lingo Table 4:
#   ARC-Challenge and MMLU-Pro: reasoning off, generate up to 1024 tokens, lenient parse of the chosen letter
#     (KV-Lingo "generate-and-parse"); ARC 300 of 1172 items, MMLU-Pro 20 per category (280) -- fixed random subsets;
#   CoQA: reasoning off, up to 128 tokens, token F1 (SQuAD normalisation) and gold-in-answer; prompt format COQA_FMT;
#   needle: 6-digit access code at 2/4/8/16K, 20 each (Hybrid-KV style).
# The source prefills all prompt tokens but the last, the translator maps that cache, the target reads the held-back
# token and decodes (KV-Lingo App. C).  Decoding is greedy and batched: each item reads its last token on its own cache,
# then the caches are left-padded and decoded together (as eval_quality.answer_batch).
# env: PAIR SDEV TDEV TASKS METHODS (tgt_native,src_native,<name>) TRS (name=path,...) COQA_FMT COQA_N OUT
import os, sys, json, re, random, time
import torch
from transformers import DynamicCache

sys.argv = sys.argv[:1]
import kvlingo as K

E = os.environ.get
TASKS = E("TASKS", "arc_c,mmlu_pro,coqa,needle").split(",")
TRS = dict(x.split("=", 1) for x in E("TRS", "").split(",") if x)
METHODS = E("METHODS", ",".join(["tgt_native", "src_native"] + list(TRS))).split(",")
FMT = E("COQA_FMT", "multi")
OUT = E("OUT", f"{K.OUT}/eval2_{K.PAIR}.json")
tok = K.tok
STOP = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
log = K.log


def user_prompt(text, system=None):
    s = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    return K.enc(s + f"<|im_start|>user\n{text}<|im_end|>\n" + K.HDR)


def items(task):
    from datasets import load_dataset
    out = []
    if task == "arc_c":
        rows = list(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"))
        random.Random(0).shuffle(rows)
        for r in rows[:int(E("ARC_N", 300))]:
            opts = "\n".join(f"{l}. {t}" for l, t in zip(r["choices"]["label"], r["choices"]["text"]))
            out.append(dict(p=user_prompt(f"{r['question']}\n\n{opts}\n\nAnswer with the label of the correct option."),
                            gold=r["answerKey"], labels=r["choices"]["label"], mx=1024))
    elif task == "mmlu_pro":
        by = {}
        for r in load_dataset("TIGER-Lab/MMLU-Pro", split="test"):
            by.setdefault(r["category"], []).append(r)
        L = "ABCDEFGHIJ"
        for cat in sorted(by):
            rs = by[cat][:]; random.Random(0).shuffle(rs)
            for r in rs[:int(E("MMLU_PER_CAT", 20))]:
                opts = "\n".join(f"{L[i]}. {o}" for i, o in enumerate(r["options"]))
                out.append(dict(p=user_prompt(f"{r['question']}\n\n{opts}\n\nAnswer with the letter of the correct option."),
                                gold=r["answer"], labels=list(L[:len(r["options"])]), cat=cat, mx=1024))
    elif task == "coqa":
        rng = random.Random(0)
        for r in load_dataset("stanfordnlp/coqa", split="validation"):
            qs, ans = r["questions"], r["answers"]["input_text"]
            j = rng.randrange(len(qs))
            if FMT == "single":   # one user turn: story, instruction, Q/A history, question
                hist = "".join(f"Q: {qs[i]}\nA: {ans[i]}\n" for i in range(j))
                p = user_prompt(f"{r['story']}\n\nAnswer the last question briefly, based on the story.\n\n{hist}Q: {qs[j]}")
            else:                 # conversational: story + first question, then alternating turns
                sysmsg = "Answer each question about the story with a short phrase." if FMT == "multi_sys" else None
                s = f"<|im_start|>system\n{sysmsg}<|im_end|>\n" if sysmsg else ""
                for i in range(j + 1):
                    u = (f"{r['story']}\n\n{qs[i]}" if i == 0 else qs[i])
                    s += f"<|im_start|>user\n{u}<|im_end|>\n"
                    if i < j:
                        s += f"<|im_start|>assistant\n{ans[i]}<|im_end|>\n"
                p = K.enc(s + K.HDR)
            if len(p) <= 1024:
                out.append(dict(p=p, gold=ans[j], mx=128))
        random.Random(1).shuffle(out)
        out = out[:int(E("COQA_N", 256))]
    elif task == "needle":
        import tasks
        arts = [" ".join(x[0]["article"].split()) for x in tasks.quality_docs()]
        for L_ in (2048, 4096, 8192, 16384):
            for s in tasks.passkey_samples(K.enc, 20, L_, arts):
                out.append(dict(p=s["h"] + s["q"], gold=s["gold"], mx=32, length=L_))
    return out


def parse_mc(out, labels):
    """Lenient letter parse: 'answer is X' / 'Answer: X' (last), bold **X, a leading 'X.', else the last '(X)' or 'X.'."""
    L = re.escape("".join(labels))
    for pat in (rf"answer is[:\s]*\**\(?([{L}])\b", rf"[Aa]nswer[:\s]*\**\(?([{L}])\b", rf"\*\*\(?([{L}])[\.\):\*\s]"):
        m = re.findall(pat, out)
        if m:
            return m[-1]
    m = re.match(rf"\s*\(?([{L}])(?:[\.\):\s]|$)", out)
    if m:
        return m.group(1)
    m = re.findall(rf"\(([{L}])\)|\b([{L}])\.", out)
    return (m[-1][0] or m[-1][1]) if m else None


def norm(s):
    s = re.sub(r"\b(a|an|the)\b", " ", re.sub(r"[^\w\s]", " ", s.lower()))
    return " ".join(s.split())


def f1(p, g):
    from collections import Counter
    p, g = norm(p).split(), norm(g).split()
    if not p or not g:
        return float(p == g)
    c = sum((Counter(p) & Counter(g)).values())
    return 0.0 if c == 0 else 2 * c / (len(p) + len(g))


def score(task, it, out):
    if task in ("arc_c", "mmlu_pro"):
        a = parse_mc(out, it["labels"])
        return dict(score=float(a == it["gold"]), parsed=a is not None)
    if task == "coqa":
        return dict(score=f1(out, it["gold"]), gold_in=float(norm(it["gold"]) in norm(out)), ntok=len(tok(out).input_ids))
    return dict(score=float(it["gold"] in re.findall(r"\d{6}", out)))


@torch.no_grad()
def decode(M, dev, kvs, lasts, ns, mx):
    """Batched greedy decode: item b reads lasts[b] on its prefix kvs[b] (n = ns[b] positions), then all decode."""
    first, caches = [], []
    for kv, last, n in zip(kvs, lasts, ns):
        c = DynamicCache()
        for l, (k, v) in enumerate(kv):
            c.update(k, v, l)
        lg = M(torch.tensor([[last]], device=dev), past_key_values=c, use_cache=True,
               cache_position=torch.tensor([n], device=dev)).logits[0, -1]
        first.append(int(lg.argmax())); caches.append(c)
    lens = [n + 1 for n in ns]
    B, Lm, nl = len(caches), max(lens), len(caches[0].layers)
    bc = DynamicCache()
    for l in range(nl):
        ks, vs = [], []
        for c, n in zip(caches, lens):
            k, v = c.layers[l].keys, c.layers[l].values
            if n < Lm:
                z = k.new_zeros(1, k.shape[1], Lm - n, k.shape[3])
                k, v = torch.cat([z, k], 2), torch.cat([z, v], 2)
            ks.append(k); vs.append(v)
            c.layers[l].keys = c.layers[l].values = None
        bc.update(torch.cat(ks, 0), torch.cat(vs, 0), l)
    mask = torch.zeros(B, Lm, dtype=torch.long, device=dev)
    for b, n in enumerate(lens):
        mask[b, Lm - n:] = 1
    pos = torch.tensor(lens, device=dev)
    nxt = torch.tensor(first, device=dev)
    outs, done = [[] for _ in range(B)], [False] * B
    for t in range(mx + 1):
        for b in range(B):
            if not done[b]:
                x = int(nxt[b])
                if x in STOP or len(outs[b]) >= mx:
                    done[b] = True
                else:
                    outs[b].append(x)
        if all(done) or t == mx:
            break
        mask = torch.cat([mask, torch.ones(B, 1, dtype=torch.long, device=dev)], 1)
        nxt = M(nxt[:, None], past_key_values=bc, attention_mask=mask, position_ids=(pos + t)[:, None], use_cache=True).logits[:, -1].argmax(-1)
    del bc
    return [tok.decode(o, skip_special_tokens=True).strip() for o in outs]


def main():
    S, T = K.load(K.SRC, K.SDEV), K.load(K.TGT, K.TDEV)
    trs = {name: K.load_tr(S, T, path) for name, path in TRS.items() if name in METHODS}
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for task in TASKS:
        its = items(task)
        tkey = f"coqa_{FMT}" if task == "coqa" else task
        rows = {m: [None] * len(its) for m in METHODS}
        outs = {m: [None] * len(its) for m in METHODS}
        budget = int(E("TOK_BUDGET", 60000))   # prefix tokens per decode batch
        order = sorted(range(len(its)), key=lambda i: len(its[i]["p"]))
        groups, cur = [], []
        for i in order:
            if cur and (len(cur) + 1) * len(its[i]["p"]) > budget or len(cur) >= 16:
                groups.append(cur); cur = []
            cur.append(i)
        groups.append(cur)
        t1 = time.time()
        for gi, g in enumerate(groups):
            ps = [its[i]["p"] for i in g]; ns = [len(p) - 1 for p in ps]; mx = its[g[0]]["mx"]
            for m in METHODS:
                with torch.no_grad():
                    if m in ("tgt_native", "src_native"):
                        M, dev = (T, K.TDEV) if m == "tgt_native" else (S, K.SDEV)
                        kvs = [K.native_kv(M, torch.tensor([p[:n]], device=dev)) for p, n in zip(ps, ns)]
                    else:
                        M, dev, kvs = T, K.TDEV, []
                        for p, n in zip(ps, ns):
                            sk, sv = K.prenorm_kv(S, torch.tensor([p], device=K.SDEV), last=n)
                            kvs.append(trs[m].translate(T, [a.to(K.TDEV) for a in sk], [a.to(K.TDEV) for a in sv]))
                    texts = decode(M, dev, kvs, [p[-1] for p in ps], ns, mx)
                    del kvs
                for i, o in zip(g, texts):
                    r = score(task, its[i], o)
                    r.update({k: its[i][k] for k in ("length", "cat") if k in its[i]})
                    rows[m][i], outs[m][i] = r, o[:400]
            if (gi + 1) % 5 == 0 or gi + 1 == len(groups):
                done = [i for i in range(len(its)) if rows[METHODS[0]][i] is not None]
                log(tkey, len(done), "/", len(its), {m: round(sum(rows[m][i]["score"] for i in done) / len(done), 4) for m in METHODS},
                    f"{(time.time() - t1) / 60:.1f} min")
        summ = {}
        for m in METHODS:
            v = rows[m]
            s = dict(n=len(v), mean=round(sum(r["score"] for r in v) / len(v), 4))
            if task == "coqa":
                s["gold_in"] = round(sum(r["gold_in"] for r in v) / len(v), 4)
                s["mean_tokens"] = round(sum(r["ntok"] for r in v) / len(v), 1)
            if task in ("arc_c", "mmlu_pro"):
                s["unparsed"] = sum(not r["parsed"] for r in v)
            if task == "mmlu_pro":
                cats = {}
                for r in v:
                    cats.setdefault(r["cat"], []).append(r["score"])
                s["macro"] = round(sum(sum(c) / len(c) for c in cats.values()) / len(cats), 4)
            if task == "needle":
                s["by_len"] = {L_: round(sum(r["score"] for r in v if r["length"] == L_) / max(1, sum(r["length"] == L_ for r in v)), 3)
                               for L_ in (2048, 4096, 8192, 16384)}
            summ[m] = s
        res.setdefault(tkey, {}).update({m: dict(summary=summ[m], items=[r["score"] for r in rows[m]], outputs=outs[m]) for m in METHODS})
        json.dump(res, open(OUT, "w"))
        log("EVAL2DONE", tkey, {m: summ[m] for m in METHODS})


if __name__ == "__main__":
    main()
