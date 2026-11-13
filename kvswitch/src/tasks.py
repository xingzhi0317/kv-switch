# Task builders.  Each sample: dict(id, hist=[(role,text)...] (A's turns, before switch),
#   turns=[(user_text, gold or None, max_new)], metric) .  B answers every turn in `turns` after the switch.
import json, random, re, string, collections
from ks import ROOT

def _norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred, golds):
    if isinstance(golds, str):
        golds = [golds]
    best = 0.0
    p = _norm(pred).split()
    for g in golds:
        gt = _norm(g).split()
        common = collections.Counter(p) & collections.Counter(gt)
        ns = sum(common.values())
        if ns == 0:
            continue
        pr, rc = ns / len(p), ns / len(gt)
        best = max(best, 2 * pr * rc / (pr + rc))
    return best


def gsm_num(s):
    s = s.replace(",", "")
    m = re.findall(r"####\s*(-?\d+\.?\d*)", s) or re.findall(r"answer is[^\d-]*(-?\d+\.?\d*)", s) or re.findall(r"-?\d+\.?\d*", s)
    return m[-1].rstrip(".") if m else None


def score(metric, pred, gold):
    if metric == "f1":
        return f1(pred, gold)
    if metric == "gsm":
        a, b = gsm_num(pred), gsm_num(gold)
        try:
            return float(a is not None and abs(float(a) - float(b)) < 1e-4)
        except Exception:
            return 0.0
    if metric == "contains":
        return float(gold in pred)
    raise ValueError(metric)


def coqa(n, k, seed=0):
    d = json.load(open(f"{ROOT}/data/coqa.json"))
    rng = random.Random(seed)
    d = [x for x in d if len(x["qs"]) >= k + 3]
    rng.shuffle(d)
    out = []
    for i, x in enumerate(d[:n]):
        sys_q = f"Read the story and answer my questions with a short phrase.\n\n{x['story']}\n\nQuestion: {x['qs'][0]}"
        msgs = [("user", sys_q), ("assistant", x["ans"][0])]
        for j in range(1, k):
            msgs += [("user", x["qs"][j]), ("assistant", x["ans"][j])]
        turns = [(x["qs"][j], x["ans"][j], 24) for j in range(k, min(k + 3, len(x["qs"])))]
        out.append(dict(id=f"coqa{i}_k{k}", hist=msgs, turns=turns, metric="f1"))
    return out


LB_PROMPT = {
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.",
    "multifieldqa_en": "Read the article and answer the question as concisely as you can, using a single phrase or sentence if possible.",
    "qasper": "Answer the question based on the article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered, write \"unanswerable\". If it is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\".",
}
LB_MAXNEW = {"hotpotqa": 32, "2wikimqa": 32, "multifieldqa_en": 64, "qasper": 64}


def longbench(tok, n_per, target_len, seed=0):
    """Doc in turn 1 (user), A acknowledges; question after the switch. Doc middle-truncated to target_len tokens."""
    d = json.load(open(f"{ROOT}/data/longbench.json"))
    rng = random.Random(seed)
    out = []
    for t, rows in d.items():
        rows = list(rows)
        rng.shuffle(rows)
        for i, r in enumerate(rows[:n_per]):
            ids = tok(r["ctx"], add_special_tokens=False).input_ids
            if len(ids) > target_len:
                h = target_len // 2
                ctx = tok.decode(ids[:h]) + tok.decode(ids[-h:])
            else:
                ctx = r["ctx"]
            u1 = f"{LB_PROMPT[t]}\n\n{ctx}\n\nI will ask you a question about this text next."
            hist = [("user", u1), ("assistant", "Understood. I have read the text. Please ask your question.")]
            q = f"{LB_PROMPT[t]}\n\nQuestion: {r['q']}\nAnswer:"
            out.append(dict(id=f"lb_{t}_{i}", task=t, hist=hist, turns=[(q, r["ans"], LB_MAXNEW[t])], metric="f1"))
    return out


def niah(tok, n, L, seed=0):
    import torch, os
    rng = random.Random(seed + L)
    hay = torch.load(os.path.expanduser("~/kvmap/reuse/c4_test.pt"))
    cities = ["Lisbon", "Nairobi", "Osaka", "Quito", "Tallinn", "Hanoi", "Bergen", "Perth", "Cusco", "Tbilisi", "Accra", "Zagreb"]
    out = []
    for i in range(n):
        city, num = rng.choice(cities), str(rng.randint(1000000, 9999999))
        needle = f" The special magic number for {city} is {num}. "
        o = rng.randrange(len(hay) - 2 * L)
        text = tok.decode(hay[o:o + L - 200].tolist())
        depth = rng.random()
        cut = int(len(text) * depth)
        cut = text.rfind(". ", 0, cut) + 2 if text.rfind(". ", 0, cut) > 0 else cut
        doc = text[:cut] + needle + text[cut:]
        hist = [("user", f"Here is a long document. Memorize it.\n\n{doc}"), ("assistant", "I have read the document.")]
        q = f"What is the special magic number for {city} mentioned in the document? Answer with the number only."
        out.append(dict(id=f"niah{L}_{i}", hist=hist, turns=[(q, num, 16)], metric="contains", depth=depth))
    return out


def gsm8k(n, nprev=3, seed=0):
    d = json.load(open(f"{ROOT}/data/gsm8k.json"))
    tr = json.load(open(f"{ROOT}/data/gsm8k_train.json"))
    rng = random.Random(seed)
    out = []
    inst = "Solve the math problem step by step. End with '#### <number>'.\n\n"
    for i in range(n):
        x = d[i]
        hist = []
        for p in rng.sample(tr, nprev):
            hist += [("user", inst + p["q"]), ("assistant", p["a"].replace("<<", "").replace(">>", ""))]
        out.append(dict(id=f"gsm{i}", hist=hist, turns=[(inst + x["q"], x["a"], 320)], metric="gsm"))
    return out


def ultrachat(n, min_turns=2, seed=0):
    d = json.load(open(f"{ROOT}/data/ultrachat.json"))
    rng = random.Random(seed)
    d = [c for c in d if len(c) >= 2 * min_turns + 2]
    rng.shuffle(d)
    out = []
    for i, c in enumerate(d[:n]):
        k = len(c) // 2 - 1
        hist = [(m["role"], m["content"]) for m in c[:2 * k]]
        out.append(dict(id=f"uc{i}", hist=hist, user=c[2 * k]["content"], reply=c[2 * k + 1]["content"], metric="kl"))
    return out
