# [kvcodesign] Task construction and scoring for eval_quality.py.  Everything here is our own code (no official
# baseline code).  Kept free of torch/transformers so that it can be unit-tested on a machine without a GPU
# (test_tasks.py); the tokenizer is passed in as `enc` (text -> list of token ids).
#
# Tasks (history H is translated, query Q is processed by the target model itself):
#   quality     QuALITY multiple choice, A-D logits                         (reading comprehension)
#   hotpotqa_e  LongBench HotpotQA-E, F1                                    (reading comprehension)
#   gsm8k       8-shot CoT, question in Q                                   (question NOT translated)
#   gsm8k_h     8-shot CoT, question inside H                               (exact information: numbers)
#   squad       SQuAD v1.1 dev, passage + question inside H, F1 / EM        (exact information: spans)
#   passkey_{2k,8k,16k}  a 6-digit code hidden in real text (QuALITY articles), question inside H, EM
#   kvret       40 "name: 6-digit number" pairs, ask for one, question inside H, EM
import re, json, string, random, collections, zipfile

TASKS = ["quality", "hotpotqa_e", "gsm8k", "gsm8k_h", "squad", "kvret", "passkey_2k", "passkey_8k", "passkey_16k"]
EXACT_TASKS = ["gsm8k_h", "squad", "kvret", "passkey_2k", "passkey_8k", "passkey_16k"]   # exact information must cross the translation
PASSKEY_LEN = {"passkey_2k": 2000, "passkey_8k": 8000, "passkey_16k": 16000}
DEPTHS = [0.1, 0.3, 0.5, 0.7, 0.9]
# Generation cap for the code tasks.  16 was too tight: translated caches often make the target answer with a full
# sentence ("The special access code mentioned in the text is ...") that was cut before the number.  The 16-token
# runs are kept in results/mx16/ as the truncation control.
CODE_MX = 64


# ---------------------------------------------------------------- scoring (format-insensitive)
def clean(s):
    """Remove markdown / LaTeX decoration that should not affect the score: ** __ ` $ \\( \\) \\boxed{x} \\text{x}."""
    s = str(s)
    for _ in range(3):
        s = re.sub(r"\\(?:boxed|text|textbf|mathrm|mathbf)\s*\{([^{}]*)\}", r"\1", s)
    for t in ("**", "__", "`", "$", "\\(", "\\)", "\\[", "\\]"):
        s = s.replace(t, "")
    return s


def norm(s):
    s = "".join(ch for ch in clean(s).lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def f1(p, golds):
    best = 0.0
    for g in golds:
        a, b = norm(p).split(), norm(g).split()
        c = sum((collections.Counter(a) & collections.Counter(b)).values())
        if c:
            pr, rc = c / len(a), c / len(b); best = max(best, 2 * pr * rc / (pr + rc))
    return best


def em(p, golds):
    return float(any(norm(p) == norm(g) for g in golds))


def gsm_extract(pred):
    t = pred.replace("**", "").replace("$", "").replace(",", "")
    for pat in (r"answer is\s*:?\s*(-?\d+\.?\d*)", r"boxed\{\s*\\?(-?\d+\.?\d*)", r"[Aa]nswer\s*:?\s*(-?\d+\.?\d*)"):
        m = re.findall(pat, t)
        if m:
            return m[-1]
    m = re.findall(r"-?\d+\.?\d*", t)
    return m[-1] if m else None


def gsm_eq(a, gold):
    try:
        return a is not None and abs(float(a.rstrip(".")) - float(gold)) < 1e-6
    except ValueError:
        return False


def digit_runs(pred):
    t = re.sub(r"(?<=\d)[,\s](?=\d{3}\b)", "", clean(pred))   # 482,913 / 482 913 -> 482913
    return re.findall(r"\d+", t)


def code_score(pred, gold):
    """Exact match: some digit run equals the gold code.  digit_acc: position-wise agreement of the best run
    (partial credit; separates 'found the place but copied it wrong' from 'did not find it')."""
    runs = digit_runs(pred)
    acc = max([sum(a == b for a, b in zip(r, gold)) / max(len(r), len(gold)) for r in runs], default=0.0)
    return dict(score=float(gold in runs), digit_acc=acc)


def score(s, pred):
    """-> dict with 'score' (the headline metric) plus task-specific extras."""
    k = s["kind"]
    if k == "mc":
        return dict(score=float(pred == s["gold"]))
    if k == "gen":
        return dict(score=f1(pred, s["gold"]), em=em(pred, s["gold"]))
    if k == "code":
        return code_score(pred, s["gold"])
    return dict(score=float(gsm_eq(gsm_extract(pred), s["gold"])))


# ---------------------------------------------------------------- task construction
def chat_split(enc, user_hist, user_query):
    h = f"<|im_start|>user\n{user_hist}"
    q = f"{user_query}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return enc(h), enc(q)


def quality_docs():
    """QuALITY validation articles in the fixed shuffled order used everywhere; the first 24 are CacheBridge
    V-calibration documents and are never used for evaluation or as filler."""
    from datasets import load_dataset
    arts = {}
    for r in load_dataset("emozilla/quality", split="validation"):
        arts.setdefault(r["article"], []).append(r)
    docs = [v for v in arts.values() if len(v) >= 4]
    random.Random(0).shuffle(docs)
    return docs[24:]


def split_sentences(text):
    return [x for x in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if x]


def passkey_samples(enc, n, length, articles):
    """A 6-digit code is stated once, at a relative depth cycling through DEPTHS, inside real article text of about
    `length` tokens.  The question is part of the history.  Deterministic (seeded per sample)."""
    sent_cache = {}

    def sents_of(j):
        if j not in sent_cache:
            ss = split_sentences(articles[j])
            sent_cache[j] = [(x, len(enc(" " + x))) for x in ss]
        return sent_cache[j]

    out = []
    for i in range(n):
        rng = random.Random(1000 + i)
        code = str(rng.randint(100000, 999999))
        j, sents, tot = (i * 3) % len(articles), [], 0
        while tot < length - 80:
            for x, c in sents_of(j):
                if tot >= length - 80:
                    break
                sents.append(x); tot += c
            j = (j + 1) % len(articles)
        depth = DEPTHS[i % len(DEPTHS)]
        sents.insert(int(depth * len(sents)), f"The special access code is {code}.")
        hi, qi = chat_split(enc, "Read the following text carefully.\n\n" + " ".join(sents) +
                            "\n\nQuestion: What is the special access code mentioned in the text? Reply with the number only.\n", "Answer:")
        out.append(dict(h=hi, q=qi, gold=code, kind="code", mx=CODE_MX, meta=dict(depth=depth)))
    return out


FIRST = "James Mary Robert Linda Michael Susan David Karen Daniel Nancy Paul Laura Kevin Emily Brian Helen Jason Sarah Eric Julia".split()
LAST = "Carter Hughes Bennett Foster Morgan Reyes Sullivan Parker Griffin Hayes Coleman Barnes Jensen Walsh Fisher Burke Lambert Tucker Porter Dalton".split()


def kvret_samples(enc, n, pairs=40):
    out = []
    for i in range(n):
        rng = random.Random(2000 + i)
        names = rng.sample([f"{a} {b}" for a in FIRST for b in LAST], pairs)
        nums = [str(x) for x in rng.sample(range(100000, 1000000), pairs)]
        t = rng.randrange(pairs)
        lines = "\n".join(f"{a}: {b}" for a, b in zip(names, nums))
        hi, qi = chat_split(enc, f"Here is a list of employee ID numbers.\n\n{lines}\n\n"
                                 f"Question: What is the ID number of {names[t]}? Reply with the number only.\n", "Answer:")
        out.append(dict(h=hi, q=qi, gold=nums[t], kind="code", mx=CODE_MX, meta=dict(pos=t / (pairs - 1))))
    return out


def build_samples(task, n, enc):
    samples = []
    if task == "quality":
        for d in quality_docs():
            for r in d:
                opts = "\n".join(f"{c}. {o}" for c, o in zip("ABCD", r["options"]))
                hi, qi = chat_split(enc, f"Read the following article.\n\n{d[0]['article']}",
                                    f"\n\nQuestion: {r['question']}\n{opts}\nAnswer with the letter of the correct option.")
                samples.append(dict(h=hi[:12000], q=qi, gold=int(r["answer"]), kind="mc"))
        samples = samples[:n]
    elif task == "hotpotqa_e":
        from huggingface_hub import hf_hub_download
        z = zipfile.ZipFile(hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset"))
        rows = [json.loads(l) for l in z.open("data/hotpotqa_e.jsonl")]
        random.Random(0).shuffle(rows)
        inst = "Answer the question based on the given passages. Only give me the answer and do not output any other words."
        for r in rows[:n]:
            hi, qi = chat_split(enc, f"{inst}\n\nThe following are given passages.\n{r['context']}",
                                f"\n\n{inst}\n\nQuestion: {r['input']}\nAnswer:")
            samples.append(dict(h=hi[:12000], q=qi, gold=r["answers"], kind="gen", mx=32))
    elif task in ("gsm8k", "gsm8k_h"):
        from datasets import load_dataset
        tr = load_dataset("openai/gsm8k", "main", split="train")
        te = load_dataset("openai/gsm8k", "main", split="test")
        shots = "".join(f"Question: {tr[i]['question']}\nAnswer: {tr[i]['answer'].replace('####', 'The answer is')}\n\n" for i in range(8))
        for r in list(te)[:n]:
            if task == "gsm8k":
                hi, qi = chat_split(enc, f"Solve the math problems.\n\n{shots}", f"Question: {r['question']}\nAnswer:")
            else:   # the question is part of the (translated) history
                hi, qi = chat_split(enc, f"Solve the math problems.\n\n{shots}Question: {r['question']}\n", "Answer:")
            samples.append(dict(h=hi, q=qi, gold=r["answer"].split("####")[-1].strip().replace(",", ""), kind="gsm",
                                mx=256 if task == "gsm8k" else 512))
    elif task == "squad":   # passage and question are both inside the translated history
        from datasets import load_dataset
        rows = list(load_dataset("rajpurkar/squad", split="validation"))
        random.Random(0).shuffle(rows)
        for r in rows[:n]:
            hi, qi = chat_split(enc, "Read the passage and answer the question with a short phrase copied from the passage.\n\n"
                                     f"Passage: {r['context']}\n\nQuestion: {r['question']}\n", "Answer:")
            samples.append(dict(h=hi, q=qi, gold=sorted(set(r["answers"]["text"])), kind="gen", mx=32))
    elif task in PASSKEY_LEN:
        samples = passkey_samples(enc, n, PASSKEY_LEN[task], [d[0]["article"] for d in quality_docs()])
    elif task == "kvret":
        samples = kvret_samples(enc, n)
    else:
        raise ValueError(f"unknown task {task}")
    return samples
