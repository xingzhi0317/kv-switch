# Quality under the CacheBridge handoff protocol: source (Qwen3-1.7B) prefills history H; target (Qwen3-4B) gets
# H's KV via a translator, processes the query Q and answers.  Thinking disabled.
# methods: native | cb_official | cb_fused | kvbridge | dense_fused | dense_fp8
# tasks  : quality (A-D logits, 0-shot) | hotpotqa_e (F1) | gsm8k (8-shot CoT, exact match)
# env: TASK N METHODS MAPPER_CB OUT
import os, json, re, string, random, collections, zipfile, glob, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

E = os.environ.get
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
ROOT = os.path.expanduser("~/kvbase")
MODELS = os.path.expanduser("~/kvswitch/models")
TASK, NS = E("TASK", "quality"), int(E("N", 200))
METHODS = E("METHODS", "native,cb_official,cb_fused,kvbridge,dense_fused,dense_fp8").split(",")
CB = E("MAPPER_CB", f"{ROOT}/mappers/cachebridge_17to4")
OUT = E("OUT", f"{ROOT}/results/quality_{TASK}.json")
dev = "cuda"
LS, LT, H, D = 28, 36, 8, 128

tok = AutoTokenizer.from_pretrained(f"{MODELS}/Qwen3-4B")
S = AutoModelForCausalLM.from_pretrained(f"{MODELS}/Qwen3-1.7B", torch_dtype=torch.bfloat16).to(dev).eval()
T = AutoModelForCausalLM.from_pretrained(f"{MODELS}/Qwen3-4B", torch_dtype=torch.bfloat16).to(dev).eval()
TR = {}
if {"cb_official"} & set(METHODS):
    from cachebridge.mapper import PageMapper
    TR["cb_official"] = PageMapper(CB, method="cachebridge", source_rope_theta=1e6, target_rope_theta=1e6, device=dev)
if {"cb_fused"} & set(METHODS):
    from fastmap import FastHeadLocal
    TR["cb_fused"] = FastHeadLocal(CB)
if {"cb_cublas32"} & set(METHODS):
    from fastmap import CublasHeadLocalF32
    TR["cb_cublas32"] = CublasHeadLocalF32(CB)
if {"cb_tf32x3"} & set(METHODS):
    from fastmap import FastHeadLocalF32
    TR["cb_tf32x3"] = FastHeadLocalF32(CB, prec="tf32x3", BM=16, num_warps=4)
if {"kvbridge", "dense_fused", "dense_fp8"} & set(METHODS):
    from kvbridge import Mapper as KBMapper, transfer as kb_transfer
    kb = KBMapper.load(f"{ROOT}/mappers/kvbridge_17to4", device=dev)
    from fastmap import FastDense
    if "dense_fused" in METHODS:
        TR["dense_fused"] = FastDense(kb)
    if "dense_fp8" in METHODS:
        TR["dense_fp8"] = FastDense(kb, fp8=True)


def chat_split(user_hist, user_query):
    h = f"<|im_start|>user\n{user_hist}"
    q = f"{user_query}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return tok(h, add_special_tokens=False).input_ids, tok(q, add_special_tokens=False).input_ids


# ---------------------------------------------------------------- tasks
def norm(s):
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def f1(p, golds):
    best = 0.0
    for g in golds:
        a, b = norm(p).split(), norm(g).split()
        c = sum((collections.Counter(a) & collections.Counter(b)).values())
        if c:
            pr, rc = c / len(a), c / len(b); best = max(best, 2 * pr * rc / (pr + rc))
    return best


samples = []
if TASK == "quality":
    from datasets import load_dataset
    q = load_dataset("emozilla/quality", split="validation")
    arts = {}
    for r in q:
        arts.setdefault(r["article"], []).append(r)
    docs = [v for v in arts.values() if len(v) >= 4]
    random.Random(0).shuffle(docs)
    docs = docs[24:]  # the first 24 (same shuffle) are CacheBridge V-calibration documents
    for d in docs:
        for r in d:
            opts = "\n".join(f"{c}. {o}" for c, o in zip("ABCD", r["options"]))
            hi, qi = chat_split(f"Read the following article.\n\n{d[0]['article']}",
                                f"\n\nQuestion: {r['question']}\n{opts}\nAnswer with the letter of the correct option.")
            samples.append(dict(h=hi[:12000], q=qi, gold=int(r["answer"]), kind="mc"))
    samples = samples[:NS]
elif TASK == "hotpotqa_e":
    from huggingface_hub import hf_hub_download
    z = zipfile.ZipFile(hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset"))
    rows = [json.loads(l) for l in z.open("data/hotpotqa_e.jsonl")]
    random.Random(0).shuffle(rows)
    inst = "Answer the question based on the given passages. Only give me the answer and do not output any other words."
    for r in rows[:NS]:
        hi, qi = chat_split(f"{inst}\n\nThe following are given passages.\n{r['context']}",
                            f"\n\n{inst}\n\nQuestion: {r['input']}\nAnswer:")
        samples.append(dict(h=hi[:12000], q=qi, gold=r["answers"], kind="gen", mx=32))
elif TASK == "gsm8k":
    from datasets import load_dataset
    tr = load_dataset("openai/gsm8k", "main", split="train")
    te = load_dataset("openai/gsm8k", "main", split="test")
    shots = "".join(f"Question: {tr[i]['question']}\nAnswer: {tr[i]['answer'].replace('####', 'The answer is')}\n\n" for i in range(8))
    for r in list(te)[:NS]:
        hi, qi = chat_split(f"Solve the math problems.\n\n{shots}", f"Question: {r['question']}\nAnswer:")
        samples.append(dict(h=hi, q=qi, gold=r["answer"].split("####")[-1].strip().replace(",", ""), kind="gsm", mx=256))


# ---------------------------------------------------------------- handoff
@torch.no_grad()
def target_cache(method, hist):
    if method == "native":
        c = DynamicCache(); T(torch.tensor([hist], device=dev), past_key_values=c, use_cache=True); return c
    sc = S(torch.tensor([hist], device=dev), use_cache=True).past_key_values
    if method == "kvbridge":
        return kb_transfer(sc, kb)
    sk = torch.stack([sc.layers[l].keys[0].transpose(0, 1) for l in range(LS)]).contiguous()  # [L,T,H,D]
    sv = torch.stack([sc.layers[l].values[0].transpose(0, 1) for l in range(LS)]).contiguous()
    n = sk.shape[1]
    if method == "cb_official":
        out = torch.cat([TR[method].map_page(torch.stack([sk[:, p:p + 2048], sv[:, p:p + 2048]]), position_start=p, output_device=dev)
                         for p in range(0, n, 2048)], 2)
    else:
        out = TR[method](sk, sv)
    c = DynamicCache()
    for l in range(LT):
        c.update(out[0, l].transpose(0, 1)[None].contiguous(), out[1, l].transpose(0, 1)[None].contiguous(), l)
    return c


LETTERS = [tok.encode(x, add_special_tokens=False)[0] for x in "ABCD"]
STOP = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}


@torch.no_grad()
def answer(c, s):
    lg = T(torch.tensor([s["q"]], device=dev), past_key_values=c, use_cache=True).logits[0, -1]
    if s["kind"] == "mc":
        return int(lg[LETTERS].argmax())
    outt, nxt = [], int(lg.argmax())
    for _ in range(s["mx"]):
        if nxt in STOP:
            break
        outt.append(nxt)
        nxt = int(T(torch.tensor([[nxt]], device=dev), past_key_values=c, use_cache=True).logits[0, -1].argmax())
    return tok.decode(outt, skip_special_tokens=True).strip()


def score(s, pred):
    if s["kind"] == "mc":
        return float(pred == s["gold"])
    if s["kind"] == "gen":
        return f1(pred, s["gold"])
    m = re.findall(r"answer is\s*\$?(-?[\d,]*\.?\d+)", pred) or re.findall(r"-?\d[\d,]*\.?\d*", pred)
    return float(bool(m) and m[0 if "answer is" in pred else -1].replace(",", "").rstrip(".") == s["gold"])


res = json.load(open(OUT)) if os.path.exists(OUT) else dict(task=TASK, samples=[])
done = len(res["samples"])
for i, s in enumerate(samples):
    if i < done:
        continue
    rec = dict(n_hist=len(s["h"]), m={})
    for m in METHODS:
        c = target_cache(m, s["h"])
        p = answer(c, s)
        rec["m"][m] = dict(pred=p if isinstance(p, (int, str)) else str(p), score=score(s, p))
        del c
    res["samples"].append(rec)
    if (i + 1) % 20 == 0 or i == len(samples) - 1:
        json.dump(res, open(OUT, "w"))
        print(TASK, i + 1, {m: round(float(np.mean([x["m"][m]["score"] for x in res["samples"]])), 4) for m in METHODS}, flush=True)
    torch.cuda.empty_cache()
json.dump(res, open(OUT, "w"))
print("QUALDONE")
