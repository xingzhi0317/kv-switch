# Why is the CacheBridge paper's Qwen3-14B re-prefill SQuAD F1 only 21.78 while our Qwen3-4B gets ~56?
# Re-prefill only (no translation): the same 200 SQuAD v1.1 dev questions (tasks.py order), the paper's prompt
# wording (Table 5/6: "Context: <passage>\n\n" + short-answer instruction + question + "Answer:"), scored with the
# standard SQuAD token F1 (tasks.f1).  Settings that the paper does not state are varied:
#   chat   = Qwen chat template, thinking off (what we use)      raw = plain text, no chat template (lm-eval style)
#   max    = generation cap 32 / 256 tokens; stops at the model's EOS ids (<|im_end|>, <|endoftext|>)
#   line1  = the same output cut at its first newline (a common answer post-processing step)
# usage: MODEL=Qwen3-4B N=200 python squad_protocol.py
import os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random
from tasks import f1, em

E = os.environ.get
MODEL, N = E("MODEL", "Qwen3-4B"), int(E("N", 200))
path = os.path.expanduser(f"~/kvswitch/models/{MODEL}")
tok = AutoTokenizer.from_pretrained(path)
m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto").eval()
torch.set_grad_enabled(False)
stop = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
rows = list(load_dataset("rajpurkar/squad", split="validation"))
random.Random(0).shuffle(rows)
rows = rows[:N]


def prompt(r, mode):
    body = f"Context: {r['context']}\n\nAnswer the question briefly, using only the context.\nQuestion: {r['question']}\nAnswer:"
    if mode == "chat":
        return f"<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return body


def gen(text, mx):
    ids = torch.tensor([tok(text, add_special_tokens=False).input_ids], device=m.device)
    out = m.generate(ids, max_new_tokens=mx, do_sample=False, eos_token_id=list(stop), pad_token_id=list(stop)[0])
    new = out[0, ids.shape[1]:].tolist()
    return tok.decode([t for t in new if t not in stop]), len(new)


res, samples = {}, []
for mode in ["chat", "raw"]:
    for mx in [32, 256]:
        sc = {"f1": [], "em": [], "f1_line1": [], "len": [], "hit_cap": []}
        for i, r in enumerate(rows):
            golds = sorted(set(r["answers"]["text"]))
            p, n = gen(prompt(r, mode), mx)
            l1 = p.strip().split("\n")[0]
            sc["f1"].append(f1(p, golds)); sc["em"].append(em(p, golds)); sc["f1_line1"].append(f1(l1, golds))
            sc["len"].append(n); sc["hit_cap"].append(float(n >= mx))
            if i < 6:
                samples.append(dict(setting=f"{mode}{mx}", gold=golds, pred=p))
        k = f"{mode}{mx}"
        res[k] = {a: round(sum(v) / len(v), 4) for a, v in sc.items()}
        print(MODEL, k, res[k], flush=True)
os.makedirs("results", exist_ok=True)
json.dump(dict(model=MODEL, n=N, scores=res, samples=samples),
          open(f"results/squad_protocol_{MODEL}.json", "w"), indent=1, ensure_ascii=False)
print("SQDONE")
