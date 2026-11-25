# LLMLingua-2 sanity check: achieved compression ratio and whether the passkey survives compression, with the exact call
# eval_quality.py uses (official llmlingua package, microsoft/llmlingua-2-xlm-roberta-large-meetingbank, rate = kept
# fraction, force_tokens ["\n", "?"]).  Tasks: passkey_2k / passkey_8k (first 40 samples), QuALITY and HotpotQA (20).
# Output: results/lingua_check.json
import os, sys, json, re, torch
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from transformers import AutoTokenizer
from llmlingua import PromptCompressor
sys.path.insert(0, os.path.expanduser("~/kvcodesign"))
from tasks import build_samples

tok = AutoTokenizer.from_pretrained(os.path.expanduser("~/kvswitch/models/Qwen3-4B"))
enc = lambda t: tok(t, add_special_tokens=False).input_ids
L = PromptCompressor(model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank", use_llmlingua2=True, device_map="cuda")
PRE = enc("<|im_start|>user\n")
out = {}
for task, n in (("passkey_2k", 40), ("passkey_8k", 40), ("quality", 20), ("hotpotqa_e", 20)):
    S = build_samples(task, n, enc)
    rows = []
    for s in S:
        text = tok.decode(s["h"][len(PRE):])
        row = dict(n_tokens=len(s["h"]))
        if task.startswith("passkey"):
            gold = str(s["gold"]); row["gold_in_original"] = gold in text
        for rate in (0.25, 0.5):
            comp = L.compress_prompt(text, rate=rate, force_tokens=["\n", "?"])["compressed_prompt"]
            row[f"kept_{rate}"] = len(enc(comp)) / max(1, len(enc(text)))
            if task.startswith("passkey"):
                row[f"gold_kept_{rate}"] = gold in comp
                row[f"gold_kept_nospace_{rate}"] = gold in re.sub(r"\s", "", comp)
        rows.append(row)
    agg = {k: round(sum(float(r[k]) for r in rows) / len(rows), 3) for k in rows[0] if k != "n_tokens"}
    agg["mean_tokens"] = round(sum(r["n_tokens"] for r in rows) / len(rows))
    out[task] = agg
    print(task, agg, flush=True)
json.dump(out, open(os.path.expanduser("~/kvcodesign/results/lingua_check.json"), "w"), indent=1)
print("LINGUADONE")
