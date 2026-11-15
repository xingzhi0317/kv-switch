# Re-score saved GSM8K predictions with the fixed answer extractor (tasks.gsm_extract).
# The old scorer failed on "The answer is **18**." and fell back to the FIRST number in the output, which
# penalised the native target (it follows the 8-shot "The answer is" format with markdown bold).
# Originals are kept as *.orig.json. Gold answers: openai/gsm8k test[:N], same order as eval_quality.py.
import os, re, json, glob, shutil
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from datasets import load_dataset
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks
ns = {"gsm_extract": tasks.gsm_extract, "gsm_eq": tasks.gsm_eq}
te = load_dataset("openai/gsm8k", "main", split="test")
gold = [r["answer"].split("####")[-1].strip().replace(",", "") for r in te]
H = os.path.expanduser("~")
files = [f for f in glob.glob(f"{H}/kvcodesign/results/quality_gsm8k*.json") + glob.glob(f"{H}/kvbase/results/quality_gsm8k*.json")
         if not f.endswith(".orig.json")]
for f in sorted(files):
    d = json.load(open(f))
    if d.get("rescored"):
        continue
    if not os.path.exists(f[:-5] + ".orig.json"):
        shutil.copy(f, f[:-5] + ".orig.json")
    old, new = {}, {}
    for i, s in enumerate(d["samples"]):
        for m, r in s["m"].items():
            old.setdefault(m, []).append(r["score"])
            r["score_old"] = r["score"]
            r["score"] = float(ns["gsm_eq"](ns["gsm_extract"](str(r["pred"])), gold[i]))
            new.setdefault(m, []).append(r["score"])
    d["rescored"] = True
    json.dump(d, open(f, "w"))
    print(os.path.basename(f), {m: (round(sum(old[m]) / len(old[m]), 3), round(sum(new[m]) / len(new[m]), 3)) for m in old})
