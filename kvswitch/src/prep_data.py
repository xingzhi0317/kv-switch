# Download / preprocess eval datasets into ~/kvswitch/data/*.json
import os, json, zipfile, glob
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from datasets import load_dataset
from huggingface_hub import hf_hub_download
D = os.path.expanduser("~/kvswitch/data"); os.makedirs(D, exist_ok=True)

def dump(name, obj):
    json.dump(obj, open(f"{D}/{name}.json", "w")); print(name, len(obj), flush=True)

if not os.path.exists(f"{D}/coqa.json"):
    ds = load_dataset("stanfordnlp/coqa", split="validation")
    dump("coqa", [dict(story=r["story"], qs=r["questions"], ans=r["answers"]["input_text"]) for r in ds])
if not os.path.exists(f"{D}/gsm8k.json"):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    tr = load_dataset("openai/gsm8k", "main", split="train")
    dump("gsm8k", [dict(q=r["question"], a=r["answer"]) for r in ds])
    dump("gsm8k_train", [dict(q=r["question"], a=r["answer"]) for r in tr.select(range(2000))])
if not os.path.exists(f"{D}/longbench.json"):
    p = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    z = zipfile.ZipFile(p); out = {}
    for t in ["hotpotqa", "2wikimqa", "multifieldqa_en", "qasper"]:
        rows = [json.loads(l) for l in z.open(f"data/{t}.jsonl")]
        out[t] = [dict(ctx=r["context"], q=r["input"], ans=r["answers"], len=r["length"]) for r in rows]
    dump("longbench", out)
if not os.path.exists(f"{D}/ultrachat.json"):
    fs = sorted(glob.glob("/data2/datasets/ultrachat_200k/data/*test_sft*")) or sorted(glob.glob("/data2/datasets/ultrachat_200k/**/*.parquet", recursive=True))
    print("ultrachat files", fs[:3])
    ds = load_dataset("parquet", data_files=fs[:1], split="train")
    dump("ultrachat", [r["messages"] for r in ds.select(range(3000))])
    fs2 = sorted(glob.glob("/data2/datasets/ultrachat_200k/data/*train_sft*"))
    ds2 = load_dataset("parquet", data_files=fs2[:1], split="train")
    dump("ultrachat_train", [r["messages"] for r in ds2.select(range(20000))])
print("DATADONE")
