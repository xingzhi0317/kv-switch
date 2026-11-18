# Fit the kvbridge mapper (arXiv 2608.03893) with the OFFICIAL code (kvbridge.fit.fit_mapper, default protocol:
# 500 FineWeb-Edu docs x 1024 tokens, stride 4, lambda 0.01) for a direction that has no released weights.
# Only the model paths and the output path are ours.  top_k=12 is the value the authors selected for 1.7B->4B;
# their README says to sweep it for a new pair -- not swept here (noted in the report).
# env: SRC TGT TOPK OUT
import os, sys, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, os.path.expanduser("~/kvbase/kvbridge"))
import transformers
# compatibility shim (ours): the official code passes dtype=..., which transformers 4.54 (our env) only accepts as
# torch_dtype=...; nothing else is changed.
_fp = transformers.AutoModelForCausalLM.from_pretrained.__func__
def _from_pretrained(cls, *a, **k):
    if "dtype" in k:
        k["torch_dtype"] = k.pop("dtype")
    return _fp(cls, *a, **k)
transformers.AutoModelForCausalLM.from_pretrained = classmethod(_from_pretrained)
from kvbridge.fit import fit_mapper
M = os.path.expanduser("~/kvswitch/models")
SRC, TGT = os.environ.get("SRC", "Qwen3-4B"), os.environ.get("TGT", "Qwen3-1.7B")
OUT = os.environ.get("OUT", os.path.expanduser("~/kvbase/mappers/kvbridge_4to17"))
t0 = time.time()
m = fit_mapper(f"{M}/{SRC}", f"{M}/{TGT}", top_k=int(os.environ.get("TOPK", 12)))
m.save(OUT)
print("FITKBDONE", OUT, round(time.time() - t0), "s", flush=True)
