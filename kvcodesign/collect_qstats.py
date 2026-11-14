# Readout-aware target metric, from CALIBRATION docs (FineWeb-Edu #0..NDOC-1, disjoint from the val docs #500+).
# K: score error for a key error e is q^T R_delta e. Averaged over relative offsets delta, the metric becomes
#    per-frequency isotropic: weight_f = (E[q_f^2] + E[q_{f+64}^2]) / 2, per (layer, kv head) pooled over its GQA group.
# V: output error is W_o,j e for each query head j sharing the kv head => metric sum_j W_o,j^T W_o,j (128x128).
# env: NDOC(32) THREADS OUT
import os, torch
E = os.environ.get
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

NDOC = int(E("NDOC", 32))
torch.set_num_threads(int(E("THREADS", 32)))
OUT = E("OUT", os.path.expanduser("~/kvcodesign/data/qstats_4b.pt"))
MODELS = os.path.expanduser("~/kvswitch/models")
texts = []
for row in load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True):
    if len(row["text"]) > 3000:
        texts.append(row["text"])
        if len(texts) == NDOC:
            break
tok = AutoTokenizer.from_pretrained(f"{MODELS}/Qwen3-4B")
m = AutoModelForCausalLM.from_pretrained(f"{MODELS}/Qwen3-4B", torch_dtype=torch.float32).eval()
L, HQ, H, D = 36, 32, 8, 128
q2 = torch.zeros(L, HQ, D, dtype=torch.float64); cnt = [0]
for l, layer in enumerate(m.model.layers):
    layer.self_attn.q_norm.register_forward_hook(lambda mod, i, o, l=l: (q2[l].add_(o[0].double().square().sum(0)), None)[1])
for i, t in enumerate(texts):
    ids = tok(t, return_tensors="pt", truncation=True, max_length=1024).input_ids
    with torch.inference_mode():
        m(ids, use_cache=False)
    cnt[0] += ids.shape[1]
    print("doc", i, flush=True)
q2 /= cnt[0]
qg = q2.view(L, H, HQ // H, D).mean(2)                      # pool GQA group
wk = 0.5 * (qg[..., :D // 2] + qg[..., D // 2:])            # per-frequency isotropic
wk = torch.cat([wk, wk], -1)                                # [L, H, D]
Mv = torch.zeros(L, H, D, D, dtype=torch.float64)
for l, layer in enumerate(m.model.layers):
    Wo = layer.self_attn.o_proj.weight.double().view(-1, HQ, D)   # [hidden, HQ, D]
    for h in range(H):
        for j in range(h * HQ // H, (h + 1) * HQ // H):
            Mv[l, h] += Wo[:, j].T @ Wo[:, j]
torch.save(dict(wk=wk.float(), Mv=Mv.float(), q2=q2.float()), OUT)
print("QSTATSDONE", flush=True)
