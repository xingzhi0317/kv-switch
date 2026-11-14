# Held-out documents for the "attention readout error" metric.
# FineWeb-Edu texts #500.. (the calibration set is texts #0..499 of the same stream, same >3000-char filter).
# For each doc (first SEQ tokens) store, all in content space (RoPE removed):
#   source K,V  [LS, n, H, D]   target K,V [LT, n, H, D]   target Q at query positions [LT, nq, HQ, D]
# env: NDOC(16) SEQ(1024) QSTART(256) QSTRIDE(8) DEVICE(cpu|cuda) THREADS OUT
import os, torch
E = os.environ.get
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

NDOC, SEQ, QS, QST = int(E("NDOC", 16)), int(E("SEQ", 1024)), int(E("QSTART", 256)), int(E("QSTRIDE", 8))
dev = E("DEVICE", "cpu")
torch.set_num_threads(int(E("THREADS", 32)))
OUT = E("OUT", os.path.expanduser("~/kvcodesign/data/val_docs.pt"))
MODELS = os.path.expanduser("~/kvswitch/models")
dt = torch.float32 if dev == "cpu" else torch.bfloat16
log = lambda *a: print(*a, flush=True)

texts = []
for row in load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True):
    if len(row["text"]) > 3000:
        texts.append(row["text"])
        if len(texts) == 500 + 4 * NDOC:
            break
tok = AutoTokenizer.from_pretrained(f"{MODELS}/Qwen3-1.7B")
ids = [tok(t, return_tensors="pt", truncation=True, max_length=SEQ).input_ids for t in texts[500:]]
ids = [x for x in ids if x.shape[1] == SEQ][:NDOC]  # full-length docs only (query positions are fixed)
assert len(ids) == NDOC, len(ids)
qpos = torch.arange(QS, SEQ, QST)


def run(name, want_q):
    m = AutoModelForCausalLM.from_pretrained(f"{MODELS}/{name}", torch_dtype=dt).to(dev).eval()
    L = m.config.num_hidden_layers
    cap = {}
    hooks = []
    for l, layer in enumerate(m.model.layers):
        a = layer.self_attn
        hooks.append(a.k_norm.register_forward_hook(lambda mod, i, o, l=l: cap.__setitem__(("k", l), o[0].to(torch.float16).cpu())))
        hooks.append(a.v_proj.register_forward_hook(lambda mod, i, o, l=l: cap.__setitem__(("v", l), o[0].to(torch.float16).cpu())))
        if want_q:
            hooks.append(a.q_norm.register_forward_hook(lambda mod, i, o, l=l: cap.__setitem__(("q", l), o[0][qpos].to(torch.float16).cpu())))
    res = []
    for i, x in enumerate(ids):
        cap.clear()
        with torch.inference_mode():
            m(x.to(dev), use_cache=False)
        n = x.shape[1]
        d = dict(k=torch.stack([cap[("k", l)] for l in range(L)]),                       # [L, n, H, D]
                 v=torch.stack([cap[("v", l)].view(n, -1, 128) for l in range(L)]))
        if want_q:
            d["q"] = torch.stack([cap[("q", l)] for l in range(L)])                       # [L, nq, HQ, D]
        res.append(d)
        log(name, "doc", i, n)
    for h in hooks:
        h.remove()
    del m
    return res


src = run("Qwen3-1.7B", False)
tgt = run("Qwen3-4B", True)
docs = [dict(ids=ids[i][0], sk=src[i]["k"], sv=src[i]["v"], tk=tgt[i]["k"], tv=tgt[i]["v"], tq=tgt[i]["q"], qpos=qpos)
        for i in range(NDOC)]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.save(docs, OUT)
log("COLLECTDONE", OUT)
