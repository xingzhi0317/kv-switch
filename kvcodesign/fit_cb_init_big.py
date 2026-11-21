# CacheBridge initial (uncalibrated) head-local mapper for a large pair, e.g. Qwen3-8B -> Qwen3-14B (paper Table 2/25:
# "w/o calibration" retains 62.12% on QuALITY for this pair).  Same recipe as fit_cb4.py's initial fit:
#   support : per target layer, the K (=8 for the paper's Qwen3 pairs) source layers with the highest head-local R^2
#             (mean of K and V, same-index head h -> h, ridge 0.01, 32768 random observations)
#   fit     : official cachebridge.offline.initialization.fit_layer (ridge 0.01, fp64 statistics)
# Data: kvbridge's pipeline (FineWeb-Edu sample-10BT, 500 texts > 3000 chars, 1024 tokens, every 4th position;
# keys RoPE-stripped), the same text file head_corr.py uses.  No value calibration.
# Output: ~/kvbase/mappers/cachebridge_{DIR}_init/layer_NNN.safetensors + fit_report.json (PageMapper format)
# usage: DIR=8to14 CUDA_VISIBLE_DEVICES=a,b python fit_cb_init_big.py      (the target model may span both GPUs)
import os, gc, json, time, torch
from safetensors.torch import save_file, load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvbridge.rope import strip_rope, get_rope_theta
from kvbridge.cache_compat import cache_layer_kv
from cachebridge.offline.initialization import fit_layer

E = os.environ.get
DIR = E("DIR", "8to14")
SRC, TGT = {"8to14": ("Qwen3-8B-ms", "Qwen3-14B-ms"), "17to4": ("Qwen3-1.7B", "Qwen3-4B")}[DIR]
K, SEQ, STRIDE = int(E("K", 8)), 1024, 4
M = os.path.expanduser("~/kvswitch/models")
FULL = E("FULL", "1") == "1"   # paper: 500 sequences of exactly 1024 tokens -> 128,000 positions at stride 4
TEXTS = os.path.expanduser("~/kvcodesign/data/" + ("fineweb_texts_full500.json" if FULL else "fineweb_texts_500.json"))
SUB = int(E("SUB", 0))          # positions used to rank source layers (0 = all, as in the paper)
OUT = E("OUT", os.path.expanduser(f"~/kvbase/mappers/cachebridge_{DIR}_init" + ("2" if FULL else "")))
dev = "cuda:0"
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
torch.set_grad_enabled(False)

tok = AutoTokenizer.from_pretrained(f"{M}/{SRC}")
if not os.path.exists(TEXTS):   # FineWeb-Edu sample-10BT in stream order, first 500 texts with >= 1024 tokens
    from datasets import load_dataset
    keep = []
    for r in load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True):
        if len(r["text"]) > 3000 and len(tok(r["text"], truncation=True, max_length=SEQ).input_ids) == SEQ:
            keep.append(r["text"])
            if len(keep) == 500:
                break
    json.dump(keep, open(TEXTS, "w"))
texts = json.load(open(TEXTS))
ids_all = [tok(t, return_tensors="pt", truncation=True, max_length=SEQ).input_ids for t in texts]
keep_all = [torch.arange(i % STRIDE, x.shape[1], STRIDE) for i, x in enumerate(ids_all)]
NPOS = sum(len(k) for k in keep_all)


def collect(name):
    """-> [2, L, N, H, D] fp16 CPU (K RoPE-stripped, V raw)."""
    m = AutoModelForCausalLM.from_pretrained(f"{M}/{name}", torch_dtype=torch.bfloat16, device_map="auto").eval()
    c = m.config
    L, H, D = c.num_hidden_layers, c.num_key_value_heads, getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)
    theta = get_rope_theta(c)
    KV = torch.empty(2, L, NPOS, H, D, dtype=torch.float16)
    n0 = 0
    for i, (ids, keep) in enumerate(zip(ids_all, keep_all)):
        cache = m(ids.to(dev), use_cache=True).past_key_values
        pos, kp = torch.arange(ids.shape[1], device=dev), keep.to(dev)
        n1 = n0 + len(keep)
        for l in range(L):
            k, v = cache_layer_kv(cache, l)
            KV[0, l, n0:n1] = strip_rope(k[0].to(dev).float(), pos, theta)[:, kp].transpose(0, 1).half().cpu()
            KV[1, l, n0:n1] = v[0].to(dev).float()[:, kp].transpose(0, 1).half().cpu()
        n0 = n1
        del cache
        if (i + 1) % 100 == 0:
            log(name, i + 1, "texts")
    del m, k, v
    gc.collect(); torch.cuda.empty_cache()
    log(name, "layers", L, "kv heads", H, "positions", NPOS,
        "GPU MB still allocated", [torch.cuda.memory_allocated(i) >> 20 for i in range(torch.cuda.device_count())])
    return KV


src = collect(SRC)
tgt = collect(TGT)
_, LS, N, H, D = src.shape
LT = tgt.shape[1]
sub = torch.randperm(N, generator=torch.Generator().manual_seed(0))[:SUB] if SUB else torch.arange(N)
log("positions", N, "used for layer ranking", len(sub))


def r2_headlocal(xs, ys):   # same as fit_cb4.py
    x = xs.to(dev).float(); y = ys.to(dev).float()
    x = x - x.mean(0); y = y - y.mean(0)
    xh, yh = x.transpose(0, 1), y.transpose(0, 1)
    G = xh.transpose(1, 2) @ xh + 0.01 * torch.eye(D, device=dev)
    W = torch.linalg.solve(G, xh.transpose(1, 2) @ yh)
    return (1 - (yh - xh @ W).square().sum((1, 2)) / yh.square().sum((1, 2))).mean().item()


R2 = torch.zeros(LT, LS)
ss = src[:, :, sub]; ts = tgt[:, :, sub]
for s in range(LS):
    for t in range(LT):
        R2[t, s] = 0.5 * (r2_headlocal(ss[0, s], ts[0, t]) + r2_headlocal(ss[1, s], ts[1, t]))
log("R2 done; best per target layer", [round(float(x), 3) for x in R2.max(1).values])
del ss, ts
support = [sorted(R2[t].topk(K).indices.tolist()) for t in range(LT)]
os.makedirs(OUT, exist_ok=True)
for t in range(LT):
    tensors, _ = fit_layer(src, tgt[:, t], selected=support[t], ridge=0.01, device=dev, dtype=torch.float64)
    save_file({k: v.contiguous() for k, v in tensors.items()}, f"{OUT}/layer_{t:03d}.safetensors")
    log("init layer", t, "support", support[t])
# energy-budget statistics on the initialization data (paper App. B: "the source-feature second moment and the initial
# mapper's value residual energy relative to re-prefill on the same 4,096 sampled initialization states", token averages
# per layer and KV head; M includes the intercept, the 1e-4 diagonal stabilisation is added by fit_cb6.py)
samp = torch.randperm(N, generator=torch.Generator().manual_seed(2))[:4096]
Ms, E0s = [], []
for t in range(LT):
    Lt = load_file(f"{OUT}/layer_{t:03d}.safetensors")
    sel = Lt["selected"].tolist()
    xv = torch.cat([src[1, s_][samp].transpose(0, 1).to(dev).float() for s_ in sel], -1)       # [H, 4096, K*D]
    vmap = torch.einsum("hnf,hfd->hnd", xv, Lt["v_weight"].to(dev).float()) + Lt["v_bias"].to(dev).float()[:, None]
    vn = tgt[1, t][samp].transpose(0, 1).to(dev).float()
    E0s.append((vmap - vn).double().square().sum(-1).mean(-1).cpu())                           # [H]
    Xb = torch.cat([xv, torch.ones_like(xv[..., :1])], -1).double()
    Ms.append(((Xb.transpose(1, 2) @ Xb) / Xb.shape[1]).float().cpu())                          # [H, p, p]
torch.save(dict(M=Ms, E0=E0s, n=len(samp)), OUT + "_stats.pt")
log("saved FineWeb budget statistics", OUT + "_stats.pt", "E0 layer 0 / last", round(float(E0s[0].mean()), 4), round(float(E0s[-1].mean()), 4))
json.dump(dict(support=support, K=K, R2=R2.tolist(), src=SRC, tgt=TGT, positions=N,
               note="official head-local initial fit only, no value calibration"), open(f"{OUT}/fit_report.json", "w"), indent=1)
log("FITDONE", OUT)
