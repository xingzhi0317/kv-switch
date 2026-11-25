# Stage 1 of the KL-trained translator (KV-Lingo / Hybrid-KV): closed-form ridge on CHAT prefixes (the training
# conversations' prompts, kl_data.py), ~0.55M prefix tokens (Hybrid-KV Part 1), every position; per target layer the
# nu = 3 source layers with the highest single-source R^2, all heads concatenated, lambda 0.01 -- official kvbridge
# functions (rank_source_layers, ridge_streaming).  Output: ~/kvbase/mappers/kvbridge3c_{DIR} (kvbridge format).
# usage: DIR=17to4 CUDA_VISIBLE_DEVICES=a python fit_kl_init.py
import os, sys, gc, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
sys.path.insert(0, os.path.expanduser("~/kvbase/kvbridge"))
sys.path.insert(0, os.path.expanduser("~/kvcodesign"))
from kvbridge.rope import strip_rope, get_rope_theta
from kvbridge.cache_compat import cache_layer_kv
from kvbridge.fit import rank_source_layers, ridge_streaming
from kvbridge import Mapper
import kl_data

DIR = os.environ.get("DIR", "17to4")
SRC, TGT = {"17to4": ("Qwen3-1.7B", "Qwen3-4B"), "4to17": ("Qwen3-4B", "Qwen3-1.7B")}[DIR]
M = os.path.expanduser("~/kvswitch/models")
K, LAM, NTOK, dev = 3, 0.01, 550000, "cuda:0"
t0 = time.time()
log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{(time.time() - t0) / 60:.1f} min]", *a, flush=True)
torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained(f"{M}/Qwen3-4B")
train, _ = kl_data.load(tok)
ids_all, n = [], 0
for x in train:
    ids_all.append(torch.tensor([x["p"]])); n += len(x["p"])
    if n >= NTOK:
        break
NPOS = n
log("init prefixes", len(ids_all), "tokens", NPOS)


def capture(name):   # as kvbridge.fit.collect_kv, every position (fp16 [N, H, D] per layer; keys in content space)
    m = AutoModelForCausalLM.from_pretrained(f"{M}/{name}", torch_dtype=torch.float16).to(dev).eval()
    c = m.config
    L, H, D = c.num_hidden_layers, c.num_key_value_heads, c.head_dim
    theta = get_rope_theta(c)
    keys = [torch.empty((NPOS, H, D), dtype=torch.float16) for _ in range(L)]
    vals = [torch.empty((NPOS, H, D), dtype=torch.float16) for _ in range(L)]
    n0 = 0
    for ids in ids_all:
        ids = ids.to(dev); pos = torch.arange(ids.shape[1], device=dev); n1 = n0 + ids.shape[1]
        cache = m.model(ids, use_cache=True).past_key_values
        for l in range(L):
            k, v = cache_layer_kv(cache, l)
            keys[l][n0:n1] = strip_rope(k[0].float(), pos, theta).transpose(0, 1).half().cpu()
            vals[l][n0:n1] = v[0].transpose(0, 1).half().cpu()
        n0 = n1
        del cache
    del m
    gc.collect(); torch.cuda.empty_cache()
    log(name, "captured", NPOS, "positions", L, "layers")
    return keys, vals


sk, sv = capture(SRC)
tk, tv = capture(TGT)
rank_idx = torch.linspace(0, NPOS - 1, min(16384, NPOS)).long()
layers, r2s = [], []
for tl in range(len(tk)):
    order, _ = rank_source_layers(sk, sv, tk[tl], tv[tl], rank_idx, lam=LAM, device=dev)
    chosen = order[:K]
    W_k, mu_xk, mu_yk, r2k = ridge_streaming(sk, tk[tl], chosen, lam=LAM, device=dev)
    W_v, mu_xv, mu_yv, r2v = ridge_streaming(sv, tv[tl], chosen, lam=LAM, device=dev)
    layers.append({"source_layers": [int(s) for s in chosen], "W_k": W_k, "mu_xk": mu_xk, "mu_yk": mu_yk,
                   "W_v": W_v, "mu_xv": mu_xv, "mu_yv": mu_yv})
    r2s.append((r2k, r2v))
mk, mv = sum(x[0] for x in r2s) / len(r2s), sum(x[1] for x in r2s) / len(r2s)
cs, ct = AutoConfig.from_pretrained(f"{M}/{SRC}"), AutoConfig.from_pretrained(f"{M}/{TGT}")
meta = {"source": SRC, "target": TGT, "n_kv_heads": int(cs.num_key_value_heads), "head_dim": int(cs.head_dim),
        "rope_theta_source": get_rope_theta(cs), "rope_theta_target": get_rope_theta(ct), "top_k": K, "lam": LAM,
        "mean_r2_k": mk, "mean_r2_v": mv,
        "calibration": {"data": "Nemotron chat prompts (kl_data.py)", "n_prefixes": len(ids_all), "token_stride": 1, "n_observations": NPOS},
        "method": "stage-1 closed form for the KL-trained translator (KV-Lingo / Hybrid-KV), chat prefixes (fit_kl_init.py)"}
Mapper(layers, meta).save(os.path.expanduser(f"~/kvbase/mappers/kvbridge3c_{DIR}"))
log("KLINITDONE R2 K", round(mk, 3), "V", round(mv, 3))
