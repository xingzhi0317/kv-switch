# RaReCache (arXiv 2610.11358) on the paper's own pair Qwen3-0.6B -> Qwen3-14B, GSM8K: the task-specific map and the
# rank-disagreement subspace, as the paper describes them (Sec 3.1-3.2, 4.1, App C.1):
#   calibration : GSM8K TRAIN prompts formatted as the evaluation prompts (tasks.gsm8k_rr), positions at stride 4 (offset
#                 rotating per prompt) until 128k positions; keys RoPE-stripped
#   map         : per target layer and K/V, ridge over the concatenated source layers (all heads), the K=8 source layers
#                 with the highest single-source R^2, lambda 0.01 -- official kvbridge functions (rank_source_layers,
#                 ridge_streaming), saved with kvbridge.Mapper (-> eval_quality "kvbridge" / "rare*" for DIR=06to14)
#   selector    : per target layer and K/V, top-128 eigenvectors of the covariance of the map's predictions on the
#                 calibration positions (rank r = 128) -> mappers/rare_vr_06to14.pt (format of rare_fit.py)
# usage: CUDA_VISIBLE_DEVICES=a,b python fit_rr.py
import os, gc, json, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from kvbridge.rope import strip_rope, get_rope_theta
from kvbridge.cache_compat import cache_layer_kv
from kvbridge.fit import rank_source_layers, ridge_streaming
from kvbridge import Mapper
import sys
sys.path.insert(0, os.path.expanduser("~/kvcodesign"))
from tasks import chat_split, gsm_rr_text

SRC, TGT, DIR = "Qwen3-0.6B-ms", "Qwen3-14B-ms", "06to14"
M = os.path.expanduser("~/kvswitch/models")
K, LAM, STRIDE, NPOS_MAX, R, dev = 8, 0.01, 4, 128000, 128, "cuda:0"
t0 = time.time()
log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{(time.time() - t0) / 60:.1f} min]", *a, flush=True)
torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained(f"{M}/{SRC}")
enc = lambda x: tok(x, add_special_tokens=False).input_ids
ids_all, keep_all, n = [], [], 0
for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="train")):
    h, q = chat_split(enc, gsm_rr_text(r["question"]), "")
    ids = torch.tensor([h + q])
    keep = torch.arange(i % STRIDE, ids.shape[1], STRIDE)
    ids_all.append(ids); keep_all.append(keep); n += len(keep)
    if n >= NPOS_MAX:
        break
NPOS = n
log("calibration prompts", len(ids_all), "positions", NPOS)


def capture(name):   # as kvbridge.fit.collect_kv (fp16 traces [N, H, D] per layer; keys content space)
    m = AutoModelForCausalLM.from_pretrained(f"{M}/{name}", torch_dtype=torch.float16, device_map="auto").eval()
    c = m.config
    L, H, D = c.num_hidden_layers, c.num_key_value_heads, c.head_dim
    theta = get_rope_theta(c)
    keys = [torch.empty((NPOS, H, D), dtype=torch.float16) for _ in range(L)]
    vals = [torch.empty((NPOS, H, D), dtype=torch.float16) for _ in range(L)]
    n0 = 0
    for ids, keep in zip(ids_all, keep_all):
        ids = ids.to(dev); kp = keep.to(dev); pos = torch.arange(ids.shape[1], device=dev)
        n1 = n0 + len(keep)
        cache = m(ids, use_cache=True).past_key_values
        for l in range(L):
            k, v = cache_layer_kv(cache, l)
            keys[l][n0:n1] = strip_rope(k[0].to(dev).float(), pos, theta)[:, kp].transpose(0, 1).half().cpu()
            vals[l][n0:n1] = v[0].to(dev).float()[:, kp].transpose(0, 1).half().cpu()
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
    r2s.append({"k": r2k, "v": r2v})
log("map fitted; mean in-sample R2 K", round(sum(x["k"] for x in r2s) / len(r2s), 3), "V", round(sum(x["v"] for x in r2s) / len(r2s), 3))
cs, ct = AutoConfig.from_pretrained(f"{M}/{SRC}"), AutoConfig.from_pretrained(f"{M}/{TGT}")
meta = {"source": SRC, "target": TGT, "n_kv_heads": int(cs.num_key_value_heads), "head_dim": int(cs.head_dim),
        "rope_theta_source": get_rope_theta(cs), "rope_theta_target": get_rope_theta(ct), "top_k": K, "lam": LAM,
        "mean_r2_k": sum(x["k"] for x in r2s) / len(r2s), "mean_r2_v": sum(x["v"] for x in r2s) / len(r2s),
        "calibration": {"task": "gsm8k train (gsm8k_rr format)", "n_prompts": len(ids_all), "token_stride": STRIDE, "n_observations": NPOS},
        "method": "RaReCache map (arXiv 2610.11358): per-layer ridge, K=8 by single-source R^2, task calibration data (fit_rr.py)"}
Mapper(layers, meta).save(os.path.expanduser(f"~/kvbase/mappers/kvbridge_{DIR}"))
# rank-disagreement subspace: top-R eigenvectors of the covariance of the map's predictions on the calibration positions
sub = torch.linspace(0, NPOS - 1, min(32768, NPOS)).long()
VR = {"Vk": [], "Vv": [], "myk": [], "myv": []}
for e in layers:
    for kind, src in (("k", sk), ("v", sv)):
        x = torch.cat([src[s][sub].reshape(len(sub), -1).to(dev).float() for s in e["source_layers"]], -1)
        y = (x - e[f"mu_x{kind}"].to(dev)) @ e[f"W_{kind}"].to(dev).float() + e[f"mu_y{kind}"].to(dev)
        my = y.mean(0); yc = (y - my).double()
        ev, U = torch.linalg.eigh(yc.T @ yc / len(sub))
        VR["V" + kind].append(U[:, -R:].flip(1).float().cpu()); VR["my" + kind].append(my.cpu())
VR = {k: torch.stack(v) for k, v in VR.items()}
torch.save(VR, os.path.expanduser(f"~/kvcodesign/mappers/rare_vr_{DIR}.pt"))
json.dump(dict(meta=meta, r2=r2s, minutes=(time.time() - t0) / 60), open(os.path.expanduser(f"~/kvcodesign/results/rr_fit_{DIR}.json"), "w"), indent=1)
log("RRFITDONE", {k: tuple(v.shape) for k, v in VR.items()})
