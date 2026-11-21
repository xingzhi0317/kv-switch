# CFM / kvbridge ridge mapper (arXiv 2608.03893, official code ~/kvbase/kvbridge) for a pair whose target does not fit on
# one 24 GB GPU (Qwen3-8B -> 14B).  kvbridge.fit.fit_mapper() loads each model with .to(device) on a single GPU, so its
# steps are run here one by one with the OFFICIAL functions:
#   capture : as kvbridge.fit.collect_kv (model in fp16 = fit_mapper's default, keys RoPE-stripped, positions i%4::4,
#             fp16 CPU traces [N, H, D] per layer); only difference: the target model is spread over two GPUs
#   ranking : kvbridge.fit.rank_source_layers on fit_mapper's subsample (linspace, 16384 positions), lam 0.01
#   fit     : kvbridge.fit.ridge_streaming, top_k=12 (kvbridge's default, = "original CFM ridge mapper" settings)
#   save    : kvbridge.Mapper(layers, meta).save()  -> ~/kvbase/mappers/kvbridge_{DIR}
# Fitting texts: the 500 FineWeb-Edu texts of exactly 1024 tokens (CacheBridge App. A: CFM and CacheBridge share them).
# usage: DIR=8to14 CUDA_VISIBLE_DEVICES=a,b python fit_cfm_big.py
import os, gc, json, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from kvbridge.rope import strip_rope, get_rope_theta
from kvbridge.cache_compat import cache_layer_kv
from kvbridge.fit import rank_source_layers, ridge_streaming
from kvbridge import Mapper

E = os.environ.get
DIR = E("DIR", "8to14")
SRC, TGT = {"8to14": ("Qwen3-8B-ms", "Qwen3-14B-ms")}[DIR]
M = os.path.expanduser("~/kvswitch/models")
TEXTS = os.path.expanduser("~/kvcodesign/data/fineweb_texts_full500.json")
OUT = os.path.expanduser(E("OUT", f"~/kvbase/mappers/kvbridge_{DIR}"))
TOPK, LAM, SEQ, STRIDE, dev = int(E("TOPK", 12)), float(E("LAM", 0.01)), 1024, 4, "cuda:0"   # LAM: ridge on the AVERAGED Gram (kvbridge convention)
t0 = time.time()
log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{(time.time() - t0) / 60:.1f} min]", *a, flush=True)
torch.set_grad_enabled(False)
texts = json.load(open(TEXTS))
tok = AutoTokenizer.from_pretrained(f"{M}/{SRC}")


def capture(name):   # kvbridge.fit.collect_kv, target spread over GPUs
    m = AutoModelForCausalLM.from_pretrained(f"{M}/{name}", torch_dtype=torch.float16, device_map="auto").eval()
    c = m.config
    L, H, D = c.num_hidden_layers, c.num_key_value_heads, c.head_dim
    theta = get_rope_theta(c)
    all_ids, all_keep = [], []
    for i, text in enumerate(texts):
        ids = tok(text, return_tensors="pt", truncation=True, max_length=SEQ).input_ids
        if ids.shape[1] < 32:
            continue
        all_ids.append(ids); all_keep.append(torch.arange(i % STRIDE, ids.shape[1], STRIDE))
    n_total = sum(len(k) for k in all_keep)
    keys = [torch.empty((n_total, H, D), dtype=torch.float16) for _ in range(L)]
    vals = [torch.empty((n_total, H, D), dtype=torch.float16) for _ in range(L)]
    n0 = 0
    for ids, keep in zip(all_ids, all_keep):
        ids = ids.to(dev); keep = keep.to(dev); pos = torch.arange(ids.shape[1], device=dev)
        n1 = n0 + len(keep)
        cache = m(ids, use_cache=True).past_key_values
        for l in range(L):
            k, v = cache_layer_kv(cache, l)
            k, v = k[0].to(dev), v[0].to(dev)
            keys[l][n0:n1] = strip_rope(k.float(), pos, theta)[:, keep].transpose(0, 1).half().cpu()
            vals[l][n0:n1] = v.float()[:, keep].transpose(0, 1).half().cpu()
        n0 = n1
        del cache
    del m
    gc.collect(); torch.cuda.empty_cache()
    log(name, "captured", n_total, "observations; GPU MB still allocated", [torch.cuda.memory_allocated(i) >> 20 for i in range(torch.cuda.device_count())])
    return keys, vals


src_keys, src_vals = capture(SRC)
tgt_keys, tgt_vals = capture(TGT)
n_obs = src_keys[0].shape[0]
rank_idx = torch.linspace(0, n_obs - 1, min(16384, n_obs)).long()
# TOPKS="1,2,4,8": one ranking per target layer, one mapper per k (OUT with "{k}" in it, e.g. .../kvbridge{k}_8to14)
TOPKS = [int(x) for x in E("TOPKS", str(TOPK)).split(",")]
orders = [rank_source_layers(src_keys, src_vals, tgt_keys[tl], tgt_vals[tl], rank_idx, lam=LAM, device=dev)[0]
          for tl in range(len(tgt_keys))]
log("ranked", len(orders), "target layers")
cs, ct = AutoConfig.from_pretrained(f"{M}/{SRC}"), AutoConfig.from_pretrained(f"{M}/{TGT}")
for K in TOPKS:
    layers, r2s = [], []
    for tl in range(len(tgt_keys)):
        chosen = orders[tl][:K]
        W_k, mu_xk, mu_yk, r2k = ridge_streaming(src_keys, tgt_keys[tl], chosen, lam=LAM, device=dev)
        W_v, mu_xv, mu_yv, r2v = ridge_streaming(src_vals, tgt_vals[tl], chosen, lam=LAM, device=dev)
        layers.append({"source_layers": [int(s) for s in chosen], "W_k": W_k, "mu_xk": mu_xk, "mu_yk": mu_yk,
                       "W_v": W_v, "mu_xv": mu_xv, "mu_yv": mu_yv})
        r2s.append({"k": r2k, "v": r2v})
    meta = {"source": SRC, "target": TGT, "n_kv_heads": int(cs.num_key_value_heads), "head_dim": int(cs.head_dim),
            "rope_theta_source": get_rope_theta(cs), "rope_theta_target": get_rope_theta(ct), "top_k": K, "lam": LAM,
            "mean_r2_k": sum(x["k"] for x in r2s) / len(r2s), "mean_r2_v": sum(x["v"] for x in r2s) / len(r2s),
            "calibration": {"n_texts": len(texts), "seq_len": SEQ, "token_stride": STRIDE, "n_observations": int(n_obs)},
            "method": "arXiv:2608.03893 closed-form per-layer ridge, content-space keys (official functions, fit_cfm_big.py)"}
    out = OUT.replace("{k}", str(K))
    Mapper(layers, meta).save(out)
    log("k", K, "saved", out, "in-sample mean R^2 K", round(meta["mean_r2_k"], 3), "V", round(meta["mean_r2_v"], 3))
log("CFMDONE")
