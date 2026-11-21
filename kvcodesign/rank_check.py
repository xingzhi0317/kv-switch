# Is the CacheBridge source-layer selection for Qwen3-8B -> 14B right?  Same data as fit_cb_init_big.py FULL=1
# (500 FineWeb-Edu texts x 1024 tokens, stride 4 = 128,000 positions).  Source-layer score (paper eq. 5):
# mean over heads of same-index head R^2, averaged over K and V; top-8 per target layer.  Three rankings:
#   sum : ridge 0.01 added to the summed X^T X           (what fit_cb_init_big.py used; = official fit_layer scale)
#   cov : ridge 0.01 added to the averaged X^T X / N      (kvbridge/CFM rank_source_layers scale)
#   kvb : the official kvbridge rank_source_layers itself on a strided subsample (checks the cov implementation)
# Prints the per-target-layer overlap with the support of the fitted mapper (INIT) and saves all scores.
# usage: CUDA_VISIBLE_DEVICES=a,b python rank_check.py
import os, gc, json, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvbridge.rope import strip_rope, get_rope_theta
from kvbridge.cache_compat import cache_layer_kv
from kvbridge.fit import rank_source_layers

E = os.environ.get
SRC, TGT = "Qwen3-8B-ms", "Qwen3-14B-ms"
M = os.path.expanduser("~/kvswitch/models")
TEXTS = os.path.expanduser("~/kvcodesign/data/fineweb_texts_full500.json")
INIT = os.path.expanduser(E("INIT", "~/kvbase/mappers/cachebridge_8to14_init2"))
dev, SEQ, STRIDE, K = "cuda:0", 1024, 4, 8
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
torch.set_grad_enabled(False)
texts = json.load(open(TEXTS))
tok = AutoTokenizer.from_pretrained(f"{M}/{SRC}")
ids_all = [tok(t, return_tensors="pt", truncation=True, max_length=SEQ).input_ids for t in texts]
keep_all = [torch.arange(i % STRIDE, x.shape[1], STRIDE) for i, x in enumerate(ids_all)]
NPOS = sum(len(k) for k in keep_all)


def collect(name):   # same as fit_cb_init_big.py
    m = AutoModelForCausalLM.from_pretrained(f"{M}/{name}", torch_dtype=torch.bfloat16, device_map="auto").eval()
    c = m.config
    L, H, D = c.num_hidden_layers, c.num_key_value_heads, c.head_dim
    theta = get_rope_theta(c)
    KV = torch.empty(2, L, NPOS, H, D, dtype=torch.float16)
    n0 = 0
    for ids, keep in zip(ids_all, keep_all):
        cache = m(ids.to(dev), use_cache=True).past_key_values
        pos, kp = torch.arange(ids.shape[1], device=dev), keep.to(dev)
        n1 = n0 + len(keep)
        for l in range(L):
            k, v = cache_layer_kv(cache, l)
            KV[0, l, n0:n1] = strip_rope(k[0].to(dev).float(), pos, theta)[:, kp].transpose(0, 1).half().cpu()
            KV[1, l, n0:n1] = v[0].to(dev).float()[:, kp].transpose(0, 1).half().cpu()
        n0 = n1
        del cache
    del m, k, v
    gc.collect(); torch.cuda.empty_cache()
    log(name, "collected", tuple(KV.shape))
    return KV


src = collect(SRC); tgt = collect(TGT)
_, LS, N, H, D = src.shape
LT = tgt.shape[1]


def r2_both(xs, ys):   # per-head ridge R^2 averaged over heads, ridge on the summed and on the averaged Gram
    x = xs.to(dev).float(); y = ys.to(dev).float()
    x = x - x.mean(0); y = y - y.mean(0)
    xh, yh = x.transpose(0, 1), y.transpose(0, 1)
    G = xh.transpose(1, 2) @ xh; C = xh.transpose(1, 2) @ yh; I = torch.eye(D, device=dev)
    out = []
    for lam in (0.01, 0.01 * N):   # 0.01 on X^T X  /  0.01 on X^T X / N
        W = torch.linalg.solve(G + lam * I, C)
        out.append((1 - (yh - xh @ W).square().sum((1, 2)) / yh.square().sum((1, 2))).mean().item())
    return out


Rs, Rc = torch.zeros(LT, LS), torch.zeros(LT, LS)
for t in range(LT):
    for s in range(LS):
        ks, kc = r2_both(src[0, s], tgt[0, t]); vs, vc = r2_both(src[1, s], tgt[1, t])
        Rs[t, s], Rc[t, s] = 0.5 * (ks + vs), 0.5 * (kc + vc)
log("scores done")
fit = json.load(open(f"{INIT}/fit_report.json"))["support"]
top = lambda R, t: sorted(R[t].topk(K).indices.tolist())
ov_sum = [len(set(top(Rs, t)) & set(fit[t])) for t in range(LT)]
ov_cov = [len(set(top(Rc, t)) & set(fit[t])) for t in range(LT)]
print("overlap with fitted support, sum-scale ranking:", ov_sum, "mean", sum(ov_sum) / LT)
print("overlap with fitted support, cov-scale ranking:", ov_cov, "mean", sum(ov_cov) / LT)
# official kvbridge ranking on a strided subsample, a few target layers (checks the cov-scale implementation)
idx = torch.arange(0, N, 8)
sk = [src[0, s] for s in range(LS)]; sv = [src[1, s] for s in range(LS)]
chk = {}
for t in (0, 10, 20, 30, LT - 1):
    order, sc = rank_source_layers(sk, sv, tgt[0, t], tgt[1, t], idx, lam=0.01, device=dev)
    chk[t] = sorted(order[:K])
    print(f"target {t:2d}: kvbridge official {chk[t]} | cov {top(Rc, t)} | sum {top(Rs, t)} | fitted {fit[t]}", flush=True)
json.dump(dict(R2_sum=Rs.tolist(), R2_cov=Rc.tolist(), fitted_support=fit, overlap_sum=ov_sum, overlap_cov=ov_cov,
               kvbridge_official_top8=chk, support_cov=[top(Rc, t) for t in range(LT)]),
          open(os.path.expanduser("~/kvcodesign/results/rank_check_8to14.json"), "w"), indent=1)
log("RANKDONE")
