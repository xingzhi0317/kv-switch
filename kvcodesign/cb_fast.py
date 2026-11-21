# CacheBridge initial mapper from sufficient statistics: read the fitting data ONCE, then rank source layers, fit any
# layer selection and compute the energy-budget statistics in seconds (fit_cb_init_big.py re-collects the KV and runs the
# official fit_layer over the full 17-19 GB traces for every target layer: ~25 min per mapper for 8B -> 14B).
#   MODE=collect : both models read the fitting texts (FineWeb-Edu, 500 x 1024 tokens, stride 4 = 128,000 positions,
#                  keys RoPE-stripped); per KV head and component (K, V) accumulate the centred statistics of ALL layers
#                  in one pass:  Sxx = Xc^T Xc over the concatenated source layers [LS*D, LS*D], Sxy = Xc^T Yc against the
#                  concatenated target layers [LS*D, LT*D], tr(Yc^T Yc) per target layer, the means; plus 4096 sampled
#                  positions of the source / target V (the paper's budget states).  Saved under STATS_DIR.
#   MODE=fit     : SEL=sum | cov | <json file with a support list>, K (8):
#                  ranking  = paper eq. 5 (same-index head ridge R^2, mean over heads, mean of K and V); ridge 0.01 on the
#                             summed Gram (sum; = fit_cb_init_big.py) or on the averaged Gram (cov; kvbridge/CFM code);
#                  fit      = official fit_layer arithmetic (centred, ridge 0.01 added to the summed Gram, fp64, b = my - mx W);
#                  budget   = M and E0 on the 4096 sampled states (as fit_cb_init_big.py)  ->  OUT/, OUT_stats.pt
#                  REF=<mapper dir>: report the weight difference to a mapper fitted with the official fit_layer.
# usage: MODE=collect CUDA_VISIBLE_DEVICES=a,b python cb_fast.py ;  MODE=fit SEL=cov OUT=... CUDA_VISIBLE_DEVICES=a python cb_fast.py
import os, gc, json, time, torch
from safetensors.torch import save_file, load_file

E = os.environ.get
MODE, DIR = E("MODE", "collect"), E("DIR", "8to14")
SRC, TGT = {"8to14": ("Qwen3-8B-ms", "Qwen3-14B-ms"), "17to4": ("Qwen3-1.7B", "Qwen3-4B")}[DIR]
M = os.path.expanduser("~/kvswitch/models")
TEXTS = os.path.expanduser("~/kvcodesign/data/fineweb_texts_full500.json")
SD = os.path.expanduser(E("STATS_DIR", f"~/kvbase/stats/{DIR}_full500"))
dev, SEQ, STRIDE, NS = "cuda:0", 1024, 4, 4096
t0 = time.time()
log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{(time.time() - t0) / 60:.1f} min]", *a, flush=True)
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False

if MODE == "collect":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kvbridge.rope import strip_rope, get_rope_theta
    from kvbridge.cache_compat import cache_layer_kv
    texts = json.load(open(TEXTS))
    tok = AutoTokenizer.from_pretrained(f"{M}/{SRC}")
    ids_all = [tok(t, return_tensors="pt", truncation=True, max_length=SEQ).input_ids for t in texts]
    keep_all = [torch.arange(i % STRIDE, x.shape[1], STRIDE) for i, x in enumerate(ids_all)]
    NPOS = sum(len(k) for k in keep_all)

    def collect(name):   # -> [2, L, N, H, D] fp16 CPU; identical to fit_cb_init_big.py
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

    src, tgt = collect(SRC), collect(TGT)
    _, LS, N, H, D = src.shape
    LT = tgt.shape[1]
    CH = 4096
    os.makedirs(SD, exist_ok=True)
    for comp, name in ((0, "k"), (1, "v")):
        mx = torch.zeros(H, LS * D, device=dev, dtype=torch.float64); my = torch.zeros(H, LT * D, device=dev, dtype=torch.float64)
        flat = lambda T, a, b: T[comp, :, a:b].to(dev).float().permute(2, 1, 0, 3).reshape(T.shape[3], b - a, -1)   # [H, c, L*D]
        for a in range(0, N, CH):   # pass 1: means
            b = min(a + CH, N)
            mx += flat(src, a, b).double().sum(1); my += flat(tgt, a, b).double().sum(1)
        mx /= N; my /= N
        Sxx = torch.zeros(H, LS * D, LS * D, device=dev, dtype=torch.float64)
        Sxy = torch.zeros(H, LS * D, LT * D, device=dev, dtype=torch.float64)
        Syy = torch.zeros(H, LT * D, device=dev, dtype=torch.float64)          # per-column sum of squares
        mxf, myf = mx.float()[:, None], my.float()[:, None]
        for a in range(0, N, CH):   # pass 2: centred products (fp32 per chunk, fp64 accumulation)
            b = min(a + CH, N)
            x = flat(src, a, b) - mxf; y = flat(tgt, a, b) - myf
            Sxx += torch.bmm(x.transpose(1, 2), x).double(); Sxy += torch.bmm(x.transpose(1, 2), y).double()
            Syy += y.double().square().sum(1)
        torch.save(dict(Sxx=Sxx.cpu(), Sxy=Sxy.cpu(), Syy=Syy.cpu(), mx=mx.cpu(), my=my.cpu()), f"{SD}/stats_{name}.pt")
        del Sxx, Sxy, Syy; torch.cuda.empty_cache()
        log("statistics", name, "saved")
    samp = torch.randperm(N, generator=torch.Generator().manual_seed(2))[:NS]   # = fit_cb_init_big.py
    torch.save(dict(src_v=src[1][:, samp].clone(), tgt_v=tgt[1][:, samp].clone(), samp=samp), f"{SD}/budget_samples.pt")
    json.dump(dict(src=SRC, tgt=TGT, N=N, LS=LS, LT=LT, H=H, D=D, texts=TEXTS, stride=STRIDE), open(f"{SD}/meta.json", "w"), indent=1)
    log("COLLECTDONE", SD)

if MODE == "fit":
    meta = json.load(open(f"{SD}/meta.json"))
    N, LS, LT, H, D = (meta[k] for k in ("N", "LS", "LT", "H", "D"))
    K, SEL = int(E("K", 8)), E("SEL", "cov")
    OUT = os.path.expanduser(E("OUT", f"~/kvbase/mappers/cachebridge_{DIR}_fast_{SEL}_k{K}"))
    ST = {c: torch.load(f"{SD}/stats_{c}.pt") for c in ("k", "v")}
    log("loaded statistics", SD)
    # ---- ranking (paper eq. 5) from the statistics: per (source s, target t, head h), 128 -> 128 ridge, in-sample R^2
    R2 = torch.zeros(LT, LS, dtype=torch.float64)
    if SEL in ("sum", "cov"):
        lam = 0.01 if SEL == "sum" else 0.01 * N
        I = torch.eye(D, device=dev, dtype=torch.float64)
        for c in ("k", "v"):
            Sxx, Sxy, Syy = ST[c]["Sxx"].to(dev), ST[c]["Sxy"].to(dev), ST[c]["Syy"].to(dev)
            for s in range(LS):
                G = Sxx[:, s * D:(s + 1) * D, s * D:(s + 1) * D]                                   # [H, D, D]
                for t in range(LT):
                    C = Sxy[:, s * D:(s + 1) * D, t * D:(t + 1) * D]                               # [H, D, D]
                    W = torch.linalg.solve(G + lam * I, C)
                    sst = Syy[:, t * D:(t + 1) * D].sum(1)
                    sse = sst - 2 * (W * C).sum((1, 2)) + (W * (G @ W)).sum((1, 2))
                    R2[t, s] += 0.5 * (1 - sse / sst).mean().cpu()
            del Sxx, Sxy, Syy
        support = [sorted(R2[t].topk(K).indices.tolist()) for t in range(LT)]
    else:
        support = json.load(open(SEL))["support"]
    log("selection", SEL, "first/last target layer", support[0], support[-1])
    # ---- fit: official fit_layer arithmetic from the statistics
    os.makedirs(OUT, exist_ok=True)
    layers = []
    for t in range(LT):
        idx = torch.cat([torch.arange(s * D, (s + 1) * D) for s in support[t]])
        L = {"selected": torch.tensor(support[t], dtype=torch.int64), "headwise": torch.tensor(True)}
        for c in ("k", "v"):
            Sxx, Sxy = ST[c]["Sxx"], ST[c]["Sxy"]
            G = Sxx[:, idx][:, :, idx].to(dev).clone(); C = Sxy[:, idx, t * D:(t + 1) * D].to(dev)
            G = 0.5 * (G + G.transpose(1, 2)); G.diagonal(dim1=1, dim2=2).add_(0.01)
            W = torch.cholesky_solve(C, torch.linalg.cholesky(G))                                 # [H, K*D, D]
            b = ST[c]["my"][:, t * D:(t + 1) * D].to(dev) - (ST[c]["mx"][:, idx].to(dev)[:, None] @ W)[:, 0]
            L[f"{c}_weight"], L[f"{c}_bias"] = W.float().cpu().contiguous(), b.float().cpu().contiguous()
        save_file(L, f"{OUT}/layer_{t:03d}.safetensors")
        layers.append(L)
    log("fitted", LT, "layers")
    # ---- energy-budget statistics on the 4096 sampled fitting states (as fit_cb_init_big.py)
    BS = torch.load(f"{SD}/budget_samples.pt")
    Ms, E0s = [], []
    for t in range(LT):
        Lt = layers[t]
        xv = torch.cat([BS["src_v"][s].transpose(0, 1).to(dev).float() for s in support[t]], -1)    # [H, NS, K*D]
        vmap = torch.einsum("hnf,hfd->hnd", xv, Lt["v_weight"].to(dev)) + Lt["v_bias"].to(dev)[:, None]
        E0s.append((vmap - BS["tgt_v"][t].transpose(0, 1).to(dev).float()).double().square().sum(-1).mean(-1).cpu())
        Xb = torch.cat([xv, torch.ones_like(xv[..., :1])], -1).double()
        Ms.append(((Xb.transpose(1, 2) @ Xb) / Xb.shape[1]).float().cpu())
    torch.save(dict(M=Ms, E0=E0s, n=NS), OUT + "_stats.pt")
    json.dump(dict(support=support, K=K, sel=SEL, R2=R2.tolist(), src=meta["src"], tgt=meta["tgt"], positions=N,
                   note="initial head-local fit from sufficient statistics (cb_fast.py), no value calibration"),
              open(f"{OUT}/fit_report.json", "w"), indent=1)
    log("saved", OUT, "E0 layer 0 / last", round(float(E0s[0].mean()), 4), round(float(E0s[-1].mean()), 4))
    if E("REF"):   # compare with a mapper fitted by the official fit_layer on the same data
        ref = os.path.expanduser(E("REF")); worst = 0.0
        rsup = json.load(open(f"{ref}/fit_report.json"))["support"]
        same = [t for t in range(LT) if rsup[t] == support[t]]
        for t in same:
            A, B = load_file(f"{ref}/layer_{t:03d}.safetensors"), layers[t]
            for k_ in ("k_weight", "k_bias", "v_weight", "v_bias"):
                worst = max(worst, float((A[k_] - B[k_]).norm() / A[k_].norm()))
        log("vs official fit_layer", ref, ":", len(same), "target layers with the same selection; worst relative weight difference", f"{worst:.2e}")
        if os.path.exists(ref + "_stats.pt"):
            R = torch.load(ref + "_stats.pt")
            e = max(float((R["E0"][t] - E0s[t]).abs().max() / R["E0"][t].abs().max()) for t in same)
            log("budget E0 worst relative difference", f"{e:.2e}")
    log("FITDONE")
