# Our low-rank shared-latent translator for a large pair (Qwen3-8B -> 14B).  Same estimator as fit_lowrank_dir.py
# (screen2.py metric "var", alpha 0.01): per kind (K, V), x = all source layers' content-space K (V) of a token, y = all
# target layers'; ridge W = (Xc^T Xc + a mean(diag) I)^-1 Xc^T Yc on the first 90% of positions; reduced rank = top-r
# eigenvectors of the per-target-layer variance-equalised prediction covariance (5 power iterations, 2304 directions);
# P = W T U_r, D = U_r^T T^-1.  Output: ~/kvcodesign/mappers/{DIR}/rrr_var_r{768,1536,2048}.pt (eval_quality rrr_k*_v*).
# Speed: the Gram / cross-product accumulation and the power iterations run on the GPU (fp32, as the CPU original);
# the ridge solve stays fp64 on the CPU.  Data: FineWeb-Edu 500 x 1024 tokens, stride 4 (the CFM / CacheBridge data).
# usage: DIR=8to14 CUDA_VISIBLE_DEVICES=a,b python fit_lowrank_big.py      (target spans both GPUs while collecting)
import os, gc, json, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvbridge.rope import strip_rope, get_rope_theta
from kvbridge.cache_compat import cache_layer_kv

E = os.environ.get
DIR, A = E("DIR", "8to14"), float(E("ALPHA", 0.01))
SRC, TGT = {"8to14": ("Qwen3-8B-ms", "Qwen3-14B-ms")}[DIR]
M = os.path.expanduser("~/kvswitch/models")
TEXTS = os.path.expanduser("~/kvcodesign/data/fineweb_texts_full500.json")
OUT = os.path.expanduser(f"~/kvcodesign/mappers/{DIR}"); os.makedirs(OUT, exist_ok=True)
RANKS, KTOP, SEQ, STRIDE, CH, dev = (768, 1536, 2048), 2304, 1024, 4, 4096, "cuda:0"
torch.set_num_threads(int(E("THREADS", 64)))
torch.backends.cuda.matmul.allow_tf32 = False
t0 = time.time()
log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{(time.time() - t0) / 60:.1f} min]", *a, flush=True)
torch.set_grad_enabled(False)
texts = json.load(open(TEXTS))
tok = AutoTokenizer.from_pretrained(f"{M}/{SRC}")
ids_all = [tok(t, return_tensors="pt", truncation=True, max_length=SEQ).input_ids for t in texts]
keep_all = [torch.arange(i % STRIDE, x.shape[1], STRIDE) for i, x in enumerate(ids_all)]
NPOS = sum(len(k) for k in keep_all)


def collect(name):   # -> {"k": [N, L*H*D], "v": ...} fp16 CPU, layer-major (= torch.stack(layers, 1).reshape(N, -1))
    m = AutoModelForCausalLM.from_pretrained(f"{M}/{name}", torch_dtype=torch.bfloat16, device_map="auto").eval()
    c = m.config
    L, H, D = c.num_hidden_layers, c.num_key_value_heads, c.head_dim
    theta = get_rope_theta(c)
    out = {kd: torch.empty(NPOS, L * H * D, dtype=torch.float16) for kd in "kv"}
    n0 = 0
    for ids, keep in zip(ids_all, keep_all):
        cache = m(ids.to(dev), use_cache=True).past_key_values
        pos, kp = torch.arange(ids.shape[1], device=dev), keep.to(dev)
        n1 = n0 + len(keep)
        for l in range(L):
            k, v = cache_layer_kv(cache, l)
            out["k"][n0:n1, l * H * D:(l + 1) * H * D] = strip_rope(k[0].to(dev).float(), pos, theta)[:, kp].transpose(0, 1).reshape(n1 - n0, H * D).half().cpu()
            out["v"][n0:n1, l * H * D:(l + 1) * H * D] = v[0].to(dev).float()[:, kp].transpose(0, 1).reshape(n1 - n0, H * D).half().cpu()
        n0 = n1
        del cache
    del m, k, v
    gc.collect(); torch.cuda.empty_cache()
    log(name, "collected", NPOS, "positions,", L, "layers; GPU MB still allocated", [torch.cuda.memory_allocated(i) >> 20 for i in range(torch.cuda.device_count())])
    return out, L


SRCD, LS = collect(SRC)
TGTD, LT = collect(TGT)
N = NPOS; ntr = int(N * 0.9)


def top_eig(Ww, G, k, iters=5, seed=0):   # as fit_lowrank_dir.py / common.py, on the GPU
    g = torch.Generator().manual_seed(seed)
    Q = torch.linalg.qr(torch.randn(Ww.shape[1], k, generator=g).to(dev))[0]
    op = lambda Mx: Ww.T @ (G @ (Ww @ Mx))
    for _ in range(iters):
        Q = torch.linalg.qr(op(Q))[0]
    B = Q.T @ op(Q)
    ev, U = torch.linalg.eigh(B.double().cpu())
    return (Q.cpu() @ U.flip(1).float())


res, report = {}, {}
for kind in ("k", "v"):
    X, Y = SRCD[kind], TGTD[kind]
    mx = sum(X[i:min(ntr, i + CH)].double().sum(0) for i in range(0, ntr, CH)) / ntr
    my = sum(Y[i:min(ntr, i + CH)].double().sum(0) for i in range(0, ntr, CH)) / ntr
    mxf, myf = mx.float(), my.float()
    G = torch.zeros(X.shape[1], X.shape[1], device=dev); C = torch.zeros(X.shape[1], Y.shape[1], device=dev)
    yy = torch.zeros(Y.shape[1], dtype=torch.float64, device=dev)
    mxd, myd = mxf.to(dev), myf.to(dev)
    for i in range(0, ntr, CH):
        j = min(ntr, i + CH)
        xc = X[i:j].to(dev).float() - mxd; yc = Y[i:j].to(dev).float() - myd
        G.addmm_(xc.T, xc); C.addmm_(xc.T, yc); yy += yc.double().square().sum(0)
    log(kind, "statistics on GPU done", tuple(G.shape), tuple(C.shape))
    G, C, yy = G.cpu(), C.cpu(), yy.cpu(); torch.cuda.empty_cache()
    Ycols = C.shape[1] // LT
    w = (yy / ntr).float().view(LT, -1).mean(1).rsqrt().repeat_interleave(Ycols)
    Gd = G.double(); Gd.diagonal().add_(A * G.diagonal().mean().item())
    W = torch.cholesky_solve(C.double(), torch.linalg.cholesky(Gd)).float(); del Gd, C
    log(kind, "ridge solved (fp64, CPU)")
    U = top_eig((W * w).to(dev), G.to(dev), KTOP); torch.cuda.empty_cache()
    log(kind, "top directions done")
    Xval, Yval = X[ntr:].float(), Y[ntr:].float()
    s = (Yval - Yval.mean(0)).square().view(-1, LT, Ycols).sum((0, 2))
    for r in RANKS:
        P = (W * w) @ U[:, :r]; Dm = U[:, :r].T.contiguous() / w
        pred = (Xval - mxf) @ P @ Dm + myf
        r2 = (1 - (pred - Yval).square().view(-1, LT, Ycols).sum((0, 2)) / s).mean().item()
        res[(kind, r)] = (P, Dm, mxf, myf); report[f"{kind}_r{r}"] = r2
        log(kind, "rank", r, "held-out R2 (mean over target layers)", round(r2, 4))
    pred = (Xval - mxf) @ W + myf
    report[f"{kind}_full"] = (1 - (pred - Yval).square().view(-1, LT, Ycols).sum((0, 2)) / s).mean().item()
    log(kind, "full-rank ridge held-out R2", round(report[f"{kind}_full"], 4))
    del W, U, G, Xval, Yval, pred
for r in RANKS:
    Pk, Dk, mxk, myk = res[("k", r)]; Pv, Dv, mxv, myv = res[("v", r)]
    torch.save(dict(Pk=Pk, Dk=Dk, Pv=Pv, Dv=Dv, mx=torch.cat([mxk, mxv]), my=torch.cat([myk, myv]),
                    meta=dict(dir=DIR, alpha=A, rank=r, metric="var", positions=N, train=ntr)), f"{OUT}/rrr_var_r{r}.pt")
json.dump(dict(dir=DIR, positions=N, train=ntr, heldout_r2=report, minutes=(time.time() - t0) / 60), open(f"{OUT}/lowrank_fit_report.json", "w"), indent=1)
log("LOWRANKDONE", OUT)
