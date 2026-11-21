# Do same-index KV heads correspond across model sizes?  (Is CacheBridge's head-local map justified for a pair?)
# Same data and statistics for every pair, so pairs are directly comparable:
#   data   : kvbridge's own calibration pipeline (FineWeb-Edu sample-10BT, first N_TEXTS texts > 3000 chars,
#            1024 tokens each, every 4th position), keys RoPE-stripped (content space), values raw.
#   per target layer t and kind (K / V), with X_s = source layer s (all heads, centred), Y_t = target layer t:
#     layer R2 [s,t]  : all-head ridge R2 of Y_t from X_s; best source layer s*(t) = argmax_s
#     head R2  [i,j]  : single source head i of s*(t) -> single target head j (128 -> 128 ridge)
#                       -> same-index mean, other mean, best mean, P(same index is the best of the 8) (chance 1/8)
#     share           : joint ridge from all heads of the top-8 source layers (CacheBridge/CFM layer budget) to Y_t;
#                       share of the explained variance of target head j coming from source heads with index j,
#                       sum(W*(G W)) over those blocks / sum over all blocks  (chance 1/8; REPORT sec. 12 used this)
# usage: SRC=Qwen3-8B TGT=Qwen3-14B CUDA_VISIBLE_DEVICES=3,4 python head_corr.py     (target may span two GPUs)
import os, gc, json, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvbridge.rope import strip_rope, get_rope_theta
from kvbridge.cache_compat import cache_layer_kv

E = os.environ.get
SRC, TGT = E("SRC", "Qwen3-1.7B"), E("TGT", "Qwen3-4B")
N_TEXTS, SEQ, STRIDE, TOPK = int(E("N_TEXTS", 500)), 1024, 4, int(E("TOPK", 8))
M = os.path.expanduser("~/kvswitch/models")
TEXTS = os.path.expanduser(f"~/kvcodesign/data/fineweb_texts_{N_TEXTS}.json")
OUT = os.path.expanduser(f"~/kvcodesign/results/head_corr_{SRC}_{TGT}.json")
dev = "cuda:0"
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
torch.set_grad_enabled(False)

if not os.path.exists(TEXTS):   # fixed text set shared by all pairs
    from datasets import load_dataset
    texts = []
    for r in load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True):
        if len(r["text"]) > 3000:
            texts.append(r["text"])
            if len(texts) == N_TEXTS:
                break
    os.makedirs(os.path.dirname(TEXTS), exist_ok=True)
    json.dump(texts, open(TEXTS, "w"))
texts = json.load(open(TEXTS))
tok = AutoTokenizer.from_pretrained(f"{M}/{SRC}")   # Qwen3 sizes share one tokenizer
ids_all = [tok(t, return_tensors="pt", truncation=True, max_length=SEQ).input_ids for t in texts]
keep_all = [torch.arange(i % STRIDE, x.shape[1], STRIDE) for i, x in enumerate(ids_all)]
NPOS = sum(len(k) for k in keep_all)


def collect(name):
    """-> K, V: [L, N, H*D] fp16 on CPU (K RoPE-stripped)."""
    m = AutoModelForCausalLM.from_pretrained(f"{M}/{name}", torch_dtype=torch.bfloat16, device_map="auto").eval()
    c = m.config
    L, H, D = c.num_hidden_layers, c.num_key_value_heads, getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)
    theta = get_rope_theta(c)
    K = torch.empty(L, NPOS, H * D, dtype=torch.float16); V = torch.empty_like(K)
    n0 = 0
    for i, (ids, keep) in enumerate(zip(ids_all, keep_all)):
        cache = m(ids.to(dev), use_cache=True).past_key_values
        pos, kp = torch.arange(ids.shape[1], device=dev), keep.to(dev)
        n1 = n0 + len(keep)
        for l in range(L):
            k, v = cache_layer_kv(cache, l)
            k = strip_rope(k[0].to(dev).float(), pos, theta)[:, kp]          # [H, n, D]
            K[l, n0:n1] = k.transpose(0, 1).reshape(len(keep), H * D).half().cpu()
            V[l, n0:n1] = v[0].to(dev).float()[:, kp].transpose(0, 1).reshape(len(keep), H * D).half().cpu()
        n0 = n1
        del cache
        if (i + 1) % 100 == 0:
            log(name, i + 1, "texts")
    del m, k, v
    gc.collect(); torch.cuda.empty_cache()   # device_map models hold reference cycles: without gc the weights stay on the GPU
    log(name, "layers", L, "kv heads", H, "head dim", D, "positions", NPOS,
        "GPU MB still allocated", [torch.cuda.memory_allocated(i) >> 20 for i in range(torch.cuda.device_count())])
    return K, V, L, H, D


Ks, Vs, LS, HS, DS = collect(SRC)
Kt, Vt, LT, HT, DT = collect(TGT)
assert HS == HT and DS == DT, "head-index correspondence needs equal KV head counts"
H, D = HS, DS


def stats(X):   # X [N, p] fp16 cpu -> centred fp32 gpu
    x = X.to(dev).float()
    return x - x.mean(0, keepdim=True)


def ridge_W(G, C):   # (G + lam I)^-1 C, lam relative to the mean diagonal
    lam = 1e-3 * float(G.diagonal().mean())
    return torch.linalg.solve(G.double() + lam * torch.eye(G.shape[0], device=dev, dtype=torch.float64), C.double()).float()


res = {"src": SRC, "tgt": TGT, "positions": NPOS, "kv_heads": H, "topk_layers": TOPK}
for kind, S, T in (("K", Ks, Kt), ("V", Vs, Vt)):
    # source-layer Gram matrices once (p = H*D per layer)
    Gs, R2 = [], torch.zeros(LS, LT)
    rows = []
    for t in range(LT):
        y = stats(T[t]); N = y.shape[0]; tv = y.square().sum(0)                   # per-column target SS
        tvh = tv.view(H, D).sum(1)                                                 # per target head
        r2l, Cst = torch.zeros(LS), []
        for s in range(LS):
            x = stats(S[s])
            if len(Gs) < LS:
                Gs.append((x.T @ x).cpu())
            G = Gs[s].to(dev); C = x.T @ y
            W = ridge_W(G, C)
            r2l[s] = float((W * C).sum() / tv.sum())                              # explained SS / total SS
            Cst.append(C.cpu())
            del x
        R2[:, t] = r2l
        sb = int(r2l.argmax())
        # head-pair R2 inside the best source layer
        G, C = Gs[sb].to(dev), Cst[sb].to(dev)
        hr = torch.zeros(H, H)                                                     # [src head i, tgt head j]
        for i in range(H):
            Gi = G[i * D:(i + 1) * D, i * D:(i + 1) * D]
            Ci = C[i * D:(i + 1) * D]                                               # [D, H*D]
            Wi = ridge_W(Gi, Ci)
            hr[i] = (Wi * Ci).sum(0).view(H, D).sum(1).cpu() / tvh.cpu()
        diag = hr.diagonal()
        off = (hr.sum(0) - diag) / (H - 1)
        same_best = float((hr.argmax(0) == torch.arange(H)).float().mean())
        rank = (hr > diag[None]).sum(0).float() + 1                                # 1 = best of 8
        # joint top-k layers, all heads: share by source head index
        top = r2l.topk(min(TOPK, LS)).indices.tolist()
        X = torch.empty(N, len(top) * H * D, device=dev)                           # [N, k*H*D], filled in place
        for j, s in enumerate(top):
            X[:, j * H * D:(j + 1) * H * D] = stats(S[s])
        Gj = X.T @ X; Cj = X.T @ y
        Wj = ridge_W(Gj, Cj)
        contrib = (Wj * (Gj @ Wj)).view(len(top), H, D, H, D).sum((0, 2, 4))      # [src head idx, tgt head]
        share_same = float(contrib.diagonal().sum() / contrib.sum())
        r2_joint = float((Wj * Cj).sum() / tv.sum())
        del X, Gj, Cj, Wj, y
        torch.cuda.empty_cache()
        rows.append(dict(t=t, best_src_layer=sb, layer_r2_best=float(r2l.max()), joint_r2_topk=r2_joint,
                         head_r2_same=float(diag.mean()), head_r2_other=float(off.mean()), head_r2_best=float(hr.max(0).values.mean()),
                         p_same_is_best=same_best, mean_rank_same=float(rank.mean()), share_same_index=share_same))
        log(kind, t, {k: round(v, 3) if isinstance(v, float) else v for k, v in rows[-1].items()})
    avg = lambda k: sum(r[k] for r in rows) / len(rows)
    res[kind] = dict(per_layer=rows, layer_r2=R2.tolist(),
                     summary={k: round(avg(k), 4) for k in ["layer_r2_best", "joint_r2_topk", "head_r2_same", "head_r2_other",
                                                            "head_r2_best", "p_same_is_best", "mean_rank_same", "share_same_index"]})
    log(kind, "SUMMARY", res[kind]["summary"])
json.dump(res, open(OUT, "w"), indent=1)
log("HCDONE", OUT)
