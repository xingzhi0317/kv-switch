# Translation error by token type (PLAN_v5 stage 2, item 2).  CPU only, our own analysis code.
# Data: 16 held-out FineWeb-Edu documents (data/val_docs.pt: token ids, source/target K,V in content space, target Q).
# Mappers: dense ridge over all source layers (data/wraw_a0.01.pt) and low-rank K768+V1536, K2048+V2048 (mappers/rrr_var_*).
# Per token j and target layer l:
#   k_err, v_err : ||yhat - y||^2 / (mean over tokens of ||y - mean||^2)   -> 1.0 means "as bad as predicting the mean"
#   logit_err    : mean over later queries i and heads of (q_i . (khat_j - k_j))^2 / mean over (i, j) of (q_i . (k_j - mean))^2
#                  (how much the K error moves the attention logits this token receives; uses the target's real queries)
# Token types (first match wins): first (position 0) | digit | punct (punctuation / whitespace only) | cap (word starting
# with an upper-case letter, not at a sentence start) | cont (sub-word continuation piece) | word (everything else).
# Output: results/tokerr.json     env: THREADS
import os, json, math, time, numpy as np, torch
torch.set_num_threads(int(os.environ.get("THREADS", 48)))
HOME = os.path.expanduser("~")
LS, LT, H, D, HQ = 28, 36, 8, 128, 32
PK, QK = LS * H * D, LT * H * D
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
st = torch.load(f"{HOME}/kvcodesign/data/stats_17to4.pt", mmap=True)
mx, my = st["mx"].float(), st["my"].float()
DOCS = torch.load(f"{HOME}/kvcodesign/data/val_docs.pt")
M = lambda r: torch.load(f"{HOME}/kvcodesign/mappers/rrr_var_r{r}.pt")
Wk, Wv = torch.load(f"{HOME}/kvcodesign/data/wraw_a0.01.pt")
m768, m1536, m2048 = M(768), M(1536), M(2048)
MAPS = {
    "dense": (lambda x: x @ Wk, lambda x: x @ Wv),
    "lowrank_k768_v1536": (lambda x: (x @ m768["Pk"]) @ m768["Dk"], lambda x: (x @ m1536["Pv"]) @ m1536["Dv"]),
    "lowrank_k2048_v2048": (lambda x: (x @ m2048["Pk"]) @ m2048["Dk"], lambda x: (x @ m2048["Pv"]) @ m2048["Dv"]),
}
from tokerr_types import token_types, TYPES, tok


def rope(x, pos):   # x [..., n, D], rotate_half convention, theta 1e6
    inv = 1.0 / (1e6 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    ang = pos.float()[:, None] * inv[None]
    cos, sin = torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)
    return x * cos + torch.cat([-x[..., D // 2:], x[..., :D // 2]], -1) * sin


def logits(q, k, qpos):   # q [LT, nq, HQ, D] rotated; k [LT, n, H, D] content space -> [LT, HQ, nq, n]
    n = k.shape[1]
    kr = rope(k.transpose(1, 2), torch.arange(n)).repeat_interleave(HQ // H, 1)       # [LT, HQ, n, D]
    return q.permute(0, 2, 1, 3) @ kr.transpose(-1, -2) / math.sqrt(D)


rows = []   # per token: doc, pos, type, then per method k/v/logit errors (mean over layers) and per-layer-third splits
per_layer = {m: {x: {t: torch.zeros(LT) for t in TYPES} for x in ("k", "v", "logit")} for m in MAPS}
cnt = {t: 0 for t in TYPES}
for di, d in enumerate(DOCS):
    n = d["sk"].shape[1]
    ty = token_types(d["ids"])
    xk = d["sk"].float().permute(1, 0, 2, 3).reshape(n, PK) - mx[:PK]
    xv = d["sv"].float().permute(1, 0, 2, 3).reshape(n, PK) - mx[PK:]
    yk = d["tk"].float().permute(1, 0, 2, 3).reshape(n, QK) - my[:QK]                  # centred targets
    yv = d["tv"].float().permute(1, 0, 2, 3).reshape(n, QK) - my[QK:]
    q = rope(d["tq"].float().transpose(1, 2), d["qpos"]).transpose(1, 2)               # [LT, nq, HQ, D] rotated
    mask = (torch.arange(n)[None, :] <= d["qpos"][:, None]).float()                    # [nq, n] visible keys
    vis = mask.sum(0).clamp_min(1)                                                     # queries that can see key j
    lay = lambda e: e.view(n, LT, H * D).sum(2)                                        # [n, LT]
    dk, dv = lay(yk.square()).mean(0), lay(yv.square()).mean(0)                        # per-layer normalisers
    s0 = logits(q, yk.view(n, LT, H, D).transpose(0, 1), d["qpos"])                    # logits of the centred true K
    dl = (s0.square() * mask).sum((1, 2, 3)) / (mask.sum() * HQ)                       # [LT]
    res = {}
    for name, (fk, fv) in MAPS.items():
        ek, ev = fk(xk) - yk, fv(xv) - yv
        se = logits(q, ek.view(n, LT, H, D).transpose(0, 1), d["qpos"])               # [LT, HQ, nq, n]
        el = ((se.square() * mask).sum((1, 2)) / (vis * HQ)).T / dl                    # [n, LT]
        res[name] = dict(k=lay(ek.square()) / dk, v=lay(ev.square()) / dv, logit=el)
        for x in ("k", "v", "logit"):
            for t in TYPES:
                sel = torch.tensor([a == t for a in ty])
                per_layer[name][x][t] += res[name][x][sel].sum(0)
    for t in ty:
        cnt[t] += 1
    for j in range(n):
        r = dict(doc=di, pos=j, type=ty[j])
        for name in MAPS:
            for x in ("k", "v", "logit"):
                r[f"{name}.{x}"] = float(res[name][x][j].mean())
        rows.append(r)
    log("doc", di, {t: ty.count(t) for t in TYPES})

out = dict(counts=cnt, types=TYPES, by_type={}, by_type_layer={}, by_position={}, examples={})
rng = np.random.default_rng(0)
for name in MAPS:
    for x in ("k", "v", "logit"):
        key = f"{name}.{x}"
        base = np.array([r[key] for r in rows if r["type"] == "word"])
        out["by_type"][key] = {}
        for t in TYPES:
            a = np.array([r[key] for r in rows if r["type"] == t])
            bs = [a[rng.integers(0, len(a), len(a))].mean() / base[rng.integers(0, len(base), len(base))].mean() for _ in range(2000)]
            out["by_type"][key][t] = dict(n=len(a), mean=float(a.mean()), ratio_to_word=float(a.mean() / base.mean()),
                                          ratio_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        out["by_type_layer"][key] = {t: (per_layer[name][x][t] / max(cnt[t], 1)).tolist() for t in TYPES}
        out["by_position"][key] = {b: float(np.mean([r[key] for r in rows if lo <= r["pos"] < hi and r["type"] != "first"]))
                                   for b, lo, hi in (("1-15", 1, 16), ("16-127", 16, 128), ("128-511", 128, 512), ("512-1023", 512, 1024))}
for t in TYPES:   # a few decoded examples per type, to check the classifier by eye
    ex = [tok.decode([int(DOCS[r["doc"]]["ids"][r["pos"]])]) for r in rows if r["type"] == t][:400:10]
    out["examples"][t] = ex
json.dump(out, open(f"{HOME}/kvcodesign/results/tokerr.json", "w"), indent=1)
print("counts", cnt)
for key, v in out["by_type"].items():
    print(f"{key:28s}", "  ".join(f"{t} {v[t]['mean']:.3f} (x{v[t]['ratio_to_word']:.2f})" for t in TYPES))
for t in TYPES:
    print(t, out["examples"][t][:25])
log("TOKERRDONE")
