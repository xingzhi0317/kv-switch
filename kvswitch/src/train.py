# C4: KL self-distillation of a low-rank residual on top of the B1 (CacheBridge-style) maps.
# W_K <- W_K + U_K V_K, W_V <- W_Va + U_V V_V (per target layer).  Loss: KL(B native || B on translated cache)
# over T continuation tokens.  Prefix length random in [NMIN, NMAX].
# env: PAIR RANK(64) STEPS(400) BS(4) LR(1e-3) NMIN(2048) NMAX(8192) T(256) TAG
import os, json, math, random, torch, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from ks import *

E = os.environ.get
PAIRS = {"P1": ("Qwen3-4B", "Qwen3-1.7B"), "P2": ("Qwen3-1.7B", "Qwen3-4B")}
PAIR = E("PAIR")
RANK, STEPS, BS, LR = int(E("RANK", 64)), int(E("STEPS", 400)), int(E("BS", 4)), float(E("LR", 1e-3))
NMIN, NMAX, T, K = int(E("NMIN", 2048)), int(E("NMAX", 8192)), int(E("T", 256)), 3
TAG = E("TAG", f"ours_r{RANK}")
A, B = Model(PAIRS[PAIR][0]), Model(PAIRS[PAIR][1])
maps = torch.load(f"{ROOT}/maps/{PAIR}.pt")[K]
dev = "cuda"
WK = maps["WK"].to(torch.bfloat16).to(dev)
WV = maps["WVa"].to(torch.bfloat16).to(dev)
src = maps["src"]
LB, Fd, Fo = WK.shape
torch.manual_seed(0)


def lr_pair():
    U = torch.zeros(LB, Fd, RANK, device=dev, requires_grad=True)
    V = (torch.randn(LB, RANK, Fo, device=dev) * 0.02).requires_grad_()
    return U, V


UK, VK = lr_pair()
UV, VV = lr_pair()
params = [UK, VK, UV, VV]
opt = torch.optim.Adam(params, lr=LR)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))

uc_tr = json.load(open(f"{ROOT}/data/ultrachat_train.json"))
uc_va = json.load(open(f"{ROOT}/data/ultrachat.json"))
c4_tr = torch.load(os.path.expanduser("~/kvmap/reuse/c4_train.pt"))[:-64 * 2048]
c4_va = torch.load(os.path.expanduser("~/kvmap/reuse/c4_test.pt"))


def sample(rng, L, val=False):
    if rng.random() < 0.5:
        uc, ids = (uc_va if val else uc_tr), []
        while len(ids) < L:
            conv = uc[rng.randrange(len(uc))]
            ids += A.conv_ids([(m["role"], m["content"]) for m in conv])[0]
        return torch.tensor(ids[:L])
    c4 = c4_va if val else c4_tr
    o = rng.randrange(len(c4) - L)
    return c4[o:o + L].long()


def feats(cA, ss, n, kind):
    if kind == "k":
        cos, sin = A.cossin(0, n)
        xs = [derope(cA.k[s][:, :n].float(), cos, sin).to(torch.bfloat16) for s in ss]
    else:
        xs = [cA.v[s][:, :n] for s in ss]
    return torch.cat([x.transpose(0, 1).reshape(n, -1) for x in xs], -1)


def tmap(X, W, U, V):
    return X @ W[:-1] + W[-1] + ((X @ U[:-1].to(X.dtype) + U[-1].to(X.dtype)) @ V.to(X.dtype))


def layer_fn(h, Xk, Xv, ukl, vkl, uvl, vvl, l, sk, sv, n, cos_p, sin_p, cos_c, sin_c, mask):
    G = B.Hq // B.Hk
    Tn = h.shape[1]
    kpre = tmap(Xk, WK[l], ukl, vkl).view(n - 1, B.Hk, B.D).transpose(0, 1)
    Kp = rope(kpre.float(), cos_p, sin_p).to(torch.bfloat16)
    Vp = tmap(Xv, WV[l], uvl, vvl).view(n - 1, B.Hk, B.D).transpose(0, 1)
    layer = B.layers[l]
    x = layer.input_layernorm(h)
    q, k, v = B.qkv(l, x)
    qp = rope(q.float(), cos_c, sin_c).to(q.dtype)
    kp = rope(k.float(), cos_c, sin_c).to(k.dtype)
    Kf = torch.cat([sk, Kp, kp[0]], 1)[None].repeat_interleave(G, 1)
    Vf = torch.cat([sv, Vp, v[0]], 1)[None].repeat_interleave(G, 1)
    o = F.scaled_dot_product_attention(qp, Kf, Vf, attn_mask=mask)
    h = h + layer.self_attn.o_proj(o.transpose(1, 2).reshape(1, Tn, -1))
    return h + layer.mlp(layer.post_attention_layernorm(h))


def kl_loss(ids, n, grad=True):
    ids = ids[None].to(dev)
    with torch.no_grad():
        cB = B.new_cache(n + T)
        B.forward(ids[:, :n], cB, want="hidden")
        ref = B.forward(ids[:, n:], cB, want="all")[0].log_softmax(-1)
        sk = [cB.k[l][:, :1].clone() for l in range(B.L)]
        sv = [cB.v[l][:, :1].clone() for l in range(B.L)]
        del cB
        cA = A.new_cache(n)
        A.forward(ids[:, :n], cA, want="hidden")
        Xks = [feats(cA, src[l], n, "k")[1:] for l in range(B.L)]  # positions 1..n-1 (pos 0 = B's own sink)
        Xvs = [feats(cA, src[l], n, "v")[1:] for l in range(B.L)]
        del cA
    cos_p, sin_p = B.cossin(1, n)
    cos_c, sin_c = B.cossin(n, n + T)
    mask = torch.arange(n + T, device=dev)[None] <= (n + torch.arange(T, device=dev))[:, None]
    with torch.set_grad_enabled(grad):
        h = B.m.model.embed_tokens(ids[:, n:])
        for l in range(B.L):
            args = (h, Xks[l], Xvs[l], UK[l], VK[l], UV[l], VV[l], l, sk[l], sv[l], n, cos_p, sin_p, cos_c, sin_c, mask)
            h = checkpoint(layer_fn, *args, use_reentrant=False) if grad else layer_fn(*args)
        lp = B.m.lm_head(B.m.model.norm(h))[0].float().log_softmax(-1)
        return (ref.exp() * (ref - lp)).sum(-1).mean()


vrng = random.Random(123)
VAL = {n: [sample(vrng, n + T, True) for _ in range(8)] for n in (2048, 8192)}


def evaluate():
    with torch.no_grad():
        return {n: sum(kl_loss(x, n, False).item() for x in xs) / len(xs) for n, xs in VAL.items()}


hist = dict(cfg=dict(PAIR=PAIR, RANK=RANK, STEPS=STEPS, BS=BS, LR=LR, NMIN=NMIN, NMAX=NMAX, T=T, K=K), ev=[(0, evaluate())], tr=[])
log("step 0 val KL", hist["ev"][-1][1])
rng = random.Random(0)
run = None
for step in range(1, STEPS + 1):
    opt.zero_grad(set_to_none=True)
    tot = 0.0
    for _ in range(BS):
        n = rng.randint(NMIN, NMAX)
        loss = kl_loss(sample(rng, n + T), n) / BS
        loss.backward()
        tot += loss.item()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); sched.step()
    run = tot if run is None else 0.95 * run + 0.05 * tot
    if step % 10 == 0:
        hist["tr"].append((step, run))
        log(f"step {step} kl(ema) {run:.4f} mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G")
    if step % 100 == 0 or step == STEPS:
        hist["ev"].append((step, evaluate()))
        log("val", step, hist["ev"][-1][1])
        torch.save(dict(UK=UK.detach().cpu(), VK=VK.detach().cpu(), UV=UV.detach().cpu(), VV=VV.detach().cpu(), k=K, hist=hist),
                   f"{ROOT}/maps/{PAIR}_{TAG}.pt")
json.dump(hist, open(f"{ROOT}/results/train_{PAIR}_{TAG}.json", "w"), indent=1)
log("TRAINDONE")
