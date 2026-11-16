# Behavioural fine-tuning of the shared low-rank translator (PLAN_v5 stage 3, step 3).  Our own code.
# Recipe (not ours: closed-form init + KL self-distillation through the frozen target is used by arXiv 2611.12062
# and, as a residual on a frozen ridge base, by CacheLink arXiv 2610.15509).  What is ours: the trainable object is
# the shared low-rank factorisation itself -- the same four matrices Pk, Dk, Pv, Dv that the closed-form
# reduced-rank ridge produced -- so the translation cost and the fused kernel (fastlr.py) are unchanged.
#   y = ((x - mx) P) D + my          K: strip source RoPE before, apply target RoPE after
# Loss: KL( target reading its own cache || target reading the translated cache ), averaged over continuation tokens.
# Both models are frozen; gradients flow through the frozen Qwen3-4B into P and D.  No labels.
# Data: FineWeb-Edu documents #600+ of the same stream (calibration used #0-499, held-out validation uses #500-599).
#   mode A (half): history = text, continuation = the next 128 tokens of the text;
#   mode B (half): history = text, then a generic request (PROMPTS below), continuation = the native target's own
#                  greedy answer (<= 64 tokens).
# None of the evaluation templates (passkey, key-value list, SQuAD, GSM8K, QuALITY, HotpotQA) is used.
# env: RK RV STEPS(samples) ACC LR(relative to each matrix's RMS) MAXN TAG NDOCS VAL_EVERY
import os, sys, json, math, time, random
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

E = os.environ.get
RK, RV = int(E("RK", 768)), int(E("RV", 1536))
STEPS, ACC, LR = int(E("STEPS", 2400)), int(E("ACC", 2)), float(E("LR", 1e-3))
MAXN, NDOCS, VAL_EVERY = int(E("MAXN", 3072)), int(E("NDOCS", 3000)), int(E("VAL_EVERY", 300))
TAG = E("TAG", "a")
HOME = os.path.expanduser("~")
MODELS = f"{HOME}/kvswitch/models"
OUT = f"{HOME}/kvcodesign/mappers/kl_k{RK}_v{RV}_{TAG}.pt"
LOGJ = f"{HOME}/kvcodesign/results/train_kl_k{RK}_v{RV}_{TAG}.json"
dev = "cuda"
LS, LT, H, D = 28, 36, 8, 128
PK, QK = LS * H * D, LT * H * D
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

tok = AutoTokenizer.from_pretrained(f"{MODELS}/Qwen3-4B")
S = AutoModelForCausalLM.from_pretrained(f"{MODELS}/Qwen3-1.7B", torch_dtype=torch.bfloat16).to(dev).eval()
T = AutoModelForCausalLM.from_pretrained(f"{MODELS}/Qwen3-4B", torch_dtype=torch.bfloat16).to(dev).eval()
for p in list(S.parameters()) + list(T.parameters()):
    p.requires_grad_(False)

# ---------------------------------------------------------------- translator (same maths as eval_quality.LowRank)
a = torch.load(f"{HOME}/kvcodesign/mappers/rrr_var_r{RK}.pt")
b = torch.load(f"{HOME}/kvcodesign/mappers/rrr_var_r{RV}.pt")
g = lambda t: t.to(dev, torch.float32)
W = {n: torch.nn.Parameter(g(t)) for n, t in (("Pk", a["Pk"]), ("Dk", a["Dk"]), ("Pv", b["Pv"]), ("Dv", b["Dv"]))}
mx, my = g(a["mx"]), g(a["my"])
inv = 1.0 / (1e6 ** (torch.arange(0, D, 2, device=dev, dtype=torch.float32) / D))
rot = lambda x: torch.cat([-x[..., D // 2:], x[..., :D // 2]], -1)


def translate(sc, n):
    """source DynamicCache (n tokens) -> target DynamicCache; differentiable w.r.t. W."""
    ang = torch.arange(n, device=dev, dtype=torch.float32)[:, None] * inv[None]
    c, s_ = torch.cat([ang.cos(), ang.cos()], -1)[None, :, None], torch.cat([ang.sin(), ang.sin()], -1)[None, :, None]
    sk = torch.stack([sc.layers[l].keys[0].transpose(0, 1) for l in range(LS)]).float()          # [LS, n, H, D]
    sv = torch.stack([sc.layers[l].values[0].transpose(0, 1) for l in range(LS)]).float()
    sk = sk * c - rot(sk) * s_
    xk = sk.permute(1, 0, 2, 3).reshape(n, PK) - mx[:PK]
    xv = sv.permute(1, 0, 2, 3).reshape(n, PK) - mx[PK:]
    yk = ((xk @ W["Pk"]) @ W["Dk"] + my[:QK]).view(n, LT, H, D).permute(1, 0, 2, 3)
    yv = ((xv @ W["Pv"]) @ W["Dv"] + my[QK:]).view(n, LT, H, D).permute(1, 0, 2, 3)
    yk = yk * c + rot(yk) * s_
    out = DynamicCache()
    for l in range(LT):
        out.update(yk[l].transpose(0, 1)[None].to(torch.bfloat16), yv[l].transpose(0, 1)[None].to(torch.bfloat16), l)
    return out


# ---------------------------------------------------------------- data
PROMPTS = [
    "Summarize the text above in one sentence.",
    "What is the main topic of the text above?",
    "Quote the first sentence of the text exactly.",
    "Quote the last sentence of the text exactly.",
    "List the proper names mentioned in the text.",
    "List the numbers and dates mentioned in the text, in the order they appear.",
    "Copy the last twenty words of the text exactly.",
    "Which sentence of the text is the most surprising? Quote it.",
]
enc = lambda t: tok(t, add_special_tokens=False).input_ids
PRE = enc("<|im_start|>user\n")
SEP = enc("\n\n")
STOP = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
log("loading documents")
docs = []
for i, row in enumerate(r for r in load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True) if len(r["text"]) > 3000):
    if i >= 500:
        docs.append(enc(row["text"])[:4096])
    if len(docs) == 100 + NDOCS:
        break
VAL_DOCS, TRAIN_DOCS = docs[:100], docs[100:]      # stream docs #500-599 (validation) and #600+ (training)
log("docs", len(TRAIN_DOCS), "train,", len(VAL_DOCS), "val; mean tokens", sum(map(len, TRAIN_DOCS)) // len(TRAIN_DOCS))


def sample(rng, pool):
    """-> dict(h, q, cont or None, mode).  History length log-uniform in [256, MAXN]."""
    n = int(math.exp(rng.uniform(math.log(256), math.log(MAXN))))
    mode = "A" if rng.random() < 0.5 else "B"
    need = n + (128 if mode == "A" else 0)
    text, j = [], rng.randrange(len(pool))
    while len(text) < need:
        text += (SEP if text else []) + pool[j]; j = (j + 1) % len(pool)
    if mode == "A":
        return dict(mode="A", h=PRE + text[:n], q=text[n:n + 128])
    q = enc(f"\n\n{rng.choice(PROMPTS)}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return dict(mode="B", h=PRE + text[:n], q=q)


@torch.no_grad()
def teacher(s):
    """native target: logits on the continuation; mode B also generates the answer greedily (<= 64 tokens)."""
    c = DynamicCache()
    T(torch.tensor([s["h"]], device=dev), past_key_values=c, use_cache=True)
    lg = T(torch.tensor([s["q"]], device=dev), past_key_values=c, use_cache=True).logits[0].float()
    if s["mode"] == "A":
        return s["q"], lg
    ans, steps, nxt = [], [lg[-1]], int(lg[-1].argmax())
    for _ in range(64):
        if nxt in STOP:
            break
        ans.append(nxt)
        l1 = T(torch.tensor([[nxt]], device=dev), past_key_values=c, use_cache=True).logits[0, -1].float()
        steps.append(l1); nxt = int(l1.argmax())
    return s["q"] + ans, torch.stack(steps)          # logits after q and after each answer token


def student_loss(s, ids, tl):
    with torch.no_grad():
        sc = S(torch.tensor([s["h"]], device=dev), use_cache=True).past_key_values
    c = translate(sc, len(s["h"]))
    sl = T(torch.tensor([ids], device=dev), past_key_values=c, use_cache=True).logits[0].float()
    sl = sl if s["mode"] == "A" else sl[len(s["q"]) - 1:]
    lp_t, lp_s = tl.log_softmax(-1), sl.log_softmax(-1)
    kl = (lp_t.exp() * (lp_t - lp_s)).sum(-1).mean()
    return kl, float((sl.argmax(-1) == tl.argmax(-1)).float().mean())


vr = random.Random(123)
VAL = [sample(vr, VAL_DOCS) for _ in range(int(E("NVAL", 24)))]
VAL_T = [teacher(s) for s in VAL]
log("validation set ready", {m: sum(s["mode"] == m for s in VAL) for m in "AB"})


@torch.no_grad()
def validate():
    r = {"A": [], "B": [], "A_top1": [], "B_top1": []}
    for s, (ids, tl) in zip(VAL, VAL_T):
        kl, top1 = student_loss(s, ids, tl)
        r[s["mode"]].append(float(kl)); r[s["mode"] + "_top1"].append(top1)
    return {k: round(sum(v) / max(len(v), 1), 4) for k, v in r.items()}


def save(step):
    torch.save(dict(Pk=W["Pk"].detach().cpu(), Dk=W["Dk"].detach().cpu(), Pv=W["Pv"].detach().cpu(), Dv=W["Dv"].detach().cpu(),
                    mx=mx.cpu(), my=my.cpu(), step=step, rk=RK, rv=RV), OUT)


# learning rate relative to each matrix's RMS (Adam's step is ~lr per element), cosine decay, short warm-up
opt = torch.optim.Adam([dict(params=[p], lr=LR * float(p.detach().square().mean().sqrt()), base=LR * float(p.detach().square().mean().sqrt())) for p in W.values()],
                       betas=(0.9, 0.99))
hist = dict(config=dict(rk=RK, rv=RV, steps=STEPS, acc=ACC, lr=LR, maxn=MAXN, ndocs=len(TRAIN_DOCS), prompts=PROMPTS), val=[], train=[])
v0 = validate(); hist["val"].append(dict(step=0, **v0)); log("step 0 (closed-form init) val", v0)
rng, run, t0 = random.Random(0), [], time.time()
updates = STEPS // ACC
for step in range(1, STEPS + 1):
    s = sample(rng, TRAIN_DOCS)
    try:
        ids, tl = teacher(s)
        kl, top1 = student_loss(s, ids, tl)
        (kl / ACC).backward()
        run.append((float(kl), top1, s["mode"]))
    except torch.OutOfMemoryError:
        log("OOM at step", step, "history", len(s["h"]), "- skipped"); opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        continue
    if step % ACC == 0:
        u = step // ACC
        f = min(1.0, u / 20) * 0.5 * (1 + math.cos(math.pi * u / updates))
        for gp in opt.param_groups:
            gp["lr"] = gp["base"] * f
        torch.nn.utils.clip_grad_norm_(list(W.values()), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
    if step % 50 == 0:
        ka = [x[0] for x in run if x[2] == "A"]; kb = [x[0] for x in run if x[2] == "B"]
        r = dict(step=step, kl_A=round(sum(ka) / max(len(ka), 1), 4), kl_B=round(sum(kb) / max(len(kb), 1), 4),
                 top1=round(sum(x[1] for x in run) / len(run), 4), sec_per_step=round((time.time() - t0) / step, 2))
        hist["train"].append(r); log("train", r); run = []
    if step % VAL_EVERY == 0 or step == STEPS:
        v = validate(); hist["val"].append(dict(step=step, **v)); log("step", step, "val", v)
        save(step); json.dump(hist, open(LOGJ, "w"), indent=1)
log("KLDONE", OUT)
