# "KL-trained translator" baseline: the recipe of KV-Lingo (arXiv 2609.32610) and Hybrid-KV (arXiv 2611.12062, Part 1),
# re-implemented (neither releases code that we found).  Our code.
#   map   : per target layer a linear map over the concatenated K (resp. V) of nu source layers (all heads), keys in
#           content space (source RoPE stripped, target RoPE applied) -- the kvbridge format.  nu = 3, as in
#           Hybrid-KV's GQA pairs ("1 to 3 source layers").
#   stage 1 (closed form): the official kvbridge fit with top_k=3 (fit_kvbridge_dir.py TOPK=3) -> mappers/kvbridge3_{DIR}
#   stage 2 (this script): all map weights fine-tuned by KL(target reading its own cache || target reading the
#           translated cache), averaged over continuation tokens; both models frozen.  Hybrid-KV: 2000 steps on
#           Nemotron-SFT-Instruction-Following-Chat-v2 conversations, prefixes up to 8K.
# Data: the same chat file, conversations NOT among the evaluation ones (data/klchat.json uuids are excluded), prefix
#   length log-uniform in [512, MAXN], continuation = the next 128 tokens of the conversation (teacher-forced).
# Output: mappers/kvbridge3kl_{DIR}/ (kvbridge format, loadable by the official Mapper.load)
# env: DIR STEPS LR MAXN VAL_EVERY
import os, sys, json, math, time, random
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
HOME = os.path.expanduser("~")
sys.path.insert(0, f"{HOME}/kvbase/kvbridge")
from kvbridge import Mapper
E = os.environ.get
DIR = E("DIR", "17to4")
SRC_M, TGT_M = {"17to4": ("Qwen3-1.7B", "Qwen3-4B"), "4to17": ("Qwen3-4B", "Qwen3-1.7B")}[DIR]
STEPS, LR, MAXN, VAL_EVERY = int(E("STEPS", 2000)), float(E("LR", 1e-4)), int(E("MAXN", 8192)), int(E("VAL_EVERY", 250))
INIT, OUT = f"{HOME}/kvbase/mappers/kvbridge3_{DIR}", f"{HOME}/kvbase/mappers/kvbridge3kl_{DIR}"
dev, D = "cuda", 128
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
M = f"{HOME}/kvswitch/models"
tok = AutoTokenizer.from_pretrained(f"{M}/Qwen3-4B")
S = AutoModelForCausalLM.from_pretrained(f"{M}/{SRC_M}", torch_dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)
T = AutoModelForCausalLM.from_pretrained(f"{M}/{TGT_M}", torch_dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)
mp = Mapper.load(INIT, device=dev, dtype=torch.float32)
LT = len(mp.layers)
P = {f"{k}{l}": torch.nn.Parameter(e[k].float().clone()) for l, e in enumerate(mp.layers) for k in ("W_k", "W_v")}
inv = 1.0 / (1e6 ** (torch.arange(0, D, 2, device=dev, dtype=torch.float32) / D))
rot = lambda x: torch.cat([-x[..., D // 2:], x[..., :D // 2]], -1)


def translate(sc, n):   # same arithmetic as the official kvbridge.transfer, differentiable in the W's
    ang = torch.arange(n, device=dev, dtype=torch.float32)[:, None] * inv[None]
    c, s_ = torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)          # [n, D]
    sk = [(lambda k: (k * c - rot(k) * s_).transpose(0, 1).reshape(n, -1))(sc.layers[l].keys[0].float()) for l in range(len(sc.layers))]
    sv = [sc.layers[l].values[0].float().transpose(0, 1).reshape(n, -1) for l in range(len(sc.layers))]
    out = DynamicCache()
    for t, e in enumerate(mp.layers):
        ch = e["source_layers"]
        yk = (torch.cat([sk[s] for s in ch], -1) - e["mu_xk"]) @ P[f"W_k{t}"] + e["mu_yk"]
        yv = (torch.cat([sv[s] for s in ch], -1) - e["mu_xv"]) @ P[f"W_v{t}"] + e["mu_yv"]
        yk = yk.view(n, -1, D).transpose(0, 1); yv = yv.view(n, -1, D).transpose(0, 1)                # [H, n, D]
        yk = yk * c + rot(yk) * s_
        out.update(yk[None].to(torch.bfloat16), yv[None].to(torch.bfloat16), t)
    return out


# ---------------------------------------------------------------- data (held out from the evaluation conversations)
from huggingface_hub import hf_hub_download
evalu = {x["uuid"] for v in json.load(open(f"{HOME}/kvcodesign/data/klchat.json")).values() for x in v}
path = hf_hub_download("nvidia/Nemotron-SFT-Instruction-Following-Chat-v2", "data/reasoning_off.jsonl", repo_type="dataset")
convs = []
for i, line in enumerate(open(path)):
    if i < 200000:          # the evaluation sets were drawn from the start of the file; train on later rows only
        continue
    d = json.loads(line)
    if d.get("uuid") in evalu:
        continue
    m = [x for x in d["messages"] if x.get("content")]
    if sum(len(x["content"]) for x in m) < 2500:
        continue
    convs.append(tok("".join(f"<|im_start|>{x['role']}\n{x['content']}<|im_end|>\n" for x in m), add_special_tokens=False).input_ids[:MAXN + 128])
    if len(convs) >= 6000:
        break
VAL, TRAIN = convs[:24], convs[24:]
log("conversations", len(TRAIN), "train,", len(VAL), "val; mean tokens", sum(map(len, TRAIN)) // len(TRAIN))


def sample(rng, conv):
    n = min(len(conv) - 128, int(math.exp(rng.uniform(math.log(512), math.log(MAXN)))))
    return conv[:n], conv[n:n + 128]


def kl_loss(h, q, grad=True):
    with torch.no_grad():
        c = DynamicCache(); T.model(torch.tensor([h], device=dev), past_key_values=c, use_cache=True)
        ref = T(torch.tensor([q], device=dev), past_key_values=c, use_cache=True).logits[0].float().log_softmax(-1)
        del c
        sc = S.model(torch.tensor([h], device=dev), use_cache=True).past_key_values
    with torch.set_grad_enabled(grad):
        lp = T(torch.tensor([q], device=dev), past_key_values=translate(sc, len(h)), use_cache=True).logits[0].float().log_softmax(-1)
        return (ref.exp() * (ref - lp)).sum(-1).mean()


vr = random.Random(5)
VALS = [sample(vr, c) for c in VAL]


def validate():
    return round(sum(float(kl_loss(h, q, grad=False)) for h, q in VALS) / len(VALS), 4)


def save(step):
    for l, e in enumerate(mp.layers):
        e["W_k"], e["W_v"] = P[f"W_k{l}"].detach(), P[f"W_v{l}"].detach()
    mp.meta["method"] = ("kvbridge-format map, closed-form init (official kvbridge fit, top_k=3) + KL distillation through "
                         "the frozen target (KV-Lingo / Hybrid-KV recipe, re-implemented)")
    mp.meta["kl_steps"] = step
    mp.save(OUT)


opt = torch.optim.Adam(P.values(), lr=LR, betas=(0.9, 0.99))
hist = dict(config=dict(dir=DIR, steps=STEPS, lr=LR, maxn=MAXN), val=[(0, validate())], train=[])
log("step 0 (closed form) val KL", hist["val"][-1][1])
rng, run, t0 = random.Random(0), [], time.time()
for step in range(1, STEPS + 1):
    h, q = sample(rng, TRAIN[rng.randrange(len(TRAIN))])
    try:
        loss = kl_loss(h, q)
        loss.backward()
        f = min(1.0, step / 50) * 0.5 * (1 + math.cos(math.pi * step / STEPS))
        for g in opt.param_groups:
            g["lr"] = LR * f
        torch.nn.utils.clip_grad_norm_(P.values(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        run.append(float(loss))
    except torch.OutOfMemoryError:
        log("OOM at step", step, "prefix", len(h), "- skipped"); opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
    if step % 50 == 0 and run:
        hist["train"].append((step, sum(run) / len(run))); log("step", step, "train KL", round(sum(run) / len(run), 4), "s/step", round((time.time() - t0) / step, 2)); run = []
    if step % VAL_EVERY == 0 or step == STEPS:
        hist["val"].append((step, validate())); log("step", step, "val KL", hist["val"][-1][1])
        save(step); json.dump(hist, open(f"{HOME}/kvcodesign/results/train_dense_kl_{DIR}.json", "w"), indent=1)
log("DENSEKLDONE", OUT)
