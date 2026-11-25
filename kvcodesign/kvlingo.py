# Paper-faithful reproduction of the KL-trained KV translator: KV-Lingo (Castin et al., Apple, arXiv 2609.32610) and the
# GQA replication in Hybrid-KV Part 1 (Kvedaras & Rasymas, arXiv 2611.12062 §5).  Replaces train_kl2.py for alignment
# with the published numbers; differences from train_kl2.py are the paper settings below.
#   capture : PRE-NORM keys (k_proj output, before k_norm and RoPE) and values (KV-Lingo default; Hybrid-KV point 1).
#             Injection runs the target's own k_norm and RoPE on the translated keys.
#   map     : per target layer, K and V separately, head-mixing full matrix over the concatenation of NU source layers
#             chosen by relative depth (KV-Lingo §3.1); bias only if BIAS=1 (Hybrid-KV y = Wx + b; KV-Lingo none).
#   stage 1 : closed form, fp64 second moments over every prefix position of FIT_N conversations (KV-Lingo App. E: 400
#             conversations), eigendecomposition with relative eigenvalue tolerance 1e-8 (no ridge).
#   stage 2 : KL(target on its own prefix cache || target on the translated cache), averaged over the continuation
#             (<= 2048 tokens), then over 8 conversations per step; AdamW, wd 0, cosine with 5% warmup, clip 1.0,
#             translator fp32, models bf16 frozen.  The source prefills all prefix tokens but the last; the target runs
#             the held-back token itself, then the continuation (KV-Lingo App. C).
#   data    : Nemotron-SFT-Instruction-Following-Chat-v2 reasoning_off (reasoning_on is not cached on the server --
#             documented deviation), split at the start of a random assistant turn, prefix <= MAXP (KV-Lingo 16K,
#             Hybrid-KV 8K), 64 validation conversations.
#   eval    : ARC-Challenge (1172), MMLU-Pro (1400, 100 per category), CoQA (256), 6-digit needle (2/4/8/16K, 20 each);
#             greedy, reasoning off; methods: target native, source native, closed form (CF), self-distilled (SD).
# env: PAIR=17to4|4to17|8to4|4to8 NU BIAS LR STEPS MAXP FIT_N TAG SDEV TDEV MODE=fit,train,eval EVAL_TASKS EVAL_N
import os, sys, json, math, time, random, re, hashlib
import torch
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

E = os.environ.get
PAIR = E("PAIR", "17to4")
NAMES = {"17": "Qwen3-1.7B", "4": "Qwen3-4B", "8": "Qwen3-8B-ms"}
SRC, TGT = (NAMES[x] for x in PAIR.split("to"))
NU, BIAS = int(E("NU", 1)), int(E("BIAS", 0))
LR, STEPS, BATCH, VAL_EVERY = float(E("LR", 1e-5)), int(E("STEPS", 5000)), int(E("BATCH", 8)), int(E("VAL_EVERY", 500))
MAXP, MAXC, FIT_N = int(E("MAXP", 16384)), 2048, int(E("FIT_N", 400))
TAG = E("TAG", f"{PAIR}_nu{NU}_b{BIAS}_lr{E('LR', '1e-5')}")
SDEV, TDEV = E("SDEV", "cuda:0"), E("TDEV", "cuda:0")
MODE = E("MODE", "fit,train,eval").split(",")
HOME = os.path.expanduser("~")
M, OUT = f"{HOME}/kvswitch/models", f"{HOME}/kvcodesign/kvlingo"
os.makedirs(OUT, exist_ok=True)
CF_PATH, SD_PATH, RES = f"{OUT}/cf_{PAIR}_nu{NU}_b{BIAS}.pt", f"{OUT}/sd_{TAG}.pt", f"{OUT}/res_{TAG}.json"
bf = torch.bfloat16
t0 = time.time()
log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{(time.time() - t0) / 60:.1f} min]", *a, flush=True)

tok = AutoTokenizer.from_pretrained(f"{M}/Qwen3-4B")
enc = lambda s: tok(s, add_special_tokens=False).input_ids
HDR = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


# ---------------------------------------------------------------- data
def conversations():
    path = f"{HOME}/kvcodesign/data/kvlingo_convs.json"
    if os.path.exists(path):
        return json.load(open(path))
    import glob
    src = glob.glob(f"{HOME}/.cache/huggingface/hub/datasets--nvidia--Nemotron-SFT-Instruction-Following-Chat-v2/snapshots/*/data/reasoning_off.jsonl")[0]
    evalu = {x["uuid"] for v in json.load(open(f"{HOME}/kvcodesign/data/klchat.json")).values() for x in v}
    out, need = [], 64 + FIT_N + 40000
    for i, line in enumerate(open(src)):
        if i < 200000:
            continue
        d = json.loads(line)
        if d.get("uuid") in evalu:
            continue
        m = [x for x in d["messages"] if x.get("content")]
        turns = [j for j, x in enumerate(m) if x["role"] == "assistant" and j > 0]
        if not turns:
            continue
        j = turns[int(hashlib.md5(d["uuid"].encode()).hexdigest(), 16) % len(turns)]   # start of a random assistant turn
        p = enc("".join(f"<|im_start|>{x['role']}\n{x['content']}<|im_end|>\n" for x in m[:j]) + HDR)
        c = enc(m[j]["content"] + "<|im_end|>")[:MAXC]
        if len(p) > 16384 or len(p) < 2 or len(c) < 2:
            continue
        out.append(dict(uuid=d["uuid"], p=p, c=c))
        if len(out) >= need:
            break
    d = dict(val=out[:64], fit=out[64:64 + FIT_N], train=out[64 + FIT_N:])
    json.dump(d, open(path, "w"))
    return d


# ---------------------------------------------------------------- models and cache plumbing
def load(name, dev):
    return AutoModelForCausalLM.from_pretrained(f"{M}/{name}", torch_dtype=bf).to(dev).eval().requires_grad_(False)


def prenorm_kv(model, ids, last=None):
    """Pre-norm keys (k_proj output) and values (v_proj output) of every layer, [n, H*D] each, for ids[:, :last]."""
    ks, vs, hs = {}, {}, []
    for i, l in enumerate(model.model.layers):
        hs.append(l.self_attn.k_proj.register_forward_hook(lambda m_, a, o, i=i: ks.__setitem__(i, o[0])))
        hs.append(l.self_attn.v_proj.register_forward_hook(lambda m_, a, o, i=i: vs.__setitem__(i, o[0])))
    try:
        with torch.no_grad():
            model.model(ids if last is None else ids[:, :last], use_cache=True)   # same kernels as the native prefill
    finally:
        for h in hs:
            h.remove()
    L = len(model.model.layers)
    return [ks[i] for i in range(L)], [vs[i] for i in range(L)]


def rot(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], -1)


def inject(model, t, yk, yv, cos, sin):
    """Translated pre-norm K/V of target layer t ([n, H*D], fp32) -> cache tensors [1, H, n, D] (bf16): the target's
    own k_norm, then its RoPE (same arithmetic as Qwen3Attention)."""
    D, n = model.config.head_dim, yk.shape[0]
    k = model.model.layers[t].self_attn.k_norm(yk.view(n, -1, D).to(bf)).transpose(0, 1)[None]   # bf16, as the target
    cos, sin = cos.to(bf), sin.to(bf)
    k = k * cos[:, None] + rot(k) * sin[:, None]
    v = yv.view(n, -1, D).transpose(0, 1)[None]
    return k, v.to(bf)


def src_layers(t, Ls, Lt):
    c = round(t * (Ls - 1) / (Lt - 1)) if Lt > 1 else 0
    lo = min(max(0, c - (NU - 1) // 2), Ls - NU)
    return list(range(lo, lo + NU))


class Prefix:   # stateless one-layer cache: the translated (or native) prefix K/V prepended to the new tokens' K/V
    def __init__(self, k, v):
        self.k, self.v = k, v

    def update(self, k, v, layer_idx, cache_kwargs=None):
        return torch.cat([self.k, k], 2), torch.cat([self.v, v], 2)


def run_target(T, kv, n, ids, grad=False):
    """Target decoder over ids (positions n..n+m-1) reading the prefix kv [(k, v)] of n positions; returns final hidden."""
    m, dev = ids.shape[1], ids.device
    h = T.model.embed_tokens(ids)
    pos = torch.arange(n, n + m, device=dev)[None]
    cos, sin = T.model.rotary_emb(h, pos)
    mask = (torch.arange(n + m, device=dev)[None] <= pos[0][:, None])[None, None]

    def layer(i, h, k, v):
        return T.model.layers[i](h, attention_mask=mask, position_ids=pos, past_key_value=Prefix(k, v), use_cache=True,
                                 cache_position=pos[0], position_embeddings=(cos, sin))
    for i, (k, v) in enumerate(kv):
        h = checkpoint(layer, i, h, k, v, use_reentrant=False) if grad else layer(i, h, k, v)
        h = h[0] if isinstance(h, tuple) else h
    return T.model.norm(h)


class Translator(torch.nn.Module):
    def __init__(self, Ls, Lt, din, dout):
        super().__init__()
        self.ch = [src_layers(t, Ls, Lt) for t in range(Lt)]
        self.Wk = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(NU * din, dout)) for _ in range(Lt)])
        self.Wv = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(NU * din, dout)) for _ in range(Lt)])
        self.bk = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(dout), requires_grad=bool(BIAS)) for _ in range(Lt)])
        self.bv = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(dout), requires_grad=bool(BIAS)) for _ in range(Lt)])

    def layer(self, t, sk, sv):
        xk = torch.cat([sk[s] for s in self.ch[t]], -1).float()
        xv = torch.cat([sv[s] for s in self.ch[t]], -1).float()
        return xk @ self.Wk[t] + self.bk[t], xv @ self.Wv[t] + self.bv[t]

    def translate(self, T, sk, sv, grad=False):
        n = sk[0].shape[0]
        dev = self.Wk[0].device
        cos, sin = T.model.rotary_emb(torch.zeros(1, n, 1, device=dev, dtype=torch.float32), torch.arange(n, device=dev)[None])
        out = []
        for t in range(len(self.Wk)):
            def f(t=t):
                yk, yv = self.layer(t, sk, sv)
                return inject(T, t, yk, yv, cos, sin)
            out.append(checkpoint(lambda t=t: f(t), use_reentrant=False) if grad else f())
        return out


# ---------------------------------------------------------------- stage 1: closed form
def fit(S, T, data):
    Ls, Lt = len(S.model.layers), len(T.model.layers)
    din = S.config.num_key_value_heads * S.config.head_dim
    dout = T.config.num_key_value_heads * T.config.head_dim
    tr = Translator(Ls, Lt, din, dout).to(TDEV)
    a = NU * din + BIAS
    XX = {kind: [torch.zeros(a, a, dtype=torch.float64, device=TDEV) for _ in range(Lt)] for kind in "kv"}
    XY = {kind: [torch.zeros(a, dout, dtype=torch.float64, device=TDEV) for _ in range(Lt)] for kind in "kv"}
    YS = {kind: [torch.zeros(dout, dtype=torch.float64, device=TDEV) for _ in range(Lt)] for kind in "kv"}   # sum y per dim
    YQ = {kind: [torch.zeros((), dtype=torch.float64, device=TDEV) for _ in range(Lt)] for kind in "kv"}     # sum y^2
    npos = 0
    for x in data:
        ids = torch.tensor([x["p"]])
        sk, sv = prenorm_kv(S, ids.to(SDEV), last=len(x["p"]) - 1)
        tk, tv = prenorm_kv(T, ids.to(TDEV), last=len(x["p"]) - 1)
        n = tk[0].shape[0]; npos += n
        for t in range(Lt):
            for kind, src, y in (("k", sk, tk[t]), ("v", sv, tv[t])):
                X = torch.cat([src[s].to(TDEV) for s in tr.ch[t]], -1).double()
                if BIAS:
                    X = torch.cat([X, torch.ones(n, 1, dtype=torch.float64, device=TDEV)], -1)
                Y = y.double()
                XX[kind][t] += X.T @ X; XY[kind][t] += X.T @ Y
                YS[kind][t] += Y.sum(0); YQ[kind][t] += (Y * Y).sum()
    r2 = {"k": [], "v": []}
    with torch.no_grad():
        for t in range(Lt):
            for kind, W, b in (("k", tr.Wk, tr.bk), ("v", tr.Wv, tr.bv)):
                lam, Q = torch.linalg.eigh(XX[kind][t])
                keep = lam > 1e-8 * lam.max()
                Wf = (Q[:, keep] / lam[keep]) @ (Q[:, keep].T @ XY[kind][t])
                sse = YQ[kind][t] - 2 * (Wf * XY[kind][t]).sum() + (Wf * (XX[kind][t] @ Wf)).sum()
                sst = YQ[kind][t] - (YS[kind][t] ** 2).sum() / npos
                r2[kind].append(float(1 - sse / sst))
                W[t].copy_(Wf[:NU * din].float())
                if BIAS:
                    b[t].copy_(Wf[NU * din].float())
    torch.save(dict(state=tr.state_dict(), ch=tr.ch, r2=r2, npos=npos), CF_PATH)
    log("FITDONE positions", npos, "R2 K", round(sum(r2["k"]) / Lt, 3), "V", round(sum(r2["v"]) / Lt, 3))
    return tr


def load_tr(S, T, path):
    Ls, Lt = len(S.model.layers), len(T.model.layers)
    din = S.config.num_key_value_heads * S.config.head_dim
    dout = T.config.num_key_value_heads * T.config.head_dim
    tr = Translator(Ls, Lt, din, dout).to(TDEV)
    tr.load_state_dict(torch.load(path, map_location=TDEV)["state"])
    return tr


# ---------------------------------------------------------------- stage 2: KL distillation
def kl_loss(S, T, tr, x, grad=True):
    p, c = x["p"], x["c"]
    pi = torch.tensor([p])
    ci = torch.tensor([[p[-1]] + c[:-1]], device=TDEV)   # held-back last prefix token, then the continuation
    n = len(p) - 1
    with torch.no_grad():
        nat = T.model(pi[:, :n].to(TDEV), use_cache=True).past_key_values
        nkv = [(nat.layers[i].keys, nat.layers[i].values) for i in range(len(T.model.layers))] if hasattr(nat, "layers") \
            else [(nat.key_cache[i], nat.value_cache[i]) for i in range(len(T.model.layers))]
        ref = run_target(T, nkv, n, ci)[0]
        del nat, nkv
        sk, sv = prenorm_kv(S, pi.to(SDEV), last=n)
        sk, sv = [a.to(TDEV) for a in sk], [a.to(TDEV) for a in sv]
    with torch.set_grad_enabled(grad):
        h = run_target(T, tr.translate(T, sk, sv, grad), n, ci, grad)[0]
        tot = 0
        for a in range(0, len(c), 512):
            def ck(h_, r_):
                lp = T.lm_head(h_).float().log_softmax(-1); lr = T.lm_head(r_).float().log_softmax(-1)
                return (lr.exp() * (lr - lp)).sum()
            tot = tot + (checkpoint(ck, h[a:a + 512], ref[a:a + 512], use_reentrant=False) if grad else ck(h[a:a + 512], ref[a:a + 512]))
        return tot / len(c)


def validate(S, T, tr, val):
    with torch.no_grad():
        return round(sum(float(kl_loss(S, T, tr, x, grad=False)) for x in val) / len(val), 4)


def train(S, T, tr, data):
    params = [p for p in tr.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
    hist = dict(config=dict(pair=PAIR, nu=NU, bias=BIAS, lr=LR, steps=STEPS, batch=BATCH, maxp=MAXP), val=[], train=[])
    hist["val"].append((0, validate(S, T, tr, data["val"]))); log("step 0 (closed form) val KL", hist["val"][-1][1])
    trn = [x for x in data["train"] if len(x["p"]) <= MAXP]
    rng, run, oom, warm = random.Random(0), [], 0, max(1, int(0.05 * STEPS))
    order = list(range(len(trn))); rng.shuffle(order); ptr = 0
    for step in range(1, STEPS + 1):
        f = min(1.0, step / warm) * 0.5 * (1 + math.cos(math.pi * max(0, step - warm) / max(1, STEPS - warm)))
        for g in opt.param_groups:
            g["lr"] = LR * f
        tot = 0.0
        for _ in range(BATCH):
            x = trn[order[ptr % len(order)]]; ptr += 1
            try:
                loss = kl_loss(S, T, tr, x) / BATCH
                loss.backward(); tot += float(loss)
            except torch.OutOfMemoryError:
                oom += 1; log("OOM prompt", len(x["p"]), "answer", len(x["c"]), "- skipped"); torch.cuda.empty_cache()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        run.append(tot)
        if step % 25 == 0:
            hist["train"].append((step, sum(run) / len(run)))
            log("step", step, "train KL", round(sum(run) / len(run), 4), "s/step", round((time.time() - t0) / step, 1), "oom", oom); run = []
        if step % VAL_EVERY == 0 or step == STEPS:
            hist["val"].append((step, validate(S, T, tr, data["val"]))); log("step", step, "val KL", hist["val"][-1][1])
            torch.save(dict(state=tr.state_dict(), ch=tr.ch, step=step), SD_PATH)
            hist["oom"] = oom; json.dump(hist, open(RES, "w"), indent=1)
    log("TRAINDONE", SD_PATH)
    return tr


# ---------------------------------------------------------------- evaluation (paper-style probes)
def norm_ans(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", re.sub(r"[^\w\s]", " ", s))
    return " ".join(s.split())


def tok_f1(p, g):
    p, g = norm_ans(p).split(), norm_ans(g).split()
    if not p or not g:
        return float(p == g)
    from collections import Counter
    c = sum((Counter(p) & Counter(g)).values())
    if c == 0:
        return 0.0
    pr, rc = c / len(p), c / len(g)
    return 2 * pr * rc / (pr + rc)


def user_prompt(text):
    return enc(f"<|im_start|>user\n{text}<|im_end|>\n" + HDR)


def eval_items(task, n):
    from datasets import load_dataset
    items = []
    if task == "arc_c":
        for r in load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"):
            opts = "\n".join(f"{l}. {t}" for l, t in zip(r["choices"]["label"], r["choices"]["text"]))
            items.append(dict(p=user_prompt(f"{r['question']}\n\n{opts}\n\nAnswer with the label of the correct option only."),
                              gold=r["answerKey"], labels=r["choices"]["label"], mx=16))
    elif task == "mmlu_pro":
        rows = list(load_dataset("TIGER-Lab/MMLU-Pro", split="test"))
        by = {}
        for r in rows:
            by.setdefault(r["category"], []).append(r)
        L = "ABCDEFGHIJ"
        for cat in sorted(by):
            rs = by[cat][:]; random.Random(0).shuffle(rs)
            for r in rs[:100]:
                opts = "\n".join(f"{L[i]}. {o}" for i, o in enumerate(r["options"]))
                items.append(dict(p=user_prompt(f"{r['question']}\n\n{opts}\n\nAnswer with the letter of the correct option only."),
                                  gold=r["answer"], labels=list(L[:len(r["options"])]), cat=cat, mx=16))
    elif task == "coqa":
        ds = list(load_dataset("stanfordnlp/coqa", split="validation"))
        rng = random.Random(0)
        for r in ds:
            qs, ans = r["questions"], r["answers"]["input_text"]
            j = rng.randrange(len(qs))
            hist = "".join(f"Q: {qs[i]}\nA: {ans[i]}\n" for i in range(j))
            p = user_prompt(f"{r['story']}\n\nAnswer the last question briefly, based on the story.\n\n{hist}Q: {qs[j]}")
            if len(p) <= 1024:
                items.append(dict(p=p, gold=ans[j], mx=128))
        random.Random(1).shuffle(items)
    elif task.startswith("needle"):
        import tasks
        arts = [" ".join(x[0]["article"].split()) for x in tasks.quality_docs()]
        for L_ in (2048, 4096, 8192, 16384):
            for s in tasks.passkey_samples(enc, 20, L_, arts):
                items.append(dict(p=s["h"] + s["q"], gold=s["gold"], mx=16, length=L_))
    return items[:n] if n else items


def score(task, it, out):
    if task in ("arc_c", "mmlu_pro"):
        m = re.search(r"\b([A-J1-9])\b", out.strip())
        return dict(score=float(bool(m) and m.group(1) == it["gold"]))
    if task == "coqa":
        return dict(score=tok_f1(out, it["gold"]), gold_in=float(norm_ans(it["gold"]) in norm_ans(out)))
    return dict(score=float(it["gold"] in re.findall(r"\d{6}", out)))


def generate(T, kv, n, last, mx):
    """Greedy decode: the target reads `last` (the held-back prefix token) on prefix kv of n positions."""
    from transformers import DynamicCache
    cache = DynamicCache()
    for i, (k, v) in enumerate(kv):
        cache.update(k, v, i)
    ids, out = torch.tensor([[last]], device=TDEV), []
    for _ in range(mx):
        lo = T(ids, past_key_values=cache, use_cache=True, cache_position=torch.tensor([n + len(out)], device=TDEV)).logits[0, -1]
        nxt = int(lo.argmax())
        if nxt in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id):
            break
        out.append(nxt); ids = torch.tensor([[nxt]], device=TDEV)
    return tok.decode(out)


def native_kv(model, ids):
    nat = model.model(ids, use_cache=True).past_key_values
    L = len(model.model.layers)
    return [(nat.layers[i].keys, nat.layers[i].values) for i in range(L)] if hasattr(nat, "layers") \
        else [(nat.key_cache[i], nat.value_cache[i]) for i in range(L)]


def evaluate(S, T, trs):
    tasks_ = E("EVAL_TASKS", "arc_c,mmlu_pro,coqa,needle").split(",")
    res = json.load(open(RES)) if os.path.exists(RES) else {}
    res.setdefault("eval", {})
    for task in tasks_:
        items = eval_items(task, int(E("COQA_N", 256)) if task == "coqa" and not E("EVAL_N") else int(E("EVAL_N", 0)))
        meths = E("EVAL_METHODS", "tgt_native,src_native," + ",".join(trs)).split(",")
        rows = {m: [] for m in meths}
        for i, it in enumerate(items):
            p, n = it["p"], len(it["p"]) - 1
            with torch.no_grad():
                if "tgt_native" in rows:
                    kvT = native_kv(T, torch.tensor([p[:n]], device=TDEV))
                    rows["tgt_native"].append(score(task, it, generate(T, kvT, n, p[-1], it["mx"]))); del kvT
                if "src_native" in rows:
                    kvS = native_kv(S, torch.tensor([p[:n]], device=SDEV))
                    rows["src_native"].append(score(task, it, generate_src(S, kvS, n, p[-1], it["mx"]))); del kvS
                sk, sv = prenorm_kv(S, torch.tensor([p], device=SDEV), last=n)
                sk, sv = [a.to(TDEV) for a in sk], [a.to(TDEV) for a in sv]
                for name, tr in trs.items():
                    if name in rows:
                        rows[name].append(score(task, it, generate(T, tr.translate(T, sk, sv), n, p[-1], it["mx"])))
            for m in rows:
                rows[m][-1].update({k: it[k] for k in ("length", "cat") if k in it})
            if (i + 1) % 50 == 0 or i + 1 == len(items):
                log(task, i + 1, "/", len(items), {m: round(sum(r["score"] for r in v) / len(v), 4) for m, v in rows.items()})
        summ = {}
        for m, v in rows.items():
            s = dict(n=len(v), mean=round(sum(r["score"] for r in v) / len(v), 4))
            if task == "coqa":
                s["gold_in"] = round(sum(r["gold_in"] for r in v) / len(v), 4)
            if task == "mmlu_pro":   # macro over categories (KV-Lingo Table 4)
                cats = {}
                for r in v:
                    cats.setdefault(r["cat"], []).append(r["score"])
                s["macro"] = round(sum(sum(c) / len(c) for c in cats.values()) / len(cats), 4)
            if task == "needle":
                s["by_len"] = {L_: round(sum(r["score"] for r in v if r["length"] == L_) / max(1, sum(r["length"] == L_ for r in v)), 3) for L_ in (2048, 4096, 8192, 16384)}
            summ[m] = s
        res["eval"].setdefault(task, {}).update(summ)
        json.dump(res, open(RES, "w"), indent=1)
        log("EVALDONE", task, summ)


def generate_src(S, kv, n, last, mx):
    global TDEV
    keep = TDEV; TDEV = SDEV
    try:
        return generate(S, kv, n, last, mx)
    finally:
        TDEV = keep


if __name__ == "__main__":
    data = conversations()
    log("conversations: train", len(data["train"]), "fit", len(data["fit"]), "val", len(data["val"]),
        "mean prefix", sum(len(x["p"]) for x in data["train"]) // len(data["train"]),
        "mean answer", sum(len(x["c"]) for x in data["train"]) // len(data["train"]))
    if MODE == ["data"]:
        sys.exit()
    if MODE == ["check"]:   # injection path must reproduce the target's own cache: pre-norm capture -> k_norm -> RoPE
        T = load(TGT, TDEV)
        x = data["val"][0]; p = x["p"]; n = len(p) - 1
        ci = torch.tensor([[p[-1]] + x["c"][:-1]], device=TDEV)
        with torch.no_grad():
            ref = run_target(T, native_kv(T, torch.tensor([p[:n]], device=TDEV)), n, ci)[0]
            tk, tv = prenorm_kv(T, torch.tensor([p], device=TDEV), last=n)
            cos, sin = T.model.rotary_emb(torch.zeros(1, n, 1, device=TDEV), torch.arange(n, device=TDEV)[None])
            kv = [inject(T, t, tk[t].float(), tv[t].float(), cos, sin) for t in range(len(tk))]
            nat = native_kv(T, torch.tensor([p[:n]], device=TDEV))
            dl = [float((a[0].float() - b[0].float()).abs().max()) for a, b in zip(kv, nat)]; dk = max(dl)
            log("per-layer max |K diff| (first 4, last)", dl[:4], dl[-1])
            h = run_target(T, kv, n, ci)[0]
            lr_, lp = T.lm_head(ref).float().log_softmax(-1), T.lm_head(h).float().log_softmax(-1)
            log("CHECK n", n, "max |K diff|", round(dk, 4), "KL", float((lr_.exp() * (lr_ - lp)).sum(-1).mean()),
                "top1 agree", float((lr_.argmax(-1) == lp.argmax(-1)).float().mean()))
        sys.exit()
    S, T = load(SRC, SDEV), load(TGT, TDEV)
    pool, fitset, ntok = data["fit"] + data["train"], [], 0
    for x in pool:   # FIT_TOK prefix positions (KV-Lingo ~2.8M, Hybrid-KV 0.55M)
        if ntok >= int(E("FIT_TOK", 2800000)):
            break
        fitset.append(x); ntok += len(x["p"]) - 1
    tr = fit(S, T, fitset) if "fit" in MODE or not os.path.exists(CF_PATH) else load_tr(S, T, CF_PATH)
    if "train" in MODE:
        tr = train(S, T, tr, data)
    if "eval" in MODE:
        trs = {"cf": load_tr(S, T, CF_PATH)}
        if os.path.exists(SD_PATH):
            trs["sd"] = load_tr(S, T, SD_PATH)
        evaluate(S, T, trs)
    log("KVLINGODONE")
