# P4: Llama-3.2-3B-Instruct -> Qwen3.5-2B (GDN hybrid, partial RoPE, different tokenizer).  B5 (2611.12062-style):
#  * full-attention layers: KV map (de-RoPE Llama -> ridge -> RoPE on B's first rotary_dim dims);
#  * GDN layers: ridge-predict B's per-token layer input (post input_layernorm) from A's KV, then B replays its own
#    linear_attn (projections + causal conv + gated-delta recurrence) to produce conv/recurrent states.
#  * tokens aligned by character coverage; template tokens map to A's template tokens proportionally.
# Both native and B5 use the same HF forward (DynamicCache) for the new input and decoding.
# env: MODE=fit|eval|time  (eval: TASK N METHODS)
import os, json, random, math, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from ks import Model, ROOT, log, rope, derope, kv_bytes
import tasks as TK
from xalign import _pieces

E = os.environ.get
MODE = E("MODE", "fit")
dev = "cuda"
A = Model("Llama-3.2-3B-Instruct")
BP = f"{ROOT}/models/Qwen3.5-2B"
Btok = AutoTokenizer.from_pretrained(BP)
Bm = AutoModelForCausalLM.from_pretrained(BP, dtype=torch.bfloat16).to(dev).eval()
for p in Bm.parameters():
    p.requires_grad_(False)
tc = Bm.config
LT = tc.layer_types
LB = tc.num_hidden_layers
FULL = [l for l in range(LB) if LT[l] == "full_attention"]
HKB, DB = tc.num_key_value_heads, tc.head_dim
H = tc.hidden_size
win = lambda c, k, L: list(range(min(max(c - k // 2, 0), L - k), min(max(c - k // 2, 0), L - k) + k))
KS = 3
src = [win(round(l * (A.L - 1) / (LB - 1)), KS, A.L) for l in range(LB)]
FD = KS * 2 * A.Hk * A.D + 1


class BF:  # minimal B family shim for alignment
    family = "qwen3"
    tok = Btok
    def bos(self): return []


Bf = BF()


def align(msgs):
    """idsA, idsB, mapB2A: every B token gets a source A token (template tokens: proportional within the piece)."""
    ia, ib, mp = list(A.bos()), [], []
    for role, text in msgs:
        ha, c, fa = _pieces(A, role, text)
        hb, _, fb = _pieces(Bf, role, text)
        for pa, pb in ((ha, hb), (None, None), (fa, fb)):
            if pa is None:
                ea = A.tok(c, add_special_tokens=False, return_offsets_mapping=True)
                eb = Btok(c, add_special_tokens=False, return_offsets_mapping=True)
                a0 = len(ia); ia += ea.input_ids
                ends = [e for _, e in ea.offset_mapping]
                j = 0
                for tid, (s, e) in zip(eb.input_ids, eb.offset_mapping):
                    last = max(e - 1, s)
                    while j < len(ends) - 1 and ends[j] <= last:
                        j += 1
                    ib.append(tid); mp.append(a0 + j if ends else max(a0 - 1, 0))
                continue
            xa = A.tok(pa, add_special_tokens=False).input_ids
            xb = Btok(pb, add_special_tokens=False).input_ids
            a0 = len(ia); ia += xa
            for i, t in enumerate(xb):
                ib.append(t); mp.append(a0 + round(i * (len(xa) - 1) / max(len(xb) - 1, 1)))
    return ia, ib, torch.tensor(mp)


def b_turn(role, text):
    return Btok(f"<|im_start|>{role}\n{text}<|im_end|>\n", add_special_tokens=False).input_ids


B_GEN = Btok("<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False).input_ids
B_STOP = {Btok.convert_tokens_to_ids("<|im_end|>"), Btok.eos_token_id, Btok.convert_tokens_to_ids("<|endoftext|>")}


def b_cos(n0, n1):
    pos = torch.arange(n0, n1, device=dev)[None, None].expand(3, 1, -1)
    cos, sin = Bm.model.rotary_emb(torch.zeros(1, device=dev, dtype=torch.float32), pos)
    return cos[0].float(), sin[0].float()  # [N, rd]


@torch.no_grad()
def b_fwd(ids, cache, n0, want="last"):
    T = len(ids)
    out = Bm(input_ids=torch.tensor([ids], device=dev), past_key_values=cache, use_cache=True,
             position_ids=torch.arange(n0, n0 + T, device=dev)[None], logits_to_keep=0 if want == "all" else 1)
    return out.logits.float()


def a_feats(cA, l, posA):
    cos, sin = A.cossin(0, int(posA.max()) + 1)
    cos, sin = cos[posA], sin[posA]
    xs = []
    for s in src[l]:
        xs.append(derope(cA.k[s][:, posA].float(), cos, sin).to(torch.bfloat16))
        xs.append(cA.v[s][:, posA])
    N = len(posA)
    X = torch.cat([x.transpose(0, 1).reshape(N, -1) for x in xs], -1)
    return torch.cat([X, torch.ones_like(X[:, :1])], -1)


# ---------------------------------------------------------------- fit
if MODE == "fit":
    NSEQ, SEQ = int(E("NSEQ", 48)), int(E("SEQ", 2048))
    uc = json.load(open(f"{ROOT}/data/ultrachat_train.json")); ucv = json.load(open(f"{ROOT}/data/ultrachat.json"))

    def seq(rng, d):
        msgs = []
        while True:
            conv = d[rng.randrange(len(d))]
            msgs += [(m["role"], m["content"]) for m in conv]
            ia, ib, mp = align(msgs)
            if len(ib) >= SEQ:
                mp = mp[:SEQ]
                return ia[:int(mp.max()) + 1], ib[:SEQ], mp

    hooks, grab = [], {}
    for l in range(LB):
        layer = Bm.model.layers[l]
        if LT[l] == "full_attention":
            hooks.append(layer.self_attn.k_norm.register_forward_hook(lambda m, i, o, l=l: grab.__setitem__(("k", l), o[0])))
            hooks.append(layer.self_attn.v_proj.register_forward_hook(lambda m, i, o, l=l: grab.__setitem__(("v", l), o[0])))
        else:
            hooks.append(layer.input_layernorm.register_forward_hook(lambda m, i, o, l=l: grab.__setitem__(("x", l), o[0])))

    def run(data, fn):
        for ia, ib, mp in data:
            cA = A.new_cache(len(ia)); A.forward(torch.tensor([ia], device=dev), cA, want="hidden")
            grab.clear()
            with torch.no_grad():
                Bm(input_ids=torch.tensor([ib], device=dev), use_cache=False)
            posA = mp.to(dev)
            for l in range(LB):
                X = a_feats(cA, l, posA).float()
                Y = torch.cat([grab[("k", l)].reshape(len(ib), -1), grab[("v", l)].reshape(len(ib), -1)], -1).float()                     if LT[l] == "full_attention" else grab[("x", l)].float()
                fn(l, X, Y)
            del cA

    S = {}
    def acc(l, X, Y):
        if l not in S:
            S[l] = [torch.zeros(FD, FD, device=dev, dtype=torch.float64), torch.zeros(FD, Y.shape[1], device=dev, dtype=torch.float64)]
        S[l][0] += (X.T @ X).double(); S[l][1] += (X.T @ Y).double()

    rng = random.Random(0)
    tr = [seq(rng, uc) for _ in range(NSEQ)]
    va = [seq(random.Random(1), ucv) for _ in range(6)]
    log("aligned data ready")
    run(tr, acc)
    LAMS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    Ws = {}
    for l in range(LB):
        Ws[l] = []
        for lam in LAMS:
            Aa = S[l][0] + torch.eye(FD, device=dev, dtype=torch.float64) * lam * S[l][0].diagonal().mean()
            Aa[-1, -1] = S[l][0][-1, -1]
            Ws[l].append(torch.linalg.solve(Aa, S[l][1]).float())
            del Aa
        del S[l]
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    E_ = {l: [[0.0, 0.0] for _ in LAMS] for l in range(LB)}
    def ev(l, X, Y):
        for i, Wl in enumerate(Ws[l]):
            E_[l][i][0] += ((X @ Wl - Y) ** 2).sum().item(); E_[l][i][1] += (Y ** 2).sum().item()
    run(va, ev)
    W, rep = {}, {}
    for l in range(LB):
        errs = [a / b for a, b in E_[l]]
        i = min(range(len(LAMS)), key=lambda j: errs[j])
        W[l] = Ws[l][i].to(torch.bfloat16).cpu(); rep[l] = dict(type=LT[l], val_rel_err=errs[i], lam=LAMS[i])
        log(l, rep[l])
    torch.save(dict(W=W, src=src), f"{ROOT}/maps/P4.pt")
    json.dump(rep, open(f"{ROOT}/results/fit_P4.json", "w"), indent=1)
    log("FITDONE")
    raise SystemExit

# ---------------------------------------------------------------- translate (B5)
MP4 = torch.load(f"{ROOT}/maps/P4.pt")
W4 = {l: w.to(dev) for l, w in MP4["W"].items()}


@torch.no_grad()
def translate(cA, ib, mp, replay=True, timer=None):
    cache = DynamicCache(config=tc)
    N = len(ib)
    posA = mp.to(dev)
    cos, sin = b_cos(0, N)
    rd = cos.shape[-1]
    t = {"kv": 0.0, "replay": 0.0}
    for l in range(LB):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record()
        Y = a_feats(cA, l, posA) @ W4[l]
        if LT[l] == "full_attention":
            k = Y[:, :HKB * DB].view(N, HKB, DB).transpose(0, 1).float()
            v = Y[:, HKB * DB:].view(N, HKB, DB).transpose(0, 1)
            k = torch.cat([rope(k[..., :rd], cos, sin), k[..., rd:]], -1).to(torch.bfloat16)
            cache.update(k[None], v[None].contiguous(), l)
            key = "kv"
        else:
            if replay:
                Bm.model.layers[l].linear_attn(Y[None], cache_params=cache)
            key = "replay"
        e1.record(); torch.cuda.synchronize(); t[key] += e0.elapsed_time(e1)
    if timer is not None:
        timer.update(t)
    return cache


def native(ib):
    cache = DynamicCache(config=tc)
    b_fwd(ib, cache, 0)
    return cache


def gen(lg, cache, n0, mx):
    out, nxt = [], int(lg[0, -1].argmax())
    for i in range(mx):
        if nxt in B_STOP:
            break
        out.append(nxt)
        nxt = int(b_fwd([nxt], cache, n0 + i)[0, -1].argmax())
    return out


# ---------------------------------------------------------------- eval
if MODE == "eval":
    TASK, NS = E("TASK"), int(E("N", 100))
    METHODS = E("METHODS", "native,none,B5,B5noreplay").split(",")
    OUT = f"{ROOT}/results/q_P4_{TASK}.json"
    S = {"uc": lambda: TK.ultrachat(NS), "coqa1": lambda: TK.coqa(NS, 1), "lb8k": lambda: TK.longbench(A.tok, NS // 4, 8192),
         "niah8k": lambda: TK.niah(A.tok, NS, 8192), "gsm": lambda: TK.gsm8k(NS)}[TASK]()
    res = json.load(open(OUT)) if os.path.exists(OUT) else dict(cfg=dict(TASK=TASK, METHODS=METHODS), samples=[])
    done = {s["id"] for s in res["samples"]}
    for si, s in enumerate(S):
        if s["id"] in done:
            continue
        ia, ib, mp = align(s["hist"])
        N = len(ib)
        cA = A.new_cache(len(ia)); A.forward(torch.tensor([ia], device=dev), cA, want="hidden")
        rec = dict(id=s["id"], N=N, m={})
        ref = None
        for m in METHODS:
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); e0.record()
            if m == "native":
                cache, n0 = native(ib), N
            elif m == "none":
                cache, n0 = DynamicCache(config=tc), 0
            else:
                cache, n0 = translate(cA, ib, mp, replay=(m == "B5")), N
            r = {}
            if s["metric"] == "kl":
                new = b_turn("user", s["user"]) + B_GEN
                rep = Btok(s["reply"], add_special_tokens=False).input_ids[:256]
                lg = b_fwd(new + rep, cache, n0, "all")
                lp = lg[0, len(new) - 1:-1].log_softmax(-1)
                if m == "native":
                    ref = lp
                else:
                    r["kl"] = (ref.exp() * (ref - lp)).sum(-1).mean().item()
                    r["top1"] = (ref.argmax(-1) == lp.argmax(-1)).float().mean().item()
            else:
                q, gold, mx = s["turns"][0]
                new = b_turn("user", q) + B_GEN
                lg = b_fwd(new, cache, n0)
                e1.record(); torch.cuda.synchronize()
                r["t_switch_ms"] = e0.elapsed_time(e1); r["peak_GB"] = torch.cuda.max_memory_allocated() / 2**30
                pred = Btok.decode(gen(lg, cache, n0 + len(new), mx), skip_special_tokens=True).strip()
                r["turns"] = [dict(pred=pred, score=TK.score(s["metric"], pred, gold))]
            rec["m"][m] = r
            del cache
        del cA
        res["samples"].append(rec)
        if len(res["samples"]) % 10 == 0:
            json.dump(res, open(OUT, "w"))
            log(TASK, len(res["samples"]), {m: round(float(np.mean([x["m"][m].get("kl", x["m"][m].get("turns", [{}])[0].get("score", 0)) for x in res["samples"]])), 3) for m in METHODS})
    json.dump(res, open(OUT, "w"))
    log("EVALDONE")

# ---------------------------------------------------------------- timing
if MODE == "time":
    out = {}
    for N in [2048, 4096, 8192, 16384]:
        ib = torch.randint(100, 50000, (N,)).tolist()
        mp = torch.arange(N)
        ia = torch.randint(100, 50000, (N,)).tolist()
        cA = A.new_cache(N); A.forward(torch.tensor([ia], device=dev), cA, want="hidden")
        new = torch.randint(100, 50000, (128,)).tolist()
        row = {}
        for m in ["native", "B5"]:
            xs = []
            for r in range(8):
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e2 = torch.cuda.Event(enable_timing=True)
                st = {}
                e0.record()
                cache = native(ib) if m == "native" else translate(cA, ib, mp, timer=st)
                e1.record()
                lg = b_fwd(new, cache, N); b_fwd([int(lg[0, -1].argmax())], cache, N + 128)
                e2.record(); torch.cuda.synchronize()
                x = dict(total=e0.elapsed_time(e2), prep=e0.elapsed_time(e1), new_and_decode=e1.elapsed_time(e2),
                         peak_GB=torch.cuda.max_memory_allocated() / 2**30, **{f"stage_{k}": v for k, v in st.items()})
                if r >= 3:
                    xs.append(x)
                del cache
            row[m] = {k: float(np.median([x[k] for x in xs])) for k in xs[0]}
            log("P4", N, m, row[m])
        out[N] = row
        del cA; torch.cuda.empty_cache()
    json.dump(out, open(f"{ROOT}/results/timing_P4.json", "w"), indent=1)
    log("TIMEDONE")
