# Quality evaluation of switch methods under the common switch protocol (PLAN §4.1).
# env: PAIR(P1|P2) TASK(coqa1|coqa5|coqa10|lb4k|lb8k|lb16k|niah4k..niah32k|gsm|uc) N METHODS(csv) KSEL(3) OURS(path|'')
import os, sys, json, math, torch
from ks import *
from translate import Translator, selective_recompute
import tasks as TK

E = os.environ.get
PAIRS = {"P1": ("Qwen3-4B", "Qwen3-1.7B"), "P2": ("Qwen3-1.7B", "Qwen3-4B"), "P3": ("Llama-3.2-3B-Instruct", "Qwen3-1.7B")}
XT = E("PAIR") == "P3"
PAIR, TASK, NS = E("PAIR"), E("TASK"), int(E("N", 100))
METHODS = E("METHODS", "native,none,B2,B1,B3").split(",")
KSEL = int(E("KSEL", 3))
OUT = E("OUT", f"{ROOT}/results/q_{PAIR}_{TASK}{E('SUFFIX', '')}.json")
A, B = Model(PAIRS[PAIR][0]), Model(PAIRS[PAIR][1])
maps = torch.load(f"{ROOT}/maps/{PAIR}.pt")
TR = {}
if XT:
    from xalign import XTranslator, conv_aligned
for m in dict.fromkeys(x[:-3] if x.endswith("_c3") else x for x in METHODS):
    if XT and m.startswith("B4"):  # KV-Lingo-style (closed-form init, char-aligned); B4k1 = 1 source layer
        TR[m] = XTranslator(A, B, maps, k=1 if m == "B4k1" else 3, vkind="WV")
    elif XT and m.startswith("ours"):
        lr = torch.load(f"{ROOT}/maps/{PAIR}_{m}.pt")
        TR[m] = XTranslator(A, B, maps, k=lr["k"], vkind="WV", lowrank=lr)
    elif m in ("B1", "B3"):
        TR[m] = Translator(A, B, maps, k=KSEL, vkind="WVa")
    elif m == "B2":
        TR[m] = Translator(A, B, maps, k=KSEL, vkind="WV")
    elif m.startswith("B1k"):  # B1 with a different number of source layers
        TR[m] = Translator(A, B, maps, k=int(m[3:]), vkind="WVa")
    elif m.endswith("_c3"):  # C3: translated cache for the 1st reply, then the background native re-prefill
        pass
    elif m.startswith("ours"):  # KL-trained low-rank residual on top of B1 (C4); file maps/{PAIR}_{m}.pt
        lr = torch.load(f"{ROOT}/maps/{PAIR}_{m}.pt")
        TR[m] = Translator(A, B, maps, k=lr["k"], vkind="WVa", lowrank=lr)
    elif m.startswith("noise"):  # deliberately degraded translator for guard calibration (M3)
        TR[m] = Translator(A, B, maps, k=1, vkind="WV")
        sig = float(m[5:])
        g = torch.Generator(device="cuda").manual_seed(0)
        for W in (TR[m].WK, TR[m].WV):
            W += (torch.randn(W.shape, generator=g, device="cuda") * W.float().std() * sig).to(W.dtype)

t = TASK
if t.startswith("coqa"):
    S = TK.coqa(NS, int(t[4:]))
elif t.startswith("lb"):
    S = TK.longbench(A.tok, NS // 4, int(t[2:-1]) * 1024)
elif t.startswith("niah"):
    S = TK.niah(A.tok, NS, int(t[4:-1]) * 1024)
elif t == "gsm":
    S = TK.gsm8k(NS)
elif t == "uc":
    S = TK.ultrachat(NS)
STOP = B.stop_ids()


def user_span(M, text):  # token range of the user *content* inside M.turn('user', text)
    pre = len(M.tok("<|im_start|>user\n" if M.family == "qwen3" else "<|start_header_id|>user<|end_header_id|>\n\n",
                    add_special_tokens=False).input_ids)
    return pre, len(M.turn("user", text)) - 1


def guard_stats(lg, new, us):
    """lg: [1,T,V] logits over the new input; new: ids; us: (a,b) user-content span inside new."""
    lp = lg[0].log_softmax(-1)
    a, b = us
    tgt = torch.tensor(new[a + 1:b], device=lg.device)
    nll = -lp[a:b - 1].gather(-1, tgt[:, None]).squeeze(-1)
    ent = -(lp[-8:].exp() * lp[-8:]).sum(-1)
    return dict(nll=nll.mean().item(), nll_last32=nll[-32:].mean().item(), ent8=ent.mean().item(), ntok=int(b - a - 1))


def prep_cache(m, ids_t, N, cA, extra):
    m = m[:-3] if m.endswith("_c3") else m
    cB = B.new_cache(N + extra)
    if m == "native":
        B.forward(ids_t, cB, want="hidden")
    elif m == "none":
        pass
    elif XT:
        TR[m].translate(cA, cB, ids_t, MP)
    else:
        TR[m].translate(cA, cB, ids_t, 0, N)
        if m == "B3":
            selective_recompute(B, cB, ids_t, N, 0.3)
    return cB


res = json.load(open(OUT)) if os.path.exists(OUT) and E("RESUME", "1") == "1" else dict(cfg=dict(PAIR=PAIR, TASK=TASK, METHODS=METHODS, KSEL=KSEL), samples=[])
done = {s["id"] for s in res["samples"]}
for si, s in enumerate(S):
    if s["id"] in done:
        continue
    if XT:
        ia, ids, MP = conv_aligned(A, B, s["hist"])
        cA = A.new_cache(len(ia) + 2048)
        A.forward(torch.tensor([ia], device="cuda"), cA, want="hidden")
    else:
        ids, _ = A.conv_ids(s["hist"])
        cA = A.new_cache(len(ids) + 2048)
        A.forward(torch.tensor([ids], device="cuda"), cA, want="hidden")
    N = len(ids)
    ids_t = torch.tensor([ids], device="cuda")
    rec = dict(id=s["id"], N=N, m={})
    lpA = None
    if s["metric"] != "kl" and not XT:  # guard stat 4: A (cache still resident at switch time) reads the same new input
        newA = A.turn("user", s["turns"][0][0]) + A.gen_prompt()
        if len(newA) <= 2048:
            lpA = A.forward(torch.tensor([newA], device="cuda"), cA, n0=N, want="all")[0].log_softmax(-1)
            ua, ub = user_span(A, s["turns"][0][0])
            lpA = lpA[ua:ub - 1]
    if s["metric"] == "kl":
        new = B.turn("user", s["user"]) + B.gen_prompt()
        rep = B.tok(s["reply"], add_special_tokens=False).input_ids[:256]
        seq = new + rep
        ref = None
        for m in (["native"] if "native" not in METHODS else []) + METHODS:
            base = len(B.bos()) if m == "none" else N
            cB = prep_cache(m, ids_t, N, cA, len(seq) + 8)
            if m == "none":
                B.forward(torch.tensor([B.bos()], device="cuda"), cB, want="hidden") if B.bos() else None
            lg = B.forward(torch.tensor([seq], device="cuda"), cB, want="all")
            lp = lg[0, len(new) - 1:-1].log_softmax(-1)
            r = guard_stats(lg[:, :len(new)], new, user_span(B, s["user"]))
            tg = torch.tensor(rep, device="cuda")
            r["reply_nll"] = -lp.gather(-1, tg[:, None]).mean().item()
            if m == "native":
                ref = lp
            else:
                r["kl"] = (ref.exp() * (ref - lp)).sum(-1).mean().item()
                r["top1"] = (ref.argmax(-1) == lp.argmax(-1)).float().mean().item()
            if m in METHODS:
                rec["m"][m] = r
            del cB
    else:
        for m in METHODS:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            extra = sum(len(B.turn("user", q)) + 4 * mx + 128 for q, _, mx in s["turns"]) + 256
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            cB = prep_cache(m, ids_t, N, cA, extra)
            if m == "none" and B.bos():
                B.forward(torch.tensor([B.bos()], device="cuda"), cB, want="hidden")
            r = dict(turns=[])
            for ti, (q, gold, mx) in enumerate(s["turns"]):
                new = B.turn("user", q) + B.gen_prompt()
                n_before = cB.n
                lg = B.forward(torch.tensor([new], device="cuda"), cB, want="all")
                if ti == 0:
                    e1.record(); torch.cuda.synchronize()
                    r["t_switch_ms"] = e0.elapsed_time(e1)
                    r["peak_GB"] = torch.cuda.max_memory_allocated() / 2**30
                    r.update(guard_stats(lg, new, user_span(B, q)))
                    if lpA is not None:
                        ua, ub = user_span(B, q)
                        lpB = lg[0, ua:ub - 1].log_softmax(-1)
                        V = min(lpA.shape[-1], lpB.shape[-1])
                        r["klAB"] = (lpA[:, :V].exp() * (lpA[:, :V] - lpB[:, :V])).sum(-1).mean().item()
                        r["top1AB"] = (lpA.argmax(-1) == lpB.argmax(-1)).float().mean().item()
                out = B.generate(lg, cB, mx, STOP)
                pred = B.tok.decode(out, skip_special_tokens=True).strip()
                r["turns"].append(dict(pred=pred, score=TK.score(s["metric"], pred, gold)))
                cB.truncate(n_before + len(B.turn("user", q)))
                cB.n = n_before  # rebuild: user turn + assistant answer in history format
                hist_tail = B.turn("user", q) + B.turn("assistant", pred)
                B.forward(torch.tensor([hist_tail], device="cuda"), cB, want="hidden")
                r.setdefault("_tail", []).extend(hist_tail)
                if m.endswith("_c3") and ti == 0:  # background upgrade finished before the next user turn
                    full = ids + r["_tail"]
                    cB.n = 0
                    B.forward(torch.tensor([full], device="cuda"), cB, want="hidden")
            r.pop("_tail", None)
            if m == "native":  # guard stat 2: B on the same input with no history
                new0 = B.bos() + B.turn("user", s["turns"][0][0]) + B.gen_prompt()
                c0 = B.new_cache(len(new0) + 8)
                lg0 = B.forward(torch.tensor([new0], device="cuda"), c0, want="all")
                a, b = user_span(B, s["turns"][0][0])
                rec["nohist"] = guard_stats(lg0[:, len(B.bos()):], new0[len(B.bos()):], (a, b))
                del c0
            rec["m"][m] = r
            del cB
    del cA
    res["samples"].append(rec)
    if len(res["samples"]) % 10 == 0 or si == len(S) - 1:
        json.dump(res, open(OUT, "w"))
        msg = {m: round(sum((x["m"][m]["turns"][0]["score"] if "turns" in x["m"][m] else -x["m"][m].get("kl", 0))
                            for x in res["samples"]) / len(res["samples"]), 3) for m in METHODS}
        log(f"{TASK} {PAIR} n={len(res['samples'])} N~{N} {msg}")
json.dump(res, open(OUT, "w"))
log("EVALDONE")
