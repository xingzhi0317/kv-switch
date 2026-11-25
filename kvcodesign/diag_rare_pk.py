# Why is RaReCache low on the 8K passkey?  Selection rates and forced-position variants.
# Same samples/map/prefill as eval_quality.  env: N
import os, sys, json, math, torch
HM = os.path.expanduser("~")
sys.path.insert(0, HM + "/kvbase/kvbridge"); sys.path.insert(0, HM + "/kvcodesign")
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from kvbridge import Mapper, transfer
import rare, tasks, inspect
N = int(os.environ.get("N", 40)); dev = "cuda"; M = HM + "/kvswitch/models"
tok = AutoTokenizer.from_pretrained(f"{M}/Qwen3-4B")
enc = lambda x: tok(x, add_special_tokens=False).input_ids
S = AutoModelForCausalLM.from_pretrained(f"{M}/Qwen3-1.7B", torch_dtype=torch.bfloat16).to(dev).eval()
T = AutoModelForCausalLM.from_pretrained(f"{M}/Qwen3-4B", torch_dtype=torch.bfloat16).to(dev).eval()
kb = Mapper.load(HM + "/kvbase/mappers/kvbridge_17to4", device=dev)
vr = {k: v.to(dev) for k, v in torch.load(HM + "/kvcodesign/mappers/rare_vr_17to4.pt").items()}
STOP = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
samples = tasks.passkey_samples(enc, N, 8000, [d[0]["article"] for d in tasks.quality_docs()])
old = json.load(open(HM + "/kvcodesign/results/quality_passkey_8k_rare.json"))["samples"]

@torch.no_grad()
def prefill(m, ids):
    c = DynamicCache()
    for a in range(0, len(ids), 2048):
        m(torch.tensor([ids[a:a + 2048]], device=dev), past_key_values=c, use_cache=True, logits_to_keep=1)
    return c

@torch.no_grad()
def answer(c, s):
    nxt = int(T(torch.tensor([s["q"]], device=dev), past_key_values=c, use_cache=True, logits_to_keep=1).logits[0, -1].argmax()); out = []
    for _ in range(s["mx"]):
        if nxt in STOP: break
        out.append(nxt); nxt = int(T(torch.tensor([[nxt]], device=dev), past_key_values=c, use_cache=True).logits[0, -1].argmax())
    return tok.decode(out, skip_special_tokens=True).strip()

def find(h, sub):
    return next(i for i in range(len(h) - len(sub), -1, -1) if h[i:i + len(sub)] == sub)
res, frac, same = {}, {}, 0
for i, s in enumerate(samples):
    h = s["h"]; n = len(h); assert n == old[i]["n_hist"]
    code = find(h, enc(s["gold"])); cset = list(range(code, code + 6))
    sent = find(h, enc(" special access code is")); sset = list(range(sent - 1, code + 7))
    qq = find(h, enc("Question")); qset = list(range(qq, n))
    sc = prefill(S, h); base = transfer(sc, kb); sco = rare.scores(base, vr)
    rank = torch.empty(n, device=dev); rank[sco.argsort(descending=True)] = torch.arange(n, device=dev).float() / n
    for k, p in (("code", cset), ("sentence", sset), ("question", qset)):
        frac.setdefault(k, []).append([rank[p].mean().item()] + [(rank[p] < r).float().mean().item() for r in (.1, .3, .5)])
    sel30, sel50 = sco.topk(math.ceil(.3 * n)).indices, sco.topk(math.ceil(.5 * n)).indices
    g = torch.Generator(device=dev).manual_seed(i)
    U = lambda a, p: torch.unique(torch.cat([a, torch.tensor(p, device=dev)]))
    E0 = torch.empty(0, dtype=torch.long, device=dev)
    V = {"translate": E0, "rare30": sel30, "rare50": sel50, "random50": torch.randperm(n, generator=g, device=dev)[:n // 2],
         "rare30+question": U(sel30, qset), "rare30+sentence": U(sel30, sset), "rare30+both": U(sel30, sset + qset),
         "sentence only": U(E0, sset), "question only": U(E0, qset), "sentence+question only": U(E0, sset + qset),
         "rare50+sentence": U(sel50, sset), "rare50+both": U(sel50, sset + qset),
         "all100": torch.arange(n, device=dev)}
    for k, sel in V.items():
        c = transfer(sc, kb)
        if len(sel): c = rare.recompute(T, c, h, sel, chunk=1024)
        p = answer(c, s); ok = int(p == s["gold"]); res.setdefault(k, []).append(ok); del c
        if k == "rare30": same += int(p == old[i]["m"]["rare30"]["pred"])
    del sc, base; torch.cuda.empty_cache()
    if i % 5 == 4: print(i + 1, {k: round(sum(v) / len(v), 2) for k, v in res.items()}, flush=True)
print("same prediction as stored rare30:", same, "/", N)
print("selection [mean rank, in top10, top30, top50]:", {k: [round(sum(x) / len(x), 2) for x in zip(*v)] for k, v in frac.items()})
json.dump(dict(res=res, frac=frac), open(HM + "/kvcodesign/results/diag_rare_pk.json", "w")); print("DONE")
