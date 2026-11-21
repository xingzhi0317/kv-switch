# Does the CacheBridge value calibration do what it should?  Held-out check, using the OFFICIAL PageMapper (the exact
# path eval_quality.py uses).  For each mapper dir: 8 QuALITY documents NOT used for calibration (docs[24:32] of the
# same shuffle) x 2 questions; the source reads the article, the mapper builds the target history cache, the target
# reads the question.  Reported per mapper:
#   readout_err[l] = || O_mapped - O_native ||^2 / || O_native ||^2 of layer l's attention output at the question
#                    positions, both computed from the SAME query states (the mapped run's), i.e. the quantity the
#                    calibration minimises (native = re-prefill history K,V);
#   kl_q           = KL(native || mapped) of the next-token distribution at the question positions (end-to-end).
# usage: DIR=17to4 python diag_cb.py mapper_dir [mapper_dir ...]       (CORRUPT runs are diagnosed the same way)
import os, sys, json, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from cachebridge.mapper import PageMapper
from cachebridge.offline.calibration import algebra
from kvbridge.rope import build_cos_sin

E = os.environ.get
DIR = E("DIR", "17to4")
SRC_M, TGT_M = {"17to4": ("Qwen3-1.7B", "Qwen3-4B"), "4to17": ("Qwen3-4B", "Qwen3-1.7B"),
                "same17": ("Qwen3-1.7B", "Qwen3-1.7B")}[DIR]
M = os.path.expanduser("~/kvswitch/models")
dev, D, THETA = "cuda", 128, 1e6
tok = AutoTokenizer.from_pretrained(f"{M}/Qwen3-1.7B")
S = AutoModelForCausalLM.from_pretrained(f"{M}/{SRC_M}", torch_dtype=torch.bfloat16).to(dev).eval()
T = S if SRC_M == TGT_M else AutoModelForCausalLM.from_pretrained(f"{M}/{TGT_M}", torch_dtype=torch.bfloat16).to(dev).eval()
torch.set_grad_enabled(False)
LS, LT = S.config.num_hidden_layers, T.config.num_hidden_layers

from datasets import load_dataset
arts = {}
for r in load_dataset("emozilla/quality", split="validation"):
    arts.setdefault(r["article"], []).append(r)
docs = [v for v in arts.values() if len(v) >= 4]
random.Random(0).shuffle(docs)
pairs = []
for d in docs[24:32]:   # held out: fit_cb2/4/5 calibrate on docs[:24]
    hist = tok(f"<|im_start|>user\nRead the following article.\n\n{d[0]['article']}", add_special_tokens=False).input_ids[:6000]
    for r in d[:2]:
        opts = "\n".join(f"{c}. {o}" for c, o in zip("ABCD", r["options"]))
        pairs.append((hist, tok(f"\n\nQuestion: {r['question']}\n{opts}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:",
                                add_special_tokens=False).input_ids))

attn_in = {}
for l in range(LT):
    T.model.layers[l].self_attn.register_forward_pre_hook(
        (lambda l: lambda mod, args, kwargs: attn_in.__setitem__(l, kwargs.get("hidden_states", args[0] if args else None)))(l),
        with_kwargs=True)


def q_states(l, hidden, n0, nq):
    a = T.model.layers[l].self_attn
    hsh = (1, nq, -1, D)
    qq = a.q_norm(a.q_proj(hidden).view(hsh)).transpose(1, 2)
    kk = a.k_norm(a.k_proj(hidden).view(hsh)).transpose(1, 2)
    vv = a.v_proj(hidden).view(hsh).transpose(1, 2)
    cos, sin = build_cos_sin(torch.arange(n0, n0 + nq, device=dev), D, THETA, dev, torch.float32)
    rot = lambda x: (x.float() * cos + torch.cat([-x[..., D // 2:], x[..., :D // 2]], -1).float() * sin)
    return rot(qq), rot(kk), vv.float()


prep = []
for hist, qids in pairs:
    n0, nq = len(hist), len(qids)
    nat = T.model(torch.tensor([hist], device=dev), use_cache=True).past_key_values
    natkv = [(nat.layers[l].keys.clone(), nat.layers[l].values.clone()) for l in range(LT)]
    ref = T(torch.tensor([qids], device=dev), past_key_values=nat, position_ids=torch.arange(n0, n0 + nq, device=dev)[None]).logits[0].float().log_softmax(-1)
    sc = S.model(torch.tensor([hist], device=dev), use_cache=True).past_key_values
    src = torch.stack([torch.stack([sc.layers[l].keys[0].transpose(0, 1) for l in range(LS)]),
                       torch.stack([sc.layers[l].values[0].transpose(0, 1) for l in range(LS)])]).cpu()   # [2, LS, n, H, D]
    prep.append(dict(n0=n0, qids=qids, natkv=[(k.cpu(), v.cpu()) for k, v in natkv], ref=ref.cpu(), src=src))
    del nat, natkv, sc
results = {}
for mdir in sys.argv[1:]:
    mp = PageMapper(mdir, method="cachebridge", source_rope_theta=THETA, target_rope_theta=THETA, device=dev)
    err = torch.zeros(LT); kl = 0.0
    for p in prep:
        n0, nq = p["n0"], len(p["qids"])
        out = torch.cat([mp.map_page(p["src"][:, :, s:s + 2048].to(dev), position_start=s, output_device=dev)
                         for s in range(0, n0, 2048)], 2)          # [2, LT, n, H, D]
        c = DynamicCache()
        for l in range(LT):
            c.update(out[0, l].transpose(0, 1)[None].to(torch.bfloat16), out[1, l].transpose(0, 1)[None].to(torch.bfloat16), l)
        lp = T(torch.tensor([p["qids"]], device=dev), past_key_values=c, position_ids=torch.arange(n0, n0 + nq, device=dev)[None]).logits[0].float().log_softmax(-1)
        ref = p["ref"].to(dev)
        kl += float((ref.exp() * (ref - lp)).sum(-1).mean())
        for l in range(LT):
            qq, kq, vq = q_states(l, attn_in[l], n0, nq)
            km, vm = c.layers[l].keys[0][:, :n0].float(), c.layers[l].values[0][:, :n0].float()
            kn, vn = p["natkv"][l][0][0].to(dev).float(), p["natkv"][l][1][0].to(dev).float()
            pos = list(range(nq))
            Om = algebra.pool(algebra.probabilities(qq, torch.cat([km, kq[0]], 1)[None], n0, pos), torch.cat([vm, vq[0]], 1))
            On = algebra.pool(algebra.probabilities(qq, torch.cat([kn, kq[0]], 1)[None], n0, pos), torch.cat([vn, vq[0]], 1))
            err[l] += float((Om - On).square().sum() / On.square().sum())
        attn_in.clear(); del c, out
    err /= len(prep)
    results[mdir] = dict(kl_q=kl / len(prep), readout_err=err.tolist(), readout_err_mean=float(err.mean()))
    print(os.path.basename(mdir), "KL at question positions", round(kl / len(prep), 4), "mean readout err", round(float(err.mean()), 4),
          "per layer", [round(float(x), 3) for x in err[::4]], flush=True)
json.dump(results, open(os.path.expanduser(f"~/kvcodesign/results/diag_cb_{DIR}.json"), "w"), indent=1)
print("DIAGDONE")
