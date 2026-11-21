# K/V reconstruction R^2 on QuALITY articles, to compare with CacheBridge Table 13 (Qwen3 8B -> 14B, QuALITY):
#   CFM ridge mapper  R2_K 0.617, R2_V 0.503;  CacheBridge (calibrated)  R2_K 0.556, R2_V -2.47.
# CacheBridge calibrates V only, so its R2_K is the R2_K of its INITIAL mapper.
# History = quality_cb history (instruction + complete article) of QuALITY documents NOT used for calibration
# (tasks.quality_docs(), i.e. docs[24:]), first NDOC documents.  Source reads it, the mapper builds the target KV,
# compared with the target's own re-prefill KV at every history position.
#   R2 (kvbridge.fit.r2_score definition): per target layer, 1 - sum of squared errors / total sum of squares about the
#   mean, all heads and dims of the layer together; keys in content space (RoPE removed); reported = mean over layers.
#   R2_coord: per-coordinate R^2 averaged over coordinates (alternative definition, for reference).
# mapper arguments: kb:<dir> (kvbridge format) | cb:<dir> (CacheBridge PageMapper format)
# usage: CUDA_VISIBLE_DEVICES=a,b,c,d python diag_r2.py kb:~/kvbase/mappers/kvbridge_8to14 cb:... ...
#        (source on GPU 0, target over GPUs 1-2, mapping and statistics on GPU 3)
import os, sys, json, gc, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from kvbridge.rope import strip_rope
from kvbridge import Mapper as KBMapper, transfer as kb_transfer
from cachebridge.mapper import PageMapper
sys.path.insert(0, os.path.expanduser("~/kvcodesign"))
from tasks import quality_docs, quality_cb_text, chat_split

E = os.environ.get
NDOC = int(E("NDOC", 20))
M = os.path.expanduser("~/kvswitch/models")
SRC, TGT, THETA = "Qwen3-8B-ms", "Qwen3-14B-ms", 1e6
d0, d3 = "cuda:0", "cuda:3"
torch.set_grad_enabled(False)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
tok = AutoTokenizer.from_pretrained(f"{M}/{SRC}")
enc = lambda x: tok(x, add_special_tokens=False).input_ids
S = AutoModelForCausalLM.from_pretrained(f"{M}/{SRC}", torch_dtype=torch.bfloat16).to(d0).eval()
T = AutoModelForCausalLM.from_pretrained(f"{M}/{TGT}", torch_dtype=torch.bfloat16, device_map="auto",
                                         max_memory={0: "0GiB", 1: "21GiB", 2: "21GiB", 3: "0GiB"}).eval()
LS, LT = S.config.num_hidden_layers, T.config.num_hidden_layers
hists = [chat_split(enc, *quality_cb_text(d[0]["article"], d[0]))[0] for d in quality_docs()[:NDOC]]
log("documents", len(hists), "history tokens", sum(map(len, hists)))
mappers = []
for a in sys.argv[1:]:
    kind, path = a.split(":", 1); path = os.path.expanduser(path)
    mp = KBMapper.load(path, device=d3) if kind == "kb" else PageMapper(path, method="cachebridge", source_rope_theta=THETA,
                                                                         target_rope_theta=THETA, device=d3)
    mappers.append((os.path.basename(path), kind, mp))
# per mapper, layer, component: sum of squared errors, sum y, sum y^2 (pooled), per-coordinate sums
acc = {name: torch.zeros(2, LT, 3, 1024, device=d3, dtype=torch.float64) for name, _, _ in mappers}   # [comp, layer, (sse, sy, syy), coord]
ntok = 0
for hist in hists:
    n = len(hist); pos = torch.arange(n, device=d3)
    nat = T.model(torch.tensor([hist], device=T.model.embed_tokens.weight.device), use_cache=True).past_key_values
    natk = [strip_rope(nat.layers[l].keys[0].to(d3).float(), pos, THETA).transpose(0, 1).reshape(n, -1) for l in range(LT)]
    natv = [nat.layers[l].values[0].to(d3).float().transpose(0, 1).reshape(n, -1) for l in range(LT)]
    del nat
    sc = S.model(torch.tensor([hist], device=d0), use_cache=True).past_key_values
    for name, kind, mp in mappers:
        if kind == "kb":
            c = DynamicCache()
            for l in range(LS):
                c.update(sc.layers[l].keys.to(d3), sc.layers[l].values.to(d3), l)
            out = kb_transfer(c, mp)
            mk = [out.layers[l].keys[0].float() for l in range(LT)]; mv = [out.layers[l].values[0].float() for l in range(LT)]
            del c, out
        else:
            sk = torch.stack([sc.layers[l].keys[0].transpose(0, 1) for l in range(LS)]).to(d3)
            sv = torch.stack([sc.layers[l].values[0].transpose(0, 1) for l in range(LS)]).to(d3)
            o = torch.cat([mp.map_page(torch.stack([sk[:, p:p + 2048], sv[:, p:p + 2048]]), position_start=p, output_device=d3)
                           for p in range(0, n, 2048)], 2)                                           # [2, LT, n, H, D]
            mk = [o[0, l].transpose(0, 1).float() for l in range(LT)]; mv = [o[1, l].transpose(0, 1).float() for l in range(LT)]
            del sk, sv, o
        A = acc[name]
        for l in range(LT):
            for ci, (y, yh) in enumerate(((natk[l], strip_rope(mk[l], pos, THETA).transpose(0, 1).reshape(n, -1)),
                                          (natv[l], mv[l].transpose(0, 1).reshape(n, -1)))):
                A[ci, l, 0] += (y - yh).double().square().sum(0); A[ci, l, 1] += y.double().sum(0); A[ci, l, 2] += y.double().square().sum(0)
        del mk, mv
    ntok += n
    del sc, natk, natv; torch.cuda.empty_cache()
res = {}
for name, _, _ in mappers:
    A = acc[name]; sse, sy, syy = A[:, :, 0], A[:, :, 1], A[:, :, 2]
    sst = syy - sy.square() / ntok                                                                       # [2, LT, 1024]
    pooled = 1 - sse.sum(-1) / sst.sum(-1)                                                                # [2, LT]
    coord = (1 - sse / sst.clamp_min(1e-12)).mean(-1)
    res[name] = dict(R2_K=float(pooled[0].mean()), R2_V=float(pooled[1].mean()), R2coord_K=float(coord[0].mean()),
                     R2coord_V=float(coord[1].mean()), per_layer_K=pooled[0].tolist(), per_layer_V=pooled[1].tolist())
    log(f"{name:40s} R2_K {res[name]['R2_K']:.3f}  R2_V {res[name]['R2_V']:.3f}   (per-coordinate mean: K {res[name]['R2coord_K']:.3f}  V {res[name]['R2coord_V']:.3f})")
json.dump(dict(docs=NDOC, tokens=ntok, paper=dict(CFM=dict(R2_K=0.617, R2_V=0.503), CacheBridge=dict(R2_K=0.556, R2_V=-2.47)),
               mappers=res), open(os.path.expanduser("~/kvcodesign/results/8to14/diag_r2.json"), "w"), indent=1)
log("R2DONE")
