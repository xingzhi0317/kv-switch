# Translation-step latency of the official implementations on one RTX 4090, Qwen3-1.7B -> Qwen3-4B.
# Each method is timed on its own native cache layout; only the translation call is timed.
#   kvbridge     : kvbridge.transfer(DynamicCache, Mapper)            (dense, top-k=12, bf16 matmul)
#   kvtransfer   : kvtransfer Mapper.apply_kv (NVIDIA CFM re-impl)     (same dense weights as kvbridge)
#   cb_dense     : CacheBridge PageMapper(method="nv_transfer")       (same dense weights, CB's runtime)
#   cb_headlocal : CacheBridge PageMapper(method="cachebridge")       (head-local k=8; weights from MAPPER_CB)
# env: LENS, REPS, WARM, MAPPER_CB (dir of layer_*.safetensors; random weights if unset), OUT
import os, json, sys, time, tempfile
import numpy as np, torch
from safetensors.torch import save_file

E = os.environ.get
ROOT = os.path.expanduser("~/kvbase")
MODELS = os.path.expanduser("~/kvswitch/models")
LENS = [int(x) for x in E("LENS", "2048,4096,8192,16384,32768").split(",")]
REPS, WARM = int(E("REPS", 10)), int(E("WARM", 3))
METHODS = E("METHODS", "kvbridge,kvtransfer,cb_dense,cb_headlocal").split(",")
OUT = E("OUT", f"{ROOT}/results/translate_latency.json")
dev = "cuda"
LS, LT, H, D = 28, 36, 8, 128
THETA = 1e6

from kvbridge import Mapper as KBMapper, transfer as kb_transfer
from transformers import DynamicCache, AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

kb = KBMapper.load(f"{ROOT}/mappers/kvbridge_17to4", device=dev)
sel = [list(map(int, e["source_layers"])) for e in kb.layers]


def ev():
    e = torch.cuda.Event(enable_timing=True); e.record(); return e


def timeit(fn, setup=None):
    xs, peaks = [], []
    for r in range(WARM + REPS):
        args = setup() if setup else ()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        a = ev(); out = fn(*args); b = ev(); torch.cuda.synchronize()
        if r >= WARM:
            xs.append(a.elapsed_time(b)); peaks.append((torch.cuda.max_memory_allocated() - base) / 2**30)
        del out, args
    return dict(p50=float(np.median(xs)), p95=float(np.percentile(xs, 95)), min=float(min(xs)), extra_peak_GB=float(np.median(peaks)))


# ---------- dense CacheBridge-format mapper built from the kvbridge weights (identical math) ----------
def dense_cb_dir():
    d = tempfile.mkdtemp(prefix="cbdense_", dir=f"{ROOT}/mappers")
    for i, e in enumerate(kb.layers):
        Wk, Wv = e["W_k"].float(), e["W_v"].float()
        bk = (e["mu_yk"].float() - e["mu_xk"].float() @ Wk).reshape(H, D)
        bv = (e["mu_yv"].float() - e["mu_xv"].float() @ Wv).reshape(H, D)
        save_file({"selected": torch.tensor(sel[i]), "k_weight": Wk.reshape(-1, H, D).contiguous().cpu(),
                   "k_bias": bk.contiguous().cpu(), "v_weight": Wv.reshape(-1, H, D).contiguous().cpu(),
                   "v_bias": bv.contiguous().cpu()}, f"{d}/layer_{i:03d}.safetensors")
    return d


def random_headlocal_dir(k=8):
    d = tempfile.mkdtemp(prefix="cbhl_", dir=f"{ROOT}/mappers")
    g = torch.Generator().manual_seed(0)
    for i in range(LT):
        c = round(i * (LS - 1) / (LT - 1)); s0 = min(max(c - k // 2, 0), LS - k)
        save_file({"selected": torch.arange(s0, s0 + k), "headwise": torch.tensor(True),
                   "k_weight": torch.randn(H, k * D, D, generator=g) * 0.03, "k_bias": torch.zeros(H, D),
                   "v_weight": torch.randn(H, k * D, D, generator=g) * 0.03, "v_bias": torch.zeros(H, D)},
                  f"{d}/layer_{i:03d}.safetensors")
    return d


res = json.load(open(OUT)) if os.path.exists(OUT) else {}
res["cfg"] = dict(pair="Qwen3-1.7B->Qwen3-4B", gpu=torch.cuda.get_device_name(), reps=REPS, warm=WARM,
                  torch=torch.__version__, kvbridge_topk=len(sel[0]))
PM = {}
if "cb_dense" in METHODS or "cb_headlocal" in METHODS:
    from cachebridge.mapper import PageMapper
    if "cb_dense" in METHODS:
        PM["cb_dense"] = PageMapper(dense_cb_dir(), method="nv_transfer", source_rope_theta=THETA, target_rope_theta=THETA, device=dev)
    if "cb_headlocal" in METHODS:
        PM["cb_headlocal"] = PageMapper(E("MAPPER_CB") or random_headlocal_dir(), method="cachebridge",
                                        source_rope_theta=THETA, target_rope_theta=THETA, device=dev)
if "kvtransfer" in METHODS:
    from kvtransfer.mapper import Mapper as KTMapper
    from kvtransfer.hf import ModelSpec
    from kvtransfer.rope import RopeCodec
    spec = lambda n, L: ModelSpec(name=n, n_layers=L, n_kv=H, head_dim=D, rope_theta=THETA, attention_scaling=1.0)
    kt = KTMapper(source=spec("Qwen3-1.7B", LS), target=spec("Qwen3-4B", LT), selected=np.array(sel), lam=0.01)
    for e in kb.layers:
        kt.W_K.append(e["W_k"]); kt.W_V.append(e["W_v"])
        kt.b_K.append(e["mu_yk"] - e["mu_xk"] @ e["W_k"]); kt.b_V.append(e["mu_yv"] - e["mu_xv"] @ e["W_v"])
    rot = Qwen3RotaryEmbedding(AutoConfig.from_pretrained(f"{MODELS}/Qwen3-4B")).to(dev)
    codec = RopeCodec(rot)

for N in LENS:
    row = res.setdefault(str(N), {})
    torch.manual_seed(N)
    src = [(torch.randn(1, H, N, D, device=dev, dtype=torch.bfloat16) * 4,
            torch.randn(1, H, N, D, device=dev, dtype=torch.bfloat16)) for _ in range(LS)]
    for m in METHODS:
        try:
            if m == "kvbridge":
                def setup():
                    c = DynamicCache()
                    for l, (k, v) in enumerate(src):
                        c.update(k, v, l)
                    return (c,)
                r = timeit(lambda c: kb_transfer(c, kb), setup)
            elif m == "kvtransfer":
                pos = torch.arange(N, device=dev)
                r = timeit(lambda: kt.apply_kv(src, pos, codec, codec, out_dtype=torch.bfloat16))
            else:
                pm = PM[m]
                P = 2048
                pages = [torch.stack([torch.stack([src[l][0][0, :, p:p + P].transpose(0, 1) for l in range(LS)]),
                                      torch.stack([src[l][1][0, :, p:p + P].transpose(0, 1) for l in range(LS)])]).contiguous()
                         for p in range(0, N, P)]  # SGLang canonical page layout [2, L, T, H, D]
                outbuf = torch.empty(2, LT, P, H, D, device=dev, dtype=torch.bfloat16)

                def run():
                    for i, pg in enumerate(pages):
                        pm.map_page(pg, position_start=i * P, output_device=outbuf.device, output=outbuf)
                    return outbuf
                r = timeit(run)
                del pages, outbuf
            row[m] = r
            print(N, m, {k: round(v, 3) for k, v in r.items()}, flush=True)
        except torch.OutOfMemoryError:
            row[m] = "OOM"; print(N, m, "OOM", flush=True)
        torch.cuda.empty_cache()
    del src; torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
print("BENCHDONE")
