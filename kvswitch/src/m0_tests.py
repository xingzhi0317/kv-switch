# M0: RoPE unit tests, forward-path equivalence, memory budget, prefill-time table.
import sys, json, torch, gc
from ks import *

res = {}
torch.manual_seed(0)

# ---------- RoPE round-trip + agreement with HF's own apply_rotary_pos_emb ----------
def rope_test(name):
    M = Model(name)
    out = {}
    for n0 in [0, 4000, 31000]:
        cos, sin = M.cossin(n0, n0 + 64)
        x = torch.randn(1, M.Hk, 64, M.D, device="cuda")
        y = rope(x, cos, sin)
        out[f"roundtrip_maxerr@{n0}"] = (derope(y, cos, sin) - x).abs().max().item()
        mod = sys.modules[type(M.layers[0].self_attn).__module__]
        q2, _ = mod.apply_rotary_pos_emb(x, x, cos[None], sin[None])
        out[f"vs_hf_maxerr@{n0}"] = (q2 - y).abs().max().item()
        # bf16 storage round trip (what really happens to a stored cache)
        yb = y.bfloat16().float()
        out[f"bf16_roundtrip_rel@{n0}"] = ((derope(yb, cos, sin) - x).norm() / x.norm()).item()
    # forward equivalence vs HF
    ids = torch.randint(100, 20000, (1, 700), device="cuda")
    ref = M.m(ids).logits.float()
    c = M.new_cache(1024)
    lg = M.forward(ids, c, want="all")
    out["fwd_vs_hf_maxabs"] = (lg - ref).abs().max().item()
    out["fwd_vs_hf_top1"] = (lg.argmax(-1) == ref.argmax(-1)).float().mean().item()
    c2 = M.new_cache(1024)
    M.forward(ids[:, :500], c2)
    lg2 = M.forward(ids[:, 500:], c2, want="all")
    out["chunked_vs_oneshot_maxabs"] = (lg2 - lg[:, 500:]).abs().max().item()
    log(name, out)
    del M; gc.collect(); torch.cuda.empty_cache()
    return out


def rope_test_qwen35():
    from transformers import AutoConfig
    import importlib
    cfg = AutoConfig.from_pretrained(f"{ROOT}/models/Qwen3.5-2B")
    tc = cfg.text_config
    mod = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    rot = mod.Qwen3_5TextRotaryEmbedding(tc).cuda()
    out = dict(head_dim=tc.head_dim, partial_rotary_factor=tc.rope_parameters["partial_rotary_factor"],
               layer_types=tc.layer_types, num_kv_heads=tc.num_key_value_heads,
               linear_k_heads=tc.linear_num_key_heads, linear_v_heads=tc.linear_num_value_heads,
               linear_k_dim=tc.linear_key_head_dim, linear_v_dim=tc.linear_value_head_dim,
               conv_kernel=tc.linear_conv_kernel_dim, n_layers=tc.num_hidden_layers)
    for n0 in [0, 31000]:
        pos = torch.arange(n0, n0 + 64, device="cuda")
        pid = pos[None, None].expand(3, 1, -1)  # text-only: t=h=w positions
        x = torch.randn(1, tc.num_key_value_heads, 64, tc.head_dim, device="cuda")
        cos, sin = rot(x, pid)
        y, _ = mod.apply_rotary_pos_emb(x, x, cos, sin)
        rd = cos.shape[-1]
        yr = torch.cat([derope(y[..., :rd], cos[0], sin[0]), y[..., rd:]], -1)
        out[f"rotary_dim"] = rd
        out[f"roundtrip_maxerr@{n0}"] = (yr - x).abs().max().item()
        out[f"passthrough_untouched@{n0}"] = (y[..., rd:] - x[..., rd:]).abs().max().item()
    log("Qwen3.5-2B", out)
    return out


res["rope"] = {n: rope_test(n) for n in ["Qwen3-1.7B", "Qwen3-4B", "Llama-3.2-3B-Instruct"]}
try:
    res["rope"]["Qwen3.5-2B"] = rope_test_qwen35()
except Exception as e:
    res["rope"]["Qwen3.5-2B"] = dict(error=repr(e)); log("qwen3.5 rope err", e)

# ---------- memory budget + prefill times ----------
PAIRS = {"P1": ("Qwen3-4B", "Qwen3-1.7B"), "P2": ("Qwen3-1.7B", "Qwen3-4B"), "P3": ("Llama-3.2-3B-Instruct", "Qwen3-1.7B")}
LENS = [2048, 4096, 8192, 16384, 32768]
CAP = torch.cuda.get_device_properties(0).total_memory
mem, pre = {}, {}
for P, (a, b) in PAIRS.items():
    torch.cuda.reset_peak_memory_stats()
    A, B = Model(a), Model(b)
    w = torch.cuda.memory_allocated()
    row = dict(weights_GB=w / 2**30, cap_GB=CAP / 2**30)
    for n in LENS:
        two = kv_bytes(A, n) + kv_bytes(B, n)
        row[f"kv2_{n}_GB"] = two / 2**30
        row[f"fits_{n}"] = bool(w + two + 1.5 * 2**30 < CAP)  # 1.5GB headroom for activations
    mem[P] = row
    log(P, row)
    for M in (A, B):
        if M.name in pre:
            continue
        pre[M.name] = {}
        for n in LENS:
            if not row[f"fits_{n}"]:
                continue
            ids = torch.randint(100, 20000, (1, n), device="cuda")
            c = M.new_cache(n)
            for _ in range(2):
                c.n = 0; M.forward(ids, c)
            ts = []
            for _ in range(5):
                c.n = 0
                tm = Timer(); tm.mark("s"); M.forward(ids, c); tm.mark("e"); ts.append(tm.result()["total"])
            pre[M.name][n] = sorted(ts)[2]
            del c; torch.cuda.empty_cache()
        log(M.name, "prefill ms", pre[M.name])
    del A, B; gc.collect(); torch.cuda.empty_cache()
res["memory"] = mem
res["prefill_ms"] = pre
json.dump(res, open(f"{ROOT}/results/m0.json", "w"), indent=1)
log("M0DONE")
