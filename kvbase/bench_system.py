# System-level measurements on one RTX 4090, Qwen3-1.7B -> Qwen3-4B (target served with HF transformers, sdpa).
#  (1) E2E TTFT parts: transfer (measured GPU->GPU here; NVLink/RDMA modeled by bandwidth), translation
#      (CacheBridge official vs ours), target prefill of the new turn + first decode step, and re-prefill.
#  (2) Concurrency: R simultaneous handoffs -> aggregate translation throughput and per-request latency.
#  (3) Serving interference: target decodes a batch while handoff translations arrive at rate r/s.
# env: PART (e2e|conc|serve|all)
import os, json, glob, time, threading, numpy as np, torch
from transformers import AutoModelForCausalLM, DynamicCache
from cachebridge.mapper import PageMapper
from fastmap import FastHeadLocal

E = os.environ.get
ROOT = os.path.expanduser("~/kvbase")
MODELS = os.path.expanduser("~/kvswitch/models")
PART = E("PART", "all")
CB = E("MAPPER_CB") or sorted(glob.glob(f"{ROOT}/mappers/cbhl_*"))[-1]
OUT = f"{ROOT}/results/system.json"
dev = "cuda:0"
LS, LT, H, D = 28, 36, 8, 128
BYTES_PER_TOK_SRC = LS * 2 * H * D * 2
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
fh = FastHeadLocal(CB, dev=dev)
pm = PageMapper(CB, method="cachebridge", source_rope_theta=1e6, target_rope_theta=1e6, device=dev)
P = 2048


def ev(s=None):
    e = torch.cuda.Event(enable_timing=True); e.record(s); return e


def med(xs):
    return float(np.median(xs))


def cb_translate(sk, sv, out, pos0=0):
    N = sk.shape[1]
    for p in range(0, N, P):
        pg = torch.stack([sk[:, p:p + P], sv[:, p:p + P]])
        pm.map_page(pg, position_start=pos0 + p, output_device=out.device, output=out[:, :, p:p + P])
    return out


def src(N, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    return ((torch.randn(LS, N, H, D, device=dev, generator=g) * 4).to(torch.bfloat16),
            torch.randn(LS, N, H, D, device=dev, generator=g).to(torch.bfloat16))


# ------------------------------------------------------------------------------- (1) E2E
if PART in ("e2e", "all"):
    B = AutoModelForCausalLM.from_pretrained(f"{MODELS}/Qwen3-4B", torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    e2e = res.setdefault("e2e", {})
    # measured GPU->GPU copy bandwidth on this host (4090s: no NVLink; P2P via PCIe/host)
    x = torch.empty(256 * 2**20, dtype=torch.uint8, device=dev)
    y = torch.empty_like(x, device="cuda:1")
    for _ in range(3):
        y.copy_(x); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        y.copy_(x)
    torch.cuda.synchronize()
    bw_pcie = 10 * x.numel() / (time.time() - t0)
    del x, y
    BW = {"same_gpu": None, "nvlink_h800": 400e9, "rdma_4x400Gb": 200e9, "pcie_measured": bw_pcie}
    e2e["bandwidth_Bps"] = BW
    for N in [int(x) for x in E("E2E_LENS", "2048,4096,8192,16384,32768").split(",")]:
        row = {}
        sk, sv = src(N)
        out = torch.empty(2, LT, N, H, D, device=dev, dtype=torch.bfloat16)
        for name, fn in (("cachebridge", lambda: cb_translate(sk, sv, out)), ("ours", lambda: fh(sk, sv, out=out))):
            ts = []
            for r in range(8):
                a = ev(); fn(); b = ev(); torch.cuda.synchronize()
                if r >= 2:
                    ts.append(a.elapsed_time(b))
            row[f"translate_{name}_ms"] = med(ts)
        # target: new turn (128 tok) + first decode on top of an N-token cache; and full re-prefill
        newq = torch.randint(100, 50000, (1, 128), device=dev)
        hist = torch.randint(100, 50000, (1, N), device=dev)
        ts_new, ts_pre = [], []
        for r in range(6):
            cache = DynamicCache()
            for l in range(LT):
                cache.update(out[0, l].transpose(0, 1)[None].contiguous(), out[1, l].transpose(0, 1)[None].contiguous(), l)
            torch.cuda.synchronize()
            a = ev()
            with torch.no_grad():
                lg = B(newq, past_key_values=cache, use_cache=True).logits[:, -1:]
                B(lg.argmax(-1), past_key_values=cache, use_cache=True)
            b = ev(); torch.cuda.synchronize()
            if r >= 2:
                ts_new.append(a.elapsed_time(b))
            del cache
        del sk, sv, out; torch.cuda.empty_cache()
        for r in range(6):
            c2 = DynamicCache()
            a2 = ev()
            with torch.no_grad():
                B(torch.cat([hist, newq], 1), past_key_values=c2, use_cache=True, logits_to_keep=1)
                B(lg.argmax(-1), past_key_values=c2, use_cache=True)
            b2 = ev(); torch.cuda.synchronize()
            del c2
            if r >= 2:
                ts_pre.append(a2.elapsed_time(b2))
            torch.cuda.empty_cache()
        row["new_turn_plus_decode1_ms"] = med(ts_new)
        row["reprefill_ttft_ms"] = med(ts_pre)
        nbytes = N * BYTES_PER_TOK_SRC
        for sc, bw in BW.items():
            tr = 0.0 if bw is None else nbytes / bw * 1e3
            for name in ("cachebridge", "ours"):
                tl_ = row[f"translate_{name}_ms"]
                serial = tr + tl_ + row["new_turn_plus_decode1_ms"]
                # 2048-token block pipeline (CacheBridge-style): transfer and translation overlap
                nb = max(1, N // P)
                piped = max(tr, tl_) + min(tr, tl_) / nb + row["new_turn_plus_decode1_ms"]
                row[f"ttft_{sc}_{name}_serial_ms"] = serial
                row[f"ttft_{sc}_{name}_pipelined_ms"] = piped
                row[f"translate_share_{sc}_{name}"] = tl_ / serial
            row[f"transfer_{sc}_ms"] = tr
        e2e[str(N)] = row
        print("e2e", N, {k: round(v, 1) for k, v in row.items() if "share" not in k}, flush=True)
        torch.cuda.empty_cache()
        json.dump(res, open(OUT, "w"), indent=1)
    del B; torch.cuda.empty_cache()

# ------------------------------------------------------------------------------- (2) concurrency
if PART in ("conc", "all"):
    conc = res.setdefault("concurrency", {})
    for N in [2048, 8192]:
        for R in [1, 4, 16]:
            if R * N > 65536:
                continue
            sks = [src(N, i) for i in range(R)]
            outs = [torch.empty(2, LT, N, H, D, device=dev, dtype=torch.bfloat16) for _ in range(R)]
            streams = [torch.cuda.Stream(device=dev) for _ in range(R)]
            row = {}
            # CacheBridge official: one stream per request (its runtime overlaps requests on separate channels)
            ts = []
            for r in range(5):
                torch.cuda.synchronize(); a = time.time()
                for i in range(R):
                    with torch.cuda.stream(streams[i]):
                        cb_translate(*sks[i], outs[i])
                torch.cuda.synchronize(); ts.append((time.time() - a) * 1e3)
            row["cachebridge_batch_ms"] = med(ts[1:])
            # ours: all requests concatenated along tokens in one launch (per-token positions)
            catk = torch.cat([s[0] for s in sks], 1); catv = torch.cat([s[1] for s in sks], 1)
            pos = torch.arange(N, device=dev).repeat(R)
            outc = torch.empty(2, LT, R * N, H, D, device=dev, dtype=torch.bfloat16)
            ts = []
            for r in range(5):
                torch.cuda.synchronize(); a = time.time()
                fh(catk, catv, out=outc, positions=pos)
                torch.cuda.synchronize(); ts.append((time.time() - a) * 1e3)
            row["ours_batch_ms"] = med(ts[1:])
            for k in ("cachebridge", "ours"):
                row[f"{k}_throughput_Mtok_s"] = R * N / row[f"{k}_batch_ms"] / 1e3
            conc[f"N{N}_R{R}"] = row
            print("conc", N, R, {k: round(v, 2) for k, v in row.items()}, flush=True)
            del sks, outs, catk, catv, outc; torch.cuda.empty_cache()
    json.dump(res, open(OUT, "w"), indent=1)

# ------------------------------------------------------------------------------- (3) serving interference
if PART in ("serve", "all"):
    B = AutoModelForCausalLM.from_pretrained(f"{MODELS}/Qwen3-4B", torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    sv_res = res.setdefault("serving", {}); res["serving_cfg"] = dict(batch=8, ctx=2048, handoff_tokens=8192, seconds=12)
    BS, CTX, NH, DUR = 8, 2048, 8192, 12.0
    ids = torch.randint(100, 50000, (BS, CTX), device=dev)
    sk, sv = src(NH)
    out = torch.empty(2, LT, NH, H, D, device=dev, dtype=torch.bfloat16)
    side = torch.cuda.Stream(device=dev, priority=0)
    main = torch.cuda.Stream(device=dev, priority=-1)

    def run(rate, method):
        cache = DynamicCache()
        with torch.no_grad():
            B(ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        torch.cuda.empty_cache()
        tok = torch.randint(100, 50000, (BS, 1), device=dev)
        steps, lat = 0, []
        nxt = time.time() + (1.0 / rate if rate else 1e9)
        pend = []
        t0 = time.time()
        while time.time() - t0 < DUR and cache.get_seq_length() < CTX + 900:
            if rate and time.time() >= nxt:  # a handoff arrives: translate on the low-priority side stream
                a = ev(side)
                with torch.cuda.stream(side):
                    (cb_translate(sk, sv, out) if method == "cachebridge" else fh(sk, sv, out=out))
                b = ev(side)
                pend.append((a, b)); nxt += 1.0 / rate
            with torch.cuda.stream(main), torch.no_grad():
                B(tok, past_key_values=cache, use_cache=True)
            main.synchronize()
            steps += 1
        torch.cuda.synchronize()
        el = time.time() - t0
        lat = [a.elapsed_time(b) for a, b in pend]
        return dict(decode_tok_s=steps * BS / el, step_ms=el / steps * 1e3, handoffs=len(lat),
                    translate_ms_under_load=med(lat) if lat else None)

    for rate in [0, 1, 2, 4]:
        for method in (["none"] if rate == 0 else ["cachebridge", "ours"]):
            r = run(rate, method)
            sv_res[f"rate{rate}_{method}"] = r
            print("serve", rate, method, r, flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
print("SYSDONE")
