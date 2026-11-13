# M4 system measurements (Case 1): T_switch stage breakdown, peak memory (C5), overlap (C6),
# AOT incremental translation (C1) vs shadow prefill (B6) background cost & interference, background
# re-prefill (C3), and correctness unit tests (incremental == one-shot, free-as-you-go == baseline).
# env: PAIR LENS(csv) TNEW(128) REPS(20) WARM(5) DELTA(512)
import os, json, math, statistics as st, torch, numpy as np
from ks import *
from translate import Translator, selective_recompute

E = os.environ.get
PAIRS = {"P1": ("Qwen3-4B", "Qwen3-1.7B"), "P2": ("Qwen3-1.7B", "Qwen3-4B")}
PAIR = E("PAIR")
LENS = [int(x) for x in E("LENS", "2048,4096,8192,16384,32768").split(",")]
TNEW, REPS, WARM, DELTA = int(E("TNEW", 128)), int(E("REPS", 20)), int(E("WARM", 5)), int(E("DELTA", 512))
A, B = Model(PAIRS[PAIR][0]), Model(PAIRS[PAIR][1])
maps = torch.load(f"{ROOT}/maps/{PAIR}.pt")
TRb1 = Translator(A, B, maps, k=3, vkind="WVa")  # B1 / ours: identical kernel cost (low-rank merged)
TRb2 = Translator(A, B, maps, k=3, vkind="WV")
dev = "cuda"
OUT = f"{ROOT}/results/timing_{PAIR}.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
res["cfg"] = dict(PAIR=PAIR, TNEW=TNEW, REPS=REPS, WARM=WARM, DELTA=DELTA)
sync = torch.cuda.synchronize
method_aot_cache = None


def ev():
    e = torch.cuda.Event(enable_timing=True); e.record(); return e


def fill_A(N, ids):
    cA = A.new_cache(N + DELTA + 64)
    A.forward(ids[:, :N], cA, want="hidden")
    return cA


def switch(method, N, ids, new, cA, lazy=False, free=False, overlap=False):
    """One switch. Returns dict of stage ms + peak GB. cA is consumed if free=True."""
    sync(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t = {}
    e0 = ev()
    if method == "native":
        if free:  # A's cache is no longer needed: drop it before B's prefill
            for l in range(A.L):
                cA.free_layer(l)
        cB = Cache(B, N + TNEW + 8)
        B.forward(ids[:, :N], cB, want="hidden"); e1 = ev()
    elif method == "aot":  # C1 with lead>=1: B's cache already translated in the background; nothing on the critical path
        cB = method_aot_cache
        e1 = ev()
    else:
        tr = TRb2 if method == "B2" else TRb1
        cB = Cache(B, N + TNEW + 8, lazy=lazy)
        if overlap:  # C6: translation on a side stream, new-input layer l waits for translated layer l
            s = torch.cuda.Stream()
            evs = [torch.cuda.Event() for _ in range(B.L)]
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                tr.translate(cA, cB, ids, 0, N, free_src=free, on_layer=lambda l: evs[l].record(s))
            e1 = e0
            lg = B.forward(new, cB, n0=N, want="last", pre_layer=lambda l: torch.cuda.current_stream().wait_event(evs[l]))
            torch.cuda.current_stream().wait_stream(s)
            e2 = ev()
        else:
            tr.translate(cA, cB, ids, 0, N, free_src=free)
            if method == "B3":
                e1a = ev()
                selective_recompute(B, cB, ids, N, 0.3)
            e1 = ev()
    if not overlap or method in ("native", "aot"):
        lg = B.forward(new, cB, n0=N, want="last")
        e2 = ev()
    nxt = lg[0, -1].argmax().view(1, 1)
    B.forward(nxt, cB, n0=N + new.shape[1], want="last")
    e3 = ev()
    sync()
    out = dict(total=e0.elapsed_time(e3), new_input=e1.elapsed_time(e2) if not overlap else None, decode1=e2.elapsed_time(e3))
    if method == "native":
        out["prefill"] = e0.elapsed_time(e1)
    elif method == "aot":
        out["translate"] = 0.0
    elif overlap:
        out["translate+new_input(overlapped)"] = e0.elapsed_time(e2)
    else:
        if method == "B3":
            out["translate"] = e0.elapsed_time(e1a); out["recompute30"] = e1a.elapsed_time(e1)
        else:
            out["translate"] = e0.elapsed_time(e1)
    out["peak_GB"] = torch.cuda.max_memory_allocated() / 2**30
    out["peak_over_start_GB"] = (torch.cuda.max_memory_allocated() - base) / 2**30
    out["_cB"] = cB
    return out


def summarize(xs):
    keys = [k for k in xs[0] if not k.startswith("_") and xs[0][k] is not None]
    return {k: dict(p50=float(np.median([x[k] for x in xs])), p95=float(np.percentile([x[k] for x in xs], 95))) for k in keys}


# ---------------- unit tests ----------------
torch.manual_seed(0)
N0 = 3000
ids = torch.randint(100, 50000, (1, N0 + 600), device=dev)
cA = fill_A(N0 + 512, ids)
c1 = Cache(B, N0 + 600); TRb1.translate(cA, c1, ids, 0, N0 + 512)
c2 = Cache(B, N0 + 512)
for a, b in [(0, 1000), (1000, 1777), (1777, N0), (N0, N0 + 512)]:  # incremental, odd boundaries
    TRb1.translate(cA, c2, ids, a, b)
inc_err = max(max((c1.k[l][:, :N0 + 512].float() - c2.k[l][:, :N0 + 512].float()).abs().max().item(),
                  (c1.v[l][:, :N0 + 512].float() - c2.v[l][:, :N0 + 512].float()).abs().max().item()) for l in range(B.L))
c3 = Cache(B, N0 + 512, lazy=True)
cA2 = fill_A(N0 + 512, ids)
TRb1.translate(cA2, c3, ids, 0, N0 + 512, free_src=True)
free_err = max((c1.k[l][:, :N0 + 512].float() - c3.k[l][:, :N0 + 512].float()).abs().max().item() for l in range(B.L))
new = ids[:, N0 + 512:N0 + 512 + 64]
lg1 = B.forward(new, c1, n0=N0 + 512, want="all")
c4 = Cache(B, N0 + 600)
s = torch.cuda.Stream(); evs = [torch.cuda.Event() for _ in range(B.L)]
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    TRb1.translate(cA, c4, ids, 0, N0 + 512, on_layer=lambda l: evs[l].record(s))
lg4 = B.forward(new, c4, n0=N0 + 512, want="all", pre_layer=lambda l: torch.cuda.current_stream().wait_event(evs[l]))
sync()
res["unit"] = dict(incremental_vs_oneshot_maxabs=inc_err, free_as_you_go_vs_baseline_maxabs=free_err,
                   overlap_vs_serial_logits_maxabs=(lg1 - lg4).abs().max().item())
log("unit", res["unit"])
del cA, cA2, c1, c2, c3, c4; torch.cuda.empty_cache()

# ---------------- T_switch / peak memory ----------------
CONFIGS = [("native", {}), ("native_freeA", dict(free=True)), ("B2", {}), ("B1", {}), ("B3", {}),
           ("C5", dict(lazy=True, free=True)), ("C5+C6", dict(lazy=True, free=True, overlap=True)), ("C1_aot", {})]
res.setdefault("switch", {})
for N in LENS:
    ids = torch.randint(100, 50000, (1, N + TNEW + 8), device=dev)
    new = ids[:, N:N + TNEW]
    row = {}
    for name, kw in CONFIGS:
        meth = {"native_freeA": "native", "C5": "B1", "C5+C6": "B1", "C1_aot": "aot"}.get(name, name)
        xs = []
        try:
            for r in range(WARM + REPS):
                cA = fill_A(N, ids)
                if meth == "aot":
                    method_aot_cache = Cache(B, N + TNEW + 8); TRb1.translate(cA, method_aot_cache, ids, 0, N)
                x = switch(meth, N, ids, new, cA, **kw)
                x.pop("_cB"); del cA
                method_aot_cache = None
                torch.cuda.empty_cache()
                if r >= WARM:
                    xs.append(x)
            row[name] = summarize(xs)
            log(PAIR, N, name, {k: round(v["p50"], 2) for k, v in row[name].items()})
        except torch.OutOfMemoryError:
            row[name] = "OOM"; log(PAIR, N, name, "OOM")
            torch.cuda.empty_cache()
    res["switch"][str(N)] = row
    json.dump(res, open(OUT, "w"), indent=1)

# ---------------- background cost: C1 (incremental translation) vs B6 (shadow prefill) ----------------
fg = torch.cuda.Stream(priority=-1)  # foreground (high priority)
bg = torch.cuda.Stream(priority=0)   # background (low priority)


def decode_loop(M, cache, n, steps=48):
    x = torch.tensor([[1000]], device=dev)
    ts = []
    with torch.cuda.stream(fg):
        for i in range(steps):
            a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
            a.record(fg); M.forward(x, cache, n0=n + i, want="last"); b.record(fg)
            ts.append((a, b))
    sync()
    return [a.elapsed_time(b) for a, b in ts]


res.setdefault("background", {})
for N in [n for n in LENS if n <= 16384]:
    ids = torch.randint(100, 50000, (1, N + DELTA + 64), device=dev)
    cA = fill_A(N, ids)
    A.forward(ids[:, N:N + DELTA], cA, n0=N, want="hidden")  # A's latest turn (DELTA tokens) is in its cache
    cB = Cache(B, N + DELTA + 64); TRb1.translate(cA, cB, ids, 0, N)
    cBs = Cache(B, N + DELTA + 64); B.forward(ids[:, :N], cBs, want="hidden")
    row = {}
    jobs = {"C1_translate_turn": lambda: TRb1.translate(cA, cB, ids, N, N + DELTA),
            "B6_shadow_prefill_turn": lambda: B.forward(ids[:, N:N + DELTA], cBs, n0=N, want="hidden")}
    for name, fn in jobs.items():
        for _ in range(3):
            fn()
        sync()
        iso = []
        for _ in range(10):
            a = ev(); fn(); b = ev(); sync(); iso.append(a.elapsed_time(b))
        row[name + "_gpu_ms"] = float(np.median(iso))
    # interference on A's foreground decoding (pre-switch): bg job issued once, at start of fg decode window
    base = decode_loop(A, cA, N + DELTA)
    base = decode_loop(A, cA, N + DELTA)
    row["A_decode_ms_alone"] = float(np.median(base))
    for name, fn in jobs.items():
        sl = []
        for _ in range(5):
            sync()
            with torch.cuda.stream(bg):
                fn()
            ts = decode_loop(A, cA, N + DELTA, steps=24)
            sl.append(ts)
        sl = np.array(sl)
        row[name + "_A_decode_ms_during"] = float(np.median(sl[:, :8]))  # first tokens overlap the bg job
        row[name + "_A_extra_ms_total"] = float(sl.sum(1).mean() - 24 * row["A_decode_ms_alone"])
    row["extra_mem_GB_both"] = kv_bytes(B, N + DELTA) / 2**30  # both keep a B-format cache during the warning period
    res["background"][str(N)] = row
    log("bg", N, row)
    del cA, cB, cBs; torch.cuda.empty_cache()
    json.dump(res, open(OUT, "w"), indent=1)

# ---------------- C3: background native re-prefill after the switch ----------------
res.setdefault("c3", {})
for N in LENS:
    try:
        ids = torch.randint(100, 50000, (1, N + 300), device=dev)
        cB = Cache(B, N + 300); B.forward(ids[:, :N], cB, want="hidden")  # translated cache stand-in (B decoding fg)
        cR = Cache(B, N + 8)
        iso = []
        for _ in range(3):
            a = ev(); B.forward(ids[:, :N], cR, n0=0, want="hidden"); b = ev(); sync(); iso.append(a.elapsed_time(b))
        alone = decode_loop(B, cB, N, 32)
        done_ev = torch.cuda.Event(enable_timing=True); st_ev = torch.cuda.Event(enable_timing=True)
        st_ev.record(bg)
        with torch.cuda.stream(bg):
            B.forward(ids[:, :N], cR, n0=0, want="hidden")
            done_ev.record(bg)
        during = decode_loop(B, cB, N, 128)
        sync()
        bg_ms = st_ev.elapsed_time(done_ev)
        res["c3"][str(N)] = dict(reprefill_alone_ms=float(np.median(iso)), reprefill_concurrent_ms=bg_ms,
                                 B_decode_ms_alone=float(np.median(alone)), B_decode_ms_during=float(np.median(during[:max(1, min(128, int(bg_ms / np.median(alone))))])),
                                 B_decode_128tok_ms_during=float(sum(during)), B_decode_128tok_ms_alone=float(np.median(alone) * 128))
        log("c3", N, res["c3"][str(N)])
        del cB, cR; torch.cuda.empty_cache()
    except torch.OutOfMemoryError:
        res["c3"][str(N)] = "OOM"; torch.cuda.empty_cache()
    json.dump(res, open(OUT, "w"), indent=1)
log("TIMINGDONE")
