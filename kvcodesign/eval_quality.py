# [kvcodesign] copy of kvbase/eval_quality.py + low-rank shared-latent translators (method rrr_k{rk}_v{rv}[_bf16]).
# Quality under the CacheBridge handoff protocol: source (Qwen3-1.7B) prefills history H; target (Qwen3-4B) gets
# H's KV via a translator, processes the query Q and answers.  Thinking disabled.
# methods: native | src_native (1.7B reads and answers) | cb_official | cb_fused | kvbridge | dense_fused | dense_fp8
#          rrr_k{rk}_v{rv}[_bf16] low-rank | lrf_bf16 / lrf_fp8 fused kernels | tid_k{rk}_v{rv} low-rank with token
#          identity (fit_tid.py) | <kvbridge or rrr_...>_l0 = same translator with target layer 0 computed exactly
#          kl_k{rk}_v{rv}_{tag}[_bf16] low-rank map fine-tuned by KL through the frozen target (train_kl.py)
# tasks  : see tasks.py (quality | hotpotqa_e | gsm8k | gsm8k_h | squad | kvret | passkey_{2k,8k,16k})
# every generated answer records n_out (tokens generated) and trunc (hit the cap); gsm8k_h also records score_256
#          rare{pct} = RaReCache (rare.py): kvbridge translation, then the target recomputes the pct% highest-scoring
#          positions | none (target sees only the new message)
#          kb3 / kb3kl: kvbridge-format map over 3 source layers, closed form / + KL distillation (train_dense_kl.py)
#          sum{pct}: the old model, which has already read the history, writes a summary (greedy, at most pct% of the
#            history length, capped at 512 tokens); the new model reads only the summary (our prompt, our code)
#          lingua{pct}: LLMLingua-2 (official llmlingua package, model microsoft/llmlingua-2-xlm-roberta-large-meetingbank)
#            compresses the history text to pct% of its tokens; the new model re-reads the compressed text | for kind "kl" tasks every method is scored by
#          KL(native target || target with this cache) over the continuation tokens (lower is better)
# env: TASK N METHODS MAPPER_CB OUT [MX] [DIR=17to4|4to17]  (4to17: Qwen3-4B -> Qwen3-1.7B, strong to weak)
# MODE=ttft: time-to-first-token after the switch instead of quality (PLAN_v6 sec. 2.3), same code paths as the quality
#   runs.  Clock starts when the target receives the source cache (the source has already read the history, cache on
#   the same GPU); it stops when the target has processed a 128-token new user message and produced its first token.
#   Everything a method does in between is included (translation, re-prefill, ...).  HF transformers, sdpa attention,
#   no CUDA graph.  env LENS (default 2048,8192,16384,32768) REPS (5, after 2 warm-ups) TTFT_OUT
#   Run ONE method per process (run_ttft.sh) so that no other method's weights sit in memory.  Before every timed
#   repetition the GPU is checked for other processes' memory (total used - ours); if another job is on the card the
#   repetition is discarded and retried after a pause (up to 30 times), and the row records how many were discarded.
import os, sys, json, numpy as np
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # before importing transformers / huggingface_hub
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

E = os.environ.get
ROOT = os.path.expanduser("~/kvbase")
sys.path.insert(0, ROOT)
MODELS = os.path.expanduser("~/kvswitch/models")
TASK, NS = E("TASK", "quality"), int(E("N", 200))
METHODS = E("METHODS", "native,cb_official,cb_fused,kvbridge,dense_fused,dense_fp8").split(",")
DIR = E("DIR", "17to4")
SRC_M, TGT_M = {"17to4": ("Qwen3-1.7B", "Qwen3-4B"), "4to17": ("Qwen3-4B", "Qwen3-1.7B"),
                "b17to17": ("Qwen3-1.7B-Base", "Qwen3-1.7B"), "17tob17": ("Qwen3-1.7B", "Qwen3-1.7B-Base"),   # DroidSpeak pair
                "same17": ("Qwen3-1.7B", "Qwen3-1.7B"),   # same17: sanity check only (every method must equal native)
                "8to14": ("Qwen3-8B-ms", "Qwen3-14B-ms")}[DIR]   # CacheBridge paper's pair; target spans two GPUs
CB = E("MAPPER_CB", f"{ROOT}/mappers/cachebridge_{DIR}")
OUT = E("OUT", os.path.expanduser(f"~/kvcodesign/results/{'' if DIR == '17to4' else DIR + '/'}quality_{TASK}.json"))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
# our low-rank maps: 17to4 from screen2.py (mappers/), other directions from fit_lowrank_dir.py (mappers/{DIR}/)
LRDIR = os.path.expanduser("~/kvcodesign/mappers" + ("" if DIR == "17to4" else f"/{DIR}"))
dev = "cuda"
H, D = 8, 128

tok = AutoTokenizer.from_pretrained(f"{MODELS}/Qwen3-4B")   # the Qwen3 sizes share one tokenizer
S = AutoModelForCausalLM.from_pretrained(f"{MODELS}/{SRC_M}", torch_dtype=torch.bfloat16).to(dev).eval()
if DIR == "8to14":   # 3 GPUs: source + mapper on cuda:0, the 14B target split over cuda:1 and cuda:2
    T = AutoModelForCausalLM.from_pretrained(f"{MODELS}/{TGT_M}", torch_dtype=torch.bfloat16, device_map="auto",
                                             max_memory={0: "0GiB", 1: "20GiB", 2: "20GiB"}).eval()
else:
    T = AutoModelForCausalLM.from_pretrained(f"{MODELS}/{TGT_M}", torch_dtype=torch.bfloat16).to(dev).eval()
LS, LT = S.config.num_hidden_layers, T.config.num_hidden_layers
TDEV = [next(T.model.layers[l].parameters()).device for l in range(LT)]   # mapped KV of layer l must live where layer l runs
TR = {}
# DroidSpeak adapted across sizes (droid.py): dsx_{direct|kvbridge|lrf}_{b} -- the reused KV comes from a translator
NEED = list(METHODS) + [x for x, pre in (("kvbridge", "dsx_kvbridge_"), ("lrf_bf16", "dsx_lrf_")) if any(m.startswith(pre) for m in METHODS)]
if {"cb_official"} & set(METHODS):
    from cachebridge.mapper import PageMapper
    TR["cb_official"] = PageMapper(CB, method="cachebridge", source_rope_theta=1e6, target_rope_theta=1e6, device=dev)
if {"cb_fused"} & set(METHODS):
    from fastmap import FastHeadLocal
    TR["cb_fused"] = FastHeadLocal(CB)
if {"cb_cublas32"} & set(METHODS):
    from fastmap import CublasHeadLocalF32
    TR["cb_cublas32"] = CublasHeadLocalF32(CB)
if {"cb_tf32x3"} & set(METHODS):
    from fastmap import FastHeadLocalF32
    TR["cb_tf32x3"] = FastHeadLocalF32(CB, prec="tf32x3", BM=16, num_warps=4)
if any(m.startswith("lingua") for m in METHODS):
    from llmlingua import PromptCompressor
    LINGUA = PromptCompressor(model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank", use_llmlingua2=True, device_map=dev)
SUM_ASK = ("<|im_end|>\n<|im_start|>user\nBefore we continue, write a concise summary of everything above. Keep every name, "
           "number, date, code and question exactly as written.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
SUM_HEAD = "Here is a summary of the earlier conversation:\n"
if any(m.startswith("rare") for m in METHODS):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rare
    VR = {k: v.to(dev) for k, v in torch.load(os.path.expanduser(f"~/kvcodesign/mappers/rare_vr_{DIR}.pt")).items()}
if {"kvbridge", "kvbridge_l0", "dense_fused", "dense_fp8"} & set(NEED) or any(m.startswith("rare") for m in METHODS):
    from kvbridge import Mapper as KBMapper, transfer as kb_transfer
    kb = KBMapper.load(f"{ROOT}/mappers/kvbridge_{DIR}", device=dev)
    from fastmap import FastDense
    if "dense_fused" in METHODS:
        TR["dense_fused"] = FastDense(kb)
    if "dense_fp8" in METHODS:
        TR["dense_fp8"] = FastDense(kb, fp8=True)
KBS = {"kvbridge": kb} if "kb" in dir() else {}
import re as _re
for m_ in [x for x in METHODS if _re.fullmatch(r"kb\d+(cb|kl)?", x)]:   # kb{n}[cb|kl]: kvbridge-format map mappers/kvbridge{n}[cb|kl]_{DIR}   # kb3cb: fit_cb5.py (all-head map + CacheBridge energy-constrained V calibration)   # train_dense_kl.py: KV-Lingo / Hybrid-KV recipe (top-3 closed form, then KL distillation)
    if m_ in METHODS:
        from kvbridge import Mapper as KBMapper, transfer as kb_transfer
        KBS[m_] = KBMapper.load(f"{ROOT}/mappers/kvbridge{m_[2:]}_{DIR}", device=dev)


class LowRank:
    """Shared-latent reduced-rank ridge: strip source RoPE -> z = (x - mu_x) P -> y = z D + mu_y -> target RoPE.
    K and V use separate latents (ranks rk, rv). Weights from screen2.py (metric 'var', alpha 0.01)."""
    def __init__(self, rk, rv, dtype=torch.float32, path=None):
        if path:   # one file holding all four matrices (train_kl.py: behaviourally fine-tuned low-rank map)
            a = b = torch.load(path)
        else:
            a = torch.load(f"{LRDIR}/rrr_var_r{rk}.pt")
            b = torch.load(f"{LRDIR}/rrr_var_r{rv}.pt")
        PK = LS * H * D
        g = lambda t: t.to(dev, dtype)
        self.Pk, self.Dk, self.Pv, self.Dv = g(a["Pk"]), g(a["Dk"]), g(b["Pv"]), g(b["Dv"])
        self.mxk, self.mxv = g(a["mx"][:PK]), g(a["mx"][PK:])
        self.myk, self.myv = g(a["my"][:LT * H * D]), g(a["my"][LT * H * D:])
        self.dt = dtype
        inv = 1.0 / (1e6 ** (torch.arange(0, D, 2, device=dev, dtype=torch.float32) / D))
        self.inv = inv

    def cs(self, n):
        ang = torch.arange(n, device=dev, dtype=torch.float32)[:, None] * self.inv[None]
        return torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)  # [n, D]

    @staticmethod
    def rot(x):
        return torch.cat([-x[..., D // 2:], x[..., :D // 2]], -1)

    @torch.no_grad()
    def __call__(self, sk, sv, chunk=2048):  # sk, sv: [LS, n, H, D] (keys rotated) -> [2, LT, n, H, D] bf16
        n = sk.shape[1]
        out = torch.empty(2, LT, n, H, D, device=dev, dtype=torch.bfloat16)
        cos, sin = self.cs(n)
        for p in range(0, n, chunk):
            q = min(n, p + chunk); m = q - p
            c, s_ = cos[None, p:q, None], sin[None, p:q, None]
            k = sk[:, p:q].float()
            k = k * c - self.rot(k) * s_                                       # strip source RoPE
            xk = k.permute(1, 0, 2, 3).reshape(m, -1).to(self.dt) - self.mxk
            xv = sv[:, p:q].permute(1, 0, 2, 3).reshape(m, -1).to(self.dt) - self.mxv
            yk = ((xk @ self.Pk) @ self.Dk + self.myk).float().view(m, LT, H, D).permute(1, 0, 2, 3)
            out[1, :, p:q] = ((xv @ self.Pv) @ self.Dv + self.myv).view(m, LT, H, D).permute(1, 0, 2, 3).to(torch.bfloat16)
            out[0, :, p:q] = (yk * c + self.rot(yk) * s_).to(torch.bfloat16)  # apply target RoPE
        return out


def rope_keys(k, n):   # k [n, H, D] content space -> rotated (target RoPE, positions 0..n-1), fp32
    inv = 1.0 / (1e6 ** (torch.arange(0, D, 2, device=dev, dtype=torch.float32) / D))
    ang = torch.arange(n, device=dev, dtype=torch.float32)[:, None] * inv[None]
    c, s_ = torch.cat([ang.cos(), ang.cos()], -1)[:, None], torch.cat([ang.sin(), ang.sin()], -1)[:, None]
    return k * c + torch.cat([-k[..., D // 2:], k[..., :D // 2]], -1) * s_


@torch.no_grad()
def tgt_l0(hist):
    """Token identity in the target model: its layer-0 K (content space) and V depend only on the token ids
    (shared tokenizer), so they are computed exactly, without reading the history.  -> k0, v0 [n, H, D] fp32"""
    lay = T.model.layers[0]; a = lay.self_attn
    x = lay.input_layernorm(T.model.embed_tokens(torch.tensor([hist], device=dev)))
    n = len(hist)
    return a.k_norm(a.k_proj(x).view(1, n, H, D))[0].float(), a.v_proj(x).view(n, H, D).float()


class LowRankTID:
    """Low-rank map with token identity (fit_tid.py): target layer 0 exact; layers 1..35 from
    y = ([x - mx, s (e - me)] P) D + my  with e = exact target layer-0 K,V of the same token."""
    def __init__(self, rk, rv, dtype=torch.float32):
        a = torch.load(os.path.expanduser(f"~/kvcodesign/mappers/tid_r{rk}.pt"))
        b = torch.load(os.path.expanduser(f"~/kvcodesign/mappers/tid_r{rv}.pt"))
        PK, QK, HD = LS * H * D, LT * H * D, H * D
        g = lambda t: t.to(dev, dtype)
        self.Pk, self.Dk, self.Pv, self.Dv = g(a["Pk"]), g(a["Dk"]), g(b["Pv"]), g(b["Dv"])
        self.mxk, self.mxv, self.me = g(a["mx"][:PK]), g(a["mx"][PK:]), g(a["me"])
        self.myk, self.myv = g(a["my"][HD:QK]), g(a["my"][QK + HD:])
        self.sk_, self.sv_, self.dt = a["s_k"], b["s_v"], dtype
        # The first token is the attention sink; identity features wreck its V (relative error 0.44 -> 1.99 offline,
        # diag_tid.py), so position 0 is translated by the plain low-rank map of the same ranks.
        self.lr = LowRank(rk, rv, dtype)

    @torch.no_grad()
    def __call__(self, sk, sv, k0, v0, chunk=2048):
        n = sk.shape[1]
        out = torch.empty(2, LT, n, H, D, device=dev, dtype=torch.bfloat16)
        cos, sin = self.lr.cs(n)
        e = torch.cat([k0.reshape(n, -1), v0.reshape(n, -1)], 1).to(self.dt) - self.me
        out[0, 0], out[1, 0] = rope_keys(k0, n).to(torch.bfloat16), v0.to(torch.bfloat16)
        for p in range(0, n, chunk):
            q = min(n, p + chunk); m = q - p
            c, s_ = cos[None, p:q, None], sin[None, p:q, None]
            k = sk[:, p:q].float()
            k = k * c - LowRank.rot(k) * s_
            xk = torch.cat([k.permute(1, 0, 2, 3).reshape(m, -1).to(self.dt) - self.mxk, e[p:q] * self.sk_], 1)
            xv = torch.cat([sv[:, p:q].permute(1, 0, 2, 3).reshape(m, -1).to(self.dt) - self.mxv, e[p:q] * self.sv_], 1)
            yk = ((xk @ self.Pk) @ self.Dk + self.myk).float().view(m, LT - 1, H, D).permute(1, 0, 2, 3)
            out[1, 1:, p:q] = ((xv @ self.Pv) @ self.Dv + self.myv).view(m, LT - 1, H, D).permute(1, 0, 2, 3).to(torch.bfloat16)
            out[0, 1:, p:q] = (yk * c + LowRank.rot(yk) * s_).to(torch.bfloat16)
        out[:, 1:, :1] = self.lr(sk[:, :1], sv[:, :1])[:, 1:]
        return out


# method suffix "_l0": any translator, with target layer 0 replaced by the exact token-identity K,V
for m in NEED:
    if m.startswith(("kl_", "tid_")):
        assert DIR == "17to4", "KL-tuned / token-identity low-rank maps exist for Qwen3-1.7B -> Qwen3-4B only"
    if m.startswith("kl_"):    # kl_k{rk}_v{rv}_{tag}[_bf16]: low-rank map fine-tuned by KL through the frozen target
        base = m[:-5] if m.endswith("_bf16") else m
        TR[m] = LowRank(0, 0, torch.bfloat16 if m.endswith("_bf16") else torch.float32, path=os.path.expanduser(f"~/kvcodesign/mappers/{base}.pt"))
        continue
    if m.startswith("tid_"):   # tid_k{rk}_v{rv}[_bf16]
        parts = m.split("_")
        TR[m] = LowRankTID(int(parts[1][1:]), int(parts[2][1:]), torch.bfloat16 if m.endswith("_bf16") else torch.float32)
        continue
    if m.endswith("_l0") and m.startswith("rrr_"):
        parts = m.split("_")
        TR[m] = LowRank(int(parts[1][1:]), int(parts[2][1:]), torch.bfloat16 if "_bf16" in m else torch.float32)
        continue
    if m in ("lrf_bf16", "lrf_fp8"):   # fused Triton kernels (fastlr.py), K 768 + V 1536
        sys.path.insert(0, os.path.expanduser("~/kvcodesign"))
        import fastlr
        from fastlr import FastLowRank
        fastlr.LS, fastlr.LT = LS, LT   # the kernels read the layer counts at call time
        assert m == "lrf_bf16" or DIR == "17to4", "FP8 activation scales were calibrated for 17to4 only"
        sc = json.load(open(os.path.expanduser("~/kvcodesign/data/fp8_scales_k768_v1536.json")))
        TR[m] = FastLowRank(768, 1536, fp8=(m == "lrf_fp8"), scales=sc, mapper_dir=LRDIR)
    if m.startswith("rrr_"):
        parts = m.split("_")
        rk, rv = int(parts[1][1:]), int(parts[2][1:])
        TR[m] = LowRank(rk, rv, torch.bfloat16 if m.endswith("_bf16") else torch.float32)


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasks import build_samples, score   # task construction and format-insensitive scoring (tasks.py)

samples = build_samples(TASK, NS, lambda t: tok(t, add_special_tokens=False).input_ids)[int(E("OFFSET", 0)):]   # OFFSET: profiling split
if E("MX"):   # optional override of the generation cap
    for s_ in samples:
        if "mx" in s_:
            s_["mx"] = int(E("MX"))


# ---------------------------------------------------------------- handoff
PRE = tok("<|im_start|>user\n", add_special_tokens=False).input_ids


@torch.no_grad()
def prefill(M, ids, c=None, chunk=8192):
    """Prefill `ids` into cache c with the decoder body only, in chunks of `chunk` tokens."""
    c = DynamicCache() if c is None else c
    for p in range(0, len(ids), chunk):
        M.model(torch.tensor([ids[p:p + chunk]], device=dev), past_key_values=c, use_cache=True)
    return c


SC = None   # MODE=ttft: the source cache, computed before the clock starts
SE = {}     # MODE=ttft: DroidSpeak E caches {a: hidden state entering layer a}, computed before the clock starts
SND = [None, None, None]   # DroidSpeak: sender KV + all hidden states of the current history, shared by droid* methods
import droid


@torch.no_grad()
def target_cache(method, hist):   # history prefills run the decoder body only (no lm_head over all positions)
    if method == "none":   # no history: the target only sees the start of the user turn, then the new message
        c = DynamicCache(); T.model(torch.tensor([PRE], device=dev), past_key_values=c, use_cache=True); return c
    if method == "native":
        return prefill(T, hist)
    if method == "src_native":   # source model (1.7B) reads the history and answers itself
        return prefill(S, hist)
    if method.startswith("dsx_"):    # DroidSpeak adapted across sizes: reused KV from {direct, kvbridge, lrf}, layers 0..b-1 recomputed
        _, base, b = method.split("_")
        if base == "direct":
            c = droid.depth_map(SC if SC is not None else prefill(S, hist), LS, LT)
        else:
            c = target_cache({"kvbridge": "kvbridge", "lrf": "lrf_bf16"}[base], hist)
        return droid.prefix_recompute(T, c, hist, int(b))
    if method.startswith("droid"):   # DroidSpeak (droid.py): sender KV + receiver recomputes layers a..b-1
        a, b = droid.parse(method)
        if SC is not None:
            return droid.receive(T, SC, SE.get(a), a, b)
        if SND[0] != hist:
            SND[:] = [None, None, None]; torch.cuda.empty_cache()
            SND[:] = [list(hist), *droid.sender(S, hist)]
        return droid.receive(T, SND[1], SND[2].get(a), a, b)
    sc = SC if SC is not None else prefill(S, hist)
    if method.startswith("sum"):    # old model summarises from its own cache, new model reads the summary
        sc = SC if SC is not None else prefill(S, hist)
        n0 = sc.get_seq_length()
        budget = min(512, max(16, int(len(hist) * int(method[3:]) / 100)))
        ask = tok(SUM_ASK, add_special_tokens=False).input_ids
        lg = S(torch.tensor([ask], device=dev), past_key_values=sc, use_cache=True).logits[0, -1]
        out, nxt = [], int(lg.argmax())
        for _ in range(budget):
            if nxt in STOP:
                break
            out.append(nxt)
            nxt = int(S(torch.tensor([[nxt]], device=dev), past_key_values=sc, use_cache=True).logits[0, -1].argmax())
        sc.crop(n0)                     # leave the old model's cache as it was
        new = PRE + tok(SUM_HEAD + tok.decode(out, skip_special_tokens=True).strip(), add_special_tokens=False).input_ids
        return prefill(T, new)
    if method.startswith("lingua"):  # LLMLingua-2 compresses the history text, new model re-reads it
        text = tok.decode(hist[len(PRE):])
        comp = LINGUA.compress_prompt(text, rate=int(method[6:]) / 100, force_tokens=["\n", "?"])["compressed_prompt"]
        new = PRE + tok(comp, add_special_tokens=False).input_ids
        return prefill(T, new)
    if method.startswith("rare"):   # RaReCache: kvbridge map + recompute the top pct% positions (rare.py)
        return rare.rarecache(T, target_cache("kvbridge", hist), hist, VR, int(method[4:]) / 100)
    if method == "kvbridge_l0":   # official dense translation, layer 0 replaced by the exact token-identity K,V
        c = target_cache("kvbridge", hist)
        k0, v0 = tgt_l0(hist)
        c.layers[0].keys = rope_keys(k0, len(hist)).transpose(0, 1)[None].to(c.layers[0].keys.dtype)
        c.layers[0].values = v0.transpose(0, 1)[None].to(c.layers[0].values.dtype)
        return c
    if method in KBS:   # kvbridge format: official weights, top-3 closed form, top-3 + KL
        mpr = KBS[method]
        n = sc.layers[0].keys.shape[2]
        if n <= 8192:
            return kb_transfer(sc, mpr)
        # long histories: the official transfer() run on 4096-token slices with explicit positions, then concatenated
        # (identical arithmetic per token; avoids the out-of-memory failure at 16K tokens on a 24 GB card)
        parts = []
        for p in range(0, n, 4096):
            cc = DynamicCache()
            for l in range(LS):
                cc.update(sc.layers[l].keys[:, :, p:p + 4096], sc.layers[l].values[:, :, p:p + 4096], l)
            parts.append(kb_transfer(cc, mpr, positions=torch.arange(p, min(n, p + 4096), device=dev)))
            del cc
        del sc
        c = DynamicCache()
        for l in range(LT):
            c.update(torch.cat([x.layers[l].keys for x in parts], 2), torch.cat([x.layers[l].values for x in parts], 2), l)
        return c
    sk = torch.stack([sc.layers[l].keys[0].transpose(0, 1) for l in range(LS)]).contiguous()  # [L,T,H,D]
    sv = torch.stack([sc.layers[l].values[0].transpose(0, 1) for l in range(LS)]).contiguous()
    n = sk.shape[1]
    del sc   # free the source cache before translating (matters at 16K tokens on a 24 GB card)
    if method == "cb_official":
        out = torch.cat([TR[method].map_page(torch.stack([sk[:, p:p + 2048], sv[:, p:p + 2048]]), position_start=p, output_device=dev)
                         for p in range(0, n, 2048)], 2)
    elif method.startswith("tid_"):
        out = TR[method](sk, sv, *tgt_l0(hist))
    else:
        out = TR[method](sk, sv)
        if method.endswith("_l0"):
            k0, v0 = tgt_l0(hist)
            out[0, 0], out[1, 0] = rope_keys(k0, n).to(out.dtype), v0.to(out.dtype)
    del sk, sv
    c = DynamicCache()
    for l in range(LT):
        k, v = out[0, l].transpose(0, 1)[None], out[1, l].transpose(0, 1)[None]
        if n <= 8192:   # long histories keep views of `out` instead of a second 2.4 GB copy (same values)
            k, v = k.contiguous(), v.contiguous()
        c.update(k.to(TDEV[l]), v.to(TDEV[l]), l)
    return c


LETTERS = [tok.encode(x, add_special_tokens=False)[0] for x in "ABCD"]
STOP = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
if TGT_M.endswith("-Base"):   # the base model ends text with <|endoftext|>
    STOP.add(tok.convert_tokens_to_ids("<|endoftext|>"))


@torch.no_grad()
def answer(c, s, M=None):   # -> (prediction, generated token ids or None, truncated at the cap?)
    M = M or T
    lg = M(torch.tensor([s["q"]], device=dev), past_key_values=c, use_cache=True).logits[0, -1]
    if s["kind"] == "mc":
        return int(lg[LETTERS].argmax()), None, False
    outt, nxt, trunc = [], int(lg.argmax()), True
    for _ in range(s["mx"]):
        if nxt in STOP:
            trunc = False
            break
        outt.append(nxt)
        nxt = int(M(torch.tensor([[nxt]], device=dev), past_key_values=c, use_cache=True).logits[0, -1].argmax())
    if trunc and nxt in STOP:   # the stop token arrived exactly at the cap
        trunc = False
    return tok.decode(outt, skip_special_tokens=True).strip(), outt, trunc


if E("MODE") == "ttft":
    import time
    from tasks import quality_docs
    LENS = [int(x) for x in E("LENS", "2048,8192,16384,32768").split(",")]
    REPS = int(E("REPS", 5))
    TOUT = E("TTFT_OUT", os.path.expanduser(f"~/kvcodesign/results/ttft_{DIR}.json"))
    text = []
    for d in quality_docs():
        text += tok("\n\n" + d[0]["article"], add_special_tokens=False).input_ids
        if len(text) > max(LENS) + 200:
            break
    tail = tok("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False).input_ids
    newmsg = tok("<|im_end|>\n<|im_start|>user\n", add_special_tokens=False).input_ids
    newmsg = newmsg + text[-(128 - len(newmsg) - len(tail)):] + tail          # the new user message: 128 tokens
    assert len(newmsg) == 128
    res = json.load(open(TOUT)) if os.path.exists(TOUT) else dict(dir=DIR, gpu=torch.cuda.get_device_name(), reps=REPS,
                                                                  note="HF transformers sdpa, no CUDA graph; ms", lens={})
    for L in LENS:
        hist = PRE + text[:L - len(PRE)]
        with torch.no_grad():
            SC = prefill(S, hist)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        for m in METHODS:
            if m in res["lens"].get(str(L), {}):
                continue
            if m.startswith("droid") and droid.parse(m)[1] > droid.parse(m)[0]:   # E cache: stored with the KV, not timed
                with torch.no_grad():
                    SE.clear(); SE.update(droid.sender(S, hist, layers=[droid.parse(m)[0]])[1])
                torch.cuda.synchronize(); base = torch.cuda.memory_allocated()
            ts, peak, busy, r = [], 0, 0, 0

            def foreign_gb():   # memory on this GPU that is not ours (other users' processes)
                free, total = torch.cuda.mem_get_info()
                return (total - free - torch.cuda.memory_reserved()) / 2**30 - 0.6   # ~0.6 GB CUDA context
            try:
                while r < 2 + REPS:
                    if foreign_gb() > 1.0:
                        busy += 1
                        if busy > 30:
                            raise RuntimeError("GPU shared with another job")
                        time.sleep(20); continue
                    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        if m == "native":      # full re-prefill: history + new message in one (chunked) pass
                            # one pass over history + new message (chunking with HF's explicit mask is ~35%
                            # slower at 16K than the single causal pass, which would penalise re-prefill)
                            c, last = DynamicCache(), hist + newmsg
                        elif m == "none":      # no history: only the new message
                            c, last = DynamicCache(), PRE + newmsg[len(tok("<|im_end|>\n<|im_start|>user\n", add_special_tokens=False).input_ids):]
                        else:
                            c, last = target_cache(m, hist), newmsg
                        nxt = int(T(torch.tensor([last], device=dev), past_key_values=c, use_cache=True, logits_to_keep=1).logits[0, -1].argmax())
                    torch.cuda.synchronize()
                    dt = (time.perf_counter() - t0) * 1000
                    peak = max(peak, torch.cuda.max_memory_allocated() - base)
                    del c
                    if foreign_gb() > 1.0:   # someone arrived during the measurement: discard and retry
                        busy += 1; continue
                    if r >= 2:
                        ts.append(dt)
                    r += 1
                ts.sort()
                row = dict(p50=ts[len(ts) // 2], p95=ts[min(len(ts) - 1, int(round(0.95 * (len(ts) - 1))))], min=ts[0],
                           extra_peak_GB=peak / 2**30, discarded_shared=busy)
            except torch.OutOfMemoryError:
                row = dict(oom=True, discarded_shared=busy)
            except RuntimeError as e:
                row = dict(error=str(e))
            torch.cuda.empty_cache()
            res["lens"].setdefault(str(L), {})[m] = row
            json.dump(res, open(TOUT, "w"), indent=1)
            print("ttft", DIR, L, m, {k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
        SC = None; torch.cuda.empty_cache()
    print("TTFTDONE")
    sys.exit(0)


@torch.no_grad()
def cont_logits(c, s, M):   # teacher-forced log-probs on the continuation (kind "kl"), fp32
    return M(torch.tensor([s["q"]], device=dev), past_key_values=c, use_cache=True).logits[0].float().log_softmax(-1)


res = json.load(open(OUT)) if os.path.exists(OUT) else dict(task=TASK, dir=DIR, samples=[])
done = len(res["samples"])
for i, s in enumerate(samples):
    if i < done:
        continue
    rec = dict(n_hist=len(s["h"]), m={}, **s.get("meta", {}))
    if s["kind"] == "kl":
        c = target_cache("native", s["h"]); ref = cont_logits(c, s, T); del c
        for m in METHODS:
            c = target_cache(m, s["h"])
            lp = cont_logits(c, s, S if m == "src_native" else T)
            kl = (ref.exp() * (ref - lp)).sum(-1)
            rec["m"][m] = dict(score=float(kl.mean()), kl_first16=float(kl[:16].mean()),
                               top1=float((lp.argmax(-1) == ref.argmax(-1)).float().mean()))
            del c
        res["samples"].append(rec)
        if (i + 1) % 20 == 0 or i == len(samples) - 1:
            json.dump(res, open(OUT, "w"))
            print(TASK, i + 1, "KL", {m: round(float(np.mean([x["m"][m]["score"] for x in res["samples"]])), 4) for m in METHODS}, flush=True)
        torch.cuda.empty_cache()
        continue
    for m in METHODS:
        c = target_cache(m, s["h"])
        p, ids, trunc = answer(c, s, S if m == "src_native" else T)
        r = dict(pred=p, **score(s, p))
        if ids is not None:
            r.update(n_out=len(ids), trunc=trunc)
            if s["kind"] == "gsm" and s["mx"] > 256:   # same greedy output cut at 256 tokens (truncation control)
                r["score_256"] = score(s, tok.decode(ids[:256], skip_special_tokens=True).strip())["score"]
        rec["m"][m] = r
        del c
    res["samples"].append(rec)
    if (i + 1) % 20 == 0 or i == len(samples) - 1:
        json.dump(res, open(OUT, "w"))
        print(TASK, i + 1, {m: round(float(np.mean([x["m"][m]["score"] for x in res["samples"]])), 4) for m in METHODS}, flush=True)
    torch.cuda.empty_cache()
json.dump(res, open(OUT, "w"))
print("QUALDONE")
