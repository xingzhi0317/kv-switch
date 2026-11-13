# Fused translation kernels.
#  * FastHeadLocal: CacheBridge-format head-local mapper (layer_*.safetensors with "headwise").
#    One Triton kernel: load source K/V tiles of the k selected layers -> de-RoPE in registers -> bf16 tensor-core
#    GEMM with fp32 accumulation over the k layers -> + bias -> target RoPE -> store into the target buffer.
#  * FastDense: kvbridge / NVIDIA dense mapper. De-RoPE all source K once (one fused elementwise pass), then one
#    GEMM per target layer on a gathered [T, k*H*D] view (bf16 or FP8 scaled_mm), RoPE fused into a Triton epilogue.
# Layout everywhere: source K,V [L_s, T, H, D] (SGLang canonical page layout split), output [2, L_t, T, H, D].
import math, glob, json
import torch, triton, triton.language as tl
from safetensors.torch import load_file


def rope_table(n0, n, D, theta, dev):
    inv = 1.0 / (theta ** (torch.arange(0, D, 2, device=dev, dtype=torch.float32) / D))
    ang = torch.arange(n0, n0 + n, device=dev, dtype=torch.float32)[:, None] * inv[None]
    return ang.cos().contiguous(), ang.sin().contiguous()  # [n, D/2]


def rope_at(positions, D, theta):
    inv = 1.0 / (theta ** (torch.arange(0, D, 2, device=positions.device, dtype=torch.float32) / D))
    ang = positions.float()[:, None] * inv[None]
    return ang.cos().contiguous(), ang.sin().contiguous()


@triton.jit
def _headlocal_kernel(SK, SV, OUT, WK, WV, BK, BV, SEL, COS, SIN,
                      T, LT, sl_stride, st_stride, ol_stride, ot_stride, okv_stride,
                      H: tl.constexpr, D: tl.constexpr, KSEL: tl.constexpr, BM: tl.constexpr):
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    t = tl.program_id(2)
    HALF: tl.constexpr = D // 2
    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < T
    ha = tl.arange(0, HALF)
    dd = tl.arange(0, D)
    c = tl.load(COS + offs_m[:, None] * HALF + ha[None, :], mask=mask_m[:, None], other=1.0)
    s = tl.load(SIN + offs_m[:, None] * HALF + ha[None, :], mask=mask_m[:, None], other=0.0)
    acc_a = tl.zeros((BM, HALF), dtype=tl.float32)
    acc_b = tl.zeros((BM, HALF), dtype=tl.float32)
    acc_v = tl.zeros((BM, D), dtype=tl.float32)
    wbase = (t * H + h) * (KSEL * D * D)
    for j in range(KSEL):
        src = tl.load(SEL + t * KSEL + j)
        rowk = SK + src * sl_stride + offs_m[:, None] * st_stride + h * D
        x1 = tl.load(rowk + ha[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
        x2 = tl.load(rowk + HALF + ha[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
        ca = (x1 * c + x2 * s).to(tl.bfloat16)   # de-RoPE, first half
        cb = (x2 * c - x1 * s).to(tl.bfloat16)   # de-RoPE, second half
        wj = WK + wbase + j * D * D
        w_aa = tl.load(wj + ha[:, None] * D + ha[None, :])
        w_ab = tl.load(wj + ha[:, None] * D + HALF + ha[None, :])
        w_ba = tl.load(wj + (HALF + ha[:, None]) * D + ha[None, :])
        w_bb = tl.load(wj + (HALF + ha[:, None]) * D + HALF + ha[None, :])
        acc_a = tl.dot(ca, w_aa, acc_a)
        acc_a = tl.dot(cb, w_ba, acc_a)
        acc_b = tl.dot(ca, w_ab, acc_b)
        acc_b = tl.dot(cb, w_bb, acc_b)
        xv = tl.load(SV + src * sl_stride + offs_m[:, None] * st_stride + h * D + dd[None, :], mask=mask_m[:, None], other=0.0)
        wv = tl.load(WV + wbase + j * D * D + dd[:, None] * D + dd[None, :])
        acc_v = tl.dot(xv, wv, acc_v)
    bk_a = tl.load(BK + (t * H + h) * D + ha)
    bk_b = tl.load(BK + (t * H + h) * D + HALF + ha)
    a = acc_a + bk_a[None, :]
    b = acc_b + bk_b[None, :]
    ka = a * c - b * s                          # target RoPE
    kb = b * c + a * s
    v = acc_v + tl.load(BV + (t * H + h) * D + dd)[None, :]
    orow = OUT + t * ol_stride + offs_m[:, None] * ot_stride + h * D
    tl.store(orow + ha[None, :], ka.to(tl.bfloat16), mask=mask_m[:, None])
    tl.store(orow + HALF + ha[None, :], kb.to(tl.bfloat16), mask=mask_m[:, None])
    tl.store(orow + okv_stride + dd[None, :], v.to(tl.bfloat16), mask=mask_m[:, None])


class FastHeadLocal:
    def __init__(self, mapper_dir, theta_s=1e6, theta_t=1e6, dev="cuda", BM=32, num_warps=8, num_stages=2):
        paths = sorted(glob.glob(f"{mapper_dir}/layer_*.safetensors"))
        Ls = [load_file(p) for p in paths]
        assert all("headwise" in L for L in Ls)
        self.sel = torch.stack([L["selected"] for L in Ls]).to(dev, torch.int32).contiguous()      # [LT, k]
        self.WK = torch.stack([L["k_weight"] for L in Ls]).to(dev, torch.bfloat16).contiguous()     # [LT, H, kD, D]
        self.WV = torch.stack([L["v_weight"] for L in Ls]).to(dev, torch.bfloat16).contiguous()
        self.BK = torch.stack([L["k_bias"] for L in Ls]).to(dev, torch.float32).contiguous()        # [LT, H, D]
        self.BV = torch.stack([L["v_bias"] for L in Ls]).to(dev, torch.float32).contiguous()
        self.LT, self.H, kD, self.D = self.WK.shape
        self.K = kD // self.D
        assert theta_s == theta_t, "kernel uses one RoPE table for source and target (same family)"
        self.theta, self.dev, self.BM, self.nw, self.ns = theta_s, dev, BM, num_warps, num_stages

    def __call__(self, sk, sv, pos0=0, out=None, positions=None):
        """sk, sv: [L_s, T, H, D] bf16 (RoPE-applied K). Returns/fills out [2, L_t, T, H, D] bf16.
        positions: optional per-token positions [T] (several requests concatenated along T)."""
        Ls, T, H, D = sk.shape
        if out is None:
            out = torch.empty(2, self.LT, T, H, D, device=sk.device, dtype=torch.bfloat16)
        if positions is None:
            cos, sin = rope_table(pos0, T, D, self.theta, sk.device)
        else:
            cos, sin = rope_at(positions, D, self.theta)
        grid = (triton.cdiv(T, self.BM), H, self.LT)
        _headlocal_kernel[grid](sk, sv, out, self.WK, self.WV, self.BK, self.BV, self.sel, cos, sin,
                                T, self.LT, sk.stride(0), sk.stride(1), out.stride(1), out.stride(2), out.stride(0),
                                H=H, D=D, KSEL=self.K, BM=self.BM, num_warps=self.nw, num_stages=self.ns)
        return out


# ------------------------------------------------------------------------------------------------ dense
@triton.jit
def _derope_kernel(SK, OUT, COS, SIN, T, total_rows, H: tl.constexpr, D: tl.constexpr, BR: tl.constexpr):
    # rows = (layer, token, head) flattened; each row is one head vector of length D
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    m = r < total_rows
    HALF: tl.constexpr = D // 2
    ha = tl.arange(0, HALF)
    tok = (r // H) % T
    x1 = tl.load(SK + r[:, None] * D + ha[None, :], mask=m[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(SK + r[:, None] * D + HALF + ha[None, :], mask=m[:, None], other=0.0).to(tl.float32)
    c = tl.load(COS + tok[:, None] * HALF + ha[None, :], mask=m[:, None], other=1.0)
    s = tl.load(SIN + tok[:, None] * HALF + ha[None, :], mask=m[:, None], other=0.0)
    tl.store(OUT + r[:, None] * D + ha[None, :], (x1 * c + x2 * s).to(OUT.dtype.element_ty), mask=m[:, None])
    tl.store(OUT + r[:, None] * D + HALF + ha[None, :], (x2 * c - x1 * s).to(OUT.dtype.element_ty), mask=m[:, None])


@triton.jit
def _rope_store_kernel(Y, OUT, COS, SIN, T, ytok_stride, ot_stride, H: tl.constexpr, D: tl.constexpr, BT: tl.constexpr):
    # Y [T, H*D] fp32/bf16 (mapped content keys) -> OUT [T, H, D] bf16 with target RoPE
    pid = tl.program_id(0)
    h = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    tok = pid * BT + tl.arange(0, BT)
    m = tok < T
    ha = tl.arange(0, HALF)
    a = tl.load(Y + tok[:, None] * ytok_stride + h * D + ha[None, :], mask=m[:, None], other=0.0).to(tl.float32)
    b = tl.load(Y + tok[:, None] * ytok_stride + h * D + HALF + ha[None, :], mask=m[:, None], other=0.0).to(tl.float32)
    c = tl.load(COS + tok[:, None] * HALF + ha[None, :], mask=m[:, None], other=1.0)
    s = tl.load(SIN + tok[:, None] * HALF + ha[None, :], mask=m[:, None], other=0.0)
    o = OUT + tok[:, None] * ot_stride + h * D
    tl.store(o + ha[None, :], (a * c - b * s).to(tl.bfloat16), mask=m[:, None])
    tl.store(o + HALF + ha[None, :], (b * c + a * s).to(tl.bfloat16), mask=m[:, None])


class FastDense:
    """Dense (kvbridge / NVIDIA CFM) mapper. fp8=True uses per-tensor-scaled FP8 E4M3 GEMMs (Ada tensor cores)."""

    def __init__(self, kb_mapper, theta=1e6, dev="cuda", fp8=False):
        self.sel = [list(map(int, e["source_layers"])) for e in kb_mapper.layers]
        self.H, self.D = kb_mapper.n_kv_heads, kb_mapper.head_dim
        self.theta, self.dev, self.fp8 = theta, dev, fp8
        self.W, self.b = [], []
        for e in kb_mapper.layers:
            Wk, Wv = e["W_k"].to(dev).float(), e["W_v"].to(dev).float()
            bk = e["mu_yk"].to(dev).float() - e["mu_xk"].to(dev).float() @ Wk
            bv = e["mu_yv"].to(dev).float() - e["mu_xv"].to(dev).float() @ Wv
            if fp8:
                sk_, sv_ = Wk.abs().max() / 448.0, Wv.abs().max() / 448.0
                # scaled_mm wants B column-major: store W^T contiguous and pass .t()
                self.W.append(((Wk / sk_).to(torch.float8_e4m3fn).t().contiguous(), sk_.reshape(()),
                               (Wv / sv_).to(torch.float8_e4m3fn).t().contiguous(), sv_.reshape(())))
            else:
                self.W.append((Wk.to(torch.bfloat16), None, Wv.to(torch.bfloat16), None))
            self.b.append((bk, bv))
        self.LT = len(self.W)

    def _mm(self, x, w, ws, bias, xs=None):
        if not self.fp8:
            return torch.addmm(bias.to(torch.bfloat16), x, w)
        return torch._scaled_mm(x, w.t(), scale_a=xs, scale_b=ws, bias=bias.to(torch.bfloat16), out_dtype=torch.bfloat16)

    def __call__(self, sk, sv, pos0=0, out=None):
        Ls, T, H, D = sk.shape
        if out is None:
            out = torch.empty(2, self.LT, T, H, D, device=sk.device, dtype=torch.bfloat16)
        cos, sin = rope_table(pos0, T, D, self.theta, sk.device)
        kc = torch.empty_like(sk)
        rows = Ls * T * H
        _derope_kernel[(triton.cdiv(rows, 256),)](sk, kc, cos, sin, T, rows, H=H, D=D, BR=256, num_warps=4)
        kc = kc.view(Ls, T, H * D)
        vv = sv.view(Ls, T, H * D)
        if self.fp8:  # per-tensor activation scales, computed once for all source layers
            ksc, vsc = kc.abs().amax() / 448.0, vv.abs().amax() / 448.0
            kc8, vv8 = (kc / ksc).to(torch.float8_e4m3fn), (vv / vsc).to(torch.float8_e4m3fn)
        for t in range(self.LT):
            idx = self.sel[t]
            if self.fp8:
                xk = kc8[idx].permute(1, 0, 2).reshape(T, -1)   # gather selected layers -> [T, k*H*D]
                xv = vv8[idx].permute(1, 0, 2).reshape(T, -1)
                yk = self._mm(xk, *self.W[t][0:2], self.b[t][0], ksc.float().reshape(()))
                yv = self._mm(xv, *self.W[t][2:4], self.b[t][1], vsc.float().reshape(()))
            else:
                xk = kc[idx].permute(1, 0, 2).reshape(T, -1)
                xv = vv[idx].permute(1, 0, 2).reshape(T, -1)
                yk = self._mm(xk, self.W[t][0], None, self.b[t][0])
                yv = self._mm(xv, self.W[t][2], None, self.b[t][1])
            _rope_store_kernel[(triton.cdiv(T, 64), H)](yk, out[0, t], cos, sin, T, yk.stride(0), out.stride(2),
                                                        H=H, D=D, BT=64, num_warps=4)
            out[1, t].view(T, H * D).copy_(yv)
        return out


# ------------------------------------------------------------------------------------------------ FP32 head-local
# Same numeric types as the CacheBridge official runtime: BF16 cache in/out, FP32 weights, FP32 math.
# PREC: "ieee" = true FP32 FMA, "tf32x3" = 3-pass TF32 (FP32-level accuracy on tensor cores).
@triton.jit
def _headlocal_f32_kernel(SK, SV, OUT, WK, WV, BK, BV, SEL, COS, SIN,
                          T, LT, sl_stride, st_stride, ol_stride, ot_stride, okv_stride,
                          H: tl.constexpr, D: tl.constexpr, KSEL: tl.constexpr, BM: tl.constexpr, PREC: tl.constexpr):
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    t = tl.program_id(2)
    HALF: tl.constexpr = D // 2
    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < T
    ha = tl.arange(0, HALF)
    c = tl.load(COS + offs_m[:, None] * HALF + ha[None, :], mask=mask_m[:, None], other=1.0)
    s = tl.load(SIN + offs_m[:, None] * HALF + ha[None, :], mask=mask_m[:, None], other=0.0)
    acc_a = tl.zeros((BM, HALF), dtype=tl.float32)
    acc_b = tl.zeros((BM, HALF), dtype=tl.float32)
    acc_va = tl.zeros((BM, HALF), dtype=tl.float32)
    acc_vb = tl.zeros((BM, HALF), dtype=tl.float32)
    wbase = (t * H + h) * (KSEL * D * D)
    for j in range(KSEL):
        src = tl.load(SEL + t * KSEL + j)
        rowk = SK + src * sl_stride + offs_m[:, None] * st_stride + h * D
        x1 = tl.load(rowk + ha[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
        x2 = tl.load(rowk + HALF + ha[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
        ca = x1 * c + x2 * s
        cb = x2 * c - x1 * s
        wj = WK + wbase + j * D * D
        acc_a = tl.dot(ca, tl.load(wj + ha[:, None] * D + ha[None, :]), acc_a, input_precision=PREC)
        acc_a = tl.dot(cb, tl.load(wj + (HALF + ha[:, None]) * D + ha[None, :]), acc_a, input_precision=PREC)
        acc_b = tl.dot(ca, tl.load(wj + ha[:, None] * D + HALF + ha[None, :]), acc_b, input_precision=PREC)
        acc_b = tl.dot(cb, tl.load(wj + (HALF + ha[:, None]) * D + HALF + ha[None, :]), acc_b, input_precision=PREC)
        rowv = SV + src * sl_stride + offs_m[:, None] * st_stride + h * D
        v1 = tl.load(rowv + ha[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
        v2 = tl.load(rowv + HALF + ha[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
        wv = WV + wbase + j * D * D
        acc_va = tl.dot(v1, tl.load(wv + ha[:, None] * D + ha[None, :]), acc_va, input_precision=PREC)
        acc_va = tl.dot(v2, tl.load(wv + (HALF + ha[:, None]) * D + ha[None, :]), acc_va, input_precision=PREC)
        acc_vb = tl.dot(v1, tl.load(wv + ha[:, None] * D + HALF + ha[None, :]), acc_vb, input_precision=PREC)
        acc_vb = tl.dot(v2, tl.load(wv + (HALF + ha[:, None]) * D + HALF + ha[None, :]), acc_vb, input_precision=PREC)
    a = acc_a + tl.load(BK + (t * H + h) * D + ha)[None, :]
    b = acc_b + tl.load(BK + (t * H + h) * D + HALF + ha)[None, :]
    va = acc_va + tl.load(BV + (t * H + h) * D + ha)[None, :]
    vb = acc_vb + tl.load(BV + (t * H + h) * D + HALF + ha)[None, :]
    orow = OUT + t * ol_stride + offs_m[:, None] * ot_stride + h * D
    tl.store(orow + ha[None, :], (a * c - b * s).to(tl.bfloat16), mask=mask_m[:, None])
    tl.store(orow + HALF + ha[None, :], (b * c + a * s).to(tl.bfloat16), mask=mask_m[:, None])
    tl.store(orow + okv_stride + ha[None, :], va.to(tl.bfloat16), mask=mask_m[:, None])
    tl.store(orow + okv_stride + HALF + ha[None, :], vb.to(tl.bfloat16), mask=mask_m[:, None])


class FastHeadLocalF32(FastHeadLocal):
    """Head-local translation with the CacheBridge numeric types (FP32 weights and math, BF16 cache in/out)."""

    def __init__(self, mapper_dir, prec="ieee", BM=32, num_warps=4, num_stages=1, **kw):
        super().__init__(mapper_dir, BM=BM, num_warps=num_warps, num_stages=num_stages, **kw)
        paths = sorted(glob.glob(f"{mapper_dir}/layer_*.safetensors"))
        Ls = [load_file(p) for p in paths]
        self.WK = torch.stack([L["k_weight"] for L in Ls]).to(self.dev, torch.float32).contiguous()
        self.WV = torch.stack([L["v_weight"] for L in Ls]).to(self.dev, torch.float32).contiguous()
        self.prec = prec

    def __call__(self, sk, sv, pos0=0, out=None, positions=None):
        Ls, T, H, D = sk.shape
        if out is None:
            out = torch.empty(2, self.LT, T, H, D, device=sk.device, dtype=torch.bfloat16)
        cos, sin = rope_table(pos0, T, D, self.theta, sk.device) if positions is None else rope_at(positions, D, self.theta)
        grid = (triton.cdiv(T, self.BM), H, self.LT)
        _headlocal_f32_kernel[grid](sk, sv, out, self.WK, self.WV, self.BK, self.BV, self.sel, cos, sin,
                                    T, self.LT, sk.stride(0), sk.stride(1), out.stride(1), out.stride(2), out.stride(0),
                                    H=H, D=D, KSEL=self.K, BM=self.BM, PREC=self.prec, num_warps=self.nw, num_stages=self.ns)
        return out


class CublasHeadLocalF32:
    """True-FP32 head-local translation (CacheBridge numerics) using cuBLAS FP32 batched GEMMs (TF32 off),
    with de-RoPE done once per source layer and no per-page / per-layer concat of the 8 selected layers
    (8 accumulated bmm's instead of one GEMM on a materialized concat). Chunked over tokens to bound memory."""

    def __init__(self, mapper_dir, theta=1e6, dev="cuda", chunk=8192):
        paths = sorted(glob.glob(f"{mapper_dir}/layer_*.safetensors"))
        Ls = [load_file(p) for p in paths]
        self.sel = [L["selected"].tolist() for L in Ls]
        self.WK = torch.stack([L["k_weight"] for L in Ls]).to(dev, torch.float32).contiguous()  # [LT,H,kD,D]
        self.WV = torch.stack([L["v_weight"] for L in Ls]).to(dev, torch.float32).contiguous()
        self.BK = torch.stack([L["k_bias"] for L in Ls]).to(dev, torch.float32)[:, :, None]   # [LT,H,1,D]
        self.BV = torch.stack([L["v_bias"] for L in Ls]).to(dev, torch.float32)[:, :, None]
        self.LT, self.H, kD, self.D = self.WK.shape
        self.K = kD // self.D
        self.theta, self.dev, self.chunk = theta, dev, chunk
        self.needed = sorted({s for ss in self.sel for s in ss})

    @torch.no_grad()
    def __call__(self, sk, sv, pos0=0, out=None):
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        Ls, T, H, D = sk.shape
        if out is None:
            out = torch.empty(2, self.LT, T, H, D, device=sk.device, dtype=torch.bfloat16)
        half = D // 2
        for c0 in range(0, T, self.chunk):
            c1 = min(T, c0 + self.chunk)
            n = c1 - c0
            cos, sin = rope_table(pos0 + c0, n, D, self.theta, sk.device)
            cos2, sin2 = torch.cat([cos, cos], -1)[None, :, None], torch.cat([sin, sin], -1)[None, :, None]
            idx = torch.tensor(self.needed, device=sk.device)
            k = sk[idx, c0:c1].float()                                   # [Ln, n, H, D]
            rh = torch.cat([-k[..., half:], k[..., :half]], -1)
            kc = (k * cos2 - rh * sin2).permute(0, 2, 1, 3).contiguous()  # content keys [Ln, H, n, D]
            vv = sv[idx, c0:c1].float().permute(0, 2, 1, 3).contiguous()
            pos = {s: i for i, s in enumerate(self.needed)}
            del k, rh
            for t in range(self.LT):
                ak = self.BK[t].expand(H, n, D).clone()
                av = self.BV[t].expand(H, n, D).clone()
                for j, s in enumerate(self.sel[t]):
                    ak.baddbmm_(kc[pos[s]], self.WK[t, :, j * D:(j + 1) * D])
                    av.baddbmm_(vv[pos[s]], self.WV[t, :, j * D:(j + 1) * D])
                rk = torch.cat([-ak[..., half:], ak[..., :half]], -1)
                kr = ak * cos2[0].transpose(0, 1) + rk * sin2[0].transpose(0, 1)   # target RoPE, [H, n, D]
                out[0, t, c0:c1] = kr.transpose(0, 1).to(torch.bfloat16)
                out[1, t, c0:c1] = av.transpose(0, 1).to(torch.bfloat16)
            del kc, vv
        torch.backends.cuda.matmul.allow_tf32 = prev
        return out
