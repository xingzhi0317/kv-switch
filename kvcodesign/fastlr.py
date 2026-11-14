# Fused translation for the shared low-rank latent mapper (reduced-rank ridge, screen2.py).
#   y = (x - mu_x) P D + mu_y, K and V with separate latents (ranks rk, rv).
# Two Triton kernels (the step is compute-bound, so the latent goes through HBM: ~4.6 KB/token vs ~270 KB/token
# of cache traffic and ~150 M MACs/token; keeping it on chip would force re-reading the 300 MB of weights per tile):
#   encode: read source K/V directly from the source cache [Ls, T, H, D] -> de-RoPE in registers (K) ->
#           GEMM with P over all (layer, head) units -> + folded bias -> latent z [T, r] (bf16, or FP8 + static scale)
#   decode: z @ D, one target (layer, head) = 128 output columns per tile -> + mu_y -> target RoPE in registers (K) ->
#           store straight into the target cache layout out[kv, layer, T, H, D]
# Every source element is read once; no intermediate content-space copy, no separate RoPE pass, no layout copy.
# FP8 (E4M3, Ada tensor cores): source activations use a static per-source-layer scale folded into P's rows,
#   z uses a static scale folded into D's rows, weights have per-output-column scales applied in the epilogue.
import os, torch, triton, triton.language as tl

LS, LT, H, D = 28, 36, 8, 128


def rope_table(positions, theta=1e6):
    inv = 1.0 / (theta ** (torch.arange(0, D, 2, device=positions.device, dtype=torch.float32) / D))
    ang = positions.float()[:, None] * inv[None]
    return ang.cos().contiguous(), ang.sin().contiguous()  # [T, D/2]


@triton.jit
def _encode(SX, P, ZB, WC, XS, Z, COS, SIN, T, R, sl_stride, st_stride, z_stride,
            NU: tl.constexpr, H: tl.constexpr, D: tl.constexpr, ROPE: tl.constexpr, FP8: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, ZFP8: tl.constexpr):
    NT = tl.cdiv(R, BN)                      # 1-D grid, N tiles fastest: programs of one token block share x in L2
    pid = tl.program_id(0)
    pid_m = pid // NT
    pid_n = pid % NT
    HALF: tl.constexpr = D // 2
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mm = offs_m < T
    mn = offs_n < R
    ha = tl.arange(0, HALF)
    if ROPE:
        c = tl.load(COS + offs_m[:, None] * HALF + ha[None, :], mask=mm[:, None], other=1.0)
        s = tl.load(SIN + offs_m[:, None] * HALF + ha[None, :], mask=mm[:, None], other=0.0)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for u in range(NU):                      # (source layer, head) units, 128 input dims each
        layer = u // H
        h = u % H
        row = SX + layer * sl_stride + offs_m[:, None] * st_stride + h * D
        x1 = tl.load(row + ha[None, :], mask=mm[:, None], other=0.0)
        x2 = tl.load(row + HALF + ha[None, :], mask=mm[:, None], other=0.0)
        if ROPE:
            x1f = x1.to(tl.float32); x2f = x2.to(tl.float32)
            a = x1f * c + x2f * s            # de-RoPE (rotate_half convention)
            b = x2f * c - x1f * s
        else:
            a = x1; b = x2
        pa = tl.load(P + (u * D + ha)[:, None] * R + offs_n[None, :], mask=mn[None, :], other=0.0)
        pb = tl.load(P + (u * D + HALF + ha)[:, None] * R + offs_n[None, :], mask=mn[None, :], other=0.0)
        if FP8:
            inv = tl.load(XS + layer)        # 1 / per-layer static scale (folded into P on the host)
            a = tl.clamp(a.to(tl.float32) * inv, -448.0, 448.0).to(tl.float8e4nv)
            b = tl.clamp(b.to(tl.float32) * inv, -448.0, 448.0).to(tl.float8e4nv)
        else:
            a = a.to(tl.bfloat16); b = b.to(tl.bfloat16)
        acc = tl.dot(a, pa, acc)
        acc = tl.dot(b, pb, acc)
    if FP8:
        acc = acc * tl.load(WC + offs_n, mask=mn, other=0.0)[None, :]
    z = acc + tl.load(ZB + offs_n, mask=mn, other=0.0)[None, :]
    zp = Z + offs_m[:, None] * z_stride + offs_n[None, :]
    if ZFP8:
        zs = tl.load(XS + NU // H)            # 1 / static latent scale
        tl.store(zp, tl.clamp(z * zs, -448.0, 448.0).to(tl.float8e4nv), mask=mm[:, None] & mn[None, :])
    else:
        tl.store(zp, z.to(tl.bfloat16), mask=mm[:, None] & mn[None, :])


@triton.jit
def _decode(Z, DW, YB, WC, OUT, COS, SIN, T, R, z_stride, ol_stride, ot_stride,
            H: tl.constexpr, D: tl.constexpr, ROPE: tl.constexpr, FP8: tl.constexpr,
            BM: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr, NU: tl.constexpr):
    # grouped ordering: GROUP_M token blocks x successive target heads, so z tiles and D tiles both get L2 reuse
    pid = tl.program_id(0)
    num_m = tl.cdiv(T, BM)
    per_group = GROUP_M * NU
    gid = pid // per_group
    first_m = gid * GROUP_M
    gsz = tl.minimum(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % per_group) % gsz
    u = (pid % per_group) // gsz             # target (layer, head) unit = 128 output columns
    t = u // H
    h = u % H
    HALF: tl.constexpr = D // 2
    NQ = NU * D
    offs_m = pid_m * BM + tl.arange(0, BM)
    mm = offs_m < T
    ha = tl.arange(0, HALF)
    kk = tl.arange(0, BK)
    acc_a = tl.zeros((BM, HALF), dtype=tl.float32)
    acc_b = tl.zeros((BM, HALF), dtype=tl.float32)
    for k0 in range(0, R, BK):
        z = tl.load(Z + offs_m[:, None] * z_stride + (k0 + kk)[None, :], mask=mm[:, None], other=0.0)
        wa = tl.load(DW + (k0 + kk)[:, None] * NQ + (u * D + ha)[None, :])
        wb = tl.load(DW + (k0 + kk)[:, None] * NQ + (u * D + HALF + ha)[None, :])
        acc_a = tl.dot(z, wa, acc_a)
        acc_b = tl.dot(z, wb, acc_b)
    if FP8:
        acc_a = acc_a * tl.load(WC + u * D + ha)[None, :]
        acc_b = acc_b * tl.load(WC + u * D + HALF + ha)[None, :]
    a = acc_a + tl.load(YB + u * D + ha)[None, :]
    b = acc_b + tl.load(YB + u * D + HALF + ha)[None, :]
    if ROPE:
        c = tl.load(COS + offs_m[:, None] * HALF + ha[None, :], mask=mm[:, None], other=1.0)
        s = tl.load(SIN + offs_m[:, None] * HALF + ha[None, :], mask=mm[:, None], other=0.0)
        a, b = a * c - b * s, b * c + a * s  # target RoPE
    o = OUT + t * ol_stride + offs_m[:, None] * ot_stride + h * D
    tl.store(o + ha[None, :], a.to(tl.bfloat16), mask=mm[:, None])
    tl.store(o + HALF + ha[None, :], b.to(tl.bfloat16), mask=mm[:, None])


CFG_BF16 = {"k": ((64, 256, 8, 2), (128, 32, 4, 4), 8), "v": ((128, 128, 4, 2), (128, 32, 4, 4), 32)}
CFG_FP8 = {"k": ((128, 256, 8, 1), (128, 32, 4, 4), 32), "v": ((128, 256, 8, 1), (256, 32, 8, 3), 16)}


def _q8_cols(W):  # per-output-column FP8 E4M3 weights; returns (fp8 weights, fp32 column scales)
    sc = W.abs().amax(0).clamp_min(1e-12) / 448.0
    return (W / sc).to(torch.float8_e4m3fn).contiguous(), sc.float().contiguous()


class FastLowRank:
    """Fused low-rank translator. mapper files: ~/kvcodesign/mappers/rrr_var_r{r}.pt (Pk, Dk, Pv, Dv, mx, my).
    fp8=True needs static activation scales: scales = dict(k_layer=[LS], v_layer=[LS], zk=float, zv=float) (amax)."""

    def __init__(self, rk=768, rv=1536, fp8=False, scales=None, dev="cuda", theta=1e6,
                 cfg=None, mapper_dir=os.path.expanduser("~/kvcodesign/mappers")):
        a = torch.load(f"{mapper_dir}/rrr_var_r{rk}.pt"); b = torch.load(f"{mapper_dir}/rrr_var_r{rv}.pt")
        PK, QK = LS * H * D, LT * H * D
        self.rk, self.rv, self.fp8, self.dev, self.theta = rk, rv, fp8, dev, theta
        # per kind: (encode BM, BN, warps, stages), (decode BM, BK, warps, stages), GROUP_M -- tuned at 32K (prof_fastlr.py)
        self.cfg = cfg or (CFG_FP8 if fp8 else CFG_BF16)
        self.parts = {}
        for kind, src, r in (("k", a, rk), ("v", b, rv)):
            Pm, Dm = src[f"P{kind}"].double(), src[f"D{kind}"].double()
            mx = src["mx"][:PK] if kind == "k" else src["mx"][PK:]
            my = src["my"][:QK] if kind == "k" else src["my"][QK:]
            zb = -(mx.double() @ Pm)                                  # fold input centering into the latent bias
            p = dict(r=r, zb=zb.float().to(dev), yb=my.float().to(dev))
            if not fp8:
                p.update(P=Pm.to(dev, torch.bfloat16).contiguous(), D=Dm.to(dev, torch.bfloat16).contiguous(),
                         wce=torch.ones(1, device=dev), wcd=torch.ones(1, device=dev),
                         xs=torch.ones(LS + 1, device=dev))
            else:
                sl = torch.as_tensor(scales[f"{kind}_layer"], dtype=torch.float64) / 448.0      # per source layer
                sz = float(scales[f"z{kind}"]) / 448.0
                P8, wce = _q8_cols(Pm * sl.repeat_interleave(H * D)[:, None])
                D8, wcd = _q8_cols(Dm * sz)
                p.update(P=P8.to(dev), D=D8.to(dev), wce=wce.to(dev), wcd=wcd.to(dev),
                         xs=torch.cat([1.0 / sl, torch.tensor([1.0 / sz], dtype=torch.float64)]).float().to(dev))
            self.parts[kind] = p
        self._z = {}

    @torch.no_grad()
    def __call__(self, sk, sv, positions=None, out=None):
        """sk, sv: source cache [LS, T, H, D] bf16 (K with source RoPE). positions: [T] (requests may be concatenated).
        returns out [2, LT, T, H, D] bf16 (K with target RoPE)."""
        T = sk.shape[1]
        if positions is None:
            positions = torch.arange(T, device=sk.device)
        cos, sin = rope_table(positions, self.theta)
        if out is None:
            out = torch.empty(2, LT, T, H, D, device=sk.device, dtype=torch.bfloat16)
        zdt = torch.float8_e4m3fn if self.fp8 else torch.bfloat16
        for i, (kind, x) in enumerate((("k", sk), ("v", sv))):
            p = self.parts[kind]
            r = p["r"]
            (BM, BN, nwe, nse), (BMd, BK, nwd, nsd), gm = self.cfg[kind]
            key = (kind, T)
            if key not in self._z:
                self._z[key] = torch.empty(T, r, device=sk.device, dtype=zdt)
            z = self._z[key]
            _encode[(triton.cdiv(T, BM) * triton.cdiv(r, BN),)](
                x, p["P"], p["zb"], p["wce"], p["xs"], z, cos, sin, T, r, x.stride(0), x.stride(1), z.stride(0),
                NU=LS * H, H=H, D=D, ROPE=(kind == "k"), FP8=self.fp8, BM=BM, BN=BN, ZFP8=self.fp8,
                num_warps=nwe, num_stages=nse)
            _decode[(triton.cdiv(T, BMd) * LT * H,)](
                z, p["D"], p["yb"], p["wcd"], out[i], cos, sin, T, r, z.stride(0), out.stride(1), out.stride(2),
                H=H, D=D, ROPE=(kind == "k"), FP8=self.fp8, BM=BMd, BK=BK, GROUP_M=gm, NU=LT * H,
                num_warps=nwd, num_stages=nsd)
        return out


class TorchLowRank:
    """Unfused reference: de-RoPE pass -> cuBLAS GEMM -> cuBLAS GEMM -> RoPE pass -> layout copy (bf16 or fp32)."""

    def __init__(self, rk=768, rv=1536, dtype=torch.bfloat16, dev="cuda", theta=1e6,
                 mapper_dir=os.path.expanduser("~/kvcodesign/mappers")):
        a = torch.load(f"{mapper_dir}/rrr_var_r{rk}.pt"); b = torch.load(f"{mapper_dir}/rrr_var_r{rv}.pt")
        PK, QK = LS * H * D, LT * H * D
        g = lambda t: t.to(dev, dtype).contiguous()
        self.Pk, self.Dk, self.Pv, self.Dv = g(a["Pk"]), g(a["Dk"]), g(b["Pv"]), g(b["Dv"])
        self.zbk = g(-(a["mx"][:PK].double() @ a["Pk"].double()).float()); self.zbv = g(-(b["mx"][PK:].double() @ b["Pv"].double()).float())
        self.ybk, self.ybv = g(a["my"][:QK]), g(b["my"][QK:])
        self.dt, self.theta = dtype, theta

    @torch.no_grad()
    def __call__(self, sk, sv, positions=None, out=None, chunk=8192):
        T = sk.shape[1]
        if positions is None:
            positions = torch.arange(T, device=sk.device)
        if out is None:
            out = torch.empty(2, LT, T, H, D, device=sk.device, dtype=torch.bfloat16)
        for p in range(0, T, chunk):
            q = min(T, p + chunk)
            self._run(sk[:, p:q], sv[:, p:q], positions[p:q], out[:, :, p:q])
        return out

    def _run(self, sk, sv, positions, out):
        T = sk.shape[1]
        c, s = rope_table(positions, self.theta)
        c, s = torch.cat([c, c], -1)[None, :, None], torch.cat([s, s], -1)[None, :, None]
        rot = lambda x: torch.cat([-x[..., D // 2:], x[..., :D // 2]], -1)
        k = sk.to(self.dt)
        kc = (k * c - rot(k) * s).to(self.dt)
        xk = kc.permute(1, 0, 2, 3).reshape(T, -1)
        xv = sv.to(self.dt).permute(1, 0, 2, 3).reshape(T, -1)
        yk = torch.addmm(self.ybk, torch.addmm(self.zbk, xk, self.Pk), self.Dk).view(T, LT, H, D).permute(1, 0, 2, 3)
        yv = torch.addmm(self.ybv, torch.addmm(self.zbv, xv, self.Pv), self.Dv).view(T, LT, H, D).permute(1, 0, 2, 3)
        yk = yk * c + rot(yk) * s
        out[0].copy_(yk); out[1].copy_(yv)
