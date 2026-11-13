# Runtime KV translation A-cache -> B-cache (Case 1, same tokenizer).
# K: de-RoPE(A) -> affine map over k source layers -> RoPE(B).  V: affine map.  Sink (pos 0) = B's own KV.
import torch
from ks import *


class Translator:
    def __init__(self, A, B, maps, k=3, vkind="WVa", lowrank=None, name=""):
        """maps: output of fit.py (dict k -> WK/WV/WVa [LB,F,Fo], src).  lowrank: optional dict with
        UK,VK,UV,VV ([LB,F,r],[LB,r,Fo]) residuals added to the maps (C4 / KL-trained)."""
        self.A, self.B, self.name = A, B, name
        m = maps[k]
        WK, WV = m["WK"].float(), m[vkind].float()
        if lowrank is not None:
            WK = WK + lowrank["UK"].float() @ lowrank["VK"].float()
            WV = WV + lowrank["UV"].float() @ lowrank["VV"].float()
        self.WK = WK.to(torch.bfloat16).to(A.dev)
        self.WV = WV.to(torch.bfloat16).to(A.dev)
        self.src = m["src"]
        # last target layer that reads each source layer (for C5 free-as-you-go)
        self.last_use = {}
        for l, ss in enumerate(self.src):
            for s in ss:
                self.last_use[s] = l
        self.CHUNK = 8192

    def _feats(self, kvA, ss, n0, n1, k_or_v):
        A = self.A
        if k_or_v == "k":
            cos, sin = A.cossin(n0, n1)
            xs = [derope(kvA[s][:, n0:n1].float(), cos, sin).to(torch.bfloat16) for s in ss]
        else:
            xs = [kvA[s][:, n0:n1] for s in ss]
        N = n1 - n0
        return torch.cat([x.transpose(0, 1).reshape(N, -1) for x in xs], -1)

    @torch.no_grad()
    def layer(self, l, cA, cB, n0, n1):
        B = self.B
        for c0 in range(n0, n1, self.CHUNK):
            c1 = min(n1, c0 + self.CHUNK)
            N = c1 - c0
            W = self.WK[l]
            yk = (self._feats(cA.k, self.src[l], c0, c1, "k") @ W[:-1] + W[-1]).view(N, B.Hk, B.D).transpose(0, 1)
            cos, sin = B.cossin(c0, c1)
            cB.k[l][:, c0:c1] = rope(yk.float(), cos, sin).to(torch.bfloat16)
            W = self.WV[l]
            yv = (self._feats(cA.v, self.src[l], c0, c1, "v") @ W[:-1] + W[-1]).view(N, B.Hk, B.D).transpose(0, 1)
            cB.v[l][:, c0:c1] = yv

    @torch.no_grad()
    def translate(self, cA, cB, ids, n0, n1, sink=1, free_src=False, on_layer=None):
        """Translate positions [n0,n1). If n0 < sink, positions < sink get B's own exact KV.
        free_src: release A's layer as soon as no later target layer needs it (C5).
        on_layer(l): callback after target layer l is written (C6 event hook)."""
        B = self.B
        do_sink = n0 < sink
        if do_sink:  # B's own KV for the first token(s): depends only on those tokens -> cached per token id
            key = tuple(ids[0, :sink].tolist())
            if getattr(self, "_sink_key", None) != key:
                tmp = B.new_cache(sink)
                B.forward(ids[:, :sink], tmp, n0=0, want="hidden")
                self._sink = ([t[:, :sink].clone() for t in tmp.k], [t[:, :sink].clone() for t in tmp.v])
                self._sink_key = key
                del tmp
            n0 = sink
        for l in range(B.L):
            if cB.k[l] is None:
                cB.alloc(l)
            if do_sink:
                cB.k[l][:, :sink] = self._sink[0][l]
                cB.v[l][:, :sink] = self._sink[1][l]
            if n1 > n0:
                self.layer(l, cA, cB, n0, n1)
            if on_layer is not None:
                on_layer(l)
            if free_src:
                for s in self.src[l]:
                    if self.last_use[s] == l and cA.k[s] is not None:
                        cA.free_layer(s)
        cB.n = max(cB.n, n1)


@torch.no_grad()
def selective_recompute(B, cB, ids, N, frac=0.3, sel=None):
    """RaReCache/CacheBlend-style: B recomputes `frac` of the translated prefix tokens.
    Selection: largest deviation between translated and exact layer-0 KV (exact layer-0 KV is cheap: it
    depends only on token embeddings); layer-0 cache is replaced by the exact one for all tokens."""
    x = B.layers[0].input_layernorm(B.m.model.embed_tokens(ids[:, :N]))
    q, k, v = B.qkv(0, x)
    cos, sin = B.cossin(0, N)
    kp = rope(k.float(), cos, sin)[0]
    dev_ = ((kp - cB.k[0][:, :N].float()).norm(dim=-1) / kp.norm(dim=-1).clamp_min(1e-6)).mean(0) + \
           ((v[0].float() - cB.v[0][:, :N].float()).norm(dim=-1) / v[0].float().norm(dim=-1).clamp_min(1e-6)).mean(0)
    cB.k[0][:, :N] = kp.to(torch.bfloat16)
    cB.v[0][:, :N] = v[0]
    nsel = max(1, int(frac * N))
    dev_[0] = -1  # sink already exact
    pos = dev_.topk(nsel).indices.sort().values if sel is None else sel
    forward_at(B, ids[:, pos], pos, cB, N)
    return pos


@torch.no_grad()
def forward_at(M, ids, pos, cache, N, chunk=4096):
    """Forward tokens at arbitrary (sorted) positions `pos` against a cache of length N; writes their KV."""
    for s in range(0, pos.numel(), chunk):
        p = pos[s:s + chunk]
        _forward_at(M, ids[:, s:s + chunk], p, cache, N)


def _forward_at(M, ids, pos, cache, N):
    import torch.nn.functional as F
    cos, sin = M.cossin(0, N)
    cos, sin = cos[pos], sin[pos]
    T = pos.numel()
    mask = torch.arange(N, device=M.dev)[None, :] <= pos[:, None]
    h = M.m.model.embed_tokens(ids)
    G = M.Hq // M.Hk
    for l in range(M.L):
        layer = M.layers[l]
        x = layer.input_layernorm(h)
        q, k, v = M.qkv(l, x)
        qp = rope(q.float(), cos, sin).to(q.dtype)
        cache.k[l][:, pos] = rope(k.float(), cos, sin).to(k.dtype)[0]
        cache.v[l][:, pos] = v[0]
        K, V = cache.k[l][None, :, :N], cache.v[l][None, :, :N]
        o = F.scaled_dot_product_attention(qp, K.repeat_interleave(G, 1), V.repeat_interleave(G, 1), attn_mask=mask)
        h = h + layer.self_attn.o_proj(o.transpose(1, 2).reshape(1, T, -1))
        h = h + layer.mlp(layer.post_attention_layernorm(h))
