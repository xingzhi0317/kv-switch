# RaReCache (arXiv 2610.11358), re-implemented from the paper (no official code found).  Our code.
#   1. map the source cache with a closed-form per-layer ridge map (here: the kvbridge mapper, official code/weights;
#      the paper fits its own map with 8 source layers per target layer for Qwen3 -- kvbridge uses 12);
#   2. score every token by rank disagreement: ||z||^2 - ||z V_r||^2 summed over target layers and K/V, where
#      z = centred prediction (mapped entry minus the map's output mean, keys in content space) and V_r = top-128
#      eigenvectors of the covariance of the map's predictions on calibration data (rare_fit.py);
#   3. the target recomputes the top ceil(rho * T) positions: one forward over those tokens (in position order,
#      chunked), each attending to all earlier prompt keys -- mapped entries for unselected positions, recomputed
#      entries for selected ones -- through an explicit 4D mask; the recomputed K/V replace the mapped entries.
# One position set is shared by all layers, as in the paper.
import math, torch

D = 128


def _strip_rope(k, theta=1e6):   # k [1, H, n, D] rotated -> content space, positions 0..n-1
    n = k.shape[2]
    inv = 1.0 / (theta ** (torch.arange(0, D, 2, device=k.device, dtype=torch.float32) / D))
    ang = torch.arange(n, device=k.device, dtype=torch.float32)[:, None] * inv[None]
    c, s = torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)
    k = k.float()
    return k * c + torch.cat([k[..., D // 2:], -k[..., :D // 2]], -1) * s


@torch.no_grad()
def scores(c, vr):
    """c: mapped target DynamicCache (batch 1); vr: dict(Vk [LT, H*D, r], Vv, myk [LT, H*D], myv) -> [T] scores."""
    s = 0
    for l in range(len(c.layers)):
        for kind, x in (("k", _strip_rope(c.layers[l].keys)), ("v", c.layers[l].values.float())):
            z = x[0].transpose(0, 1).reshape(x.shape[2], -1) - vr["my" + kind][l]          # [T, H*D]
            s = s + z.square().sum(1) - (z @ vr["V" + kind][l]).square().sum(1)
    return s


@torch.no_grad()
def recompute(T, c, hist, sel, chunk=4096):
    """Target model T recomputes positions `sel` (sorted LongTensor) of history `hist` (list of ids) in cache c."""
    dev = c.layers[0].keys.device
    n, sel = len(hist), sel.sort().values
    ids = torch.tensor(hist, device=dev)
    is_sel = torch.zeros(n, dtype=torch.bool, device=dev); is_sel[sel] = True
    done = 0
    for a in range(0, len(sel), chunk):
        p = sel[a:a + chunk]; k = len(p)
        base = n + done                                              # mapped entries + already recomputed ones
        m_map = (torch.arange(n, device=dev)[None] <= p[:, None]) & ~is_sel[None]   # [k, n]
        m_prev = torch.ones(k, done, dtype=torch.bool, device=dev)                  # earlier chunks precede p
        m_cur = p[None, :] <= p[:, None]                                             # [k, k] causal within chunk
        mask = torch.cat([m_map, m_prev, m_cur], 1)[None, None]                      # [1, 1, k, base + k]
        T(ids[p][None], position_ids=p[None], past_key_values=c, attention_mask=mask, use_cache=True)
        done += k
    for l in range(len(c.layers)):                                   # move recomputed entries into their slots
        K, V = c.layers[l].keys, c.layers[l].values
        K[:, :, sel] = K[:, :, n:n + len(sel)]; V[:, :, sel] = V[:, :, n:n + len(sel)]
        c.layers[l].keys, c.layers[l].values = K[:, :, :n].contiguous(), V[:, :, :n].contiguous()
    return c


@torch.no_grad()
def rarecache(T, c, hist, vr, rho):
    n = len(hist)
    k = math.ceil(rho * n)
    if k == 0:
        return c
    sel = scores(c, vr).topk(k).indices
    return recompute(T, c, hist, sel)
