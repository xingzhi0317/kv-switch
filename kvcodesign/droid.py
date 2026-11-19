# DroidSpeak (arXiv 2411.02820), re-implemented: the paper releases no code.  Our code.
# Setting of the paper: sender and receiver come from the SAME base model (same architecture and widths, different
# weights).  Here: Qwen3-1.7B-Base <-> Qwen3-1.7B (post-trained from it).
#   * The receiver reuses the sender's KV cache for every layer except ONE contiguous group of layers [a, b), which it
#     recomputes itself (paper: one contiguous group, because each reuse->recompute switch adds error).
#   * Recomputation starts from the sender's hidden state at the input of layer a (the paper's "E cache").
#   * The group is chosen offline (droid_profile.py): 50 HotpotQA contexts, all contiguous spans at 2-layer granularity,
#     the fewest recomputed layers whose F1 stays within 5% of the receiver's own F1.
# Choices the paper leaves open (ours, stated in the report):
#   - E = residual stream entering layer a (HF hidden_states[a]; hidden_states[0] = token embeddings of the SENDER).
#   - Layers after the group also reuse the sender's KV (Fig. 9 caption: "other layers reuse").
#   - Only the history tokens are recomputed; the new message is then prefilled by the receiver over the assembled
#     cache, like every other method in eval_quality.py.
#   - The E cache is precomputed with the sender's KV (the paper assumes stored caches), so it is not timed.
import torch
from transformers import DynamicCache


def parse(method):   # "droid{a}_{b}" -> (a, b): recompute layers a..b-1; droid0_0 = reuse every layer
    a, b = method[5:].split("_")
    return int(a), int(b)


@torch.no_grad()
def sender(S, ids, layers=None, chunk=8192):
    """Sender prefill -> (KV cache, {a: hidden state entering layer a, [1, n, d]}) for a in `layers` (default: all)."""
    dev = S.device
    c, hs = DynamicCache(), {}
    for p in range(0, len(ids), chunk):
        out = S.model(torch.tensor([ids[p:p + chunk]], device=dev), past_key_values=c, use_cache=True,
                      output_hidden_states=layers is None or len(layers) > 0)
        if out.hidden_states is not None:
            for a in (range(len(out.hidden_states) - 1) if layers is None else layers):
                hs.setdefault(a, []).append(out.hidden_states[a])
        del out
    return c, {a: torch.cat(v, 1) for a, v in hs.items()}


@torch.no_grad()
def receive(T, sc, e, a, b):
    """Assemble the receiver's cache: sender KV for layers outside [a, b); layers a..b-1 recomputed by the receiver
    from e = the sender's hidden state entering layer a (one causal pass over the history)."""
    L = len(T.model.layers)
    c = DynamicCache()
    for l in range(a):
        c.update(sc.layers[l].keys, sc.layers[l].values, l)
    if b > a:
        n = e.shape[1]
        pos = torch.arange(n, device=e.device)[None]
        h = e.to(T.dtype)
        pe = T.model.rotary_emb(h, pos)
        for l in range(a, b):   # attention_mask=None + empty layer cache -> plain causal attention (sdpa is_causal)
            h = T.model.layers[l](h, attention_mask=None, position_ids=pos, past_key_value=c, use_cache=True,
                                  cache_position=pos[0], position_embeddings=pe)
            h = h[0] if isinstance(h, tuple) else h
        del h
    for l in range(b, L):
        c.update(sc.layers[l].keys, sc.layers[l].values, l)
    return c
