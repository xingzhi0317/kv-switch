import os; os.environ["MODE"] = "none"
import sys; sys.argv = ["x"]
exec(open("p4.py").read().split("# ---------------------------------------------------------------- fit")[0])
from transformers import DynamicCache
msgs = [("user", "Tell me a short fact about Lisbon and Côte d'Ivoire, 12,345 times."), ("assistant", "Lisbon is the capital of Portugal.")]
ia, ib, mp = align(msgs)
print(len(ia), len(ib), [(Btok.decode([ib[j]]), A.tok.decode([ia[mp[j]]])) for j in range(len(ib))][:20])
new = b_turn("user", "And Paris?") + B_GEN
full = Bm(input_ids=torch.tensor([ib + new], device=dev)).logits.float()[0, len(ib):]
c = DynamicCache(config=tc)
b_fwd(ib, c, 0)
lg = b_fwd(new, c, len(ib), "all")[0]
print("split-vs-full maxabs", (lg - full).abs().max().item(), "top1", (lg.argmax(-1) == full.argmax(-1)).float().mean().item())
# replay sanity: feeding B's TRUE layer inputs to the replay path must reproduce native exactly
grab = {}
hs = [Bm.model.layers[l].input_layernorm.register_forward_hook(lambda m, i, o, l=l: grab.__setitem__(l, o)) for l in range(LB)]
hk = {}
for l in FULL:
    Bm.model.layers[l].self_attn.k_norm.register_forward_hook(lambda m, i, o, l=l: hk.__setitem__(("k", l), o))
    Bm.model.layers[l].self_attn.v_proj.register_forward_hook(lambda m, i, o, l=l: hk.__setitem__(("v", l), o))
with torch.no_grad():
    Bm(input_ids=torch.tensor([ib], device=dev), use_cache=False)
for h in hs: h.remove()
c2 = DynamicCache(config=tc)
N = len(ib); cos, sin = b_cos(0, N); rd = cos.shape[-1]
with torch.no_grad():
    for l in range(LB):
        if LT[l] == "full_attention":
            k = hk[("k", l)][0].transpose(0, 1).float(); v = hk[("v", l)][0].view(N, HKB, DB).transpose(0, 1)
            k = torch.cat([rope(k[..., :rd], cos, sin), k[..., rd:]], -1).to(torch.bfloat16)
            c2.update(k[None], v[None].contiguous(), l)
        else:
            Bm.model.layers[l].linear_attn(grab[l], cache_params=c2)
lg2 = b_fwd(new, c2, N, "all")[0]
print("oracle-replay-vs-native maxabs", (lg2 - lg).abs().max().item(), "top1", (lg2.argmax(-1) == lg.argmax(-1)).float().mean().item())
