# Checks for rare.recompute (GPU): recomputing ALL positions must reproduce the native cache's next-token logits;
# recomputing none must leave the translation unchanged; a partial set must lie in between (KL to native).
import os, sys, torch
sys.path.insert(0, os.path.expanduser("~/kvbase/kvbridge")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from kvbridge import Mapper, transfer
import rare
M = os.path.expanduser("~/kvswitch/models"); dev = "cuda"
tok = AutoTokenizer.from_pretrained(f"{M}/Qwen3-4B")
S = AutoModelForCausalLM.from_pretrained(f"{M}/Qwen3-1.7B", torch_dtype=torch.bfloat16).to(dev).eval()
T = AutoModelForCausalLM.from_pretrained(f"{M}/Qwen3-4B", torch_dtype=torch.bfloat16).to(dev).eval()
kb = Mapper.load(os.path.expanduser("~/kvbase/mappers/kvbridge_17to4"), device=dev)
vr = {k: v.to(dev) for k, v in torch.load(os.path.expanduser("~/kvcodesign/mappers/rare_vr_17to4.pt")).items()}
text = "The quick brown fox jumps over the lazy dog. My locker code is 482913 and my name is Alice. " * 60
hist = tok(text, add_special_tokens=False).input_ids[:3000]
q = tok(" What is the locker code?", add_special_tokens=False).input_ids
with torch.no_grad():
    c = DynamicCache(); T(torch.tensor([hist], device=dev), past_key_values=c, use_cache=True)
    ref = T(torch.tensor([q], device=dev), past_key_values=c, use_cache=True).logits[0].float().log_softmax(-1)
    sc = S(torch.tensor([hist], device=dev), use_cache=True).past_key_values

    def kl(cache):
        lp = T(torch.tensor([q], device=dev), past_key_values=cache, use_cache=True).logits[0].float().log_softmax(-1)
        return float((ref.exp() * (ref - lp)).sum(-1).mean())

    k_tr = kl(transfer(sc, kb))
    k_all = kl(rare.recompute(T, transfer(sc, kb), hist, torch.arange(len(hist), device=dev), chunk=1024))
    k_none = kl(rare.rarecache(T, transfer(sc, kb), hist, vr, 0.0))
    k30 = kl(rare.rarecache(T, transfer(sc, kb), hist, vr, 0.3))
    rnd = torch.randperm(len(hist), device=dev)[:900]
    k30r = kl(rare.recompute(T, transfer(sc, kb), hist, rnd))
print(f"KL to native: translation {k_tr:.4f} | recompute all {k_all:.5f} | rho=0 {k_none:.4f} | rho=0.3 (score) {k30:.4f} | 30% random {k30r:.4f}")
assert k_all < 0.02 and abs(k_none - k_tr) < 1e-6 and k30 < k_tr
print("RARE OK")
