import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer
p = "/home/wxz/kvswitch/models/Qwen3.5-2B"
tok = AutoTokenizer.from_pretrained(p)
m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16).cuda().eval()
print(type(m).__name__, sum(x.numel() for x in m.parameters()) / 1e9)
msgs = [{"role": "user", "content": "What is 17*23? Answer briefly."}]
s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
print(repr(s))
ids = tok(s, return_tensors="pt").input_ids.cuda()
out = m.generate(ids, max_new_tokens=30, do_sample=False)
print(repr(tok.decode(out[0, ids.shape[1]:])))
from transformers import DynamicCache
with torch.no_grad():
    o = m(ids, use_cache=True)
pk = o.past_key_values
print(type(pk).__name__, [type(l).__name__ for l in getattr(pk, "layers", [])][:5])
l0 = pk.layers[0]; l3 = pk.layers[3]
print({k: (tuple(v.shape) if hasattr(v, "shape") else v) for k, v in vars(l0).items() if not k.startswith("_")})
print({k: (tuple(v.shape) if hasattr(v, "shape") else v) for k, v in vars(l3).items() if not k.startswith("_")})
