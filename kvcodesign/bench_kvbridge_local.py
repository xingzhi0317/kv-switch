# Reproduce kvbridge's own published accuracy table (README "Accuracy under a fully mapped cache", Qwen3-1.7B -> 4B,
# top_k 12: HellaSwag 63.3 / ARC-C 42.7 / MMLU 55.1, 4B standalone 68.7 / 53.9 / 69.2, WikiText-2 prefix ppl 13.25 / 11.59)
# with the AUTHOR'S evaluation code: the question construction (load_bench_data), the batched log-likelihood scoring
# (score_mc_suite) and the WikiText-2 prefix perplexity of experiments/fit_eval_modal.py are imported and called as is;
# only the Modal cloud wrapper is stubbed out.  Mapped condition = official kvbridge.transfer with the published mapper
# (HF Siddharth85/kvbridge-qwen3-1.7b-to-qwen3-4b = ~/kvbase/mappers/kvbridge_17to4).  fp16 models, as bench_k().
# usage: MODE=mapped|standalone N=2000 CUDA_VISIBLE_DEVICES=a python bench_kvbridge_local.py
import os, sys, json, types, importlib.util, torch

E = os.environ.get
MODE, N = E("MODE", "mapped"), int(E("N", 2000))
HOME = os.path.expanduser("~")


class _Any:   # stands in for every modal object (Image chains, App, Volume, decorators): the helpers we call never touch it
    def __getattr__(self, k): return _Any()
    def __call__(self, *a, **k): return _Any()


sys.modules["modal"] = types.SimpleNamespace(Image=_Any(), App=_Any(), Volume=_Any(), Secret=_Any(), Mount=_Any(), gpu=_Any())
spec = importlib.util.spec_from_file_location("fem", f"{HOME}/kvbase/kvbridge/experiments/fit_eval_modal.py")
fem = importlib.util.module_from_spec(spec); spec.loader.exec_module(fem)

from transformers import AutoModelForCausalLM, AutoTokenizer
from kvbridge import Mapper, transfer
M = f"{HOME}/kvswitch/models"
device, dtype = "cuda", torch.float16
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained(f"{M}/Qwen3-1.7B")
m_tgt = AutoModelForCausalLM.from_pretrained(f"{M}/Qwen3-4B", torch_dtype=dtype).to(device).eval()
if MODE == "mapped":
    m_src = AutoModelForCausalLM.from_pretrained(f"{M}/Qwen3-1.7B", torch_dtype=dtype).to(device).eval()
    mp = Mapper.load(f"{HOME}/kvbase/mappers/kvbridge_17to4", device=device)

    def make_cache(prefix_ids):   # source prefill -> official transfer
        return transfer(m_src(prefix_ids.unsqueeze(0), use_cache=True).past_key_values, mp)
else:
    def make_cache(prefix_ids):   # the 4B's own prefill
        return m_tgt(prefix_ids.unsqueeze(0), use_cache=True).past_key_values

bench = fem.load_bench_data(tok, N)
mc = fem.score_mc_suite(m_tgt, tok, bench, make_cache, device, tag=f"[{MODE}]")
wt = fem.wikitext_prefix_ppl(m_tgt, fem.load_wikitext_chunks(tok), make_cache, device, tag=f"[{MODE}]")
out = dict(mode=MODE, n_samples=N, mc=mc, wikitext=wt)
os.makedirs(f"{HOME}/kvcodesign/results/kvbridge_bench", exist_ok=True)
json.dump(out, open(f"{HOME}/kvcodesign/results/kvbridge_bench/{MODE}.json", "w"), indent=1)
print("BENCHDONE", json.dumps({k: round(v["acc_norm" if v["metric"] == "acc_norm" else "acc"] * 100, 1) for k, v in mc.items()}), "ppl", round(wt["ppl"], 2))
