# Cross-model KV cache translation: experiments

Research notes and code for switching a conversation from model A to model B by translating A's KV cache into B's format.

| Path | What it is |
|---|---|
| `PLAN.md`, `PLAN_v2.md`, `PLAN_v3.md` | Successive research plans (in Chinese) |
| `kvswitch/` | Phase 1: same-device switching study (Qwen3-1.7B/4B, Llama-3.2-3B, Qwen3.5-2B). Code in `src/`, write-up in `report.md`, raw results in `results/`, figures in `figures/` |
| `kvbase/` | Phase 2: translation-step latency vs. the official CacheBridge / kvbridge / kvtransfer implementations; fused Triton kernels in `fastmap.py`, write-up in `REPORT.md`, raw results in `results/` |
| `kvcodesign/` | Phase 3: baseline reproduction (KVBridge, CacheBridge, KL-trained translators, RaReCache, DroidSpeak, LLMLingua-2, SpecPrefill) and low-rank maps; full write-up in `REPORT.md`, raw per-question results in `results/` |
| `BASELINE_REPORT.md` | One-page summary of the baseline reproduction (in Chinese), including what does not match the papers |

Baselines are the authors' own code: [CacheBridge](https://github.com/WHU-XingyuQu/CacheBridge), [kvbridge](https://github.com/YunQipao/kvbridge), [kvtransfer](https://github.com/Susmith4710/kvtransfer). They are not vendored here; mapper weights are not included.
