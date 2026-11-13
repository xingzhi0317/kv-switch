# kvswitch — 同设备多模型切换时的 KV cache 转换

服务器路径 `~/kvswitch/`（本地副本：`Desktop/kv_switch_plan/kvswitch/`）。Python 环境 `~/kvmap/.venv`。
所有实验均为单卡（RTX 4090 24GB）。结论与验收结果见 [report.md](report.md)。

## 目录
| 路径 | 内容 |
|---|---|
| `src/ks.py` | 统一推理路径：手写 Qwen3 / Llama-3.2 前向，读写预分配的外部 KV cache（K 存 post-RoPE）。原生与所有方法共用这一条路径，只有 cache 来源不同 |
| `src/translate.py` | 运行时翻译器（去 RoPE → 多源层仿射映射 → 加 B 的 RoPE；sink=B 自己的第 0 个 token）；C5 边译边释放、C6 逐层事件回调；B3 选择性重算 |
| `src/fit.py` | Case 1 闭式拟合：B2（NVIDIA 式岭回归）、B1（CacheBridge 式，V 按 B 的注意力输出校准）；k=1/3 源层 |
| `src/train.py` | C4：在 B1 上用 KL 自蒸馏训练 rank-r 低秩残差，前缀长度随机 2K–8K |
| `src/xalign.py`, `src/fit_x.py` | Case 2（P3）：按字符覆盖对齐 token；模板/特殊 token 由 B 自己重算；闭式初始化 |
| `src/tasks.py` | 任务构造：CoQA（切换点 k=1/5/10，切换后 3 轮）、LongBench 4 子集（文档在第 1 轮）、找针、GSM8K 多轮、UltraChat KL |
| `src/eval.py` | 质量评测（切换协议 §4.1），每样本记录分数、T_switch、峰值显存、守卫统计量 |
| `src/timing.py` | M4 系统测量：阶段拆分计时、峰值显存、C1/C5/C6、B6 影子 prefill 后台开销与干扰、C3、单元测试 |
| `src/analyze.py` | 汇总：保留率与 bootstrap 95% CI、有害翻转、守卫 ROC、图 |
| `src/m0_tests.py` | M0：RoPE 单测（Qwen3 / Llama-3.2 / Qwen3.5）、前向等价、显存预算、prefill 时间表 |
| `results/` | 原始结果 JSON（逐样本），`tables.md`（汇总表），`summary.json`，`guard.json`，`timing_*.json`，`m0.json` |
| `figures/` | 图 |
| `maps/` | 拟合/训练得到的映射（P1/P2/P3 闭式，`*_ours_r*.pt` 低秩残差） |

## 复现
```bash
cd ~/kvswitch/src && source ~/kvmap/.venv/bin/activate
python prep_data.py                                   # 数据（HF_ENDPOINT=https://hf-mirror.com）
python m0_tests.py                                    # M0
A=Qwen3-4B B=Qwen3-1.7B TAG=P1 python fit.py          # B1/B2 映射（P2: A=Qwen3-1.7B B=Qwen3-4B TAG=P2）
PAIR=P1 STEPS=600 RANK=64 python train.py             # C4（消融：RANK=16；NMIN=NMAX=2048 TAG=ours_r64_fix2k）
PAIR=P1 TASK=coqa5 N=100 METHODS=native,B2,B1,B3 python eval.py      # 质量；TASK ∈ uc coqa{1,5,10} lb8k niah{4,8,16,32}k gsm
PAIR=P1 TASK=coqa5 N=100 METHODS=ours_r64,ours_r64_c3,B1_c3 SUFFIX=_ours python eval.py
PAIR=P1 python timing.py                              # 计时/显存/后台开销
A=Llama-3.2-3B-Instruct B=Qwen3-1.7B TAG=P3 python fit_x.py && PAIR=P3 TASK=lb8k N=200 METHODS=native,none,B4,B4k1 python eval.py
python analyze.py                                     # -> results/tables.md, results/guard.json, figures/*.png
```
方法名：`native`=B0 重新 prefill；`none`=B 不看历史（下界）；`B1`/`B2`/`B3` 见计划 §3；`B1k1`=1 个源层；
`ours_r64`=C4 翻译器；`*_c3`=第一轮用翻译 cache、之后用后台原生重算的 cache（C3）；`noise*`=故意降质的翻译器（M3 标定）。
