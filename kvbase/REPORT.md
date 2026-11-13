# 只优化"翻译"这一步：与官方实现的对比

> 场景：B 拿到 A 的 KV cache 后，把它翻译成 B 的 KV cache。只优化这一步的耗时，质量不能比官方实现差。
> 模型对：Qwen3-1.7B → Qwen3-4B（kvbridge 有官方发布的映射器，与 kvtransfer 的测试对也接近）。硬件：单张 RTX 4090（24GB，Ada）。
> 服务器路径 `~/kvbase/`，本地副本在本目录。原始结果：`results/*.json`。

## 1. 对比对象（全部跑的是官方代码）

| 名称 | 来源 | 映射结构 | 翻译实现 |
|---|---|---|---|
| CacheBridge | [WHU-XingyuQu/CacheBridge](https://github.com/WHU-XingyuQu/CacheBridge) `PageMapper.map_page` | head-local：每个目标 head 读 k=8 个源层同编号 head | PyTorch FP32 einsum，每 2048 token 一页、按层 Python 循环 |
| kvbridge | [YunQipao/kvbridge](https://github.com/YunQipao/kvbridge) `transfer`，官方发布的 1.7B→4B 映射器 | 稠密：读 12 个源层的全部 head | PyTorch bf16 `addmm` |
| kvtransfer（NVIDIA 方法） | [Susmith4710/kvtransfer](https://github.com/Susmith4710/kvtransfer) `Mapper.apply_kv`，用与 kvbridge 相同的权重 | 稠密 | PyTorch |
| CacheBridge 跑稠密映射器 | CacheBridge `PageMapper(method="nv_transfer")`，同一份稠密权重 | 稠密 | 同 CacheBridge |

**CacheBridge 的映射器**：官方只发布了 14B→32B 的拟合流程，且"按 R² 选源层"的初始化器没有开源。1.7B→4B 的映射器按以下方式得到：
- 官方有的部分直接调用官方代码：kvbridge 的 FineWeb-Edu 采集（500×1024 token，步长 4），CacheBridge 的 `fit_layer`（head-local 中心化岭回归，λ=0.01），CacheBridge 的校准代数（`probabilities/pool/gram/solve`，岭为 0.01×平均对角）。
- 官方没开源的部分按论文补上：按 head-local R² 选 8 个源层，以及在 24 篇 QuALITY 文档 × 4 题上逐层顺序做 V 校准。
- **只测翻译耗时时，映射器的数值不影响速度**，所以耗时表在拟合完成前就可以测（用同形状的随机权重）。

## 2. 我们的优化

1. **head-local 融合 kernel**（`fastmap.py: FastHeadLocal`，Triton）：一个 kernel 完成全部步骤，不落中间张量。
   - 按 token 块读 8 个源层的 K/V；
   - 在寄存器里去 RoPE（拆成前后两半直接算）；
   - bf16 张量核矩阵乘，fp32 累加 8 个源层；
   - 加偏置，再加 B 的 RoPE，直接写进输出。
   - 分块参数（每块 32 token、8 warp、2 级流水）经实测扫参选出。
   - 支持把多个请求沿 token 维拼起来一次翻译（每个 token 带自己的位置），用于高并发。
2. **稠密映射器**（`FastDense`）：
   - 所有源层的 K 一次性去 RoPE（融合 kernel）；
   - 每个目标层一个 GEMM；
   - 加 RoPE 融合在写回里；
   - 可选 FP8（E4M3，逐张量缩放，Ada FP8 张量核）。

**数值一致性**（随机输入，4K token）：

| | 与谁比 | K 相对误差 | V 相对误差 |
|---|---|---|---|
| 我们的 head-local（bf16） | CacheBridge 官方（FP32） | 0.31% | 0.26% |
| 我们的稠密（bf16） | kvbridge 官方 | 0.44% | 0.47% |
| 我们的稠密（FP8） | kvbridge 官方 | 3.6% | 3.3% |

下游质量见 §5。

## 3. 单请求翻译耗时（ms，p50，10 次）

| 历史长度 | CacheBridge 官方 | **我们 head-local** | 加速 | kvbridge 官方 | kvtransfer | CB 跑稠密 | 我们稠密 bf16 | 我们稠密 FP8 |
|---|---|---|---|---|---|---|---|---|
| 2K | 30.5 | **4.0** | 7.6× | 50.0 | 44.2 | 91.5 | 43.4 | 23.9 |
| 4K | 62.0 | **7.5** | 8.3× | 93.2 | 84.0 | 183.3 | 85.5 | 50.0 |
| 8K | 121.9 | **14.9** | 8.2× | 185.6 | 164.6 | 368.0 | 165.4 | 99.0 |
| 16K | 243.7 | **29.7** | 8.2× | 393.4 | 325.3 | 744.3 | 331.3 | 201.1 |
| 32K | 487.2 | **59.8** | 8.1× | 812.3 | 677.0 | 1492.0 | 663.4 | 399.1 |

- head-local 在 32K 时的理论下限约 40–60ms（4.9 TFLOP 计算，约 20GB 访存）。我们的 59.8ms 已接近下限。
- 稠密映射器每 token 计算量是 head-local 的 12 倍，本身是算力瓶颈：bf16 只能和 kvtransfer 持平，FP8 再快约 1.7 倍。
- 峰值额外显存：CacheBridge 官方每次翻译固定多占约 2.5GB（FP32 中间张量）；kvbridge 和 kvtransfer 随长度线性增长，32K 时约 10–11GB。我们只需要输出缓冲。

## 4. 系统层面

### 4.1 高并发下的翻译吞吐（多个切换请求同时到达）

| 场景 | CacheBridge 官方（每请求一个 stream） | 我们（拼成一次 kernel） |
|---|---|---|
| 2K × 1 | 30.9ms，0.07 M token/s | 3.9ms，0.53 M token/s |
| 2K × 4 | 116ms，0.07 M token/s | 15.0ms，0.54 M token/s |
| 2K × 16 | 1114ms，**0.03** M token/s | 60.5ms，0.54 M token/s |
| 8K × 1 | 124ms，0.07 M token/s | 15.2ms，0.54 M token/s |
| 8K × 4 | 468ms，0.07 M token/s | 60.5ms，0.54 M token/s |

我们的吞吐稳定在约 54 万 token/s，是 CacheBridge 官方的 7.7 倍；16 个并发时差距扩大到 18 倍，因为官方实现在多 stream 下互相干扰、吞吐反而下降。

### 4.2 对服务吞吐的影响
B 在服务 8 条请求的解码（上下文 2K），同时每秒有 r 个 8K 长度的切换请求，在低优先级 stream 上做翻译：

| 每秒切换数 r | 解码吞吐（CB 官方） | 解码吞吐（我们） | 负载下单次翻译耗时（CB / 我们） |
|---|---|---|---|
| 0 | 185 tok/s | 185 tok/s | — |
| 1 | 175（−5%） | 182（−2%） | 156 / 24 ms |
| 2 | 166（−10%） | 182（−1%） | 155 / 23 ms |
| 4 | 146（**−21%**） | 174（**−6%**） | 156 / 22 ms |

### 4.3 端到端 TTFT 与"翻译什么时候是瓶颈"

TTFT = 传输 A 的 cache + 翻译 + B 处理新消息（128 token）并解码第一个 token。
- **B 的那部分**用 HF transformers（sdpa，eager）实测，约 96–147ms。它被 kernel 启动开销主导，用 CUDA graph 的生产引擎会小得多，那样翻译的占比会更高。
- **传输带宽**：本机 4090 之间实测 15.5 GB/s（无 NVLink，走 PCIe）；NVLink（H800，400 GB/s）和跨机 RDMA（4×400Gb/s ≈ 200 GB/s）按带宽换算。
- **流水线**：按 CacheBridge 的 2048 token 分块，传输和翻译重叠。

| 历史长度 | 从头重读 | CB 同卡 | **我们同卡** | CB NVLink（流水） | 我们 NVLink（流水） | CB PCIe（流水） | 我们 PCIe（流水） | 翻译占比 CB → 我们（同卡） |
|---|---|---|---|---|---|---|---|---|
| 2K | 160 | 130 | **103** | 130 | 103 | 144 | 117 | 24% → 4% |
| 4K | 320 | 158 | **104** | 159 | 105 | 173 | 130 | 39% → 7% |
| 8K | 705 | 234 | **126** | 235 | 126 | 249 | 173 | 53% → 12% |
| 16K | 1649 | 353 | **136** | 354 | 136 | 368 | 227 | 70% → 22% |
| 32K | 4272 | 641 | **207** | 642 | 208 | 657 | 393 | **77% → 29%** |

（单位 ms。）结论：
- **同卡和 NVLink/RDMA 下，翻译是 CacheBridge 端到端时延的主要部分**：8K 以上占一半以上，32K 占 77%。我们把端到端 TTFT 降了 1.3 倍（2K）到 3.1 倍（32K）。
- **PCIe 下**：传输本身（32K 时 242ms）成为主要部分。CacheBridge 的流水线能把它的翻译藏在传输后面一部分，但它的翻译（495ms）比传输还慢，所以仍然是瓶颈。我们的翻译（60ms）完全藏在传输后面，端到端由传输决定。
- **高并发**：吞吐差 7.7–18 倍，对服务吞吐的干扰从 −21% 降到 −6%（§4.1、§4.2）。
- **短历史（2K）**：翻译只占 24%，优化的端到端收益有限（1.3 倍）。
- **注意**：CacheBridge 原文在 H800 上用 4 卡张量并行，每张卡只翻译 1/4 的 head，算力也更强，所以在它的硬件上翻译的绝对耗时和占比会更小。本报告的结论适用于单卡、消费级或推理卡的部署。

## 5. 质量（加速不能掉质量）

（进行中：CacheBridge 映射器拟合完成后，评测"CB 官方 vs 我们的融合版"在 QuALITY、HotpotQA-E、GSM8K 上的结果。）

稠密映射器（kvbridge 发布的权重），QuALITY 前 100 题：

| | 从头重读 | kvbridge 官方 | 我们稠密 bf16 | 我们稠密 FP8 |
|---|---|---|---|---|
| 准确率 | 73% | 45% | 44% | 43% |
