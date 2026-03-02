---
title: "模型微调与 vLLM 探索项目"
description: "基于 L20 记录的微调路线与推理验证"
date: 2026-01-22
category: "项目"
tags: ["LoRA", "DeepSpeed", "vLLM", "L20"]
---

## 背景与目标

这是围绕 L20 服务器进行的一次工程实践，目标是建立一条“微调 → 推理服务 → 指标回归”的可复用路径：训练侧以 LoRA 降低成本，分布式训练依赖 DeepSpeed，推理服务统一使用 vLLM，并把关键参数的含义、失败形态和排查方式沉淀成可复现的记录。

## 记录来源

关键实验过程留存在 L20 server.txt 中（原始记录见：`https://github.com/yl2581961856-collab/ML-sys-learning-project/blob/main/L20%20server.txt`），包括环境、调参、训练日志与推理结果。这里整理为可复用的路线与结论。磁盘分区与 fdisk 细节不展开，优先保留对“微调 LoRA + 推理 vLLM + 失败排查 + kernel/graph 性能观察”有用的部分。

<a id="reader-guide"></a>
## 读者指南（第三方视角）

- 你能从这页得到什么：一条可复现的“训练 → 验证 → 服务化 → 回归”最小闭环，以及每个关键旋钮的工程含义
- 你该怎么读：先看「快速开始」，再按需跳转到 LoRA/ZeRO 与 vLLM Kernel 分析，最后用「可复现附录」把你的每次实验写成可回归条目
- 你什么时候会用到：当你 OOM、吞吐不稳定、vLLM 启动卡死/超时、或想解释“为什么这样配参数”给第三方听

<a id="quickstart"></a>
## 快速开始（最短复现路径）

1. 环境确认：GPU/驱动/CUDA/多卡通信是否健康（`nvidia-smi`、`nvidia-smi topo -m`）
2. 数据最小闭环：先用 50–100 条样本跑通一轮（确保格式、tokenizer、图像路径都正确）
3. 训练起步：LoRA + ZeRO-2（能跑稳再上 ZeRO-3），优先 `bf16`，先把 `max_length`/分辨率压到可控
4. 训练产物验证：不要先合并权重，直接挂载 LoRA 做推理对比（固定对照 prompt）
5. vLLM 服务化：把 `gpu-memory-utilization`、`max-model-len` 写死成“服务侧边界”
6. 性能/Kernel 观察：先把 TTFT/TPOT/吞吐跑出基线，再决定要不要碰 CUDA Graph、Chunked Prefill、Speculative Decoding

## 本页目录（索引）

- 快速入口
  - [读者指南（第三方视角）](#reader-guide)
  - [快速开始（最短复现路径）](#quickstart)
- 路线概览
  - [微调路线](#fine-tune-path)
  - [推理验证](#inference-verify)
  - [复盘要点](#postmortem)
  - [可复现附录（给未来的自己和同伴）](#repro-appendix)
- 归档（把 2026-03 系列并入本项目）
  - [归档入口（2026-03 系列）](#archive-2026-03)
  - [微调的深水区：LoRA 与 DeepSpeed 的工程抉择](#archive-lora-deepspeed)
  - [跨越软件的边界：拓扑、通信与操作系统底层](#archive-topology-os)
  - [推理即未来：vLLM vs TGI vs SGLang 的架构对决](#archive-inference-arch)

<a id="fine-tune-path"></a>
## 微调路线

- 数据准备：指令格式统一、最小闭环可复现
- 训练策略：LoRA + DeepSpeed（ZeRO-2 起步）
- 训练产出：权重保存、超参记录、训练指标对齐

<a id="inference-verify"></a>
## 推理验证

- vLLM 启动与服务化脚本
- 接口回归与输出质量检查
- 吞吐、TTFT、显存占用记录

<a id="archive-2026-03"></a>
## 归档：2026-03 深入拆解（并入本项目）

下面三块内容原本是独立文章，现在统一归档到“模型微调与 vLLM 探索项目”中，作为项目过程中的阶段性沉淀，便于你后续继续补全每次训练命令、参数释义与结果对比。

<a id="archive-lora-deepspeed"></a>
### 归档：微调的深水区：LoRA 与 DeepSpeed 的工程抉择

微调最容易被低估的部分不是“脚本怎么写”，而是“显存/通信/吞吐怎么平衡”。在 48GB 档（例如 L20）的单机多卡上，这些抉择决定你能不能把一套流程长期跑稳。

#### 1）数据与 OOM 的博弈：显存峰值到底被谁吃掉了

以常见的训练形态拆账：参数、梯度、优化器状态、激活值（activation）和临时 buffer。

- 参数（fp16/bf16）：约 `2 bytes * 参数量`
- 梯度（通常 fp16/bf16 或 fp32）：约 `2~4 bytes * 参数量`
- Adam 优化器状态（常见 fp32）：`m` + `v` 两份，约 `8 bytes * 参数量`
- 激活值：与 batch、序列长度、模型层数强相关，峰值往往来自长序列/高分辨率视觉分支/大 batch 的中间特征

针对 500–1000 张高分辨率图片的数据集，微调阶段最常见的两个“误判”：

- 以为数据量不大就不会 OOM，但单 step 的激活峰值与你 dataset 总量无关
- 以为 LoRA 一定省显存，但当你把 batch/分辨率/序列长度推上去时，激活值依然能把显存打穿

##### 可复现命令（模板）

下面以“Transformers + PEFT + DeepSpeed”为例给出命令模板。你只需要替换路径与模型名。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=warn

deepspeed --num_gpus 4 train.py \
  --model_name_or_path <MODEL> \
  --dataset <DATASET_PATH_OR_NAME> \
  --output_dir <OUT_DIR> \
  --bf16 true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --max_steps 2000 \
  --logging_steps 10 \
  --save_steps 200 \
  --deepspeed ds_zero2.json \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules q_proj,k_proj,v_proj,o_proj
```

参数解释（按“最容易踩坑”的顺序）：

- `--per_device_train_batch_size`：真正触发激活峰值的第一按钮。先从 1 起步，用累积步数补回有效 batch
- `--gradient_accumulation_steps`：用时间换显存，注意它会改变有效学习率（常见做法是按有效 batch 线性缩放）
- `--bf16 true`：如果硬件支持，优先 bf16（数值更稳），显存与 fp16 同级
- `--deepspeed ds_zero2.json`：把“省显存的代价”显式写进配置（下文专门拆 ZeRO）
- `--lora_r / --lora_alpha`：LoRA 的有效更新幅度关键参数（下一节）

#### 2）LoRA 不是银弹：Rank/Alpha 如何影响收敛

直觉上可以把 LoRA 看成把某些线性层的权重更新限制在一个低秩子空间中。这里两个参数经常被“拍脑袋”：

- `r`（rank）：可表达能力上限。太小会欠拟合、太大又会吃掉你省下来的显存与吞吐
- `alpha`：对 LoRA 更新的缩放。常见实现里有效缩放大致与 `alpha / r` 同阶，改变它会改变每步更新幅度

实践建议（偏工程，不追求唯一答案）：

- 先固定 `alpha/r`，再调 `r` 看“能力上限”和“显存/吞吐”曲线
- 如果出现 loss 前期下不去、或者非常慢但不发散：先检查学习率与有效 batch，再回头看 `alpha/r`
- 如果出现很快震荡甚至 NaN：先把学习率降下来，再考虑减小 `alpha` 或增大 `r`（降低单方向更新的尖锐程度）

##### 快速验证：不合并权重，直接挂载 LoRA 跑推理

工程上最省时间的做法是“训练产物能立刻被推理进程加载验证”，避免合并全参/再导出一次。

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <BASE_MODEL> \
  --enable-lora \
  --lora-modules lora=<LORA_DIR> \
  --served-model-name <NAME>
```

这类验证能回答两个最关键的问题：

- LoRA 权重是否真的生效（输出风格/能力是否变化）
- 训练是否过拟合/欠拟合（用一组固定对比 prompt 快速看趋势）

#### 2.1）基于 L20 server.txt 的补充（不含磁盘分区）

这份工程实践的硬件/软件底座（用于对齐你的复现环境与预期）：

- OS：CentOS 8.5（4.18 内核系）
- GPU：NVIDIA L20 × 4（46GB × 4），CPU 64 核
- Driver：580.105.08（记录中同时出现过 575/580 安装介质）
- CUDA：记录为 13.0

建议你在每次实验日志开头固定贴三段输出（第三方视角读起来会非常安心）：

```bash
cat /etc/redhat-release || cat /etc/os-release
uname -r
nvidia-smi
```

#### 2.2）ZeRO-3 + LoRA 的“能跑稳”配方（从日志反推）

原始记录里出现过一组非常典型的“从 OOM 走向稳定”的组合拳（核心不是某个单点参数，而是顺序）：

1. 先启用 `PYTORCH_ALLOC_CONF=expandable_segments:True`，降低碎片化导致的“看起来还剩显存但突然 OOM”
2. 再把 `max_length` 拉回可控范围（长 CoT/多模态会让激活值暴涨）
3. 必须启用 `gradient_checkpointing`（以时间换显存）
4. 仍然不够再上 ZeRO-3 分摊参数/optimizer

一个“可复制”的启动模板（命令行参数以占位符表示，适配 Swift/Transformers 任一训练入口都成立）：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=warn
export PYTORCH_ALLOC_CONF=expandable_segments:True

deepspeed --num_gpus 4 train.py \
  --model_name_or_path <BASE_MODEL> \
  --dataset <DATASET_JSONL> \
  --output_dir <OUT_DIR> \
  --bf16 true \
  --gradient_checkpointing true \
  --max_seq_len <MAX_LEN> \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --max_steps 2000 \
  --save_steps 200 \
  --logging_steps 10 \
  --deepspeed ds_zero3.json \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05
```

下面给一份可以直接粘贴的 `ds_zero3.json`（把“省显存的代价”显式写出来，避免玄学）：

```json
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 8,
  "steps_per_print": 50,
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 134217728,
    "allgather_bucket_size": 134217728
  },
  "gradient_clipping": 1.0,
  "wall_clock_breakdown": false
}
```

读这个配置的方式（第三方视角最关心“为什么这样配”）：

- `stage: 3`：把参数也切碎分摊，换显存上限；代价是更多 all-gather/reduce-scatter 调度
- `overlap_comm: true`：让通信尽量藏到计算间隙里，缓解吞吐掉得太厉害
- `bucket_size`：越大越省通信调用次数，但也更占瞬时 buffer；遇到尖刺/不稳定再回头调它

#### 3）DeepSpeed ZeRO 策略解剖：收益边界与代价

把 ZeRO 当成“把冗余状态切碎分摊”的工程手段更好理解：省显存的同时，你在通信与碎片化上付出成本。

- ZeRO-1：分片优化器状态
- ZeRO-2：在 ZeRO-1 基础上再分片梯度
- ZeRO-3：进一步分片参数本身（显存更省，通信与调度更重）

##### 选择建议（单机多卡、L20 档的常见轨迹）

- 优先试 ZeRO-2：通常是“收益/复杂度”最稳的甜点位
- ZeRO-3 用于“模型太大/序列太长”导致 ZeRO-2 仍 OOM 的场景，但要预期：
  - 通信更频繁，吞吐可能明显下降
  - 参数分片带来更多 all-gather/reduce-scatter 的调度开销

##### ZeRO-3 的核心矛盾：参数分片 vs 通信开销

当你把参数切碎后，计算某个 layer 时不得不把它临时聚齐，算完又切回去；如果通信链路（PCIe/NVLink/NIC）不够快，你省下的显存会以吞吐的形式吐回去。

因此 ZeRO-3 的调参通常围绕两件事：

- 通过 bucket/overlap 等策略，减少通信对计算的“硬阻塞”
- 通过 offload（CPU/NVMe）在 OOM 与速度之间找一个能接受的点

##### 推荐资源（含可渲染视频）

- LoRA 论文：Low-Rank Adaptation of Large Language Models（arXiv:2106.09685）
- DeepSpeed 官方仓库与文档：https://github.com/microsoft/DeepSpeed
- ZeRO 论文：ZeRO: Memory Optimizations Toward Training Trillion Parameter Models（arXiv:1910.02054）
- PEFT（LoRA 常用实现）：https://github.com/huggingface/peft
- PEFT 文档：https://huggingface.co/docs/peft/
- Accelerate + DeepSpeed 指南：https://huggingface.co/docs/accelerate/usage_guides/deepspeed

<div class="embed">
  <iframe
    src="https://player.bilibili.com/player.html?bvid=BV1fb421t7KN&page=1&high_quality=1"
    scrolling="no"
    frameborder="0"
    allowfullscreen="true"
  ></iframe>
</div>

<a id="archive-topology-os"></a>
### 归档：跨越软件的边界：拓扑、通信与操作系统底层

越往后做，你会发现“训练不稳定/吞吐上不去/延迟抖动”很少是模型结构的问题，更多来自通信与 I/O 的系统性约束。把这些约束讲清楚，是普通炼丹师写不出来的深度。

#### 1）拓扑与通信瓶颈：先把机器画出来

第一件事不是改代码，而是把拓扑看清楚。

```bash
nvidia-smi topo -m
```

你关心的不是“有几张卡”，而是卡与卡之间的路径：

- NVLink：通常带宽最高、延迟最低
- PCIe：带宽与拓扑高度相关（同 root complex vs 跨 NUMA）
- 跨 NUMA / 跨 CPU socket：容易出现隐性带宽上限与抖动

##### 用一句话理解 NCCL 的 Ring/Tree

- Ring：更像“均匀搬运”，带宽利用率通常好，延迟随规模增长更线性
- Tree：更像“分层汇聚”，延迟可能更低，但更依赖拓扑与实现细节

工程上最重要的不是背定义，而是知道你在看什么日志：

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV
```

如果你在多卡训练里看到吞吐忽高忽低，优先排查：

- 实际走的是 NVLink 还是退化到 PCIe
- 是否因为跨 NUMA 访问导致通信/拷贝绕路
- 是否因为数据加载/I/O 抖动把 GPU 吃不满（下一节）

#### 2）操作系统层面的隐形损耗：Page Cache 与内存颠簸

加载大量高分辨率图片时，最容易被忽视的是 Page Cache 行为：

- 你以为“读盘慢”，但很多时候是在“内存被 cache 撑爆后反复回收”
- 训练进程的 RSS、Page Cache、数据增强临时 buffer 一起把系统内存打满，就会出现抖动

快速观察（更关心趋势而不是精确数）：

```bash
free -h
cat /proc/meminfo | head -n 30
```

你想看到的是：

- `MemAvailable` 是否持续下滑并触底
- `Cached` 是否在增长后被频繁回收（伴随延迟尖峰）

##### 训练吞吐“锯齿状”的典型根因链路

1. data loader 预取高分辨率样本，触发大量 page fault 与 cache fill
2. OS 回收 cache，导致后续批次再次 page fault
3. CPU 侧解码/增强拉长 step 时间，GPU 等数据
4. 训练吞吐出现周期性锯齿

#### 3）I/O 模型对比：同步 I/O vs io_uring（以 checkpoint 为例）

当你做大模型训练时，“写 checkpoint”经常被当成不可避免的停顿，但系统层的差异非常真实：

- 同步 I/O：简单，但容易在高并发/大文件时阻塞用户态线程
- io_uring：把提交与完成分离，减少 syscalls 与上下文切换，在大量小块读写/并发队列时优势明显

如果你的工程里已经有 Rust 侧基础设施（例如你熟悉 Monoio），把 checkpoint/数据加载的 I/O 逻辑独立出来做压测，收益往往比再调一个训练参数更大。

##### 推荐资源

- NCCL 官方文档：https://docs.nvidia.com/deeplearning/nccl/
- NVIDIA Collective Communication（入门）：https://developer.nvidia.com/nccl
- `nvidia-smi topo` 与 NVLink/PCIe 基础参考：https://docs.nvidia.com/deploy/nvidia-smi/
- NCCL 测试工具（快速验证带宽/拓扑）：https://github.com/NVIDIA/nccl-tests
- io_uring 官方文档（Linux）：https://kernel.dk/io_uring.pdf

<a id="archive-inference-arch"></a>
### 归档：推理即未来：vLLM vs TGI vs SGLang 的架构对决

预训练的成本越来越被大厂垄断，而真正让 AI 落地产生价值的，是端到端的低延迟推理环境（尤其是 Agent 运行时）。你能不能把吞吐吃满、把尾延迟压平，决定系统能不能进生产。

#### 1）为什么推理是下一片主战场

- 成本结构：训练是集中式重投入；推理是长期在线成本 + 用户体验
- 约束形态：推理更像 OS 调度问题（资源分配、抢占、缓存复用），而不是单次大算子跑多快

#### 2）vLLM 的核心突破：PagedAttention = “显存分页管理”

传统 KV Cache 的问题不是“太大”，而是“碎片化 + 难复用”。vLLM 的关键做法是把 KV Cache 拆成固定大小的 block，并用类似页表的映射把“逻辑连续”映射到“物理不连续”，从而：

- 显存利用率更高（减少空洞）
- 请求之间更容易共享前缀 KV（系统提示词/长前缀场景收益显著）
- 调度器可以更稳定地做 continuous batching

#### 2.1）Kernel/Graph 视角：为什么 vLLM “启动慢但跑起来很稳”

原始记录里有一段很典型的现象：TP=4 启动时卡在 KV Cache / CUDA Graph 阶段，随后触发共享内存广播超时（`No available shared memory broadcast block found`）。把它拆成系统步骤更好理解：

1. 权重加载：大体积权重从磁盘 → CPU 内存 → PCIe/NVLink → 多卡显存（TP=4 还需要切分与分发）
2. 分布式握手：NCCL 初始化、建立通信 ring、校准缓冲区
3. KV Cache Profiling：根据 `gpu-memory-utilization` 计算可用 KV cache，并把显存切成很多 block（PagedAttention 的“圈地”）
4. CUDA Graph Capture：为了降低 launch 开销与稳定尾延迟，提前把常见形状的执行图捕获下来

第三步与第四步的工程含义是：你用更长的冷启动时间，换取更稳定的吞吐与 tail latency（更像线上系统要的东西）。

当你在 Docker 里跑多卡推理，建议把“IPC/共享内存”当成一等公民来配置（否则你会在高负载 NCCL 或 graph capture 阶段遇到随机卡死/超时）：

- 优先保障共享内存：`--shm-size` 够大；必要时使用 `--ipc=host`（接受隔离性换稳定性）
- 接受启动慢：模型大 + LoRA + 图捕获会拉长启动时间，先别把它误判成“挂了”

#### 2.2）vLLM + LoRA 的 kernel 线索：你应该盯什么日志

原始日志里出现过 `Using default LoRA kernel configs`、以及一段包含 `cudagraph_specialize_lora` / `combo_kernels` 的配置输出。它们透露的信息是：

- vLLM 不只是“调用 PyTorch”，它在把关键路径做 kernel 级别的组合与图捕获
- LoRA 作为额外分支会影响图捕获与 kernel 选择（所以“能跑”和“跑得稳”中间有一条很长的工程鸿沟）

把 kernel 探索做成可复现实验，建议按下面顺序推进（每一步都能给第三方解释清楚“我在验证什么”）：

1. 建立基线：固定模型、固定 `max-model-len`、固定并发形态，先拿到 TTFT/TPOT/吞吐
2. 观测瓶颈归因：用系统级 profiler 看“时间花在哪类 kernel 上”（attention、GEMM、layernorm、采样、LoRA 分支）
3. 再做旋钮对比：一次只改一个变量，例如 CUDA Graph 与否、Chunked Prefill 的大小、Speculative Decoding 的策略

一个最小的观察模板（不依赖特定框架，目的只是把 kernel 时间线抓出来）：

```bash
nsys profile -t cuda,nvtx,osrt --stats=true -o vllm_report \
  python -m vllm.entrypoints.openai.api_server \
    --model <MODEL> \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --port 8000
```

你最终要回答的不是“哪个库更快”，而是两类工程问题：

- 热点 kernel 是谁：decode 阶段到底是被 attention、采样、还是某些融合 kernel 限制
- 上界在哪里：显存（KV cache）/带宽（HBM/PCIe）/并发（队列与批处理）哪个先顶住了

##### 可复现命令（模板）

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <MODEL> \
  --served-model-name <NAME> \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --port 8000
```

参数解释：

- `--gpu-memory-utilization`：给 KV Cache 留出稳定空间的关键旋钮。太大容易在峰值时 OOM，太小吞吐上不去
- `--max-model-len`：直接影响 KV Cache 上界，先按业务真实上限设置，而不是“越大越好”

<div class="embed">
  <iframe
    src="https://player.bilibili.com/player.html?bvid=BV1GWjjzfE1b&page=1&high_quality=1"
    scrolling="no"
    frameborder="0"
    allowfullscreen="true"
  ></iframe>
</div>

#### 3）TGI：Rust Router + 持续批处理的工程化优势

TGI 的价值不只在“跑得快”，更在“生产工程化”：路由、队列、并发控制、观测、降级路径都更清晰。它用 Rust + Python + gRPC 组合，在高并发时更容易把 tail latency 压住。

##### 可复现命令（Docker 模板）

```bash
docker run --rm --gpus all --shm-size 1g -p 8080:80 \
  -v <HF_CACHE>:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id <MODEL> \
  --max-input-length 4096 \
  --max-total-tokens 8192
```

参数解释：

- `--shm-size`：很多“看起来像随机崩溃”的问题，根因是共享内存不足导致的异常行为
- `--max-input-length / --max-total-tokens`：把服务侧的上界写死，避免被异常请求拖垮

<div class="embed">
  <iframe
    src="https://player.bilibili.com/player.html?bvid=BV1g14y1o7Yz&page=1&high_quality=1"
    scrolling="no"
    frameborder="0"
    allowfullscreen="true"
  ></iframe>
</div>

#### 4）SGLang：RadixAttention 的“前缀复用降维打击”

当你的业务 prompt 有大量共享前缀（系统提示词、多轮对话固定模板、few-shot 示例），SGLang 通过 Radix Tree 组织前缀 KV Cache，实现跨请求复用：

- Prefill 计算不再“每个请求从头算一遍”
- 对长前缀场景，首 token 延迟与吞吐都会出现质变

<div class="embed">
  <iframe
    src="https://player.bilibili.com/player.html?bvid=BV1CzVNzMEBP&page=1&high_quality=1"
    scrolling="no"
    frameborder="0"
    allowfullscreen="true"
  ></iframe>
</div>

#### 5）怎么选：一张工程化对照表（给决策用）

| 维度 | vLLM | TGI | SGLang |
| --- | --- | --- | --- |
| 适合的第一目标 | 快速把吞吐做上去 | 把线上稳定性与可观测做完整 | 在“强前缀复用/结构化调用”里拿到质变 |
| 部署形态 | Python 为主，集成成本低 | 生产工程化组件更齐（路由/队列/并发/观测） | 更偏“程序化编排 + 复用缓存” |
| Tail latency 控制 | 依赖你对 batching/并发形态的约束 | Router/队列策略更明确，更容易控 P99 | 前缀复用做得好时尾延迟收益明显 |
| KV Cache 策略 | PagedAttention，显存碎片与复用更友好 | 更像“生产系统”工程（具体依赖模型与配置） | Radix Tree 前缀复用，对共享模板特别强 |
| 你什么时候会换它 | 需要把工程化补齐时 | 需要更强生态/特定算子/更快落地实验时 | 业务形态不适合前缀复用或需要更成熟生态时 |

更实用的选型方式（按约束反推）：

- 先上 vLLM：你要的是最快把服务跑起来、快速压吞吐/显存，且业务 prompt 变化较大
- 先上 TGI：你最在乎的是线上稳定、观测、限流/降级与尾延迟治理，且愿意用更“服务化”的栈
- 先上 SGLang：你有大量共享前缀（系统提示词/固定模板/few-shot），或需要把推理过程写成结构化程序（Agent 工作流）

##### 推荐资源

- vLLM：https://github.com/vllm-project/vllm
- PagedAttention 论文：Efficient Memory Management for Large Language Model Serving with PagedAttention（arXiv:2309.06180）
- PagedAttention 工程解读（博文）：https://codepointer.substack.com/p/vllm-pagedattention-saving-millions
- TGI：https://github.com/huggingface/text-generation-inference
- TGI 引擎文档：https://huggingface.co/docs/inference-endpoints/en/engines/tgi
- SGLang：https://github.com/sgl-project/sglang
- SGLang 论文：SGLang: Efficient Execution of Structured Language Model Programs（arXiv:2312.07104）
- RadixAttention 细节拆解（博文）：https://tom-jerr.github.io/notes/sglang/RadixAttention%20%E4%BD%A0%E9%9C%80%E8%A6%81%E7%9F%A5%E9%81%93%E7%9A%84%E5%85%A8%E9%83%A8%E7%BB%86%E8%8A%82/

## 产出与下一步

- 训练到推理的完整闭环
- 参数与脚本可复用模板
- 继续补齐稳定性与长文本场景验证

- 业务目标与场景约束梳理
- 数据可得性与标注成本评估
- 指标定义与验证方案设计

<a id="postmortem"></a>
## 复盘要点

- 长文本场景需要严格控制 max_seq_len
- vLLM 的算子兼容性需要提前验证
- L20 上的瓶颈集中在 decode 阶段显存带宽

<a id="repro-appendix"></a>
## 可复现附录（给未来的自己和同伴）

### A）环境与版本锁定

建议把“能跑的那一次”的环境写成可复制的最小集合：

- CUDA Driver / CUDA Toolkit / cuDNN
- Python 版本、PyTorch 版本、Transformers/PEFT/DeepSpeed/vLLM 版本
- GPU 型号与数量、显存大小、互联（PCIe/NVLink）

如果你使用容器，优先记录：

- Base Image（含 tag）
- 运行时关键参数（`--gpus`、`--shm-size`、挂载目录、网络模式）

### B）训练命令参数速查（你每次改动都能解释）

下面是这类项目里“最容易变成玄学”的参数组，建议每次实验都在日志里明确写下取值与理由。

- 批量与有效学习率
  - `per_device_train_batch_size`
  - `gradient_accumulation_steps`
  - `learning_rate`（是否按有效 batch 线性缩放）
- 序列与显存上界
  - `max_seq_len` / `max_length` / `max_model_len`
  - 图像分辨率/patch size（如为多模态）
- LoRA
  - `lora_r`、`lora_alpha`、`lora_dropout`
  - `target_modules`（改了哪些层，为什么）
- DeepSpeed / ZeRO
  - ZeRO stage（1/2/3）
  - bucket/overlap（通信与吞吐的权衡点）
  - offload（CPU/NVMe）策略（以及代价）

### C）推理服务参数速查（vLLM）

把推理侧当成“线上系统”，至少记录以下旋钮：

- `--gpu-memory-utilization`：决定 KV cache 的可用空间与 OOM 风险边界
- `--max-model-len`：直接决定 KV cache 上界，建议按业务上限设置
- `--served-model-name`：便于多模型/多 LoRA 并存时做路由与回归

### D）结果记录模板（建议每次训练都填）

你可以把下面的模板贴到每次实验日志的开头，强制把“输入、改动、输出”写清楚：

- 实验 ID：`YYYYMMDD-HHMM-<短描述>`
- 目标：一句话（例如：在不 OOM 的前提下把有效 batch 提到 X）
- 数据：样本数、分辨率/长度分布、清洗规则版本
- 训练：命令 + 关键参数 + ZeRO stage + LoRA r/alpha
- 资源：GPU 数、显存、互联、CPU/RAM、磁盘吞吐
- 结果：loss 曲线摘要、关键评测、推理回归结果
- 结论：这次改动到底带来了什么（吞吐/显存/质量/稳定性）

### E）常见失败模式（快速排查清单）

- OOM 但看不懂是谁吃的：先降低 `max_seq_len`/分辨率，再动 batch；同时核对 ZeRO stage
- 吞吐忽高忽低：先排查数据加载与 CPU 解码是否抖动，再看 NCCL 拓扑与通信
- 推理能跑但延迟抖动大：先检查 KV cache 是否频繁逼近上限，再检查 batching 配置与并发请求形态
