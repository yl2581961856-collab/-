---
title: "HPC 核心概念"
description: "高性能计算基础概念与术语梳理"
---

## 关键术语

- 计算 vs 通信：算得再快也会被“搬数据”拖住，先分清瓶颈属于 compute bound 还是 bandwidth/latency bound
- Latency vs Bandwidth：小消息看延迟，大消息看带宽；优化手段往往完全不同
- Topology（拓扑）：节点内 PCIe/NVLink、节点间 IB/RoCE；决定 collective 的上限
- NUMA：CPU 插槽与内存分布不均匀，线程/内存放错位置会出现“看不见的降速”
- Collective：All-Reduce / All-Gather / Reduce-Scatter / Broadcast；训练里的通信大头
- All-Reduce：把每张卡的梯度聚合成一致结果，通常是 data parallel 的关键路径
- Ring / Tree：两类典型通信拓扑；Ring 吞吐稳定，Tree 延迟更低但更吃拓扑与实现
- Overlap（通信计算重叠）：把通信塞到计算间隙里，减少关键路径时长
- Kernel Fusion：减少 launch 与访存次数，用“更少更大的 kernel”降低开销
- Roofline：用 FLOPs 与带宽上限给性能“画边界”，避免盲目优化
- Strong / Weak Scaling：强缩放固定问题规模加机器，弱缩放问题规模随机器线性增长

## 通信模型

- 单机多卡
  - PCIe：带宽相对有限，All-Reduce 容易成为瓶颈
  - NVLink：更高带宽更低延迟，常见做法是按 NVLink domain 分层聚合
- 多机多卡
  - IB/RoCE：决定跨节点通信上限，通常比单机更容易成为瓶颈
  - Hierarchical All-Reduce：先节点内聚合，再节点间聚合，最后节点内广播
- 训练并行
  - Data Parallel：梯度 All-Reduce 为主
  - Tensor Parallel：激活/中间张量频繁 All-Gather/Reduce-Scatter
  - Pipeline Parallel：微批调度 + stage 间激活传输，关注 bubble 与带宽
- 系统视角
  - 同步通信：简单可靠，但容易被最慢节点拖住
  - 异步/松同步：吞吐更高，但一致性与收敛分析更难

## 性能指标

- 吞吐（Throughput）：tokens/s、samples/s、images/s，优先用“端到端”指标
- 延迟（Latency）：TTFT、TPOT、P99/P999，线上更关注尾延迟
- 通信占比：step time 中通信耗时比例；判断是否要换拓扑/并行策略
- 显存占用：参数/梯度/优化器状态/激活值/KV cache 分账，定位 OOM 根因
- 带宽利用率：HBM/PCIe/NVLink/IB 的有效带宽是否接近理论值
- 资源利用率：SM occupancy、tensor core 利用率、CPU 解码占用、I/O 吞吐
