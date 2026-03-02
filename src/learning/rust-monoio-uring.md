---
title: "Rust + Monoio + io_uring"
description: "异步运行时、并发模型与性能观察"
---

## 重点结论

- io_uring 的核心价值是把“每次 I/O 都要陷入内核”的成本摊薄：更少的 syscalls、更少的上下文切换、更稳定的尾延迟
- 真正决定收益的不是“换了 API”，而是并发形态：当你有大量 in-flight I/O 时收益更明显
- 事件循环要做的是两件事：提交（submit）与收割（reap）；中间必须有明确的 backpressure，不然会把内存与延迟炸穿
- 对于模型工程常见场景（数据加载、checkpoint、日志/指标落盘），先把 I/O 形态分类：顺序/随机、小/大、缓存/直读，策略完全不同

## 核心模型

- 两个 ring：Submission Queue（SQ）与 Completion Queue（CQ）
  - SQE：你要做的 I/O 请求（read/write/open/fsync…）
  - CQE：内核完成后的回执（结果码、返回字节数等）
- 一次循环的基本流程
  - 填 SQE（准备请求）
  - `submit`（把请求交给内核）
  - `wait/peek` CQE（拿到完成事件）
  - 处理结果并回收资源（buffer、fd、状态机）
- 注册资源（可选但常见）
  - register buffers / files：减少每次请求的元数据开销，提高稳定性
  - 固定 buffer 也意味着要更严格地管理生命周期与复用策略
- 运行时视角（Monoio 的常见做法）
  - 单线程 reactor + 轻量任务调度：减少跨线程迁移与锁争用
  - 对“热点 I/O”用批量提交/批量收割：减少循环开销

## 关键踩坑

- Kernel/发行版差异：同样的 io_uring 程序在不同内核版本上行为可能不一致，线上要把内核版本当成依赖管理
- Buffer 生命周期：提交后 buffer 必须一直有效直到 CQE 返回；错误的复用会产生难复现的脏数据
- Backpressure 缺失：SQ 塞满后继续提交会导致排队、内存上涨、尾延迟飙升，必须对 in-flight 数量做硬上限
- Direct I/O 对齐：走 O_DIRECT 时经常要求对齐与 block size 约束，不满足就会 `EINVAL`
- Page Cache 抖动：大量高分辨率数据加载时，页缓存会挤压其它内存，表现为吞吐波动与抖动
- “I/O 变快但端到端没变”：最终瓶颈常在 CPU 解码/预处理、锁争用或磁盘本身，先用 profile 把关键路径圈出来
