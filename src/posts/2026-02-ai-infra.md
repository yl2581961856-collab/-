---
title: "AI 基础设施与硬件适配阶段"
description: "NPU 适配与并发模型的核心观察"
date: 2026-02-15
category: "部署"
tags: ["Ascend 910B", "ASR", "Rust", "Monoio"]
---

这阶段的核心目标是把模型部署从“能跑”推进到“可维护、可复现”。

## 关键工作

- 完成 ASR 模型在 Ascend 910B 上的推理适配
- 梳理驱动与 CANN 版本依赖
- 设计 Rust + Monoio 的高并发网关结构

## 下一步

- 归纳性能瓶颈与测量方法
- 整理部署脚本并归档
