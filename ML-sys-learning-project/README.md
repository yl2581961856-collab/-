# ML-sys

面向 GPU 架构、CUDA 与 Triton 的学习型仓库，围绕项目迭代沉淀基础设施、推理与内核实践。

## 项目定位
- 以项目驱动学习：用可复现实验 + 可运行代码理解 GPU 架构与 Kernel 设计
- 记录 L20 服务器的推理/训练/内核实验过程与结论
- README 持续更新为教程入口，实验细节保留在原始记录与后续文档中

## 示例项目：project-vl-inference-example
当前示例聚焦多卡 L20 上的 vLLM 推理与服务化，作为学习体系中的一个可复现实验样例。

**包含内容（基于 L20 服务器记录）**
- vLLM Docker 部署与参数选择
- Nginx 反向代理与静态报告访问
- 批量推理脚本调用与结果落盘
- OOM 与启动耗时的排查过程
- Kernel 兼容性与瓶颈分析（如 XQA / SM 架构差异）

原始记录：`L20 server.txt`

## 基线环境（L20 服务器）
| 项目 | 配置 |
|---|---|
| OS | CentOS Linux 8.5.2111 |
| Kernel | 4.18.0-553.6.1.el8.x86_64 |
| NVIDIA Driver | 580.105.08 |
| CUDA | 13.0 |
| GPU | NVIDIA L20 × 4 |
| 显存 | 46GB × 4 |
| CPU | 64 核 |

## 目录规划
- `examples/`：示例项目集合（含 app/web/worker、kernels、benchmarks、docs、assets、scripts 等）
- `L20 server.txt`：原始实验记录（完整过程/命令/日志）

## 复现实验（简版）
**vLLM Docker 启动示例**
```bash
docker run --rm -it \
  --gpus all \
  --shm-size 16g \
  -p 8000:8000 \
  -v /data/models:/models:ro \
  -v /data/logs:/logs \
  vllm/vllm-openai:latest \
  --model /models/Qwen3-VL-30B-A3B-Instruct \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.78 \
  --max-model-len 16384
```

**接口验证示例**
```bash
curl -sS -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen3-VL-30B-A3B-Instruct",
    "messages": [{"role":"user","content":"用一句话解释 MoE 的优势。"}],
    "max_tokens": 64,
    "temperature": 0.2
  }' | head -c 800
```

## 关键观察（来自 L20 实验记录）
- Qwen3-VL-30B-A3B 在长文本场景下吞吐约 865 tokens/s，可视作保底性能
- Prefix Caching 在 RAG 场景中带来约 4.2x 的加速
- L20 (SM89) 不支持 XQA 算子，H100 (SM90) 才支持
- 推理瓶颈主要集中在 Decode 阶段的显存带宽

## 数据与图表（待补）
> 这些图表会用 Pandas 生成并保存在 `examples/project-vl-inference-example/assets/plots/`，先预留位置。

- 吞吐 vs 序列长度
  - `examples/project-vl-inference-example/assets/plots/throughput_vs_seq.png`
- TTFT / ITL（P50/P95/P99）
  - `examples/project-vl-inference-example/assets/plots/latency_ttft_itl.png`
- KV Cache 利用率与并发关系
  - `examples/project-vl-inference-example/assets/plots/kv_cache_util.png`
- 显存占用 vs max_model_len
  - `examples/project-vl-inference-example/assets/plots/vram_vs_maxlen.png`
- Prefix Cache 命中率 vs 加速比
  - `examples/project-vl-inference-example/assets/plots/prefix_cache_speedup.png`

## 待做事项（从 L20 记录中整理）
- Speculative Decoding：ngram / draft model / EAGLE 对比
- Chunked Prefill：长 prompt 与短请求混合下的调参
- Prefill-Decode Disaggregation：与 PagedAttention 的关系
- 更系统的 Triton Kernel 实践与性能对比

## 更新方式
- README 作为总入口与路线图
- 原始实验记录持续保留，新的结论沉淀到 `examples/project-vl-inference-example/docs/`

