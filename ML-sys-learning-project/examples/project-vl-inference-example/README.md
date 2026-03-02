# project-vl-inference-example - L20 推理与 Infra 基线

示例项目，用于整理和复现实验：L20 多卡推理、vLLM 部署、性能瓶颈分析与后续 Kernel 实践。

## 目录结构
- `app/`：服务与接口代码
- `web/`：前端静态页面
- `worker/`：任务执行与队列消费
- `kernels/`：CUDA / Triton 内核实验
- `benchmarks/`：性能测试脚本与结果
- `configs/`：配置文件与启动参数
- `data/`：示例输入或小样本（大数据不入库）
- `scripts/`：一键运行与辅助脚本
- `docs/`：项目内文档与实验记录
- `assets/plots/`：图表输出（Pandas 生成）

## 复现实验（占位）
- vLLM 启动参数：待整理
- API 调用示例：待整理
- 关键性能结论：待整理

## 图表占位（Pandas 输出）
- `assets/plots/throughput_vs_seq.png`
- `assets/plots/latency_ttft_itl.png`
- `assets/plots/kv_cache_util.png`
- `assets/plots/vram_vs_maxlen.png`
- `assets/plots/prefix_cache_speedup.png`

## TODO
- 梳理 app/web/worker 的运行方式与配置说明
- 补充完整运行说明与依赖
- 持续沉淀实验结论到 `docs/`

