# 个人维护网站

这个仓库用于维护一个个人网站，目标是持续迭代知识体系，并记录实践中踩过的坑与复盘总结。

当前阶段聚焦 AI-Infrastructure 能力栈：从算子与 CUDA C++ 开始，逐步过渡到 PyTorch API 的工程化理解；推理侧重点掌握 PagedAttention、Speculative Decode、vLLM 等关键机制，并围绕大模型架构与系统瓶颈建立完整视图；微调侧覆盖 LoRA、DeepSpeed 并行等实践路径。

## 维护方式

- `src/posts/` 用于阶段性总结与专题文章
- `learning/`、`projects/`、`explorations/` 用于主题笔记与实践归档

## 本地开发

```bash
npm install
npm run dev
```

访问：

- http://localhost:8080

## 构建

```bash
npm run build
```

输出目录为 `dist/`。
