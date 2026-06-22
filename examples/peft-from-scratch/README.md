# peft-from-scratch

配套书库《高效微调 PEFT 从零：在 TypeScript 里手写 LoRA / QLoRA / Adapter》的可运行代码。

**纯算法、离线 CPU 可跑、种子确定可复现。** 不需要任何 LLM / API key / 网络。所有随机性来自 `src/core/prng.ts` 的全局种子 PRNG；在每个 stage 顶部 `seed(1234)` 即可把输出复现到末位小数。

## 设计原则

- **真训练**：每个 stage 用 `src/core` 的极小自动微分张量引擎，在玩具任务上真做几十~几百步梯度下降，打印真实的 loss 下降曲线。
- **诚实数字**：可训练参数占比、ΔW 低秩性、合并前后等价误差等都是代码真算出来的。能 wall-clock 的真测；估算量（如显存）标 `(est.)`。
- **toy-scale 免责声明**：玩具尺度下绝对值（loss 数值、显存 MB、收敛步数）偏乐观。可迁移的是**机制与曲线形状**：可训练参数占比的数量级、loss 曲线形状、ΔW 低秩结构、合并等价性。每个 stage 末尾固定打印该提示。

## 运行

需要 Node 18+。先安装开发依赖（tsx / typescript / @types/node）：

```bash
npm install
```

类型检查：

```bash
npm run typecheck   # tsc --noEmit
```

按章运行：

```bash
npm run stage01   # 训练并 dump 一个玩具基座（task A），后续章节从它出发
npm run stage02   # 全量微调（full fine-tuning）基线：迁移到 task B，记录可训练参数 = 100%
npm run stage03   # LoRA：冻结基座，只训练低秩 BA 增量
npm run stage04   # rank 扫描：r 取不同值，看容量 vs 参数量 trade-off
npm run stage05   # Adapter：瓶颈式插入层
npm run stage06   # Prefix-Tuning：可训练虚拟 token / 前缀 KV
npm run stage07   # QLoRA：量化冻结基座 + LoRA，看显存进一步下降
npm run stage08   # 合并：把 LoRA 增量并回基座，验证合并前后输出等价
```

每个 stage 是独立的 `tsx` 入口，加载即跑 `main()`。**不要相互 import stage 文件。** 共享逻辑全在 `src/core`。

## 共享 core 契约

| 模块 | 职责 |
|------|------|
| `core/prng.ts` | 全局种子 PRNG：`seed/uniform/normal/randint/shuffle` + 初始化 std |
| `core/tensor.ts` | 极小反向自动微分张量引擎（1D/2D），支持 `requires_grad` 冻结 + `numericalGradCheck` |
| `core/nn.ts` | `Linear/Embedding/LayerNorm/Sequential/MultiHeadAttention/TransformerBlock`；`Module.trainable()/frozen()/numParams()/freeze()` |
| `core/optim.ts` | `SGD(momentum)` / `Adam`，只更新 `requires_grad=true` 的参数，支持 paramGroups |
| `core/data.ts` | 种子确定的玩具任务（copy/reverse/sort/分类）+ 「预训练 A → 微调 B」成对生成器 + batch 迭代 |
| `core/viz.ts` | ASCII 可视化：`lossCurve/sparkline/histogram/heatmap/bar` |
| `core/mem.ts` | 显存估算器 `estBytes`（纯算术，不真分配）：全量 vs PEFT 对比 |
| `core/checkpoint.ts` | 内存内 `dump/loadBase/freeze`：冻结基座 = 共享只读权重 |

核心不变量：梯度 `+=` 累积（每步前必须 `zeroGrad`）；冻结叶子（`requires_grad=false`）参与前向、传播梯度给输入、但自身 `.grad` 永远为 0、优化器永不移动它——这正是 PEFT 便宜的原因。
