# 序列模型从零：RNN 到 Mamba

配套书库《序列模型从零：RNN 到 Mamba》的可跑代码。用纯 TypeScript 从零拆解序列模型的演化主线：从 vanilla RNN 的长依赖困境，经 LSTM/GRU 的门控、attention 的全局连接，到 SSM / Mamba 的 O(n) 选择性扫描。

**定位**：这是训练框架书的「夹册」——前接 RNN 史前史、后接 attention 之后的 SSM。刻意不重复 autograd→nanoGPT / LLM 推理 / RLHF；attention 仅作为对照点做最小复现。

## 设计原则

- **纯算法，零运行时依赖**：只用 TypeScript 标准库，自带 `Float64Array` 自动微分引擎。不引入 PyTorch / 任何 ML 框架。
- **离线 CPU 可跑**：每个 stage 在普通笔记本上几秒到几十秒跑完，不需要 GPU / 网络 / API key。
- **确定性可复现**：所有随机走 `core/prng.ts` 的种子 PRNG，同种子同机器 → bit 级一致。
- **诚实数字**：打印的量化都由代码真算 / 真测；wall-clock 真测，复杂度用解析 MAC 计数佐证。toy 任务的绝对值偏乐观——可迁移的是机制与曲线形状（loss 下降趋势、梯度 norm 量级、耗时随 n 的标度斜率），不是 SOTA 数字。
- **反 survey**：每个 stage 都 demo 失败模式（梯度消失 / 爆炸、记忆horizon崩塌），不只 happy path。

## 跑法

依赖（`tsx` / `typescript` / `@types/node`）装好后：

```bash
npm install          # 一次性
npm run typecheck    # tsc --noEmit，确认类型干净

npm run stage01      # RNN：vanilla RNN cell + 长依赖任务
npm run stage02      # 梯度消失 / 爆炸：随 span 增长的梯度 norm
npm run stage03      # LSTM / GRU：门控如何延长记忆 horizon
npm run stage04      # 字符级语言模型：perplexity 与采样
npm run stage05      # Attention：作为对照点的最小复现
npm run stage06      # SSM：线性状态空间扫描
npm run stage07      # Mamba：选择性 SSM 与 O(n) 标度
```

每个 stage 文件顶部注释声明 seed、模型规模（参数量）、训练步数、预计耗时。

## 共享底座 `src/core/`

纯 TS、零依赖、确定性引擎，所有 stage 复用：

| 模块 | 职责 |
|------|------|
| `prng.ts` | 种子 PRNG（mulberry32）+ 采样器（normal / randint / shuffle） |
| `tensor.ts` | 极小动态计算图自动微分（add/sub/mul 广播、matmul、transpose、reshape/slice/concat、激活、softmax、cross-entropy、reductions、backward 拓扑排序） |
| `nn.ts` | `Module` 基类 + `Linear` / `Embedding` / `LayerNorm` primitive（RNN/LSTM/GRU/Attention/SSM cell 由各 stage 自己搭建并导出） |
| `optim.ts` | `SGD`（含 momentum）/ `Adam` + `clipGradNorm`（返回裁剪前 global norm） |
| `data.ts` | toy 任务生成器：copyTask / addingProblem / parityTask / delayedRecall / charSeq + 批次工具 |
| `plot.ts` | ASCII 可视化：sparkline / lineChart / histogram / heatmap / bar |
| `metrics.ts` | accuracy / perplexity / timeit（真 wall-clock）/ countMACs（解析复杂度） |

> 各 stage **不要** import 其它 stage 文件（加载即跑 `main()`）。要复用某个 cell，从教它的那个 stage 显式 export / import。
