# 机制可解释性从零

> 在一个玩具 Transformer 里把电路拆开。

配套书《机制可解释性从零：在一个玩具 Transformer 里把电路拆开》的可运行代码仓。

**专家级、可跑、反 survey**：代码真能跑，机制是真的，打印的数字是代码真算/真测出来的。纯算法、离线 CPU 可跑、确定性可复现 —— 不需要任何 LLM / API key / 网络。

## 核心理念

我们训练一个几千参数的 `TinyTransformer`（1-2 层、1-4 头、d_model 16-32）去解一个**有精确地面真值**的合成任务（模加法 / 复制 / induction / skip-trigram），然后用一整套可解释性工具把它的内部电路拆开。

因为模型小、任务有 oracle（精确正确答案），每个"我们发现了 X 电路"的论断都能用真实数字验证，而不是看着 attention 图讲故事。

### 诚实边界（贯穿全书）

toy 模型小、任务合成，所以**绝对数字偏乐观**（loss / 探针准确率 / patch 恢复率），不外推到大模型。可迁移的是**机制与曲线形状**：

- induction head 的 QK 对角化模式
- logit lens 逐层逼近答案
- activation patching 把因果责任集中到少数组件
- SAE 在叠加（superposition）下拆出可命名特征

这些关系在任何尺度成立，绝对值不成立 —— 书中逐处诚实标注。

## 跑法

```bash
npm install          # 装 dev 依赖（tsx / typescript / @types/node）
npm run typecheck    # 类型检查，应 0 error

npm run stage01      # 训练研究对象 + 引擎正确性（gradCheck）+ 失败模式
npm run stage02      # 注意力可视化（ASCII 热图）
npm run stage03      # 线性探针（probing）+ 随机基线对照
npm run stage04      # logit lens：中间残差流投到 unembedding
npm run stage05      # activation patching：因果定位
npm run stage06      # induction head：QK 对角化模式
npm run stage07      # 稀疏自编码器（SAE）：拆叠加特征
npm run stage08      # 完整定位流水线：多工具交叉印证
```

每个 stage 离线 CPU 几秒跑完，打印实测数字，并**必演一个失败模式**（不只 happy path）。

## 架构

共享底座在 `src/core/`（零运行时依赖，全书共用一个随机源）。每个 `src/stageNN-*.ts` 是一章的可运行实验，只依赖 `core`，互不 import。

### `src/core/` 模块契约

| 模块 | 职责 |
|------|------|
| `rng.ts` | `mulberry32` 种子 PRNG（全书唯一随机源）+ `gaussian` / `sampleCategorical` / `argmax` / `argmaxRandomTie` / `shuffle`。禁用 `Math.random`。 |
| `autograd.ts` | 极小张量 autograd 引擎：`Tensor` + 反向传播；算子 `matmul/add/mul/transpose/softmax/logSoftmax/relu/gelu/layerNorm/embeddingLookup`；`noGrad()` 推理上下文；`gradCheck()` 有限差分对拍（引擎正确性的地基）。 |
| `nn.ts` | 从 core 算子拼出的层 + **hook 系统**：`Module` / `Linear` / `Embedding` / `LayerNorm` / `MultiHeadSelfAttention`（导出每头注意力矩阵）/ `TransformerBlock` / `TinyTransformer`。每个 `forward(x, hooks?)` 可在命名激活点（`resid_pre/attn_out/resid_mid/mlp_out/resid_post/head_z`）读取或替换激活 —— 这是 logit lens / patching / SAE 的统一抓取点。 |
| `optim.ts` | `SGD` / `AdamW` / `clipGradNorm` / `cosineWarmup` / `crossEntropy`，用于训练研究对象。 |
| `tasks.ts` | toy 研究对象数据生成器（地面真值可精确计算）：`modAdd(p)` / `copyTask` / `inductionTask` / `skipTrigram`。每个给 `vocab` / `makeBatch` / `oracle`（精确正确答案）/ `scorablePositions`。 |
| `model_zoo.ts` | `trainToyModel(task, cfg)`：确定性训练并**缓存**权重（同种子复现），各 stage 直接加载同一研究对象，保证各章定位结果可互相印证。 |
| `interp.ts` | 可解释性工具箱：`runWithCache` / `logitLens` / `linearProbe`（带随机基线）/ `patch`（activation patching）/ `ablate`（零/均值消融）/ `trainSAE` / `saeEncode` / `topSAEfeatures`。 |
| `viz.ts` | ASCII 可视化（核心交付形态）：`asciiHeatmap` / `asciiBar` / `asciiSparkline` / `asciiScatter` / `heatmapDiff`。 |

### 为什么是 hook 系统

全书的可解释性技术（缓存、logit lens、probing、patching、ablation、SAE）都建立在 `nn.ts` 里**一个** hook 机制上：

- **缓存** = "观察并存下来"的 hook
- **patching** = "观察并替换成另一次运行的激活"的 hook
- **ablation** = "观察并替换成零/均值"的 hook

因为共用一套 hook 命名（如 `blocks.0.resid_post`），一个组件在不同工具里指的是同一个东西 —— 这种一致性正是"patch 是否落在 lens 指出的地方？"这类跨技术交叉印证能成立的前提。

## 质量保证

无 test 框架。质量靠：

- `autograd.ts::gradCheck()` —— 有限差分对拍解析梯度，引擎正确性的地基（stage01 对全模型跑一次）
- `npm run typecheck` —— TypeScript strict + `noUnusedLocals/Parameters`
- 每个 stage 打印实测数字 + 必演失败模式
