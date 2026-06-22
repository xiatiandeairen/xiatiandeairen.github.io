# moe-from-scratch

《混合专家 MoE 从零做透》配套教学代码。用纯 TypeScript 把 Mixture-of-Experts 拆透：从一个最小 MoE 层，到 Top-k 路由、不可导门控的处理、负载均衡辅助损失、容量丢弃（capacity drop）、专家坍塌（expert collapse），最后到一个能跑通的 capstone。

**特点**：纯算法、零运行时依赖、离线 CPU 可跑、确定性可复现（所有随机源走种子 PRNG）。每个 stage 在 CPU 上几秒收敛。

## 跑法

```bash
npm install          # 仅 devDependencies：tsx / typescript / @types/node
npm run typecheck    # tsc --noEmit，校验全仓类型

npm run stage01      # 最小 MoE 层：gate + experts，稀疏 vs dense
npm run stage02      # Top-k 路由：每 token 只激活 k 个专家
npm run stage03      # 不可导路由：argmax 不可导，门控权重怎么把梯度引回 router
npm run stage04      # 负载均衡损失：Switch/GShard 风格 aux loss 压平利用率
npm run stage05      # 容量丢弃：每专家容量上限，溢出 token 被丢，drop rate 实测
npm run stage06      # 专家坍塌：路由退化成常数，熵塌缩 / CV 飙升，及缓解
npm run stage07      # capstone：把以上机制合到一个可训练的 MoE 上
```

> stage 文件加载即跑 `main()`，互不 import。每个 stage 顶部注释声明 seed 与预期输出范围，便于 CI diff。

## 共享底座 `src/core/`

纯 TypeScript、零依赖、Node 原生可跑，所有 stage 复用：

| 模块 | 内容 |
|------|------|
| `prng.ts` | 种子 PRNG（mulberry32）：`rng(seed)` → `{next, normal, int, shuffle, pick}`。全书唯一随机源，禁用 `Math.random`。 |
| `tensor.ts` | 极小 reverse-mode autograd（micrograd 风格，支持向量/矩阵）：`Value` + `backward()` + `gradCheck()` 数值对拍。 |
| `nn.ts` | 层原语：`Linear` / `Embedding` / `LayerNorm` / `Sequential` / `Expert`（2 层 MLP）/ `crossEntropy`，均基于 tensor。 |
| `optim.ts` | `SGD(lr, momentum)` 与 `Adam(lr, b1, b2, eps)`，`step()` / `zeroGrad()`。 |
| `data.ts` | seeded toy 数据：`makeClusters` / `makeModularTask` / `makeTokenStream`，均暴露 ground-truth 分组以对照路由是否学到语义分工。 |
| `viz.ts` | ASCII 可视化：`plotLoss` / `bar` / `heatmap` / `hist`。 |
| `metrics.ts` | MoE 度量：`routingEntropy` / `expertUtilization` / `loadBalanceLoss` / `activatedFLOPs` / `denseFLOPs` / `coefficientOfVariation`。 |

底座自检：`npx tsx src/core/_smoke.ts`（gradCheck + 优化器下降 + 指标手算对照 + 数据确定性）。

## 诚实数字约定

打印的所有量化都由代码真算/真测：

- **FLOPs** 是按 matmul 维度真实计数的 MAC（乘加），量化的是**算法工作量**（稀疏 vs dense 的 `≈ k/E` 比例可迁移），不是吞吐；wall-clock 受硬件/kernel 影响，toy 规模偏乐观。
- **收敛曲线 / 利用率 CV / 路由熵** 是 toy、合成数据上的实测值，**绝对值偏乐观**；可迁移的是**机制与曲线形状**（稀疏省算力比例、aux loss 压平利用率 CV、坍塌时熵塌缩），不是绝对精度。
- 估算量会显式标 `(est.)`。
