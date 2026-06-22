# diffusion-from-scratch

《扩散与生成模型从零：在 2D toy 数据上手写 DDPM 与采样器》配套代码。

纯 TypeScript、零运行时依赖、离线 CPU 可跑、种子可复现。在 2D toy 点云上手写完整的 DDPM
(去噪扩散概率模型)：前向加噪 → score / 噪声网络 → 训练 → 多种采样器 → classifier-free
guidance → latent diffusion。**不写 U-Net / 卷积 / 注意力**——那是图像专属；本书聚焦扩散机制
本身，在 2D 上一个 MLP 就足够把 DDPM 讲透，而且你能用 ASCII 散点图一眼看出模型有没有学会
整个分布。

## 诚实声明

toy 模型只有几万参数、2D 数据、几百步训练，所以**绝对的 loss 值 / 样本质量偏乐观**；真实图像
扩散要大模型、上亿参数、海量数据。但**可迁移的是机制与曲线形状**：loss 随训练下降的趋势、采样
步数↑样本质量↑的权衡、guidance 强度↑多样性↓的取舍方向——在 toy 上观察到的相对关系与生产一致。
能 wall-clock 的数字真测；估算的标 `(est.)`。

## 运行

```bash
npm install            # 安装 dev 依赖 (tsx / typescript / @types/node)
npm run typecheck      # tsc --noEmit，确认类型干净

npm run stage01        # 前向加噪：q(x_t|x_0)，linear vs cosine 调度，ᾱ_t 单调降到 ~0
npm run stage02        # 反向过程：从 N(0,I) 一步步去噪的骨架与失败模式
npm run stage03        # score / 噪声网络：网络学的就是 ∇log p(x)
npm run stage04        # DDPM 训练：ε-prediction 目标 + loss 曲线
npm run stage05        # 采样器：DDPM vs DDIM，步数 vs 质量权衡
npm run stage06        # classifier-free guidance：guidance 强度 vs 多样性
npm run stage07        # latent diffusion：先压缩再在 latent 空间扩散
```

每个 stage 一个文件、独立可跑，几秒内输出：loss 下降 + 一张 ASCII 图 + 一个被代码触发的失败模式。

## 共享 core

`src/core/` 是全书地基，零依赖、纯 TS、种子可复现。所有 stage 的数值原语都从这里取，**任何 stage
不得各写一份数值底座**（否则梯度 bug 无法归因）：

| 文件 | 职责 |
|------|------|
| `core/rng.ts` | 种子 PRNG（mulberry32 + Box–Muller）。全书所有 "noise" 都来自这里 |
| `core/tensor.ts` | 极小张量 + 反向自动微分（真梯度，非黑箱）+ `numericalGradCheck` |
| `core/nn.ts` | `Linear` / `MLP` / `SinusoidalEmbedding`，统一去噪网络形状 |
| `core/optim.ts` | `SGD` / `Adam`（DDPM 默认） |
| `core/schedule.ts` | 噪声调度 `linearSchedule` / `cosineSchedule`，前向/反向共用常数表 |
| `core/data.ts` | toy 2D 数据：`twoMoons` / `mixtureOfGaussians` / `swissRoll2D` / `spiral` |
| `core/plot.ts` | ASCII 可视化：散点 / 热图 / loss 曲线 / 直方图 |

写新 stage 的约定见 `src/core/*` 各文件头注释。注意：**不要在一个 stage 里 import 另一个 stage
文件**（stage 文件加载即跑 `main()`）。
