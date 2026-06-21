# tiny-dl-from-scratch

配套书库《训练框架从零：从 autograd 到 nanoGPT》的可跑代码仓。

从一个标量 autograd 引擎出发，自底向上搭出张量引擎、网络层、优化器、训练循环，最后拼出一个能在
CPU 上分钟级过拟合玩具语料的 nanoGPT。**纯算法、离线、零运行时依赖**：不需要任何 LLM / API key /
网络 / GPU。所有随机性走显式 seed，结果 bit-for-bit 可复现。

## 设计原则

- **专家级、可跑、反 survey**：每个 stage 都是能 `npx tsx` 直接跑出真实数字的程序，不是伪代码。
- **诚实数字**：打印的量化要么真测（wall-clock）、要么真算（loss / grad-check 误差）。估算值标 `(est.)`。
- **玩具数据的诚实边界**：spiral / moons / 玩具语料的收敛**绝对值偏乐观**；可迁移的是**趋势**——
  train loss 单调下降、grad-check 误差 < 1e-6、过拟合小数据时 train loss→0 而 val 不降。
- **必演失败模式**：每个 stage 不只跑 happy path，还会复现一个典型 bug / 不稳定现象。

## 共享底座 `src/core/`

后续 stage 一律复用，不重造：

| 模块 | 职责 |
|------|------|
| `core/rng.ts` | 种子 PRNG（mulberry32）+ `randn` / `shuffle` / `kaiming` / `xavier` |
| `core/autograd.ts` | 计算引擎单一真相源：`Value`（标量）+ `Tensor`（n 维）+ 全套算子 + `backward()` + `noGrad()` |
| `core/nn.ts` | `Module` 基类 + `Linear` / `Embedding` / `LayerNorm` / `Dropout` / `Sequential` |
| `core/optim.ts` | `SGD` / `Adam` / `AdamW` + `clipGradNorm` + `cosineWarmup` |
| `core/data.ts` | `makeSpiral` / `makeMoons` / `charDataset`（玩具语料，CPU 分钟级过拟合） |
| `core/metrics.ts` | `crossEntropy` / `mseLoss` / `accuracy` / `gradCheck` / `timeIt` / `lossCurveAscii` / `paramCount` |

## 跑法

```bash
npm install              # 装 tsx / typescript / @types/node（仅 devDeps）
npm run typecheck        # tsc --noEmit，确认类型干净

npm run stage01          # 标量 autograd：手搭计算图 + 反向传播
npm run stage02          # 张量引擎：Float64Array + 广播 + matmul 反向
npm run stage03          # 网络层：Linear / LayerNorm / Embedding
npm run stage04          # 优化器：SGD vs Adam vs AdamW + 梯度裁剪 + LR 调度
npm run stage05          # 训练循环：spiral 分类，loss 曲线 + 过拟合演示
npm run stage06          # 注意力：scaled dot-product + causal mask
npm run stage07          # Transformer block：attention + MLP + 残差 + LayerNorm
npm run stage08          # nanoGPT：字符级语言模型，过拟合玩具语料 + 采样
```

每个 stage 是独立可执行文件，`main()` 在加载时自动跑。**不要在一个 stage 里 import 另一个
stage**（会触发对方的 `main()`）；要复用逻辑就放进 `src/core/`。

## 复现性

同机同 seed，跨次运行输出一致。所有随机来源（初始化、shuffle、dropout mask、batch 采样）都从
`core/rng.ts` 的种子 PRNG 取，禁止直接 `Math.random()`。
