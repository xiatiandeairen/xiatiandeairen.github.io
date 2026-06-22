# math-for-builders-from-scratch

《开发者的数学与物理》配套教学代码仓。从零用 TypeScript 把开发者真正用得上的数学/物理拆开重写：每个 stage 都是**可跑的真算法**，不是 survey。

设计原则：

- **纯算法、离线 CPU 可跑**：零运行时依赖，不需要任何 LLM / API key / 网络。
- **确定性**：所有随机都走 `src/core/rng.ts` 的种子 PRNG，同 seed bit-for-bit 可复现。
- **诚实数字**：打印的量化要么代码真算出来，要么真 wall-clock 测出来；估算标 `(est.)`；toy/合成数据会说明绝对值偏乐观、能迁移的是相对趋势/加速比。
- **讲 why，demo 失败模式**：注释解释为什么这样做、不变量、失败模式；每个 stage 不只跑 happy path，还演示它什么时候出错。

## 共享底座 `src/core/`

每个 stage 复用 core，不重造：

- `core/rng.ts` — 种子 PRNG（mulberry32）+ `sampleUniform` / `sampleNormal` / `shuffle`。
- `core/linalg.ts` — `number[]` / `number[][]` 上的 `dot/add/scale/norm/normalize` 与 `matmul/transpose/identity/matVec`，可读优先、不依赖 BLAS。
- `core/stats.ts` — `mean/variance/std/quantile/histogram`。
- `core/plot.ts` — `asciiBar` / `asciiLine`，把数值画成 console 可读图。

## 跑法

```bash
npm install            # 装 tsx / typescript / @types/node (devDeps)
npm run typecheck      # tsc --noEmit

npm run stage01        # 贝叶斯与基率陷阱
npm run stage02        # 统计：抽样分布与 CLT
npm run stage03        # 线性代数：matmul / 投影 / PCA
npm run stage04        # 最优化：梯度下降 / 动量
npm run stage05        # 离散数学：图 / 组合 / 递推
npm run stage06        # 数值计算：浮点与数值稳定性
npm run stage07        # 信息论：熵 / 交叉熵 / KL / 编码
npm run stage08        # 密码学数论：模幂 / 素性 / RSA
npm run stage09        # 可计算性：状态机 / 停机 / 复杂度
npm run stage10        # 傅里叶：DFT 即 matmul / FFT / 频谱
npm run stage11        # 统计力学：蒙特卡洛 / 相变
npm run stage12        # 硬件与性能：缓存 / 局部性 / roofline
```

每条 stage 也可直接 `npx tsx src/stage01-bayes.ts` 跑。

> 注：stage01 已是完整可跑示例；stage02–stage12 当前是已接好 core 接口、可运行的脚手架占位（header 注释写明各自 SCOPE / 失败模式 / core 契约），由各 stage 作者按注释填充。
