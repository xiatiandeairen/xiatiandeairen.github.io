# RL 与后训练 从零：PPO / GRPO / RLHF / DPO

从零用 TypeScript 手写强化学习与大模型后训练的核心算法。**纯算法、离线 CPU 可跑、零运行时依赖**（无 LLM / 无 API key / 无网络）。种子 PRNG 驱动，run-to-run **bit-for-bit 可复现**。

每个 stage 打印的奖励 / 胜率 / KL / regret / 准确率，**都由代码实时计算实测**，无硬编码 assert。曲线以文本表 / ASCII sparkline 呈现。

## 设计原则

- **可验证地面真值**：所有玩具环境（bandit / gridworld / 偏好世界）的最优解可精确计算（最优臂、价值迭代 V\*、`trueRewardFn`），曲线一律对照它，"学到了没"是测出来的事实。
- **必演失败模式**：不只 happy path。greedy 被锁死、无 baseline 的高方差、PPO 去裁剪后过冲、RM 吃标注噪声、RLHF 的 reward hacking —— 都直接量化打印。
- **诚实数字**：合成环境远比真实 RLHF 简单，**绝对值偏乐观**；可迁移的是**相对趋势**（KL 系数↑⇒漂移↓但学得慢；噪声↑⇒RM 准确率↓；reward hacking 出现在代理奖励与真奖励解耦处）。

## 运行

```bash
npm install          # 仅装 devDependencies（tsx / typescript / @types/node）
npm run typecheck    # tsc --noEmit，确认类型干净
npm run stage00      # 或直接：npx tsx src/stage00-landscape.ts
```

每个 stage 都可独立直跑，加载即跑 `main()`：

| 命令 | 文件 | 主题 |
|------|------|------|
| `npm run stage00` | `src/stage00-landscape.ts` | 地形图 + core 自检 + 复现性证明 |
| `npm run stage01` | `src/stage01-bandit.ts` | Bandit：探索 vs 利用，regret 实测 |
| `npm run stage02` | `src/stage02-gridworld-value.ts` | Gridworld：信用分配 + 价值自举，对照精确 V\* |
| `npm run stage03` | `src/stage03-reinforce.ts` | REINFORCE：策略梯度 + baseline 降方差 |
| `npm run stage04` | `src/stage04-ppo.ts` | PPO：裁剪代理目标 = 内建信任域 |
| `npm run stage05` | `src/stage05-reward-model.ts` | Reward Model：从偏好对学奖励（Bradley–Terry） |
| `npm run stage06` | `src/stage06-rlhf.ts` | RLHF：PPO×RM + KL 拴参考策略，量出 reward hacking |
| `npm run stage07` | `src/stage07-dpo.ts` | DPO：直接偏好优化（无 RM、无采样、闭式损失） |
| `npm run stage08` | `src/stage08-grpo.ts` | GRPO：组内相对优势替代价值网络 |
| `npm run stage09` | `src/stage09-rlvr-reasoning.ts` | RLVR：可验证奖励驱动推理（R1 配方） |

## 共享底座 `src/core/`

零依赖的"确定性实验台"，全书一个随机源。

- **`rng.ts`** — `mulberry32` 种子 PRNG（全书唯一随机源，禁用 `Math.random`）、`gaussian`、`sampleCategorical`、`argmax` / `argmaxRandomTie`。
- **`probability.ts`** — `softmax` / `logSoftmax` / `logProb`（数值稳定的 log π）、`entropy`、`klCategorical`（精确分布 KL）、`klFromLogprobs`（PPO/RLHF 的 k1/k3 采样估计量并列对比）。
- **`envs.ts`** — 离线确定性玩具环境：`makeBandit`（k 臂高斯，可算 regret）、`makeGridworld`（带洞/目标/可选风，已知 V\*）、`makeContextualBandit`（state→最优动作有 ground truth）。
- **`preference.ts`** — 合成偏好世界：已知 `trueRewardFn`、`generatePairs`（Bradley–Terry 标注 + 可注入翻转噪声）、`goldRanking`（评估地面真值）。RM/DPO/GRPO 共用。
- **`metrics.ts`** — 诚实数字工具：`RunningMean`、`regret`、`winRateVsRef`、`klToRef`、`rewardModelAccuracy`、`pearson` / `spearman`、`movingAvg`、`asciiSparkline`、`timeIt`、`optimalValueIteration`。

> 注：stage01–09 当前为**可运行骨架**（已实现真实算法 + 实测打印，但范围聚焦核心机制）；各章配套文本会进一步展开（UCB/Thompson、GAE/critic、IPO/KTO、adaptive-KL 等）。
