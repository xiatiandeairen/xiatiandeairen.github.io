---
title: "PPO 从零：clip、KL 与不把策略训崩"
slug: "04"
collection: "rl-posttrain"
order: 4
summary: "把第 3 章 REINFORCE 那些零件——策略梯度、baseline——组装成能在噪声里活下来的 PPO：actor-critic、GAE、以及最关键的 clipped surrogate。讲透 clip ε / GAE λ / KL 三个旋钮各防哪种崩法，以及 clip 和 KL penalty 为什么是两套不同的限制机制。这是第 6 章 RLHF 的 RL 引擎，也是第 8 章 GRPO 去掉 critic 后剩下的那个壳。"
topics:
  - "强化学习"
tags: []
createdAt: "2026-06-21T08:00:00.000Z"
updatedAt: "2026-06-21T08:00:00.000Z"
---

REINFORCE（第 3 章那个最朴素的策略梯度）有个致命脾气：它把一整批 rollout（采样出来的轨迹）只用一次，然后丢掉；更要命的是它对学习率毫无防御——一次过大的更新能把策略推到一个再也回不来的角落。第 3 章我们靠手调小 lr 勉强让它收敛，但那是在 12 状态的玩具环境里。真到了 RLHF（第 6 章），reward 来自一个会犯错的奖励模型，策略每一步都在被噪声拉扯，「一次更新把策略训崩」不是边缘情况，是默认结局。PPO（Proximal Policy Optimization，近端策略优化）就是冲着这两个病来的，而且一次治两个：用 critic 压方差、用 clip 限步长。这一章我们把它拆到零件级跑起来，对照价值迭代算出的精确最优值 V*，看每个旋钮到底防的是哪种崩法。

配套代码在 `examples/rl-posttrain-from-scratch/src/stage04-ppo.ts`，跑 `npm run stage04`。环境和第 2 章同一个 3x4 gridworld，start@0，goal@11，中间一个 hole@7（踩进去拿负回报）。它的价值迭代精确解 `V*[start]=0.6335`——这是「最好能拿到多少折扣回报」的 ground truth，所以「PPO 学没学会」不是看曲线顺不顺眼，是看贪婪策略的回报离 0.6335 还差多少。

## PPO 的三个零件，按依赖顺序装

PPO 不是一个新算法，是三个老零件的组装。理解它最快的路径就是按依赖顺序一个个装上去，看每个零件解决了 REINFORCE 的哪个具体痛点。

### 零件一：critic —— 把 baseline 从「常数」升级成「按状态变」

第 3 章我们给 REINFORCE 加过 baseline（基线，从回报里减掉一个不影响梯度无偏性的量来压方差），但那是个全局常数。critic 是把它升级成「逐状态的价值估计」V(s)——这个状态平均能拿多少回报。于是 advantage（优势，这个动作比该状态的平均水平好多少）就有了一个随状态变化的、信息量大得多的参照。

代码里 critic 就是一张表 `V[state]`，靠普通 SGD 回归到 bootstrap 回报目标（`stage04-ppo.ts:189`）：

```typescript
// Critic regression: V(s) <- V(s) + lrCritic·(ret - V(s))
net.V[s] += lrCritic * (ret[i] - net.V[s]);
```

注意它是用回报目标 `ret` 来回归的，而 `ret = advantage + V(s)`（见下一节 GAE）。critic 越准，下一轮算出的 advantage 方差越小——这是 PPO 比 REINFORCE 平稳的一半原因。基线测试里 critic loss 末值收到 `0.0000`（`[1]` 段输出），意思是 critic 完全追上了回报，advantage 信号干净。

为什么是 actor-critic 而不是纯 value-based（像 Q-learning）？因为我们最终要的是一个能直接采样的随机策略（RLHF 里就是语言模型本身），actor（策略网络）必须显式存在。critic 只是 actor 的辅助——它不决定动作，只负责给 actor 提供低方差的「这步走得好不好」的评分。

### 零件二：GAE —— 一个旋钮在偏差和方差之间滑动

advantage 怎么算？有两个极端。一个是纯 TD（时序差分，只看一步）：`δ_t = r_t + γV(s_{t+1}) − V(s_t)`，方差小，但被 critic 的误差污染（critic 不准它就有偏）。另一个是纯 MC（蒙特卡洛，看完整条轨迹的实际回报减 V）：无偏，但把整条轨迹的随机性都灌进来，方差爆炸。

GAE（Generalized Advantage Estimation，广义优势估计）是这两者的几何加权混合，用一个旋钮 λ 连续地在它们之间滑（`stage04-ppo.ts:99`）：

```typescript
// λ=0 -> A_t = δ_t (纯 TD，全靠 critic)；λ=1 -> A_t = MC return - V(s_t)
const delta = st.reward + gamma * nextV - V[st.s];
gae = delta + gamma * lam * gae;
adv[i] = gae;
nextV = V[st.s];
```

这里有个新手必踩的坑，代码注释（`stage04-ppo.ts:94`）专门标了：episodes 是首尾拼在一个数组里的，到 `done` 处**必须**把 `gae` 和 `nextV` 重置（`stage04-ppo.ts:106`），否则优势会跨轨迹泄漏——上一条轨迹的功劳算到下一条头上，credit assignment（功劳分配）直接错乱。这是那种「不报错但悄悄训不好」的 bug，比崩溃更难查。

`[4]` 段实测把 λ 的偏差-方差权衡量化了：

| λ | 早期 adv std | 收敛轮 | critic 末 loss |
|---|---|---|---|
| 0.0（纯 TD） | 0.1605 | 5 | 0.0000 |
| 1.0（纯 MC） | 0.2528 | 4 | 0.0000 |

λ=1 的早期 advantage std 是 0.2528，明显高于 λ=0 的 0.1605——MC 把整条轨迹的噪声都灌进了优势。这正是 λ 这个旋钮控制的东西。可迁移的判断：critic 准的时候 λ 调低（省方差），critic 烂的时候 λ 调高（别被 critic 的偏差带偏）。实践默认值 λ≈0.95 就是这个折中——绝大多数时候你不会去动它，但你得知道它在权衡什么，否则 critic 出问题时你不知道该往哪拧。

> ⚠ 注意这里有个反直觉的实测结果：本环境里**先稳定收敛的反而是 λ=1（MC）**（第 4 轮 vs TD 的第 5 轮）。别把它当成「MC 更好」的普适结论——这是 12 状态确定性环境的特例：轨迹短、随机性小，MC 的方差劣势没暴露出来，反而它的无偏性占了便宜。换到真实的长 horizon、强随机环境，结论会反过来。这正是玩具环境要警惕的地方：相对趋势（λ 在偏差/方差间折中）可迁移，绝对名次不可迁移。

### 零件三：clipped surrogate —— PPO 的灵魂，一个内建的信任域

前两个零件让 advantage 又准又稳，但还没解决 REINFORCE 的核心病：rollout 用一次就扔、单步可以走太远。clipped surrogate（裁剪代理目标）一次解决这两个。

它让你在**同一批数据上反复更新好几个 epoch**（代码里 `epochs: 6`），同时拒绝让任何一次更新把策略 π 推出 π_old 的 1±ε 范围。这就是一个内建的信任域（trust region，把更新限制在「还能信任当前数据」的范围内）。看代码（`stage04-ppo.ts:167`）：

```typescript
const lo = 1 - clip;
const hi = 1 + clip;
const clamped = Math.max(lo, Math.min(hi, ratio));
// 是否落在被裁剪的（更悲观的）那一支？
const usingClip = clip > 0 && clamped * A < ratio * A;
```

目标是 `L = min(ratio·A, clip(ratio, 1±ε)·A)`。这个 `min` 取的是悲观的那一支。关键在梯度（`stage04-ppo.ts:177`）：被裁剪那一支对 θ 的梯度是 0（clamp 后的 ratio 是常数，对参数求导为零），所以一旦某个动作的概率在「被奖励的方向」上移动了 ε，目标就**变平、停止推动**。代码直接体现成「这一支 live 时就跳过这个样本这一 epoch 的 actor 更新」：

```typescript
if (!usingClip) {
  const gradScale = lrActor * A * ratio;
  // ... 更新 logits
}
```

这就是信任域的实现方式——不是加个惩罚项软性拉回，而是硬性地让超界样本的梯度归零。

这里还有个隐蔽但要命的实现细节，注释（`stage04-ppo.ts:39`）专门警告了：`logpOld`（采集数据时那个策略下的 log 概率）必须在 rollout 当下就冻结存下来（`stage04-ppo.ts:79`），因为 PPO 在一个 batch 内是 off-policy（用旧策略采的数据训新策略）的，ratio `π_new/π_old` 需要这个冻结的旧值。如果你图省事在更新后重新算 `logpOld`，ratio 永远是 1，clip 永远不触发——PPO 就**悄悄退化回了 vanilla 策略梯度**，而代码照样跑、loss 照样降，你根本看不出来。这类 bug 是 PPO 实现里最经典的坑。

## 失败模式：关掉 clip，看策略怎么断崖

讲清楚 clip 怎么设计的，最有说服力的方式是把它关掉看会怎样。`[3]` 段就是这个对照实验：同样的 seed、同样的大学习率（lrActor=0.8）、同样的 epoch 数，唯一区别是 ratio 裁不裁。

```
有 clip(ε=0.2)  峰值KL=0.2850  最大单轮回报跌幅=0.0000  回报 起点-0.738->终点0.634（单调爬升）
无 clip         峰值KL=6.2200  最大单轮回报跌幅=1.3720（@轮4）  回报 起点0.634->终点-0.738（爬升后断崖）
```

读这两行：

- **有 clip**：起点 −0.738（未训练的 argmax 策略一直撞墙），单调爬到 0.634（≈V*），最大单轮跌幅 0.0000——一路向上不回头。
- **无 clip**：第一次更新就太猛，直接把策略蹦到了目标附近（0.634），但好景不长，第 4 轮某次过大的更新把策略甩出取信区，单轮回报暴跌 1.3720，**直接落进负回报区间**（掉进 hole），终点 −0.738。

最能说明问题的是 KL（Kullback-Leibler 散度，衡量新旧策略分布差多远）峰值：无 clip 是 **6.22**，有 clip 是 **0.29**——差了一个数量级还多。clip 把每一步的 KL 死死拴住，曲线才能单调爬到 V*。这就是 PPO 内建信任域的全部价值：它不是让你训得更快，是让你**训得下去而不崩**。在玩具环境里崩了重跑就行；在 RLHF 里一次崩溃可能烧掉几千美元算力还得回滚。

> 这个对照实验的超参（seed 7、lr 0.8）是经验选出来让对比**诚实**的，不是 cherry-pick 来吹 clip 的。注释（`stage04-ppo.ts:341`）说明了：同一组设定下，clipped 跑爬到 V* 不回头，no-clip 跑半路崩盘——唯一变量就是裁不裁 ratio。

## ✦ clip 和 KL penalty 是两套机制，不是一回事

这是面试里少有人答全的点，也是真正理解 PPO 信任域的分水岭。很多人把 clip 和 KL penalty 混为一谈，觉得「都是限制策略别变太多」。它们的目标是相关的，但**作用的对象和数学量完全不同**：

- **clip 限的是 ratio**——逐样本（per-token，在 RLHF 里）地把 `π_new(a)/π_old(a)` 截在 `[1−ε, 1+ε]`。它是一个**硬性、局部、无量纲**的约束，作用在「单个动作的概率比」上。超界就梯度归零，简单粗暴。
- **KL penalty 限的是分布距离**——把 `KL(π_new ‖ π_old)`（或在 RLHF 里是 `KL(π ‖ π_ref)`，对参考策略的散度）作为一个**软性、全局、有量纲**的惩罚项加进目标，或者用作 early stopping（KL 超阈值就停止本轮更新）。它作用在「整个分布的距离」上。

一句话区分：**clip 管「单个动作别动太多」，KL 管「整个策略分布别飘太远」**。一个是逐点裁剪，一个是整体度量。它们防的崩法也不同——clip 防单次过大更新的瞬间失控（上面的断崖），KL 防多步累积的缓慢漂移（每步都合规但走了 100 步就跑没影了）。

代码里这两者都出现了，但角色不同。clip 是**训练时真正起作用的约束机制**（`stage04-ppo.ts:167`）。而 KL 在本章是作为**诊断量**被测出来的，不是惩罚项——`ppoUpdate` 在所有 epoch 跑完后用 k3 估计器算近似 KL（`stage04-ppo.ts:196`）：

```typescript
// approx KL(new||old) over the batch, k3 estimator
const logRatio = logpNew - logpOld;
klSum += Math.exp(logRatio) - 1 - logRatio; // k3
```

k3 是 OpenAI/TRL 实际在用的那个估计器——永远 ≥0、低方差。这里它的作用是「让信任域可观测」：clip 把它压小（峰值 0.0979，`[1]` 段），no-clip 让它飙到 6.22。它是你判断「这一步到底走了多远」的仪表盘。

到了第 6 章 RLHF，你会看到这两套机制**同时上**：PPO 自带的 clip 限单步 ratio，再外加一个对参考模型（SFT 模型）的 KL penalty 防止策略为了讨好奖励模型而漂离原本的语言能力（reward hacking 的一种防线）。它们不冗余，是两层不同粒度的保险。

### clip ε 旋钮：信任域松紧的直接控制

`[2]` 段扫了三个 ε，把「ε 越大信任域越松」量化出来：

| ε | 峰值 KL | clip 均值 | 收敛轮 | 与 V* 差距 |
|---|---|---|---|---|
| 0.05 | 0.0781 | 0.165 | 3 | 0.0000 |
| 0.20 | 0.0979 | 0.069 | 5 | 0.0000 |
| 0.50 | 0.2778 | 0.056 | 8 | 0.0000 |

读这张表要小心，里面有个反直觉处：

- ε 越小（0.05），KL 被拴得越紧（峰值 0.0781），收敛反而**最快**（第 3 轮）。
- ε 越大（0.50），信任域越松，峰值 KL 抬到 0.2778，收敛反而**最慢**（第 8 轮）。

为什么松反而慢？因为本环境太简单，策略要走的路很短，「拴得紧、小步快走」比「松了大步乱晃」更高效——大 ε 带来的波动拖慢了稳定收敛。但别把「ε 小更好」当普适结论：在难环境里 ε 太小会让策略动弹不得、学不动。ε=0.2 是工业界默认值，就是因为它在「敢动」和「别崩」之间取了个对大多数任务都还行的折中。这又是一个「相对趋势可迁移、绝对名次不可迁移」的例子——你能带走的是「ε 控制信任域松紧、松了波动大」，不是「0.05 最快」。

clip 均值这一列也值得看：它是「ratio 越界被裁的样本比例」。基线跑里它从早期高（策略在大动）降到后期低（收敛了不怎么动），均值 0.069（`[1]` 段）。这是个好用的健康指标——如果你的 PPO 训练里 clip fraction 一直贴着 0 或一直很高，都说明旋钮没调对（lr 太小没东西可裁 / lr 太大一直在撞墙）。

## ⚡ 前沿：GRPO 就是这个 PPO 拆掉一个零件

把 PPO 拆到零件级，最大的回报在第 8 章才兑现。DeepSeek 在训练推理模型时用的 GRPO（Group Relative Policy Optimization，组相对策略优化）本质上是 **PPO 的「去 critic 化」变体**——它把上面那个零件一（critic）整个删掉了。

回想一下 advantage 的来源：本章是 critic + GAE 算出来的。GRPO 的做法是：对同一个 prompt 采样一组（group）回答，用「这个回答的 reward 减去这组回答的平均 reward」直接当 advantage——用组内相对排名替代 critic 的绝对价值估计。于是 critic 网络、GAE、bootstrap 回归全都不需要了。在 LLM 训练里这是巨大的省钱：critic 是一个和策略模型同等量级的网络，去掉它直接砍掉一半显存和算力。

这就是为什么本章值得拆到零件级：第 8 章不需要重新讲 PPO，**只需要替换 advantage 的来源**——把「critic + GAE」这一块换成「组内归一化的 reward」，clip、ratio、epoch 复用这一整套不动。两章对比一目了然：PPO 和 GRPO 的差异就精确地落在 advantage 怎么来这一个点上，其余骨架完全共享。本章代码里 `computeGae`（`stage04-ppo.ts:99`）算出来的 `adv` 数组，到第 8 章会被一个完全不同的函数填充，但喂给 `ppoUpdate`（`stage04-ppo.ts:143`）的接口一字不变。

那为什么 PPO 当年要带 critic？因为在通用 RL（gridworld、机器人、游戏）里，reward 往往稀疏、轨迹长，没有 critic 提供的逐状态 baseline，方差大到训不动。GRPO 能扔掉 critic，靠的是 LLM 任务的特殊结构：同一 prompt 可以廉价地采很多组回答,用组内对比就能造出足够好的相对 advantage——这在「一条轨迹采样很贵」的传统 RL 里做不到。

**这里是真正的开放问题**：去 critic 化是不是免费午餐，目前没有定论，仍在研究中。critic 提供的是逐状态、低方差的 baseline；组内归一化提供的是每个 prompt 一个的、依赖采样组大小的 baseline。前者理论上方差更可控，后者省资源但在 reward 信号本身就嘈杂或组内方差小的时候可能不稳。GRPO 在数学推理这类「答案可验证、reward 干净」的任务上表现亮眼，但它是否在 reward 更软、更主观的对齐任务（比如开放式对话的偏好）上同样能替代 critic，还是个**没有通用答案的问题**——学界和各家实验室都还在 ablation。所以本章别记成「critic 过时了」，而要记成「critic 是一种 baseline 来源，GRPO 提供了另一种，各有适用边界，边界在哪还没划清」。

## 诚实声明与小结

代码末尾（`stage04-ppo.ts:405`）有句话我得原样转达：这是 12 状态确定性 gridworld + 表格 actor/critic，**绝对回报乐观、收敛快**。「5 轮收敛到 V*」「critic loss 归零」这些绝对数字到真实环境会全部失真——真实 PPO 训练动辄几千几万步，loss 永远不会真的归零。

可迁移的是**相对趋势和因果**：

1. clip 限单步 KL——这是 RLHF 复用它的根因，no-clip 会断崖（峰值 KL 6.22 vs 0.29）。
2. λ 在 critic 偏差与 rollout 方差之间折中——critic 准调低、critic 差调高。
3. clip（限 ratio，逐点硬约束）和 KL penalty（限分布距离，全局软约束）是两套机制，第 6 章会同时见到。
4. PPO 的零件可拆——第 8 章 GRPO 只换 advantage 来源。

下一步（第 5 章）是把 reward 从「环境给的真值」换成「学出来的奖励模型」，那是 RLHF 真正噪声的来源；然后第 6 章把本章的 PPO 引擎接到那个噪声 reward 上，你会亲眼看到——为什么 clip 和 KL penalty 这两道防线，到那时候才真正开始救命。
