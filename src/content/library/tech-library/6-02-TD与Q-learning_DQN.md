---
title: "TD 与 Q-learning / DQN"
slug: "6-02"
collection: "tech-library"
group: "强化学习"
order: 6002
summary: "强化学习域 · 深化篇 面向有工程经验、要把 RL（含 LLM 对齐用 RL）吃透成专家的读者。术语保留英文。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-14T20:02:17.000Z"
updatedAt: "2026-06-14T20:02:17.000Z"
---
> 强化学习域 · 深化篇
> 面向有工程经验、要把 RL（含 LLM 对齐用 RL）吃透成专家的读者。术语保留英文。

---

## TL;DR（先给结论）

- **TD（Temporal-Difference）是 RL 的中枢思想**：用「下一步的估计」去更新「当前步的估计」（bootstrapping），不必等到 episode 结束拿到真实 return。它把 Monte Carlo 的「采样」和 Dynamic Programming 的「bootstrapping」缝在一起。
- **Q-learning 是 TD 的 control 版 + off-policy**：直接学最优动作价值 `Q*`，更新里用 `max_a' Q(s',a')`（贪心的 target），而 behavior policy 可以是带探索的 ε-greedy。Watkins & Dayan 1992 证明了它在表格情形下 w.p.1 收敛到 `Q*`。
- **DQN = Q-learning + 神经网络 + 两个稳定性补丁**：(1) **experience replay** 打散样本相关性、提高数据复用；(2) **target network**（冻结的 `θ⁻`）让 TD target 在一段时间内不动，缓解「追自己尾巴」的发散。这是 2013 NeurIPS workshop → 2015 Nature 的工作，让 RL 第一次从原始像素端到端打 Atari 到人类水平。
- **核心失败模式**：maximization bias（`max` 操作系统性高估）→ Double DQN 解；deadly triad（function approximation + bootstrapping + off-policy 三者同时出现易发散）；replay/target 调参敏感。
- **可运行 demo 两个**：(1) 纯 numpy 表格 Q-learning 解 GridWorld，看 reward 上升 + 学出最短路径；(2) 基于 cleanrl 风格的最小 PyTorch DQN 解 CartPole-v1，看 episodic return 爬到接近 500。**两个 demo 都设计为可运行，请在你的环境验证。**
- **和 LLM 对齐的连接**：经典 RLHF 用的是 PPO（policy gradient 家族，第 4-5 章），**不是** DQN。但 (a) 价值函数 / advantage 估计的 TD 思想贯穿 PPO 的 critic 与 GAE；(b) 近年出现把 token-level RLHF 显式建成 Q-learning / value-based 形式的工作（如 ILQL、token-level DPO 的 Q 视角）。把 TD/Q 吃透，是读懂 critic、value baseline、reward-to-go 这些 RLHF 组件的前提。本章末有一节专门讲这个连接，并明确标注哪些是 fact、哪些是「待核」。

---

## 前置依赖（读这章前你该有的东西）

**概念前置**
- MDP 五元组 `(S, A, P, R, γ)`、policy `π`、state value `V^π`、action value `Q^π`、Bellman equation 与 Bellman optimality equation。（第 1 章内容；本章会快速回顾 `Q*` 的定义，但不重推。）
- 知道 return `G_t = Σ_k γ^k r_{t+k+1}` 的定义。

**工程前置（跑 demo 用）**
- Python 3.9+。
- `numpy`（表格 demo 唯一依赖）。
- DQN demo：`torch`（CPU 即可，CartPole 不需要 GPU）、`gymnasium`、`stable-baselines3`（仅用它的 `ReplayBuffer`，避免重写 buffer 的边界 bug）。
  ```bash
  pip install "gymnasium[classic-control]" torch numpy stable-baselines3
  ```
- 想跑官方原版对照：`pip install cleanrl`（可选，本章 demo 不强依赖）。

**读法建议**
- 想最快建立直觉：直接跳到「§4 表格 Q-learning demo」跑起来，再回头看推导。
- 想做 LLM 对齐：§1（TD 思想）+ §6.4（与 RLHF 的连接）+ §3.3（overestimation / Double DQN）这三段最高优先。

---

## 模板六段

> 六段：① 设计考古 ② 核心机制（源码精读）③ 可运行 demo ④ 方案对比 ⑤ 失败模式与真坑 ⑥ 与 LLM 对齐的连接。

---

## §1 设计考古：MC → TD → Q-learning → DQN

### 1.1 一条主线：怎么不等到底就更新

强化学习要解的核心估计问题是：**给定一个 policy，状态/动作值是多少**（prediction），以及**怎么改进 policy**（control）。两个极端的做法：

- **Dynamic Programming (DP)**：已知环境模型 `P, R`，用 Bellman 方程做全宽度（full backup）迭代。问题：要模型，且每步要遍历所有后继状态，state 空间大就炸。
- **Monte Carlo (MC)**：不用模型，跑完整 episode，用真实 return `G_t` 当 target 更新 `V(s) ← V(s) + α[G_t − V(s)]`。问题：必须等 episode 结束；高方差（一条轨迹的随机性全压在 `G_t` 上）；不能用于持续型（non-episodic）任务。

**TD 的洞察（Sutton 1988）**：不必等到底。用「当前奖励 + 对下一状态的现有估计」当 target：

```
TD(0):  V(S_t) ← V(S_t) + α [ R_{t+1} + γ V(S_{t+1}) − V(S_t) ]
                              └────────── TD target ──────────┘
                            └──────────── TD error δ_t ───────────┘
```

这里 `R_{t+1} + γ V(S_{t+1})` 既**采样**了一步真实交互（像 MC），又**bootstrap** 了 `V(S_{t+1})` 这个现有估计（像 DP）。代价是引入 bias（你 bootstrap 的是一个还没收敛的估计），收益是大幅降低 variance、可在线、可用于持续任务。这正是 bias-variance trade-off 的经典化身。

> 【设计考古 · fact】Temporal-difference learning 由 Richard S. Sutton 在 1988 年论文 *"Learning to predict by the methods of temporal differences"*（*Machine Learning*）系统化提出，思想根源可追溯到 Arthur Samuel 的跳棋程序与 Sutton–Barto 的 actor–critic。来源：Wikipedia "Temporal difference learning"（已 WebFetch 核实，见文末 Sources）。

### 1.2 从 prediction 到 control：Q-learning 的诞生

TD(0) 学的是 `V`，做 control 不方便（要 `V` + 模型才能挑动作）。直接学 **action value** `Q(s,a)` 就能 `argmax_a Q(s,a)` 选动作，不需要模型。

**Q-learning（Watkins 1989）** 给出 control 更新：

```
Q(S_t, A_t) ← Q(S_t, A_t) + α [ R_{t+1} + γ max_a' Q(S_{t+1}, a') − Q(S_t, A_t) ]
                                            └── 注意这里是 max ──┘
```

关键点：target 里用的是 **`max_a' Q`**（即「假设下一步走最优」），而**实际**用来产生 `A_t` 的 behavior policy 可以是别的（通常 ε-greedy 带探索）。target policy（贪心）≠ behavior policy（探索）→ 这就是 **off-policy**。

> 【设计考古 · fact】Q-learning 由 Chris Watkins 在 1989 年博士论文 *Learning from Delayed Rewards* 提出；Watkins & Dayan 1992（*Machine Learning*, 8, 279–292）给出第一个严格收敛证明：**只要所有 state-action 被无限次重复采样、且 Q 值用离散（表格）表示，Q-learning 以概率 1 收敛到最优 `Q*`**。来源：WebFetch 核实 Watkins & Dayan 1992 摘要页（gatsby.ucl.ac.uk）+ Wikipedia "Q-learning"。

对照 **SARSA（on-policy）**，update 里把 `max` 换成「实际下一步要走的动作」`A_{t+1}`：

```
SARSA:      Q(S_t,A_t) ← Q(S_t,A_t) + α [ R_{t+1} + γ Q(S_{t+1}, A_{t+1}) − Q(S_t,A_t) ]
Q-learning: Q(S_t,A_t) ← Q(S_t,A_t) + α [ R_{t+1} + γ max_a' Q(S_{t+1}, a') − Q(S_t,A_t) ]
```

> 【设计考古 · fact】SARSA（State–Action–Reward–State–Action）由 Rummery & Niranjan 1994 以 "Modified Connectionist Q-Learning" 之名提出，Rich Sutton 起的 "SARSA" 名最初只是个脚注。SARSA 评估的是它**实际在跑的** policy（含探索），所以在 cliff-walking 这类例子里会学出更「保守安全」的路；Q-learning 学最优路但训练中可能更冒险。来源：WebFetch 核实 Wikipedia "SARSA"。

这条 on/off-policy 的区别后面会反复出现——它正是 deadly triad 里「off-policy」那一条，也是 DQN 能用 replay buffer（存的是旧 policy 产的数据）的根本原因：off-policy 算法本来就允许用别的 policy 产的数据学。

### 1.3 把 Q 换成神经网络：DQN 与两个补丁

表格 Q-learning 在 `|S|` 大（比如 Atari 的像素）时不可行——你没法给每个像素组合存一个 Q 值。自然想法：用函数逼近 `Q(s,a; θ)`，拿神经网络当函数。但直接这么做会**发散**，原因有二：

1. **样本高度相关**：RL 数据是一条时间序列，连续帧极相似。用相关样本做 SGD，违反 i.i.d. 假设，梯度估计差、容易振荡。
2. **target 在动（追自己尾巴）**：target `r + γ max_a' Q(s',a'; θ)` 和你正在优化的 `Q(s,a; θ)` **共享同一套 θ**。你每更新一步 θ，target 也跟着变，等于在追一个一直跑的目标，正反馈下极易发散。

**DQN 的两个补丁正是对症下药：**

- **Experience Replay**：把 transition `(s,a,r,s',done)` 存进一个 buffer `D`，训练时**随机**采 minibatch。随机采样打散了时间相关性（恢复近似 i.i.d.），且每条数据可被复用多次，数据效率高。
- **Target Network**：复制一份网络 `Q(·; θ⁻)` 当 target 专用，参数 `θ⁻` **冻结**，每 `C` 步才从 `θ` 同步一次。这样在 `C` 步内 target 固定，把「移动靶」变「固定靶」，回归问题变稳。

> 【设计考古 · fact】DQN 来自 Mnih et al. 两篇：(1) 2013 *"Playing Atari with Deep Reinforcement Learning"*（NeurIPS Deep Learning Workshop，arXiv:1312.5602），首次用 CNN 从原始像素端到端学 control，在 7 个 Atari 游戏 6 个超过此前最好、3 个超过人类专家；(2) 2015 *"Human-level control through deep reinforcement learning"*（*Nature*），加入 target network、扩到 49 个游戏达人类水平。来源：WebFetch 核实 arXiv:1312.5602 摘要 + Wikipedia "Deep reinforcement learning"。Nature 全文 PDF 被 paywall/二进制无法逐字提取，本章涉及 Nature 版细节（如 target network 的 `θ⁻`、每 C 步更新、达人类水平表述）均来自上述可访问二手源，已逐条标注。

> 【fact · 引用原文】2013 论文摘要原文（WebFetch 核实 arXiv:1312.5602）：*"We present the first deep learning model to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning. The model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards."*

#### DQN 的损失函数（Nature 版本）

DQN 把 Q-learning 的 TD error 写成一个回归 loss，对 θ 做 SGD：

```
L(θ) = E_{(s,a,r,s') ~ D} [ ( y − Q(s,a; θ) )² ]
其中 target:  y = r + γ · max_a' Q(s', a'; θ⁻)        （非终止）
              y = r                                    （终止 s' 时去掉 bootstrap 项）
```

注意 target 用 `θ⁻`（冻结的 target network），而被优化的是 `θ`（online network）。这是「示意，非逐字」的标准写法——具体符号在不同文献略有差异，但语义一致：**冻结 target、对 online 网络回归 TD target**。Algorithm 1（"Deep Q-learning with experience replay"）的步骤（综合二手源，标【示意，非逐字】）：

```
初始化 replay memory D（容量 N）
初始化 Q 网络（随机权重 θ），target 网络 θ⁻ ← θ
for episode = 1..M:
    初始化状态 s
    for t = 1..T:
        以概率 ε 随机选 a，否则 a = argmax_a Q(s,a; θ)      # ε-greedy 探索
        执行 a，得到 r, s'
        把 (s,a,r,s',done) 存进 D
        从 D 随机采 minibatch (s_j,a_j,r_j,s'_j,done_j)
        y_j = r_j                          if done_j
            = r_j + γ max_a' Q(s'_j,a'; θ⁻) otherwise
        对 (y_j − Q(s_j,a_j; θ))² 做一步梯度下降更新 θ
        每 C 步：θ⁻ ← θ                     # 同步 target network
        s ← s'
```

这套伪代码和后面要精读的 cleanrl/dqn.py **逐行对应**——读完源码你会发现工业实现就是把这段忠实翻译成 PyTorch，外加一堆工程细节（vectorized env、soft update、logging）。

---

## §2 核心机制：源码精读（cleanrl/dqn.py 逐行）

> 选 **CleanRL** 的 `dqn.py` 做精读，因为它是「单文件、无抽象、可直接对照论文」的参考实现，社区公认教学价值高。下面所有标【真实源码 vwxyzjn/cleanrl@cleanrl/dqn.py】的代码块均由 WebFetch 从 `raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/dqn.py` **逐字取得**（已核实）。ReplayBuffer 来自 stable-baselines3，单独标注。

### 2.1 Q 网络：就是个 3 层 MLP

【真实源码 vwxyzjn/cleanrl@cleanrl/dqn.py】

```python
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)
```

**逐行注解：**
- `np.array(env.single_observation_space.shape).prod()`：把 obs 的 shape 展平成一个标量维度。CartPole 的 obs 是 4 维 `(4,)`，所以输入层是 4。这一行的写法是为了兼容多维 obs（虽然 classic control 用不到）。
- 隐藏层 `120 → 84`，两层 ReLU。**这是 classic control 的小 MLP，不是 Atari 的 CNN**。Atari 版在 `dqn_atari.py`，结构是卷积层。
- 输出层 `env.single_action_space.n`：CartPole 是 `Discrete(2)`，所以输出 2 个值——**对应两个动作各自的 Q 值**。这是 value-based 方法的关键设计：网络一次前向输出**所有动作**的 Q，挑动作只需对输出取 argmax，不用对动作空间做循环。（注意：这种「一次输出所有动作 Q」的结构只适用于**离散**动作；连续动作得用别的架构，比如 DDPG/SAC 的 critic 吃 `(s,a)` 输出标量——这是后续章节内容。）

> 【真坑】`forward` 返回的是 raw Q 值，**没有任何 softmax/激活**。Q 值是回归量不是概率，别手贱加 softmax——那是 policy-based 或 C51 distributional 才做的事。

### 2.2 ε-greedy 的线性退火

【真实源码 vwxyzjn/cleanrl@cleanrl/dqn.py】

```python
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
```

**逐行注解：**
- 从 `start_e`（默认 1.0，纯随机）线性降到 `end_e`（默认 0.05），跨越 `duration` 步，之后 `max(..., end_e)` 把它钳在地板值。
- `duration = exploration_fraction * total_timesteps`（默认 `0.5 * 500000 = 250000`）：前一半训练时间在退火探索率，后一半保持 0.05 的小探索。
- 这是 exploration-exploitation 的工程化：**早期多探索（建 buffer 多样性），后期多利用（收敛）**。这是 value-based 方法标配——因为 Q-learning 是 off-policy，behavior policy（ε-greedy）和 target policy（greedy）天然分离，可以独立调 ε。

调用处【真实源码 vwxyzjn/cleanrl@cleanrl/dqn.py】：

```python
epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
if random.random() < epsilon:
    actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
else:
    q_values = q_network(torch.Tensor(obs).to(device))
    actions = torch.argmax(q_values, dim=1).cpu().numpy()
```

**注解：**
- `random.random() < epsilon` → 随机动作（探索）；否则走 `argmax(q_values)`（利用）。和伪代码的 ε-greedy 一一对应。
- `dim=1`：因为 cleanrl 用 vectorized env，`q_values` shape 是 `(num_envs, n_actions)`，对动作维 argmax。

### 2.3 训练核心：TD target 与 loss（最关键的 11 行）

【真实源码 vwxyzjn/cleanrl@cleanrl/dqn.py】

```python
if global_step > args.learning_starts:
    if global_step % args.train_frequency == 0:
        data = rb.sample(args.batch_size)
        with torch.no_grad():
            target_max, _ = target_network(data.next_observations).max(dim=1)
            td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
        old_val = q_network(data.observations).gather(1, data.actions).squeeze()
        loss = F.mse_loss(td_target, old_val)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

（上面省去了中间的 logging 行，logging 不影响算法语义；完整文件里 `loss = F.mse_loss(...)` 后还有 `if global_step % 100 == 0:` 的 TensorBoard 记录块。）

**这是整章最该看懂的一段，逐行拆：**

1. `if global_step > args.learning_starts`：默认 `learning_starts=10000`。**前 1 万步只往 buffer 里灌数据、不训练**。为什么？buffer 太空时随机采的 minibatch 多样性差、还高度相关，过早训练等于用坏数据带歪网络。这是个被反复验证的工程默认值。
2. `if global_step % args.train_frequency == 0`：默认 `train_frequency=10`。**每 10 个环境步才训一次**。不是每步都更新——降算力、也让 buffer 多积累新鲜样本再训。
3. `data = rb.sample(args.batch_size)`：从 replay buffer **随机**采 128 条 transition。**这就是 experience replay 打散相关性的落点。**
4. `with torch.no_grad():`：**target 的计算不参与梯度**。这是 DQN 正确性的命门——target 是「监督信号」，必须 detach。忘了这个会让梯度同时流向 target 和 prediction，训练直接乱掉。
5. `target_max, _ = target_network(data.next_observations).max(dim=1)`：用 **target network（`θ⁻`）** 算 `max_a' Q(s', a'; θ⁻)`。注意是 `target_network` 不是 `q_network`——**这就是 target network 补丁的落点**。`.max(dim=1)` 对动作维取最大，对应 Q-learning 的 `max_a'`。
6. `td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())`：
   - 完整还原 `y = r + γ · max_a' Q(s',a'; θ⁻)`。
   - **`(1 - dones)` 是终止处理**：episode 结束时 `done=1`，整个 bootstrap 项归零，`y = r`。对应伪代码 `y_j = r_j if done_j`。**漏掉这个 `(1-done)` 是新手最常见的 bug**——会在终止状态错误地 bootstrap 一个不存在的「下一状态」，把价值估计带偏。
7. `old_val = q_network(data.observations).gather(1, data.actions).squeeze()`：用 **online network（`θ`）** 算 `Q(s,a; θ)`。`gather(1, data.actions)` 的作用是：网络输出所有动作的 Q（shape `(B, n_actions)`），但我们只要**当时实际执行的那个动作** `a` 对应的 Q（shape `(B,1)`），`gather` 按 `data.actions` 的索引把它挑出来。这一步常被忽视——TD error 比较的是「**实际动作**的预测 Q」和「target」，不是所有动作。
8. `loss = F.mse_loss(td_target, old_val)`：标准 MSE，即 `(y − Q(s,a;θ))²` 的 batch 平均。对应 DQN 的 `L(θ)`。
9. `optimizer.zero_grad(); loss.backward(); optimizer.step()`：标准 PyTorch 三步走，对 `θ` 做一步梯度下降。注意梯度**只流向 online network**（因为 target 在 `no_grad` 里、且 target_network 的参数不在 optimizer 里）。

> 【真坑 · 必看】第 4 和第 5 行是 DQN「能不能学起来」的两个开关：**(a) target 必须 `no_grad`/detach；(b) target 必须用 `target_network` 而不是 `q_network`**。任意一个写错，loss 看起来还在降，但 Q 值会发散或学不出策略。这类 bug 的隐蔽性在于「不会报错、只是不收敛」，调试时优先 review 这两行。

### 2.4 target network 同步：soft update

【真实源码 vwxyzjn/cleanrl@cleanrl/dqn.py】

```python
if global_step % args.target_network_frequency == 0:
    for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
        target_network_param.data.copy_(
            args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
        )
```

**注解：**
- 每 `target_network_frequency`（默认 500）步同步一次。
- `tau * θ + (1-tau) * θ⁻` 是 **Polyak / soft update**。注意 cleanrl 默认 `tau=1.0`，此时退化成 `θ⁻ ← θ`，即**硬同步**（hard update），和 Nature DQN 原版的「每 C 步直接复制」一致。
- 写成 soft 形式是为了通用：`tau<1` 时 target network 缓慢跟随 online network（DDPG/SAC 常用 `tau≈0.005` 的小步软更新，每步都更）。cleanrl 这里保留接口、默认值对齐原版。

> 【设计选择对比】**Hard update（每 C 步复制，DQN 原版）vs Soft update（每步 Polyak，DDPG 风格）**：hard update 在 C 步内 target 完全静止，回归目标最稳但 target 「过时」；soft update 让 target 平滑漂移，更新更频繁但每次幅度极小。两者都是为同一个目的——**让 target 比 online 变得慢**——只是节奏不同。

### 2.5 超参数全表（默认值，逐字核实）

【真实源码 vwxyzjn/cleanrl@cleanrl/dqn.py · @dataclass Args】（以下默认值均逐字核实）

| 超参 | 默认值 | 作用 / 注解 |
|---|---|---|
| `env_id` | `"CartPole-v1"` | 默认环境 |
| `total_timesteps` | `500000` | 总环境步数 |
| `learning_rate` | `2.5e-4` | Adam 学习率（对齐原版数值，但优化器换成 Adam，原版是 RMSProp） |
| `num_envs` | `1` | 并行环境数 |
| `buffer_size` | `10000` | replay buffer 容量 N |
| `gamma` | `0.99` | 折扣因子 γ |
| `tau` | `1.0` | target 更新率（1.0 = 硬同步） |
| `target_network_frequency` | `500` | 每多少步同步一次 target network（C） |
| `batch_size` | `128` | 从 buffer 采的 minibatch 大小 |
| `start_e` | `1` | ε 起始（纯探索） |
| `end_e` | `0.05` | ε 地板 |
| `exploration_fraction` | `0.5` | 在前 50% 训练步内退火 ε |
| `learning_starts` | `10000` | 前 1 万步只灌 buffer 不训练 |
| `train_frequency` | `10` | 每 10 步训练一次 |

> 【fact · 与原版差异】CleanRL 官方文档（WebFetch 核实 docs.cleanrl.dev/rl-algorithms/dqn）说明 `dqn.py` 与 Nature 原版的工程差异：(1) classic control 用 3 层 MLP（120→84）而非 CNN；(2) 优化器 Adam 而非 RMSProp；(3) target 用 Polyak soft update 接口（默认 `tau=1.0` 等价硬更新）。官方 benchmark：`dqn.py` 在 CartPole-v1 上达 **488.69 ± 16.11**（500k 步内接近满分 500）。

### 2.6 replay buffer 的实现（stable-baselines3）

cleanrl 通过 `from cleanrl_utils.buffers import ReplayBuffer` 引入，底层是 stable-baselines3 的 `ReplayBuffer`。实例化【真实源码 vwxyzjn/cleanrl@cleanrl/dqn.py】：

```python
rb = ReplayBuffer(
    args.buffer_size,
    envs.single_observation_space,
    envs.single_action_space,
    device,
    handle_timeout_termination=False,
)
```

buffer 的 `add` / `sample` 核心【真实源码 DLR-RM/stable-baselines3@stable_baselines3/common/buffers.py】（WebFetch 核实，节选 add 的存储与 sample 的随机索引部分）：

```python
def add(self, obs, next_obs, action, reward, done, infos):
    ...
    self.observations[self.pos] = np.array(obs)
    ...
    self.next_observations[self.pos] = np.array(next_obs)
    self.actions[self.pos] = np.array(action)
    self.rewards[self.pos] = np.array(reward)
    self.dones[self.pos] = np.array(done)
    ...
    self.pos += 1
    if self.pos == self.buffer_size:
        self.full = True
        self.pos = 0          # 环形覆盖：满了从头覆盖最旧数据
```

```python
def sample(self, batch_size, env=None):
    ...
    if self.full:
        batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
    else:
        batch_inds = np.random.randint(0, self.pos, size=batch_size)
    return self._get_samples(batch_inds, env=env)
```

**注解：**
- buffer 是**环形（ring）**结构：`self.pos` 到顶就归零、`self.full=True`，之后新数据**覆盖最旧的**。所以它永远保留「最近 N 条」transition。
- `sample` 用 `np.random.randint` **均匀随机**抽索引 → 这就是 experience replay 打散相关性的机制实现。满了之后避开 `self.pos`（正在写的位置，next_obs 尚未配对完整）以防采到坏样本——这是个容易写错的边界，正是「借用成熟 buffer 而非自己撸」的理由。
- 返回的 `ReplayBufferSamples` 是个 namedtuple，字段 `observations / actions / next_observations / dones / rewards`——正好对上 §2.3 里 `data.next_observations`、`data.actions` 等访问。

至此，cleanrl/dqn.py 的算法骨架已逐行拆完。下一节我们脱离框架，自己从零写两个 demo。

---

## §3 方案对比

### 3.1 MC vs TD vs DP（prediction 层面）

| 维度 | Dynamic Programming | Monte Carlo | TD(0) |
|---|---|---|---|
| 需要环境模型 `P,R`？ | 需要 | 不需要 | 不需要 |
| bootstrapping（用估计更新估计）？ | 是 | 否 | 是 |
| 采样真实交互？ | 否（全宽度） | 是 | 是 |
| 何时能更新 | 每次扫全状态 | episode 结束后 | **每步** |
| 能用于持续型任务？ | 是 | 否（需 episode） | 是 |
| variance | 低 | **高** | 低-中 |
| bias | 无（精确） | 无（无偏） | **有**（bootstrap 估计有偏） |
| 适用边界 | 状态小、模型已知 | episodic、能跑到底、要无偏 | 在线、长/无限 horizon、要低方差 |

**具体场景：**
- 你在做一个**有精确模型的小棋盘**（如 4×4 格子、转移确定）→ DP（value iteration）最快最准。
- 你在做**离线策略评估、episode 短且能跑完、且极在意无偏**（比如某些 OPE 场景的 baseline）→ MC 的无偏性有价值。
- 你在做**长 horizon、在线、需要边跑边学**（绝大多数 deep RL、机器人、游戏）→ TD，几乎是默认选择。

### 3.2 SARSA vs Q-learning（control 层面）

| 维度 | SARSA（on-policy） | Q-learning（off-policy） |
|---|---|---|
| target 用的动作 | 实际下一步动作 `A_{t+1}`（含探索） | 贪心 `max_a' Q` |
| 学的是 | 当前行为策略（含探索）的 Q | 最优策略 `Q*` |
| 能用 replay buffer / 旧数据？ | 受限（数据需贴近当前 policy） | **能**（off-policy 天然支持） |
| cliff-walking 行为 | 学**安全**路（绕开悬崖边） | 学**最优**路（贴悬崖走，训练中可能掉下去） |
| 收敛 | 收敛到 ε-soft 最优 | 收敛到 `Q*`（表格、充分探索下） |

**具体场景 / 边界：**
- **在线、探索本身有真实代价**（真机器人，掉下悬崖=摔坏；训练中的失败有成本）→ SARSA 更稳妥，因为它把探索风险算进了所学策略。
- **能用经验回放、想要最优策略、探索代价低**（模拟器里随便撞）→ Q-learning / DQN。**DQN 选 Q-learning 而非 SARSA，正是因为要配 replay buffer——buffer 里全是旧 policy 产的数据，只有 off-policy 算法能正确消费。**

### 3.3 DQN vs Double DQN（解 overestimation）

| 维度 | DQN | Double DQN |
|---|---|---|
| target | `r + γ max_a' Q(s',a'; θ⁻)` | `r + γ Q(s', argmax_a' Q(s',a'; θ); θ⁻)` |
| 动作**选择**用哪个网络 | target network（隐含在 max 里） | **online network** θ |
| 动作**评估**用哪个网络 | target network θ⁻ | target network θ⁻ |
| 核心问题 | `max` 操作 + 估计误差 → **系统性高估** Q | 选择/评估解耦 → 显著降低高估 |
| 实现改动 | — | 仅改 target 一行；replay/网络全复用 |

> 【fact】Double DQN 来自 van Hasselt, Guez, Silver, *"Deep Reinforcement Learning with Double Q-learning"*（AAAI 2016，arXiv:1509.06461）。核心论点：标准 Q-learning/DQN 因 `max` 操作叠加估计误差而**系统性高估** action value（maximization bias）；修法是把**动作选择**（用 online 网络 argmax）和**动作评估**（用 target 网络取值）解耦。原文结论：*"not only reduces the observed overestimations... but that this also leads to much better performance on several games."* 来源：WebFetch 核实 arXiv:1509.06461 摘要页。

**为什么 `max` 会高估（直觉）**：假设真实 Q 值都相等，但你的估计有零均值噪声。`max` 总是挑「估计偏高」的那个 → 期望上 `E[max(估计)] ≥ max(真值)`，偏差恒为正。bootstrap 又把这个正偏差一路往前传，越滚越大。Double DQN 用两套独立估计让「选的人」和「打分的人」不是同一个，斩断这个正反馈。这个 maximization bias 在第 4-5 章 RLHF 里也有回声——reward model 的 over-optimization（policy 钻 reward model 的高估漏洞）在结构上同源。

### 3.4 DQN vs C51（distributional，进阶一瞥）

cleanrl 同目录有 `c51.py`【真实源码 vwxyzjn/cleanrl@cleanrl/c51.py，已 WebFetch 核实其语义】：它不学单个 Q 值，而学**回报的整个分布**（用 101 个 atom 在 `[v_min, v_max]` 上的离散分布表示），loss 从 MSE 换成 categorical cross-entropy，target 计算多一步「投影回 support 网格」。

| 维度 | DQN | C51 (Categorical DQN) |
|---|---|---|
| 网络输出 | 每动作一个标量 Q | 每动作一个分布（101 atoms 的 PMF） |
| loss | MSE on Q | cross-entropy on 分布 |
| 额外步骤 | — | 把平移后的 target atoms 投影回原 support |
| 收益 | 简单 | 建模回报不确定性，常更稳/更强 |
| 适用边界 | baseline、够用 | 想榨性能、回报多模态/高方差时 |

**何时不用 distributional**：原型期、问题简单、调试成本敏感时——C51 的投影步骤多、超参（`v_min/v_max/n_atoms`）要调，过早上反而增加 debug 面。先把 vanilla DQN 跑通再说。

---

## §4 ⭐ 可运行 Demo（重中之重）

> **两个 demo 都设计为可运行，请在你的环境验证。** 我会给完整代码、运行步骤、预期输出，并和上面的源码呼应。Demo A 只依赖 numpy（吃透 TD 更新本身）；Demo B 用 PyTorch + gymnasium 复刻 cleanrl/dqn.py 的最小骨架（看 episodic return 爬升）。

### Demo A：表格 Q-learning 解 GridWorld（纯 numpy）

**目标**：在一个 `4×4` 网格里，agent 从左上 `(0,0)` 走到右下 `(3,3)`（目标格 reward=+1，其余每步 reward=0，可加小的 step penalty）。用纯表格 Q-learning 学出最短路径，观察 (1) 每个 episode 的步数下降、(2) 学出的贪心策略指向终点。这是 §1.2 Q-learning 更新公式的最直接落地，没有任何神经网络，便于看清 `max` bootstrap 到底在干嘛。

**依赖**：仅 `numpy`。

**完整代码**（保存为 `demo_a_tabular_q.py`，标「设计为可运行，请在你环境验证」）：

```python
import numpy as np

# ---------------- GridWorld 环境（自包含，无外部依赖） ----------------
# 4x4 网格，状态用 0..15 编号 (s = row*4 + col)。
# 动作: 0=上 1=下 2=左 3=右。撞墙则停在原地。
# 终点 = 15 (3,3)，到达 reward=+1 并结束；每走一步 reward = -0.01（鼓励走短路）。
GRID = 4
N_STATES = GRID * GRID
N_ACTIONS = 4
GOAL = N_STATES - 1
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

def step(s, a):
    r, c = divmod(s, GRID)
    dr, dc = ACTIONS[a]
    nr, nc = r + dr, c + dc
    if 0 <= nr < GRID and 0 <= nc < GRID:      # 合法移动
        ns = nr * GRID + nc
    else:                                       # 撞墙，原地不动
        ns = s
    if ns == GOAL:
        return ns, 1.0, True                    # 到终点：+1，done
    return ns, -0.01, False                     # 普通一步：小负奖励

# ---------------- 表格 Q-learning ----------------
def train(episodes=500, alpha=0.5, gamma=0.95,
          eps_start=1.0, eps_end=0.05, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))         # Q 表，全 0 初始化
    steps_per_ep = []
    for ep in range(episodes):
        # ε 线性退火（呼应 cleanrl 的 linear_schedule）
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / (episodes * 0.7))
        s, done, steps = 0, False, 0
        while not done and steps < 100:         # 上限 100 步防止死循环
            # --- ε-greedy 行为策略（off-policy 的 behavior policy）---
            if rng.random() < eps:
                a = rng.integers(N_ACTIONS)     # 探索
            else:
                a = int(np.argmax(Q[s]))        # 利用
            ns, r, done = step(s, a)
            # --- Q-learning 更新（§1.2 的公式逐字落地）---
            # target = r + γ * max_a' Q(ns, a')；终止则去掉 bootstrap 项
            td_target = r + (0.0 if done else gamma * np.max(Q[ns]))
            td_error = td_target - Q[s, a]
            Q[s, a] += alpha * td_error          # Q(s,a) ← Q(s,a) + α·δ
            s = ns
            steps += 1
        steps_per_ep.append(steps)
    return Q, steps_per_ep

def greedy_policy_str(Q):
    arrow = {0: '^', 1: 'v', 2: '<', 3: '>'}
    cells = []
    for s in range(N_STATES):
        cells.append('G' if s == GOAL else arrow[int(np.argmax(Q[s]))])
    return '\n'.join(' '.join(cells[r*GRID:(r+1)*GRID]) for r in range(GRID))

if __name__ == "__main__":
    Q, steps = train()
    # 打印学习曲线（每 50 episode 的平均步数，应当下降并趋近最短路 6 步）
    print("episode  avg_steps(last 50)")
    for i in range(0, len(steps), 50):
        chunk = steps[i:i+50]
        print(f"{i:7d}  {np.mean(chunk):.2f}")
    print("\n最短路最优步数 = 6（从 (0,0) 到 (3,3) 需 3 下 + 3 右）")
    print("\n学出的贪心策略（每格指向它认为最优的方向，G=终点）:")
    print(greedy_policy_str(Q))
```

**运行步骤：**
```bash
pip install numpy
python demo_a_tabular_q.py
```

**预期输出（设计为可运行，数值因 seed 略有差异，趋势应一致）：**
```
episode  avg_steps(last 50)
      0   38.xx        ← 早期纯探索，乱走，步数高
     50   12.xx
    100    7.xx
    ...
    450    6.xx        ← 收敛到接近最优 6 步

最短路最优步数 = 6（从 (0,0) 到 (3,3) 需 3 下 + 3 右）

学出的贪心策略（每格指向它认为最优的方向，G=终点）:
v > v > ...           ← 箭头整体指向右下，沿对角线把 agent 导向终点
...
> > > G
```

**和源码的呼应：**
- `td_target = r + (0.0 if done else gamma * np.max(Q[ns]))` ↔ cleanrl 的 `td_target = data.rewards + gamma * target_max * (1 - data.dones)`。`np.max(Q[ns])` 就是 `target_max`，`(0.0 if done ...)` 就是 `(1 - dones)`。**这里没有 target network**——表格情形下 Q-learning 本身就收敛，target network 是 DQN 为「函数逼近不稳定」加的补丁，表格不需要。这正说明：**replay 和 target network 不是 Q-learning 的本质，是把 Q-learning 塞进神经网络后才需要的工程稳定器**。
- `if rng.random() < eps:` ↔ cleanrl 的 `if random.random() < epsilon:`，一模一样的 ε-greedy。

**这个 demo 教会你什么**：跑一次你会直观看到「步数随 episode 下降」——这就是 reward 在上升（步数少=累计 step penalty 少+早拿到 +1）。把 `alpha` 调到 `1.0` 或 `0.01`、把 `gamma` 调到 `0.5` 再跑，能亲手感受学习率/折扣对收敛速度与策略的影响。

### Demo B：最小 DQN 解 CartPole-v1（PyTorch + gymnasium）

**目标**：用 ~80 行复刻 cleanrl/dqn.py 的**算法核心**（去掉 vectorized env、logging、CLI），在 CartPole-v1 上训练，观察 **episodic return 从 ~10 爬到接近 500**。这是 §2 源码精读的「自己手写一遍」版本，每一块都对得上。

**依赖**：
```bash
pip install "gymnasium[classic-control]" torch numpy stable-baselines3
```
（用 stable-baselines3 的 `ReplayBuffer` 避免重写 buffer 的边界 bug，与 cleanrl 同款。CPU 即可，不需要 GPU。）

**完整代码**（保存为 `demo_b_dqn_cartpole.py`，标「设计为可运行，请在你环境验证」）：

```python
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3.common.buffers import ReplayBuffer

# ---------------- 超参（对齐 cleanrl/dqn.py 默认值，仅缩短 total_timesteps 便于快速看曲线）----------------
SEED = 1
TOTAL_TIMESTEPS = 60_000          # cleanrl 默认 500k；CartPole 6 万步通常已能看到明显爬升
LEARNING_RATE = 2.5e-4
BUFFER_SIZE = 10_000
GAMMA = 0.99
TAU = 1.0                          # 1.0 = 硬同步 target network
TARGET_NET_FREQ = 500
BATCH_SIZE = 128
START_E, END_E = 1.0, 0.05
EXPLORATION_FRACTION = 0.5
LEARNING_STARTS = 10_000
TRAIN_FREQUENCY = 10

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cpu")      # CartPole 用 CPU 足够

# ---------------- Q 网络（与 cleanrl 的 QNetwork 等价：3 层 MLP 120->84->n_actions）----------------
class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, n_actions),
        )
    def forward(self, x):
        return self.net(x)

def linear_schedule(start_e, end_e, duration, t):     # 与 cleanrl 逐字等价
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

# ---------------- 环境 ----------------
env = gym.make("CartPole-v1")
obs_dim = int(np.array(env.observation_space.shape).prod())
n_actions = env.action_space.n

q_network = QNetwork(obs_dim, n_actions).to(device)
target_network = QNetwork(obs_dim, n_actions).to(device)
target_network.load_state_dict(q_network.state_dict())   # θ⁻ ← θ 初始化
optimizer = torch.optim.Adam(q_network.parameters(), lr=LEARNING_RATE)

# stable-baselines3 ReplayBuffer 期望 vectorized 接口，这里用单环境的 obs/space 适配
rb = ReplayBuffer(
    BUFFER_SIZE,
    env.observation_space,
    env.action_space,
    device,
    handle_timeout_termination=False,
)

# ---------------- 训练主循环 ----------------
obs, _ = env.reset(seed=SEED)
ep_return, ep_returns = 0.0, []
for global_step in range(TOTAL_TIMESTEPS):
    # --- ε-greedy 选动作（呼应 §2.2）---
    epsilon = linear_schedule(START_E, END_E, EXPLORATION_FRACTION * TOTAL_TIMESTEPS, global_step)
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            q_values = q_network(torch.Tensor(obs).to(device))
            action = int(torch.argmax(q_values).item())

    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    ep_return += reward

    # 存入 replay buffer（SB3 buffer 接口；infos 传 [{}]）
    rb.add(np.array([obs]), np.array([next_obs]),
           np.array([action]), np.array([reward]), np.array([terminated]), [{}])
    obs = next_obs

    if done:
        ep_returns.append(ep_return)
        # 每 20 个 episode 打印一次最近 20 局平均 return
        if len(ep_returns) % 20 == 0:
            print(f"step={global_step:6d}  episodes={len(ep_returns):4d}  "
                  f"avg_return(last20)={np.mean(ep_returns[-20:]):6.1f}  eps={epsilon:.2f}")
        obs, _ = env.reset()
        ep_return = 0.0

    # --- 训练（呼应 §2.3，逐行对应）---
    if global_step > LEARNING_STARTS and global_step % TRAIN_FREQUENCY == 0:
        data = rb.sample(BATCH_SIZE)
        with torch.no_grad():                                   # target 不回传梯度
            target_max, _ = target_network(data.next_observations).max(dim=1)
            td_target = data.rewards.flatten() + GAMMA * target_max * (1 - data.dones.flatten())
        old_val = q_network(data.observations).gather(1, data.actions).squeeze()
        loss = F.mse_loss(td_target, old_val)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        # --- soft/hard 同步 target network（呼应 §2.4）---
        if global_step % TARGET_NET_FREQ == 0:
            for tp, qp in zip(target_network.parameters(), q_network.parameters()):
                tp.data.copy_(TAU * qp.data + (1.0 - TAU) * tp.data)

env.close()
print("\n训练结束。最后 20 局平均 return =", round(np.mean(ep_returns[-20:]), 1))
print("（CartPole-v1 满分 500，~475 视为 solved）")
```

**运行步骤：**
```bash
pip install "gymnasium[classic-control]" torch numpy stable-baselines3
python demo_b_dqn_cartpole.py
```

**预期输出（设计为可运行；DQN 在 CartPole 上有名地不稳，曲线会震荡但总体上行；具体数值随 seed/版本变化）：**
```
step= 11000  episodes=  ...  avg_return(last20)=  20.x  eps=0.78
step= 20000  episodes=  ...  avg_return(last20)=  45.x  eps=0.60
step= 35000  episodes=  ...  avg_return(last20)= 180.x  eps=0.30
step= 50000  episodes=  ...  avg_return(last20)= 350.x  eps=0.05
...
训练结束。最后 20 局平均 return = 4xx.x
（CartPole-v1 满分 500，~475 视为 solved）
```

> **诚实预期管理**：vanilla DQN 在 CartPole 上**不保证单调上升**，常见「爬到很高又突然跌回去再爬」的现象（catastrophic forgetting / 价值高估导致的策略崩溃）。60k 步通常能看到明显爬升、可能触及 400+，但**不一定每个 seed 都稳定 solved**。官方 cleanrl 用 500k 步 + 多 seed 才报到 488.69 ± 16.11。**如果你想看到更稳的曲线，把 `TOTAL_TIMESTEPS` 调回 `500_000`、或换 seed 多跑几次**——这本身就是 §5 「RL 训练不稳」的第一手体感。

**和源码的呼应（逐块对照）：**

| Demo B 代码块 | cleanrl/dqn.py 对应 | 说明 |
|---|---|---|
| `QNetwork` 3 层 MLP | §2.1 同款 | 唯一差别：去掉了 vectorized 的 `single_*_space` 包装 |
| `linear_schedule` | §2.2 逐字等价 | — |
| ε-greedy `if random.random() < epsilon` | §2.2 | — |
| `td_target = ... GAMMA * target_max * (1 - dones)` | §2.3 第 6 行 | **核心 TD target，含终止处理** |
| `gather(1, data.actions)` | §2.3 第 7 行 | 取实际动作的 Q |
| `F.mse_loss` + 三步优化 | §2.3 第 8-9 行 | — |
| target network soft update | §2.4 | `TAU=1.0` 硬同步 |
| `learning_starts` / `train_frequency` 守卫 | §2.3 第 1-2 行 | — |

**这个 demo 教会你什么**：把它跑起来你就**亲手验证了 §2 的源码精读不是纸上谈兵**。想做消融实验、感受每个补丁的作用，见章末「五件套 · 代码题」——那里给了三个最有教学价值的改动方向（关掉 target network、关掉 replay、忘记 `(1-done)`），跑一次就知道这些「补丁」到底防的是什么灾难。

---

## §5 失败模式、真坑与根因

### 5.1 算法层失败模式

**(1) Maximization bias / overestimation（系统性高估）**
- **现象**：训练中 Q 值估计持续走高、远超真实可达 return；策略时好时坏。
- **根因**：§3.3 讲过——`max` 操作挑「估计偏高」的动作，叠加估计噪声 → 期望上正偏，bootstrap 再放大。
- **解**：Double DQN（选择/评估解耦）。这是最常见、最该先上的修法。

**(2) Deadly Triad（致命三要素）**
- **现象**：训练发散，loss 爆炸或 Q 值 → ±∞。
- **根因**：当 **function approximation（神经网络）+ bootstrapping（TD）+ off-policy（如 replay/Q-learning）** 三者**同时**出现时，理论上不保证收敛，实践中易发散。表格 Q-learning 没有第一条（无逼近），所以收敛有保证；DQN 三条全占，正是它要 target network + replay 来「凑合稳住」的根本原因。
- **缓解**：target network、合适的学习率、gradient clipping、reward/observation 归一化。**注意是「缓解」不是「根治」——deadly triad 没有干净的通用解，这是 deep RL 不稳的理论底色。**

**(3) 终止状态 bootstrap 错误**
- **现象**：策略学不出来或价值估计莫名偏高，但 loss 看起来正常。
- **根因**：target 里漏了 `(1 - done)`，在 episode 终止后还 bootstrap 了一个不存在的「下一状态」的价值。
- **解**：永远写 `td_target = r + gamma * max_next_q * (1 - done)`。对照 §2.3 第 6 行。**这是新手 DQN 第一大 bug。**

### 5.2 工程层真坑

**(4) target 忘记 detach / 用错网络**（§2.3 真坑框已强调）
- **现象**：训练发散或不收敛，且**不报错**。
- **根因**：target 没 `no_grad`（梯度流进 target），或 target 用了 online network（退化成「追自己尾巴」）。
- **解**：target 计算包在 `with torch.no_grad():`，且必须用 `target_network`。Review DQN 代码先看这两行。

**(5) learning_starts 太小 / buffer 太空就训练**
- **现象**：早期训练把网络带歪，后面难恢复。
- **根因**：buffer 样本少且相关，过早 SGD 用了坏数据。
- **解**：`learning_starts`（cleanrl 默认 10000）让 buffer 先积累足够多样的数据。

**(6) ε 退火太快 / 太慢**
- **现象**：太快 → 探索不足，卡在次优策略；太慢 → 一直随机走，学得慢。
- **根因**：exploration-exploitation 失衡。
- **解**：`exploration_fraction`（默认在前 50% 步退火）是个稳健起点，按任务难度调。

**(7) buffer 自己撸出边界 bug**
- **现象**：偶发采到不完整 transition（next_obs 错位）。
- **根因**：环形 buffer 满了之后没避开正在写的位置（§2.6 SB3 的 `(randint(1, N) + pos) % N` 正是为此）。
- **解**：用成熟实现（SB3 / cleanrl_utils）。这是「不重复造轮子」的典型理由。

**(8) 把 DQN 用在连续动作 / 想当然加 softmax**
- **现象**：要么跑不起来（动作空间不是离散），要么策略全乱。
- **根因**：DQN 的「网络输出所有动作 Q + argmax」结构**只适用于离散动作**；Q 是回归量，加 softmax 是范畴错误。
- **解**：连续动作用 DDPG/TD3/SAC（后续章节）；Q 输出保持 raw。

### 5.3 调试 checklist（DQN 不收敛时按序排查）

1. target 是否 `no_grad` + 用 `target_network`？（最高频）
2. `(1 - done)` 是否在 target 里？
3. `gather` 是否取的是**实际动作**的 Q？
4. `learning_starts` 是否够大、buffer 是否已有足够数据？
5. ε 退火是否合理（早期够探索）？
6. 学习率是否过大（试着 ×0.5）？
7. 多跑几个 seed——可能只是 RL 固有的方差，不是 bug。

> 第 7 条不是玩笑：RL 单 seed 的结果**不可信**。声称「这个改动有效」前，至少 ≥3 个 seed 看分布。这是 RL 实证的基本卫生（呼应「N=1 不算 feasibility」的通则）。

---

## §6 与 LLM 对齐（RLHF/RLAIF）的连接

> 这一节把 TD/Q 显式接到你的真实目标——LLM 对齐。**先划清 fact 和待核的边界**，不把「相关」吹成「就是用 DQN」。

### 6.1 fact：主流 RLHF 用 PPO，不是 DQN

经典 RLHF pipeline（InstructGPT / ChatGPT 路线）的 RL 阶段用的是 **PPO**（Proximal Policy Optimization，policy-gradient 家族），**不是 value-based 的 DQN**。原因（fact-level 推理）：

- LLM 的「动作空间」是整个词表（数万 token），DQN 那套「网络输出所有动作 Q + argmax」在这个规模上不现实（虽然技术上 logits 也是每 token 一个值，但 value-based 的训练稳定性与 credit assignment 在长序列上很差）。
- RLHF 要在「不偏离原模型太远」(KL 约束) 的前提下微调一个已经很强的 policy（pretrained LM），policy-gradient + KL penalty 的 PPO 框架天然契合。

所以：**别把本章的 DQN 直接套到 RLHF 上**。DQN 是 value-based 的代表、TD 思想的集大成者，但 RLHF 主线在 policy-based 那条（第 4-5 章）。

### 6.2 fact：TD 思想本身贯穿 RLHF 的 critic 与 advantage

虽然不用 DQN，但**本章的 TD 内核在 RLHF 里无处不在**：

- PPO 有一个 **critic / value network** `V(s)`，它的训练**就是 TD 回归**——用 `r + γ V(s')` 当 target 拟合 `V(s)`，和 §1.1 的 TD(0) 同源。
- PPO 的 advantage 估计用 **GAE（Generalized Advantage Estimation）**，GAE 的核心是 **TD error 的指数加权和** `Σ (γλ)^k δ_{t+k}`，其中 `δ_t = r_t + γ V(s_{t+1}) − V(s_t)` 正是 §1.1 的 TD error。**不懂 TD error 就读不懂 GAE。**
- 「reward-to-go」「value baseline 降方差」这些 RLHF 工程常识，根都在本章的 MC vs TD、bias-variance trade-off。

所以本章对做 RLHF 的价值是：**它是读懂 PPO 里 critic / GAE / advantage 这半边的前置**。你以后调 RLHF 看到 `value_loss`、`gae_lambda`、`gamma` 这些超参，全是本章概念的延续。

### 6.3 fact：maximization bias 与 reward over-optimization 的结构同源

§3.3 的 maximization bias（policy 钻 Q 估计的高估漏洞）和 RLHF 里著名的 **reward model over-optimization / reward hacking**（policy 钻 reward model 的漏洞，把 RM 分数刷高但真实质量下降）在结构上是**同一类问题**：优化器会系统性地利用「估计模型」的误差。理解前者有助于直觉理解后者，以及为什么 RLHF 要 KL penalty、early stopping、RM ensemble 这些「别太信任估计」的手段。（此为结构类比，非「同一算法」的断言。）

### 6.4 待核 / 谨慎区：value-based RL 在 LLM 上的近期尝试

近年确有把 RL-for-LLM 显式建成 **value-based / Q-learning 形式**的研究方向（如把 token 生成建成 MDP 后用类 Q-learning 的离线 RL、ILQL 一类 implicit-language-Q 方法；以及从 Q-learning 视角重新解读 DPO 的工作）。

> 【待核】上述「ILQL / token-level Q 视角 / DPO 的 Q 解释」属于我训练知识中的方向性记忆，**本章未对这些具体论文做 WebFetch 逐字核实**，请勿当作精确出处引用。这里只用来说明：**value-based / TD 思想在 LLM 对齐里不是死路，而是一个活跃但非主流的分支**。要写进正式材料前需补 fetch 原论文核实。

**给做对齐的你的结论**：把本章吃透成专家，直接收益是**读懂 PPO 的 critic 半边（TD/GAE/advantage）**——这是 fact、是刚需；间接收益是对 over-optimization 等问题有更深的结构直觉。是否要深入 value-based-for-LLM 这条线，看你具体方向，且需自行核实最新文献。

---

## 五件套（章末固定结构）

### 一、本章 TL;DR（再压一遍）
TD = 用下一步估计更新当前估计（bootstrapping），缝合 MC 的采样与 DP 的 bootstrapping。Q-learning = TD 的 off-policy control 版，target 用 `max`，表格情形 w.p.1 收敛。DQN = Q-learning + 神经网络 + experience replay（打散相关性）+ target network（稳住移动靶）。核心坑：overestimation（→Double DQN）、deadly triad（理论无解只能缓解）、终止 bootstrap、target 忘 detach。对 RLHF：DQN 本身不用，但 TD 内核贯穿 PPO 的 critic/GAE。

### 二、关键术语对照（中 / 英）
| 中文 | English | 一句话 |
|---|---|---|
| 时序差分 | Temporal-Difference (TD) | 用 `r + γV(s')` 当 target |
| 自举 | bootstrapping | 用估计去更新估计 |
| TD 误差 | TD error δ | `r + γV(s') − V(s)` |
| 动作价值 | action value Q(s,a) | 在 s 做 a 之后的期望 return |
| 异策略 | off-policy | target policy ≠ behavior policy |
| 同策略 | on-policy | 学的就是在跑的策略 |
| 经验回放 | experience replay | 存 transition 随机采样 |
| 目标网络 | target network θ⁻ | 冻结的 target 专用网络 |
| 最大化偏差 | maximization bias | `max` 系统性高估 |
| 致命三要素 | deadly triad | 逼近+bootstrap+off-policy 易发散 |

### 三、自测题（5 道，附答案要点）
1. **为什么表格 Q-learning 不需要 target network，DQN 却需要？**
   答：表格情形无函数逼近，Q-learning 收敛有理论保证；DQN 引入神经网络后，target 和 prediction 共享参数形成移动靶，且占齐 deadly triad 三条，需 target network 把靶冻住缓解发散。
2. **`td_target = r + γ·max_next_q·(1 - done)` 里的 `(1-done)` 删掉会怎样？**
   答：episode 终止后仍 bootstrap 一个不存在的下一状态价值，价值估计偏高、策略学不好，且不报错——典型隐蔽 bug。
3. **Q-learning 和 SARSA 在 cliff-walking 上行为差异及原因？**
   答：SARSA（on-policy）把探索风险算进所学策略，学安全路；Q-learning（off-policy）学最优路（贴悬崖），训练中可能掉下去。原因：target 一个用实际动作、一个用 `max`。
4. **Double DQN 改了 DQN 的哪一行？解决什么？**
   答：改 target 一行——动作**选择**用 online 网络 argmax、动作**评估**用 target 网络取值。解决 `max` 带来的 overestimation。
5. **RLHF 主线用 DQN 吗？TD 思想在 RLHF 哪里出现？**
   答：不用，主线是 PPO（policy-gradient）。但 TD 贯穿 PPO 的 critic（TD 回归 `V`）和 GAE（TD error 的指数加权和）。

### 四、动手实验（基于 Demo，跑了才算数）
- 跑 **Demo A**，把 `alpha` 设为 `1.0` 和 `0.01` 各跑一次，对比收敛速度与最终策略——感受学习率。
- 跑 **Demo A**，把 `gamma` 从 `0.95` 降到 `0.5`，观察策略是否还指向终点——感受折扣对长期规划的影响。
- 跑 **Demo B**，用 3 个不同 `SEED`（1/2/3）各跑一次，记录最后 20 局平均 return 的**分布**——亲手验证「RL 单 seed 不可信」。

### 五、代码题（= 扩展 Demo，每题都是对 Demo B 的最小改动）

> 这三题是本章最有教学价值的消融实验，**每题改动 ≤10 行**，跑一次就知道对应「补丁」防的是什么灾难。

**代码题 1：关掉 target network（证明它真的在稳住训练）**
- 改动：把 §2.3 那段里的 `target_network(data.next_observations)` 改成 `q_network(data.next_observations)`（target 直接用 online 网络），并删掉 target network 同步那段。
- 预期：训练明显变不稳，Q 值更容易发散、return 曲线震荡更剧烈甚至崩溃。
- 学到：target network 不是装饰，它就是把「移动靶」变「固定靶」的那个补丁。

**代码题 2：关掉 experience replay（证明相关性会害死训练）**
- 改动：不从 buffer 随机采，改成只用**当前这一步**的 transition 做更新（`batch_size=1` 且每步用最新 transition，不存不采）。
- 预期：训练大幅恶化——样本高度相关，梯度估计差，难收敛。
- 学到：replay 的随机采样是恢复近似 i.i.d.、让 SGD 能工作的关键。

**代码题 3：忘记 `(1-done)`（复现新手第一大 bug）**
- 改动：把 `td_target = data.rewards.flatten() + GAMMA * target_max * (1 - data.dones.flatten())` 里的 `* (1 - data.dones.flatten())` 删掉。
- 预期：价值估计偏高，策略质量下降，且 loss 看起来「正常」——体会这个 bug 为什么隐蔽。
- 学到：终止状态必须切断 bootstrap；这类「不报错只是不收敛」的 bug 要靠 review 终止处理来抓。

**进阶代码题 4（选做）：实现 Double DQN**
- 改动：把 target 计算改成——先用 `q_network` 在 `next_observations` 上 `argmax` 选动作，再用 `target_network` 在该动作上 `gather` 取值。即 `r + γ · Q_target(s', argmax_a Q_online(s',a))`。约 3 行。
- 预期：Q 值高估缓解，CartPole 上训练通常更稳。
- 学到：§3.3 的「选择/评估解耦」如何用 3 行代码落地。

---

## Sources（本章实际 WebFetch / WebSearch 取证记录）

> 标注规则回顾：【真实源码 repo@path】= 逐字取得；【示意，非逐字】= 语义正确的标准写法；【待核】= 未逐字核实的记忆性内容。下列 URL 均在写作前实际 fetch 过。

**真实源码（逐字核实）**
- `vwxyzjn/cleanrl@cleanrl/dqn.py` — https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/dqn.py （QNetwork / linear_schedule / ε-greedy / 训练块 / target 同步 / Args 默认值 / make_env / ReplayBuffer 实例化，均逐字取得）
- `vwxyzjn/cleanrl@cleanrl/c51.py` — https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/c51.py （C51 语义核实，未逐字引用大段代码）
- `DLR-RM/stable-baselines3@stable_baselines3/common/buffers.py` — https://raw.githubusercontent.com/DLR-RM/stable-baselines3/master/stable_baselines3/common/buffers.py （ReplayBuffer add/sample 逐字节选）

**论文 / 史料（出处核实，部分全文 PDF 因 paywall/二进制无法逐字提取，细节取自可访问摘要与权威二手源）**
- Mnih et al. 2013, *Playing Atari with Deep Reinforcement Learning*, NeurIPS DL Workshop, arXiv:1312.5602 — https://arxiv.org/abs/1312.5602 （摘要逐字核实；Algorithm 1 / loss 标【示意，非逐字】，PDF 二进制未能逐字提取）
- Mnih et al. 2015, *Human-level control through deep reinforcement learning*, *Nature* — Nature 全文 paywall（https://www.nature.com/articles/nature14236 重定向至授权页；PDF 二进制未能解析）；target network / 达人类水平等表述取自 Wikipedia "Deep reinforcement learning"（https://en.wikipedia.org/wiki/Deep_reinforcement_learning，已核实）
- Watkins & Dayan 1992, *Q-learning*, *Machine Learning* 8, 279–292 — 摘要与收敛定理核实：http://www.gatsby.ucl.ac.uk/~dayan/papers/wd92.html ；citation 经 WebSearch 核实
- van Hasselt, Guez, Silver 2016, *Deep RL with Double Q-learning*, AAAI 2016, arXiv:1509.06461 — https://arxiv.org/abs/1509.06461 （overestimation 论点与 target 公式核实）
- Wikipedia "Temporal difference learning" — https://en.wikipedia.org/wiki/Temporal_difference_learning （Sutton 1988 史料、TD(0) 公式、MC/DP 对比）
- Wikipedia "Q-learning" — https://en.wikipedia.org/wiki/Q-learning （更新公式、Watkins 1989 史料、off-policy）
- Wikipedia "SARSA" — https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action （SARSA 公式、on-policy、cliff-walking）
- CleanRL docs "DQN" — https://docs.cleanrl.dev/rl-algorithms/dqn/ （CartPole benchmark 488.69±16.11、与原版工程差异、变体列表）
- Gymnasium CartPole-v1 — https://gymnasium.farama.org/environments/classic_control/cart_pole/ （obs/action/reward/termination/solved 阈值）

**未逐字核实（明确标注的待核内容）**
- §6.4 的 ILQL / token-level Q 视角 / DPO 的 Q 解释 —【待核】，属训练知识中的方向性记忆，本章未对具体论文做 WebFetch，引用前需补核实。
