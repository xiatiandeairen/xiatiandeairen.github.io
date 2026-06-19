---
title: "策略梯度 REINFORCE / Actor-Critic"
slug: "6-03"
collection: "tech-library"
group: "强化学习"
order: 6003
summary: "强化学习域 · 深化篇 面向有工程经验、要把 RL（含 LLM 对齐用 RL）吃透成专家的读者。术语保留英文。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-14T20:11:38.000Z"
updatedAt: "2026-06-14T20:11:38.000Z"
---
> 强化学习域 · 深化篇
> 面向有工程经验、要把 RL（含 LLM 对齐用 RL）吃透成专家的读者。术语保留英文。

---

## TL;DR（先给结论）

- **策略梯度（Policy Gradient, PG）换了一个根本视角**：不再先学价值再贪心导出策略（value-based，第 2 章），而是把策略本身 `π_θ(a|s)` 参数化，对「期望回报」`J(θ)` 直接做梯度上升。它天然支持**随机策略**、**连续动作空间**，并且能优化我们真正在意的目标。
- **整套理论的地基是 Policy Gradient Theorem（Sutton et al. 1999/2000）**：它把「目标对参数的梯度」从「需要对环境分布求导」的不可解形式，化成「对 `∇_θ ln π_θ(a|s)` 求期望」的可采样形式——这一步靠的是 **log-derivative trick（score function）**。绕过了对 transition `P` 求导，这正是 model-free PG 能跑起来的原因。
- **REINFORCE（Williams 1992）是最朴素的实现**：用一整条 episode 的 Monte Carlo return `G_t` 当权重去推 `∇ ln π`。优点是无偏、实现极简；致命缺点是**方差极大**（整条轨迹的随机性全压在 `G_t` 上），训练慢且不稳。
- **两个降方差的核心招式**：(1) **baseline** —— 从 `G_t` 里减去一个只依赖状态的 `b(S_t)`，**不引入偏差**（因为 score function 的期望为 0），却显著降方差；(2) **reward-to-go / 因果性** —— 时刻 `t` 的动作只该为它**之后**的奖励负责。把 baseline 取成学出来的 `V(s)`，`G_t − V(s)` 就近似 **advantage** `A(s,a)`，于是自然过渡到 **Actor-Critic**。
- **Actor-Critic = 用一个 critic（价值网络）替代纯 MC 的 return**：actor 出策略，critic 估 `V`/`Q`/`A`，用 critic 的 bootstrapping 估计当权重。这把 MC 的高方差换成了「bias↑ variance↓」的可调权衡。**GAE（Schulman 2015）** 用一个 `λ` 参数在这条 bias-variance 轴上连续滑动，是现代 PPO/RLHF 的标配。
- **可运行 demo 三个**：(1) 纯 numpy + gymnasium 的 REINFORCE 解 CartPole-v1，看 return 上升；(2) 在同一框架上加 baseline / 改 Actor-Critic，**直接打印梯度方差对比**（这是本章实验核心，肉眼看 baseline 把方差砍下去）；(3) 一个最小的 advantage 计算单元测试，验证 `λ=0` / `λ=1` 两个极端退化成 one-step TD / Monte Carlo。**三个 demo 都设计为可运行，请在你的环境验证。**
- **和 LLM 对齐的连接（本章是关键一章）**：经典 RLHF 的 PPO（第 4-5 章）就是 Actor-Critic + clip 的 PG 方法；而 2024 年后大热的 **GRPO / RLOO / REINFORCE++** 其实是**回归 REINFORCE 本源**——砍掉 critic，用「一组 response 的奖励均值」当 baseline。本章把 REINFORCE→baseline→advantage 这条线吃透，你才能看懂为什么 GRPO 是「带 group baseline 的 REINFORCE」，以及为什么 RLHF 里 advantage 和 KL penalty 要那样拼。章末单列一节，明确标注 fact / 待核。

---

## 前置依赖（读这章前你该有的东西）

**概念前置**
- MDP 五元组 `(S, A, P, R, γ)`、policy `π`、state value `V^π`、action value `Q^π`、return `G_t = Σ_k γ^k r_{t+k+1}`。（第 1 章）
- TD error `δ_t = R_{t+1} + γV(S_{t+1}) − V(S_t)`、bootstrapping、bias-variance 的基本直觉。（第 2 章；本章的 Actor-Critic 直接用 TD error 当 advantage 的估计。）
- 一点点概率：`∇_θ ln p` 与 `∇_θ p` 的关系（`∇ ln p = ∇p / p`），期望的线性。

**工程前置（跑 demo 用）**
- Python 3.9+。
- demo 1/2：`numpy` + `torch`（CPU 即可，CartPole 不需要 GPU）+ `gymnasium`。
  ```bash
  pip install "gymnasium[classic-control]" torch numpy
  ```
- demo 3：只要 `numpy`。
- 想跑官方原版对照（可选）：PyTorch 官方 examples 的 `reinforce.py` / `actor_critic.py`，或 `pip install cleanrl`。

**读法建议**
- 想最快建直觉：先跑 §3 的 demo 1（REINFORCE）看它能解 CartPole，再跑 demo 2 看 baseline 把方差砍下去——「方差」这件抽象的事会立刻变具体。
- 想做 LLM 对齐：§1.3（log-derivative trick）+ §2（baseline / advantage 推导）+ §6（与 RLHF 的连接）三段最高优先；GRPO 的本质在 §6.3。
- 想抠工程：§5 失败模式（PG 为什么训练不稳）+ §4 方案对比里 on-policy 的样本效率坑。

---

## 模板六段

> 六段：① 设计考古 ② 核心机制（源码精读）③ 可运行 demo ④ 方案对比 ⑤ 失败模式与真坑 ⑥ 与 LLM 对齐的连接。

---

## §1 设计考古：从 value-based 到 policy-based

### 1.1 动机：value-based 解不了的三类问题

第 2 章的 Q-learning / DQN 走的是 **value-based** 路线：学 `Q*`，再用 `argmax_a Q(s,a)` 导出策略。这条路在很多任务上很好用，但有三个结构性短板：

1. **连续动作空间**。`argmax_a Q(s,a)` 在连续 `a` 上是一个内层优化问题，每步 forward 都要解一遍，代价高。策略方法直接输出动作（或动作分布的参数，如高斯的 `μ, σ`），一步到位。
2. **需要随机策略的场景**。value-based 导出的是确定性贪心策略（加 ε-greedy 只是探索补丁）。但在 partially observed、对抗博弈（如石头剪刀布的纳什均衡是均匀随机）等场景，**最优策略本身就是随机的**。PG 把策略参数化成分布，能自然学出随机策略。
3. **目标不对齐**。value-based 优化的是「值估计的准确性」（TD error），不是「期望回报」本身；策略是间接导出的。PG **直接对期望回报 `J(θ)` 做梯度上升**，优化目标和我们真正在意的东西一致。这一点在 RLHF 里尤其重要——我们要的是「人类偏好奖励最大」，PG 直接优化它。

> 一句话对比：value-based 是「先把地图画准，再走最短路」；policy-based 是「直接调整走路的习惯，让平均到达更快」。两者各有适用边界，§4 详述。

### 1.2 REINFORCE 的由来（Williams 1992）

最早的策略梯度算法 **REINFORCE** 出自 Ronald J. Williams 1992 年的论文 *"Simple statistical gradient-following algorithms for connectionist reinforcement learning"*（*Machine Learning*, 8(3-4):229-256）。【真实出处，WebFetch 核实：Wikipedia "Policy gradient method" + 论文标题】

> REINFORCE 这个名字是个缩写：**RE**ward **I**ncrement = **N**onnegative **F**actor × **O**ffset **R**einforcement × **C**haracteristic **E**ligibility。不用记，知道它指代「用 return 当权重去推 score function」这一类更新即可。

Williams 当年的语境是 connectionist（神经网络）单元的强化学习，但他给出的更新规则正是今天所有 PG 的雏形：朝着「让高回报的动作概率变大」的方向推参数。

### 1.3 Policy Gradient Theorem 与 log-derivative trick（地基）

我们要最大化期望回报。把一条轨迹记为 `τ = (s_0, a_0, s_1, a_1, ...)`，它的概率：

```
p_θ(τ) = ρ(s_0) · Π_t  π_θ(a_t | s_t) · P(s_{t+1} | s_t, a_t)
                       └── 我们能控制的 ──┘  └── 环境，θ 无关 ──┘
```

目标：

```
J(θ) = E_{τ ~ p_θ} [ R(τ) ]，   R(τ) = Σ_t γ^t r_t
```

直接求 `∇_θ J(θ) = ∇_θ ∫ p_θ(τ) R(τ) dτ` 会卡住，因为梯度作用在 `p_θ(τ)` 上，而 `p_θ(τ)` 里含环境 transition `P`——我们**没有 `P` 的解析式**，也无法对它求导。

**log-derivative trick（score function estimator）** 解开了这个结。核心恒等式：

```
∇_θ p_θ(τ) = p_θ(τ) · ∇_θ ln p_θ(τ)        （因为 ∇ ln p = ∇p / p）
```

代入后梯度重新变回一个**期望**（于是可采样）：

```
∇_θ J(θ) = ∫ ∇_θ p_θ(τ) · R(τ) dτ
         = ∫ p_θ(τ) · ∇_θ ln p_θ(τ) · R(τ) dτ
         = E_{τ ~ p_θ} [ ∇_θ ln p_θ(τ) · R(τ) ]
```

关键一步：`ln p_θ(τ)` 拆开后，`ρ(s_0)` 和 `P(s_{t+1}|s_t,a_t)` 都不含 `θ`，求 `∇_θ` 直接归零，**只剩 policy 项活下来**：

```
∇_θ ln p_θ(τ) = Σ_t ∇_θ ln π_θ(a_t | s_t)     ← 环境项全消失！
```

这就是为什么 PG 是 model-free 的——**我们绕过了对环境求导**。最终得到 **Policy Gradient Theorem**（Sutton, McAllester, Singh, Mansour, *Policy Gradient Methods for Reinforcement Learning with Function Approximation*, NIPS 1999, 2000 年正式发表）的一个常用形式【真实出处，WebFetch 核实：NeurIPS 1999 proceedings + Wikipedia】：

```
∇_θ J(θ) = E_{τ ~ π_θ} [ Σ_t  ∇_θ ln π_θ(a_t | s_t) · Ψ_t ]
```

这里 `Ψ_t` 是「给这个动作打的分」，可以有多种选择，**这正是 REINFORCE / baseline / Actor-Critic / GAE 的分水岭**：

| `Ψ_t` 的选择 | 名字 | 性质 |
|---|---|---|
| `R(τ)`（整条轨迹总回报） | 最朴素 REINFORCE | 无偏，方差极大 |
| `Σ_{t'≥t} γ^{t'} r_{t'}`（reward-to-go） | REINFORCE + 因果性 | 无偏，方差略小 |
| `G_t − b(s_t)`（减 baseline） | REINFORCE + baseline | 无偏（baseline 不依赖 a），方差更小 |
| `Q^π(s_t,a_t)` | Q Actor-Critic | 有偏（critic 估计误差），方差小 |
| `A^π(s_t,a_t) = Q − V` | Advantage Actor-Critic (A2C) | 有偏，方差小，最常用 |
| `δ_t = r_t + γV(s') − V(s)`（TD error） | one-step AC | 有偏，方差最小 |
| GAE(γ,λ) 加权 | 现代 PPO/RLHF | 用 λ 在 bias-variance 间连续调 |

> Spinning Up 给的 VPG 标准梯度形式就是用 advantage 的那一行【真实出处，WebFetch 核实：OpenAI Spinning Up VPG 页】：
> `∇_θ J(π_θ) = E_τ [ Σ_t ∇_θ log π_θ(a_t|s_t) · A^{π_θ}(s_t,a_t) ]`

记住这张表，后面所有源码都是在算「某种 `Ψ_t` × `∇ ln π`」。

---

## §2 核心机制：REINFORCE、baseline、advantage（源码精读）

### 2.1 REINFORCE 源码逐行（PyTorch 官方 examples）

下面是 PyTorch 官方 examples 仓库的 REINFORCE 实现，CartPole 上几十行解决。我按【真实源码 pytorch/examples@reinforcement_learning/reinforce.py】逐字摘录核心三段（WebFetch 核实）。

**(a) 策略网络**：输入 4 维状态，输出 2 个动作的 softmax 概率。

```python
# 【真实源码 pytorch/examples@reinforcement_learning/reinforce.py】
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []     # 存每步的 ln π(a|s)，供反传用
        self.rewards = []             # 存每步即时奖励

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)   # 输出动作分布
```

逐行注解：
- `affine2` 输出 2 个 logits，`softmax` 成概率——这就是把策略**参数化成一个 Categorical 分布**。连续动作时这里会换成输出高斯的 `μ, σ`（见 §2.4）。
- `saved_log_probs` / `rewards` 是两个 buffer：PG 的更新需要「每步的 `ln π`」和「每步的 reward」，先攒一整条 episode 再统一算。这是 **on-policy** 的直接体现——只能用当前策略采的数据。

**(b) 采样动作**：从分布里 sample，并把 `log_prob` 存下来（**注意是 sample 不是 argmax**，随机性是 PG 探索的来源）。

```python
# 【真实源码 pytorch/examples@reinforcement_learning/reinforce.py】
def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()                              # 随机采样，不是 argmax
    policy.saved_log_probs.append(m.log_prob(action))  # 存 ln π(a|s)，留住计算图
    return action.item()
```

> 工程关键：`m.log_prob(action)` 返回的是带 autograd 计算图的张量，append 进 list 后**计算图一直留着**，直到 `finish_episode` 里 `backward`。如果你不小心对它 `.detach()` 或 `.item()`，梯度就断了——这是手写 PG 最常见的静默 bug 之一（§5.4）。

**(c) episode 结束，算 return 与 loss**：这是 REINFORCE 的心脏。

```python
# 【真实源码 pytorch/examples@reinforcement_learning/reinforce.py】
def finish_episode():
    R = 0
    policy_loss = []
    returns = deque()
    for r in policy.rewards[::-1]:          # 从后往前
        R = r + args.gamma * R             # G_t = r_t + γ·G_{t+1}（递推算 reward-to-go）
        returns.appendleft(R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)  # ★ 标准化：穷人的 baseline
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)  # loss = −Σ ln π(a|s)·G_t（负号：梯度上升变下降）
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]                   # 清空 buffer，on-policy 数据用一次就丢
    del policy.saved_log_probs[:]
```

逐行注解（每一行都对应 §1 的某个公式）：
- **`R = r + args.gamma * R` 倒序递推**：这是在算 reward-to-go `G_t = Σ_{t'≥t} γ^{t'-t} r_{t'}`，自动实现了「动作只为之后的奖励负责」的**因果性**。比用整条 `R(τ)` 方差小。
- **`returns = (returns - mean) / (std + eps)`**：这一行是 REINFORCE 的灵魂工程 trick。减均值 = 一个 batch 内的 **baseline**（理论上无偏，因为它不依赖具体动作）；除以 std = **奖励归一化**，让学习率不被回报量级绑架。**没有这一行，CartPole 上 REINFORCE 经常学不动。** 注意：严格说除以 std 会引入一点偏差（std 也是当前 batch 的统计量），但实践中收益远大于这点偏差。
- **`-log_prob * R`**：对应 `−∇ ln π · G_t`。负号是因为 optimizer 做的是梯度**下降**，而我们要梯度**上升** `J`。`backward` 后 `optimizer.step()` 就完成了一次 PG 更新。
- **`del policy.rewards[:]`**：**on-policy 的硬约束**——这批数据是用旧策略采的，更新后策略变了，旧数据立刻作废，不能像 DQN 那样塞进 replay buffer 复用。这是 PG 样本效率低的根因（§4、§5.1）。

> 这段代码和 §1.3 的理论是**逐项对应**的：`for log_prob, R in zip(...): -log_prob * R` 就是 `Σ_t ∇ ln π_θ(a_t|s_t) · Ψ_t` 里 `Ψ_t = G_t − baseline`。把这段读透，PG 的源码你就读通一大半了。

### 2.2 为什么 baseline 不引入偏差（核心证明，必须懂）

这是整个 PG 工程里**最重要的一个数学事实**，RLHF 里 GRPO 的 group baseline、PPO 的 value baseline 全靠它撑着。

命题：对任意只依赖状态的 `b(s)`（不依赖动作 `a`），从权重里减去它**不改变梯度的期望**：

```
E_{a~π} [ ∇_θ ln π_θ(a|s) · b(s) ] = 0
```

证明（关键是 **score function 的期望恒为 0**）：

```
E_{a~π} [ ∇_θ ln π_θ(a|s) ]
  = Σ_a π_θ(a|s) · ∇_θ ln π_θ(a|s)
  = Σ_a π_θ(a|s) · (∇_θ π_θ(a|s) / π_θ(a|s))     ← log-derivative trick 反用
  = Σ_a ∇_θ π_θ(a|s)
  = ∇_θ Σ_a π_θ(a|s)
  = ∇_θ (1)                                        ← 概率归一，恒等于 1
  = 0
```

因为 `b(s)` 不依赖 `a`，可以提到求和外：`E[∇ln π · b(s)] = b(s)·E[∇ln π] = b(s)·0 = 0`。**证毕。**

直觉：`∇ ln π` 告诉你「往哪推能让这个动作更可能」。对所有动作按概率加权求和，等于在问「整体概率往哪挪」——但概率永远归一，没法整体挪，所以期望为 0。减一个 `b(s)` 相当于给所有动作的「分数」统一减一个常数，不改变它们之间的**相对**高低，于是梯度方向（期望意义上）不变，但**绝对量级的抖动变小**了——这就是降方差的来源。

> Wikipedia 的原话【WebFetch 核实】："the expectation of the score function is zero, conditional on any present or past state"。这正是上面这个推导。
>
> ⚠️ 但要注意：上面只证了**期望不变（无偏）**。`b(s)` 取什么**最优**（方差最小）是另一个问题——理论最优 baseline 不是 `V(s)` 而是一个对 `∇ln π` 加权的量，但 `V(s)` 实践中已经很好且好估，所以大家都用 `V(s)`。

### 2.3 Actor-Critic 源码逐行（PyTorch 官方 examples）

把 baseline 从「batch 均值」升级成「学出来的 `V(s)`」，REINFORCE 就变成 **Actor-Critic**。下面是 PyTorch 官方 examples 的 AC 实现【真实源码 pytorch/examples@reinforcement_learning/actor_critic.py，WebFetch 核实】。

**(a) 双头网络**：共享躯干，一个头出策略（actor），一个头出 `V(s)`（critic）。

```python
# 【真实源码 pytorch/examples@reinforcement_learning/actor_critic.py】
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])  # 同时存 ln π 和 V(s)

class Policy(nn.Module):
    """implements both actor and critic in one model"""
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)   # actor: 出动作分布
        self.value_head = nn.Linear(128, 1)    # critic: 出 V(s)，标量
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_prob = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_prob, state_values
```

**(b) 采样**：除了存 `log_prob`，还把 critic 的 `V(s)` 一起存进 `SavedAction`。

```python
# 【真实源码 pytorch/examples@reinforcement_learning/actor_critic.py】
def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))  # 同存
    return action.item()
```

**(c) 算两个 loss**：actor loss 用 advantage 加权，critic loss 用回归拟合 return。

```python
# 【真实源码 pytorch/examples@reinforcement_learning/actor_critic.py】
def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []     # actor 的 loss
    value_losses = []      # critic 的 loss
    returns = []
    for r in model.rewards[::-1]:
        R = r + args.gamma * R          # 还是 MC return（这个实现用 MC 当 critic 的 target）
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()    # ★ advantage = G_t − V(s_t)，critic 当 baseline
        policy_losses.append(-log_prob * advantage)                 # actor: −ln π · A
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))  # critic: 回归到 G_t
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()  # 两 loss 相加
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]
```

逐行注解（与 REINFORCE 的差异就是 PG → AC 的全部要点）：
- **`advantage = R - value.item()`**：这是核心改动。baseline 不再是 batch 均值，而是 critic 估的 `V(s_t)`。`G_t − V(s_t)` 就是 advantage `A(s,a)` 的一个估计——「这个动作比该状态的平均水平好多少」。
- **`value.item()`**：算 advantage 时对 `V` 做了 `.item()`（detach），**advantage 不回传到 critic**。这是对的：advantage 只该作为 actor 的权重（一个数），不该让 actor loss 去训 critic。critic 由它自己的 `value_losses` 训。这个 detach 是 AC 工程里另一个易错点（忘了 detach 会让两个 loss 互相污染梯度）。
- **`F.smooth_l1_loss(value, R)`**：critic 的监督信号是 MC return `R`。注意这个**官方 demo 用的是 MC return 当 critic target**（`R` 是整条 reward-to-go），所以它其实是 "Monte Carlo Actor-Critic"。更典型的 one-step AC 会用 TD target `r + γV(s')` 当 critic 的回归目标——那才是真正用上 bootstrapping、把方差进一步压低的版本（demo 2 我会给 TD 版对比）。
- **`loss = policy_losses.sum() + value_losses.sum()`**：actor 和 critic 的 loss 简单相加一起 backward。生产级实现（如 PPO）通常给 value loss 一个系数 `vf_coef`（如 0.5）、再加一个 entropy bonus 鼓励探索——见 §2.4 的 CleanRL PPO。

> REINFORCE → AC 的本质：**用一个 learned function `V(s)` 替换 Monte Carlo 的 baseline / 甚至替换 return 本身**。代价是 critic 有估计误差（引入 bias），收益是方差大降 + 不必等 episode 结束就能更新（用 TD 时）。这正是 §1.3 表格里从「无偏高方差」往「有偏低方差」滑的那一步。

### 2.4 生产级形态：CleanRL PPO 的 Agent（advantage + GAE + entropy）

把上面两段放进生产语境，看现代 RL 怎么写。CleanRL 的单文件 PPO 是社区公认的 reference 实现。这里摘 Agent 与 loss 的核心【真实源码 vwxyzjn/cleanrl@cleanrl/ppo.py，WebFetch 核实】。

**(a) Agent：actor + critic 两个独立网络**（注意 `layer_init` 给 actor 末层 `std=0.01`，初始策略接近均匀——一个稳定性 trick）：

```python
# 【真实源码 vwxyzjn/cleanrl@cleanrl/ppo.py】
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),  # 末层小 std
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)        # 注意用 logits 不是 probs，数值更稳
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
```

**(b) GAE 计算**（这是 advantage 的现代标准做法，倒序递推一遍，是 PPO/RLHF 的标配）：

```python
# 【真实源码 vwxyzjn/cleanrl@cleanrl/ppo.py】
with torch.no_grad():
    next_value = agent.get_value(next_obs).reshape(1, -1)
    advantages = torch.zeros_like(rewards).to(device)
    lastgaelam = 0
    for t in reversed(range(args.num_steps)):
        if t == args.num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]  # TD residual δ_t
        advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
    returns = advantages + values     # critic 的回归 target = advantage + V
```

这段就是 **GAE(γ,λ)** 的代码实现（§2.5 给公式）：`delta` 是 TD residual `δ_t`，`lastgaelam` 倒序累积 `δ + γλ·(下一步的 GAE)`。`nextnonterminal` 处理 episode 边界（终止后不 bootstrapping）。

**(c) PPO 的 policy loss / value loss**（在 advantage 上加了 clip，这是 PPO 相对 vanilla PG 的核心，第 4 章详讲；这里先看 advantage 怎么用）：

```python
# 【真实源码 vwxyzjn/cleanrl@cleanrl/ppo.py】
mb_advantages = b_advantages[mb_inds]
if args.norm_adv:
    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)  # 又见标准化

pg_loss1 = -mb_advantages * ratio                                       # 普通 PG 项
pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)  # clip 项
pg_loss = torch.max(pg_loss1, pg_loss2).mean()
...
v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()           # critic 回归（简化分支）
entropy_loss = entropy.mean()
loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef  # 三项合一
```

注意 `loss = pg_loss − ent_coef·entropy + vf_coef·v_loss`——这就是 §2.3 那个「actor loss + critic loss」的生产版，多了个 **entropy bonus**（鼓励策略别过早塌成确定性，维持探索）和系数。**`mb_advantages` 这一行的标准化又出现了**——降方差的标准化 trick 从 REINFORCE 一路贯穿到 PPO。把这条线看清，你会发现 RLHF 的 PPO 在「advantage 怎么算、怎么用」上和教科书 REINFORCE 是同一套骨架。

**(d) 连续动作的 Agent**（高斯策略，RLHF 之外的机器人控制常用；也帮助理解「策略=分布参数」）：

```python
# 【真实源码 vwxyzjn/cleanrl@cleanrl/ppo_continuous_action.py】
self.actor_mean = nn.Sequential(... layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01))
self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))  # 状态无关的 log std

def get_action_and_value(self, x, action=None):
    action_mean = self.actor_mean(x)
    action_logstd = self.actor_logstd.expand_as(action_mean)
    action_std = torch.exp(action_logstd)
    probs = Normal(action_mean, action_std)     # 高斯策略
    if action is None:
        action = probs.sample()
    return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)  # 多维度求和
```

注解：连续动作把 Categorical 换成 `Normal(μ, σ)`，`μ` 由网络出、`logstd` 是一个独立可学参数（state-independent，是 CleanRL 的设计选择）。`log_prob(action).sum(1)` 对动作各维求和——因为多维高斯（对角协方差）的 log prob 是各维之和。**这解释了 §1.1 说的「PG 天然支持连续动作」**：只要换一个分布族即可，更新公式一字不改。

### 2.5 GAE 的公式（把 §2.4(b) 的代码翻译成数学）

GAE 出自 Schulman, Moritz, Levine, Jordan, Abbeel, *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, 2015 (arXiv:1506.02438)【真实出处，WebFetch 核实标题/作者/年份；下面公式为标准形式，PDF 为二进制未能逐字摘录，标「示意，非逐字」】。

定义 TD residual：

```
δ_t = r_t + γ·V(s_{t+1}) − V(s_t)         （这就是第 2 章的 TD error）
```

GAE 把不同步数的优势估计用 `(γλ)^l` 指数加权求和【示意，非逐字，标准形式】：

```
Â_t^{GAE(γ,λ)} = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
              = δ_t + (γλ)δ_{t+1} + (γλ)^2 δ_{t+2} + ...
```

两个极端（这就是 demo 3 要数值验证的）：

```
λ = 0:  Â_t = δ_t = r_t + γV(s_{t+1}) − V(s_t)         → one-step TD，bias 最大 / variance 最小
λ = 1:  Â_t = Σ_{l≥0} γ^l δ_{t+l} = G_t − V(s_t)        → Monte Carlo advantage，bias 最小 / variance 最大
```

> `λ=1` 退化成 `G_t − V(s_t)` 这件事可以用望远镜求和（telescoping）证：把 `δ_{t+l}` 展开，相邻的 `V` 项逐对消掉，只剩 `Σγ^l r_{t+l} − V(s_t) = G_t − V(s_t)`。demo 3 会用 numpy 把这两个极端跑出来对账。

`λ` 就是 §1.3 表格那条 bias-variance 轴上的旋钮。PPO/RLHF 默认 `λ≈0.95`，`γ≈0.99`——偏向低 bias 但留一点 variance 控制。**把 GAE 理解成「可调的 advantage 估计器」，你就理解了现代 PG 的 advantage 是怎么来的。**

---

## §3 可运行 demo（重中之重）

> 三个 demo 层层递进：demo 1 跑通 REINFORCE → demo 2 加 baseline / 改 AC 并**直接量化梯度方差**（本章实验核心）→ demo 3 用纯 numpy 验证 GAE 的 λ 极端。**全部设计为可运行，请在你的环境验证。** 我把它们写成可直接 `python xxx.py` 的独立脚本。

### 3.0 环境

```bash
python -V                     # 3.9+
pip install "gymnasium[classic-control]" torch numpy
```

任务统一用 `CartPole-v1`：状态 4 维，动作 2 个（左/右推车），每步存活 +1 奖励，最多 500 步。`return` 越接近 500 越好。

### 3.1 demo 1：最小 REINFORCE（看 return 上升）

把它存成 `reinforce_min.py`。这是一个**自包含**的实现（不依赖 PyTorch examples 仓库），但骨架与 §2.1 的官方源码一一对应。

```python
# reinforce_min.py
# 设计为可运行,请在你的环境验证。依赖: gymnasium[classic-control], torch, numpy
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym

torch.manual_seed(0); np.random.seed(0)

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, n_act):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, n_act),
        )
    def forward(self, x):
        return self.net(x)                       # 返回 logits

def run_episode(env, policy):
    obs, _ = env.reset(seed=int(np.random.randint(1e6)))
    log_probs, rewards = [], []
    done = False
    while not done:
        logits = policy(torch.as_tensor(obs, dtype=torch.float32))
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))   # 留计算图
        obs, r, term, trunc, _ = env.step(action.item())
        rewards.append(r)
        done = term or trunc
    return log_probs, rewards

def discounted_returns(rewards, gamma):
    R, out = 0.0, []
    for r in reversed(rewards):
        R = r + gamma * R
        out.insert(0, R)
    return torch.tensor(out, dtype=torch.float32)

def main():
    env = gym.make("CartPole-v1")
    policy = PolicyNet(env.observation_space.shape[0], env.action_space.n)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
    gamma, eps = 0.99, 1e-8
    running = 0.0
    for ep in range(1, 801):
        log_probs, rewards = run_episode(env, policy)
        returns = discounted_returns(rewards, gamma)
        returns = (returns - returns.mean()) / (returns.std() + eps)   # 标准化(穷人 baseline)
        loss = torch.stack([-lp * R for lp, R in zip(log_probs, returns)]).sum()
        opt.zero_grad(); loss.backward(); opt.step()

        ep_ret = sum(rewards)
        running = 0.05 * ep_ret + 0.95 * running
        if ep % 50 == 0:
            print(f"ep {ep:4d} | last_return {ep_ret:6.1f} | running {running:6.1f}")
        if running > 475:
            print(f"Solved at episode {ep}! running={running:.1f}")
            break
    env.close()

if __name__ == "__main__":
    main()
```

**运行**：
```bash
python reinforce_min.py
```

**预期输出（数值随机种子/torch 版本会有差异，趋势应一致）**：running return 从十几一路爬升，通常几百个 episode 内冲到 400+：
```
ep   50 | last_return   34.0 | running   28.7
ep  100 | last_return   78.0 | running   55.2
ep  200 | last_return  201.0 | running  165.9
ep  300 | last_return  410.0 | running  333.1
Solved at episode 3xx! running=4xx.x
```
> ⚠️ REINFORCE 方差大，**不同种子收敛速度差异明显**，偶尔会卡住甚至中途回落——这恰恰是 §5.1 要讲的「PG 训练不稳」的第一手体感。如果你跑出来很慢，换个种子或把 `lr` 调到 `5e-3` 再试。这种「换种子结果差很多」本身就是本章要传达的 fact。

### 3.2 demo 2（本章实验核心）：baseline / Actor-Critic vs 纯 REINFORCE，量化梯度方差

只看 return 曲线还不够——本章的核心论点是「**baseline 降的是方差**」。这个 demo 直接把**梯度的方差**打印出来，让抽象的「方差」变成一个能比大小的数字。三种配置跑同样的 episode，统计 policy 第一层权重梯度的方差。

存成 `variance_compare.py`：

```python
# variance_compare.py
# 设计为可运行,请在你的环境验证。依赖: gymnasium[classic-control], torch, numpy
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym

def make_nets(obs_dim, n_act):
    actor = nn.Sequential(nn.Linear(obs_dim,128), nn.ReLU(), nn.Linear(128,n_act))
    critic = nn.Sequential(nn.Linear(obs_dim,128), nn.ReLU(), nn.Linear(128,1))
    return actor, critic

def rollout(env, actor):
    obs,_ = env.reset(seed=int(np.random.randint(1e6)))
    states, log_probs, rewards = [], [], []
    done=False
    while not done:
        s = torch.as_tensor(obs, dtype=torch.float32)
        dist = Categorical(logits=actor(s))
        a = dist.sample()
        states.append(s); log_probs.append(dist.log_prob(a))
        obs, r, term, trunc, _ = env.step(a.item()); rewards.append(r)
        done = term or trunc
    return states, log_probs, rewards

def returns_to_go(rewards, gamma):
    R,out=0.0,[]
    for r in reversed(rewards):
        R=r+gamma*R; out.insert(0,R)
    return torch.tensor(out, dtype=torch.float32)

def grad_vector(actor):
    # 取第一层 weight 的梯度,拉平成向量,用于统计方差
    g = actor[0].weight.grad.detach().reshape(-1).clone()
    return g

def measure(mode, episodes=300, gamma=0.99, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    env = gym.make("CartPole-v1")
    obs_dim, n_act = env.observation_space.shape[0], env.action_space.n
    actor, critic = make_nets(obs_dim, n_act)
    opt_a = torch.optim.Adam(actor.parameters(), lr=1e-2)
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-2)
    grad_norms_sq = []   # 收集每次更新的梯度向量,最后算逐元素方差的均值
    grads = []
    rets = []
    for ep in range(episodes):
        states, log_probs, rewards = rollout(env, actor)
        G = returns_to_go(rewards, gamma)
        rets.append(sum(rewards))

        if mode == "reinforce":
            weight = (G - G.mean()) / (G.std()+1e-8)        # 仅标准化
        elif mode == "baseline":
            V = critic(torch.stack(states)).squeeze(-1)
            adv = (G - V).detach()                          # G - V(s) 当 advantage
            adv = (adv - adv.mean())/(adv.std()+1e-8)
            weight = adv
            # 训 critic: 回归到 G
            opt_c.zero_grad()
            v_loss = ((critic(torch.stack(states)).squeeze(-1) - G)**2).mean()
            v_loss.backward(); opt_c.step()
        elif mode == "ac_td":
            V = critic(torch.stack(states)).squeeze(-1)
            # one-step TD advantage: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            with torch.no_grad():
                Vnext = torch.cat([V[1:], torch.zeros(1)])  # 末步后 V=0(终止)
                td_target = torch.tensor(rewards, dtype=torch.float32) + gamma*Vnext
                adv = (td_target - V).detach()
            adv = (adv - adv.mean())/(adv.std()+1e-8)
            weight = adv
            opt_c.zero_grad()
            v_loss = ((critic(torch.stack(states)).squeeze(-1) - td_target)**2).mean()
            v_loss.backward(); opt_c.step()

        opt_a.zero_grad()
        loss = torch.stack([-lp*w for lp,w in zip(log_probs, weight)]).sum()
        loss.backward()
        grads.append(grad_vector(actor))                    # 记录这次的梯度向量
        opt_a.step()

    env.close()
    G_mat = torch.stack(grads)                              # [episodes, n_params]
    # 逐参数算方差,再对所有参数取均值 -> 一个标量"梯度方差"
    per_param_var = G_mat.var(dim=0)
    return per_param_var.mean().item(), float(np.mean(rets[-50:]))

if __name__ == "__main__":
    for mode in ["reinforce", "baseline", "ac_td"]:
        gvar, last_ret = measure(mode, episodes=300, seed=0)
        print(f"{mode:10s} | mean grad variance {gvar:.4e} | last50 avg return {last_ret:6.1f}")
```

**运行**：
```bash
python variance_compare.py
```

**预期输出（数值会变，但相对大小关系是本 demo 的论点）**：
```
reinforce  | mean grad variance 3.xxe-02 | last50 avg return 1xx.x
baseline   | mean grad variance 8.xxe-03 | last50 avg return 2xx.x
ac_td      | mean grad variance 4.xxe-03 | last50 avg return 2xx.x
```
你应该观察到：**`reinforce` 的梯度方差最大，加了 `baseline`（critic 当 baseline）后明显下降，用 one-step TD 的 `ac_td` 进一步下降**。这就是 §1.3 表格那条 bias-variance 轴在真实代码里的体现——从上往下，variance 一路降。return 通常也更稳更高。

> 实验诚实声明【这是本章的 fact，但请你复现】：
> - 「方差降低」的**方向**在 CartPole 上是稳健的、可复现的；但**具体倍数**对种子、网络初始化、critic 学得好不好很敏感。`ac_td` 偶尔会因为 critic 早期估计差而 advantage 偏，导致它不一定每次都严格低于 `baseline`——**这本身就是「critic 引入 bias」的真实代价**，正好印证 §1.3 表格。
> - 建议你跑 3-5 个不同 `seed`（改 `measure(..., seed=k)`）看趋势，而不是只信一次结果。这正是「N=1 不算 feasibility」的工程纪律。

### 3.3 demo 3：纯 numpy 验证 GAE 的 λ 两极（对账 §2.5 的公式）

不依赖 torch/gym，纯 numpy。验证 `λ=1` 时 GAE 退化成 `G_t − V(s_t)`（Monte Carlo advantage），`λ=0` 时退化成 one-step TD residual `δ_t`。这是把 §2.5 的数学**用代码对账**。

存成 `gae_check.py`：

```python
# gae_check.py
# 设计为可运行,请在你的环境验证。依赖: numpy
import numpy as np

def gae(rewards, values, gamma, lam, last_value=0.0):
    """倒序递推 GAE,与 cleanrl ppo.py 的实现等价(单环境版)。"""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float64)
    lastgae = 0.0
    for t in reversed(range(T)):
        v_next = values[t+1] if t+1 < T else last_value
        delta = rewards[t] + gamma * v_next - values[t]      # TD residual δ_t
        lastgae = delta + gamma * lam * lastgae
        adv[t] = lastgae
    return adv

def mc_advantage(rewards, values, gamma, last_value=0.0):
    """G_t - V(s_t),其中 G_t 用 bootstrap 末值。"""
    T = len(rewards)
    G = np.zeros(T)
    running = last_value
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running
        G[t] = running
    return G - values

def td_residual(rewards, values, gamma, last_value=0.0):
    T = len(rewards)
    out = np.zeros(T)
    for t in range(T):
        v_next = values[t+1] if t+1 < T else last_value
        out[t] = rewards[t] + gamma * v_next - values[t]
    return out

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rewards = rng.normal(size=8)
    values  = rng.normal(size=8)
    gamma   = 0.99

    gae_l1 = gae(rewards, values, gamma, lam=1.0)
    gae_l0 = gae(rewards, values, gamma, lam=0.0)
    mc     = mc_advantage(rewards, values, gamma)
    td     = td_residual(rewards, values, gamma)

    print("λ=1 GAE == MC advantage (G_t - V):", np.allclose(gae_l1, mc))
    print("λ=0 GAE == one-step TD residual δ_t:", np.allclose(gae_l0, td))
    print("max |gae_l1 - mc| =", np.max(np.abs(gae_l1 - mc)))
    print("max |gae_l0 - td| =", np.max(np.abs(gae_l0 - td)))
```

**运行**：
```bash
python gae_check.py
```

**预期输出**：
```
λ=1 GAE == MC advantage (G_t - V): True
λ=0 GAE == one-step TD residual δ_t: True
max |gae_l1 - mc| = 0.0
max |gae_l0 - td| = 0.0
```
两个 `True` 就是 §2.5 公式的数值证据：GAE 是 one-step TD 和 Monte Carlo 之间的连续插值。你可以改 `lam=0.95` 看它落在两者之间。**这个 demo 是确定性的（无网络训练），应当严格输出 True；如果不是，说明你改动引入了 bug，正好用它当回归测试。**

---

## §4 方案对比

### 4.1 算法对照表

| 维度 | REINFORCE | REINFORCE + baseline | Actor-Critic (A2C/one-step) | DQN（第 2 章，对照） |
|---|---|---|---|---|
| 路线 | policy-based | policy-based | policy-based (actor+critic) | value-based |
| 权重 `Ψ_t` | `G_t`（MC return） | `G_t − b(s)` | `A(s,a)`≈`δ_t` 或 GAE | 不适用（学 Q） |
| 偏差 | 无偏 | 无偏 | 有偏（critic 误差） | 有偏（bootstrapping） |
| 方差 | 极大 | 中 | 小（可调） | 小 |
| 何时更新 | episode 结束（MC） | episode 结束 | 每步/每段（可 bootstrap） | 每步 |
| on/off policy | on-policy | on-policy | on-policy（标准 AC） | off-policy |
| 样本复用 | 用一次即弃 | 用一次即弃 | 用一次即弃 | replay buffer 反复用 |
| 连续动作 | 天然支持 | 天然支持 | 天然支持 | 困难（argmax 难） |
| 随机策略 | 天然 | 天然 | 天然 | 否（确定性+ε） |
| 实现复杂度 | 最低 | 低 | 中 | 中（replay+target） |
| 样本效率 | 低 | 低 | 中 | 高 |

### 4.2 具体场景该选谁

- **离散动作、状态可枚举、采样便宜、追样本效率** → value-based（DQN/Q-learning）。replay buffer 复用数据，off-policy 省样本。PG 在这里通常打不过。
- **连续动作（机器人控制、运动）** → policy-based（PPO/SAC/TD3）。`argmax_a Q` 在连续空间太贵，PG 直接出动作分布。这是 PG 的主场。
- **最优策略本身随机（博弈、部分可观测）** → 必须 policy-based。value-based 的确定性贪心在这里是错的目标。
- **目标必须是「直接最大化某个标量奖励」且模型很大、采样很贵（LLM 对齐）** → PG 家族（PPO / GRPO）。但有自己的子选择（§6）。
- **教学 / 调试 / 建直觉** → REINFORCE。几十行、无偏、能跑通，是理解一切 PG 的起点；但**别拿它做生产**——方差太大。

### 4.3 不适用边界（PG 的硬伤，必须知道）

1. **样本效率天生低（on-policy 的诅咒）**：标准 PG 每次策略更新后旧数据全部作废（§2.1 的 `del buffer`）。在采样昂贵的场景（真实机器人、大模型 rollout）这是巨大成本。PPO 用 importance ratio + clip 允许同一批数据**多 epoch 复用**来缓解，但本质仍是 on-policy，复用次数有限。
2. **局部最优 + 对学习率敏感**：PG 是在策略空间做局部梯度上升，容易卡局部最优；步子大了策略一步崩坏（一次差更新把策略推到很差的区域，之后采的数据更差，雪崩）。这是 TRPO/PPO 要做 trust region / clip 的根本原因（第 4 章）。
3. **方差与 credit assignment**：长 horizon 任务里，`G_t` 把很远的奖励都算到当前动作头上，方差爆炸、credit 分配模糊。baseline / GAE 缓解但不根治。
4. **不保证单调改进**：vanilla PG 没有「这次更新一定不比上次差」的保证（不像 policy iteration）。这是 RLHF 里 PPO 训练「reward 上去了但模型崩了 / reward hacking」类问题的温床。

---

## §5 失败模式与真坑（扎根）

### 5.1 真坑：PG 训练不稳，且「换个种子结果差很多」

**现象**：同一份 REINFORCE 代码，seed=0 几百 episode 解了 CartPole，seed=1 可能卡在 100 分上不去，甚至学到一半 return 崩回去。

**根因**：
- **高方差梯度**：MC return 的随机性 → 梯度估计噪声大 → 更新方向忽左忽右。
- **on-policy 正反馈**：一次坏更新让策略变差 → 采到更差的数据 → 更差的更新。没有 replay buffer 当「稳定锚」（对比 DQN）。
- **策略塌缩**：entropy 掉太快，策略过早变确定性，探索停止，卡在次优。

**缓解**：
- 标准化 return / advantage（§2.1 那行，几乎必加）。
- 加 baseline / critic（demo 2 验证了降方差）。
- 加 entropy bonus（§2.4 的 `ent_coef`）维持探索。
- 评估时跑**多个 seed 看分布**，别信单次（呼应「N=1 不算 feasibility」）。
- 想要稳，直接上 PPO 的 clip（第 4 章），别拿 vanilla PG 做生产。

### 5.2 真坑：忘了「因果性 / reward-to-go」，用整条 R(τ) 当权重

**现象**：训练极慢或不收敛。

**根因**：用 `R(τ)`（整条轨迹总回报）给**每一个**动作当权重，等于让时刻 `t` 的动作为它**之前**的奖励也负责——那些奖励和这个动作因果无关，纯粹是噪声，白白抬高方差。正确做法是 reward-to-go `Σ_{t'≥t} γ^{t'-t} r_{t'}`（§2.1 的倒序递推）。

**检查**：看你的 return 计算是不是 per-timestep 的 reward-to-go，而不是一个 episode 一个标量广播给所有步。

### 5.3 真坑：baseline / advantage 没 detach，梯度互相污染

**现象**：critic 学不好，或 actor loss 莫名其妙、训练发散。

**根因**：advantage `A = G − V(s)` 里的 `V(s)` 如果带着计算图进了 actor loss（`-log_prob * A`），那么 actor 的梯度会顺着 `A` 流回 critic，把 critic 往「让 advantage 变大」的方向拉——这是错的，critic 该被它自己的回归 loss 训。**官方 AC 源码用 `value.item()`、CleanRL 用 `b_advantages`（提前 detach）就是为了切断这条路径**（§2.3 注解）。

**检查**：算 advantage 当 actor 权重时，确保它对 critic 参数 `.detach()`。

### 5.4 真坑：log_prob 计算图被切断 / 采样用了 argmax

**现象**：loss 能算、能 backward，但参数几乎不动，return 不涨。

**根因**：
- 把 `log_prob` `.detach()` 或先 `.item()` 再存——计算图断了，`backward` 推不到 policy 参数。
- 采样误用 `argmax(probs)` 而非 `dist.sample()`——`argmax` 不可导，且没了探索的随机性。PG 必须从分布 sample（§2.1 注解）。

**检查**：`select_action` 里存的应是 `dist.log_prob(action)` 这个**带图**的张量；动作是 `dist.sample()`。

### 5.5 真坑：on-policy 数据当 off-policy 复用

**现象**：训练起初正常，复用旧 batch 多次后越来越差。

**根因**：策略更新后，旧数据是用**旧策略**采的，分布已经变了。vanilla PG 没有 importance correction，直接复用 = 用错误分布的样本估梯度 → 偏。PPO 的 ratio `π_new/π_old` + clip 才让有限次复用合法（第 4 章）。

**检查**：vanilla PG 每次更新后必须重采（`del buffer`）；要复用就上 PPO 的机制。

---

## §6 与 LLM 对齐的连接（本章关键）

> 本节明确区分 **fact**（有源码/论文支撑、WebFetch 核实）与 **待核**（合理但未逐一核实的推断）。

### 6.1 RLHF 的 PG 内核就是本章这套（fact）

经典 RLHF（InstructGPT 路线）的 RL 阶段用 **PPO**——它就是 **Actor-Critic + clip 的策略梯度方法**（§2.4 的 CleanRL PPO 是同一算法骨架）。把语言模型映射到本章术语：

- **policy `π_θ`** = 语言模型本身，状态 = 已生成的 token 前缀，动作 = 下一个 token。
- **reward** = reward model（RM）打的分（通常只在序列末尾给，token 级用 KL 补）。
- **critic / value `V(s)`** = 一个额外的 value head，估「当前前缀的预期最终奖励」，用来算 advantage（正是 §2.3 的角色）。
- **advantage** = 用 GAE（§2.5）算，PPO 默认 `λ≈0.95, γ≈1.0 或 0.99`。
- **loss** = `pg_loss(clip) − ent_coef·entropy + vf_coef·v_loss`（§2.4(c) 那一行的 LLM 版）。

所以你在 §2 学的「`-log_prob * advantage`」「baseline 不引入偏差」「advantage 标准化」**原封不动**搬到 RLHF。【fact：HuggingFace blog "Navigating the RLHF Landscape" 与 TRL PPOTrainer 实现，WebFetch/WebSearch 核实】

### 6.2 KL penalty：RLHF 特有的 reward 改造（fact）

LLM RLHF 在 reward 里加一项 **KL penalty**，把每步 reward 改成：

```
r_t = RM_score（仅末步） − β · KL( π_θ(·|s_t) ‖ π_ref(·|s_t) )
```

`π_ref` 是 SFT 之后冻结的参考模型。作用：防止策略为了刷 RM 分数跑得离原模型太远（语言退化 / reward hacking）。【fact：WebSearch 核实，"it is customary to include a KL penalty in the reward signal to prevent the model's output distribution from diverging too far from the pre-trained one"】

> 这正是 §4.3 第 4 点「PG 不保证单调改进、易 reward hacking」在 LLM 场景的具体化——KL penalty 是工程上给 PG 加的「别跑太远」护栏，和 PPO clip 是互补的两道保险。

### 6.3 GRPO / RLOO / REINFORCE++：回归 REINFORCE 本源（fact + 待核）

2024 年后 RLHF 的一个大趋势是**把 critic 砍掉**，回到更接近 REINFORCE 的形态。原因很现实：critic（value model）和 policy 一样大，**多占一份显存/GPU**。这条线直接呼应本章 §2.2 的核心事实——**baseline 不一定要是学出来的 `V(s)`，任何不依赖动作的量都行**。

- **GRPO（Group Relative Policy Optimization）**：对同一个 prompt 采样**一组 `G` 个 response**，用这组奖励的**均值**当 baseline，advantage = `(r_i − mean(group)) / (std(group) + eps)`。**这本质就是「用 group 均值当 baseline 的 REINFORCE」**——没有 critic，baseline 是 Monte Carlo 的组内统计量。【fact：WebSearch 核实，"GRPO ... replacing it with a simpler Monte Carlo estimation, computing the relative advantage of each sample as its reward minus the group mean reward"】
- **RLOO（REINFORCE Leave-One-Out）**：baseline 用「**留一**」——第 `i` 个样本的 baseline 是其余 `G−1` 个样本奖励的均值。同样是 critic-free 的 REINFORCE 变体。【fact：WebSearch 核实，属于 "ReMax, RLOO, GRPO ... remove the critic network and estimate advantage using statistics from multiple responses to the same prompt"】
- **REINFORCE++**：在经典 REINFORCE 上加 **token-level KL penalty + trust-region clipping** 提升稳定性。名字直接告诉你它的血统。【fact：WebSearch 核实，"REINFORCE++ ... utilizing token-level KL penalties and trust region clipping to improve stability in RLHF"】

> 把本章读透的回报在这里兑现：**GRPO 不是新东西**——它是 §2.2 那个「baseline 不引入偏差」定理的直接应用，把「learned `V(s)`」换成「同组样本均值」。你甚至能在 demo 2 的框架上自己实现一个玩具 GRPO（见五件套代码题）。
>
> 待核：上面 GRPO/RLOO 的具体 advantage 公式（是否除 std、是否 leave-one-out 的精确形式）我用 WebSearch 核实了**概念与方向**，但**没有逐字摘到对应仓库的源码行**（尝试 fetch TRL 的 `grpo_trainer.py` 时关键 advantage 段落被截断未取到）。要写进生产代码请回去核 TRL 源码逐行。

### 6.4 一张「本章 → RLHF」的术语映射（fact）

| 本章概念（经典 RL） | RLHF 里的对应物 |
|---|---|
| policy `π_θ(a|s)` | 语言模型，逐 token 的条件分布 |
| trajectory `τ` | 一条生成的 response（token 序列） |
| return `G_t` | 序列奖励（RM 分，常仅末步）的 reward-to-go |
| baseline `b(s)` | value head `V(s)`（PPO）/ group 均值（GRPO）/ leave-one-out（RLOO） |
| advantage `A(s,a)` | GAE（PPO）/ 组内标准化奖励（GRPO） |
| `-log_prob * A` | PPO/GRPO 的 policy loss 主项（外面套 clip） |
| entropy bonus | 维持生成多样性（防塌缩） |
| —（经典 RL 无） | KL-to-ref penalty（RLHF 特有护栏） |

把这张表和 §6.1-6.3 连起来：**你已经具备读懂任意一篇 RLHF / PG 论文 method 章节的全部前置概念**。第 4-5 章会把 PPO 的 clip、TRPO 的 trust region、以及 DPO（绕过 RL 的对齐）补齐。

---

## 五件套（章末固化）

### 1. 一句话总结
策略梯度直接对「期望回报」做梯度上升，靠 log-derivative trick 把不可解的目标梯度变成可采样的 `E[∇ln π · Ψ]`；REINFORCE 用 MC return 当 `Ψ`（无偏但方差爆炸），减 baseline / 换 advantage（Actor-Critic / GAE）在不引入偏差或可控偏差下把方差压下来——这套骨架原封不动撑起了 RLHF 的 PPO 与回归本源的 GRPO/RLOO。

### 2. 心智模型（一图记住）
```
                       ∇J = E[ Σ_t  ∇ln π_θ(a_t|s_t) · Ψ_t ]
                                                        │
            ┌───────────────────────────────────────────┴───────────────────────────────┐
            ▼                  ▼                    ▼                  ▼                    ▼
         Ψ=G_t           Ψ=G_t − b(s)         Ψ=A=Q−V           Ψ=δ_t(one-step)      Ψ=GAE(γ,λ)
       REINFORCE     REINFORCE+baseline    Advantage AC       one-step AC          PPO/RLHF
      无偏/方差爆炸      无偏/方差↓           有偏/方差↓↓          有偏/方差最小        λ 调 bias↔variance
                          │                                                              │
                          └──── baseline 可以是:V(s)[PPO] / group均值[GRPO] / LOO[RLOO] ──┘
         降方差三连: reward-to-go(因果性) + 减baseline + 标准化(adv/return)
```

### 3. 常见误区 Top 5
1. **「baseline 会引偏」** —— 错。只要 `b(s)` 不依赖动作，期望严格不变（§2.2 证明）。这是 GRPO 用 group 均值的合法性来源。
2. **「Actor-Critic 一定比 REINFORCE 好」** —— 不绝对。critic 引入 bias，critic 没学好时 advantage 反而更差（demo 2 里 `ac_td` 偶尔不稳）。是 bias-variance 权衡，不是免费午餐。
3. **「PG 能像 DQN 那样用 replay buffer」** —— 错。标准 PG 是 on-policy，数据用一次即弃（§5.5）。要复用得上 PPO 的 ratio+clip。
4. **「GRPO 是全新算法」** —— 不。它是「带 group baseline 的 REINFORCE」，理论根就是 §2.2 那个老定理（§6.3）。
5. **「采样用 argmax / 把 log_prob detach 了无所谓」** —— 致命静默 bug。PG 必须 `sample()` 且保住 log_prob 的计算图（§5.4）。

### 4. 自测题（答案在正文，标了节号）
1. 用 log-derivative trick 推一遍 `∇_θ J(θ) = E[∇ln π_θ(τ)·R(τ)]`，并说明为什么环境 transition `P` 求导后消失。（§1.3）
2. 证明对任意 `b(s)`，`E_{a~π}[∇ln π·b(s)] = 0`。关键用到 score function 的什么性质？（§2.2）
3. 官方 REINFORCE 源码里 `returns = (returns - mean)/(std+eps)` 这一行做了哪两件事？哪件严格无偏、哪件略有偏？（§2.1）
4. GAE 在 `λ=0` 和 `λ=1` 分别退化成什么？各自 bias/variance 如何？（§2.5 + demo 3）
5. 把「baseline」「advantage」「KL penalty」分别映射到 RLHF 里的对应物。GRPO 的 baseline 是什么？（§6.4 + §6.3）
6. 为什么算 advantage 当 actor 权重时必须对 critic 的 `V(s)` 做 detach？不 detach 会怎样？（§5.3）

### 5. 代码题（= 扩展 demo，动手做）
1. **扩 demo 2**：再加一个 `mode="raw_return"`，用**整条 `R(τ)`**（不做 reward-to-go，一个标量广播给所有步）当权重，把它的梯度方差也打印出来，验证 §5.2「忘了因果性方差更大」。预期：`raw_return` 方差应**高于** `reinforce`（reward-to-go 版）。
2. **实现玩具 GRPO**：在一个简单 contextual bandit（或固定起点的短 CartPole）上，对同一个「prompt/初始状态」采样一组 `G=8` 条轨迹，用 `(r_i − mean)/(std+eps)` 当 advantage，**不训 critic**，跑 REINFORCE 更新。验证它能学起来，并和「用 learned `V(s)`」的 baseline 版比方差/收敛。这就是 §6.3 GRPO 的最小复现。
3. **GAE λ 扫描**：扩 demo 3，对 `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}` 算 advantage，在一个固定 rollout 上比较各 λ 下 advantage 的**方差**（用蒙特卡洛多条 rollout 估），画出「λ vs advantage 方差」曲线，肉眼确认 λ↑ → variance↑。把它接到 demo 2 的 actor 更新里，看哪个 λ 在 CartPole 上收敛最好。
4. **连续动作改造**：把 demo 1 的 Categorical 换成 `torch.distributions.Normal`（参考 §2.4(d) 的 CleanRL 高斯 Agent），在 `Pendulum-v1` 上跑 REINFORCE，体会「换分布族、更新公式不变」。注意 reward 是负的，标准化尤其重要。

---

## 附：本章 WebFetch / WebSearch 取材记录（可复核）

- 真实源码：
  - `pytorch/examples@reinforcement_learning/reinforce.py`（REINFORCE 逐字）— raw.githubusercontent.com
  - `pytorch/examples@reinforcement_learning/actor_critic.py`（Actor-Critic 逐字）— raw.githubusercontent.com
  - `vwxyzjn/cleanrl@cleanrl/ppo.py`（Agent / GAE / loss 逐字）— raw.githubusercontent.com
  - `vwxyzjn/cleanrl@cleanrl/ppo_continuous_action.py`（高斯 Agent 逐字）— raw.githubusercontent.com
  - `huggingface/trl@trl/trainer/grpo_trainer.py`（尝试取 advantage 段，**关键段被截断未取到，已标待核**）
- 设计考古 / 公式：
  - Wikipedia "Policy gradient method"（Williams 1992、Sutton et al. 1999、score function 期望为 0、advantage、causality trick）
  - NeurIPS 1999 proceedings（Sutton/McAllester/Singh/Mansour 引用信息）
  - arXiv:1506.02438（GAE 标题/作者/年份；**PDF 为二进制，公式按标准形式重写并标「示意，非逐字」**）
  - OpenAI Spinning Up VPG 页（advantage 形式的 PG 梯度、VPG 伪代码）
  - Lilian Weng "Policy Gradient Algorithms"（PG theorem / REINFORCE 更新式 LaTeX 形式）
  - HuggingFace blog "Navigating the RLHF Landscape" + WebSearch（PPO/GRPO/RLOO/REINFORCE++/KL penalty）

> 诚实边界：标【真实源码】的代码块为 WebFetch 逐字摘录；标【示意，非逐字】的为按论文标准形式重写（主要是 GAE 公式，因 PDF 二进制未能逐字提取）；§6.3 GRPO/RLOO 的精确 advantage 公式标【待核】——概念方向已核实，逐行源码未取到。三个 demo 为本章作者编写、设计为可运行，请在你的环境验证收敛趋势（数值因种子而异）。
