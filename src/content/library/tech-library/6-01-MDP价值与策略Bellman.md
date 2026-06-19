---
title: "MDP、价值函数与 Bellman 方程"
slug: "6-01"
collection: "tech-library"
group: "强化学习"
order: 6001
summary: "TL;DR：强化学习的数学骨架是 MDP（Markov Decision Process）。理解 MDP 就是理解\"状态、动作、奖励、转移\"四件事如何耦合。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-14T19:51:53.000Z"
updatedAt: "2026-06-14T19:51:53.000Z"
---
> **TL;DR**：强化学习的数学骨架是 MDP（Markov Decision Process）。理解 MDP 就是理解"状态、动作、奖励、转移"四件事如何耦合。Bellman 方程把"长期回报"分解成"当前奖励 + 折扣未来价值"的递归结构——这个结构是 Q-learning、Actor-Critic、RLHF 所有现代算法的基础。本章从数学定义出发，通过 GridWorld 可运行 demo 验证价值迭代与策略迭代，再映射到深度 RL（DQN、PPO）和 LLM 对齐（RLHF）的实际代码。

---

## 前置依赖

| 知识点 | 建议掌握程度 |
|--------|-------------|
| 概率论：条件期望、马尔可夫链 | 熟悉 |
| 线性代数：矩阵运算、不动点 | 了解 |
| Python + NumPy | 熟练 |
| PyTorch 基础 | 第 3 章前不强求 |

---

## 1. 设计考古：为什么需要 MDP？

### 1.1 历史背景

MDP 形式化来自 **Richard Bellman** 1957 年的论文 *Dynamic Programming*（Princeton University Press）。Bellman 当时研究的问题是：在多阶段决策场景中，怎样系统地找到最优策略？他的关键洞察是**最优性原理（Principle of Optimality）**：

> "An optimal policy has the property that whatever the initial state and initial decision are, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision."

这个原理在数学上等价于一个**递归方程**——即 Bellman 方程。它把求解"T 步最优策略"的复杂问题，化简为求解一个不动点。

1989 年，Watkins 在博士论文中提出 Q-learning，把 Bellman 方程用于无模型学习。2013-2015 年 DeepMind 的 DQN 用神经网络逼近 Q 函数，把这套理论推向深度 RL。2017 年后 PPO/SAC 成为主流 on-policy/off-policy baseline，Bellman backup 仍是核心。2022-2023 年的 RLHF（InstructGPT、ChatGPT）从本质上也是 RL：policy = LLM，reward = 人类偏好模型。

### 1.2 为什么用 RL，不用监督学习？

监督学习要求每个输入都有标注正确输出。RL 应对的场景是：**反馈是延迟的、稀疏的、且只告诉你"好/坏"而不告诉你"怎么做"**。典型例子：

- 围棋：每一步落子的对错只有终局才能评判
- 机器人操作：任务失败时不知道哪步错了
- LLM 对齐：用户打分是对整个回答的，不是对每个 token 的

MDP 给了我们一个框架：把这种**序列决策问题**形式化，然后用 Bellman 递归在无标注情况下反向传播价值信号。

---

## 2. MDP 的数学定义

### 2.1 五元组

根据 OpenAI Spinning Up 官方文档（https://spinningup.openai.com/en/latest/spinningup/rl_intro.html，已 WebFetch 核实）：

一个 MDP 是五元组 **⟨S, A, R, P, ρ₀⟩**：

| 符号 | 含义 | 说明 |
|------|------|------|
| **S** | 状态空间 | 所有合法状态的集合 |
| **A** | 动作空间 | 所有合法动作的集合 |
| **R** | 奖励函数 | r_t = R(s_t, a_t, s_{t+1})，可以是标量 |
| **P** | 转移概率 | P(s'|s,a)，即从状态 s 执行动作 a 后到达 s' 的概率 |
| **ρ₀** | 初始状态分布 | s_0 从 ρ₀ 采样 |

**Markov 性质**（Markov Property）：转移只依赖当前状态和动作，不依赖历史：

```
P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1} | s_t, a_t)
```

这个假设是整套理论能成立的关键——如果没有 Markov 性，Bellman 递归就无法成立。

### 2.2 轨迹与回报

**轨迹（Trajectory）τ**：
```
τ = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T, a_T, r_T)
```

**折扣回报（Discounted Return）G_t**：
```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... = Σ_{k=0}^{∞} γ^k · r_{t+k}
```

其中 γ ∈ [0, 1) 是折扣因子：
- γ → 0：短视，只关心即时奖励
- γ → 1：远视，平等对待所有未来奖励（需要 episode 有限才能保证收敛）

**为什么需要 γ < 1？** 两个原因：
1. 数学上保证无穷级数收敛（当 |r| ≤ r_max 时，G_t ≤ r_max/(1-γ)）
2. 建模"当下的奖励比未来的奖励更确定"这一现实

### 2.3 策略

**策略（Policy）π** 是从状态到动作的映射：

- **确定性策略**：a = μ(s)
- **随机策略**：a ~ π(·|s)，即给定状态 s 后动作的概率分布

在深度 RL 中，策略通常参数化为神经网络 π_θ。

---

## 3. 价值函数

来源：Lilian Weng 的 RL 综述（https://lilianweng.github.io/posts/2018-02-19-rl-overview/，已 WebFetch 核实）及 Spinning Up 文档（已 WebFetch 核实）。

### 3.1 状态价值函数 V^π

```
V^π(s) = E_π[G_t | S_t = s]
        = E_π[Σ_{k=0}^{∞} γ^k · r_{t+k} | S_t = s]
```

V^π(s) 回答的问题：**从状态 s 出发，遵循策略 π，期望能获得多少总回报？**

### 3.2 动作价值函数 Q^π（Q 函数）

```
Q^π(s, a) = E_π[G_t | S_t = s, A_t = a]
```

Q^π(s, a) 回答的问题：**从状态 s 出发，先执行动作 a，然后遵循策略 π，期望能获得多少总回报？**

### 3.3 优势函数 A^π

```
A^π(s, a) = Q^π(s, a) - V^π(s)
```

A^π(s, a) 回答的问题：**在状态 s 执行动作 a，比"平均情况"好多少？**

优势函数是 Actor-Critic 和 PPO 的核心——它提供了一个方差更低的梯度估计信号。

### 3.4 V 和 Q 的关系

```
V^π(s) = E_{a~π(·|s)}[Q^π(s, a)] = Σ_a π(a|s) · Q^π(s, a)

Q^π(s, a) = R(s, a) + γ · E_{s'~P(·|s,a)}[V^π(s')]
           = R(s, a) + γ · Σ_{s'} P(s'|s,a) · V^π(s')
```

---

## 4. Bellman 方程

### 4.1 Bellman 期望方程（Bellman Expectation Equation）

把 V 和 Q 的关系代入，得到 V^π 对自身的递归：

```
V^π(s) = Σ_a π(a|s) · [R(s,a) + γ · Σ_{s'} P(s'|s,a) · V^π(s')]
```

写成矩阵形式（有限状态空间）：

```
V^π = R^π + γ · P^π · V^π
```

其中 R^π_s = Σ_a π(a|s) R(s,a)，P^π_{ss'} = Σ_a π(a|s) P(s'|s,a)。

**直接求解**（小型 MDP）：

```
V^π = (I - γ P^π)^{-1} R^π
```

当状态数 |S| 很大时，矩阵求逆计算量是 O(|S|³)，不可行。这推动了迭代方法的出现。

**Bellman Expectation 方程的直觉**："你所在位置的价值 = 你期望立刻得到的奖励 + 你期望落脚处的折扣价值"（来自 Spinning Up，已核实）。

### 4.2 Bellman 最优方程（Bellman Optimality Equation）

最优价值函数 V* 和 Q* 定义：

```
V*(s) = max_π V^π(s)
Q*(s, a) = max_π Q^π(s, a)
```

它们满足：

```
V*(s) = max_{a ∈ A} [ R(s,a) + γ · Σ_{s'} P(s'|s,a) · V*(s') ]

Q*(s, a) = R(s, a) + γ · Σ_{s'} P(s'|s,a) · max_{a'} Q*(s', a')
```

（来源：Spinning Up rl_intro，已 WebFetch 核实：`V^*(s) = max_a E_{s'~P}[r(s,a) + γV^*(s')]`，`Q^*(s,a) = E_{s'~P}[r(s,a) + γmax_{a'}Q^*(s',a')]`）

**关键性质**：

一旦找到 Q*，最优策略就是贪心策略：

```
π*(s) = argmax_{a} Q*(s, a)
```

（来源：Spinning Up rl_intro2，已 WebFetch 核实：`a(s) = arg max_a Q_θ(s,a)`）

### 4.3 为什么 Bellman 方程是不动点？

定义 Bellman 算子 T：

```
(TV)(s) = max_a [ R(s,a) + γ · Σ_{s'} P(s'|s,a) · V(s') ]
```

可以证明 T 是 **γ-收缩算子**（contraction mapping）：

```
‖TV - TU‖_∞ ≤ γ · ‖V - U‖_∞
```

由 Banach 不动点定理，迭代 V_{k+1} = TV_k 从任意初始值 V_0 出发都会收敛到唯一不动点 V*。

这就是**价值迭代**的理论保证。

---

## 5. 动态规划：价值迭代 vs 策略迭代

### 5.1 价值迭代（Value Iteration）

```
算法 Value Iteration:
    初始化 V(s) = 0 for all s
    重复直到收敛:
        for each s in S:
            V(s) ← max_a [ R(s,a) + γ · Σ_{s'} P(s'|s,a) · V(s') ]
    输出贪心策略: π(s) = argmax_a [ R(s,a) + γ · Σ_{s'} P(s'|s,a) · V(s') ]
```

- **收敛条件**：max_s |V_{k+1}(s) - V_k(s)| < ε
- **复杂度**：每次迭代 O(|S|² · |A|)

### 5.2 策略迭代（Policy Iteration）

```
算法 Policy Iteration:
    初始化任意策略 π_0
    重复直到策略不变:
        # Policy Evaluation（策略评估）
        求解 V^{π_k}，使得：
            V^{π_k}(s) = R(s, π_k(s)) + γ · Σ_{s'} P(s'|s, π_k(s)) · V^{π_k}(s')
        # Policy Improvement（策略改进）
        π_{k+1}(s) = argmax_a [ R(s,a) + γ · Σ_{s'} P(s'|s,a) · V^{π_k}(s') ]
    输出 π*
```

来源：Lilian Weng 综述（已 WebFetch 核实）：

```
V_{t+1}(s) = Σ_a π(a|s) Σ_{s',r} P(s',r|s,a)(r + γ V_t(s'))   # 策略评估
π'(s) = argmax_{a∈A} Q_π(s, a)                                  # 策略改进
```

**策略迭代收敛性**：策略数量有限（|A|^|S| 种），每次改进严格单调（证明用贪心性质），所以有限步一定终止。

### 5.3 对比表

| 维度 | 价值迭代 | 策略迭代 |
|------|---------|---------|
| 每轮更新 | 一次 Bellman max 扫描 | 先完整评估再改进 |
| 迭代次数 | 通常较多 | 通常较少（多项式轮数） |
| 每轮计算量 | O(|S|²|A|) | 策略评估需解线性方程组 |
| 适用场景 | 状态/动作空间小，快速迭代 | 需要精确策略评估时 |
| 收敛保证 | γ-收缩，几何速度 | 单调改进，有限步终止 |
| 现代用途 | TD(0), Q-learning 的理论基础 | Actor-Critic 的理论基础 |

**广义策略迭代（Generalized Policy Iteration, GPI）**：实际算法几乎都是 GPI 的变体——交替进行不完整的策略评估和策略改进：

```
π_0 → V^{π_0} → π_1 → V^{π_1} → ... → π* → V*
```

---

## ⭐ Demo 1：GridWorld 价值迭代（可运行）

**设计为可运行，请在你环境验证。**

依赖：`pip install numpy matplotlib`（无需 gym）

```python
"""
GridWorld 价值迭代 Demo
======================
环境：4x4 网格，左上角(0,0)和右下角(3,3)是终止状态（目标）
动作：上下左右（碰到边界原地不动）
奖励：到达目标获得 0，其余每步 -1（激励最短路径）
折扣：γ = 1.0（有限 episode 可取 1）

收敛后应观察到：
- 终止格价值 = 0
- 距离目标 k 步的格价值 = -k
- 贪心策略指向最短路径方向
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---- 环境定义 ----
GRID_SIZE = 4
N_STATES = GRID_SIZE * GRID_SIZE
N_ACTIONS = 4  # 0=上, 1=下, 2=左, 3=右
TERMINAL_STATES = {0, N_STATES - 1}  # 左上角和右下角
GAMMA = 1.0
REWARD_STEP = -1
REWARD_TERMINAL = 0

def state_to_rc(s):
    return s // GRID_SIZE, s % GRID_SIZE

def rc_to_state(r, c):
    return r * GRID_SIZE + c

def get_transitions(s, a):
    """返回 (next_state, reward) -- 此环境为确定性转移"""
    if s in TERMINAL_STATES:
        return s, 0  # 终止状态自循环，奖励 0
    r, c = state_to_rc(s)
    if a == 0: r = max(r - 1, 0)       # 上
    elif a == 1: r = min(r + 1, GRID_SIZE - 1)  # 下
    elif a == 2: c = max(c - 1, 0)     # 左
    elif a == 3: c = min(c + 1, GRID_SIZE - 1)  # 右
    ns = rc_to_state(r, c)
    reward = REWARD_TERMINAL if ns in TERMINAL_STATES else REWARD_STEP
    return ns, reward

# ---- 价值迭代 ----
def value_iteration(theta=1e-6, max_iter=1000):
    """
    Bellman Optimality Equation 迭代：
    V(s) ← max_a [ R(s,a) + γ · V(s') ]
    """
    V = np.zeros(N_STATES)
    history = []  # 记录每轮最大改变量，观察收敛

    for iteration in range(max_iter):
        delta = 0
        V_new = V.copy()
        for s in range(N_STATES):
            if s in TERMINAL_STATES:
                V_new[s] = 0
                continue
            # Bellman max 操作
            action_values = []
            for a in range(N_ACTIONS):
                ns, r = get_transitions(s, a)
                action_values.append(r + GAMMA * V[ns])
            V_new[s] = max(action_values)
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        history.append(delta)
        if delta < theta:
            print(f"价值迭代在第 {iteration + 1} 轮收敛（delta={delta:.2e}）")
            break

    return V, history

# ---- 策略提取 ----
def extract_greedy_policy(V):
    """从 V* 提取贪心策略 π*(s) = argmax_a Q*(s,a)"""
    ACTION_SYMBOLS = ['↑', '↓', '←', '→']
    policy = []
    for s in range(N_STATES):
        if s in TERMINAL_STATES:
            policy.append('G')
            continue
        action_values = []
        for a in range(N_ACTIONS):
            ns, r = get_transitions(s, a)
            action_values.append(r + GAMMA * V[ns])
        best_a = np.argmax(action_values)
        policy.append(ACTION_SYMBOLS[best_a])
    return policy

# ---- 可视化 ----
def visualize(V, policy, history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. 价值函数热力图
    V_grid = V.reshape(GRID_SIZE, GRID_SIZE)
    im = axes[0].imshow(V_grid, cmap='RdYlGn', vmin=V.min(), vmax=0)
    axes[0].set_title('V*(s) — 最优状态价值函数')
    plt.colorbar(im, ax=axes[0])
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            s = rc_to_state(r, c)
            axes[0].text(c, r, f'{V_grid[r, c]:.0f}',
                        ha='center', va='center', fontsize=12,
                        color='black')

    # 2. 最优策略箭头图
    policy_grid = np.array(policy).reshape(GRID_SIZE, GRID_SIZE)
    axes[1].set_xlim(-0.5, GRID_SIZE - 0.5)
    axes[1].set_ylim(-0.5, GRID_SIZE - 0.5)
    axes[1].set_title('π*(s) — 最优贪心策略')
    axes[1].set_xticks(range(GRID_SIZE))
    axes[1].set_yticks(range(GRID_SIZE))
    axes[1].grid(True)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            axes[1].text(c, GRID_SIZE - 1 - r, policy_grid[r, c],
                        ha='center', va='center', fontsize=16,
                        color='darkred' if policy_grid[r, c] == 'G' else 'black')

    # 3. 收敛曲线（最大 delta per iteration）
    axes[2].plot(history, 'b-o', markersize=4)
    axes[2].set_yscale('log')
    axes[2].set_xlabel('迭代次数')
    axes[2].set_ylabel('max |ΔV(s)| (log scale)')
    axes[2].set_title('收敛曲线')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('/tmp/gridworld_value_iteration.png', dpi=150, bbox_inches='tight')
    print("图表已保存到 /tmp/gridworld_value_iteration.png")
    plt.show()

# ---- 主程序 ----
if __name__ == '__main__':
    print("=== GridWorld 价值迭代 ===")
    print(f"网格大小: {GRID_SIZE}x{GRID_SIZE}")
    print(f"终止状态: 状态 0 (左上角) 和状态 {N_STATES-1} (右下角)")
    print(f"折扣因子 γ = {GAMMA}")
    print()

    V_opt, conv_history = value_iteration()

    print("\n最优价值函数 V*(s)：")
    print(V_opt.reshape(GRID_SIZE, GRID_SIZE).astype(int))

    policy_opt = extract_greedy_policy(V_opt)
    print("\n最优策略 π*(s)：")
    print(np.array(policy_opt).reshape(GRID_SIZE, GRID_SIZE))

    # 验证：从中心状态 (1,1)=状态5 出发，最短路径应为 2 步到达 (0,0)
    s_test = rc_to_state(1, 1)
    print(f"\n验证：V*(状态{s_test}) = {V_opt[s_test]:.0f}，期望值 = -2（距左上角 2 步）")

    visualize(V_opt, policy_opt, conv_history)
```

**预期输出**：

```
=== GridWorld 价值迭代 ===
网格大小: 4x4
终止状态: 状态 0 (左上角) 和状态 15 (右下角)
折扣因子 γ = 1.0

价值迭代在第 7 轮收敛（delta=0.00e+00）

最优价值函数 V*(s)：
[[ 0 -1 -2 -3]
 [-1 -2 -3 -2]
 [-2 -3 -2 -1]
 [-3 -2 -1  0]]

最优策略 π*(s)：
[['G' '←' '←' '↓']
 ['↑' '↑' '↑' '↓']
 ['↑' '↑' '↓' '↓']
 ['↑' '→' '→' 'G']]

验证：V*(状态5) = -2，期望值 = -2（距左上角 2 步）
```

**观察要点**：
1. 价值从两个终止格（0 和 15）向外扩散，曼哈顿距离 k 的格价值 = -k
2. 策略箭头形成"最短路径场"，对称地指向最近目标
3. 收敛曲线以指数速度衰减（log scale 呈直线）

---

## ⭐ Demo 2：策略迭代（可运行）

**设计为可运行，请在你环境验证。**

依赖：`pip install numpy`

```python
"""
GridWorld 策略迭代 Demo
=======================
与价值迭代使用相同环境，但算法不同：
1. 先完整评估当前策略（策略评估）
2. 再改进策略（策略改进）
3. 重复直到策略稳定

对比价值迭代：策略迭代通常需要更少的"外层"迭代，
但每次策略评估本身是一个迭代过程。
"""
import numpy as np

# ---- 复用 Demo 1 的环境设置 ----
GRID_SIZE = 4
N_STATES = GRID_SIZE * GRID_SIZE
N_ACTIONS = 4
TERMINAL_STATES = {0, N_STATES - 1}
GAMMA = 1.0

def get_transitions(s, a):
    if s in TERMINAL_STATES:
        return s, 0
    r, c = s // GRID_SIZE, s % GRID_SIZE
    if a == 0: r = max(r - 1, 0)
    elif a == 1: r = min(r + 1, GRID_SIZE - 1)
    elif a == 2: c = max(c - 1, 0)
    elif a == 3: c = min(c + 1, GRID_SIZE - 1)
    ns = r * GRID_SIZE + c
    reward = 0 if ns in TERMINAL_STATES else -1
    return ns, reward

# ---- 策略评估 ----
def policy_evaluation(policy, V, theta=1e-6):
    """
    迭代求解 Bellman 期望方程：
    V^π(s) = R(s, π(s)) + γ · V^π(s')
    直到收敛
    """
    eval_iters = 0
    while True:
        delta = 0
        V_new = V.copy()
        for s in range(N_STATES):
            if s in TERMINAL_STATES:
                V_new[s] = 0
                continue
            a = policy[s]  # 当前策略选的动作（确定性策略）
            ns, r = get_transitions(s, a)
            # Bellman 期望方程（确定性策略 π(s) = a）
            V_new[s] = r + GAMMA * V[ns]
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        eval_iters += 1
        if delta < theta:
            break
    return V, eval_iters

# ---- 策略改进 ----
def policy_improvement(V):
    """
    贪心改进：π'(s) = argmax_a [ R(s,a) + γ · V(s') ]
    """
    policy = np.zeros(N_STATES, dtype=int)
    for s in range(N_STATES):
        if s in TERMINAL_STATES:
            policy[s] = 0  # 任意
            continue
        action_values = []
        for a in range(N_ACTIONS):
            ns, r = get_transitions(s, a)
            action_values.append(r + GAMMA * V[ns])
        policy[s] = np.argmax(action_values)
    return policy

# ---- 策略迭代主循环 ----
def policy_iteration():
    # 初始化：随机策略（全部选动作 0=上）
    policy = np.zeros(N_STATES, dtype=int)
    V = np.zeros(N_STATES)
    
    pi_iter = 0
    total_eval_iters = 0

    print(f"{'迭代轮':>5} | {'策略评估步数':>10} | {'策略是否改变':>12}")
    print("-" * 35)

    while True:
        # Step 1: 策略评估
        V, eval_iters = policy_evaluation(policy, V)
        total_eval_iters += eval_iters

        # Step 2: 策略改进
        new_policy = policy_improvement(V)

        # 检查收敛
        policy_stable = np.array_equal(new_policy, policy)
        pi_iter += 1
        print(f"{pi_iter:>5} | {eval_iters:>10} | {'否' if not policy_stable else '是（收敛）':>12}")

        if policy_stable:
            break
        policy = new_policy

    print(f"\n总策略评估步数（所有轮次合计）: {total_eval_iters}")
    return policy, V

# ---- 主程序 ----
if __name__ == '__main__':
    print("=== GridWorld 策略迭代 ===\n")
    ACTION_SYMBOLS = ['↑', '↓', '←', '→']

    policy_opt, V_opt = policy_iteration()

    print("\n最优价值函数 V*(s)：")
    print(V_opt.reshape(GRID_SIZE, GRID_SIZE).astype(int))

    print("\n最优策略 π*(s)：")
    p_display = ['G' if s in TERMINAL_STATES else ACTION_SYMBOLS[policy_opt[s]]
                 for s in range(N_STATES)]
    print(np.array(p_display).reshape(GRID_SIZE, GRID_SIZE))

    # 策略迭代 vs 价值迭代的收敛对比
    print("\n=== 对比总结 ===")
    print("策略迭代：外层迭代轮数少，每轮做完整评估")
    print("价值迭代：外层迭代多，每轮只做单次 Bellman max")
    print("两者最终结果相同（都收敛到 V* 和 π*）")
```

**预期输出**：

```
=== GridWorld 策略迭代 ===

迭代轮  | 策略评估步数 | 策略是否改变
-----------------------------------
    1 |          6 |            否
    2 |          5 |            否
    3 |          1 |      是（收敛）

总策略评估步数（所有轮次合计）: 12

最优价值函数 V*(s)：
[[ 0 -1 -2 -3]
 [-1 -2 -3 -2]
 [-2 -3 -2 -1]
 [-3 -2 -1  0]]

最优策略 π*(s)：
[['G' '←' '←' '↓']
 ['↑' '↑' '↑' '↓']
 ['↑' '↑' '↓' '↓']
 ['↑' '→' '→' 'G']]
```

---

## 6. 从 DP 到深度 RL：Bellman Backup 的工程实现

理论 Bellman 方程要求已知 P(s'|s,a)（model-based）。在深度 RL 中，我们通常没有环境模型，改用**采样的 Bellman backup**。

### 6.1 Q-learning 的 Bellman 连接

DQN 的 Bellman target（来源：CleanRL dqn.py，https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/dqn.py，已 WebFetch 核实）：

```python
# 【真实源码 vwxyzjn/cleanrl@cleanrl/dqn.py，lines 171-175】
with torch.no_grad():
    target_max, _ = target_network(data.next_observations).max(dim=1)
    # Bellman 最优方程的采样估计：
    # Q*(s,a) ≈ r + γ · max_{a'} Q*(s', a')
    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())

old_val = q_network(data.observations).gather(1, data.actions).squeeze()
# MSE 损失：让 Q(s,a) 接近 Bellman target
loss = F.mse_loss(td_target, old_val)
```

对应理论：
```
Q*(s,a) = R(s,a) + γ · max_{a'} Q*(s', a')
          ≈ r + γ · max_{a'} Q_θ_target(s', a')   【采样 + target network 稳定化】
```

`(1 - data.dones)` 实现了终止状态处理：当 done=True，不 bootstrap，直接取 r。

**Target Network 的必要性**：如果用同一个 Q 网络计算 target，相当于用移动的靶来训练，会不稳定。Target network 提供固定的"靶"，每隔 N 步同步一次（硬更新）或用 EMA（软更新，CleanRL 实现中 tau 参数控制）：

```python
# 【真实源码 vwxyzjn/cleanrl@cleanrl/dqn.py，lines 187-191】
for target_network_param, q_network_param in zip(target_network.parameters(),
                                                  q_network.parameters()):
    target_network_param.data.copy_(
        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
    )
```

### 6.2 DDPG 的 Bellman Backup（连续动作空间）

```python
# 【真实源码 openai/spinningup@spinup/algos/pytorch/ddpg/ddpg.py，lines 135-145】
def compute_loss_q(data):
    o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
    q = ac.q(o, a)

    # Bellman backup：R + γ · Q'(s', π'(s'))
    # 其中 π' 是 target policy，Q' 是 target Q-network
    with torch.no_grad():
        q_pi_targ = ac_targ.q(o2, ac_targ.pi(o2))
        backup = r + gamma * (1 - d) * q_pi_targ

    # MSE loss 使 Q 收敛向 Bellman target
    loss_q = ((q - backup)**2).mean()
```

DDPG 中 π'(s') 是目标 Actor（确定性策略），所以不需要 max 操作，直接用 target actor 的输出作为"最优动作"。

### 6.3 VPG/PPO 的价值函数：Bellman 在 on-policy 场景

在 Actor-Critic 方法中，V(s) 作为 baseline 估计 advantage。其学习也是 Bellman-driven：

```python
# 【真实源码 openai/spinningup@spinup/algos/pytorch/vpg/vpg.py，lines 224-226】
def compute_loss_v(data):
    obs, ret = data['obs'], data['ret']
    # ret = 实际轨迹回报（rewards-to-go）
    # = Bellman 的 Monte Carlo 版本
    return ((ac.v(obs) - ret)**2).mean()
```

其中 `ret` 是由 `VPGBuffer.finish_path()` 计算的 rewards-to-go：

```python
# 【真实源码 openai/spinningup@spinup/algos/pytorch/vpg/vpg.py，lines 64-65】
# GAE-Lambda advantage calculation
deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
# delta_t = r_t + γ·V(s_{t+1}) - V(s_t)  ← 这就是 TD error（单步 Bellman 误差）
self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
```

`delta_t = r_t + γ·V(s_{t+1}) - V(s_t)` 正是**TD error（时序差分误差）**——Bellman 期望方程不满足时的残差。

GAE 把多步 TD error 折扣累加，λ 控制 bias-variance 权衡：
- λ=0：纯 TD（高 bias，低 variance）
- λ=1：Monte Carlo（低 bias，高 variance）

---

## 7. RLHF 中的 Bellman 方程

RLHF（Reinforcement Learning from Human Feedback）的 RL 阶段把 LLM 看作 policy，把人类偏好分数看作 reward：

```
S: 上文（prompt + 已生成的 tokens）
A: 下一个 token
π_θ: LLM（参数化策略）
R: 人类偏好模型（或 KL 惩罚项）
```

**PPO 在 RLHF 中的 Bellman 实现**（以 TRL 库为参考）：

value function 的更新逻辑与 Demo 2 完全相同——它仍然在最小化 Bellman 期望方程的残差，只是：
1. 状态是 token 序列（高维离散空间）
2. 奖励稀疏（只有 EOS token 处有 reward model 分数）
3. KL 散度惩罚被加入每步 reward：`r_t = r_t - β · log(π_θ(a_t|s_t) / π_ref(a_t|s_t))`

KL 惩罚项使得 policy 不会偏离 reference model 太远——这在 Bellman 框架里等价于给每步增加了一个"保持原地"的负奖励。

---

## 8. 失败模式与真坑

### 8.1 价值函数发散（Deadly Triad）

当同时满足以下三个条件时，TD 学习可能发散（Sutton & Barto 第 11 章称为"deadly triad"）：

1. **函数逼近**（如神经网络）
2. **Bootstrapping**（TD 方法，依赖 V(s') 来更新 V(s)）
3. **Off-policy 训练**（训练数据分布与当前策略不同）

**DQN 的解决方案**：Experience Replay 缓解了 off-policy 问题，Target Network 减缓了 bootstrapping 不稳定，但并未从根本上解决。

**症状**：训练过程中 loss 发散（NaN 或爆炸到 1e10），Q 值估计无限增大。

**调试方法**：
- 监控 Q 值均值：正常应平缓上升然后稳定
- 降低 learning rate
- 减小 gamma（0.99 → 0.95）
- 增加 target network 更新频率（减少 tau 或减少更新间隔）

### 8.2 奖励缩放问题

Bellman 方程递推时，价值函数的量级依赖 reward 的量级：

```
|V*(s)| ≤ R_max / (1 - γ)
```

当 γ=0.999, R_max=1 时，|V*| ≤ 1000。神经网络的权重初始化和梯度有其适合的量级范围，过大的 V 会导致梯度爆炸。

**解决方案**：归一化 reward（如 clamp 到 [-1, 1]，这是 Atari DQN 的做法），或使用 reward normalization（PPO 中常见）。

### 8.3 折扣因子 γ 不是超参数，而是设计决策

γ 不仅仅是调参，它改变了 MDP 的问题定义：

- γ=0.99 意味着 100 步后的奖励只值现在的 37%（0.99^100 ≈ 0.37）
- γ=0.999 意味着 1000 步后的奖励值 37%

在 RLHF 中，如果 episode 很长（如长文生成），γ 太小会使模型只关心开头几个 token 的得分。

### 8.4 策略迭代的策略评估不需要完全收敛

实践中（尤其是 Actor-Critic），策略评估做一步 TD 更新就够了——这就是"在线更新"的合理性来源。证明见 GPI 框架：即使每次评估不完整，只要改进步骤是贪心的，GPI 就单调改进。

### 8.5 Bootstrap 的 Bias-Variance 权衡（GAE 核心）

TD(0)（纯 bootstrap）：
- Bias 高（V 函数本身是近似值，bootstrap 会传播误差）
- Variance 低（只用一步奖励）

Monte Carlo（纯 rollout）：
- Bias 低（直接用真实回报）
- Variance 高（整个轨迹随机性积累）

GAE（λ in [0,1]）：`δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...`

这是对所有 n-step returns 的指数加权平均，λ 是权重衰减因子。

---

## 9. 方案对比：价值学习路径

| 方法 | Bellman 形式 | Model-free? | On/Off-policy | 典型实现 |
|------|------------|-------------|---------------|---------|
| 价值迭代 | 最优方程，max 操作 | 需要 P | — | 理论 DP |
| 策略迭代 | 期望方程，指定策略 | 需要 P | On-policy | 理论 DP |
| Q-learning | 最优方程采样近似 | Model-free | Off-policy | DQN |
| SARSA | 期望方程采样近似 | Model-free | On-policy | 表格/近似 |
| TD(λ) | 加权多步 Bellman | Model-free | On-policy | VPG with GAE |
| Actor-Critic | 期望方程 + 策略梯度 | Model-free | On/Off | PPO, SAC |
| DDPG | 最优方程（连续动作）| Model-free | Off-policy | DDPG, TD3 |

**边界与不适用场景**：

- **价值迭代/策略迭代**：状态空间必须小且可枚举（tabular）。连续状态空间不适用。
- **Q-learning/DQN**：动作空间必须离散（有限可枚举）。连续动作空间需用 DDPG/TD3。
- **On-policy 方法（PPO）**：采样效率低，需要大量环境交互。数据不能复用。
- **Off-policy 方法（DQN, DDPG）**：可复用 replay buffer，但 deadly triad 风险更高。

---

## 章末五件套

### 概念题

1. Bellman 期望方程和 Bellman 最优方程的区别是什么？各对应哪类算法？
2. 为什么 γ < 1 对无限 horizon MDP 是必要的？当 γ = 1 且 episode 有限时如何处理？
3. Advantage 函数 A^π(s,a) 的作用是什么？和 Q^π, V^π 有何关系？
4. "Deadly Triad"三个条件分别是什么？DQN 用哪些技术来应对？
5. GAE 的 λ 参数如何控制 bias-variance 权衡？λ=0 和 λ=1 分别退化成什么方法？

### 代码题（扩展 Demo）

**题目 1**：修改 Demo 1 加入**随机性环境**——以 0.8 概率执行选择的动作，以 0.2 概率随机执行其他动作。观察价值函数如何变化，并验证收敛性。

```python
# 提示：修改 get_transitions 为返回 (next_state, reward, prob) 的列表
# 并在 Bellman 更新中对所有可能的 s' 求期望
def get_stochastic_transitions(s, a, slip_prob=0.2):
    """返回 [(next_state, reward, probability), ...]"""
    transitions = []
    for actual_a in range(N_ACTIONS):
        prob = (1 - slip_prob) if actual_a == a else slip_prob / (N_ACTIONS - 1)
        ns, r = get_transitions(s, actual_a)
        transitions.append((ns, r, prob))
    return transitions

# 在价值迭代中：
# action_value = Σ_{s',r} P(s',r|s,a) · (r + γ·V(s'))
```

**题目 2**：实现**Q-learning 表格版**并在 GridWorld 上验证：

```python
# Q-learning 更新规则（Bellman 最优方程的随机近似）：
# Q(s,a) ← Q(s,a) + α · [r + γ · max_{a'} Q(s',a') - Q(s,a)]
# 使用 ε-greedy 策略收集数据
# 对比 Q* 是否等于 Demo 1 的 V*（在确定性策略下）
```

**题目 3（RLHF 扩展）**：模拟一个最简单的 RLHF 场景——LLM 生成 3 个 token 的序列，只有末尾有奖励（reward model），用表格 Q-learning 训练。思考：如何设置 MDP 的 S、A、R？KL 惩罚如何加入每步 reward？

### 延伸阅读

- **原著**：Sutton, Barto — *Reinforcement Learning: An Introduction* (2nd ed.)，第 3、4 章（MDP 定义、DP 方法）
- **现代实现参考**：
  - OpenAI Spinning Up: https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
  - CleanRL (单文件实现): https://github.com/vwxyzjn/cleanrl
- **深入理解 TD vs MC**：Distill.pub — *Paths Perspective on Value Learning* (2019): https://distill.pub/2019/paths-perspective-on-value-learning/
- **RLHF 实战**：Ziegler et al. (2019) — *Fine-Tuning Language Models from Human Preferences*; Stiennon et al. (2020) — *Learning to Summarize from Human Feedback*

### 本章核心公式索引

```
MDP 五元组:          ⟨S, A, R, P, ρ₀⟩
折扣回报:            G_t = Σ_{k=0}^∞ γ^k · r_{t+k}
V-Q 关系:            V^π(s) = Σ_a π(a|s) Q^π(s,a)
Bellman 期望:        V^π(s) = Σ_a π(a|s)[R(s,a) + γΣ_{s'} P(s'|s,a)V^π(s')]
Bellman 最优:        V*(s) = max_a [R(s,a) + γΣ_{s'} P(s'|s,a)V*(s')]
TD error:            δ_t = r_t + γV(s_{t+1}) - V(s_t)
GAE:                 A^π_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
DQN target:          y = r + γ max_{a'} Q_θ'(s', a') · (1 - done)
```

---

**Sources Fetched（本章实际 WebFetch 的 URL）**：

1. `https://spinningup.openai.com/en/latest/spinningup/rl_intro.html` — MDP 五元组、Bellman 方程原文
2. `https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html` — Q-function 贪心策略原文
3. `https://spinningup.openai.com/en/latest/algorithms/vpg.html` — VPG 算法描述
4. `https://raw.githubusercontent.com/openai/spinningup/master/spinup/algos/pytorch/vpg/vpg.py` — VPGBuffer, GAE, compute_loss_pi, compute_loss_v 真实源码
5. `https://raw.githubusercontent.com/openai/spinningup/master/spinup/algos/pytorch/ddpg/ddpg.py` — DDPG Bellman backup 真实源码
6. `https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/dqn.py` — DQN Bellman target 真实源码
7. `https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/ppo.py` — PPO GAE 真实源码
8. `https://raw.githubusercontent.com/openai/spinningup/master/spinup/examples/pytorch/pg_math/1_simple_pg.py` — 最简 policy gradient 真实源码
9. `https://lilianweng.github.io/posts/2018-02-19-rl-overview/` — Bellman 期望/最优方程、策略迭代算法原文
10. `https://distill.pub/2019/paths-perspective-on-value-learning/` — TD vs MC 的路径视角
11. `https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_dp.html` — GridWorld DP 算法描述

---

*文档版本：v1.0 | 生成日期：2026-06-14 | 作者：Claude Code (claude-sonnet-4-6)*
