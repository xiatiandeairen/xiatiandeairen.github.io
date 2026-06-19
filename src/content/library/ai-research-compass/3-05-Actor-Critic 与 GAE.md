---
title: "Actor-Critic 与 GAE:把方差压下来"
slug: "3-05"
collection: "ai-research-compass"
group: "强化学习专家课程"
order: 3005
summary: "这一章把你从\"知道策略梯度 ∇J = E[∇log π · R] 但一训就抖得没法看\"带到\"能从第一性原理推出为什么它方差大、为什么减 baseline 不引入偏差、critic 是哪一种 baseline 的极致、GAE 又是怎么用一个 λ 在偏差和方差之间连续调旋钮\"。"
topics:
  - "AI 研究"
tags: []
createdAt: "2026-06-19T05:56:03.000Z"
updatedAt: "2026-06-19T05:56:03.000Z"
---
> 这一章把你从"知道策略梯度 ∇J = E[∇log π · R] 但一训就抖得没法看"带到"能从第一性原理推出为什么它方差大、为什么减 baseline 不引入偏差、critic 是哪一种 baseline 的极致、GAE 又是怎么用一个 λ 在偏差和方差之间连续调旋钮"。读完你应当能在白板上独立重推 GAE 的指数加权形式,能解释清楚为什么 λ→0 低方差高偏差、λ→1 无偏高方差,并且能照着把 A2C+GAE 写出来。

## 为什么需要这一章:策略梯度能跑,但跑不稳

上一章我们拿到了策略梯度定理(policy gradient theorem):

```
∇_θ J(θ) = E_τ~π_θ [ Σ_t ∇_θ log π_θ(a_t | s_t) · Ψ_t ]
```

其中 `Ψ_t` 是"给第 t 步动作打的分"。最朴素的 REINFORCE 取 `Ψ_t = R(τ)`(整条轨迹的总回报),稍好一点取 `Ψ_t = G_t = Σ_{k≥t} γ^{k-t} r_k`(从 t 往后的折扣回报,即 reward-to-go)。这个估计器是**无偏的**——它的期望恰好等于真实梯度。无偏听起来很好,但它有一个致命的实践问题:**方差大到几乎没法用**。

直观说三个来源。第一,`G_t` 是一条采样轨迹上未来所有随机性的累加——环境转移的随机、策略采样的随机、奖励本身的随机,全部叠在一个标量里;轨迹越长,这个标量的方差越大,粗略地随 horizon 线性甚至更快增长。第二,这个高方差的标量直接乘在 `∇log π` 上,梯度的方向被一两条"运气特别好/特别差"的轨迹主导。第三,RL 里你通常只能用很少的样本就要更新一次(采样昂贵),小 batch 把高方差暴露得淋漓尽致。

**结果就是:loss 曲线剧烈震荡、有时直接发散、对随机种子极度敏感、要么学不动要么学一半崩掉。** RL 比监督学习难训,一大半难在这里。

这一章的全部任务,就是系统性地把这个方差压下去,而**不引入(或只引入可控的、能用一个旋钮调节的)偏差**。我们会沿这条主线推进:

1. 先证明"减一个 baseline 不改变梯度期望"——这是所有方差削减的合法性来源;
2. 推出最优 baseline,说明为什么用价值函数 `V(s)` 当 baseline 是个极好的近似;
3. 引入 critic,得到 advantage actor-critic;同时讲清楚 bootstrapping(自举)在降方差的同时引入了多少偏差;
4. 把 k-step advantage 写出来,看到一个偏差-方差的离散谱;
5. **核心:推导 GAE,证明它是 k-step advantage 的指数加权平均,λ 在这条谱上连续插值**;
6. 讲 A2C(同步)/ A3C(异步)怎么并行收集数据进一步稳住训练;
7. 给可运行的 A2C+GAE 代码,做方差/偏差/复杂度分析,列坑。

读这一章请始终带着一个问题:**我现在做的每一步,是在减方差还是在加偏差?代价是什么?** 这是 actor-critic 全部设计哲学。

### 预备:几个记号和一个会反复用的事实

记策略 `π_θ(a|s)`(actor),状态价值 `V^π(s) = E_π[G_t | s_t=s]`,动作价值 `Q^π(s,a) = E_π[G_t | s_t=s, a_t=a]`。优势函数(advantage)定义为

```
A^π(s,a) = Q^π(s,a) − V^π(s)
```

它的物理意义非常清楚:**在状态 s 下,选动作 a 比"按当前策略平均地随便选"好多少**。正的优势意味着这个动作高于平均、该加强;负的该削弱。直觉上,用 A 而不是 G 来给动作打分,等于"减掉了这个状态本身的水平",只保留"动作相对好坏"这个真正该被学习信号——这正是降方差的核心思想。

一个会反复用到的恒等式,叫 **score function / log-derivative trick**:

```
∇_θ π_θ(a|s) = π_θ(a|s) · ∇_θ log π_θ(a|s)
```

它只是 `∇log f = ∇f / f` 的移项,但它让"对一个概率分布求梯度"变成"在该分布下求期望",是把梯度写成可采样期望的关键。还有一个推论:**任何不依赖动作 a 的量,在策略分布下乘 score function 求期望都为零**:

```
E_{a~π_θ(·|s)} [ ∇_θ log π_θ(a|s) ] = Σ_a π_θ(a|s) ∇_θ log π_θ(a|s)
                                     = Σ_a ∇_θ π_θ(a|s)
                                     = ∇_θ Σ_a π_θ(a|s)
                                     = ∇_θ 1 = 0
```

记住最后这个 **= 0**,它是下一节 baseline 不引入偏差的全部秘密。

## 第一步:baseline 为什么能减方差又不引入偏差

### 问题:能不能从 Ψ_t 里减掉点东西

我们想把 `Ψ_t = G_t` 换成 `G_t − b(s_t)`,其中 `b(s_t)` 是一个只依赖状态、不依赖动作的 **baseline**。希望:减完之后方差更小,但梯度的期望(也就是它指向的方向)一点没变。第一个诉求(无偏)能不能保证?

### 推导:减任意状态相关 baseline 不改变梯度期望

只需证明被减掉的那一项期望为零。看单个时间步 t,先对动作求条件期望:

```
E_{a_t~π}[ ∇log π_θ(a_t|s_t) · b(s_t) ]
   = b(s_t) · E_{a_t~π}[ ∇log π_θ(a_t|s_t) ]   # b(s_t) 与 a_t 无关,提出来
   = b(s_t) · 0                                  # 用上一节的 = 0
   = 0
```

再对状态分布和其余时间步取期望(它们都不影响这一项已经是 0),整条求和里每个 t 的 baseline 贡献都是 0。于是:

```
E_τ[ Σ_t ∇log π_θ(a_t|s_t) · (G_t − b(s_t)) ]
   = E_τ[ Σ_t ∇log π_θ(a_t|s_t) · G_t ]   # baseline 项整体为 0
   = ∇_θ J(θ)
```

**结论:对任意只依赖状态的 baseline `b(s_t)`,策略梯度估计器保持无偏。** 这给了我们极大的自由——可以任意挑 `b` 去压方差,完全不用担心把梯度方向带歪。注意"只依赖状态"这个条件是必要的:如果 baseline 依赖动作 `a_t`,上面那步 `b · E[∇log π] = b · 0` 就不成立(b 不能提到期望外),无偏性就破了。这也是为什么后面 critic 用的是 `V(s)`(状态价值)而不是 `Q(s,a)`。

### 推导:什么样的 baseline 最能减方差

无偏管够了,现在挑方差最小的。考虑单步项 `g(a) = h(a)·(G − b)`,其中 `h(a) = ∇log π_θ(a|s)`(向量)。我们要选标量 `b` 最小化 `Var[g]`。对每个梯度分量,方差 = E[g²] − (E[g])²,而 E[g] 与 b 无关(刚证的无偏),所以只需最小化 E[g²]:

```
E[g²] = E[ h² (G − b)² ]                       # h² 指逐分量平方
d/db E[g²] = E[ h² · 2(G − b)(−1) ] = 0
        ⇒ E[h² G] = b · E[h²]
        ⇒ b* = E[h² G] / E[h²]
```

**最优 baseline 是"用梯度幅度平方加权的平均回报"** `b* = E[‖∇log π‖² G] / E[‖∇log π‖²]`。它不是简单的 `E[G]`,而是 score 幅度加权版。但实践中这个量难估、还得对每个梯度分量分别算,所以**大家退而用一个非常接近且语义极好的近似:`b(s) = V^π(s)`**。为什么 V 是好近似?因为 `E[G_t | s_t] = V^π(s_t)`,V 就是回报的条件均值,减掉它能消除"不同状态本身回报水平不同"带来的那一大块方差(这通常是方差的主要来源)。代价是它不是逐分量最优,但简单、可学、可解释——把 `G_t − V(s_t)` 一眼看成"这步比该状态的平均表现好多少",正是 advantage 的样子。

把 `b = V` 代回,`Ψ_t = G_t − V(s_t)`。这里 `G_t` 是蒙特卡洛回报(无偏但高方差),`V` 是 critic 估计。这一步已经是一个能用的算法了(REINFORCE with baseline),但 `G_t` 那部分方差还在。下一步我们连 `G_t` 也用 critic 换掉。

## 第二步:引入 critic,得到 Actor-Critic

### 问题:G_t 还是太抖,能不能也用估计替代

`G_t` 是一条轨迹未来奖励的实采样和,方差随 horizon 累积。一个想法是:不要等整条轨迹跑完再算 `G_t`,而是只走一步,然后用 critic 估计"剩下的部分"。这就是 **bootstrapping(自举)**:用一个估计去估计另一个估计。

### 推导:从蒙特卡洛 advantage 到 TD advantage

回忆 advantage 的最干净估计需要 `Q(s,a) − V(s)`。一个对 `Q(s_t,a_t)` 的单样本无偏估计是"走一步拿到 `r_t`,加上下个状态的折扣价值":

```
Q^π(s_t,a_t) = E[ r_t + γ V^π(s_{t+1}) | s_t,a_t ]   # 贝尔曼方程,精确
```

把它代进 advantage,定义 **1-step TD residual(时序差分残差)** δ_t:

```
δ_t = r_t + γ V(s_{t+1}) − V(s_t)
```

这个 `δ_t` 极其重要,记住它的含义:**用真实拿到的一步奖励 r_t 修正后的价值,减去修正前的价值——即"这步带来的惊喜(surprise)"**。如果 V 是真值 `V^π`,那么对动作取期望 `E_{a_t}[δ_t | s_t] = E[r_t + γV(s_{t+1}) − V(s_t)] = V(s_t) − V(s_t) = 0`,而对固定动作 `δ_t` 正是 `A^π(s_t,a_t)` 的一个无偏估计:

```
E[ δ_t | s_t, a_t ] = Q^π(s_t,a_t) − V^π(s_t) = A^π(s_t,a_t)   (当 V=V^π)
```

所以最朴素的 actor-critic 取 `Ψ_t = δ_t`(1-step advantage 估计)。注意这里有个**关键的 caveat,新手最容易忽略**:上面这个无偏只在 `V = V^π`(critic 是真值)时成立。现实里 critic 是学出来的、带误差的,于是 `δ_t` 引入了**偏差**——这是用 bootstrapping 换来低方差所必须付的价。我们用一行刻画这个偏差:

```
若 V = V^π + ε(估计误差),则
δ_t = r_t + γ(V^π(s_{t+1})+ε_{t+1}) − (V^π(s_t)+ε_t)
    = [真实 1-step advantage 项] + (γ ε_{t+1} − ε_t)
偏差来自 (γ ε_{t+1} − ε_t) 这个 critic 误差残留项。
```

### 偏差 vs 方差:这是一条谱,不是两个点

现在我们有两个极端:

- **蒙特卡洛(MC):** `Ψ_t = G_t − V(s_t)`。`G_t` 是真采样回报,**无偏**(critic 误差只通过减去的 `V(s_t)` 进来,而那是 baseline,不影响无偏性——再次感谢第一节);但 `G_t` 累积了整条轨迹的随机性,**高方差**。
- **1-step TD:** `Ψ_t = δ_t`。只看一步真实奖励 + 一步 bootstrap,**低方差**(随机性只来自一步 + 一个 critic 值);但 critic 不准就**有偏**。

中间地带是 **k-step advantage**:走 k 步真实奖励,然后 bootstrap:

```
Â_t^{(k)} = ( Σ_{l=0}^{k-1} γ^l r_{t+l} ) + γ^k V(s_{t+k}) − V(s_t)
```

- k=1:`Â^{(1)} = r_t + γV(s_{t+1}) − V(s_t) = δ_t`,最低方差最高偏差。
- k=∞(到 episode 结束):`Â^{(∞)} = G_t − V(s_t)`,蒙特卡洛,最高方差最低偏差。

**所以"用多少步真实奖励、何时切换到 bootstrap"是一个连续的偏差-方差权衡。** k 越大,真实信息越多(偏差小)、累积随机越多(方差大);k 越小反之。问题来了:k 是个离散整数,而且哪个 k 最好依赖任务、甚至依赖 critic 当前有多准。能不能不硬选一个 k,而是把所有 k 的估计**加权融合**,用一个连续旋钮控制权重?这正是 GAE。

把 k-step advantage 用 `δ` 重写一下,为下一节铺路。利用望远镜求和(telescoping):`Σ_{l=0}^{k-1} γ^l δ_{t+l}` 展开,中间的 `V` 项首尾相消:

```
Σ_{l=0}^{k-1} γ^l δ_{t+l}
  = Σ_{l=0}^{k-1} γ^l ( r_{t+l} + γV(s_{t+l+1}) − V(s_{t+l}) )
  = Σ_{l=0}^{k-1} γ^l r_{t+l}  +  Σ_{l=0}^{k-1} ( γ^{l+1}V(s_{t+l+1}) − γ^l V(s_{t+l}) )
                                  └─────── 这是望远镜和,逐项相消 ───────┘
  = Σ_{l=0}^{k-1} γ^l r_{t+l}  +  ( γ^k V(s_{t+k}) − V(s_t) )
  = Â_t^{(k)}
```

**关键中间结论:k-step advantage 恰好等于前 k 个 TD residual 的折扣和** `Â_t^{(k)} = Σ_{l=0}^{k-1} γ^l δ_{t+l}`。这个等式是 GAE 推导的地基,务必自己验算一遍那步望远镜相消。

## 第三步:GAE 的完整推导(本章核心)

### 问题:把所有 k-step advantage 加权融合

我们手上有一族估计 `{Â^{(1)}, Â^{(2)}, Â^{(3)}, ...}`,从高偏差低方差到低偏差高方差排成一列。最自然的融合是**几何加权**(权重随 k 指数衰减,用参数 λ∈[0,1] 控制衰减速度),并归一化使权重和为 1。GAE 的定义就是这样一个加权平均:

```
Â_t^{GAE(γ,λ)} := (1 − λ) Σ_{k=1}^{∞} λ^{k-1} Â_t^{(k)}
```

`(1−λ)` 是归一化常数(因为 `Σ_{k≥1} λ^{k-1} = 1/(1−λ)`,乘上去权重和为 1)。λ 越大,权重越向大 k(更接近 MC)倾斜;λ 越小,越集中在小 k(更接近 1-step TD)。下面证明它能化简成那个著名的、极其好实现的指数加权 δ 求和形式。

### 推导:Â^{GAE} = Σ (γλ)^l δ_{t+l}

把上一节的 `Â^{(k)} = Σ_{l=0}^{k-1} γ^l δ_{t+l}` 代入定义,然后交换求和次序。这是整章最该亲手推一遍的一步:

```
Â_t^{GAE(γ,λ)}
  = (1−λ) Σ_{k=1}^{∞} λ^{k-1} ( Σ_{l=0}^{k-1} γ^l δ_{t+l} )          # 代入 k-step = δ 折扣和

  交换 k 与 l 的求和次序。原约束:k≥1 且 0≤l≤k-1,等价于:l≥0 且 k≥l+1。
  ⇒ = (1−λ) Σ_{l=0}^{∞} γ^l δ_{t+l} ( Σ_{k=l+1}^{∞} λ^{k-1} )       # 固定 l,对所有 k≥l+1 求 λ^{k-1}

  内层等比和:令 m=k-1,Σ_{m=l}^{∞} λ^m = λ^l / (1−λ)
  ⇒ = (1−λ) Σ_{l=0}^{∞} γ^l δ_{t+l} · λ^l / (1−λ)                   # (1−λ) 与分母约掉

  ⇒ = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
```

**得到 GAE 的核心公式:**

```
Â_t^{GAE(γ,λ)} = Σ_{l≥0} (γλ)^l δ_{t+l},   其中 δ_t = r_t + γV(s_{t+1}) − V(s_t)
```

漂亮在哪:它把"对无穷多个 k-step 估计做加权平均"这个看似昂贵的操作,压缩成"对 TD residual 序列做一个折扣因子为 `(γλ)` 的几何加权和"。而几何加权和有递推:

```
Â_t^{GAE} = δ_t + (γλ) Â_{t+1}^{GAE}
```

这意味着**从轨迹末尾往前反向扫一遍就能 O(T) 算出所有时间步的 advantage**,实现起来只有几行。下一节的代码就是这个递推。

### 两个端点:λ 把偏差-方差连成一条线

把 λ 取极端,验证它确实在 1-step TD 和 MC 之间插值:

**λ = 0:** `(γλ)^l = 0` 对所有 l≥1,只剩 l=0 项:

```
Â_t^{GAE(γ,0)} = δ_t = r_t + γV(s_{t+1}) − V(s_t)
```

这就是 **1-step TD advantage**:**最低方差**(随机性只来自一步奖励 + 相邻两个 V),**最高偏差**(完全依赖 critic 的 bootstrap,critic 不准则偏差大)。

**λ = 1:** `(γλ)^l = γ^l`,变成

```
Â_t^{GAE(γ,1)} = Σ_{l≥0} γ^l δ_{t+l}
```

用上一节望远镜和(令 k→∞,假设 episode 终止或 γ<1 使尾项可忽略):

```
Σ_{l≥0} γ^l δ_{t+l} = Σ_{l≥0} γ^l r_{t+l} − V(s_t) = G_t − V(s_t)
```

这就是**蒙特卡洛 advantage**:**无偏**(`G_t` 是真采样回报,critic 只作为 baseline `−V(s_t)` 进来,不引入偏差),**最高方差**(累积整条轨迹的随机)。

**所以 λ 是一个把偏差-方差谱"连续化"的旋钮:λ:0→1 对应 偏差↓、方差↑。** γ 也影响这个权衡(下面专门讲),但 GAE 的设计精髓是:把 γ 留给"问题本身的折扣/有效视野",把 λ 留给"估计器的偏差-方差调节",两个旋钮职责分离。典型默认 `γ=0.99, λ=0.95`(数值见文末「待核」,但这个量级是社区常用经验值)。

### γ 和 λ 各管什么:两个旋钮的分工

很多人把 γ 和 λ 混作一谈,这里讲清楚分工——这是 GAE 论文最重要的概念贡献之一。

- **γ(折扣因子)** 出现在 `δ_t = r_t + γV(s_{t+1}) − V(s_t)` 里,它定义了**优化目标本身**:我们要最大化的是折扣回报 `Σ γ^t r_t`。γ 越小,智能体越短视(有效视野约 `1/(1−γ)` 步),也顺带降低方差(远期奖励被压扁,随机性贡献小),但**引入对真实目标(通常是无折扣或更长视野)的偏差**——你优化的是一个被人为缩短视野的代理目标。
- **λ(GAE 参数)** 不改变目标,只改变**怎么估计这个目标下的 advantage**:在"多信 critic 的 bootstrap(小 λ,低方差高偏差)"和"多信真实采样回报(大 λ,高方差低偏差)"之间插值。

一句话:**γ 决定"你想要什么"(并带来视野截断偏差),λ 决定"你怎么估"(并带来估计偏差-方差权衡)。** 实践经验是 γ 由任务有效视野决定(连续控制常 0.99,长视野任务更高),λ 通常取 0.9~0.97 这个区间,因为大多数任务里 1-step 偏差太大、纯 MC 方差太大,中间偏 MC 一点效果最好。

## 第四步:A2C 与 A3C——并行收集进一步稳训练

### 问题:单条轨迹的样本相关性

即便有了 GAE,如果你用单个 actor 串行跑一条轨迹再更新,相邻样本高度时间相关(都来自同一条轨迹),小 batch 的梯度估计仍然抖。监督学习靠 shuffle 打散相关性,RL 在线采样没法直接 shuffle。解法:**并行跑多个 actor,把不同 actor 当前步的数据拼成一个 batch**,样本来自不同轨迹/不同状态,相关性天然被打散,梯度更稳。

### A3C(Asynchronous Advantage Actor-Critic,2016,Mnih 等)

A3C 的做法是**异步**:开 N 个 worker 线程,每个有自己的环境副本和一份策略参数,各自跑 t_max 步、各自算梯度,然后**异步地、无锁地**把梯度累加到一份全局共享参数上(Hogwild 风格)。优点:不需要 GPU、CPU 多核就能跑;异步本身的"参数轻微不一致"还起到一点正则/探索作用。缺点:**梯度是用稍旧的参数算的(stale gradient),引入偏差和噪声;异步逻辑复杂、难复现**。

### A2C(Advantage Actor-Critic,A3C 的同步版)

后来发现异步带来的好处主要是"并行打散相关性",而不是"异步"本身。于是有了 **A2C**:N 个 actor **同步**走 t_max 步,**等所有 actor 都走完**,把所有 `(s,a,r)` 拼成一个大 batch,算一次 GAE、做一次梯度更新,再把新参数广播给所有 actor。

```
A2C 主循环(同步):
  for iteration:
      for 每个并行 env(共 N 个),并行执行:
          用当前 π_θ 走 t_max 步,记录 (s,a,r, V(s), done)
      # 现在有 N×t_max 条 transition
      对每个 env 的每条轨迹,反向扫算 GAE → Â_t,以及 return R_t = Â_t + V(s_t)
      把所有 transition 拼成 batch
      算 actor loss、critic loss、entropy bonus,一次反传更新 θ, φ
```

**A2C 通常比 A3C 更稳、更易复现、GPU 利用率更高**(批量大、无 stale gradient),实践中已基本取代 A3C。注意:A2C 是 **on-policy**——每次更新用的数据必须来自当前(或上一拍)策略,更新后数据即作废,这跟 DQN 那种 off-policy 经验回放是两种范式(样本效率上 on-policy 吃亏,稳定性和实现简单性上占优)。

## 第五步:把它写出来——A2C + GAE

下面是体现机制的核心代码(PyTorch 风格,省略环境/网络搭建噪声)。三块:GAE 计算、loss、训练循环。

### GAE 反向累加(O(T),最该会写的一段)

```python
import torch

def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """
    rewards:    [T]  每步即时奖励 r_t
    values:     [T]  critic 对 s_t 的估计 V(s_t)
    dones:      [T]  s_{t+1} 是否为终止(0/1),终止则切断 bootstrap
    last_value: 标量 V(s_T),用于 t=T-1 的 bootstrap(若末步终止则传 0)
    返回:advantages [T], returns [T]
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0                                    # Â_{t+1} 的累加器,从末尾往前
    for t in reversed(range(T)):                 # 反向扫一遍,利用递推
        # 下一个状态的价值;若 s_{t+1} 终止,则没有未来,bootstrap 置 0
        next_value = last_value if t == T - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        # δ_t = r_t + γ V(s_{t+1})·(非终止) − V(s_t)
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        # Â_t = δ_t + (γλ)·Â_{t+1}·(非终止)   ← GAE 的核心递推
        gae = delta + gamma * lam * next_nonterminal * gae
        advantages[t] = gae
    returns = advantages + values                # critic 的回归目标:R_t = Â_t + V(s_t)
    return advantages, returns
```

两个关键工程点,新手必踩:(1) `next_nonterminal` 在 done 处把 `(γλ)·Â_{t+1}` 和 `γV(s_{t+1})` 都切断,否则会把上一条 episode 的未来"漏"进这一条,advantage 全错;(2) `returns = advantages + values`,因为 `Â_t = R_t − V(s_t)` ⇒ `R_t = Â_t + V(s_t)`,这个 `R_t` 是 GAE 给 critic 配套的回归目标(也叫 λ-return,记 `R_t^λ`),**critic 要学的不是裸 MC return,而是这个跟 advantage 一致的目标**——保持二者一致才不会自相矛盾。

### Loss:actor + critic + entropy

```python
def a2c_loss(logp, values, advantages, returns, entropy,
             vf_coef=0.5, ent_coef=0.01):
    """
    logp:       [B]  当前策略对采样动作的 log π_θ(a|s)
    values:     [B]  critic 当前对这些 s 的 V_φ(s)
    advantages: [B]  GAE 算出的 Â_t(已 detach,不回传到 critic)
    returns:    [B]  R_t^λ,critic 的回归目标(detach)
    entropy:    [B]  策略熵 H(π_θ(·|s)),鼓励探索
    """
    advantages = advantages.detach()             # ★ advantage 当常数,梯度只走 ∇logπ
    # actor:最大化 E[logπ · Â]  ⇒  最小化它的相反数
    policy_loss = -(logp * advantages).mean()
    # critic:把 V_φ(s) 回归到 R_t^λ(MSE)
    value_loss = 0.5 * (returns.detach() - values).pow(2).mean()
    # entropy bonus:加大熵防过早收敛到确定性策略(reward hacking 的常见诱因)
    entropy_loss = -entropy.mean()
    return policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
```

**最容易写错的一行是 `advantages.detach()`。** advantage 在 actor 的目标里**必须当作常数系数**:策略梯度是 `E[∇log π · Â]`,梯度只应该流过 `∇log π`。如果忘了 detach,梯度会顺着 `Â`(里面含 `V_φ`)反传到 critic,把 actor loss 和 critic 的训练耦合在一起,数学上不再是策略梯度,训练会诡异地不收敛——这是新手 debug 半天找不到的经典坑。同理 critic 那边 `returns.detach()`,因为 `returns` 里也含 `V_φ`(bootstrap),它是 critic 的"目标",目标不该有梯度(否则就是 critic 自己追自己,类似 DQN 里要用 target network 的动机的简化版)。

### 训练循环骨架

```python
for iteration in range(num_iters):
    # 1) rollout:N 个并行 env 各走 t_max 步(向量化环境一次 step 全部 env)
    buf = []                                     # 存 (s,a,r,logp,V,done)
    obs = envs.reset_if_needed()
    for step in range(t_max):
        with torch.no_grad():
            dist = policy(obs)                   # actor 给出动作分布
            value = critic(obs)                  # critic 给出 V(s)
        action = dist.sample()
        logp = dist.log_prob(action)
        next_obs, reward, done, _ = envs.step(action)
        buf.append((obs, action, reward, logp, value, done))
        obs = next_obs

    # 2) bootstrap 末状态价值,逐 env 算 GAE
    with torch.no_grad():
        last_value = critic(obs)                 # [N]
    adv, ret = compute_gae_per_env(buf, last_value, gamma, lam)

    # 3) 一次梯度更新(A2C:整批一次;PPO:这里会做多 epoch + clip)
    dist = policy(batch_obs); value = critic(batch_obs)
    logp = dist.log_prob(batch_act); entropy = dist.entropy()
    loss = a2c_loss(logp, value, adv, ret, entropy)
    optimizer.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(params, max_norm=0.5)   # ★ 梯度裁剪,RL 几乎必加
    optimizer.step()
```

`clip_grad_norm_` 那行不是可选项:策略梯度偶发的大梯度(一条极端轨迹)会瞬间把网络打飞,裁剪全局范数是 RL 训练最便宜有效的稳定手段之一。

## 量化分析:方差、偏差、复杂度、样本效率

### 方差:为什么 critic 能压方差(给个量级)

设奖励有界 `|r| ≤ R_max`。蒙特卡洛回报 `G_t = Σ_{l≥0} γ^l r_{t+l}` 的方差,因为它是大量(有效 `1/(1−γ)` 项)随机奖励的折扣和,粗略地随有效视野 `H_eff = 1/(1−γ)` 增长——量级上 `Var[G_t] = O(H_eff · σ_step²)` 级别(σ_step 为单步随机性,精确常数依赖相关结构)。而 GAE 的 advantage 是 TD residual 的几何加权和 `Σ (γλ)^l δ_{t+l}`,有效项数约 `1/(1−γλ)`。所以**有效方差随 `1/(1−γλ)` 增长**:

```
MC      有效视野 ≈ 1/(1−γ)         (λ=1)
GAE     有效视野 ≈ 1/(1−γλ)        (一般 λ<1)
1-step  有效视野 ≈ 1               (λ=0)
```

举个数:`γ=0.99`。MC 有效视野 `1/(1−0.99)=100`;GAE 取 `λ=0.95`,`γλ=0.9405`,有效视野 `1/(1−0.9405)≈17`。**单这一步就把"方差累积窗口"从 100 步缩到 ~17 步,方差量级降了约 6 倍**,而偏差只引入"对那 ~17 步之外用 critic 估"这一点点(且 critic 越准偏差越小)。这就是 GAE 性价比极高的定量来源。

### 偏差:critic 误差怎么传进 advantage

设 critic 误差 `ε(s) = V(s) − V^π(s)`。由 `Â^{GAE} = Σ(γλ)^l δ_{t+l}`,且 `δ_{t+l}` 里 `ε` 的贡献是 `γ ε(s_{t+l+1}) − ε(s_{t+l})`,加权求和后又是一个望远镜结构,净偏差大致正比于 `ε` 在被加权窗口内的"残留"。定性结论:**λ 越小,bootstrap 越多,critic 误差进入越多 ⇒ 偏差越大;λ→1 时 bootstrap 项相消殆尽(退化为 `G_t − V(s_t)`),critic 误差只剩 baseline 项 `−ε(s_t)`,不进入梯度期望,偏差→0。** 这跟前面端点分析完全一致,互相印证。

### 计算与空间复杂度

- **GAE 计算:** 反向扫一遍轨迹,每步 O(1),总 **O(T)** 时间、O(T) 空间(存 δ 或直接存 advantage)。可忽略,远小于网络前反传。
- **每次迭代主成本:** rollout 的 `N×t_max` 次前向 + 一次 batch 的前反传,跟普通深度学习一个量级。actor/critic 共享 backbone 还能省一半前向。
- **A2C vs A3C:** A2C 同步,GPU 上 batch 大、利用率高;A3C 异步多 CPU 线程,无 GPU 也能跑但有锁/stale 开销和复现难题。

### 样本效率:on-policy 的硬约束

A2C/GAE 是 **on-policy**:每批数据更新一次就作废(更新后策略变了,旧数据不再服从当前 `π_θ`,直接复用会引入分布偏移)。对比 off-policy(DQN/SAC 用 replay buffer 反复用旧数据),on-policy **样本效率明显更低**。这是用"稳定 + 实现简单 + 无重要性采样的高方差问题"换来的代价。**PPO 正是在 A2C+GAE 基础上,用一个 clipped 重要性比值,让同一批数据能安全地多走几个 epoch 的梯度,部分缓解样本效率问题**——这是下一章的主角,而它的 advantage 计算用的就是本章的 GAE,一字不差。

## 设计权衡与常见坑

- **λ 选多大?** 经验区间 0.9~0.97,默认 0.95。太小(→0)偏差主导,critic 没学好时 advantage 系统性带歪;太大(→1)退化成高方差 MC,白用了 critic。**调参顺序建议:先固定 γ 由任务视野定,再扫 λ。**
- **γ 不是越大越好。** γ 太接近 1,有效视野极长,方差爆炸且 credit assignment 极难;γ 太小则短视、对真实长期目标有偏。γ 是"目标"层面的选择,不要拿它当方差旋钮用——那是 λ 的活。
- **忘了 detach advantage / returns。** 前面强调过,最高频的静默 bug,梯度乱流导致训练莫名其妙不收敛。
- **done 处不切断 bootstrap。** episode 边界不置 `next_nonterminal=0`,会把下一条 episode 的价值漏进来,advantage 整体污染。向量化多环境时每个 env 的 done 要独立处理。
- **critic 学不动 → advantage 全是噪声。** 如果 critic loss 不降,`δ_t` 接近随机,actor 拿到的是噪声信号。常见原因:value 目标量级太大没归一化、学习率不匹配、共享 backbone 时 critic 梯度被 actor 淹没(可调 `vf_coef` 或分开网络)。
- **reward scaling / advantage normalization。** 实践里几乎都会对 advantage 做 batch 内标准化(减均值除标准差),进一步稳住梯度尺度;但要注意这会轻微改变隐含目标,属于"工程上有效、理论上近似"的操作。
- **熵塌缩(entropy collapse)与 reward hacking。** 没有 entropy bonus 或系数太小,策略会过早变确定性,卡在次优、或钻奖励函数的空子(reward hacking,RL 的头号噩梦)。监控策略熵曲线:它不该太快掉到接近 0。
- **别用 Q(s,a) 当 baseline 求"无偏"。** baseline 必须只依赖状态;依赖动作的 baseline 会破坏无偏性(虽有 action-dependent baseline 的进阶研究,但默认不要这么干)。

## 动手练习

**练习 1(推导题,核心)。** 不看正文,从 `Â_t^{GAE} := (1−λ) Σ_{k≥1} λ^{k-1} Â_t^{(k)}` 出发,独立推出 `Â_t^{GAE} = Σ_{l≥0} (γλ)^l δ_{t+l}`。
- 提示:先证引理 `Â_t^{(k)} = Σ_{l=0}^{k-1} γ^l δ_{t+l}`(望远镜相消);再代入、交换 k/l 求和次序,内层对 `k≥l+1` 求等比和;注意 `(1−λ)` 归一化常数最后约掉。卡住就检查求和上下限的等价改写 `{k≥1, 0≤l≤k-1} ⇔ {l≥0, k≥l+1}`。

**练习 2(分析题)。** 取 `γ=0.99`。分别计算 `λ ∈ {0, 0.9, 0.95, 0.99, 1.0}` 时 GAE 的"有效视野" `1/(1−γλ)`,列成表。据此论证:(a) 为什么 `λ=0.95` 是常用默认;(b) 当 critic 很不准(误差大)时,你应该把 λ 往哪个方向调,为什么;(c) 当采样极其昂贵(每条轨迹很贵、batch 很小)时,又该往哪调,为什么。
- 提示:critic 不准 ⇒ 偏差是主要敌人 ⇒ 少 bootstrap;采样贵 ⇒ 单批方差是主要敌人 ⇒ 多 bootstrap。两个诉求方向相反,体会"没有万能 λ"。

**练习 3(编码题)。** 把正文 `compute_gae` 扩展成向量化多环境版 `compute_gae_per_env`,输入形状 `[T, N]`(N 个并行 env),正确处理逐 env 的 `dones` 和逐 env 的 `last_value`。然后写一个单元测试:构造一条 `dones` 全 0、`V≡0`、`r` 已知的轨迹,验证 `λ=1` 时 GAE 输出等于手算的 `G_t`,`λ=0` 时等于 `δ_t = r_t + γV(s_{t+1}) − V(s_t)`。
- 提示:`λ=1, V=0` 时 `Â_t = Σ γ^l r_{t+l}`,反向累加 `g = r_t + γ g` 即可对照。注意 `next_value` 在 `t=T-1` 用 `last_value`。

**练习 4(推导+分析题)。** 证明:当 critic 是真值 `V=V^π` 时,GAE(对任意 λ∈[0,1])给出的 advantage 估计在期望意义下都无偏,即 `E[Â_t^{GAE} | s_t,a_t]` 对所有 λ 一致(等于真 advantage 的某个表示)。再说明:为什么现实中不同 λ 会给出不同表现——偏差来自哪里?
- 提示:`V=V^π` 时 `E[δ_{t+l}]` 沿轨迹的结构使加权和的期望与 λ 无关(都对应真 advantage);现实偏差来自 `V≠V^π`,而不同 λ 对 critic 误差的放大不同(见"偏差"小节)。

## 源码 / 论文导读

- **GAE 原论文:** Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*(ICLR 2016,arXiv:1506.02438)。重点读第 3 节 GAE 的定义与 `Σ(γλ)^l δ` 推导、以及把 γ 和 λ 解耦为"两个分别控制偏差的旋钮"的论述(第 3-4 节)。本章的核心推导即出自此。
- **A3C 原论文:** Mnih et al., *Asynchronous Methods for Deep Reinforcement Learning*(ICML 2016)。读 advantage actor-critic 的目标、异步更新机制、entropy regularization 的引入动机。A2C 是其同步变体(OpenAI Baselines 博客明确化)。
- **策略梯度与 baseline 的系统处理:** Sutton & Barto, *Reinforcement Learning: An Introduction*(2nd ed.)第 13 章,baseline 不引入偏差的证明、actor-critic 的呈现。要把第一节的 `=0` 推导和书里对齐。
- **开源实现(强烈建议对照):**
  - **CleanRL** 的 `ppo.py` / `a2c` 系列单文件实现——GAE 那段反向循环和本章 `compute_gae` 几乎一一对应,是最好的"照着读"材料(单文件、无抽象层)。
  - **TRL**(HuggingFace)在做 RLHF 的 PPO 时,advantage 计算同样是 GAE;看它在语言模型场景下 `compute_advantages` 的实现,体会 GAE 怎么从游戏迁到 LLM 对齐。
  - **verl**(字节开源的 RLHF/RL 训练框架)里大规模 PPO 的 advantage 模块,看工业级实现怎么处理向量化、归一化、分布式。
  - **Stable-Baselines3** 的 `A2C` / `PPO` 的 `RolloutBuffer.compute_returns_and_advantage`——生产级、带 done 处理和归一化,适合对照工程细节。

(以上库的具体函数名/路径随版本变动,标「待核」;但"找它们的 GAE/advantage 计算函数"这个定位是稳定的。)

## 小结与承上启下

这一章把策略梯度从"能跑但抖"推到了"能稳"。主线只有一句话:**在不引入(或只引入可控)偏差的前提下,把方差系统性压下来。** 路径是:

1. 减一个**只依赖状态的 baseline 不改变梯度期望**(`E[∇log π · b(s)] = 0`),这是一切方差削减的合法性来源;最优 baseline 是 score 加权回报,实践用 `V(s)` 这个极好近似。
2. 引入 **critic**,用 bootstrap 把高方差的 MC 回报换成低方差的 TD 估计,代价是 critic 误差带来的偏差——于是有了从 1-step TD 到 MC 的**偏差-方差谱**,刻画为 k-step advantage `Â^{(k)} = Σ_{l=0}^{k-1} γ^l δ_{t+l}`。
3. **GAE** 用几何加权把所有 k 融合,化简为 `Â^{GAE} = Σ(γλ)^l δ_{t+l}`,λ 在偏差(λ→0,1-step,低方差高偏差)和方差(λ→1,MC,无偏高方差)之间**连续插值**;γ 管"目标与视野",λ 管"估计的偏差-方差",两个旋钮分工。
4. **A2C/A3C** 用并行 actor 打散样本相关性,A2C 同步版更稳更易复现,已成主流;它是 on-policy,样本效率是它的硬约束。

在整门课里,这一章是**承前启后的枢纽**:前面(策略梯度)给了无偏但不稳的地基,这里把它变得可训;而它直接通向下一章 **PPO**——PPO 的 advantage 就是本章的 GAE 原封不动,它只是再加一个 clipped 重要性比值,让 on-policy 数据能安全多用几轮,把样本效率也补上。把本章的 `compute_gae` 和 detach 这两个点吃透,你就已经写出了 PPO 的一半。再往后,这套"actor 出动作、critic 估优势、GAE 算信号"的骨架会几乎原样迁移到 RLHF / 推理模型训练里——只不过环境换成了语言模型的生成过程,奖励换成了人类偏好或可验证奖励。**方差控制不是 RL 的一个技巧,是 RL 能不能真正落地的命门**,这一章是你掌握它的起点。
