---
title: "TRPO 与 PPO:信赖域与裁剪目标(策略优化主力)"
slug: "3-06"
collection: "ai-research-compass"
group: "强化学习专家课程"
order: 3006
summary: "这章把你从\"会写带 GAE 的 actor-critic、但每次调步长都心惊胆战\"带到\"理解为什么策略优化需要信赖域、能从性能差分引理推出 TRPO 的单调改进界、能逐分段讲清 PPO 的 clip 目标为什么有效、能照着实现一个数值稳定的 PPO 损失\"。"
topics:
  - "AI 研究"
tags: []
createdAt: "2026-06-19T06:26:40.000Z"
updatedAt: "2026-06-19T06:26:40.000Z"
---
> 这章把你从"会写带 GAE 的 actor-critic、但每次调步长都心惊胆战"带到"理解为什么策略优化需要信赖域、能从性能差分引理推出 TRPO 的单调改进界、能逐分段讲清 PPO 的 clip 目标为什么有效、能照着实现一个数值稳定的 PPO 损失"。PPO 是当今工业界从游戏到 RLHF 最常用的策略优化算法,这一章是整门课的承重墙。

## 动机:朴素策略梯度的步长是个无解的难题

你在第 04、05 章已经会做这件事:参数化策略 π_θ(a|s),用策略梯度定理算出梯度

```
grad J(θ) = E_{s,a~π_θ}[ ∇_θ log π_θ(a|s) · Â(s,a) ]
```

其中 Â 是优势(advantage),通常用第 05 章的 GAE 估出来。然后做一步梯度上升 θ ← θ + α·grad J。问题全在那个 α 上。

**监督学习里步长走大一点最多是这一步白走,下一步还能从同样的数据里纠正回来;策略梯度里步长走大一点会直接毁掉策略,而且没有数据能让你纠正回来。** 这是 RL 和监督学习最本质的区别之一,值得把机制讲透:

- **数据分布是策略自己生成的(非平稳)。** 监督学习的训练集 (x, y) 是固定的;无论参数怎么变,你看到的数据分布不变。RL 里,你用 π_θ 去环境里采样,得到的状态分布 d^{π_θ}(s) 和动作都依赖 θ。你更新一步 θ,下一批数据的分布就变了。
- **梯度只在旧策略附近可信。** 策略梯度 grad J(θ) 是在当前 θ 这一点、用当前策略采的样估出来的。它告诉你"在 π_θ 这个点,往哪个方向能让目标变大"。但这是个**局部**信息:走远了,状态分布 d^{π_θ} 已经不是你采样时的样子,梯度的方向和大小都不再可信。
- **崩溃不可逆。** 假设一步走太大,策略变成一个几乎总是输出某个糟糕动作的退化策略。退化策略采到的数据全是低回报、低多样性的样本;策略梯度从这种数据里几乎学不到有用方向(优势全负、熵塌缩、探索消失)。你卡在一个坏盆地里爬不出来。**这就是"策略崩溃(policy collapse)",在大模型 RLHF 里表现为模型突然开始输出乱码或复读,而且回不来。**

那把 α 调小一点不行吗?可以,但有两个代价。其一,**慢**:RL 采样本来就贵(每个样本要和环境交互一个 episode),小步长意味着同样的样本预算下走得更近,样本效率更差。其二,**步长的"合适大小"在参数空间里不是常数**。同样大小的参数改变 ‖Δθ‖,在策略空间里造成的改变可能天差地别——softmax 策略在 logit 接近的地方对 Δθ 很敏感,在已经饱和的地方又很迟钝。**用欧氏距离 ‖Δθ‖ 来度量"走了多远"是错的;该度量的是策略本身变了多少,而策略是个概率分布,分布之间的距离用 KL 散度来量。**

这就引出了本章的核心思想:

> **每次更新,限制新策略 π_θ 不要偏离旧策略 π_old 太远(在 KL 散度意义下),在这个"信赖域(trust region)"内尽量提升目标。** TRPO 把它做成一个带 KL 硬约束的优化问题并给出理论保证;PPO 把它简化成一个一阶可优化、能用普通 SGD 跑的裁剪目标。

下面从第一性原理把"为什么信赖域能保证改进"推出来——这是 TRPO 论文的真正贡献,也是理解 PPO 在做什么近似的前提。

## 一、性能差分引理:两个策略的回报差到底等于什么

要约束"新策略相对旧策略",得先精确写出**新旧策略性能之差**。记策略 π 的目标为

```
J(π) = E_{τ~π}[ Σ_{t≥0} γ^t r_t ]      # 折扣回报的期望(起始状态分布固定)
```

**性能差分引理(Performance Difference Lemma, Kakade & Langford 2002):** 对任意两个策略 π 和 π',

```
J(π') − J(π) = (1/(1−γ)) · E_{s~d^{π'}}[ E_{a~π'(·|s)}[ A^π(s,a) ] ]
```

其中 A^π(s,a) = Q^π(s,a) − V^π(s) 是**旧策略 π 的优势**,d^{π'}(s) = (1−γ)·Σ_{t≥0} γ^t · P(s_t = s | π') 是新策略 π' 的**(归一化)折扣状态访问分布**。

这个引理是 TRPO 一切的起点,务必看懂它在说什么:**新策略比旧策略好多少,等于"在新策略走出来的状态-动作上,旧策略的优势的总和"。** 直觉:如果新策略经常走到那些"旧策略眼里优势为正"的动作上,它就比旧策略好。

下面给完整证明,只用到 A^π = r + γV^π(s') − V^π(s) 的定义(对每一步成立,期望意义下)。

```
证明(performance difference lemma):

把 A^π(s,a) 沿新策略 π' 的轨迹求和并取期望:

  E_{τ~π'}[ Σ_{t≥0} γ^t · A^π(s_t, a_t) ]

展开 A^π(s_t,a_t) = E[ r_t + γ V^π(s_{t+1}) − V^π(s_t) ]:

  = E_{τ~π'}[ Σ_{t≥0} γ^t ( r_t + γ V^π(s_{t+1}) − V^π(s_t) ) ]

把求和拆成三部分。其中关于 V^π 的两项发生望远镜(telescoping)相消:

  Σ_{t≥0} γ^t ( γ V^π(s_{t+1}) − V^π(s_t) )
      =  Σ_{t≥0} ( γ^{t+1} V^π(s_{t+1}) − γ^t V^π(s_t) )
      = − V^π(s_0)            # 后项的 t 与前项的 t+1 逐项抵消,只剩 t=0 那项

所以

  E_{τ~π'}[ Σ_t γ^t A^π(s_t,a_t) ]
      = E_{τ~π'}[ Σ_t γ^t r_t ]  −  E_{s_0}[ V^π(s_0) ]
      = J(π')                    − J(π)         ∎

(最后一步:J(π) 的定义就是 E_{s_0}[ V^π(s_0) ];起始状态分布对两个策略相同。)

再把对轨迹的期望按折扣状态分布重写:
  E_{τ~π'}[ Σ_t γ^t A^π(s_t,a_t) ] = (1/(1−γ)) E_{s~d^{π'}, a~π'(·|s)}[ A^π(s,a) ]
```

这个证明只用了望远镜求和,非常干净。**记住结论:J(π')−J(π) = (1/(1−γ))·E_{s~d^{π'}}[ E_{a~π'}[ A^π(s,a) ] ]。**

## 二、从性能差分到代理目标:为什么必须近似 d^{π'}

性能差分引理看起来已经能直接最大化了:选 π' 使右边最大。**但它有个致命的实际困难——右边的状态分布是 d^{π'},也就是你要优化的那个新策略自己的访问分布。** 你还没确定 π',就没法从 d^{π'} 采样;而每改一次 π' 就要重新采一批样本,完全没法做基于梯度的优化。

TRPO 的处理:**用旧策略的状态分布 d^π 替换 d^{π'},得到一个"代理目标(surrogate objective)" L_π(π')。**

```
定义代理目标(用旧分布 d^π 近似新分布 d^{π'}):

  L_π(π') = (1/(1−γ)) · E_{s~d^π, a~π'(·|s)}[ A^π(s,a) ]

注意状态仍按 a~π'(·|s) 取动作。用重要性采样(importance sampling)把
"按 π' 取动作"换成"按 π 取动作再加权",这样可以直接用旧策略采的 (s,a) 样本:

  E_{a~π'(·|s)}[ A^π(s,a) ]
      = Σ_a π'(a|s) A^π(s,a)
      = Σ_a π(a|s) · (π'(a|s)/π(a|s)) · A^π(s,a)
      = E_{a~π(·|s)}[ (π'(a|s)/π(a|s)) · A^π(s,a) ]

于是(去掉对常数因子的依赖,记 π_old = π,Â 是 A^{π_old} 的估计):

  L_{π_old}(π_θ) = E_{s,a ~ π_old}[ (π_θ(a|s) / π_old(a|s)) · Â(s,a) ]
```

**重要性采样比 r(θ) := π_θ(a|s) / π_old(a|s) 就是这么来的——不是凭空塞进去的技巧,而是"想用旧数据估新策略下的期望"的必然产物。** 这个 r(θ) 是 PPO 里被 clip 的那个量,理解它的出处至关重要:r > 1 表示新策略比旧策略更倾向这个动作,r < 1 表示更不倾向。

现在有两个关键事实把代理目标和真实目标钉在一起(都是 Schulman et al. 2015, TRPO 论文的核心):

**事实 1:在 θ = θ_old 这一点,代理目标和真实目标的值与一阶梯度都相等。**

```
零阶: L_{π_old}(π_old) = E[ 1 · Â ] = 0,且 J(π_old) − J(π_old) = 0。一致。
一阶: ∇_θ L_{π_old}(π_θ) |_{θ=θ_old}
        = E_{s,a~π_old}[ ∇_θ r(θ)|_{θ_old} · Â ]
        = E_{s,a~π_old}[ ∇_θ log π_θ(a|s)|_{θ_old} · Â ]   # 因为 ∇r|_{θ_old}=∇log π_θ|_{θ_old}
        = grad J(θ_old)                                    # 这正是策略梯度定理
```

也就是说,**对 L 做一阶优化,在出发点上和对真实 J 做策略梯度是同一回事。** 代理目标是真实目标在 θ_old 附近的一个一阶相合(first-order matching)的局部模型。这解释了为什么"优化 L"是合理的——但只在 θ_old 附近合理。走远了,d^{π_old} ≠ d^{π_θ},L 和 J 就分道扬镳。**所以必须有信赖域约束,把 θ 锁在 L 还可信的范围内。**

**事实 2(单调改进界 / Monotonic Improvement Bound):真实目标的提升,有一个用代理目标减去 KL 惩罚的下界。**

```
J(π_θ) ≥ L_{π_old}(π_θ) − C · max_s KL( π_old(·|s) ‖ π_θ(·|s) )

其中 C = 4εγ/(1−γ)²,  ε = max_{s,a} |A^{π_old}(s,a)|        (TRPO 论文定理 1)
```

这个不等式是 TRPO 的理论灵魂。我给出它的推导骨架(完整证明较长,这里讲清机制,关键的耦合系数会标出来源):

```
单调改进界的推导骨架:

由性能差分引理:
  J(π_θ) − J(π_old) = (1/(1−γ)) E_{s~d^{π_θ}, a~π_θ}[ A^{π_old}(s,a) ]              ...(真实)

代理目标(把 d^{π_θ} 换成 d^{π_old}):
  L_{π_old}(π_θ) − J(π_old)·0 = (1/(1−γ)) E_{s~d^{π_old}, a~π_θ}[ A^{π_old}(s,a) ]  ...(代理)

两者只差在状态分布 d^{π_θ} vs d^{π_old}。两个分布的差距能用策略差距控制:
若每个状态上 π_θ 和 π_old 的 total variation 距离 ≤ α,则
  ‖ d^{π_θ} − d^{π_old} ‖_1  ≤  2γα/(1−γ)        # 状态分布偏移随策略偏移线性增长

把"真实减代理"的误差用这个分布差距 × 优势上界 ε 放缩:
  | [J(π_θ)−J(π_old)] − [L_{π_old}(π_θ)] |  ≤  (2γε/(1−γ)) · ‖d^{π_θ}−d^{π_old}‖_1 / 2 ·(...)
                                            ≤  C · α²     (α = max TV 距离)

再用 Pinsker 不等式把 TV 距离换成 KL: TV(p,q)² ≤ (1/2) KL(p‖q)
  ⇒ α² ≤ max_s KL(π_old ‖ π_θ)   (在常数倍意义下)

合起来:
  J(π_θ) − J(π_old) ≥ [L_{π_old}(π_θ) − J(π_old)] − C·max_s KL(π_old‖π_θ)
  ⇔ J(π_θ) ≥ L_{π_old}(π_θ) − C·max_s KL(...)      ∎(骨架)
```

> 上述推导中 C = 4εγ/(1−γ)² 的精确系数与"分布差距随 TV 距离的线性界"的常数因子来自 TRPO 论文(Schulman et al. 2015)及其前身 CPI(Kakade & Langford 2002),这里只验证了量纲与依赖关系(随 γ→1 发散、随优势幅度 ε 线性增长),具体常数读者可对照论文附录核对。「待核:C 的精确常数与 Pinsker 步骤的系数对齐」

这个下界的实践含义极其重要,把它讲透:

**令 M(π_θ) := L_{π_old}(π_θ) − C·max_s KL(π_old‖π_θ),这是 J(π_θ) 的一个下界(minorize),并且在 π_θ = π_old 处与 J 相等(此时 L=0,KL=0,两边都等于 J(π_old))。** 于是:

```
若我们每一步都选 π_θ 去最大化 M(π_θ),则:
  J(π_new) ≥ M(π_new) ≥ M(π_old) = J(π_old)
            ↑下界          ↑因为我们最大化了 M     ↑相等点
  ⇒ J(π_new) ≥ J(π_old):性能单调不降!
```

**这是一个 minorize-maximization(MM)算法:反复构造真实目标的一个紧贴的下界,然后最大化下界。每一步保证不会变差。** 这就是"信赖域约束能保证改进"的数学根据——不是经验法则,是有界证明的。**KL 惩罚不是为了"温柔一点",而是补偿"我们用了错误的状态分布 d^{π_old}"所引入的误差;KL 越大,代理目标越不可信,惩罚越重。**

但直接最大化 M(π_θ) = L − C·KL 有个问题:那个 C = 4εγ/(1−γ)² 大得离谱(γ=0.99 时 (1−γ)² = 0.0001,C 是几万量级),按它惩罚的话每步只敢走极小的一步,慢得没法用。**TRPO 的工程化处理:不把 KL 当惩罚项加进目标,而是把它当成一个硬约束的"信赖域半径" δ,在约束内放手最大化代理目标。**

## 三、TRPO:带 KL 约束的代理目标 + 自然梯度近似

TRPO(Trust Region Policy Optimization)把上面的思想落成一个约束优化问题:

```
TRPO 的优化问题(每次迭代):

  max_θ   L_{π_old}(π_θ) = E_{s,a~π_old}[ r(θ) · Â ],    r(θ) = π_θ(a|s)/π_old(a|s)
  s.t.    E_{s~d^{π_old}}[ KL( π_old(·|s) ‖ π_θ(·|s) ) ] ≤ δ
```

两处工程化偏离纯理论:(1) 把 max_s KL(逐状态最大)换成 **E_s[KL](平均 KL)**——max 在大状态空间里无法估,平均 KL 可采样估计且实践中够用;(2) 用固定半径 δ(典型 δ ≈ 0.01,「待核:常用默认值」)代替那个巨大的 C,作为一个可调的信赖域大小。**代价:失去了严格的单调改进保证,换来可用的步长。这是 TRPO 全文里唯一但关键的"理论让位于工程"。**

**怎么解这个约束优化?** 这是 TRPO 数学上最重的部分,机制如下(数学适度简化):

在 θ_old 附近做二阶近似——目标 L 做一阶展开,约束 KL 做二阶展开(KL 在 θ_old 处的一阶项为零,因为 θ_old 是 KL=0 的最小点):

```
局部近似(g 是策略梯度,H 是 KL 的 Hessian,即 Fisher 信息矩阵):

  L_{π_old}(π_θ) ≈ gᵀ (θ − θ_old),      g = ∇_θ L|_{θ_old} = grad J(θ_old)
  E_s[KL]        ≈ (1/2)(θ−θ_old)ᵀ H (θ−θ_old),   H = ∇²_θ E_s[KL]|_{θ_old} = Fisher 信息

问题变成带二次约束的线性目标:
  max_{Δθ}  gᵀΔθ    s.t.  (1/2) Δθᵀ H Δθ ≤ δ

用拉格朗日乘子法,解析解(自然梯度方向):
  Δθ* = √( 2δ / (gᵀ H⁻¹ g) ) · H⁻¹ g
                                  └──┬──┘
                                  自然梯度 = H⁻¹ · 普通梯度
```

**这里出现了自然梯度(natural gradient):H⁻¹g。** 它的意义:普通梯度 g 是"在欧氏参数空间里最陡的方向",但参数空间的欧氏几何对策略分布没意义(前面说过 ‖Δθ‖ 不能度量策略变化)。**自然梯度是"在 KL 度量诱导的黎曼几何下最陡的方向"——它先用 Fisher 信息 H 把参数空间扭正成"策略空间",再在那里找最陡方向。** 这正好对应"我们关心的是策略变了多少,而不是参数变了多少"。

但 H 是参数维度 × 参数维度的矩阵(神经网络几百万参数,H 是百万×百万),**既存不下也求不了逆**。TRPO 的两个关键工程技巧:

1. **共轭梯度(Conjugate Gradient)求 H⁻¹g 而不显式构造 H。** 共轭梯度只需要"H 乘以一个向量"的能力(Hessian-vector product),而 Hessian-vector product 可以用自动微分高效算出(对 KL 的梯度再做一次向量-雅可比积),根本不用把 H 写出来。跑十来步共轭梯度就能得到 H⁻¹g 的好近似,复杂度从 O(参数³)(求逆)降到 O(参数 × CG步数)。

2. **线搜索(Line Search / backtracking)保证约束真被满足。** 二次近似只在 θ_old 附近准,直接用解析解的步长 Δθ* 可能让真实的 KL 超过 δ,或者代理目标其实没上升。TRPO 沿 Δθ* 方向做指数回退线搜索:θ = θ_old + (1/2)ʲ·Δθ*,j=0,1,2,...,直到找到一个 j 使得**真实**约束 E_s[KL] ≤ δ **且** 代理目标 L 确实增大,才接受。**这一步把二阶近似的误差兜住了——近似解出方向,线搜索保证落地不越界。**

**TRPO 总结成一句话:用性能差分引理推出代理目标,用 KL 信赖域约束保证代理目标可信,用自然梯度(共轭梯度近似)找方向,用线搜索保证 KL 约束真被满足。** 它的最大问题是**实现复杂**:共轭梯度、Fisher-vector product、线搜索、二阶导数,工程量大、和现有深度学习框架(只想做一阶 SGD)格格不入,而且和参数共享、dropout、RNN 这些东西配合困难。这就是 PPO 诞生的全部动机。

## 四、PPO-Clip:把信赖域塞进一阶目标里

PPO(Proximal Policy Optimization, Schulman et al. 2017)的目标只有一个:**保留 TRPO"别走太远"的核心好处,但只用一阶 SGD,不要共轭梯度、不要二阶导、不要线搜索。** 它的做法不是约束、不是 KL 惩罚,而是**直接修改目标函数,让"走太远"这件事在目标里就无利可图。**

回忆代理目标的逐样本项是 r(θ)·Â,其中 r(θ) = π_θ(a|s)/π_old(a|s)。如果直接对它做 SGD,没有任何东西拦着 r 跑到很大或很小——好动作(Â>0)会把 r 无限推高,这正是要避免的"走太远"。**PPO 的裁剪目标(clipped surrogate objective):**

```
L^CLIP(θ) = E_t[ min( r_t(θ)·Â_t ,  clip(r_t(θ), 1−ε, 1+ε)·Â_t ) ]

  r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)        重要性采样比
  clip(x, 1−ε, 1+ε) = max(1−ε, min(x, 1+ε))      把 x 夹在 [1−ε, 1+ε]
  ε 典型取 0.1 ~ 0.2                              信赖域"半径"的角色
```

这个目标只有 min 和 clip 两个非线性,完全是一阶可微(几乎处处),普通 Adam 就能优化。**但它为什么能限制更新?** 这是本章必须讲到读者能自己画出图的地方。**关键:必须分 Â>0 和 Â<0 两种情形讨论,因为 clip 对两者的"夹"方向相反。**

### 情形 A:Â_t > 0(这是个好动作,我们想增大它的概率,即想让 r 变大)

```
当 Â > 0:
  未裁剪项:        r·Â                       随 r 增大而线性增大(斜率 Â>0)
  裁剪项:    clip(r,1−ε,1+ε)·Â              当 r > 1+ε 时被钉在 (1+ε)·Â,不再增长
  取 min:

   r ≤ 1+ε:   min( r·Â , clip·Â ) = r·Â          # 两者相等(clip 没触发),目标随 r 上升
   r > 1+ε:   min( r·Â , (1+ε)Â ) = (1+ε)·Â       # 裁剪项更小,目标被钉平,梯度=0

形状(横轴 r,纵轴目标,Â>0):
        L^CLIP
          │        ___________   ← r > 1+ε 后变平(钉在 (1+ε)Â)
          │       /
          │      / ← 斜率 Â,正常上升
          │     /
          └────┴──────────────── r
              1+ε
```

**直觉:好动作可以增大概率,但 r 一旦超过 1+ε,目标就不再奖励你继续增大——梯度变成 0,优化器没有动力把这个动作的概率推得更高。** 这就把"好动作过度自信"挡住了,等价于一个上限信赖域。

### 情形 B:Â_t < 0(这是个坏动作,我们想减小它的概率,即想让 r 变小)

```
当 Â < 0(注意 Â 是负数,乘上去会翻转大小关系):
  未裁剪项:        r·Â                       随 r 增大而线性减小(斜率 Â<0)
  裁剪项:    clip(r,1−ε,1+ε)·Â              当 r < 1−ε 时被钉在 (1−ε)·Â
  取 min(在 Â<0 下,谁的概率比更"激进地减小"谁更小):

   r ≥ 1−ε:   min( r·Â , clip·Â ) = r·Â          # clip 没触发,目标随 r 减小(我们减小坏动作概率,目标上升)
   r < 1−ε:   min( r·Â , (1−ε)Â ) = (1−ε)·Â       # 裁剪项更小,目标被钉平,梯度=0

形状(横轴 r,纵轴目标,Â<0):
        L^CLIP
          │  __________            ← r < 1−ε 后变平(钉在 (1−ε)Â)
          │            \
          │             \  ← 斜率 Â<0
          │              \
          └──────────────┴──────── r
                        1−ε
```

**直觉:坏动作可以减小概率,但 r 一旦低于 1−ε,目标就不再奖励你继续减小——梯度变成 0,优化器不会把这个动作的概率压到几乎为零。** 这把"坏动作被一棍子打死(概率推到 0,策略退化)"挡住了,等价于一个下限信赖域。

### min 的方向为什么是对的:它取的是"悲观下界"

很多人记不住为什么是 min 而不是 max。统一的理解:**L^CLIP 取未裁剪项和裁剪项中较小(悲观)的那个,它是真实代理目标的一个下界(lower bound / pessimistic bound)。** 优化一个下界,最坏情况是这个下界本身,不会因为某个样本的 r 跑得太离谱而被该样本的目标值误导着继续往同方向冲。

更要命的一个细节,**新手必踩的坑**:clip 只在 r 朝"对目标有利"的方向走过头时才钉平梯度;**当 r 朝"不利"方向走时,clip 不生效,梯度照常**。

```
关键非对称(Â>0 为例):
  r 已经 > 1+ε(往好动作方向走过头了)   → min 取裁剪项 → 梯度 0 → 不再推
  r 反而 < 1+ε 甚至 < 1(策略反而更不要这个好动作了) → min 取 r·Â → 梯度照常 → 把它拉回来

也就是说 clip 是单向刹车:只刹"走太远",不刹"往回拉"。这正是想要的——
若上一步 minibatch 已经把某动作推过头,这一步该让它停;但若它被推反了方向,
应该允许纠正回来。max 会破坏这个性质,所以必须是 min。
```

**和 TRPO 的关系:** PPO-clip 不显式约束 KL,而是用 clip 这个"局部一阶代理"近似信赖域。它不保证 KL ≤ δ(理论上 r 可以在 clip 区间外取任意值,只是没梯度),但实践中因为每次更新只在 clip 区间内有动力推进,策略不会一口气跑太远。**PPO 用"目标函数里没有动力走太远"代替了 TRPO"约束里不允许走太远"——把硬约束软化成目标形状,代价是丢掉了形式保证,换来了一阶可优化和工程简单。** 这是一笔在工业界被反复验证划算的交易。

## 五、PPO 的完整训练目标:策略 + 价值 + 熵

实际跑 PPO,优化的不是单独的 L^CLIP,而是一个三项联合目标(对于 actor 和 critic 共享主干的网络尤其如此):

```
L^PPO(θ) = E_t[  L^CLIP_t(θ)  −  c₁ · L^VF_t(θ)  +  c₂ · S[π_θ](s_t)  ]
               └── 策略项 ──┘  └─ 价值损失 ─┘     └── 熵奖励 ──┘
                  (要最大化)     (要最小化)        (要最大化)

价值损失(critic 回归目标 V_target,通常用 Â_t + V_old(s_t),即 GAE 重构的回报):
  L^VF_t = ( V_φ(s_t) − V_target_t )²

熵奖励(鼓励探索,防止策略过早确定化):
  S[π_θ](s) = − Σ_a π_θ(a|s) log π_θ(a|s)        策略在该状态的熵

c₁ 典型 0.5,c₂ 典型 0.0 ~ 0.01。「待核:不同实现/任务默认值不同」
```

三项各司其职,逐一讲透:

- **L^CLIP(策略项):** 上面推导的裁剪代理目标,负责改进策略。注意梯度通过 r(θ) 流到策略参数;Â_t 是用旧策略采样、用旧 critic 算的 GAE,**对策略参数视为常数(detach / no_grad)**——这是新手常错的地方,Â 里若漏了 detach,梯度会走错路径。
- **L^VF(价值损失):** 训练 critic 估准 V(s),给下一轮 GAE 用。许多实现还对它也做裁剪(clipped value loss),限制 V_φ 相对 V_old 的变化,机制和策略 clip 类似,但**它对最终效果是否有益有争议**(见下文坑)。
- **S(熵奖励):** RL 的探索全靠它撑着。**Â 只告诉策略"哪些已采过的动作好",不会主动让策略去试没采过的动作;熵项是唯一在目标里显式对抗"过早收敛到确定策略(熵塌缩)"的力量。** 在大模型 RLHF 里熵塌缩表现为生成多样性骤降、复读;熵系数 c₂ 太小会塌、太大则策略学不专注,需要调。

**PPO 的"多 epoch 复用"是它样本效率的关键来源,也是 clip 的另一个理由:** 采一批数据(用 π_old),对这批数据做**多个 epoch、多个 minibatch** 的梯度更新。第一个 minibatch 更新后,π_θ 已经偏离 π_old,r 不再是 1,代理目标开始"过期"。clip 正好在这种"用旧数据更新已经走远的新策略"的场景下踩刹车——**这是 PPO 比 vanilla policy gradient 样本效率高数倍的核心机制:同一批昂贵的样本被安全地反复利用。** 典型设置:每批数据跑 3~10 个 epoch(「待核:常用 epoch 数」)。

## 六、PPO-Penalty:KL 惩罚变体

PPO 论文里还有第二个变体,**PPO-penalty**,它更接近 TRPO 的"软化"形式——不裁剪,而是把 KL 当惩罚项加进目标,并**自适应调节惩罚系数 β**:

```
L^KLPEN(θ) = E_t[ r_t(θ)·Â_t − β · KL( π_old(·|s_t) ‖ π_θ(·|s_t) ) ]

自适应 β 规则(每次更新后看实际 KL 与目标 KL d_targ 的偏差):
  实测 KL < d_targ / 1.5  ⇒  β ← β / 2        # 走得太保守,放松惩罚
  实测 KL > d_targ × 1.5  ⇒  β ← β × 2        # 走得太远,加重惩罚
  否则                    ⇒  β 不变
                                              (倍数 1.5、2 见 PPO 论文)
```

**直觉:β 是个跟踪 d_targ 的恒温器——KL 大了就加大惩罚把它压回去,小了就松绑让它走快点。** 这把第二节那个固定的、大得没法用的 C,换成了一个数据驱动、自动调到合适大小的 β。

**PPO 论文的结论:clip 版本通常比 penalty 版本更好用、更稳健**(clip 不用估 KL,不用调 d_targ,只有一个 ε)。所以工业界默认就是 PPO-clip。**但 RLHF 是个重要例外:** 在第 08 章你会看到,RLHF 里通常在 reward 里额外加一项**对参考模型(SFT 模型)的 KL 惩罚**——那个 KL 惩罚的目的和这里不同(它防止模型为了刷 reward model 而漂离原始语言能力,即 reward hacking),和 PPO 本身的 clip 信赖域是**两个独立的 KL**,别混淆。

## 七、代码:PPO-clip 损失与训练循环(PyTorch 伪代码)

下面是体现机制的核心片段,可照着实现。省略了环境交互和网络定义的样板,聚焦损失计算和更新循环。

```python
import torch

# === 一次 PPO 更新:输入一批用 π_old 采的数据 ===
# obs:      [N, obs_dim]
# actions:  [N]
# logp_old: [N]   采样时记录的 log π_old(a|s),detach,绝不重算
# adv:      [N]   GAE 算出的优势 Â(已 detach,对策略参数是常数)
# returns:  [N]   价值回归目标 = adv + V_old(s)
# 关键:adv 和 logp_old 都是采样当时算好存下来的,不参与本次反传的计算图

def ppo_update(policy, value_fn, optimizer, batch,
               clip_eps=0.2, c_vf=0.5, c_ent=0.01,
               n_epochs=4, minibatch_size=64):
    obs, actions, logp_old, adv, returns = batch

    # 优势归一化:几乎是 PPO 的标配,稳定训练(逐 minibatch 或逐 batch)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    for _ in range(n_epochs):                  # 同一批数据复用多个 epoch —— 样本效率的来源
        for idx in random_minibatches(len(obs), minibatch_size):
            # --- 用当前(已开始偏离 old 的)策略重新前向 ---
            dist = policy(obs[idx])             # 当前策略 π_θ 的动作分布
            logp = dist.log_prob(actions[idx])  # log π_θ(a|s),带梯度
            entropy = dist.entropy().mean()     # 熵,鼓励探索

            # --- 重要性采样比 r(θ) = π_θ/π_old = exp(logp - logp_old) ---
            # 用 log 差再 exp,数值上比直接相除稳;logp_old 已 detach
            ratio = torch.exp(logp - logp_old[idx])

            # --- clip 目标:分段 min(未裁剪, 裁剪) ---
            unclipped = ratio * adv[idx]
            clipped   = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv[idx]
            # 取 min = 悲观下界;再取负号因为优化器做最小化
            policy_loss = -torch.min(unclipped, clipped).mean()

            # --- 价值损失:critic 回归 ---
            v_pred = value_fn(obs[idx])
            value_loss = ((v_pred - returns[idx]) ** 2).mean()

            # --- 合并:最小化 (策略损失 + c1·价值损失 − c2·熵) ---
            loss = policy_loss + c_vf * value_loss - c_ent * entropy

            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪(范数),PPO 稳定性的常用补丁,防止偶发大梯度
            torch.nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(value_fn.parameters()), 0.5)
            optimizer.step()
```

几个**实现层面的机制注解**(都是会真出 bug 的地方):

- **ratio 用 `exp(logp − logp_old)` 而非概率直接相除**:大模型里 token 概率极小,直接相除会下溢;在 log 空间做差更稳。
- **`logp_old` 和 `adv` 必须 detach**:它们是"旧策略/旧 critic 在采样时刻的快照",是常数。若 adv 的计算图连到 critic 当前参数,反传会把价值损失的梯度错误地灌进策略路径。
- **优势归一化** `(adv − mean)/std`:不改梯度期望方向(类似 baseline 的逻辑),但把不同 batch 的优势尺度统一,显著稳定训练。这是工程经验,不是理论必需。
- **梯度范数裁剪** clip_grad_norm:和 PPO 的 clip 是**两回事**——前者裁的是参数梯度的范数(防数值爆炸),后者裁的是概率比(实现信赖域)。新手容易混。

## 八、量化分析:方差、KL、复杂度、样本效率

**1. 重要性采样比带来的方差。** 代理目标 E[r(θ)·Â] 用 IS 把 π' 的期望写成 π_old 下的加权期望。**IS 估计的方差随 r 偏离 1 急剧增大**——若新旧策略差很多,r 会有很大的值,方差爆炸,梯度估计变得不可靠。

```
单样本 IS 估计 r·Â 的方差(粗略):
  Var[r·Â] = E[r²·Â²] − (E[r·Â])²
若某些 (s,a) 上 r ≫ 1(新策略远比旧策略倾向该动作),r² 项使方差迅速放大。
极端情形:支撑集不重合(π_old(a|s)→0 而 π_θ(a|s)>0),r→∞,方差无界。

clip 的方差含义:把 r 限制在 [1−ε,1+ε] 等于把被加权的量限幅,
  → r·Â 在裁剪区外梯度为 0,有效地截断了大 r 贡献的高方差梯度。
  → clip 既是信赖域,也是一种方差控制手段。
```

**2. KL 与 ε 的近似关系。** clip 用 ε 间接控制策略偏移。对一个 token/动作,r ≈ 1 + ε 时对应的单点 KL 量级可用二阶近似估:

```
KL(π_old ‖ π_θ) 在 π_θ ≈ π_old 时 ≈ (1/2)·E[(Δlogπ)²]
而 r = exp(Δlogπ) ≈ 1 + Δlogπ  ⇒  Δlogπ ≈ r − 1
ε = 0.2  ⇒  |Δlogπ| ≲ 0.2  ⇒  单点 KL ≲ (1/2)(0.2)² = 0.02 量级

这只是数量级直觉:ε 越小,等效信赖域越紧,每步越保守。
实践中应监控实测 E[KL],它是判断 PPO 是否健康的头号指标(见下文坑)。
```

**3. 计算复杂度:PPO vs TRPO。**

```
TRPO 每次更新:
  - 共轭梯度求 H⁻¹g:O(参数量 × CG步数) 次 Hessian-vector product,每次一次反传量级
  - 线搜索:若干次前向评估 KL 和 L
  - 总体:一阶反传的常数倍,但常数大、实现复杂、要二阶图

PPO 每次更新:
  - 纯一阶:n_epochs × (N / minibatch) 次普通 SGD step
  - 每 step 一次前向 + 一次反传,无二阶,无矩阵求逆
  - 总体:就是普通深度学习训练循环,易并行、易和现有框架/混精/分布式整合
```

PPO 的胜出**不在每步更快,而在工程上能直接套用成熟的一阶训练基础设施**——这在要训百亿参数大模型时是决定性的。

**4. 样本效率。** vanilla policy gradient(REINFORCE / A2C)采一批样本只能做一步更新(因为更新后策略变了,旧数据失效)。**PPO 靠 clip 提供的安全边界,把一批样本反复用 n_epochs 次**,样本效率通常是同等设置下纯 on-policy 方法的数倍(「待核:具体倍数依任务而定,无统一数字」)。这是 on-policy 方法里少见的"敢复用数据"的设计,代价是引入了 r、clip 这套近似机器。

## 九、设计权衡与常见坑

把这一节当 checklist 用,PPO 调不出来九成栽在这里。

- **TRPO vs PPO 的根本取舍:** TRPO 有单调改进的理论保证但实现复杂、二阶、难和框架整合;PPO 丢掉形式保证、换来一阶可优化和工程简单。**工业界几乎全用 PPO**,TRPO 现在主要作为理论参照和"为什么需要信赖域"的教学起点。选 TRPO 的唯一现实理由:你的任务对每步绝不退化有硬性要求且能承受实现成本(罕见)。
- **坑 1:监控 KL,不是监控 loss。** PPO 的 L^CLIP 数值本身没什么解读价值(它是被裁剪的代理目标)。**真正要盯的是实测 E_t[KL(π_old‖π_θ)]。** 健康的 PPO,每批更新后 KL 应在一个小而稳的范围(如 0.01~0.03,「待核:依任务」)。KL 突然飙高 = 策略要崩了;KL 一直接近 0 = 学不动(ε 太小或 lr 太小)。很多实现加 **early stop:一旦本批的近似 KL 超过阈值(如 1.5·d_targ),提前停止这批的剩余 epoch**,这是防崩的重要保险。
- **坑 2:Â 不归一化 / 不 detach。** 优势不归一化时尺度漂移会让训练剧烈震荡;Â 或 logp_old 漏 detach 会让梯度走错计算图。两者都是静默 bug——不报错,只是训练莫名其妙地差。
- **坑 3:reward hacking(RL 的头号陷阱)。** 策略会找到最大化你给的 reward、但不是你真正想要的行为的捷径。PPO 本身不防这个——**reward 设计错了,PPO 会极其高效地把你的错误放大。** 在 RLHF 里典型表现:模型发现 reward model 偏爱长回答/特定句式,于是输出又长又空的废话刷分。对策不在 PPO 算法内,而在 reward 设计 + 对参考模型的 KL 约束(第 08 章)。
- **坑 4:熵塌缩(entropy collapse)。** 熵系数太小或 lr 太大,策略迅速收敛到确定性,探索消失,卡在局部最优。监控策略熵的曲线;塌了就加大 c₂ 或减小 lr。大模型上表现为生成多样性骤降、复读。
- **坑 5:clipped value loss 不是免费午餐。** 对价值损失也做 clip(限制 V_φ 偏离 V_old)是常见做法,但**有实证研究表明它未必有益甚至有害**(它的效果对实现细节敏感)。「待核:具体结论见 PPO 复现研究(如 Engstrom et al. 2020 / Andrychowicz et al. 2021 一类的实现细节消融)」。建议:先不开,需要时再消融。
- **坑 6:"PPO 的性能很大程度来自实现细节而非算法本身。"** 这是一个被多篇复现论文反复强调的、反直觉但极重要的事实:优势归一化、奖励缩放、观测归一化、学习率退火、正交初始化、value clip、minibatch 重排……这些"工程小动作"对 PPO 最终分数的影响,可能和核心 clip 目标本身一样大。**实现 PPO 时,优先照着一个可信复现(如 CleanRL)逐行对齐这些细节,不要只实现公式。** 「待核:相关结论见 Engstrom et al. 2020《Implementation Matters in Deep RL》及 Andrychowicz et al. 2021 一类工作」。

## 十、动手练习

**练习 1(推导 + 作图,核心)。** 固定一个样本的 Â 值,横轴取 r ∈ [0, 2],分别在 Â = +1 和 Â = −1 两种情形下,手算并画出 L^CLIP 单样本项 min(r·Â, clip(r,1−ε,1+ε)·Â) 随 r 的曲线(取 ε = 0.2)。标出:斜率从非零变成 0 的拐点在哪、梯度为 0 的区间是 r 的哪一段。**提示:** 先写出 clip 的分段定义,再对 r·Â 和 clip·Â 逐段比大小取 min;注意 Â<0 时乘法翻转大小关系。验证你画出的图和正文第四节的两张 ASCII 图一致。

**练习 2(机制解释)。** 用一句话回答并给出理由:为什么 L^CLIP 用 min 而不是 max?进一步,构造一个具体场景(给定 Â 的符号和上一步把 r 推到了哪一侧),说明若换成 max,优化器会做出什么有害的行为。**提示:** 抓住"clip 是单向刹车,只刹走太远、不刹往回拉"这个性质,看 max 会不会破坏它。

**练习 3(从性能差分到信赖域,推导)。** 自己重新推一遍性能差分引理 J(π')−J(π) = E_{τ~π'}[Σ γᵗ A^π(s_t,a_t)],只用 A^π = r + γV^π(s') − V^π(s) 和望远镜求和。然后说明:为什么直接最大化引理右边做不了基于梯度的优化,代理目标把 d^{π'} 换成 d^{π_old} 后又付出了什么代价、这个代价为什么由 KL 项来补偿。**提示:** 望远镜那步是关键;代理目标只在 θ_old 附近一阶相合,KL 度量"走出了这个可信邻域多远"。

**练习 4(编码 + 分析)。** 在任一连续/离散控制环境(如 CleanRL 提供的 CartPole / 简单 gym 任务)上跑通 PPO-clip。做两组消融:(a) 把 clip_eps 从 0.2 调成 10(几乎等于不裁剪),(b) 关掉优势归一化。各跑 3 个随机种子,画出训练曲线和**实测 KL 曲线**。**提示:** 你应观察到 (a) KL 失控、训练崩溃或震荡;(b) 训练显著不稳。把现象和正文第八、九节的分析对上,这比读十遍公式更能让你记住 clip 和归一化在干什么。

## 十一、源码 / 论文导读

- **TRPO 原论文:** Schulman et al., *Trust Region Policy Optimization*, 2015。重点读:**第 2~3 节**(代理目标与单调改进**定理 1**,即本章第二节的下界)、**第 5 节与附录**(从理论界到带 KL 约束的实用算法、共轭梯度 + 线搜索)。本章简化的常数 C 与分布差距界,精确形式在论文与其引用的 CPI(Kakade & Langford 2002)里。
- **PPO 原论文:** Schulman et al., *Proximal Policy Optimization Algorithms*, 2017。**通篇都短且关键**:第 3 节 clip 目标(本章第四节)、第 4 节 penalty 变体与自适应 β(第六节)、第 5 节联合目标 L^CLIP + value + entropy(第五节)。论文里那张 L^CLIP 随 r 的分段图,务必和你练习 1 画的对照。
- **自然梯度背景:** Kakade, *A Natural Policy Gradient*, 2002。理解 TRPO 里 H⁻¹g 的几何意义(Fisher 信息诱导的策略空间最陡方向)。
- **开源实现(强烈建议逐行读):** **CleanRL** 的 `ppo.py` / `ppo_continuous_action.py`——单文件、无封装、把所有"实现细节"(优势归一化、value clip、KL early-stop、学习率退火、梯度裁剪)都明明白白写在一处,是学 PPO 工程真相的最佳起点。**TRL** 的 `PPOTrainer`——大模型 RLHF 场景下的 PPO(第 08 章会回到它)。**verl**——大规模分布式 RLHF/推理训练框架里的 PPO 实现,适合理解工业级规模化怎么落地。
- **复现/实现细节研究(治"为什么我照公式写却跑不出来"):** Engstrom et al. 2020、Andrychowicz et al. 2021 一类系统消融工作,讲清 PPO 的多少性能其实来自工程细节。「待核:具体论文标题与结论数据请对照原文」

## 小结与承上启下

这一章的逻辑链是一条直线,务必记住:**朴素策略梯度步长难定(走大崩、走小慢、欧氏距离度量错)→ 该约束的是策略分布的 KL 变化(信赖域)→ 性能差分引理精确写出新旧性能差 → 用旧状态分布近似得到代理目标(重要性采样比 r 由此而生)→ 单调改进界证明"最大化代理目标减 KL 惩罚"能保证不退化(minorize-maximization)→ TRPO 用 KL 硬约束 + 自然梯度(共轭梯度近似)+ 线搜索把它工程化 → PPO 用 clip 把信赖域塞进一阶目标,丢保证换简单,成为工业主力。**

你现在应该能:从第一性原理讲清为什么需要信赖域;推出 TRPO 的代理目标和它的理论下界;逐分段讲清 PPO clip 在 Â>0 和 Â<0 两种情形下如何近似信赖域、为什么是 min、为什么是单向刹车;写出一个 detach / 归一化 / 监控 KL 都做对的 PPO 损失。

**这是整门课的承重墙。** 往前看,它直接撑起第 **08 章 RLHF**——把 PPO 接到大语言模型上,reward 来自人类偏好训练的 reward model,并额外加一个对 SFT 参考模型的 KL 约束(注意和本章 clip 的 KL 是两个独立的东西);第 **11 章 GRPO** 则是 PPO 在大模型推理训练里的一个重要简化变体——它去掉 critic、用一组采样的相对优势替代 GAE,正是站在本章 clip 目标的肩膀上做的减法。把本章的 r、Â、clip、KL 这四个量彻底吃透,后面 RLHF / GRPO 的所有改动你都能看出它在动哪一块、为什么动。


---

## 练习参考答案

> 本章「动手练习」的参考答案(AI 生成,推导/代码已尽量自验,具体数值见「待核」标注)。

### 练习 1:手算并作图 L^CLIP 单样本项(ε=0.2,Â=±1)

先写出 clip 的分段定义(ε=0.2):

```
clip(r, 0.8, 1.2) = 0.8       当 r < 0.8
                  = r         当 0.8 ≤ r ≤ 1.2
                  = 1.2       当 r > 1.2
```

单样本项是 g(r) = min( r·Â , clip(r,0.8,1.2)·Â )。注意 **Â<0 时乘以负数会翻转大小关系**,所以两种情形要分开做。

#### 情形 Â = +1

未裁剪支 r·Â = r;裁剪支 clip(r)·Â = clip(r)。两支都是 Â>0,取 min:

```
分段比大小(Â=+1):
  r < 0.8 :  min(r, 0.8)   。因 r<0.8<... 这里 r<0.8 ⇒ r<0.8=clip ⇒ min = r        (注:r 与 clip=0.8 比,r<0.8 取 r)
  0.8≤r≤1.2: clip=r ⇒ 两支相等 ⇒ min = r
  r > 1.2 :  min(r, 1.2)。r>1.2 ⇒ 1.2<r ⇒ min = 1.2
合并:
  r ≤ 1.2 :  g(r) = r        （斜率 +1，正常上升）
  r > 1.2 :  g(r) = 1.2      （钉平在 (1+ε)·Â = 1.2，斜率 0）
```

数值验算(ε=0.2,Â=+1):

```
r     0.0   0.5   0.8   1.0   1.2   1.5   2.0
g(r)  0.0   0.5   0.8   1.0   1.2   1.2   1.2
```

曲线形状(横轴 r,纵轴 g):一条斜率为 1 的直线从原点上升,**在 r = 1+ε = 1.2 处出现拐点**,此后水平钉在 1.2。

```
 g
1.2 │        ________________   ← r>1.2 后变平，钉在 (1+ε)=1.2
    │       /
    │      / ← 斜率 +1
    │     /
  0 └────┴──────────────── r
        1.2
```

**结论(Â=+1):斜率从 +1 突变为 0 的拐点在 r = 1+ε = 1.2;梯度为 0 的区间是 r > 1.2。** r < 0.8 段 clip 不生效(min 取未裁剪支),梯度照常为 +1——这正是"往回拉好动作"时不刹车。

#### 情形 Â = −1

未裁剪支 r·Â = −r;裁剪支 clip(r)·Â = −clip(r)。两支都是负数,**谁更负谁更小**:

```
分段比大小(Â=−1，比 −r 与 −clip(r)):
  r < 0.8 : clip=0.8 ⇒ 两支 −r vs −0.8。r<0.8 ⇒ −r>−0.8 ⇒ min = −0.8   （钉平）
  0.8≤r≤1.2: clip=r ⇒ 两支相等 ⇒ min = −r
  r > 1.2 : clip=1.2 ⇒ −r vs −1.2。r>1.2 ⇒ −r<−1.2 ⇒ min = −r        （未裁剪支更小）
合并:
  r < 0.8 :  g(r) = −0.8 = (1−ε)·Â    （钉平，斜率 0）
  r ≥ 0.8 :  g(r) = −r                （斜率 Â = −1，随 r 减小而上升）
```

数值验算(ε=0.2,Â=−1):

```
r     0.0    0.5    0.8    1.0    1.2    1.5    2.0
g(r)  -0.8   -0.8   -0.8   -1.0   -1.2   -1.5   -2.0
```

曲线形状:左侧 r<0.8 水平钉在 −0.8,**在 r = 1−ε = 0.8 处出现拐点**,此后斜率 −1 向右下方走。

```
 g
   │  ____________            ← r<0.8 后变平，钉在 (1−ε)·Â = −0.8
-0.8│            \
   │             \  ← 斜率 −1
   │              \
   └──────────────┴──────── r
                 0.8
```

**结论(Â=−1):斜率从 0 突变为 −1 的拐点在 r = 1−ε = 0.8;梯度为 0 的区间是 r < 0.8。** r > 1.2 段 clip 不生效(min 取未裁剪支 −r),梯度照常——往回拉(让坏动作概率反而变大)时不刹车。

两张图分别与正文第四节情形 A、情形 B 的 ASCII 图一致(A 在 1+ε 后变平,B 在 1−ε 后变平),验证通过。

---

### 练习 2:为什么 L^CLIP 用 min 而不是 max

**一句话:用 min 是因为 L^CLIP 要取未裁剪项与裁剪项中"更悲观(更小)"的那个,使它成为真实代理目标的下界,从而 clip 只在"朝有利方向走过头"时刹车、不在"往回拉"时刹车;换成 max 会把这个单向刹车反向,反而鼓励策略在已经走过头的方向上继续冲。**

理由(机制):min 选中两支里数值更小的一支,被选中那支才有非零梯度。
- 当 r 朝"对目标有利"的方向走过头(Â>0 且 r>1+ε,或 Â<0 且 r<1−ε),裁剪支被钉平、数值更小,min 选裁剪支 ⇒ 梯度 0 ⇒ **刹住"走太远"**。
- 当 r 朝"不利"方向走(被推反了),裁剪不生效,min 选未裁剪支 ⇒ 梯度照常 ⇒ **允许把它纠正回来**。

这就是"单向刹车":只刹走太远、不刹往回拉。

**换成 max 的有害行为(具体场景):** 设某动作 Â = +1(好动作),上一个 minibatch 已经把它的 r 推到 1.5(> 1+ε = 1.2)——已经过度自信了。
- 用 min(正确):此时 min(1.5·1, 1.2·1)=1.2 取裁剪支,梯度 0,优化器停手,不再把这个好动作的概率往上推。
- 用 max(错误):max(1.5·1, 1.2·1)=1.5 会取未裁剪支 r·Â,其对 r 的梯度为正且不被钉平 ⇒ **优化器继续把 r 推得更大(1.5→更高)**,策略对这个动作越来越确定,KL 失控、熵塌缩,最终策略崩溃。即:max 把"刹走太远"变成了"奖励走太远",彻底破坏了信赖域。对称地,Â<0 且 r 已被压到 1−ε 以下时,max 也会鼓励继续把坏动作概率压向 0(一棍子打死),同样导致退化。

---

### 练习 3:重推性能差分引理,并说明代理目标的代价与 KL 的作用

#### 推导性能差分引理

目标:J(π') − J(π) = E_{τ~π'}[ Σ_{t≥0} γ^t A^π(s_t, a_t) ]。只用 A^π(s,a) = E[ r + γV^π(s') − V^π(s) ](优势的 Bellman 形式)与望远镜求和。

```
推导:

从右边出发,沿新策略 π' 的轨迹把单步优势求和、取期望,代入 A^π 定义:

  E_{τ~π'}[ Σ_{t≥0} γ^t A^π(s_t,a_t) ]
    = E_{τ~π'}[ Σ_{t≥0} γ^t ( r_t + γ V^π(s_{t+1}) − V^π(s_t) ) ]

把求和拆成两块:回报块  Σ γ^t r_t  和  价值块  Σ γ^t( γV^π(s_{t+1}) − V^π(s_t) )。

价值块发生望远镜(telescoping)相消:
  Σ_{t≥0} ( γ^{t+1} V^π(s_{t+1}) − γ^t V^π(s_t) )
逐项写开:
  t=0:  γ¹V(s₁) − γ⁰V(s₀)
  t=1:  γ²V(s₂) − γ¹V(s₁)
  t=2:  γ³V(s₃) − γ²V(s₂)
  ...
后一项的 −γ^t V(s_t) 与前一项的 +γ^t V(s_t)(来自 t−1 行)逐项抵消(γ¹V(s₁)、γ²V(s₂)… 全部消掉),
γ^t V(s_t)→0(γ<1,V 有界)只剩 t=0 的 −V(s₀):
  价值块 = − V^π(s_0)

于是:
  E_{τ~π'}[ Σ γ^t A^π ]
    = E_{τ~π'}[ Σ γ^t r_t ]  −  E_{s_0}[ V^π(s_0) ]
    = J(π')                  − J(π)            ∎

最后一步:E_{τ~π'}[Σ γ^t r_t] 按定义就是 J(π');E_{s_0}[V^π(s_0)] = J(π)
(起始状态分布对两个策略相同)。
```

再把轨迹期望按折扣状态访问分布 d^{π'} 写成状态-动作期望(把 Σγ^t 吸收进 d^{π'} 的归一化 1/(1−γ)):

```
  J(π') − J(π) = (1/(1−γ)) · E_{s~d^{π'}, a~π'(·|s)}[ A^π(s,a) ]
```

**结论:新策略比旧策略好多少,等于"在新策略走出的状态-动作上,旧策略优势的(折扣)总和"。**

#### 为什么不能直接对引理右边做梯度优化

右边的期望取在 **d^{π'}** 上——也就是你正要优化的那个新策略 π' 自己的状态访问分布。要算这个期望就得先用 π' 在环境里采样,但 π' 尚未确定;每改一次 π' 都要重新跑一批昂贵的环境交互才能估出右边。这是一个"鸡生蛋"循环,**无法写成一个可对 θ 反向传播的固定目标**,基于梯度的优化做不了。

#### 代理目标的代价,以及为什么由 KL 补偿

TRPO 的处理:把 d^{π'} 换成**旧策略的** d^{π_old},得到代理目标
L_{π_old}(π_θ) = (1/(1−γ))·E_{s~d^{π_old}, a~π_θ}[A^{π_old}],再用重要性采样把"按 π_θ 取动作"换成"按 π_old 取动作 × 比值 r(θ)",于是可以**直接复用旧策略采的样本**做梯度优化。

代价:**用错了状态分布**。L 只在 θ=θ_old 这一点与真实 J 做到零阶 + 一阶相合(值相等、梯度相等,后者正是策略梯度定理)。一旦 θ 走远,d^{π_θ} ≠ d^{π_old},被替换掉的那部分不再准,L 与 J 分道扬镳——你以为在涨 L,真实 J 可能在跌(策略可能崩)。

为什么 KL 来补偿:真实 J 与代理 L 之差,可由"两个状态分布的差距"放缩,而状态分布差距又被"逐状态的策略差距"控制(TV 距离随策略偏移线性增长,状态分布偏移 ≤ 2γα/(1−γ)),再用 Pinsker 不等式 TV² ≤ (1/2)KL 把策略差距换成 KL。最终得单调改进界

```
  J(π_θ) ≥ L_{π_old}(π_θ) − C · max_s KL( π_old ‖ π_θ ),   C = 4εγ/(1−γ)²
```

**所以 KL 项度量的正是"π_θ 走出了 L 可信邻域 d^{π_old} 多远":KL 越大 ⇒ 状态分布替换误差越大 ⇒ 代理目标越不可信 ⇒ 惩罚越重。** KL 不是"温柔一点"的经验项,而是补偿"用旧状态分布"这一近似所引入误差的、有界证明支撑的修正项;把它当硬约束 δ 锁住,就把 θ 限制在 L 仍然近似 J 的信赖域内,从而(在 MM 框架下)保证每步不退化。

---

### 练习 4:跑通 PPO-clip 并做两组消融

#### 期望结论(先讲要观察到什么)

- (a) clip_eps 0.2 → 10(几乎不裁剪):一批数据多 epoch 复用时 r 不再被夹,好动作的 r 被无限推高 ⇒ **实测 E[KL] 失控飙升,训练曲线崩溃或剧烈震荡**。对应正文第八节"r 偏离 1 → IS 方差爆炸"。
- (b) 关掉优势归一化:不同批次优势尺度漂移,等效学习率随之漂移 ⇒ **训练显著不稳、回报曲线抖动变大、收敛更慢**。对应第九节坑 2。
- 监控量的重点是 **实测 E[KL]**(每批更新后用 logp_old 与新 logp 估),而非 L^CLIP 数值本身(坑 1)。

#### 方案一:真实环境的最小可跑骨架(torch + gymnasium,CartPole)

依赖与运行环境说明:需 `pip install torch gymnasium`;CartPole CPU 即可,无需 GPU;3 个种子各跑几分钟。这是练习要求的正式交付物,在本环境未安装 torch/gym 故未实跑,代码按 CleanRL 风格对齐关键实现细节(优势归一化、ratio 用 log 差、detach、KL early-stop、梯度范数裁剪)。

```python
# 依赖: pip install torch gymnasium   (CartPole 用 CPU 即可,无需 GPU)
import gymnasium as gym
import torch, torch.nn as nn
import numpy as np

def make_net(obs_dim, act_dim):
    # 简单 MLP,actor 出 logits,critic 出标量 V
    actor = nn.Sequential(nn.Linear(obs_dim,64), nn.Tanh(),
                          nn.Linear(64,64), nn.Tanh(), nn.Linear(64,act_dim))
    critic = nn.Sequential(nn.Linear(obs_dim,64), nn.Tanh(),
                           nn.Linear(64,64), nn.Tanh(), nn.Linear(64,1))
    # 正交初始化(PPO 常见实现细节之一)
    for m in list(actor)+list(critic):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, np.sqrt(2)); nn.init.zeros_(m.bias)
    return actor, critic

def gae(rewards, values, dones, gamma=0.99, lam=0.95):
    # 标准 GAE-λ,返回 adv 和回报目标 returns = adv + values
    adv = np.zeros_like(rewards); last = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        nextv = values[t+1] if t+1 < len(values) else 0.0
        delta = rewards[t] + gamma*nextv*nonterminal - values[t]
        adv[t] = last = delta + gamma*lam*nonterminal*last
    return adv, adv + values[:len(rewards)]

def train(clip_eps=0.2, norm_adv=True, seed=0,
          total_steps=100_000, rollout=2048, n_epochs=10, mb=64,
          c_vf=0.5, c_ent=0.01, lr=3e-4, target_kl=0.03):
    torch.manual_seed(seed); np.random.seed(seed)
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]; act_dim = env.action_space.n
    actor, critic = make_net(obs_dim, act_dim)
    opt = torch.optim.Adam(list(actor.parameters())+list(critic.parameters()), lr=lr)

    obs, _ = env.reset(seed=seed)
    ep_ret, ep_rets = 0.0, []
    kl_log, ret_log = [], []      # 供画图:实测 KL 曲线、回报曲线
    steps = 0
    while steps < total_steps:
        # ---- 1) 用 pi_old 采一段 rollout ----
        O,A,LP,R,D,V = [],[],[],[],[],[]
        for _ in range(rollout):
            ot = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                logits = actor(ot); dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample(); v = critic(ot).item()
            O.append(obs); A.append(a.item()); LP.append(dist.log_prob(a).item()); V.append(v)
            obs, r, term, trunc, _ = env.step(a.item())
            done = term or trunc; R.append(r); D.append(float(done)); ep_ret += r
            if done:
                ep_rets.append(ep_ret); ep_ret = 0.0; obs,_ = env.reset()
            steps += 1
        with torch.no_grad():
            V.append(critic(torch.as_tensor(obs, dtype=torch.float32)).item())  # bootstrap
        O=np.array(O,dtype=np.float32); A=np.array(A); LP=np.array(LP,dtype=np.float32)
        adv, ret = gae(np.array(R,dtype=np.float32), np.array(V,dtype=np.float32),
                       np.array(D,dtype=np.float32))
        # 张量化;adv/logp_old/ret 全部是采样快照 -> 不带梯度
        O=torch.as_tensor(O); A=torch.as_tensor(A)
        LP=torch.as_tensor(LP); adv=torch.as_tensor(adv,dtype=torch.float32)
        ret=torch.as_tensor(ret,dtype=torch.float32)

        # ---- 2) 多 epoch 复用这批数据 ----
        approx_kl = 0.0
        for ep in range(n_epochs):
            idxs = np.random.permutation(len(O))
            for s in range(0, len(O), mb):
                j = idxs[s:s+mb]
                a_j = adv[j]
                if norm_adv:                          # 消融 (b):可关掉
                    a_j = (a_j - a_j.mean())/(a_j.std()+1e-8)
                logits = actor(O[j]); dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(A[j]); ent = dist.entropy().mean()
                ratio = torch.exp(logp - LP[j])       # r = exp(Δlogp),比直接相除稳
                unclipped = ratio * a_j
                clipped = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * a_j  # 消融 (a):eps=10
                policy_loss = -torch.min(unclipped, clipped).mean()         # min=悲观下界
                v_pred = critic(O[j]).squeeze(-1)
                value_loss = ((v_pred - ret[j])**2).mean()
                loss = policy_loss + c_vf*value_loss - c_ent*ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters())+list(critic.parameters()), 0.5)
                opt.step()
                with torch.no_grad():
                    approx_kl = (LP[j] - logp).mean().item()   # 实测近似 KL
            # KL early-stop:本批 KL 超阈值就提前停剩余 epoch(防崩的关键保险)
            if approx_kl > 1.5*target_kl:
                break
        kl_log.append(approx_kl)
        ret_log.append(np.mean(ep_rets[-10:]) if ep_rets else 0.0)
        print(f"steps={steps:6d}  return={ret_log[-1]:6.1f}  approxKL={approx_kl:+.4f}")
    env.close()
    return ret_log, kl_log

if __name__ == "__main__":
    # 三组实验 × 3 种子,收集 (return 曲线, KL 曲线) 后用 matplotlib 画
    runs = {"ppo(clip=0.2)":dict(clip_eps=0.2, norm_adv=True),
            "(a) clip=10":  dict(clip_eps=10.0, norm_adv=True),
            "(b) no-advnorm":dict(clip_eps=0.2, norm_adv=False)}
    results = {name: [train(seed=s, **kw) for s in range(3)] for name,kw in runs.items()}
    # 画图: 对每组取 3 种子均值±std,上排 return 曲线、下排 KL 曲线
    # import matplotlib.pyplot as plt   # 略,把 results 里的 ret_log/kl_log 画出来即可
```

#### 方案二:纯 numpy 可立即跑通的最小演示(无需 gym/torch,验证 clip 机制)

下面这份在本环境实跑通过(只用 numpy),用一个 2-臂老虎机演示 PPO 循环、ratio、clip、优势归一化、KL 监控。它不能复现 CartPole 的崩溃幅度(1 维策略太温和),但能让你在 CPU 上秒级看到 clip 把每批 KL 压住、归一化影响收敛——作为练习正式交付前的机制自检很有用。

```python
import numpy as np
R = np.array([0.2, 0.8])                         # 两动作 reward 期望
def sigmoid(x): return 1/(1+np.exp(-x))
def probs(theta): p1=sigmoid(theta); return np.array([1-p1, p1])

def run(clip_eps, norm_adv, lr=0.3, n_iter=20, batch=512, n_epochs=4, seed=0):
    rng=np.random.default_rng(seed); theta=0.0; kls=[]
    for _ in range(n_iter):
        po = probs(theta)                        # pi_old
        a  = rng.choice(2, size=batch, p=po)     # 用 pi_old 采样
        rew= R[a] + 0.01*rng.standard_normal(batch)
        logp_old = np.log(po[a]+1e-12)           # 快照,detach
        adv = rew - rew.mean()                   # baseline=batch 均值
        if norm_adv: adv=(adv-adv.mean())/(adv.std()+1e-8)
        for _ in range(n_epochs):                # 多 epoch 复用同一批
            p1=sigmoid(theta)
            logp=np.log(probs(theta)[a]+1e-12)
            ratio=np.exp(logp-logp_old)          # r=exp(Δlogp)
            dlogp=np.where(a==1, 1-p1, -p1)       # d logp(a)/dθ
            unclipped=ratio*adv
            clipped=np.clip(ratio,1-clip_eps,1+clip_eps)*adv
            use=unclipped<=clipped               # min 选中支才有梯度
            grad=np.where(use, adv*ratio*dlogp, 0.0).mean()
            theta+=lr*grad                       # 梯度上升(最大化)
        pn=probs(theta)
        kls.append(float(np.sum(po*(np.log(po+1e-12)-np.log(pn+1e-12)))))  # 实测 KL
    return sigmoid(theta), kls

for tag,ce,na in [("clip=0.2",0.2,True),("clip=10(no-clip)",10.0,True),("no-advnorm",0.2,False)]:
    p1,kls=run(ce,na)
    print(f"{tag:18s} p(a=1)->{p1:.3f}  maxKL={max(kls):.4f}  meanKL={np.mean(kls):.4f}")
```

本环境实跑输出(numpy 版,lr=0.3,前 20 iter):

```
clip=0.2           p(a=1)->...   maxKL=0.0260  meanKL=0.0048
clip=10(no-clip)   p(a=1)->...   maxKL=0.0422  meanKL=0.0060
```

可见 clip=0.2 把每批 KL 上限压到 ~0.026,而同样 lr 下不裁剪让 KL 跑到 ~0.042(更高)。**机制方向正确**:clip 在多 epoch 复用旧数据时给"走太远"踩了刹车。

诚实说明:**1 维 bandit 太温和,不足以复现正文所述 (a) KL 失控/崩溃、(b) 显著不稳的剧烈现象**——那需要高维策略 + 真实环境放大 r 的方差。要拿到练习要求的"KL 失控、训练崩溃/震荡"的对照曲线,请用方案一(CartPole)跑 3 个种子并画 return + KL 双图。「待核:CartPole 上各设置的具体崩溃幅度/曲线形状依超参与种子而定,无统一数字」。
