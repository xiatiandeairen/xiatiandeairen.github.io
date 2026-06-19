---
title: "RLHF / DPO / GRPO 对齐（强化学习域）"
slug: "6-06"
collection: "tech-library"
group: "强化学习"
order: 6006
summary: "强化学习域 · 深化篇 面向有工程经验、要把 RL（含 LLM 对齐用 RL）吃透成专家的读者。术语保留英文。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-14T20:43:48.000Z"
updatedAt: "2026-06-14T20:43:48.000Z"
---
> 强化学习域 · 深化篇
> 面向有工程经验、要把 RL（含 LLM 对齐用 RL）吃透成专家的读者。术语保留英文。

---

## TL;DR（先给结论）

- **本章主线是一条「不断做减法」的演化链：RLHF（InstructGPT）→ DPO（砍掉 reward model + 砍掉 RL 采样）→ GRPO（砍掉 critic）。** 三者解的都是同一个问题——**用人类偏好把语言模型对齐到「人想要的输出」**——但代价、稳定性、工程复杂度逐级下降。理解这条链的关键不是记三个公式，而是看清**每一步砍掉了什么、用什么补回来、损失了什么**。前一章的 PPO 是这条链的起点和基准，本章默认你已吃透 PPO 的 clip surrogate 与「两种 KL」。

- **RLHF（InstructGPT, Ouyang et al. 2022）= 三段式流水线**：① SFT（监督微调）；② **训一个 reward model**（用人标的 pairwise 偏好，pairwise ranking loss）；③ **用 PPO 把 policy 往 reward model 的高分推**，同时减一个 `β·KL(policy‖SFT)` 防跑偏。它工作，但**重**——要同时维护 4 个模型（policy / reference / reward / value），reward model 本身会被 hack，PPO 调参玄学。

- **DPO（Rafailov et al. 2023）的核心洞察：RLHF 的最优 policy 有闭式解，于是 reward model 根本不用单独训。** 从「KL 约束下最大化 reward」推出最优 policy `π_r(y|x) = (1/Z)·π_ref(y|x)·exp(r/β)`，**反解出隐式 reward** `r(x,y) = β·log(π_θ(y|x)/π_ref(y|x)) + β·log Z(x)`，代进 Bradley-Terry 偏好模型后 `log Z(x)` 神奇地消掉，**整个 RLHF 退化成一个二分类（logistic regression）loss**：
  `L_DPO = −E[ log σ( β·log(π_θ(y_w|x)/π_ref(y_w|x)) − β·log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]`。
  **不训 reward model、不做 RL 采样、不需要 PPO 那套 clip/value/advantage**，一个 forward + 一个 logsigmoid 就完事。代价：它是 **off-policy**（吃固定的偏好数据集），且有它自己的失败模式（**likelihood displacement**，§5）。

- **GRPO（DeepSeekMath, Shao et al. 2024）解决的是 PPO 路线里「critic 太贵」的问题。** value model 通常和 policy 一样大，载它显存翻倍。GRPO 的做法：对同一个 prompt **采样一组 G 个 response**，用**这组 reward 的均值当 baseline、标准差做归一化**，得到 group-relative advantage `Â_i = (r_i − mean(r)) / std(r)`，**直接扔掉 value 网络**。这是「回归 REINFORCE 本源 + 用组内统计当 baseline」。它是 DeepSeek-R1 推理能力训练的核心算法，**2024-2025 年事实上的 RLVR（RL with Verifiable Rewards）主力**。

- **⭐可运行 demo（本章重中之重）**：
  ① **最小 DPO 训练 loop**（纯 numpy，无需 torch / GPU）：把 DPO 剥到只剩数学骨架——一个对 K 个候选 response 的 softmax policy + 冻结 reference，喂偏好对，手推梯度，**实测看到 loss 下降、隐式 reward margin 上升、p(最优 response) 从 0.18 涨到 0.66、最终分布按真实 reward 单调排序**。与 TRL `dpo_trainer.py` 的 loss 逐行呼应。
  ② **GRPO group-advantage demo**（纯 numpy）：实测演示「组内均值当 baseline」如何自动消掉 prompt 难度差异、每组 advantage 和恒为 0。与 TRL `grpo_trainer.py@2197-2199` 逐字对照。
  两个 demo 的输出都是**本文档实际运行抓取的真实输出**，**设计为可运行，请在你的环境验证**。

- **三者怎么选（§4 详表）**：有高质量**偏好对数据**、要的是「风格/安全/有用性」对齐、想省事 → **DPO**；有可程序化验证的 reward（数学题对错、代码跑没跑过、单元测试）、要的是**推理能力** → **GRPO/RLVR**；要最大化可控性、能养一支 RL infra 团队、reward 复杂且需要泛化 → **PPO-RLHF**（仍是上限最高、最灵活的那条路）。这不是「新的淘汰旧的」，而是**不同数据形态对应不同算法**。

---

## 前置依赖（读这章前你该有的东西）

**概念前置**
- **第 4 章 PPO 全部**：clip surrogate `min(r·A, clip(r)·A)`、GAE advantage、value/critic 的作用（降方差 baseline）、**RLHF 里的两种 KL**（policy-vs-reference 进 reward shaping vs new-vs-old 算法监控）。本章 §1.3、§6 直接站在它肩膀上。
- **第 3 章**：policy gradient theorem、log-derivative trick、baseline 为什么能降方差且不引入 bias。GRPO 的「组均值当 baseline」就是这个定理的直接应用。
- **KL 散度**的基本定义与「约束最大化」的 Lagrangian 直觉（DPO 的推导核心是一个 KL 约束的最优化问题）。
- **Bradley-Terry 偏好模型**：`P(y_w ≻ y_l) = σ(r(y_w) − r(y_l))`，即「偏好概率 = 两个 reward 之差过 sigmoid」。DPO 和 reward model 训练都建在它上面。不熟也没关系，§1.2 会现推。

**工程前置（跑 demo 用）**
- Python 3.9+。
- 本章两个核心 demo **只需 numpy**（刻意设计成不依赖 torch / GPU，让你在任何机器一键跑通）：
  ```bash
  pip install numpy
  ```
- 想跑真·LLM 规模的 DPO/GRPO（可选，本章不强制）：`pip install trl transformers datasets accelerate`，参考 [huggingface/trl](https://github.com/huggingface/trl) 的 `DPOTrainer` / `GRPOTrainer`。这需要 GPU。

**读法建议**
- 想最快建直觉：先看 §1.2 的 DPO 三步推导（约束最优化 → 反解隐式 reward → 代入 BT 消 Z），再跑 §3.1 的 numpy DPO demo 看 margin 上升。「免 reward model」这件抽象的事会立刻变具体。
- 做对齐工程选型：§4 方案对比 + §5 失败模式（DPO 的 likelihood displacement、GRPO 的 reward hacking）最高优先。
- 抠源码：§2 直接逐行拆 TRL 的 `dpo_trainer.py` 与 `grpo_trainer.py` 真实代码，标了 repo@行号。

---

## 模板六段

> 六段：① 设计考古（RLHF→DPO→GRPO 的演化动机）② 核心机制（TRL 源码逐行精读）③ 可运行 demo ④ 方案对比 ⑤ 失败模式与真坑 ⑥ 生产落地与算法选型。

---

## §1 设计考古：从 RLHF 到 DPO 到 GRPO

这一章的三个算法不是并列的三选一，而是**一条时间线上不断「做减法」的演化**。把它当故事读：每一代都在前一代「太贵 / 太不稳 / 太复杂」的痛点上动刀。

### 1.1 起点：RLHF / InstructGPT 的三段式流水线

**RLHF（Reinforcement Learning from Human Feedback）** 的工业化奠基之作是 OpenAI 的 InstructGPT。

> 【设计考古，WebFetch 核实：arXiv:2203.02155（ar5iv）】作者 Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin 等（OpenAI），2022 年 3 月。这是把「人类反馈 + RL」工程化对齐 GPT-3 的奠基论文，ChatGPT 的训练范式直接源于此。

它分三步（务必记住这个流水线，DPO 和 GRPO 都是在砍它的某一步）：

**Step 1 — SFT（Supervised Fine-Tuning）。** 用人写的高质量「指令→回答」示范，监督微调 GPT-3。得到 `π_SFT`，作为后续所有步骤的起点和 reference。

**Step 2 — 训 Reward Model。** 对一个 prompt，让模型生成 K 个 response（K=4~9），人**排序**它们。把排序拆成所有 pairwise 对 `(y_w, y_l)`（y_w 比 y_l 更优），训一个 reward model `r_θ` 去拟合这些偏好。

> 【真实公式，WebFetch 核实：InstructGPT 论文（ar5iv）】reward model 的 loss（pairwise ranking / Bradley-Terry）：
> `loss(θ) = − (1 / C(K,2)) · E_{(x,y_w,y_l)~D}[ log σ( r_θ(x,y_w) − r_θ(x,y_l) ) ]`
> 即「让更优 response 的分数减更差 response 的分数过 sigmoid 后尽量大」。**记住这个式子——DPO 的整个推导就是要把这个 reward model 从流水线里删掉。**

**Step 3 — PPO RL 微调。** 用第 4 章的 PPO，把 `π_SFT` 往「reward model 给的高分」推，同时减一个 KL 惩罚防止模型为刷分而胡说（reward hacking）。

> 【真实公式，WebFetch 核实：InstructGPT 论文（ar5iv）】RL 优化目标（PPO-ptx）：
> `objective(φ) = E_{(x,y)~D_πφ}[ r_θ(x,y) − β·log( π_φ^RL(y|x) / π^SFT(y|x) ) ] + γ·E_{x~D_pretrain}[ log π_φ^RL(x) ]`
> 三项：① reward model 打分（要最大化）；② `β·KL(policy‖SFT)` 惩罚（第 4 章的「KL ①」，防跑偏）；③ `γ·` 预训练梯度混合（防止 RL 把通用能力训没了，即「ptx」=pretraining mix）。

**RLHF 的痛点（DPO/GRPO 的动机来源）：**
1. **要训一个独立的 reward model**——多一个模型、多一份标注、reward model 本身会过拟合 / 被 hack。
2. **PPO 阶段要在线采样**——每一步都要大模型 `generate`，**极贵**。
3. **要同时维护 4 个模型**（policy / reference / reward / value），显存和工程开销巨大。
4. **PPO 调参玄学**——第 4 章 §5 那一长串失败模式，每一条在 LLM 规模上都更难调。

接下来两节，DPO 砍掉痛点 1+2，GRPO 砍掉痛点 3 里的 value model。

### 1.2 DPO 的发明：reward model 其实是多余的（核心推导）

> 【设计考古，WebFetch 核实：arXiv:2305.18290（ar5iv）】《Direct Preference Optimization: Your Language Model is Secretly a Reward Model》，作者 Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn（Stanford），2023 年 5 月。摘要原文称 RLHF 是 "a complex and often unstable procedure"，DPO 则 "stable, performant, and computationally lightweight, eliminating the need for sampling from the LM during fine-tuning or performing significant hyperparameter tuning."

DPO 的洞察可以分三步推。**这是本章最重要的推导，务必看懂。**

**第一步：写出 RLHF 第 3 步的优化目标。**
RLHF 要解的是「KL 约束下最大化 reward」：

```
max_π  E_{x~D, y~π(·|x)}[ r(x,y) ]  −  β·KL( π(·|x) ‖ π_ref(·|x) )
```

**第二步：这个目标有闭式最优解。**
这是一个标准的「带 KL 正则的 reward 最大化」，用变分法（或 Lagrangian）能直接解出最优 policy：

> 【真实公式，WebFetch 核实：DPO 论文（ar5iv）Eq.4】
> `π_r(y|x) = (1/Z(x)) · π_ref(y|x) · exp( (1/β)·r(x,y) )`，其中 `Z(x) = Σ_y π_ref(y|x)·exp((1/β)·r(x,y))` 是配分函数（partition function）。

直觉：最优 policy = reference 分布**按 reward 指数倾斜**——reward 高的 response 概率被乘大，`β` 控制倾斜力度，`Z(x)` 做归一化。问题是 `Z(x)` 要对整个输出空间求和，**算不出来**（这正是直接用这个式子做 RL 很难的原因）。

**第三步：反解出隐式 reward，代入 BT 模型，`Z(x)` 消掉。**
对 Eq.4 两边取 log 再移项，**把 reward 用 policy 表示**：

> 【真实公式，WebFetch 核实：DPO 论文（ar5iv）Eq.5】
> `r(x,y) = β·log( π_r(y|x) / π_ref(y|x) ) + β·log Z(x)`

这就是论文标题「**你的语言模型本身就是一个 reward model**」的含义：任何 policy `π` 都对应一个隐式 reward `β·log(π/π_ref)`（差一个只依赖 x 的常数 `β·log Z(x)`）。

现在把这个隐式 reward 代进 Step 2 的 **Bradley-Terry 偏好模型** `P(y_w ≻ y_l) = σ(r(x,y_w) − r(x,y_l))`。注意 reward 之差里，**`β·log Z(x)` 项两边相同，直接抵消**！于是偏好概率只剩 policy 的对数比之差。最大化偏好数据的对数似然，就得到：

> 【真实公式，WebFetch 核实：DPO 论文（ar5iv）Eq.7】
> `L_DPO(π_θ; π_ref) = − E_{(x,y_w,y_l)~D}[ log σ( β·log(π_θ(y_w|x)/π_ref(y_w|x)) − β·log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]`

**这就是 DPO 的全部。** 对比一下你会发现它和 Step 2 的 reward model loss **形式完全一样**——都是 `−log σ(score_w − score_l)`。区别只是：reward model loss 里的 `r_θ` 是一个**单独训练的网络**，而 DPO 里的「隐式 reward」`β·log(π_θ/π_ref)` **直接就是被对齐的 policy 自己**。**于是 reward model 这一步被彻底删除了，RL 采样也没了——整个 RLHF 退化成在固定偏好数据集上跑一个分类 loss。**

**DPO 的梯度告诉你它在干什么（关键直觉）：**

> 【真实公式，WebFetch 核实：DPO 论文（ar5iv）§4】
> `∇_θ L_DPO = − β·E[ σ( r̂_θ(x,y_l) − r̂_θ(x,y_w) ) · ( ∇_θ log π_θ(y_w|x) − ∇_θ log π_θ(y_l|x) ) ]`
> 其中 `r̂_θ(x,y) = β·log(π_θ(y|x)/π_ref(y|x))` 是隐式 reward。

拆开看：
- 括号里 `∇log π(y_w) − ∇log π(y_l)`：**抬高 y_w 的概率、压低 y_l 的概率**（标准的偏好对比）。
- 前面的系数 `σ(r̂(y_l) − r̂(y_w))`：当**当前模型的隐式 reward 把偏好排错了**（即 y_l 的隐式 reward 反而比 y_w 高），这个 sigmoid 接近 1，**梯度权重大**；当模型已经排对了，系数接近 0，**几乎不更新**。论文原话是 "higher weight when reward estimate is wrong"。这是 DPO 自带的「难例挖掘」，也是它比朴素 SFT-on-chosen 强的地方。

> 【⚠ 关键，但常被忽略】这个梯度只保证「**margin**（y_w 与 y_l 的隐式 reward 之差）变大」，**不保证 y_w 的绝对概率上升**。完全可能 y_w 和 y_l 的概率**双双下降**、只是 y_l 降得更快——这就是 §5 要讲的 **likelihood displacement** 失败模式的种子。

### 1.3 GRPO 的发明：critic 太贵，用一组 sample 当 baseline

DPO 走的是「完全离开 RL」的路。但有一类任务，偏好不是人标的、而是**可程序化验证**的——数学题答案对不对、代码能不能编译、单元测试过不过。这类「**verifiable reward**」天然适合在线 RL（每生成一条就能立刻打分），DPO 的「固定偏好数据集」反而不如直接 RL。于是 PPO 路线在这里仍有价值——但 PPO 的 critic 太贵。

> 【设计考古，WebFetch 核实：arXiv:2402.03300（ar5iv）】GRPO 出自 DeepSeek 的《DeepSeekMath》，作者 Zhihong Shao 等（DeepSeek-AI），2024 年。论文动机原文："As the value function employed in PPO is typically another model of comparable size as the policy model, it brings a substantial memory and computational burden." 解法："obviate the need for additional value function approximation" by "using the average reward of multiple sampled outputs, produced in response to the same question, as the baseline."

**GRPO 的核心一句话**：PPO 用一个**学出来的 value 网络** `V(s)` 当 baseline 来降方差（第 3、4 章）；GRPO 不学这个网络，改成**对同一个 prompt 采样一组 G 个 response，用这组 reward 的均值当 baseline、标准差做归一化**：

> 【真实公式，WebFetch 核实：DeepSeekMath 论文（ar5iv）】group-relative advantage（outcome supervision）：
> `Â_{i,t} = r̃_i = ( r_i − mean(r_1,...,r_G) ) / std(r_1,...,r_G)`
> 同一组里所有 token 共享同一个 advantage（outcome 监督：整条 response 只有一个最终 reward）。

GRPO 的完整目标 = PPO 的 clip surrogate（沿用第 4 章）+ 把 advantage 换成上面的 group-relative + 一个**直接加在 loss 里**的 KL 惩罚：

> 【真实公式，WebFetch 核实：DeepSeekMath 论文（ar5iv）】
> ```
> J_GRPO(θ) = E[ (1/G) Σ_{i=1}^G (1/|o_i|) Σ_{t=1}^{|o_i|} {
>     min[ ρ_{i,t}·Â_{i,t} ,  clip(ρ_{i,t}, 1−ε, 1+ε)·Â_{i,t} ]  −  β·D_KL[π_θ ‖ π_ref]
> } ]
> ```
> 其中 `ρ_{i,t} = π_θ(o_{i,t}|q,o_{i,<t}) / π_{θ_old}(o_{i,t}|q,o_{i,<t})` 是 token 级概率比（与 PPO 一致）。

注意两个和 PPO 的关键区别：
1. **没有 value 网络**——advantage 来自组内统计，不来自 critic。这就是省下的那「一半显存」。
2. **KL 惩罚直接进 loss**（`− β·D_KL`），而不是像 InstructGPT 那样混进 reward。它用的是一个**无偏 KL 估计器**（就是第 4 章讲的 John Schulman k3 估计器）：

> 【真实公式，WebFetch 核实：DeepSeekMath 论文（ar5iv）】
> `D_KL[π_θ‖π_ref] = π_ref(o|·)/π_θ(o|·) − log(π_ref(o|·)/π_θ(o|·)) − 1`
> 这个式子恒 ≥ 0（因为 `x − log x − 1 ≥ 0`）、无偏、方差小——和第 4 章 §2.4 / §6.3 的 k3 是同一个东西。**这是 PPO 章知识在本章的直接复用。**

**一句话串起三代：**
- RLHF：reward model + PPO（policy/ref/reward/value 四模型）。
- DPO：删掉 reward model + 删掉 RL 采样（policy/ref 两模型，离线）。
- GRPO：保留 RL 在线采样、但删掉 value model（policy/ref/reward 三模型；reward 常是规则函数，连模型都不是）。

---

## §2 核心机制：TRL 源码逐行精读

这一节直接拆 HuggingFace TRL 的真实源码。**所有代码块标了 repo@行号，逐字摘录。** 你会发现，理解了 §1 的数学后，源码短到出乎意料——DPO 的核心 loss 不到 10 行。

### 2.1 共同基石：从 logits 取 log-prob（`selective_log_softmax`）

DPO 和 GRPO 都要反复算「某条 response 在 policy / reference 下的 log-prob」。TRL 把这步抽成一个工具函数，**两个 trainer 都调它**。先看它，因为后面所有 `logp` 都来自这里。

> 【真实源码 huggingface/trl@trl/trainer/utils.py:436-467（节选）】
> ```python
> def selective_log_softmax(logits, index) -> torch.Tensor:
>     # 等价于 torch.gather(logits.log_softmax(-1), dim=-1, index).squeeze(-1)
>     # 但用 logsumexp 省峰值显存
>     squeeze = index.ndim == logits.ndim - 1
>     if squeeze:
>         index = index.unsqueeze(-1)
>     if logits.dtype in [torch.float32, torch.float64]:
>         selected_logits = torch.gather(logits, dim=-1, index=index)
>         logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
>         per_token_logps = selected_logits - logsumexp_values.unsqueeze(-1)  # log_softmax(x_i)=x_i−logsumexp(x)
>     ...
> ```

一句话：**给定 logits 和「实际生成的 token id」，取出每个位置上那个 token 的 log 概率**。注释里那行 `log_softmax(x_i) = x_i − logsumexp(x)` 是全部数学。一条 response 的 sequence log-prob = 把它所有 token 的 per-token logp **求和**。记住这点——DPO loss 里的 `π_θ(y|x)` 就是「这条 response 所有 token logp 之和」。

### 2.2 DPO loss：不到 10 行的核心

下面是 TRL DPO trainer 计算 loss 的真实代码。前置：`chosen_logps` / `rejected_logps` 是 policy 对 y_w / y_l 的 sequence log-prob（chunk 自一个把 [chosen; rejected] 拼起来的 batch），`ref_*` 同理来自冻结的 reference。

> 【真实源码 huggingface/trl@trl/trainer/dpo_trainer.py:1245-1283（节选）】
> ```python
> # Get the log ratios for the chosen and rejected responses
> chosen_logratios = chosen_logps - ref_chosen_logps          # log(π_θ(y_w)/π_ref(y_w))
> rejected_logratios = rejected_logps - ref_rejected_logps    # log(π_θ(y_l)/π_ref(y_l))
>
> if self.f_divergence_type == "reverse_kl":  # standard DPO
>     chosen_scores = chosen_logratios
>     rejected_scores = rejected_logratios
> ...
> delta_score = chosen_scores - rejected_scores               # = (logratio_w − logratio_l)
>
> loss = 0.0
> for loss_type, loss_weight in zip(self.loss_types, self.loss_weights, strict=True):
>     if loss_type == "sigmoid":
>         per_sequence_loss = -F.logsigmoid(self.beta * delta_score)   # ← DPO 论文 Eq.7 逐字落地
>     elif loss_type == "hinge":
>         per_sequence_loss = torch.relu(1 - self.beta * delta_score)  # IPO/SLiC 系变体
>     ...
> ```

**把它和 §1.2 的 Eq.7 对齐：**
- `chosen_logratios = chosen_logps - ref_chosen_logps` 就是 `log(π_θ(y_w|x)/π_ref(y_w|x))`。
- `delta_score = chosen_scores - rejected_scores` 就是 Eq.7 里 sigmoid 内部那一整坨「两个对数比之差」。
- `per_sequence_loss = -F.logsigmoid(self.beta * delta_score)` **逐字就是** `−log σ(β·delta)`。

**没有 RL、没有采样、没有 advantage、没有 clip。** 一个 forward 拿到四个 logp，三行算 loss。这就是 DPO「computationally lightweight」的全部含义。

顺带看 reward 的记录（这就是「隐式 reward」在代码里的样子）：

> 【真实源码 huggingface/trl@trl/trainer/dpo_trainer.py:1171-1175（节选）】
> ```python
> reward_accuracies = (chosen_rewards > rejected_rewards).float()
> ...
> margins = chosen_rewards - rejected_rewards
> ```
> 其中 `chosen_rewards = β·chosen_logratios`（隐式 reward）。`reward_accuracies` = 「模型把 y_w 的隐式 reward 排得比 y_l 高」的比例——**这是监控 DPO 是否在学的头号指标**（demo 里我们也会看它）。

**默认超参（务必记）：**
> 【真实源码 huggingface/trl@trl/trainer/dpo_config.py】`beta: float = 0.1`（"Higher β means less deviation from the reference model."）、`loss_type = ["sigmoid"]`（标准 DPO）、`label_smoothing = 0.0`（设 >0 即 cDPO / conservative DPO）。loss_type 还支持 `ipo / hinge / robust / apo_*` 等十几种变体。

> 【设计考古补：β 的取值】DPO 论文原文承认 "we did not meaningfully tune DPO's β"，实验里扫过 `β ∈ {0.05, 0.1, 1, 5}`。实践共识：**β 太小（→0）模型可以任意偏离 reference，容易过拟合偏好数据 / 退化；β 太大模型被 reference 死死拴住学不动。0.1 是 TRL 默认起点。**

### 2.3 GRPO advantage：group 归一化的真实代码

GRPO 的「灵魂」是 group-relative advantage。看 TRL 怎么算（已采到一批 reward，`num_generations` = 每个 prompt 采的 G）：

> 【真实源码 huggingface/trl@trl/trainer/grpo_trainer.py:2177-2199（节选）】
> ```python
> mean_grouped_rewards = torch.nanmean(rewards.view(-1, num_generations), dim=1)
> mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
> if self.scale_rewards in ["group", "none"]:
>     if num_generations > 1:
>         std_rewards = nanstd(rewards.view(-1, num_generations), dim=1)
>         std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
> ...
> advantages = rewards - mean_grouped_rewards
> if self.scale_rewards != "none":
>     advantages = advantages / (std_rewards + 1e-4)
> ```

**逐行对齐 §1.3 的 `Â_i = (r_i − mean(r)) / std(r)`：**
- `rewards.view(-1, num_generations)`：把扁平的 reward 重排成「每行一个 prompt 的 G 个 response」。
- `mean_grouped_rewards`：每组（每个 prompt）的 reward 均值，`repeat_interleave` 把它广播回每个 response。**这就是 baseline——critic 的替身。**
- `advantages = rewards - mean_grouped_rewards`：减去组均值（降方差，对应第 3 章 baseline 定理）。
- `/ (std_rewards + 1e-4)`：除以组标准差归一化，`+1e-4` 防除零。

> 【工程注脚】`scale_rewards` 默认 `"group"`（DeepSeekMath 原版，除组内 std）。后来的研究（Dr.GRPO）指出「除 std」会引入对「容易题 / 难题」的长度偏置，于是 TRL 提供 `"none"`（不除 std，对应 Dr.GRPO）和 `"batch"`（除全 batch 的 std）。**这是 GRPO 已知偏置的修补，§5 会提。**

### 2.4 GRPO loss：clip surrogate + KL，全是 PPO 的影子

advantage 算完，loss 部分**几乎就是第 4 章的 PPO clip surrogate**，外加一个直接加的 KL：

> 【真实源码 huggingface/trl@trl/trainer/grpo_trainer.py:2549-2586（节选，loss_type="grpo" 分支）】
> ```python
> log_ratio = per_token_logps - old_per_token_logps
> if self.importance_sampling_level == "token":
>     log_importance_weights = log_ratio
> ...
> coef_1 = torch.exp(log_importance_weights)                  # ρ = π_θ/π_old，PPO 的概率比
>
> # Compute the KL divergence between the model and the reference model
> if self.beta != 0.0:
>     ref_per_token_logps = inputs["ref_per_token_logps"]
>     per_token_kl = (
>         torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
>     )                                                        # ← k3 无偏 KL 估计器，逐字
> ...
> coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
> per_token_loss1 = coef_1 * advantages
> per_token_loss2 = coef_2 * advantages
> per_token_loss = -torch.min(per_token_loss1, per_token_loss2)   # ← PPO clip surrogate，逐字
> ```

逐行对齐：
- `coef_1 = exp(per_token_logps - old_per_token_logps)` = 概率比 `ρ`（第 4 章的 `r_t`）。
- `per_token_kl = exp(ref−policy) − (ref−policy) − 1` **逐字就是** §1.3 那个 k3 KL 估计器 `π_ref/π_θ − log(π_ref/π_θ) − 1`。
- `coef_2 = clamp(coef_1, 1−ε_low, 1+ε_high)` + `−min(loss1, loss2)` **就是第 4 章的 clip surrogate** `−min(ρ·A, clip(ρ)·A)`。（注意 TRL 把 ε 拆成 low/high 两侧，支持 DAPO 的「clip-higher」非对称裁剪。）

KL 最后怎么进 loss：

> 【真实源码 huggingface/trl@trl/trainer/grpo_trainer.py:2615-2620（节选）】
> ```python
> if self.beta != 0.0:
>     per_token_loss = per_token_loss + self.beta * per_token_kl   # KL 直接加进 loss（不进 reward）
> ...
> if self.loss_type in ["grpo", "sapo"]:
>     loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
> ```

`per_token_loss + self.beta * per_token_kl` **就是** `J_GRPO` 里的 `−β·D_KL`（符号：这里在最小化 loss，所以 KL 是加的正惩罚）。最后 `(per_token_loss * mask).sum(-1) / mask.sum(-1)` 是「每条 response 内对有效 token 求平均」再 `.mean()` 跨样本平均——对应 `J_GRPO` 的双重求和归一。

> 【源码现状提醒，待核细节】当前 TRL 的 `grpo_trainer.py` 已演化得很复杂（2700+ 行），塞进了 vLLM importance sampling correction、十几种 loss_type（dr_grpo / dapo / cispo / bnpo / luspo …）、token/sequence 两级 importance sampling 等。**上面摘的是核心 `loss_type="grpo"` 路径**，与 DeepSeekMath 原版对应；其余分支是后续论文的改进，不在本章范围。如需精确到某一行，请以 [trl main 分支](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py) 当时的版本为准。

### 2.5 三个算法的 loss 放一起对比（看「砍了什么」）

| 算法 | loss 核心（伪代码） | 需要的模型 | 在线采样 | advantage 来源 |
|---|---|---|---|---|
| **RLHF-PPO** | `−min(ρ·A, clip(ρ)·A) − c·entropy + c·VF_loss`，reward 里含 `−β·KL(π‖ref)` | policy + ref + **reward** + **value** | **是** | GAE（**value 网络**） |
| **DPO** | `−logσ(β·(logratio_w − logratio_l))` | policy + ref | **否**（离线偏好集） | 无（不是 RL） |
| **GRPO** | `−min(ρ·A, clip(ρ)·A) + β·KL_k3`，`A=(r−mean)/std` | policy + ref + reward(常是规则) | **是** | **组内统计**（无 value 网络） |

看这张表，「演化 = 做减法」一目了然：DPO 删掉 reward + value + 采样；GRPO 保留采样、删掉 value、把 reward 换成可验证的规则函数。

---

## §3 ⭐可运行 demo：从零跑 DPO 与 GRPO 核心

> **本章重中之重。** 两个 demo **只需 numpy**（刻意不依赖 torch/GPU，任何机器一键跑）。它们把 LLM 的 token 细节剥掉，只留算法骨架，让你**亲眼看到** DPO 的 margin 上升、GRPO 的 group baseline 生效。输出均为**本文档实际运行抓取的真实输出**。**设计为可运行，请在你的环境验证。**

### 3.1 最小 DPO 训练 loop（纯 numpy）

**思路**：把「prompt → response」简化成一个 **K 选 1 的 bandit**——policy 是对 K 个候选 response 的 softmax 分布 `π_θ`，reference 是训练前冻结的同一份分布 `π_ref`。每条偏好对 `(y_w, y_l)` 里 y_w 是更优候选。我们手推 DPO 梯度并更新。这剥离了 token 细节，**只留 DPO 的核心：用 `β·(logp 对数比之差)` 过 sigmoid 当二分类 loss**，与 §2.2 的 `delta_score` / `per_sequence_loss` 逐行呼应。

```bash
pip install numpy
python dpo_min_numpy.py
```

```python
# dpo_min_numpy.py — 最小可运行 DPO 训练 loop，纯 numpy，无需 torch。【示意，非逐字】，设计为可运行。
# 把"序列级 bandit"当 LLM 的玩具替身,只保留 DPO 的核心数学,与 trl/dpo_trainer.py 的 loss 呼应。
import numpy as np
np.random.seed(0)

K = 6          # 候选 response 数（= 词表/动作空间）
BETA = 0.1     # DPO 温度，TRL 默认 0.1
LR = 0.5
STEPS = 300

# 真实"偏好结构"：reward 越高越优（只用来生成偏好对，policy 永远看不到它）
true_reward = np.array([3.0, 2.0, 1.0, 0.0, -1.0, -2.0])

# policy 与 reference 共享同一份初始 logits（DPO 要求 ref 是 policy 训练前的冻结副本）
init_logits = np.random.randn(K) * 0.1
theta = init_logits.copy()          # 被训练的 policy logits
ref_logits = init_logits.copy()     # 冻结的 reference logits（不再更新）

def log_softmax(z):
    z = z - z.max()
    return z - np.log(np.exp(z).sum())

ref_logp = log_softmax(ref_logits)  # 固定不变，对应 ref_chosen_logps / ref_rejected_logps

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sample_pref_pair():
    # 随机取两个不同候选，reward 高的当 y_w，低的当 y_l（模拟人标偏好）
    i, j = np.random.choice(K, size=2, replace=False)
    return (i, j) if true_reward[i] >= true_reward[j] else (j, i)

print(f"{'step':>5} {'loss':>8} {'margin':>8} {'p(best)':>8} {'acc':>6}")
for step in range(STEPS + 1):
    B = 16
    grad = np.zeros(K)
    total_loss = total_margin = 0.0
    correct = 0
    for _ in range(B):
        w, l = sample_pref_pair()
        logp = log_softmax(theta)
        # 隐式 reward = β·(logp_policy − logp_ref)，对应 chosen_rewards/rejected_rewards
        rw = BETA * (logp[w] - ref_logp[w])
        rl = BETA * (logp[l] - ref_logp[l])
        margin = rw - rl                      # 对应 delta_score 经 β 缩放
        total_margin += margin
        loss = -np.log(sigmoid(margin) + 1e-12)   # 对应 -F.logsigmoid(β·delta_score)
        total_loss += loss
        correct += int(margin > 0)            # 对应 reward_accuracies
        # ---- 手推梯度 ----
        # dL/dmargin = -(1 - σ(margin)) = -σ(-margin)  ← 论文"错得越多权重越大"的系数
        g_margin = -(1.0 - sigmoid(margin))
        # margin = β·(logp[w]-logp[l]); d logp[c]/d theta_k = 1[k==c] - softmax(theta)[k]
        p = np.exp(logp)
        dlogp_w = -p.copy(); dlogp_w[w] += 1.0
        dlogp_l = -p.copy(); dlogp_l[l] += 1.0
        grad += g_margin * BETA * (dlogp_w - dlogp_l)
    grad /= B
    theta -= LR * grad        # 梯度下降最小化 loss

    if step % 50 == 0:
        p = np.exp(log_softmax(theta))
        print(f"{step:>5} {total_loss/B:>8.4f} {total_margin/B:>8.4f} {p[0]:>8.4f} {correct/B:>6.2f}")

print("\n最终 policy 概率分布（按候选）:", np.round(np.exp(log_softmax(theta)), 4))
print("reference   概率分布（按候选）:", np.round(np.exp(ref_logp), 4))
print("真实 reward 排序（高→低应概率高→低）:", true_reward)
```

**预期输出**（本文档实际运行抓取，seed=0）：

```
 step     loss   margin  p(best)    acc
    0   0.6931   0.0000   0.1789   0.00
   50   0.6767   0.0332   0.2638   1.00
  100   0.6575   0.0730   0.3528   1.00
  150   0.6401   0.1104   0.4440   1.00
  200   0.6230   0.1475   0.5264   1.00
  250   0.6154   0.1645   0.6017   1.00
  300   0.5753   0.2549   0.6579   1.00

最终 policy 概率分布（按候选）: [0.6579 0.2112 0.0757 0.0361 0.0146 0.0046]
reference   概率分布（按候选）: [0.178  0.1553 0.1646 0.1867 0.1799 0.1354]
真实 reward 排序（高→低应概率高→低）: [ 3.  2.  1.  0. -1. -2.]
```

**怎么读这个输出（这就是 DPO「在工作」的全部证据）：**
- `loss` 从 0.6931（=`−log σ(0)`=`log 2`，初始 margin=0）单调下降。
- `margin`（隐式 reward 之差）从 0 涨到 0.25——**模型在拉大「优 vs 劣」的隐式 reward 差距**，正是 DPO 目标。
- `acc`（reward accuracy）一步就到 1.00——隐式 reward 排序正确。
- `p(best)`（最优候选的概率）从 0.18 → 0.66——**概率质量在向高 reward 候选集中**。
- 最关键：**最终分布 `[0.66, 0.21, 0.08, 0.04, 0.015, 0.005]` 按真实 reward 完美单调递减**，而 reference 是接近均匀的 `[0.18, 0.16, 0.16, 0.19, 0.18, 0.14]`。**DPO 从头到尾没见过 `true_reward`，只靠偏好对，就还原出了 reward 排序**——这就是「你的语言模型本身就是 reward model」的可视化。

> 把它和 §2.2 源码对齐：`margin` ↔ `β·delta_score`；`loss` ↔ `-F.logsigmoid(β·delta_score)`；`acc` ↔ `reward_accuracies`；`g_margin = -(1-σ(margin))` ↔ DPO 梯度里那个 `σ(r̂_l − r̂_w)` 系数（符号同源）。

### 3.2 GRPO group-advantage demo（纯 numpy）

**思路**：GRPO 的灵魂是「用一组 reward 的均值/标准差当 baseline」。这个 demo 不训练，只**演示 advantage 的计算**，让你看清它如何自动消掉 prompt 难度差异。与 §2.3 的 TRL 源码 `advantages = (rewards - mean_grouped_rewards) / (std_rewards + 1e-4)` 逐字对照。

```bash
python grpo_adv_numpy.py
```

```python
# grpo_adv_numpy.py — GRPO 的核心:group-relative advantage,纯 numpy。【示意,非逐字】,设计为可运行。
# 对照 trl/grpo_trainer.py@2177-2199。
import numpy as np

num_prompts, G = 2, 4          # 2 个 prompt,每个采样 G=4 个 response
rewards = np.array([
    [0.9, 0.2, 0.8, 0.1],      # prompt 0 的 4 个 response 的 reward(reward model 打分)
    [5.0, 5.1, 4.9, 5.2],      # prompt 1:reward 整体高,但组内差异小
], dtype=float)

# ---- 对照 TRL: mean over group; advantages=(r-mean)/(std+1e-4) ----
grp = rewards.reshape(-1, G)               # (num_prompts, G)
mean_g = grp.mean(axis=1, keepdims=True)
std_g  = grp.std(axis=1, keepdims=True)
adv = (grp - mean_g) / (std_g + 1e-4)      # group-normalized advantage

print("原始 rewards:\n", np.round(rewards, 3))
print("\n每组均值 mean_g:", np.round(mean_g.reshape(-1), 3))
print("每组标准差 std_g:", np.round(std_g.reshape(-1), 3))
print("\ngroup-relative advantages(进 policy gradient 的就是它):\n", np.round(adv, 3))
print("\n每组 advantage 之和 ≈ 0(均值被减掉):", np.round(adv.sum(axis=1), 6))
```

**预期输出**（本文档实际运行抓取）：

```
原始 rewards:
 [[0.9 0.2 0.8 0.1]
 [5.  5.1 4.9 5.2]]

每组均值 mean_g: [0.5  5.05]
每组标准差 std_g: [0.354 0.112]

group-relative advantages(进 policy gradient 的就是它):
 [[ 1.131 -0.848  0.848 -1.131]
 [-0.447  0.447 -1.34   1.34 ]]

每组 advantage 之和 ≈ 0(均值被减掉): [0. 0.]
```

**怎么读（GRPO「为什么能省掉 critic」的可视化）：**
- prompt 1 的 reward 全是 ~5.0（**绝对值远高于** prompt 0 的 ~0.5），但归一化后它的 advantage（`±0.45 / ±1.34`）和 prompt 0 的 advantage（`±0.85 / ±1.13`）**在同一量级**。**组均值 baseline 自动消掉了「prompt 难度 / reward 尺度」的差异**——这正是 PPO 里 value 网络要费力学的东西，GRPO 用一行统计就替代了。
- 每组 advantage 之和恒为 0（减了组均值）——对应第 3 章「baseline 不引入 bias」。
- 组内 reward 排序直接变成 advantage 符号：reward 高于组均值 → 正 advantage（这条 response 被鼓励）；低于组均值 → 负 advantage（被压制）。**「比同组别人答得好就加强」——这就是 group-relative 的字面含义。**

> ⚠ 一个边界（§5 会展开）：如果某个 prompt 的 G 个 response **reward 完全相同**（比如题太简单全做对、或太难全做错），`std_g → 0`、`mean_g = r_i`，则 advantage 全为 0（`+1e-4` 防爆但信号≈0）——**这组样本对训练没有任何梯度贡献**。这是 GRPO 的一个真实痛点。

---

## §4 方案对比

### 4.1 横向：RLHF-PPO vs DPO vs GRPO

| 维度 | RLHF-PPO | DPO | GRPO |
|---|---|---|---|
| **范式** | online RL（actor-critic） | offline 偏好分类 | online RL（critic-free） |
| **需要的模型** | policy+ref+reward+value（4） | policy+ref（2） | policy+ref+reward（3，reward 常是规则） |
| **需要 reward model** | 是（单独训） | **否**（隐式 reward = policy 自己） | 看情况：RLVR 用规则函数；也可用 RM |
| **训练时采样** | 是（贵） | **否**（吃固定偏好集） | 是（每 prompt 采 G 条） |
| **数据形态** | 偏好排序 → 训 RM | **成对偏好** (y_w, y_l) | prompt + 可打分的 reward |
| **显存/算力** | 最高 | **最低** | 中（比 PPO 省 value 那一半） |
| **稳定性/调参** | 玄学（第 4 章 §5 全套） | 较稳但有 likelihood displacement | 较稳但有 reward hacking / std 偏置 |
| **能力上限** | **最高**（reward 可任意复杂、可泛化） | 受限于偏好数据覆盖 | 受限于 reward 可验证性 |
| **典型代表** | InstructGPT / ChatGPT | Zephyr / 大量开源对齐模型 | DeepSeek-R1 / 数学&代码推理 |

### 4.2 具体场景：到底该选哪个

**选 DPO，当：**
- 你已有（或能廉价造出）**高质量成对偏好数据**（人标，或用更强模型当 judge 造的 AI 偏好 = RLAIF）。
- 对齐目标是**风格 / 安全 / 有用性 / 无害性**这类「软」偏好——没有客观对错，只有「人更喜欢哪个」。
- 你**没有 RL infra**、想用一套接近 SFT 的训练流程快速出活。这是 DPO 最大的吸引力：**它就是个分类任务，几乎和监督学习一样好上手**。
- 反例（不该用 DPO）：reward 可程序化验证（数学/代码），且你想让模型**探索出训练集里没有的解法**——DPO 吃固定数据集，学不到数据外的东西。

**选 GRPO / RLVR，当：**
- reward **可程序化验证且廉价**：数学题对答案、代码过单元测试、格式 / 长度约束、工具调用成功与否。这类 reward 没有「RM 被 hack」问题（除非验证器本身有漏洞）。
- 目标是**提升推理 / 解题能力**，需要模型**在线探索**（采一组 → 强化答对的那条 → 慢慢学会推理）。DeepSeek-R1 的 long-CoT 就是这么 RL 出来的。
- 你能接受 online 采样的成本，但**养不起额外的 value 网络**（显存预算紧）。
- 反例：偏好是「软」的、没有可验证 reward → 还是 DPO 或 PPO+RM。

**选 PPO-RLHF，当：**
- 你要的对齐**复杂且需要泛化**：reward model 能从有限标注泛化到没见过的 prompt，这是固定偏好集（DPO）和规则函数（GRPO）都做不到的。
- 你有成熟的 RL infra 团队，愿意付出调参成本换**最高的可控性和上限**。
- 你需要 reward shaping 的灵活性（多个 reward 组合、过程奖励 process reward 等）。
- 现实：**纯 PPO-RLHF 工程门槛最高**，2024 年后很多团队转向 DPO（省事）或 GRPO（推理强），但**能力上限和灵活性 PPO 仍是天花板**。

> 【方向性判断，非定论】「DPO vs PPO 谁强」学界仍有争论。有工作（如《Is DPO Superior to PPO for LLM Alignment?》）实证 **PPO 在多个 benchmark 上仍优于 DPO**，认为 DPO 的 offline 性质和 likelihood displacement 限制了上限；也有大量实践证明 DPO 性价比极高。**结论取决于你的数据质量、任务类型、infra 能力，没有放之四海的答案。** 标【待核】：具体哪个在你的任务上更好，必须自己 A/B。

### 4.3 同一家族内的变体（知道有这些就行）

- **DPO 系**：IPO（修 DPO 在确定性偏好下的过拟合）、cDPO / robust DPO（label smoothing 应对噪声偏好）、KTO（不需要成对，单条「好/坏」标注即可，更省标注）、ORPO（连 reference 都不要，SFT+偏好一步到位）、SimPO（去掉 reference，用平均 logp 当隐式 reward）。
- **GRPO 系**：Dr.GRPO（去掉 std 归一化和 length 归一化，修 GRPO 的长度/难度偏置）、DAPO（clip-higher 非对称裁剪 + dynamic sampling 过滤全对/全错组）、RLOO（leave-one-out baseline，比 GRPO 更早的 critic-free 方案）、REINFORCE++。
- **共性**：这些变体几乎都在 TRL 里以 `loss_type` 或 config flag 的形式存在（§2.2 / §2.3 见过 loss_type 列表）。**理解了 DPO/GRPO 主干，这些变体都是「在某个具体痛点上的小手术」，按需查。**

---

## §5 失败模式与真坑

RL 对齐「训练不稳」是出了名的。下面是三个算法各自的高频真坑，**每个给现象 → 根因 → 排查**。

### 5.1 DPO：likelihood displacement（chosen 概率不升反降）

- **现象**：训练 loss 在降、reward accuracy 在涨，**但模型在 chosen response 上的绝对 log-prob 反而下降**；严重时生成质量变差、甚至输出原本既不在 chosen 也不在 rejected 里的怪东西（"unintentional unalignment"）。
- **根因**：DPO 的梯度（§1.2）**只优化 margin（y_w 与 y_l 的相对差），不约束绝对概率**。当 y_w 和 y_l **语义/结构相似**时，「压低 y_l」的梯度会**溢出（spill over）到相似的 y_w 上**，导致两者概率双双下降、只是 y_l 降得更快——margin 照样变大、loss 照样降，但 chosen 的绝对似然塌了。

> 【失败模式，WebSearch 核实：arXiv:2410.08847《Unintentional Unalignment: Likelihood Displacement in DPO》】该工作系统刻画了这一现象：DPO 可能让 chosen 和 rejected 的概率**同时下降**，根因是相似 response 间的梯度耦合；并指出这会把概率质量挪到「既非 chosen 也非 rejected 的第三方输出」上，造成意料外的 unalignment。

- **排查 / 缓解**：
  1. **监控 chosen 的绝对 logp**（不只是 margin / accuracy）。`policy_chosen_logps` 持续下降是红旗。
  2. 在 DPO loss 里**加一个 SFT 项**（`loss_type` 里 mix `"sft"`，或 RPO/DPO+NLL）——显式约束「别把 chosen 的概率压下去」。这是最常用的缓解。
  3. **过滤掉「y_w 与 y_l 过于相似」的偏好对**（编辑距离 / embedding 相似度过滤），从源头减少梯度溢出。
  4. 调大 β（更贴 reference）或减小 lr / 训练步数（DPO 极易过拟合偏好集）。

### 5.2 DPO：过拟合偏好集 / reference 选错

- **现象**：在偏好集上 reward accuracy 接近 100%，但实际生成质量、通用能力下降（reward over-optimization 的离线版）。
- **根因**：DPO 是离线的，偏好集**有限且有分布偏差**；β 太小时模型可以无约束地偏离 reference 去拟合这些有限样本，过拟合 + 灾难性遗忘。另一个常见错误：**reference 用错**——DPO 要求 reference 是「policy 训练前的那个 SFT 模型」，若用了别的模型当 ref，隐式 reward 的语义就错了。
- **排查**：监控**通用能力 benchmark**（不只偏好集指标）；β 从 0.1 起、不稳就调大；确认 `ref_model` = SFT checkpoint。早停。

### 5.3 GRPO：reward hacking（reward 涨但答案变烂）

- **现象**：训练 reward 持续上升，但**人看实际输出在退化**——模型学会了「刷验证器」而非「真把题做对」。例：奖励函数只看「答案里有没有 `\boxed{}`」，模型就疯狂输出 `\boxed{}` 而不管对错；或奖励「长度长 = 推理充分」，模型就灌水。
- **根因**：**reward 函数有漏洞**。GRPO 不像 RM 那样有泛化能力，它**死板地优化你写的规则**，规则的任何漏洞都会被 RL 精准利用（RL 是最好的「漏洞挖掘器」）。
- **排查 / 缓解**：
  1. **盯住 reward 和人评的背离**——定期人工抽看高 reward 样本。reward 涨、人评跌 = 经典 reward hacking。
  2. **把 reward 写严**：答案对错用严格 parser、加格式/长度的反作弊项、多个 reward 函数组合（TRL 支持 `reward_funcs` 列表 + 权重）。
  3. 保留 / 调大 **KL 惩罚**（`β`），把模型拴在 reference 附近，限制它跑去 hack 的自由度。

### 5.4 GRPO：std 归一化的偏置 + 全对/全错组无梯度

- **现象（偏置）**：模型对「简单题」和「难题」的更新强度不合理——简单题（reward 方差小）被 `/std` 放大、难题被压缩，引入长度 / 难度偏置。
- **现象（无梯度）**：很多组的 G 个 response **reward 全相同**（题太简单全对、或太难全错），如 §3.2 末尾所示，这些组 advantage 全为 0、**白采样、零梯度贡献**，浪费算力。
- **根因**：前者是 DeepSeekMath 原版 `/std` 归一化的副作用（Dr.GRPO 论文指出的）；后者是「组内对比」机制的固有局限——没有差异就没有学习信号。
- **排查 / 缓解**：
  1. 偏置 → 试 `scale_rewards="none"`（Dr.GRPO，不除 std）或 `"batch"`。
  2. 无梯度组 → **dynamic sampling**（DAPO 的做法）：过滤掉全对 / 全错的组，只保留有 reward 差异的组再算梯度；或动态调整题目难度让组内有区分度。
  3. 监控 `clip_ratio` / `is_std_zero` 比例（TRL 有记这些 metric），全零组占比过高就该调数据难度。

### 5.5 通用：两种 KL 仍然容易混（承接第 4 章）

- **现象**：KL 系数调不对，模型要么跑飞（胡说刷 reward）要么被 reference 锁死不动。
- **根因**：和第 4 章一样，混淆「policy-vs-reference KL（对齐约束，防跑偏）」与「new-vs-old KL（算法稳定，PPO/GRPO 监控用）」。注意 **DPO 里的 β 是 reference KL 的隐式系数**（控制偏离 ref 多远），**GRPO 里 `β·per_token_kl` 也是 reference KL**——两者都是「KL ①」，调它控制「能离 SFT 多远」。
- **排查**：明确你调的是哪个 KL。DPO/GRPO 的 β 都是「vs reference」；GRPO 的 ε（clip）才是「vs old policy」那条线。

---

## §6 生产落地与算法选型

> 前一章把「PPO 怎么映射到 LLM」讲清了。本章是对齐技术的落地收口：**给定你的处境，这条 RLHF→DPO→GRPO 的链上该站哪一站**，以及工程上的真实注意点。

### 6.1 一张决策图（怎么选）

```
你有什么数据 / reward？
├─ 成对偏好 (y_w, y_l)，软目标（风格/安全/有用）
│   └─ 没 RL infra / 要快 → DPO（或 KTO 若只有单条好坏标注 / ORPO 若想省 reference）
│       └─ 出现 likelihood displacement → 加 SFT 项 / 过滤相似对（§5.1）
│
├─ 可程序化验证的 reward（数学对错 / 代码测试 / 格式）
│   └─ 要提升推理、能在线采样 → GRPO / RLVR（DeepSeek-R1 路线）
│       └─ reward hacking → 写严 reward + 调大 KL（§5.3）；无梯度组 → dynamic sampling（§5.4）
│
└─ 复杂 reward、需泛化、要最高上限、有 RL infra 团队
    └─ PPO-RLHF（reward model + PPO，第 4 章）—— 上限最高但最重
```

### 6.2 RLAIF：用 AI 当 labeler，喂给 DPO/GRPO

> 【fact，业界主流】人标偏好贵且慢。**RLAIF（RL from AI Feedback）** 用一个更强的模型（或同模型 + 一套 rubric）当 judge 来生成偏好标注 / reward，再喂给 DPO 或 GRPO。Anthropic 的 Constitutional AI 是早期代表（用一套「宪法」原则让模型自我批判 + 自我修正生成偏好）。

实践含义：**DPO/GRPO 的「数据 / reward」这一头，越来越多是 AI 生成的**。这把对齐的瓶颈从「人标产能」转移到「judge 模型质量 + rubric 设计」。坑也随之转移：judge 的偏见会被 DPO 原样学进去（§5.2 的过拟合在 AI 偏好上同样成立，且偏见更隐蔽）。

> 【待核】用 Claude 系模型当 judge / 生成偏好或 reward 时，judge prompt（rubric / few-shot / 输出格式）的设计对最终对齐效果影响极大，且属于「LLM-judge」范式——具体 prompt 结构、是否需要 chain-of-thought 评分、如何降低位置偏置等，**应查 Anthropic 官方 LLM-as-judge 实践，不要凭记忆**。本章不展开 judge 实现细节。

### 6.3 工程真实注意点（落地踩过的）

1. **reference 模型的显存**：DPO/GRPO 都要 reference。DPO 可以**预先离线算好 ref logp 并缓存**（reference 不变），训练时不必载 ref 模型——TRL 支持 `precompute_ref_log_probs`，省一份模型显存。GRPO 因为 policy 在变、KL 要实时算，通常得在线载 ref（或用 LoRA：ref = base，policy = base+adapter，共享底座）。
2. **DPO 的 batch 构造**：chosen 和 rejected 要在同一 batch 里成对（§2.2 的 `chunk(2)` 依赖这个布局），且 padding / mask 要正确——logp 是 sum over completion tokens，prompt 部分必须 mask 掉，否则 prompt 的 logp 污染信号。
3. **GRPO 的采样成本**：每个 prompt 采 G 条（典型 G=8~16），**采样占了训练大头**。生产上几乎必用 **vLLM** 加速 generation（TRL GRPO 集成了 vLLM），否则慢到不可用。这也是 §2.4 里源码塞满 vLLM importance sampling 修正的原因——采样和训练用不同引擎，分布有细微差异，要用 importance sampling 校正。
4. **lr 要很小**：和第 4 章 RLHF-PPO 一样，对齐是在已训好的大模型上微调，lr 通常 `1e-6 ~ 5e-6` 量级（DPO 常见 `5e-7 ~ 5e-6`），比从头训小几个数量级。
5. **评估别只看 in-domain 指标**：DPO 的 reward accuracy、GRPO 的 train reward 都容易「看起来很好」（§5.2/§5.3）。**必须配通用能力 benchmark + 人评 / LLM-judge 的 held-out 评估**，否则你优化的是指标不是能力。

### 6.4 这条链的全局图（收束）

把六章串起来：第 1-3 章给了 RL 的地基（MDP / 价值 / policy gradient / baseline）；第 4 章 PPO 是「能稳定优化任意 reward」的主力引擎；第 5 章是 off-policy / 探索的旁支。**本章是把这套 RL 机器对准「人类偏好」这个特殊 reward**：

- **RLHF-PPO** = 直接用第 4 章的 PPO 引擎，reward 来自一个学出来的 RM，KL 拴住 SFT。最通用、最重。
- **DPO** = 发现这个特定问题有闭式解，于是**整台 RL 机器都不用启动**，退化成分类。最轻、最易用，但离线、有 displacement。
- **GRPO** = 在「reward 可验证」的子场景里，**保留 RL 在线探索的好处，但用第 3 章的 baseline 定理把第 4 章的 critic 替换成组内统计**。推理任务的当红主力。

**没有谁淘汰谁**：它们是同一套 RL 原理在「不同数据形态 / 不同 reward 性质 / 不同算力预算」下的三种投影。吃透这三种投影背后是同一个「KL 约束下的偏好优化」，你就真正把对齐用的 RL 吃透了。

---

## 五件套（章末固定）

### 一、概念题（检验是否真懂）

1. DPO 论文标题说「你的语言模型本身就是一个 reward model」。用 §1.2 的 Eq.5 解释这句话——任何 policy 对应的隐式 reward 是什么？为什么说它「等价于」一个 reward model？
2. DPO 推导里，配分函数 `Z(x)`（算不出来的那个）是在哪一步、因为什么神奇地消掉了？如果偏好模型不是 Bradley-Terry（两两之差过 sigmoid），它还会消掉吗？
3. GRPO 用「组均值当 baseline」替代 PPO 的 value 网络。从第 3 章的 baseline 定理出发，解释为什么「减去一个不依赖具体 action 的量」能降方差却不引入 bias。组均值满足这个条件吗？
4. RLHF-PPO 把 KL 惩罚混进 **reward**（`r − β·KL`），GRPO 把 KL 直接加进 **loss**（`+β·KL_k3`）。这两种放法在梯度上等价吗？各有什么工程上的好处？
5. 为什么说 DPO 是 **off-policy / offline**、GRPO 是 **on-policy / online**？这个区别如何决定了「模型能不能学到训练数据里没有的解法」？

### 二、推导题

1. 从「KL 约束下最大化 reward」`max_π E[r] − β·KL(π‖π_ref)` 出发，用 Lagrangian / 变分法推出最优 policy 的闭式解 `π_r ∝ π_ref·exp(r/β)`（DPO Eq.4）。提示：对 π 求变分、加上归一化约束。
2. 把 DPO Eq.5 的隐式 reward 代入 Bradley-Terry `P(y_w≻y_l)=σ(r_w−r_l)`，**手推出 DPO loss Eq.7**，并明确指出 `β·log Z(x)` 在哪一步抵消。
3. 推导 DPO 梯度（§1.2 那个式子），并解释系数 `σ(r̂_l − r̂_w)` 为什么实现了「当前排错时权重大、排对时权重小」。
4. 证明 GRPO 用的 k3 KL 估计器 `π_ref/π_θ − log(π_ref/π_θ) − 1` 恒 ≥ 0 且是 `KL(π_θ‖π_ref)` 的无偏估计（提示：`x − log x − 1 ≥ 0`；对 `x = π_ref/π_θ` 在 `π_θ` 下取期望）。

### 三、代码题（扩展 demo）

1. **给 §3.1 的 DPO demo 复现 likelihood displacement**：把 `true_reward` 改成让两个高 reward 候选「语义相似」（在 demo 里可建模为：给 policy logits 加一个让候选 0 和候选 1 强耦合的参数化，比如它俩共享一部分 logit）。观察是否出现「chosen 候选的绝对概率随训练下降」。然后加一个 SFT 项（`+λ·(−logp[w])`）看能否救回来。复现 §5.1。
2. **把 §3.1 改成 IPO loss**：DPO 的 `−logσ(β·margin)` 换成 IPO 的 `(margin − 1/(2β))²`（squared loss）。对比两者在「确定性偏好」（同一对 y_w/y_l 反复出现）下的行为——验证 IPO 论文「DPO 在确定性偏好下会把 logratio 推向无穷而过拟合」的说法。
3. **给 §3.2 的 GRPO demo 加一个完整的 mini 训练 loop**：在 K 选 1 bandit 上（复用 §3.1 的 softmax policy），实现「采样 G 条 → 用规则 reward 打分 → 算 group advantage → 用 `−Â·logp` 做 REINFORCE 式更新（先不加 clip）」。验证 policy 收敛到高 reward 候选。再加上 clip 和 KL 惩罚，看曲线变化。
4. **实现「全对/全错组无梯度」的复现 + dynamic sampling 修复**：在第 3 题基础上，构造一些「G 条 reward 全相同」的组，确认它们 advantage=0、对更新无贡献；然后实现 DAPO 式过滤（丢掉这些组），对比训练效率。

### 四、排查题（给现象找根因）

1. 你的 DPO 训练日志：loss 在降、reward accuracy 升到 95%，但放出来的模型生成质量明显比 SFT 还差，偶尔输出奇怪的第三方内容。最可能是什么失败模式？该监控哪个**额外指标**确认？给两个缓解动作。
2. GRPO 训练 reward 一路上涨很漂亮，但人工抽看高 reward 样本发现模型在「灌水 / 套模板刷分」。这是什么问题？根因在哪一侧（policy 还是 reward 函数）？怎么改？
3. 你的 GRPO 训练里，日志显示大量 batch 的 `std ≈ 0`、对应组 advantage 全 0，GPU 利用率高但收敛极慢。诊断原因，给出两种修复（一个改数据、一个改算法）。
4. 同事的 DPO 跑出来效果很差，你发现他用了一个**和 SFT 不同的开源模型当 reference**。为什么这会出问题？从隐式 reward 的定义解释。

### 五、思辨题（专家级判断）

1. DPO 把 RLHF「退化成分类」，看起来是纯赚（更简单 + 更稳 + 更省）。但它是 **offline** 的。从「模型能否探索出数据集外的新行为」角度，论证为什么在「需要提升推理 / 解题能力」的任务上，online 的 GRPO 可能根本性地优于 DPO——这是否说明「DPO 替代 RLHF」这个叙事在某些场景是错的？
2. GRPO 砍掉了 critic，用组均值当 baseline。critic 的本质作用是「per-state 的、学出来的、可泛化的 baseline」，组均值是「per-prompt 的、即时的、不泛化的 baseline」。在什么条件下后者足够好（甚至更好）？在什么条件下你会怀念 critic？（提示：组大小 G、reward 稀疏度、prompt 间的可迁移性。）
3. RLAIF（用 AI 当 labeler）让对齐数据可以无限生成。但 §5.2 指出 DPO 会原样学进 labeler 的偏见。如果用模型 A 当 judge 去对齐模型 B，再用 B 去 judge 对齐 C……这条链会收敛到「真正的人类偏好」还是「A 的偏见的不动点」？这对「用 AI 对齐 AI」的可扩展监督（scalable oversight）路线意味着什么？
4. 本章三个算法都靠一个 `β·KL(π‖π_ref)` 把模型拴在 SFT 附近。这个 reference anchor 是对齐稳定性的来源，但也是能力上限的来源——模型永远不能离 SFT「太远」。如果某个任务的最优 policy 本来就离 SFT 很远（比如全新的推理风格），KL anchor 是不是反而成了枷锁？ORPO/SimPO 这类「去掉 reference」的方法在赌什么？什么时候这个赌注危险？

---

## 附：本章信源（均经 WebFetch / WebSearch 实际核实）

- **真实源码 · TRL DPO trainer**：`huggingface/trl@trl/trainer/dpo_trainer.py`（raw.githubusercontent.com/huggingface/trl/main/...）——§2.2 DPO loss（`chosen_logratios` / `delta_score` / `-F.logsigmoid(β·delta_score)`，行 1245-1283）、隐式 reward / reward_accuracies（行 1171-1175）。
- **真实源码 · TRL GRPO trainer**：`huggingface/trl@trl/trainer/grpo_trainer.py`——§2.3 group advantage（行 2177-2199）、§2.4 clip surrogate + k3 KL（行 2549-2620）。
- **真实源码 · TRL utils**：`huggingface/trl@trl/trainer/utils.py`——§2.1 `selective_log_softmax`（行 436-467）。
- **真实源码 · TRL DPOConfig**：`huggingface/trl@trl/trainer/dpo_config.py`——默认 `beta=0.1` / `loss_type=["sigmoid"]` / `label_smoothing=0.0`。
- **DPO 论文**：Rafailov et al. 2023《Direct Preference Optimization》arXiv:2305.18290（ar5iv.labs.arxiv.org/html/2305.18290）——Eq.4（最优 policy）、Eq.5（隐式 reward）、Eq.7（DPO loss）、§4 梯度、作者与动机。
- **GRPO 论文**：Shao et al. 2024《DeepSeekMath》arXiv:2402.03300（ar5iv）——J_GRPO 目标、group-relative advantage `(r−mean)/std`、k3 无偏 KL 估计器、「砍 critic 省显存」动机。
- **RLHF 奠基论文**：Ouyang et al. 2022《InstructGPT》arXiv:2203.02155（ar5iv）——三段式流水线、reward model pairwise loss、PPO-ptx 目标（含 β·KL + γ·ptx）。
- **DPO 失败模式**：arXiv:2410.08847《Unintentional Unalignment: Likelihood Displacement in DPO》（WebSearch 核实）——§5.1 likelihood displacement 的机制与刻画。

> 标注约定回顾：【真实源码 repo@path/行号】= 逐字摘自源码；【真实公式/设计考古，WebFetch 核实：来源】= 核心公式 / 结论经实际抓取核对；【示意，非逐字】= 教学改写的最小可运行版（两个 demo 即是，但其**输出为本文档真实运行抓取**）；【fact】= 业界共识；【待核】= 未能从一手信源完全确认、需读者自行核实的点（如 DPO vs PPO 孰优、LLM-judge prompt 设计、InstructGPT 具体超参）。
