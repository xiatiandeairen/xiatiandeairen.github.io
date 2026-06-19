---
title: "Capstone:从零跑一个对齐 / 推理训练闭环"
slug: "3-16"
collection: "ai-research-compass"
group: "强化学习专家课程"
order: 3016
summary: "这一章把前面所有零件——PPO 的 clip 目标(06)、DPO 的离线偏好优化(09)、RLVR 的可验证奖励(10)、GRPO 的组内基线(11)、稳定性调参(14)——拼成一条你能在单卡上从头跑通、并写出实验报告的完整管线。读完你不再是\"知道公式\",而是\"知道每条曲线该长什么样、走歪了该拧哪个旋钮\"。"
topics:
  - "AI 研究"
tags: []
createdAt: "2026-06-17T15:20:26.000Z"
updatedAt: "2026-06-17T15:20:26.000Z"
---
> 这一章把前面所有零件——PPO 的 clip 目标(06)、DPO 的离线偏好优化(09)、RLVR 的可验证奖励(10)、GRPO 的组内基线(11)、稳定性调参(14)——拼成一条**你能在单卡上从头跑通、并写出实验报告**的完整管线。读完你不再是"知道公式",而是"知道每条曲线该长什么样、走歪了该拧哪个旋钮"。

## 1. 这章要交付什么

前面的章节是分立的。06 章给你 PPO 的目标函数,09 章给你 DPO 的封闭解,11 章给你 GRPO 怎么去掉 critic,14 章告诉你训练为什么会崩。但一个真正的从业者,价值不在于会背这些公式,而在于:**给你一个任务、一张显卡、两天时间,你能不能让模型在可验证指标上真的变好,并且讲清楚它为什么变好。**

这一章就是那次"真的跑一遍"。我们选一个**可自动验证对错**的小任务,用一个**单卡能跑的小模型**,实现一条端到端管线:

```
准备 prompt 集 + 验证器
        ↓
对每个 prompt 采样一组 G 个回答          ← 在线 rollout
        ↓
验证器给每个回答打分 rᵢ ∈ {0,1} 或标量    ← RLVR,无需训练 reward model
        ↓
组内归一化得到优势 Aᵢ                     ← GRPO 的核心,替代 critic
        ↓
带 KL 惩罚的 clip 策略梯度更新            ← PPO 的目标 + 14 章的稳定性约束
        ↓
监控 KL / 平均奖励 / 熵 / 验证集正确率     ← 评估纪律,防 Goodhart
        ↓
对照 SFT baseline,写实验报告
```

**为什么是这条路线而不是 RLHF 全家桶?** 经典 RLHF(收集人类偏好 → 训 reward model → PPO)有三个移动部件(reward model、critic、policy),每个都能单独崩,联调成本极高,不适合作为第一次从零跑通的目标。我们把它**降复杂度**到两个关键简化:

- **用可验证奖励(RLVR)代替 reward model**:任务对错由代码判定(算术结果对不对、字符串变换是否正确、逻辑题答案是否匹配),奖励无噪声、无法被 reward model 漏洞反向利用,排除了一整类 reward hacking。
- **用 GRPO 代替 PPO 的 critic**:GRPO 用"同一 prompt 下一组回答的平均奖励"当基线,省掉一个需要单独训练、单独调参、单独崩溃的 value network。

这两步把"三个会崩的部件"压到"一个策略网络 + 一个确定性验证器",是任何人从零起步最该选的配置。本章默认走 **RLVR + GRPO 在线路线**,同时给出 **DPO 离线路线**的完整实现和"何时用哪个"的判据。

## 2. 预备:把任务变成可验证的 RL 问题

### 2.1 任务选择:可验证性是第一约束

RL 训练的成败,80% 在你按下 `train()` 之前就决定了——决定在**任务和奖励的定义**上。新手最常见的死法不是代码 bug,是**奖励定义里藏着一条比"真做对任务"更便宜的捷径**,模型必然找到它。

选任务的硬标准:**对任意一个回答,你能不能写一段不超过 20 行的确定性代码,判定它对不对?** 满足这条的小任务:

- **小学算术**:`prompt = "37 × 24 = ?"`,验证器解析模型输出里的最终数字,与 `37*24` 比较。
- **字符串变换**:`prompt = "reverse: hello"`,验证器比对输出是否等于 `"olleh"`。
- **简单逻辑/计数**:`prompt = "How many 'r' in 'strawberry'?"`,验证器比对数字。

本章用**两位数乘法**作主任务(难度刚好:SFT 小模型能做对一部分但远非全对,有提升空间;验证零歧义)。

### 2.2 验证器:奖励函数的真身

在 RLVR 里,**验证器就是奖励函数**。它必须满足三个性质,缺一个都会埋雷:

1. **确定性**:同一回答永远同一分数。否则优势估计的方差里混入奖励噪声,无法收敛。
2. **难以走捷径**:奖励只奖励"真的解对",不奖励"长得像解对"。
3. **解析鲁棒**:模型输出格式会乱七八糟,验证器要能从噪声里抠出答案,而不是因为格式不符就误判。

下面是骨架。注意 `extract_answer` 的设计——这是 RLVR 实践中最容易被低估的一环:

```python
import re

def extract_answer(text: str):
    """从模型自由文本里抠出最终数字答案。
    实践要点:取'最后一个'数字,因为模型常把中间步骤的数字也写出来,
    最终答案通常在末尾(对应 'the answer is X' / '= X' 模式)。"""
    nums = re.findall(r'-?\d+', text.replace(',', ''))
    if not nums:
        return None
    return int(nums[-1])

def verifier(prompt: str, response: str, gold: int) -> float:
    """RLVR 奖励函数:对返回 1.0,错或解析失败返回 0.0。
    可选:加一个微小的格式分,鼓励模型给出可解析的输出。"""
    pred = extract_answer(response)
    if pred is None:
        return 0.0            # 解析失败 = 答错,不给同情分
    return 1.0 if pred == gold else 0.0
```

**坑(reward shaping 反噬)**:很多人忍不住加 shaping——比如"答案接近就给部分分",`reward = 1 - |pred-gold|/gold`。对乘法这是灾难:模型会学会输出"接近但不等于"的数(因为精确解很难、近似解的期望奖励反而高),最终正确率不升反降。**纯 0/1 奖励虽然稀疏,但它诚实**。只有当任务确实是连续度量(如代码通过的测试比例)时,才用连续奖励。

**格式分要不要加?** 可以加一个很小的(如 +0.1)"输出里能解析出数字"的分,帮助早期模型脱离"完全不给数字"的局部最优。但**格式分必须远小于正确分**,否则模型会去优化格式而非正确性——这是一次微型 Goodhart,后面 5.3 节展开。

### 2.3 SFT baseline:没有它你不知道 RL 有没有用

**铁律:任何 RL 结果都必须对照一个不做 RL 的 baseline。** 否则你无法区分"RL 起作用了"和"这个模型本来就行/不行"。最小 baseline 是**直接评估基座模型(或一个轻量 SFT 后的模型)在验证集上的正确率**。我们记这个数为 `acc_sft`,RL 的全部意义就是把验证集正确率从 `acc_sft` 推上去,且不靠作弊。

## 3. 数学核心:从 PPO 到 GRPO 的优势估计

这一节是全章的理论重心。我们从策略梯度出发,推出为什么需要基线、为什么 GRPO 的组内基线是合法的、它和 PPO 的 critic 基线在数学上是什么关系。**这些推导你在 06 和 11 章见过零件,这里把它们接成一条逻辑链并算清方差。**

### 3.1 策略梯度与基线:为什么可以"白减一个数"

策略 πθ 在 prompt `q` 下生成回答 `o`,获得奖励 `R(q,o)`。我们要最大化期望奖励 `J(θ) = E_{q,o~πθ}[R(q,o)]`。策略梯度定理给出:

```
∇θ J(θ) = E_{q, o~πθ}[ R(q,o) · ∇θ log πθ(o|q) ]
```

推导(对单个 q,省略 q 记号):

```
J(θ) = Σ_o πθ(o) R(o)
∇J  = Σ_o ∇πθ(o) R(o)
    = Σ_o πθ(o) [∇πθ(o)/πθ(o)] R(o)        # 乘除同一个 πθ(o)
    = Σ_o πθ(o) [∇ log πθ(o)] R(o)          # ∇log f = ∇f / f,这一步是 log-trick
    = E_{o~πθ}[ R(o) ∇ log πθ(o) ]
```

**这个估计无偏但方差极大**:`R` 在我们的任务里是 0/1,但乘上 `∇log πθ` 后,不同样本的梯度幅度天差地别。降方差的标准武器是**减去一个基线 b**(任何不依赖于动作 `o` 的量):

```
∇J = E_{o~πθ}[ (R(o) - b) ∇ log πθ(o) ]
```

**为什么减 b 不引入偏差?** 关键引理:`E_{o~πθ}[ b · ∇ log πθ(o) ] = 0`。证明:

```
E_o[ b ∇ log πθ(o) ] = b Σ_o πθ(o) ∇ log πθ(o)
                     = b Σ_o πθ(o) · [∇πθ(o)/πθ(o)]
                     = b Σ_o ∇πθ(o)
                     = b ∇ Σ_o πθ(o)
                     = b ∇ (1)              # 概率归一化,Σπ=1
                     = 0
```

所以减去任何与 `o` 无关的 `b`,**期望不变(无偏)**,但方差可以大幅下降。这就是"优势" `A = R - b` 的合法性来源。**最优基线(最小化方差)是接近 `E[R]` 的值**——这正是 GRPO 要估的东西。

### 3.2 GRPO:用组内均值当基线

PPO 用一个**学出来的 value network** `V(q)` 当基线,`A = R - V(q)`。代价:多一个网络,多一套训练,多一个崩溃点(value 估不准 → 优势带偏 → 策略乱跑)。

GRPO 的洞察:**对同一个 prompt `q`,直接采 G 个回答 `{o₁,...,o_G}`,用这一组的平均奖励当基线。** 这就是个蒙特卡洛估计的 `E[R|q]`,完全不需要训练。组内优势:

```
对 prompt q 采样 G 个回答,得奖励 r₁,...,r_G
μ = mean(r₁..r_G)
σ = std(r₁..r_G)
Aᵢ = (rᵢ − μ) / (σ + ε)        # 组内标准化优势
```

**直觉(讲人话)**:在同一道题的 G 个答案里,比平均水平好的答案优势为正(被鼓励),比平均差的为负(被抑制)。模型学的是"在这道题上,什么样的答案比我自己的平均水平更好"——一个**自我对比**的信号,不需要外部 value 估计。

**为什么除以 σ?** 这是把不同 prompt 的优势缩放到可比尺度。简单题(大家都对,σ→0)和难题(对错参半,σ 大)的原始优势量级不同;除以 σ 让每道题对梯度的贡献均衡。**但这里藏着 GRPO 的一个已知偏差**,5.2 节专门讲。

**组内基线的无偏性**:`μ` 是从同组样本算的,严格说它和 `oᵢ` 相关(`oᵢ` 参与了 `μ` 的计算),所以 `A = r - μ` 不是 3.1 里那个"与动作无关的 b"。这会引入一个 O(1/G) 量级的小偏差。实践中 G≥8 时偏差可忽略,且降方差的收益远大于这点偏差——这是一个**偏差换方差**的经典权衡。**留意:这是真实存在的偏差,不是 bug,G 越大越小。**

### 3.3 方差对比:为什么组内基线有效

设单个 prompt 的奖励方差为 Var[R]。

- **无基线**:梯度估计方差 ∝ E[R²] · E[‖∇log π‖²]。对 0/1 奖励,E[R²]=E[R]=p(正确率),量级被 p 主导。
- **组内基线**:优势 `A=R-μ` 的方差 ≈ Var[R] = p(1-p)。当 p 接近 0 或 1 时,p(1-p) → 0,**优势趋于零,梯度自然变小**——模型在"几乎全对"或"几乎全错"的题上不再瞎更新。

这个性质很重要:**GRPO 自动给"信息量低"的题降权**。一道全对的题(σ=0,所有 A=0)对梯度零贡献,这是对的——模型已经会了,不需要再学。**坑**:如果一个 batch 里大量 prompt 都全对或全错,有效梯度信号会非常稀疏,训练看起来"在跑但不动"。解决办法是**课程化采样**(优先采 p≈0.5 的题)或**过滤掉组内全对/全错的 prompt**(很多 GRPO 实现里叫 dynamic sampling)。

### 3.4 把优势接进 clip 目标

有了 `Aᵢ`,更新式直接套用 PPO 的 clipped surrogate(06 章)。对回答 `oᵢ` 的每个 token `t`,定义重要性比率:

```
ρᵢ,ₜ(θ) = πθ(oᵢ,ₜ | q, oᵢ,<ₜ) / π_old(oᵢ,ₜ | q, oᵢ,<ₜ)
```

`π_old` 是采样时那一版策略(rollout 用的)。GRPO 把整条回答的优势 `Aᵢ` 赋给该回答的每个 token(token 级共享 sequence 级优势),目标函数:

```
L_clip(θ) = E_i E_t [ min( ρᵢ,ₜ · Aᵢ ,  clip(ρᵢ,ₜ, 1−ε, 1+ε) · Aᵢ ) ]
```

**clip 在做什么(机制)**:当 `Aᵢ>0`(好答案),我们想增大 `ρ`(更可能生成它),但 clip 把 `ρ` 的上限锁在 `1+ε`,**防止一步把某个 token 概率推太高**;当 `Aᵢ<0`(坏答案),clip 锁下限 `1−ε`,防止一步把它压太狠。`min` 取的是"更保守"的那个分支。这是 PPO 稳定性的核心:**用旧策略采的数据,只信任策略没偏离太远的那部分更新**。ε 典型取 0.2。

**为什么不能直接用 ρ·A(不 clip)?** 因为 rollout 数据是 `π_old` 采的,要对 `θ` 做多步更新,`θ` 一偏离 `π_old`,重要性比率 `ρ` 就会爆炸/塌缩,梯度方差失控——这正是 importance sampling 在分布偏移大时的通病,clip 是它的工程补丁。

### 3.5 KL 惩罚:别让模型为了刷分忘了说人话

只优化 `L_clip`,模型会朝"最大化验证器奖励"狂奔,**代价是偏离原始模型、丢失语言能力、甚至坍缩成几句套话**(reward hacking 的一种)。我们加一个 KL 惩罚,把策略锚在参考模型 `π_ref`(通常是 SFT/基座那一版,冻结)附近:

```
L(θ) = L_clip(θ) − β · E[ KL( πθ(·|q) ‖ π_ref(·|q) ) ]
```

GRPO 原文用的是一个**无偏、低方差、恒正**的 KL 估计(k3 估计器,来自 Schulman 的 "Approximating KL Divergence" 笔记):

```
对每个 token,设 lr = π_ref(oₜ|·) / πθ(oₜ|·)         # 注意是 ref/θ
KL_estimate(token) = lr − log(lr) − 1
```

**为什么是 `lr − log lr − 1` 而不是直接 `log(πθ/π_ref)`?**

```
朴素估计   k1 = log(πθ/π_ref)            # 无偏,但方差大,且可正可负
平方估计   k2 = ½(log(πθ/π_ref))²         # 低方差,但有偏
k3        = lr − log(lr) − 1,  lr=π_ref/πθ
```

k3 的妙处:对任意 `lr>0`,`lr − log lr − 1 ≥ 0`(因为 `log x ≤ x−1`,取等当 `x=1`),所以**每个样本的 KL 估计恒为非负**,不会出现"KL 惩罚项变成奖励"的诡异情况;同时可以证明它无偏(`E_{πθ}[k3] = KL(πθ‖π_ref)`)。验算非负性:令 `f(x)=x−log x−1`,`f'(x)=1−1/x`,在 `x=1` 处 `f'=0` 且 `f''=1/x²>0`,故 `x=1` 是全局最小,`f(1)=0`,因此 `f(x)≥0`。**这就是为什么实现 GRPO 时 KL 项要用这个形式而不是随手写 log 比值。**

无偏性验算(关键一步):

```
E_{o~πθ}[ lr − log lr − 1 ],  其中 lr = π_ref(o)/πθ(o)
  = E_{πθ}[π_ref/πθ] − E_{πθ}[log(π_ref/πθ)] − 1
  = Σ_o πθ·(π_ref/πθ) − E_{πθ}[log(π_ref/πθ)] − 1
  = Σ_o π_ref           − E_{πθ}[log(π_ref/πθ)] − 1
  = 1 + E_{πθ}[log(πθ/π_ref)] − 1            # Σπ_ref=1;翻转 log 符号
  = E_{πθ}[log(πθ/π_ref)]
  = KL(πθ ‖ π_ref)    ✓
```

**β 的角色就是松紧带**:β 大 → 模型被死死拴在参考模型旁,学不动;β 小 → 模型放飞自我,可能 reward hack。调 β 是本章稳定性的头号旋钮,5.1 节量化。

> 补充一个常见疑惑:KL 既可以**作为损失项**(上面这样,放进梯度),也可以**只作监控**(有些实现把 KL 控制完全交给 clip + 早停,β=0)。两种都有人用。本章默认 β>0 作损失项,因为它对小模型更稳;但你应该**始终监控 KL 曲线**,无论 β 是否为 0。

## 4. 完整实现:一个能跑的 GRPO 训练循环

下面是去掉框架噪声、只留机制的核心代码。它能体现整条闭环,可以照着填进 TRL/verl 或自己的训练脚本。**为聚焦机制,这里用全参数更新 + 简化的单步 PPO inner-loop(每批数据更新一次);生产里通常 LoRA + 多 epoch inner-loop。**

### 4.1 采样 + 打分 + 算优势

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def rollout(model, tokenizer, prompts, golds, G, temperature=1.0, max_new=256):
    """对每个 prompt 采 G 个回答,验证器打分,算组内优势。
    返回展平的 (input_ids, response_mask, advantages, old_logprobs)。"""
    batch = []
    for q, gold in zip(prompts, golds):
        enc = tokenizer(q, return_tensors="pt").to(model.device)
        # 一个 prompt 复制 G 份,温度采样得到 G 个不同回答
        out = model.generate(
            **enc, do_sample=True, temperature=temperature, top_p=1.0,
            num_return_sequences=G, max_new_tokens=max_new,
        )
        rewards = []
        for seq in out:
            resp = tokenizer.decode(seq[enc.input_ids.shape[1]:],
                                    skip_special_tokens=True)
            rewards.append(verifier(q, resp, gold))   # ← RLVR 打分
        r = torch.tensor(rewards, dtype=torch.float32)
        # ★ GRPO 组内标准化优势
        adv = (r - r.mean()) / (r.std() + 1e-6)
        # 记录 old_logprob 供 clip 用(此处略去 logprob 抽取细节,见 4.2)
        batch.append((q, gold, out, adv, r))
    return batch
```

**温度必须 >0 且组内不同**:`num_return_sequences=G` + `do_sample=True` 才能在同一 prompt 下采到**有差异**的回答。如果温度太低(接近贪婪),G 个回答几乎一样 → 组内 σ≈0 → 所有优势≈0 → **零梯度,训练完全不动**。这是新手最常见的"训练不动"的根因之一(参见 5.1)。

### 4.2 GRPO 损失:clip + KL

```python
def grpo_loss(model, ref_model, input_ids, attn_mask, response_mask,
              old_logprobs, advantages, eps=0.2, beta=0.04):
    """计算 GRPO 的 token 级 clipped + KL 损失。
    response_mask: 1 表示该 token 属于回答(计入损失),0 表示 prompt(不计)。"""
    # 当前策略对每个 token 的 log 概率
    logits = model(input_ids, attention_mask=attn_mask).logits[:, :-1]
    logp = torch.gather(F.log_softmax(logits, -1), 2,
                        input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

    # 参考策略(冻结)的 log 概率,供 KL 用
    with torch.no_grad():
        ref_logits = ref_model(input_ids, attention_mask=attn_mask).logits[:, :-1]
        ref_logp = torch.gather(F.log_softmax(ref_logits, -1), 2,
                                input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

    mask = response_mask[:, 1:].float()
    adv = advantages.unsqueeze(1)                      # (B,1) 广播到每个 token

    # ★ clipped surrogate(PPO 目标)
    ratio = torch.exp(logp - old_logprobs)             # ρ = πθ/π_old
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
    pg_loss = -torch.min(unclipped, clipped)           # 取保守分支,负号=最大化

    # ★ KL 惩罚,k3 估计器:lr=π_ref/πθ, kl = lr - log lr - 1 ≥ 0
    log_lr = ref_logp - logp                           # log(π_ref/πθ)
    kl = torch.exp(log_lr) - log_lr - 1.0              # 恒 ≥ 0

    loss_tok = pg_loss + beta * kl
    # 按 token 平均(只统计 response token)
    loss = (loss_tok * mask).sum() / mask.sum()
    kl_mean = (kl * mask).sum() / mask.sum()           # 监控用
    return loss, kl_mean.detach()
```

**两个易错点**:(1)`logits[:, :-1]` 配 `input_ids[:, 1:]` 是标准的 next-token 错位对齐,错位写反会让 logprob 全错;(2)**一定要用 response_mask 把 prompt token 排除**——只对模型"生成的"token 算梯度,对 prompt token 算梯度是没有意义的(那不是策略的决策)且会污染信号。

### 4.3 主训练循环 + 监控

```python
def train(model, ref_model, tokenizer, dataset, optimizer,
          G=8, eps=0.2, beta=0.04, temperature=1.0,
          steps=1000, eval_every=50, val_set=None):
    for step in range(steps):
        prompts, golds = dataset.sample_batch()        # 一批 prompt
        batch = rollout(model, tokenizer, prompts, golds, G, temperature)

        # —— 拼成训练张量(此处省略 padding/packing 细节)——
        input_ids, attn_mask, resp_mask, old_lp, adv = collate(batch, tokenizer)

        loss, kl = grpo_loss(model, ref_model, input_ids, attn_mask,
                             resp_mask, old_lp, adv, eps, beta)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 防爆梯度
        optimizer.step()

        # ===== 监控四件套:缺一不可 =====
        mean_reward = torch.cat([b[4] for b in batch]).mean().item()   # 平均奖励
        entropy = compute_entropy(model, input_ids, attn_mask, resp_mask) # 策略熵
        log_metrics(step=step, loss=loss.item(), kl=kl.item(),
                    mean_reward=mean_reward, entropy=entropy,
                    grad_norm=grad_norm.item())

        # ===== 在干净验证集上测真实正确率(防 Goodhart)=====
        if val_set and step % eval_every == 0:
            acc = evaluate_accuracy(model, tokenizer, val_set)  # 贪婪解码,温度=0
            log_metrics(step=step, val_acc=acc)
```

**注意验证集评估用贪婪解码(temperature=0)**,而训练用温度采样。原因:你最终部署时是贪婪/低温推理,**评估指标要匹配部署条件**,而不是匹配训练时的高熵采样分布。用训练时的采样去评估,正确率会虚高。

## 5. 监控、调参与防 Goodhart——这章的方法论收尾

### 5.1 四条曲线该长什么样,以及它们打架时拧哪个旋钮

监控不是"记下来好看",而是**诊断工具**。健康的 GRPO 训练,四条曲线的典型形态:

| 指标 | 健康形态 | 异常 → 诊断 |
|---|---|---|
| **平均奖励** | 缓慢上升,有噪声 | 暴涨后骤跌 = reward hacking 或策略崩;长期平 = 无梯度(看熵/σ) |
| **KL(πθ‖π_ref)** | 缓慢上升后趋稳,维持在小值(如 <10) | 飙升 = 偏离参考太远,即将崩;贴地为 0 = β 太大,学不动 |
| **熵** | 缓慢下降(模型变确定),但不归零 | 骤降到接近 0 = 坍缩(只会输出几句话);回升 = 训练发散 |
| **验证集正确率** | 稳定上升,超过 acc_sft | 不升但训练奖励升 = **Goodhart!**(刷的是验证器漏洞不是真能力) |

**三个核心旋钮和它们的因果**:

1. **KL 系数 β**(头号旋钮)。β↑ → KL 被压低、熵保持高、训练慢但稳;β↓ → 学得快但 KL 易飙、易坍缩。**调法**:从 β≈0.04 起,看 KL 曲线。KL 持续飙升 → 加大 β;KL 贴地且奖励不动 → 减小 β。典型范围 0.001~0.1(待核:具体最优值强依赖任务和模型规模)。

2. **采样温度 T**。T 控制**探索量**,直接决定组内 σ。T 太低 → 组内回答雷同 → σ≈0 → 优势≈0 → **零梯度(最常见的"训练不动")**;T 太高 → 回答全是噪声、奖励全 0 → 也没信号。**调法**:从 T≈1.0 起;若组内 σ 经常≈0,升温(如 1.2);若回答语无伦次、奖励几乎全 0,降温。

3. **组大小 G**。G 越大 → 组内基线 `μ` 估得越准(3.2 的 O(1/G) 偏差越小)、方差越低,但**计算成本线性上升**(每步要采 batch×G 个回答)。**权衡**:G=8 是常见甜点;G<4 时基线估计噪声大、训练抖;G 很大收益递减。单卡显存紧张时,G 和 batch size 抢资源,优先保 G≥8。

**这三个旋钮会互相耦合**:升温增加探索 → KL 更容易涨 → 可能需要同步加大 β。所以**一次只动一个旋钮**,否则你分不清是谁的效果(这是消融纪律,5.4 节)。

### 5.2 GRPO 的两个已知偏差(读源码前必须知道)

GRPO 的原始优势公式 `A=(r−μ)/(σ+ε)` 在后续研究中被指出有两个系统性偏差(对应近期的 Dr.GRPO 等改进工作):

- **长度偏差**:损失对 response token 数做平均(`/mask.sum()`),会让**长回答里每个 token 的梯度被稀释**。结果:负优势的长回答受到的惩罚被摊薄,模型倾向于**把错误回答写得更长**来减小损失。缓解:有些实现改成按"固定常数"或"组内最大长度"归一,而非按各自实际长度。
- **难度偏差(除 σ 引入)**:除以 σ 会**过度放大简单题(σ 小)的优势**。一道几乎全对的题,个别答错的样本会得到巨大的负优势,反而让训练被简单题主导。缓解:Dr.GRPO 提出**去掉除以 σ 这一步**,只做 `A=r−μ`(回到 3.1 的纯基线形式,牺牲跨题尺度归一但去掉这个偏差)。

**你不必现在就上这些修正**,但跑通后如果观察到"回答越训越长""简单题主导梯度",要知道这是 GRPO 的已知特性而非你的 bug。**(待核:Dr.GRPO 的具体公式与发表细节请查原论文,本章只陈述方向性结论。)**

### 5.3 防 Goodhart:评估纪律

Goodhart 定律:**当一个度量变成目标,它就不再是好度量。** 在 RLVR 里这具体表现为:模型找到验证器的漏洞,在"训练奖励"上刷分,但"真实能力"没涨甚至下降。

防御手段(从弱到强):

1. **训练验证器 ≠ 评估验证器**:用一套独立的、更严格的评估逻辑跑验证集。训练验证器可以宽松(抠最后一个数字),评估验证器应严格(要求格式完全正确 + 答案对)。两者背离时立刻警惕。
2. **留出集分布外测试**:训练用两位数乘法,评估时加一些三位数乘法、加法,看泛化。如果只在训练分布上涨、分布外不涨,模型学的是窄技巧不是算术。
3. **人工抽样读输出**:每隔若干步,人眼读 10 条模型输出。reward hacking 在曲线上有时看不出,但读输出一眼就能发现(比如模型开始输出固定模板、复读、或用奇怪格式骗过验证器)。
4. **监控熵不归零**:熵骤降到接近 0 是坍缩的强信号——模型放弃多样性,赌一个固定输出。

**铁律**:**奖励(训练信号)和评估指标(决策依据)必须解耦。** 你用奖励训练,但你用一套模型在训练中看不到、改不了的评估指标来决定"这次 RL 到底好不好"。这是整个 RL 对齐工程里最容易被违反、违反后最致命的纪律。

### 5.4 实验报告模板:让你的结果可被检验

RL 实验极易自欺(随机种子、一次跑通就报喜)。一份合格的"RL 训练实验报告"必须包含:

```
# RL 训练实验报告

## 1. 任务 (Task)
- 任务定义:两位数乘法,prompt 格式 / 数据来源 / 训练集 N 条 / 验证集 N 条
- 验证器:确定性规则(贴出 extract_answer + verifier 逻辑)
- 评估验证器:与训练验证器的差异(防 Goodhart)

## 2. 方法 (Method)
- 基座模型:名称 + 规模(如 Qwen2.5-0.5B,待核具体型号)
- 算法:GRPO(或 DPO),关键超参 G / ε / β / T / lr / batch / steps
- SFT baseline:acc_sft = ?(必填,没有它整张报告作废)

## 3. 曲线 (Curves)
- 四条核心曲线:平均奖励 / KL / 熵 / 验证集正确率(对 step)
- 标注关键事件:何时收敛、何时出现异常、是否早停

## 4. 消融 (Ablation)
- 至少一组对照:如 β∈{0.01, 0.04, 0.1} 各跑一次,比较验证集正确率与 KL
- 一次只变一个变量;报告随机种子,最好 ≥2 个种子看方差
- 关键结论必须 ≥2 个种子 / ≥3 个样本支撑(N=1 的 100% 命中可能只是运气)

## 5. 结论 (Conclusion)
- RL vs SFT baseline:验证集正确率 acc_rl 比 acc_sft 提升多少?
- 提升是真实能力还是 Goodhart?证据:分布外测试 / 人工读样本
- 失败/局限:哪里没work,下一步改什么
```

**为什么强制消融?** 因为不做消融的 RL 结果几乎没有可信度——你不知道提升来自算法、来自某个超参的偶然好值、还是来自随机种子。**"我调了一下 β 就涨了"不是结论,"β 从 0.04→0.01 在 3 个种子上平均提升 5 个点、KL 同步升高但未失控"才是结论。**

## 6. 另一条路:DPO 离线对齐(何时用它替代 GRPO)

GRPO 是在线的(每步要现采样、现打分)。但有时你**没有验证器、只有人类偏好对**(对同一 prompt,标注者说"回答 A 比回答 B 好"),且不想在线 rollout。这时用 **DPO(09 章)**。

### 6.1 DPO 损失的来历(为什么不需要 reward model)

经典 RLHF 要先训 reward model `r(x,y)`,再 PPO 优化 `E[r] − β·KL`。DPO 的洞察:**KL 正则化的 RL 目标有封闭最优解,把 reward 反解出来代入偏好模型,reward model 就被消掉了。**

推导骨架(09 章已细讲,这里复述关键三步):

```
第1步:KL 正则 RL 的最优策略(对目标 max E[r] − β KL(π‖π_ref))有闭式解:
   π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r(x,y)/β )

第2步:反解 reward(取 log 整理):
   r(x,y) = β log( π*(y|x)/π_ref(y|x) ) + β log Z(x)

第3步:代入 Bradley-Terry 偏好模型 P(y_w ≻ y_l) = σ( r(x,y_w) − r(x,y_l) )。
   配对相减时,β log Z(x) 这一项对同一个 x 相同,直接抵消!
   ⟹ 只剩可优化的 policy-ratio,reward model 彻底消失。
```

DPO 最终损失(把策略当作隐式 reward 直接做偏好分类):

```
L_DPO = − E_{(x, y_w, y_l)} [ log σ( β log(πθ(y_w|x)/π_ref(y_w|x))
                                    − β log(πθ(y_l|x)/π_ref(y_l|x)) ) ]
```

其中 `y_w` 是偏好回答(win),`y_l` 是被拒回答(lose),σ 是 sigmoid。

```python
def dpo_loss(policy_logps, ref_logps, beta=0.1):
    """policy_logps/ref_logps: dict with keys 'chosen','rejected',各是序列总 logprob。
    直接把'策略相对参考的对数比'当作隐式 reward 做偏好分类。"""
    pi_logratio  = policy_logps['chosen'] - policy_logps['rejected']
    ref_logratio = ref_logps['chosen']    - ref_logps['rejected']
    logits = pi_logratio - ref_logratio          # 隐式 reward 差
    loss = -F.logsigmoid(beta * logits).mean()
    # 监控:隐式 reward 与准确率(chosen 是否真的被打更高分)
    chosen_rw  = beta * (policy_logps['chosen']   - ref_logps['chosen']).detach()
    rejected_rw= beta * (policy_logps['rejected'] - ref_logps['rejected']).detach()
    acc = (chosen_rw > rejected_rw).float().mean()
    return loss, acc
```

**DPO 的 β 和 GRPO 的 β 角色一致**:控制对参考模型的偏离强度。DPO 里 β 太大 → 更新太猛、易过拟合偏好对、模型变怪;太小 → 学不动。

### 6.2 决策表:DPO vs RLVR+GRPO

| 维度 | DPO(离线偏好) | RLVR + GRPO(在线可验证) |
|---|---|---|
| **前提数据** | 有成对偏好 (y_w, y_l) | 有确定性验证器(能自动判对错) |
| **在线采样** | 不需要(用固定数据集) | 需要(每步现采样现打分) |
| **计算成本** | 低(像监督训练) | 高(rollout 占大头) |
| **奖励来源** | 人类/模型偏好(可能有噪声/偏置) | 规则验证(无噪声,无 hacking 风险低) |
| **典型任务** | 风格/有用性/无害性等主观对齐 | 数学/代码/逻辑等有客观对错的推理 |
| **稳定性** | 较稳(无 rollout 分布偏移) | 较难调(KL/温度/组大小都要管) |
| **失效模式** | 偏好数据偏置被放大;离线分布与策略脱节 | reward hacking;采样崩;KL 失控 |

**一句话判据**:**能写出自动验证器的客观任务(数学/代码/逻辑)→ RLVR+GRPO;只有主观偏好对、且要省算力 → DPO。** 本章主任务(乘法)是前者的典型,所以走 GRPO;如果你的任务是"让回答更礼貌",没有对错只有偏好,那就走 DPO。

**进阶提示**:两者不互斥。工业界常见 pipeline 是 `SFT → DPO(快速对齐风格)→ RLVR+GRPO(在可验证任务上拔高推理)`。这正是把本章两条路线串起来用。

## 7. 设计权衡与常见坑(集中清单)

把散落各节的坑收拢成一张排错表——这是你跑歪时最该先看的:

- **训练奖励涨、验证集不涨** → 头号警报:Goodhart。检查训练/评估验证器是否解耦,人工读样本找漏洞。
- **训练完全不动(梯度≈0)** → 组内 σ≈0。升采样温度;或过滤掉全对/全错的 prompt(它们贡献零梯度)。
- **奖励暴涨后崩、熵骤降到 0** → 策略坍缩/reward hacking。加大 β、降 lr、加梯度裁剪、上早停。
- **KL 持续飙升** → 偏离参考太远。加大 β,或减小学习率/clip ε。
- **回答越训越长但没更对** → GRPO 长度偏差(5.2)。换长度归一方式。
- **简单题主导梯度** → 除以 σ 的难度偏差(5.2)。考虑去掉 σ 归一(Dr.GRPO 方向)。
- **奖励有微小噪声就是不收敛** → 验证器不确定性。确保验证器是纯函数,杜绝随机性。
- **结果无法复现** → 没固定/没报告随机种子。RL 方差大,务必多种子。
- **shaping 奖励反而变差** → 模型找到 shaping 的捷径。回退到诚实的 0/1 奖励。
- **OOM** → rollout 的 G×batch 太大或 max_new_tokens 太长。先砍 batch 保 G,用 LoRA,梯度检查点。

**贯穿性权衡**:RL 对齐的核心张力永远是**"优化奖励" vs "保持原能力/语言质量"**——KL 惩罚就是这条张力的旋钮。调 RL 本质上是在这两者间找平衡点,没有放之四海的最优 β,只有"对你这个任务、这个模型、这个验证器"的经验最优。

## 8. 源码 / 论文导读

**论文(读哪部分)**:

- **PPO**(Schulman et al., 2017,*Proximal Policy Optimization Algorithms*):读 clipped surrogate objective 那一节(论文 Eq. 7 附近),理解 ε-clip 的动机。本章 3.4 的目标函数就来自这里。
- **DPO**(Rafailov et al., 2023,*Direct Preference Optimization*):读第 4 节的推导(从 KL 正则 RL 的闭式解反解 reward),这是 6.1 三步推导的来源。
- **GRPO**(Shao et al., 2024,DeepSeekMath 论文):读 GRPO 一节——组内优势、去 critic 的动机、KL 用 k3 估计器的写法。本章 3.2、3.5 对应这里。**(年份/作者为公开史实;具体公式编号待核对原文。)**
- **RLVR / 可验证奖励**:理念散见于近期推理模型工作(如 DeepSeek-R1 报告)——核心是"用规则验证器代替学习的 reward model"。读它如何定义算术/代码任务的奖励。
- **KL 估计器**:Schulman 的博客笔记 *Approximating KL Divergence*,讲清 k1/k2/k3 三个估计器的偏差-方差权衡。本章 3.5 的 `lr−log lr−1` 恒非负与无偏推导对应这里。

**开源库(读哪个模块)**:

- **TRL**(HuggingFace):读 `GRPOTrainer` 和 `DPOTrainer` 两个类。`GRPOTrainer` 的 `_generate_and_score_completions` 是 rollout+打分的真实实现,`compute_loss` 对应本章 4.2;`DPOTrainer.concatenated_forward` + `dpo_loss` 对应 6.1。**这是你最该照着读的工程实现。**
- **verl**(字节,volcengine/verl):大规模 RLHF/GRPO 框架,读它的 actor-rollout 分离架构,理解工业级如何把采样和训练解耦、如何分布式 rollout。规模上来后看这个。
- **CleanRL**:读它的 `ppo.py` 单文件实现,理解 PPO 的完整训练循环(advantage 计算、clip、多 epoch update)。虽然是 RL 经典环境(非 LLM),但**单文件、无抽象**,是理解 PPO 机制最快的代码。
- **OpenRLHF**:另一个 LLM RLHF 框架,reward model + PPO + DPO 都有,可对照 TRL 看不同工程取舍。

**读法建议**:先用 CleanRL 的 `ppo.py` 把 PPO 训练循环吃透(脱离 LLM 复杂度),再读 TRL 的 `GRPOTrainer` 看它怎么把 PPO 套到 LLM + 组内优势上,最后照本章 4 节自己拼一个最小版。**别一上来就读 verl——它的分布式抽象会淹没机制。**

## 9. 动手练习

**练习 1(编码,核心)——跑通闭环并填报告**
用一个单卡能跑的小模型(0.5B~1.5B 级,待核具体可用型号),实现本章管线:两位数乘法 prompt 集 + 验证器 + GRPO 训练循环 + 四条监控曲线。要求:(a)先测 SFT baseline 正确率 `acc_sft`;(b)跑 RL,记录四条曲线;(c)对照 baseline,按 5.4 模板写实验报告。
*提示*:先确认 rollout 里组内 σ 不是常 0(打印出来看);温度从 1.0 起;G=8;β=0.04;先用很小的训练集(几百条)+ 几十步确认管线不报错,再放大。

**练习 2(推导)——证明 GRPO 组内基线的偏差量级**
3.2 节说组内基线 `μ`(用同组样本算)会引入 O(1/G) 偏差。请形式化:设组内 G 个奖励 i.i.d.,优势 `Aᵢ=rᵢ−μ`,其中 `μ=(1/G)Σrⱼ`。证明 `E[Aᵢ·∇log π(oᵢ)]` 与"用真实期望 `E[r]` 当基线"相比,差异是 O(1/G) 量级。
*提示*:把 `μ` 拆成 `(1/G)rᵢ + (1/G)Σ_{j≠i}rⱼ`,前一项使 `Aᵢ` 和 `rᵢ`(进而和 `oᵢ`)相关,系数 1/G;算这一项的期望贡献。

**练习 3(分析)——制造并诊断 reward hacking**
故意在验证器里留一个漏洞(例如:只要输出里**包含** gold 数字就给满分,不要求是最终答案)。跑 RL,观察四条曲线哪条先报警、报警形态如何。然后修复漏洞重跑,对比。
*提示*:漏洞版大概率出现"训练奖励飙升但验证集(用严格验证器)不涨";读模型输出,你会看到它学会"把 gold 数字塞进任意位置"。这是 Goodhart 的活体标本。

**练习 4(推导 + 编码)——DPO 与 GRPO 在 β→0 的行为对比**
分析:当 β→0 时,DPO 损失 `−log σ(β·logits)` 的梯度会怎样?GRPO 的 KL 惩罚项会怎样?各自意味着什么失效模式?用一个玩具二分类偏好数据验证 DPO 在极小 β 下是否过拟合。
*提示*:β→0 时 `log σ(β·x) ≈ log(1/2) + βx/2`,梯度 ∝ β,会非常小——但这是损失对 logits 的缩放,不是说学不动,而是隐式 reward 的尺度塌缩;思考这对"chosen 比 rejected 高多少"的约束意味着什么。

## 10. 小结与承上启下

这一章是全课的收口。我们把分散的零件接成了**一条单卡可复现的 RL 对齐/推理训练闭环**,并且——这是更重要的——交付了**一套不自欺的方法论**:

- **机制上**:从策略梯度 → 基线降方差 → GRPO 组内优势 → PPO clip → KL 惩罚(k3 估计器),每一步都给了能验算的推导。你现在应该能解释"为什么减基线不引入偏差""为什么 KL 用 `lr−log lr−1`""为什么 clip 能稳住 importance sampling"。
- **工程上**:rollout → 打分 → 算优势 → clip+KL 更新 → 四件套监控,代码骨架可直接落地到 TRL/verl。
- **纪律上**:SFT baseline 必做、奖励与评估解耦、防 Goodhart、强制消融、多种子、实验报告模板。**这套纪律比任何单个算法都更决定你的 RL 工作可不可信。**

**回扣全课**:06 章的 PPO clip 在 3.4 落地;09 章的 DPO 在第 6 节给了可运行实现和决策表;10 章的 RLVR 在 2.2 成为验证器骨架;11 章的 GRPO 是 3.2/3.3 的方差推导主角;14 章的稳定性调参(KL 系数、温度、组大小、坍缩诊断)在第 5 节被**实测化**成了一张排错表。

**走出这门课之后**,真实工业场景会比本章复杂得多:多节点分布式 rollout、奖励模型与验证器混合、过程奖励(PRM)、更长的推理链、在线 vs 离线混合训练。但**底层逻辑不变**:定义一个不能被作弊的奖励,在保持原能力(KL 锚定)的前提下,用低方差的优势估计稳定地把它推上去,并用解耦的评估指标诚实地检验你有没有真的变好。**你现在具备了这套底层逻辑,剩下的是规模和工程的事。**

把练习 1 真的跑一遍——读完不动手,这章的价值会蒸发一半。
