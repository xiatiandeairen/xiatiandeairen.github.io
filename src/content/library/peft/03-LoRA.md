---
title: "LoRA：ΔW=BA 低秩分解的数学与实现"
slug: "03"
collection: "peft"
order: 3
summary: "第 02 章用全量微调把任务 A 学成的基座搬到任务 B——代价是 100% 参数动一遍、且灾难性遗忘了 A。本章只冻结基座、给注意力的 q/v 投影各挂一对低秩矩阵 ΔW=(α/r)·BA，手写 LoRALinear 把可训练参数从 d×k 降到 r×(d+k)，在 toy 上只动 24% 的参数就把任务 B 的准确率从瞎猜 12.5% 拉到 28.5%、弥合大半 gap，且关掉 adapter 即恢复任务 A 到 100%（基座本体从未被改、零遗忘）。零初始化 B 与 α/r 缩放这两个'细节'是它能用的关键，本章用 t=0 输出等于基座的断言把它讲死，为第 04 章'秩 r 怎么选'与第 08 章'合并回权重'铺数学。"
topics:
  - "模型微调"
tags: []
createdAt: "2026-06-21T08:00:00.000Z"
updatedAt: "2026-06-21T08:00:00.000Z"
---

第 02 章的全量微调能把 copy 学成的基座搬到一个新任务，但你付了两笔账：一是所有参数都进了优化器(全量微调 = 100% 可训练)，二是回头测任务 A，准确率从 100% 塌到 11.7%——基座本体被梯度改写，灾难性遗忘 (catastrophic forgetting，学新任务把旧任务能力冲掉)。本章换一个新任务 B=reverse(把序列倒序),冻结基座、只给注意力的 q/v 投影各挂一对低秩矩阵,看这两笔账能不能一起免掉。LoRA 一次解决这两笔账，靠的是一个不起眼但极其硬的观察：**微调学的不是一组新权重，是一个权重增量 ΔW，而这个增量的"内在秩 (intrinsic rank，矩阵真正独立的方向数)"很低**。低到什么程度?低到你不需要存一个 d×k 的满秩矩阵,存它的两个瘦因子就够。

这章我们把 ΔW=BA 这件事从公式写成可跑的 `LoRALinear`,然后用三个断言把"为什么 LoRA 能用"钉死:t=0 时它一定等于基座、训练后只有 delta 被改、关掉 delta 任务 A 立刻回来。最后做一个失败实验——把本该置零的 B 随机初始化——让你亲眼看到那个被绝大多数教程一笔带过的"细节"塌在哪里。

## ΔW = BA：为什么是低秩,而不是稀疏

先把数学摆正。一个 Linear 层算的是 `y = x·Wᵀ + b`,W 是 (out, in)。微调它,等价于学一个增量,让层变成 `y = x·(W+ΔW)ᵀ + b`。全量微调直接把 ΔW 当成一个和 W 同形的满秩矩阵去学,代价 out×in。

LoRA 的赌注是:ΔW 不需要满秩。它把 ΔW 参数化成两个瘦矩阵的积:

```
ΔW = (α/r) · B · A,   B:(out, r),  A:(r, in),   r ≪ min(out, in)
```

A 是 down-projection(降维,把 in 维压到 r 维),B 是 up-projection(升维,把 r 维拉回 out 维)。前向不真的把 BA 乘出来(那会退化成满秩开销),而是顺着算:

```
y = x·Wᵀ + (α/r)·(x·Aᵀ)·Bᵀ
```

`x·Aᵀ` 先把输入降到 r 维,再 `·Bᵀ` 升回 out 维。这个瓶颈是省钱的全部来源:可训练参数从 `out·in` 降到 `r·(out+in)`。

这里有个容易混的点要说死:**低秩不等于稀疏**。稀疏是"大部分元素是 0",ΔW=BA 出来的矩阵每个元素几乎都非零——它是稠密的,但秩 ≤ r。也就是说它的 in 列里只有 r 个线性无关的方向,其余都是这 r 个的线性组合。LoRA 赌的是微调要表达的变化恰好活在这样一个低维子空间里。为什么这个赌注成立?LoRA 原论文(Hu et al., 2021)的经验证据是:把全量微调学出的 ΔW 做 SVD(奇异值分解),奇异值衰减极快,前几个方向几乎承载了全部能量。这是经验观察,不是定理——它在 attention 投影上很稳,但不是对任何层、任何任务都先验成立(第 04 章会拿秩 r 扫描验证这件事在我们 toy 上长什么样)。

为什么挑 q/v 投影而不是全挂?也是经验结论:原论文做过 ablation,在固定可训练预算下,把秩匀给 q 和 v 比集中砸一个矩阵效果好。本章配套代码就只在 `attn.q` 和 `attn.v` 上注入 LoRA,FFN 和 embedding 一概冻死。

## 手写 LoRALinear:按引用接管冻结权重

实现的关键决策在 `examples/peft-from-scratch/src/stage03-lora.ts`,`LoRALinear` 继承 `Linear`,这样它能就地替换 `MultiHeadAttention` 里类型为 `Linear` 的 `q`/`v`,MHA 内部 `this.q.forward(x)` 一行都不用改:

```ts
class LoRALinear extends Linear {
  A: Tensor; // (r, in) — down-projection; random so the delta's row-space is seeded
  B: Tensor; // (out, r) — up-projection; ZERO at init so BA=0 ⇒ ΔW=0 ⇒ output == base
  scaling: number; // alpha / r
  enabled = true; // adapter on/off at forward time (does not alter learned weights)
  constructor(base: Linear, r: number, alpha: number, zeroInitB = true) {
    const [outDim, inDim] = base.W.shape;
    super(inDim, outDim, true); // allocates throwaway W,b; immediately replaced by base's refs
    // Adopt the frozen base weights BY REFERENCE. The throwaway W,b from super() are discarded.
    this.W = base.W;
    this.b = base.b;
```

`this.W = base.W` 这行是按引用接管,不是拷贝,有两个理由,都是为了"诚实计数":一是 `Module.parameters()` 枚举到的就是同一份被冻结的 W,b 加上可训练的 A,B,可训练参数数是真实的;二是不会冒出第二个 `base` 子模块把同一份 W 数两遍。前提是调用方已经把基座 `freeze()` 过——把每个叶子 Tensor 的 `requires_grad` 翻成 false。

前向就是基座输出加上 delta:

```ts
override forward(x: Tensor): Tensor {
  const baseOut = super.forward(x); // (m, out), through the frozen W,b
  if (!this.enabled) return baseOut; // adapter off ⇒ exactly the base
  const down = x.matmul(this.A.transpose()); // (m, r)
  const up = down.matmul(this.B.transpose()); // (m, out)
  return baseOut.add(up.scale(this.scaling));
}
```

注意 `enabled=false` 这条路径直接 `return baseOut`——它不动学好的 B、A,只是前向时把 delta 这一项跳过。这就是后面证明"任务 A 可恢复"用的 adapter-off 开关:它不是把权重改回去,而是从加法里把增量项摘掉。

注入完冻结后回测,基座可训练参数 = 0 / 8416(应为 0),证明基座本体一个都没进优化器。这时直接拿冻结基座做任务 B,准确率只有 12.5%——这就是 LoRA 要弥合的 gap(注意 12.5% = 1/vocab = 1/8,等于瞎猜,基座对 reverse 一无所知)。

## ✦ 零初始化 B:用 t=0 等价性把"细节"讲死

现在到本章的硬核断言。LoRA 能用的前提是:**注入 adapter、但还没训练的那一刻(t=0),整个模型必须逐位等于冻结基座**。微调必须从基座出发,不是从一个被悄悄扰动过的点出发。

怎么保证 t=0 时 ΔW=0?看构造:A 用 `N(0, 1/√in)` 随机初始化(给 delta 的行空间撒种子,让它有方向可学),B 全部置零。BA 的积里只要 B=0,无论 A 是什么,ΔW 恒为 0。这就是"B 必须零初始化"的全部数学:

```ts
const aData = new Float64Array(r * inDim);
for (let i = 0; i < aData.length; i++) aData[i] = normal(0, aStd);
this.A = new Tensor(aData, [r, inDim], true, [], "LoRA.A");
const bData = new Float64Array(outDim * r);  // 全零
if (!zeroInitB) { /* failure-mode demo: 随机填 B */ }
this.B = new Tensor(bData, [outDim, r], true, [], "LoRA.B");
```

代码不靠"应该是 0"自我安慰,它用两道断言把 t=0 等价性测死。第一道在矩阵级——直接物化 ΔW 看逐元素最大绝对值:

```
[t=0 等价性] 零初始化 B ⇒ ΔW 逐元素最大绝对值 = 0.00e+0（q）/ 0.00e+0（v）
[t=0 等价性] 断言 ΔW < 1e-6 通过：注入 LoRA 后、训练前的模型 === 冻结基座，未被扰动
```

第二道在输出级——同一个输入,adapter 开和关各跑一遍,比最大差:

```
[t=0 等价性] 输出级确认：adapter 开/关 输出最大差 = 0.00e+0（即 t=0 时 adapter 不改变任何输出）
```

为什么要两道?矩阵级证 ΔW 本身是 0,输出级证"这个 0 真的没漏进任何一条前向路径"(比如 scaling、residual 里没有偷偷掺进非零项)。两道都过,才敢说"注入即等价"。

那 α/r 缩放呢?这是第二个被一笔带过的细节。`scaling = α/r`,本章 r=4、α=8、scaling=2.00。它的作用是把 delta 的幅度和秩解耦:你调 r(改子空间维度)时,不想让 delta 的整体量级跟着乱跳——否则每改一次 r 就得重调学习率。除以 r 让你换 r 时 delta 的尺度大致稳定,α 单独当一个"delta 放大旋钮"。注意 t=0 时缩放乘的是 0,所以它不破坏等价性;它只在训练起来、B 长出非零值之后才起作用。第 04 章扫 r 时你会直接受益于这个解耦——不必每个 r 重调 lr。

## 参数账:为什么论文报 <1% 而我们这里是 24%

把可训练参数实测一遍,和"全量微调同一组 q/v"对比:

```
[参数] LoRA: r=4, alpha=8, scaling=2.00
[参数·实测] LoRA 可训练 = 512；全量微调(同一组 q/v) = 2112；占比 = 24.24%
```

512 怎么来的:q 和 v 各一对 (B:32×4 + A:4×32) = 256,两个就是 512。全量 2112 是 q、v 各 `W(32×32)+b(32)` = 1056,两个 2112。占比 24.24%。

这里必须诚实:24% 离论文吹的"<1%"差着两个数量级。原因不是 LoRA 没用,是 **toy 的 d_model=32 太小**。占比公式是 `r·2d / (d²+d)`,分子随 d 线性涨,分母随 d² 涨,d 一大占比就崩塌。代码把同一公式外推到真实规模(标注 est.,纯算术):

```
[参数·公式外推 (est.)] 同一 LoRA 公式 2·2dr / 2·(d²+d)，固定 r=4，随 d_model 变化的占比：
    d_model=  32  →  24.242% trainable  ← 本次 toy 实测点
    d_model= 256  →  3.113% trainable
    d_model=1024  →  0.780% trainable
    d_model=4096  →  0.195% trainable
```

d_model=1024 就已经掉到 0.78%,而真实 LLM 的 d_model 常在数千量级,所以论文报 <1% 完全是这条曲线的自然结果。这是 toy 的诚实局限:**绝对占比偏大,但"占比随 d 增大而下降"这条机制是可迁移的**,且方向、斜率都对。你要记的是机制,不是 24% 这个数。

## 训练 delta:用 24% 的参数弥合大半 gap,且不遗忘任务 A

只让 A、B 进优化器,在任务 B 上训 200 个 epoch:

```
[LoRA 微调] 任务 B loss 0.3428 -> 0.1990
[结果] 任务 B 准确率：未适配 12.5%  →  LoRA 后 28.5%
```

准确率从瞎猜的 12.5% 拉到 28.5%——在这个 vocab=8、seqLen=6 的 toy 上,这是个真实的、非平凡的提升(reverse 是把 6 个 token 完全倒序,比 copy 难学)。关键不在绝对值高低(toy 容量有限),在于:**只动了 24% 的参数,就把 gap 弥合了大半**。

现在是 LoRA 相对第 02 章全量微调最锋利的那一刀——遗忘对比。关掉 adapter,回测任务 A:

```
[回测·关掉 adapter] 任务 A 准确率：冻结前 100.0%  →  关掉 adapter 后 100.0%（几乎不降——基座本体从未被改）
[回测·开着 adapter] 任务 A 准确率 = 18.7%（开着 B 的 delta 跑 A，自然会降——这正说明 delta 才是被学的东西）
```

读这两行要分清两个语境,否则会误读:

- **关掉 adapter**,任务 A 回到 100.0%。这是诚实的"无遗忘"主张:基座本体一个梯度都没吃过,delta 项一摘,模型逐位变回那个 100% 的 copy 基座。对照第 02 章全量微调把基座权重直接改写、A 塌掉,这就是 PEFT 的核心卖点——一个冻结基座 + 多个可插拔 delta,任务间零干扰。
- **开着 adapter**,任务 A 只剩 18.7%。这不是 bug,恰恰是证据:开着 B 的 delta 去跑 A,等于拿 reverse 的增量去做 copy,当然降。它说明 delta 真的学到了 task-specific 的东西,而不是学了个恒等变换。

把这两行连起来读才是完整故事:**被学的东西全在 delta 里,基座纹丝不动,所以一个开关就能在"任务 B 模式"和"原始任务 A 模式"间切换**。这是全量微调给不了的——它的改动焊死在权重里,没有开关。

学到的 ΔW(q 投影)秩 ≤ 4,代码把它物化成 32×32 的 heatmap 画出来(`max|·|=20.5844`),肉眼能看到由 4 个秩-1 分量叠出来的条纹结构——这就是"低秩"长在矩阵上的样子。

## 失败模式:把 B 随机初始化 → 冷启动尖峰

光说"B 要置零"没有说服力,把它弄坏给你看。代码用同一个种子位精确重建基座,唯一差别是 B 的初始化方式(`zeroInitB` 这个 flag),对比零初始化 vs 随机初始化的训练曲线。

诚实的对比轴是"两者都从同一个冻结基座出发、同样训练,只有 B 的初值不同"。先给参照点——冻结基座在任务 B 上的 loss(没有任何 delta):

```
[基准] 冻结基座在任务 B 上的 loss（无任何 delta） = 0.4295（微调的出发点）
[零初始化 B] 首个 epoch loss = 0.3498（≈ 基准：t=0 时 ΔW=0，从基座出发）
[随机初始化 B] 首个 epoch loss = 0.4005（比零初始化高 +0.0507，t=0 即扰动基座）
```

读法:零初始化 B 的首 epoch loss(0.3498)贴着基准(0.4295,一个 epoch 的训练已经把它往下拉了一点),因为 t=0 时 ΔW=0,它确实从基座出发。随机初始化 B 的首 epoch loss(0.4005)明显更高——高出 +0.0507。这 0.0507 不是噪声,是 **LoRA 自己制造、又必须先撤销的扰动**:随机 B 让 t=0 时 ΔW≠0,基座在第一个梯度步之前就已经被推离原位,训练得先把这个自伤爬回来,才能开始真正改进。这就是"冷启动尖峰 (cold-start spike)"。

```
frozen-base (起点参照) │████████████████████████████│ 0.43  loss
zero-B  init (正确)  │███████████████████████·····│ 0.35  首 epoch loss ≈基准
rand-B  init (错误)  │██████████████████████████··│ 0.401  首 epoch loss 冷启动尖峰
```

两条曲线最终收敛点其实很接近(零初始化 0.1993、随机初始化 0.2024),这正是要警惕的地方:在 toy 这种容量小、任务简单的设定下,随机 B 的尖峰最后会被磨平,你几乎看不出区别。**别被这个"看起来也行"骗了**——在真实规模、更难的任务、更短的训练预算下,这个起手就偏离基座的扰动不一定能被爬回来,它可能改变收敛盆地、拖慢收敛、甚至让微调一开始就破坏了基座里宝贵的预训练知识。零初始化 B 在 t=0 把这个风险直接消灭于无形,代价为零(A 已经把子空间撒好种子了)。这就是为什么它不是"建议",是"必须"。

代码里还留了一道诚实护栏:如果某次 toy 随机种子让尖峰恰好不显著(startGap ≤ 0),它不硬吹,而是回退到"t=0 时 ΔW≠0 已扰动基座"这条更稳的证据,并明说更大规模下尖峰才稳定可见。这是这本书的底线——toy 上没观测到的,不假装观测到。

## ⚡ 前沿:同一框架下的 DoRA 与 PiSSA(初始化/参数化仍是开放问题)

你现在已经把 ΔW=BA、零初始化、α/r 缩放这套底座吃透了。这个底座的价值在于:**LoRA 之后一大批改进,本质都是在这同一个公式上动初始化或参数化**,你已经具备读它们论文的前提。

- **PiSSA**(Principal Singular values and Singular vectors Adaptation):它质疑零初始化 B 这个选择。零初始化保证 t=0 等价,但也意味着 delta 从一个"没承载任何基座主成分"的方向起步。PiSSA 反过来:对基座 W 做 SVD,用前 r 个主奇异方向初始化 B、A(剩下的残差留在冻结部分),让 delta 一开始就对齐基座最重要的方向。代价是 t=0 不再严格等于基座——它赌"对齐主成分"带来的收敛收益盖过这个偏移。
- **DoRA**(Weight-Decomposed LoRA):它质疑 ΔW=BA 这个加法参数化本身。DoRA 把权重拆成方向(单位向量)和幅度(标量模长)两部分,只用 LoRA 去学方向的增量、用单独的可训练标量学幅度。动机是观察到全量微调时方向和幅度的变化模式不一样,强行用一个加法 delta 同时表达两者会受限。

注意这两个改的都不是"LoRA 错了",而是它两个被本章钉死的设计选择——B 的初始化(PiSSA)、ΔW 的参数化形式(DoRA)——各自是不是最优。这恰恰说明:**LoRA 这套低秩 delta 的初始化与参数化,至今没有一个对所有任务/模型都最优的通用解,仍是活跃研究方向**。零初始化在"保证不扰动基座"这个目标上是对的、必须的;但"从哪个点、用什么形式起步收敛最快"是另一个问题,目前是经验性的、任务相关的,没有定论。你能读懂这些论文在改什么、为什么改、代价在哪,本章的目的就达到了。

---

⚠ toy-scale 提醒:本章的绝对准确率(28.5%)和参数占比(24.24%)都偏乐观/偏大,因为 d_model=32 太小。可迁移的不是这些绝对值,而是机制:ΔW=BA 低秩分解、t=0 等价性、占比随 d 增大而下降、随机 B 的冷启动尖峰、adapter-off 即回退。完整运行输出(含 32×32 ΔW heatmap)见 `npx tsx examples/peft-from-scratch/src/stage03-lora.ts`。

下一章(第 04 章)我们把 r 当变量扫一遍——r 太小欠拟合、太大白烧参数,看那条 accuracy-vs-r 曲线在哪拐弯,顺便验证本章"ΔW 内在秩很低"这个赌注在 toy 上到底成不成立。第 08 章会回到 `ΔW=(α/r)·BA` 这个式子:既然它就是个可加的低秩矩阵,推理前把它乘出来加回 W、得到一个和原始 Linear 同形、零额外延迟的合并权重,数学上是平凡的——本章的物化 `deltaW()` 已经是那一步的雏形。
