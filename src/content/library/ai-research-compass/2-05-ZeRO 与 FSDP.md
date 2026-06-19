---
title: "ZeRO 1/2/3 与 FSDP:用通信换显存的分片术"
slug: "2-05"
collection: "ai-research-compass"
group: "MLSys专家课程"
order: 2005
summary: "这一章把你从\"数据并行(DP)能加速但每张卡都得装下整个模型,所以装不下大模型\"带到\"理解 ZeRO 如何把 参数/梯度/优化器状态 切成 1/N 分到每张卡,让纯数据并行也能训练 70B 甚至更大的模型\",并最终落到 PyTorch FSDP 的源码层面,让你能看懂 unshard/reshard…"
topics:
  - "AI 研究"
tags: []
createdAt: "2026-06-17T13:55:05.000Z"
updatedAt: "2026-06-17T13:55:05.000Z"
---
> 这一章把你从"数据并行(DP)能加速但每张卡都得装下整个模型,所以装不下大模型"带到"理解 ZeRO 如何把 参数/梯度/优化器状态 切成 1/N 分到每张卡,让纯数据并行也能训练 70B 甚至更大的模型",并最终落到 PyTorch FSDP 的源码层面,让你能看懂 unshard/reshard 在哪一行触发、为什么 1.5 倍通信量能被预取掩盖。

## 0. 这一章在整门课里的位置

第 01 章给了你一本**显存账本**:在 fp16 混合精度 + Adam 优化器下,**每个参数稳定地占用约 16 字节**。这本账是后面所有并行/省显存技术的判据,我们这一章会反复用它。第 02 章讲了数据并行靠 Ring All-Reduce 同步梯度,带宽最优,但有一个致命缺陷:**每张卡都存了一份完整的模型副本(参数 + 梯度 + 优化器状态)**,所以单卡装不下的模型,纯 DP 一样装不下。第 03/04 章用张量并行和流水线并行解决了"装不下"的问题,但它们要改模型代码、对网络拓扑敏感、调起来痛苦。

ZeRO(Zero Redundancy Optimizer)问的是另一个问题:**能不能不改模型代码、还是用数据并行那套 API,但是把每张卡的冗余存储干掉?** 答案是能。ZeRO 的洞察非常朴素——**DP 里 N 张卡存了 N 份一模一样的东西,这是纯粹的冗余;把它们切成 N 份、每张卡只存 1/N,显存就能降到接近 1/N。** 代价是通信:你需要在用到某块参数前把它从别的卡那里收集回来。这就是"用通信换显存"。

**FSDP(Fully Sharded Data Parallel)是 PyTorch 官方对 ZeRO-3 的原生实现。** 学完这一章,你应该能:(1) 用账本算出 ZeRO-1/2/3 在 N 卡下单卡到底省多少;(2) 在白板上走一遍 ZeRO-3 的一步训练,说清每个 all-gather / reduce-scatter 发生在什么时刻、收集和释放的是什么;(3) 推导出 ZeRO-3 相对 DP 多了约 50% 通信量,并解释为什么这部分通信几乎不增加 wall-clock 时间;(4) 看懂 FSDP 的 `FlatParamHandle` 和 `_runtime_utils` 在做什么。

## 1. 预备:把第 01 章那本账拆细

我们必须先把账本的每一项摊开,因为 ZeRO 的三个 stage 恰好就是按这三类状态来切的。设模型参数量为 `Ψ`(读作 psi,DeepSpeed 原论文用的符号),混合精度 + Adam 的单卡显存(不含激活)按第 01 章是:

```
单参数 16 字节的来源(fp16 mixed precision + Adam):
  fp16 参数     权重前向/反向用    2 字节
  fp16 梯度     反向产生          2 字节
  ----- 以上是"模型本体",每步都要参与计算 -----
  fp32 参数主副本  优化器更新用      4 字节   ┐
  fp32 Adam m(一阶动量)          4 字节   ├─ 这三项统称"优化器状态" K=12
  fp32 Adam v(二阶动量)          4 字节   ┘
  ------------------------------------------------
  合计                            16 字节 / 参数
```

我们把这三大类记成三个系数:

```
参数(parameters)     P = 2Ψ 字节     (fp16)
梯度(gradients)      G = 2Ψ 字节     (fp16)
优化器状态(opt states) O = K·Ψ 字节    其中 K = 12(fp32 主副本 4 + m 4 + v 4)
```

**为什么优化器状态最大?** 因为它要存 fp32(高精度,4 字节而非 2),而且有三份(主副本、m、v)。所以 `O = 12Ψ` 是 `P` 或 `G` 的 6 倍。这个 6:1 的比例是理解 ZeRO 的关键直觉:**显存大头在优化器状态,所以先切它(ZeRO-1)收益最大。**

纯数据并行(DP / DDP)在 N 张卡上,每张卡的模型相关显存是:

```
M_DP = P + G + O = (2 + 2 + 12)·Ψ = 16Ψ 字节   (与 N 无关,每张卡都一样)
```

这就是"冗余":N 张卡总共存了 16ΨN 字节,但有用信息只有 16Ψ,剩下 16Ψ(N−1) 全是重复。ZeRO 要回收的就是这部分。

> 直觉锚点:7B 模型,`Ψ = 7×10⁹`。DP 下单卡模型显存 = 16 × 7×10⁹ = 112 GB。已经超过单张 80GB A100/H100 的容量了——这就是为什么 7B 全参数训练在单卡上连优化器状态都放不下,必须切。

## 2. ZeRO 的三级切分:切什么、省多少

ZeRO 是一个**渐进式**方案,三个 stage 逐级激进,每一级在前一级基础上再多切一类状态。核心约束是:**计算的那一刻,被用到的张量必须是完整的;不被用到时,可以只持有自己的分片。**

### 2.1 ZeRO-1:只切优化器状态(Pos)

**机制。** 优化器状态 `O` 在前向、反向计算中根本用不到——它只在 `optimizer.step()` 更新参数那一刻才被读写。所以 ZeRO-1 把 `O` 沿参数维度切成 N 份,卡 i 只持有第 i 份优化器状态(`O/N`)。参数 `P` 和梯度 `G` 仍然每张卡存完整一份(和 DP 一样)。

更新流程:反向得到完整 fp16 梯度 → 各卡只把**属于自己那 1/N 参数**的梯度拿来,用自己持有的那 1/N fp32 主副本和 m/v 做 Adam 更新 → 更新出新的 1/N fp16 参数 → 用一次 **all-gather** 把各卡更新好的参数分片拼回完整参数,供下一步前向使用。

```
卡 i 在 step() 里只更新参数的第 i 段:
  shard_i = slice(all_params, i·Ψ/N, (i+1)·Ψ/N)     # 这张卡负责的参数段
  grad_i  = 对应段的 fp16 梯度(各卡此时都有完整梯度,取自己那段即可)
  m_i, v_i, master_i = 这张卡持有的 fp32 状态(只有 1/N)
  Adam 更新 master_i,写回 fp16 shard_i
  all_gather(shard_i across N ranks) -> 完整 fp16 参数,下一步前向用
```

**显存账。**

```
M_ZeRO1 = P + G + O/N = 2Ψ + 2Ψ + 12Ψ/N
当 N → ∞:  M_ZeRO1 → 4Ψ   (省掉了全部 12Ψ 优化器状态,只剩参数+梯度)
```

**省多少?** 从 16Ψ 降到 4Ψ + 12Ψ/N。N=8 时 = 4Ψ + 1.5Ψ = 5.5Ψ,**约为 DP 的 34%**。优化器状态是大头(占 16 中的 12),所以光切它就省掉一大半。

### 2.2 ZeRO-2:再切梯度(Pos+g)

**机制。** 梯度 `G` 在反向算完、喂给 optimizer.step() 之后就没用了。而且关键是:**卡 i 做更新只需要它负责那段参数的梯度,根本不需要别段的梯度。** 所以梯度也可以切——但切法和优化器状态不同,要配合反向传播的过程。

DP 里同步梯度用的是 **all-reduce**(每张卡都得到完整的、求和平均后的梯度)。ZeRO-2 把它换成 **reduce-scatter**:每张卡只收到**它负责那一段**的、已经求和平均好的梯度。这样卡 i 反向产生完整梯度后,经过 reduce-scatter,只保留 `G/N`,其余立即释放。

```
反向过程中,每算完一层的梯度,逐渐积累;在合适的桶(bucket)边界:
  reduce_scatter(local_full_grad) -> 卡 i 只留下自己那段的归约后梯度 grad_i(G/N)
  其余分片发给对应的属主卡后,本地释放
随后 step():卡 i 用 grad_i + 自己的优化器状态分片更新 shard_i
然后 all_gather(shard_i) -> 完整参数(同 ZeRO-1)
```

**显存账。**

```
M_ZeRO2 = P + G/N + O/N = 2Ψ + 2Ψ/N + 12Ψ/N = 2Ψ + 14Ψ/N
当 N → ∞:  M_ZeRO2 → 2Ψ   (只剩完整的 fp16 参数)
```

N=8 时 = 2Ψ + 1.75Ψ = 3.75Ψ,**约为 DP 的 23%**。

**关键细节(新手坑):** ZeRO-2 不能在反向全部结束后才一次性切梯度——那样反向过程中梯度仍是完整的,峰值显存没降下来。正确做法是**边反向边 reduce-scatter**:反向是从输出层往输入层走的,先算出的层先归约、先释放。这要求实现里把参数分桶,梯度一凑齐一个桶就立刻 reduce-scatter。FSDP 的 reshard 时机就是这么设计的,后面会讲。

### 2.3 ZeRO-3:连参数也切(Pos+g+p)

**机制。** ZeRO-1/2 还留着一份完整参数 `P=2Ψ`,这是最后的冗余。ZeRO-3 把它也切掉:**卡 i 平时只持有参数的第 i 段 `P/N`。** 但前向/反向计算时显然需要完整的某层参数,怎么办?**用到的时候临时收集,用完立刻丢掉。**

这就是 ZeRO-3 的灵魂——**参数的生命周期被压缩到"即用即取、用完即弃"(just-in-time gather)**:

```
前向到第 L 层时:
  all_gather(layer_L_params across N ranks)  # 把该层完整参数临时拼出来
  y = layer_L.forward(x)                      # 用完整参数算前向
  free(gathered layer_L_params)               # 立刻释放,只留回自己那 1/N 分片
反向到第 L 层时:
  all_gather(layer_L_params)                  # 反向同样需要完整参数,再收集一次
  grads = layer_L.backward(...)               # 算梯度(完整)
  reduce_scatter(grads) -> 卡 i 留下 grad_i(G/N)  # 梯度归约+切分,一步到位
  free(gathered layer_L_params)               # 再次释放完整参数
step():
  卡 i 用 grad_i + opt_state 分片更新 param_i(P/N) —— 全程只碰自己那段,无需 all-gather!
```

注意 ZeRO-3 的 step **不需要** ZeRO-1/2 末尾那个 all-gather:因为参数本来就是切着存的,更新完还是切着存,下一步前向再 all-gather 取用。

**显存账(模型本体部分)。**

```
M_ZeRO3 = P/N + G/N + O/N = (2 + 2 + 12)Ψ/N = 16Ψ/N
当 N → ∞:  M_ZeRO3 → 0   (理论上单卡模型显存可任意小!)
```

N=8 时 = 16Ψ/8 = 2Ψ,**约为 DP 的 12.5%(即 1/N)**。这就是标题说的"**数据并行拿到模型并行的显存效率**":ZeRO-3 让每张卡的常驻模型显存正比于 1/N,和张量并行/流水线并行一个量级,但你用的还是数据并行的编程模型(不用手动切矩阵、不用写 send/recv)。

### 2.4 三级对比一张表

设 `K=12`(Adam fp32 三件套),DP 基线 16Ψ:

```
方案       常驻显存(模型本体)         N=8       N=64      N→∞     额外操作
----------------------------------------------------------------------------------
DP/DDP     (2+2+12)Ψ = 16Ψ            16.0Ψ     16.0Ψ     16Ψ     all-reduce 梯度
ZeRO-1     2Ψ+2Ψ+12Ψ/N = 4Ψ+12Ψ/N   5.5Ψ      4.19Ψ     4Ψ      step 后 all-gather 参数
ZeRO-2     2Ψ+(2+12)Ψ/N = 2Ψ+14Ψ/N   3.75Ψ     2.22Ψ     2Ψ      reduce-scatter 梯度
ZeRO-3     (2+2+12)Ψ/N = 16Ψ/N        2.0Ψ      0.25Ψ     0Ψ      前/反向 all-gather 参数 + reduce-scatter 梯度
```

**别忘了还有激活(activations)。** 上表只算了"模型本体 + 优化器状态"。激活显存由 batch、序列长度、层数决定(第 01 章和第 08 章的内容),ZeRO **不切激活**(那是激活重计算 / 张量并行 / 序列并行的活)。所以真实单卡峰值 = `16Ψ/N(ZeRO-3) + 激活 + 临时收集的那一层完整参数`。后面会看到那个"临时一层"也是峰值的一部分。

## 3. ZeRO-3 一步训练:逐时刻走一遍

把上面的伪代码摊开成真正的时间线,这是面试和调试时最容易讲错的地方。设模型 L 层,卡数 N,关注一张卡(rank=0)上发生了什么。

```
========================  ZeRO-3 一个 training step(rank 0 视角) ========================

[初始态] rank 0 内存里:每层参数只有自己那 1/N 分片;优化器状态只有 1/N;无完整层。

--- 前向 (layer 0 -> L-1) ---
for L in 0..layers:
    1. all_gather(P_L)        # 向其余 N-1 张卡要 P_L 的它们那部分,拼出完整 P_L(临时,大小 2Ψ_L)
    2. y = forward(x, P_L)    # 用完整 P_L 算这一层
    3. 缓存反向所需 activation # (若开 checkpoint 则只存输入,见 §5)
    4. reshard(P_L)           # 释放 P_L 的非己分片,内存回到只剩 1/N。临时显存峰值在此处回落
# 前向结束:得到 loss

--- 反向 (layer L-1 -> 0) ---
for L in (layers-1)..0:
    5. all_gather(P_L)        # 反向同样需要完整 P_L,再次临时收集(第二次!)
    6. grad_full = backward(...) # 算出该层完整梯度(大小 2Ψ_L)
    7. reduce_scatter(grad_full) # 归约+切分:rank 0 只留下 grad_L 的第 0 段(2Ψ_L/N),其余发走
    8. reshard(P_L)           # 再次释放完整 P_L
# 反向结束:rank 0 手里是各层梯度的"自己那段"

--- 更新 ---
    9. optimizer.step()       # rank 0 用 自己的梯度分片 + 自己的优化器状态分片
                              # 更新 自己的参数分片(P/N)。全程不跨卡,无通信!
```

**两个必须记住的点:**

1. **每层参数被 all-gather 了两次**(一次前向 step 2,一次反向 step 5),因为前向用完就 reshard 丢掉了,反向得重新收集。这是 ZeRO-3 比 ZeRO-2 多出通信的根源。如果舍不得这次重收集而在前向后保留完整参数,那就退化成 ZeRO-2 了(参数不切)。**这是一个明确的显存↔通信权衡:ZeRO-3 选择多通信一次来换取参数不常驻。**

2. **临时显存峰值** = 常驻 `16Ψ/N` + 当前正在处理的那一层(或那个 FSDP unit)的完整参数 `2Ψ_unit` + 激活。所以**单元(unit)切得越细,临时峰值越低,但 all-gather 次数越多、每次越小、通信效率越低**。这是 FSDP `auto_wrap` 粒度的核心权衡(§4.3)。

## 4. 通信量推导:为什么 ZeRO-3 只比 DP 多约 50%

这是本章最重要的定量结论,也是最反直觉的地方:**ZeRO-3 把显存降到 1/N,听起来要付出巨大通信代价,但实际上通信量只比普通 DP 多约 1.5 倍。** 我们来推导。

### 4.1 集合通信原语的通信量(每张卡收发字节)

先要一个基础事实(第 02 章证过的带宽最优实现下的结论):**对总大小为 `M` 字节的数据,在 N 张卡上做 all-reduce、all-gather、reduce-scatter,每张卡需要收发的数据量都约为 `M·(N−1)/N ≈ M`(N 大时)。** 关键关系:

```
all-reduce = reduce-scatter + all-gather
  reduce-scatter:每卡通信量 ≈ M·(N-1)/N ≈ M
  all-gather:    每卡通信量 ≈ M·(N-1)/N ≈ M
  all-reduce:    每卡通信量 ≈ 2M·(N-1)/N ≈ 2M   (= 上面两步之和)
```

**直觉:** all-reduce 在 Ring 实现里就是先 reduce-scatter(把每段归约到一张卡)再 all-gather(把归约结果广播回所有卡),所以是两段、约 2M;单独的 reduce-scatter 或 all-gather 各约 1M。把 M 记成"被通信对象的总字节数"(注意是总大小,不是单卡分片大小)。

### 4.2 DP 基线通信量

DP 每步只在反向后对**梯度**做一次 all-reduce。梯度总大小 = `2Ψ`(fp16)。所以:

```
Comm_DP = all-reduce(2Ψ) ≈ 2 × 2Ψ = 4Ψ   (每卡每步收发约 4Ψ 字节)
```

### 4.3 ZeRO-3 通信量

逐项数 ZeRO-3 一步里的所有集合通信(对象总大小都按 fp16 = 2Ψ 计):

```
前向:每层 all-gather 一次参数,全模型合计 all-gather 整套参数一遍
    -> all-gather(2Ψ)              ≈ 2Ψ
反向:每层再 all-gather 一次参数,全模型合计又一遍
    -> all-gather(2Ψ)              ≈ 2Ψ
反向:每层 reduce-scatter 一次梯度,全模型合计整套梯度一遍
    -> reduce-scatter(2Ψ)          ≈ 2Ψ
------------------------------------------------------------
Comm_ZeRO3 ≈ 2Ψ + 2Ψ + 2Ψ = 6Ψ   (每卡每步收发约 6Ψ 字节)
```

### 4.4 比值

```
Comm_ZeRO3 / Comm_DP = 6Ψ / 4Ψ = 1.5
```

**结论(加粗记牢):ZeRO-3 的通信量约为普通 DP 的 1.5 倍。** 直觉版:DP 是 1 次 all-reduce(=reduce-scatter + all-gather,共 2 个单位);ZeRO-3 是 2 次 all-gather(前向 + 反向)+ 1 次 reduce-scatter,共 3 个单位。3 / 2 = 1.5。

**与 ZeRO-1/2 对比:**

```
ZeRO-2:反向 reduce-scatter 梯度(2Ψ)+ step 后 all-gather 参数(2Ψ)= 4Ψ ≈ DP,通信量基本不变
ZeRO-1:同上,也是 reduce-scatter + all-gather ≈ 4Ψ ≈ DP（实现上 ZeRO-1 常用 all-reduce+本地切，量级同 DP）
ZeRO-3:6Ψ = 1.5 × DP
```

所以 **ZeRO-1/2 几乎是"免费午餐"——通信量和 DP 持平,显存却大降;ZeRO-3 才需要多付 50% 通信**。这解释了工程上的选型直觉:**显存够用就停在 ZeRO-2,真的装不下才上 ZeRO-3。**

### 4.5 为什么这 50% 几乎不增加 wall-clock 时间

通信量多 50% 不等于训练慢 50%,因为现代实现把通信**和计算重叠**了。关键机制是**预取(prefetch)**:

```
朴素(串行,慢):
  [gather L0] -> [compute L0] -> [gather L1] -> [compute L1] -> ...
  通信和计算交替,GPU 算的时候网卡闲着,网卡传的时候 GPU 闲着 -> 时间是两者之和

预取(重叠,快):
  算 L_k 的同时,后台异步 all-gather L_{k+1} 的参数
  [gather L0]
            [compute L0 | gather L1]       <- compute L0 与 gather L1 并行
                        [compute L1 | gather L2]
                                    [compute L2 | ...]
  只要 单层计算时间 ≥ 单层 all-gather 时间,通信就被完全藏在计算背后 -> 时间 ≈ max(计算, 通信) ≈ 计算
```

**能否藏住,取决于"计算/通信比"(compute-to-communication ratio)。** 一层的计算时间正比于该层 FLOPs,通信时间正比于该层参数字节数 / 带宽。Transformer 里 FLOPs ≈ 2 × 参数量 × token 数,所以:

```
计算时间 ≈ (2 · Ψ_layer · tokens) / (GPU 算力 FLOP/s)
通信时间 ≈ (2 · Ψ_layer) / (互联带宽 Byte/s)         # fp16 参数 2 字节/参
比值 ∝ tokens × (带宽 / 算力)
```

**直觉:batch/序列越大(tokens 越多),每层计算越重,越容易把通信藏住。** 反过来,小 batch + 慢互联(比如跨节点只有以太网而非 NVLink/InfiniBand)时,通信藏不住,ZeRO-3 就会暴露出通信瓶颈,这时候要么加大 batch、要么换 ZeRO-2、要么上 §5 的混合分片。**这是 ZeRO-3 最大的实战坑:它对互联带宽敏感,跨慢速节点会原形毕露。** 具体的"在多少带宽下能藏住多少"取决于硬件,这里不给死数字「待核」。

## 5. FSDP:ZeRO-3 的 PyTorch 原生实现

DeepSpeed 的 ZeRO 是 PyTorch 之上的一个外挂框架;**FSDP(`torch.distributed.fsdp`)是 PyTorch 把 ZeRO-3 做进官方运行时的版本。** 二者机制等价,但 FSDP 和 autograd、`torch.compile`、checkpoint 的集成更原生。下面讲 FSDP 的几个核心设计,这些就是它和朴素 ZeRO-3 实现的区别所在。

> 说明:PyTorch 有两代 FSDP——FSDP1(`FullyShardedDataParallel` 类,基于 `FlatParamHandle`)和 FSDP2(`fully_shard` 函数式 API,基于 `DTensor`,新代码推荐)。本节机制讲解以 FSDP1 的 `FlatParamHandle` 为主(源码更直观、最能体现机制),API 示例给两代。两代的 unshard/reshard 时机和通信策略一致。具体的类名/路径以你本地的 PyTorch 版本为准「待核」。

### 5.1 FlatParameter:为什么要把一堆参数拍扁成一个

**问题:** 一个 FSDP 单元(比如一个 Transformer block)里有几十个张量(各 linear 的 weight/bias、layernorm 的 gamma/beta……)。如果对每个张量单独做 all-gather,会发起几十次小通信,**每次通信都有固定启动开销(latency),小消息一多,延迟主导,带宽利用率极低。**

**FSDP 的解法——FlatParameter:** 把这个单元里所有原始参数**按字节首尾相接拼成一个一维大张量(flat parameter)**,然后只对这一个大张量做切分和 all-gather。

```
单元内原始参数:  W1[768,768]  b1[768]  W2[768,3072]  b2[3072]  γ[768] ...
                 ↓ 全部 flatten 成 1D 再 concat
flat_param:      [───────────── 一根超长的一维张量(总 numel = 各参数 numel 之和) ─────────────]
                 ↓ 沿这根张量按 rank 等分(末尾 pad 到能被 N 整除)
rank0 持有:      [══ 第 0 段 ══]
rank1 持有:                    [══ 第 1 段 ══]   ...
```

**好处:** (1) 一次 all-gather 搞定整个单元,通信高效;(2) 切分简单——就是一维数组切 N 段,不用关心每个原始张量的形状;(3) 内存连续,collective 走的是一块连续 buffer。**代价/坑:** 切分边界不对齐原始张量边界(一个张量可能被劈到两张卡上),所以 FSDP 内部必须维护"flat 偏移 → 原始张量 view"的映射,在 unshard 后把 flat buffer 重新切成各参数的 view 还给模块。这个映射逻辑就在 `FlatParamHandle` 里。

源码锚点:`torch/distributed/fsdp/_flat_param.py` 的 `FlatParamHandle` 类——`flatten_tensors`(拍扁)、`_get_unflat_views`(把 flat buffer 切回各原始张量 view)、`shard`(按 rank 切分)。读懂这三个方法,你就懂了 FlatParameter 的全部。

### 5.2 unshard / reshard:收集与释放的时机

FSDP 把 ZeRO-3 的 "all-gather → compute → free" 封装成两个动作:

- **unshard(对应 all-gather):** 把当前单元的 flat_param 从各 rank 收集成完整 flat_param,再 unflatten 成各原始张量 view,挂回模块,让 `forward`/`backward` 能正常访问 `self.weight`。
- **reshard(对应 free):** 计算完成后,释放收集来的完整 flat_param,只保留本 rank 的分片,把临时显存还回去。

**时机(这是 FSDP 行为的核心,务必记住):**

```
前向:
  进入单元 forward 前 (pre-forward hook):  unshard      -> 完整参数就位
  单元 forward 计算
  离开单元 forward 后 (post-forward hook): reshard      -> 立即释放(若该单元前向后还要被反向用)
反向:
  进入单元 backward 前 (pre-backward hook): unshard     -> 重新收集(因前向已 reshard)
  单元 backward 计算梯度
  离开单元 backward 后 (post-backward hook):
        reduce_scatter(grad) -> 本 rank 只留 grad 分片
        reshard               -> 释放完整参数
```

这些 hook 的注册和触发逻辑在 **`torch/distributed/fsdp/_runtime_utils.py`**,关键函数名(版本可能略有出入「待核」):`_pre_forward` / `_post_forward` / `_pre_backward_hook` / `_post_backward_hook`,以及做收集的 `_unshard` 和做归约的 `_reduce_grad`(或类似命名)。**想真正理解 FSDP,就从 `_runtime_utils.py` 这几个 hook 读起,对照 §3 的时间线,一一对上。**

**一个微妙但重要的优化——根单元(root)前向后不 reshard:** 反向马上就要用最外层单元的参数,如果前向一结束就 reshard、反向立刻又 unshard,等于白白多一次收集+释放。所以 FSDP 对部分单元(尤其马上要反向的)可以选择**前向后不立即 reshard**,这由 `reshard_after_forward` 参数控制(见 §5.4)。

### 5.3 通信预取(prefetch):藏住通信的工程实现

§4.5 说"算第 k 层时后台 gather 第 k+1 层",FSDP 用两条独立的 CUDA stream + 显式预取来实现:

- **计算流(default stream)** 跑 forward/backward。
- **通信流(separate stream)** 跑 all-gather / reduce-scatter,与计算流并行,靠 CUDA event 做依赖同步。
- **前向预取(`forward_prefetch`):** 在执行单元 k 的计算前,提前发起单元 k+1 的 unshard。
- **反向预取(backward_prefetch,默认 `BACKWARD_PRE`):** 反向时,在计算单元 k 的梯度前,提前发起单元 k−1(反向顺序里的下一个)的 unshard。反向预取尤其重要,因为反向的执行顺序在运行时才确定,FSDP 通过记录前向顺序来预测反向顺序。

```
两条流并行(反向片段示意):
计算流:   ...[backward unit_k][reduce_scatter 在通信流]...
通信流:   [unshard unit_{k-1}]  <- 与 backward unit_k 重叠
          只要 unshard 在 unit_{k-1} 的 backward 真正开始前完成,就完全藏住
```

源码锚点:`_runtime_utils.py` 里的 `_prefetch_handle` / `_get_handle_to_prefetch`(预取决策),以及 stream 管理。预取策略由 `BackwardPrefetch.BACKWARD_PRE`(更激进、显存峰值略高)和 `BACKWARD_POST`(更省显存、重叠差些)切换——这又是一个显存↔速度权衡。

### 5.4 FSDP 实战代码

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, BackwardPrefetch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

# 假设已 dist.init_process_group("nccl") 且每 rank 绑定一张卡
torch.cuda.set_device(local_rank)
model = build_my_transformer().cuda()   # 普通模型,无需手动切

# 关键 1:auto_wrap_policy —— 决定"每个 FSDP 单元(切分粒度)是什么"
# 通常按 Transformer block 包一层:每个 block 是一个 unshard/reshard 单元
auto_wrap = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={MyTransformerBlock},   # 你的 block 类
)

# 关键 2:混合精度策略(对应第 01/07 章:参数/梯度/归约各自的 dtype)
mp = MixedPrecision(
    param_dtype=torch.bfloat16,      # all-gather 出来的参数用 bf16 算
    reduce_dtype=torch.float32,      # 梯度 reduce-scatter 用 fp32 归约,防数值误差累积
    buffer_dtype=torch.bfloat16,
)

model = FSDP(
    model,
    auto_wrap_policy=auto_wrap,
    sharding_strategy=ShardingStrategy.FULL_SHARD,   # = ZeRO-3(参数+梯度+优化器全切)
    # 其他选项:SHARD_GRAD_OP = ZeRO-2(参数前向后不 reshard,只切梯度+优化器)
    #          HYBRID_SHARD  = 节点内 ZeRO-3 + 节点间 DP 复制(见 §6)
    mixed_precision=mp,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE, # 反向预取,藏通信
    forward_prefetch=True,
    limit_all_gathers=True,          # 限制在途 all-gather 数,控显存峰值(rate limiter)
    device_id=torch.cuda.current_device(),
)

# 优化器要在 FSDP 包装之后创建,这样它只看到本 rank 的参数分片(优化器状态自然只占 1/N)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 训练循环和普通 DP 完全一样 —— 这就是 ZeRO 的卖点:不改训练代码
for batch in loader:
    optimizer.zero_grad()
    loss = model(batch).loss     # 前向:逐单元 unshard->算->reshard
    loss.backward()              # 反向:逐单元 unshard->算->reduce_scatter->reshard
    optimizer.step()             # 各 rank 只更新自己的参数分片
```

**逐项对应机制:**

- `ShardingStrategy.FULL_SHARD` ↔ ZeRO-3;`SHARD_GRAD_OP` ↔ ZeRO-2(`SHARD_GRAD_OP` 的语义就是"前向后不 reshard 参数",从而参数常驻、只切梯度和优化器状态)。FSDP 没有单独的 ZeRO-1 strategy。
- `auto_wrap_policy` ↔ §3 说的"单元粒度":包得越细(每个 block 一个单元),临时峰值越低、通信次数越多;包得越粗,反之。**新手坑:整个模型只包一层(不 auto_wrap),那 unshard 一次就把整个模型 all-gather 出来,临时峰值等于完整模型,ZeRO-3 省显存的意义全没了。** 必须 auto_wrap 到合适粒度。
- `reduce_dtype=fp32` 是数值稳定的关键(见第 07 章):梯度归约若用 bf16/fp16,大量小梯度求和会因有效位数不足而丢失精度,所以归约升到 fp32。
- 优化器**必须在 FSDP 包装后创建**——否则它会看到完整参数,优化器状态就不是 1/N 了。这是高频踩坑点。

FSDP2(新 API)等价写法:

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
# 自底向上:先 shard 每个 block,再 shard 顶层
for block in model.layers:
    fully_shard(block, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                                                       reduce_dtype=torch.float32))
fully_shard(model, mp_policy=...)
# FSDP2 基于 DTensor,参数本身就是 sharded 的 DTensor;reshard_after_forward 参数控制 ZeRO-2/3 行为
```

### 5.5 和激活重计算(activation checkpointing)组合

ZeRO 切的是参数/梯度/优化器,**不碰激活**。但激活在大模型里往往是显存第二大头(尤其长序列)。所以 FSDP **几乎总是和激活重计算一起用**(第 08 章详讲):前向不存中间激活,反向时重算。

二者组合时有一个**协同效应,也是一个坑**:

```
开了 checkpoint 后,被 checkpoint 的单元在反向时要"重新前向一遍",
这次重新前向也需要完整参数 -> 又触发一次 unshard!
所以一个被 checkpoint 的 FSDP 单元,反向阶段可能 unshard 两次:
  (重算前向的 unshard) + (算梯度的 unshard)   <- 取决于实现是否复用
```

正确做法是让 **checkpoint 的边界和 FSDP wrap 的边界对齐**(都按 Transformer block 切),这样 FSDP 能识别并尽量复用同一次 unshard,避免重复收集。PyTorch 提供 `apply_activation_checkpointing` 工具来对齐二者。**坑:** 如果 checkpoint 粒度和 FSDP 粒度不一致(比如 checkpoint 包两个 block、FSDP 包一个),重算时的 unshard 时机会错乱,白白多通信甚至 OOM。

## 6. HYBRID_SHARD:跨节点慢互联的折中

§4.5 说 ZeRO-3 跨慢速节点会暴露通信瓶颈。FSDP 给的解法是 **HYBRID_SHARD**:

- **节点内**(8 张卡走 NVLink,带宽极高):用 FULL_SHARD(ZeRO-3),享受 1/8 显存。
- **节点间**(走 InfiniBand/以太网,带宽相对低):用 DDP 式复制(每个节点存一份完整分片集),节点间只在反向做一次梯度 all-reduce,**而不是把昂贵的参数 all-gather 拉到慢速跨节点链路上**。

```
8 卡/节点、共 4 节点(32 卡)为例:
  FULL_SHARD(32 卡全切):参数切成 1/32,但 all-gather 要跨节点 -> 慢链路传参数,易瓶颈
  HYBRID_SHARD:        参数在节点内 8 卡切成 1/8;4 个节点各持一份完整的 1/8 分片集
                       前向/反向的参数 all-gather 只在节点内(快);
                       节点间只做梯度 all-reduce(量小、且本就要做)
  代价:显存只降到 1/8 而非 1/32(每节点内复制了 4 份)
```

**这是显存↔通信的又一次权衡的具象化:** 当跨节点带宽是瓶颈时,牺牲一部分显存收益(1/8 而非 1/N),换取把昂贵的参数收集限制在节点内的高速链路上。**选型规则:** 卡数 ≤ 单节点 → FULL_SHARD;卡数远超单节点且跨节点带宽紧张 → 考虑 HYBRID_SHARD。

## 7. 设计权衡与常见坑(汇总)

**核心权衡(一句话):ZeRO 三级是同一根"显存↔通信"滑杆上的三个刻度。** 越往 ZeRO-3 走,显存越省(16Ψ → 16Ψ/N),通信越多(1× → 1.5× DP),对互联带宽越敏感。

1. **选 stage 的规则:** 用第 01 章账本算 `16Ψ/N + 激活`,能放下就停在能放下的最低 stage。优先级:DDP(够用最好,通信最省、最简单)< ZeRO-2(免费降显存)< ZeRO-3(真装不下才上,多 50% 通信)。

2. **ZeRO-1/2 是免费午餐,ZeRO-3 不是。** ZeRO-1/2 通信量 ≈ DP,ZeRO-3 = 1.5× DP。所以别无脑全开 FULL_SHARD。

3. **wrap 粒度(auto_wrap)是 ZeRO-3 第一杀手坑。** 不 wrap = 整模型一次 all-gather = 临时峰值等于完整模型,完全失去意义。一般按 Transformer block wrap。

4. **优化器必须在 FSDP 包装后创建。** 否则优化器状态按完整参数分配,不是 1/N,显存白省。

5. **梯度归约用 fp32(reduce_dtype)。** bf16/fp16 归约会丢精度,训练发散。

6. **跨节点慢链路 → ZeRO-3 通信藏不住。** 小 batch + 慢互联时计算/通信比低,通信暴露;用 HYBRID_SHARD 或加大 batch 或退 ZeRO-2。

7. **checkpoint 和 FSDP 粒度要对齐。** 否则重算前向时重复 unshard,多通信甚至 OOM。

8. **ZeRO 不切激活。** 长序列的激活显存得靠激活重计算/序列并行(第 08 章)解决,别指望 ZeRO。

9. **临时峰值 ≠ 常驻显存。** OOM 经常发生在某个大单元 unshard 出完整参数的瞬间,而不是常驻态。调 `limit_all_gathers=True` 限制在途 all-gather 数能压峰值。

10. **CPU offload 是更激进的一档(本章未展开):** ZeRO 还能把分片好的优化器状态/参数 offload 到 CPU 内存甚至 NVMe(ZeRO-Offload / ZeRO-Infinity),进一步降 GPU 显存,代价是 PCIe 传输,通常只在显存极度受限时用。

## 8. 动手练习

**练习 1(估算题,核心)——用第 01 章账本算三级单卡显存。**
模型 `Ψ = 13×10⁹`(13B),fp16 混合精度 + Adam(每参 16 字节,其中优化器状态 12),`N = 16` 张卡,先忽略激活。
(a) 算 DP、ZeRO-1、ZeRO-2、ZeRO-3 各自的单卡模型显存(GB)。
(b) 单卡 80GB,哪些 stage 能放下(仍忽略激活)?
(c) 若激活峰值固定为 20GB,加上 ZeRO-3 临时收集一个 block 的完整参数(设每 block ≈ Ψ/40),ZeRO-3 单卡峰值是多少?还放得下吗?
*提示:* DP=16Ψ,ZeRO-1=4Ψ+12Ψ/N,ZeRO-2=2Ψ+14Ψ/N,ZeRO-3=16Ψ/N。1 字节单位换算 GB 除以 10⁹(用十进制 GB 即可)。临时峰值 = 常驻 + 激活 + 一个 block 完整参数(2 × Ψ/40 字节)。

**练习 2(推导题)——重新推导通信比。**
(a) 从 "all-reduce = reduce-scatter + all-gather,各约 1 个单位 M" 出发,写出 DP 和 ZeRO-3 每步每卡的通信量(用 Ψ 表示),验证比值 = 1.5。
(b) 如果优化器换成 SGD with momentum(优化器状态只有 fp32 主副本 4 + momentum 4 = 8 字节/参,K=8),ZeRO-3 的**显存比** vs DP 变成多少?**通信比**变不变?为什么?
*提示:* (b) 显存:DP=2+2+8=12Ψ,ZeRO-3=12Ψ/N;但通信量只和被通信的 fp16 参数/梯度(各 2Ψ)有关,与优化器状态 dtype 无关——所以通信比仍是 1.5。这道题考你"显存账"和"通信账"是两本独立的账。

**练习 3(编码/阅读题)——给 FSDP 时间线打桩。**
用 `torch.cuda.memory_allocated()` 在一个 2 层小 Transformer 上,分别在 (i) 进入每个 block 前向前、(ii) 该 block 前向后、(iii) 反向前、(iv) 反向后 打印显存。用 FSDP `FULL_SHARD` 包装,观察是否出现 §3 时间线预测的"unshard 涨、reshard 落、反向再涨再落"的锯齿。
*提示:* 用 forward/backward hook 或直接在 block 的 forward 里插 print。关掉 prefetch(`forward_prefetch=False`)更容易看清锯齿;对比开 prefetch 时锯齿是否变模糊(因为通信和计算重叠了)。

**练习 4(分析题)——什么时候 ZeRO-3 反而更慢?**
设单层 all-gather 参数耗时 `t_comm = 2Ψ_layer / B`(B 为带宽),单层前向计算耗时 `t_compute = 2Ψ_layer·s / F`(s 为该 micro-batch 的 token 数,F 为算力)。
(a) 写出"通信能被计算完全藏住"的条件(用 s、B、F 表示)。
(b) 给定 B 跨节点骤降为节点内的 1/10,要维持藏得住,s 需要怎么变?这解释了什么实战现象?
*提示:* (a) 条件是 `t_compute ≥ t_comm`,化简得 `s ≥ F/B`(Ψ_layer 约掉)。(b) B 降 10 倍,F/B 涨 10 倍,s 要涨 10 倍才能藏住——即跨慢节点要用大得多的 batch,否则通信暴露。这就是 §6 HYBRID_SHARD 存在的理由。

## 9. 源码 / 论文导读

**论文:**
- **ZeRO 原论文**:Rajbhandari et al., *"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"* (SC 2020)。重点读:第 5 节对 Pos / Pos+g / Pos+g+p 三级的显存公式(它用 `Ψ`、`K`=12 这套记号,和本章一致),以及第 7 节的通信量分析(它给出 ZeRO-3 为 1.5× baseline 的论证)。**先看那张三级显存对比表,和本章 §2.4 对照。**
- **FSDP 论文/技术报告**:Zhao et al., *"PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel"* (VLDB 2023)。重点读:FlatParameter 设计动机、unshard/reshard 与 autograd hook 的集成、通信预取与 rate limiter(对应本章 §5)。
- **ZeRO-Offload / ZeRO-Infinity**(第 7 节坑里提到的 CPU/NVMe offload):Ren et al. 2021 / Rajbhandari et al. 2021,显存极限场景再读。

**PyTorch 源码(按本章顺序读,效率最高):**
- `torch/distributed/fsdp/_flat_param.py` → `FlatParamHandle`:看 `flatten_tensors`、`shard`、`_get_unflat_views`,理解 §5.1 的拍扁与切分。
- `torch/distributed/fsdp/_runtime_utils.py` → 前/后向 hook(`_pre_forward`/`_post_forward`/`_pre_backward_hook`/`_post_backward_hook`)、`_unshard`、梯度归约、`_prefetch_handle`:**这是 FSDP 的心脏**,对照本章 §3 时间线和 §5.2/5.3 逐函数过一遍。具体函数名随版本会有出入「待核」,以本地源码为准。
- `torch/distributed/fsdp/wrap.py` → `transformer_auto_wrap_policy`、`size_based_auto_wrap_policy`:理解 §3/§5.4 的 wrap 粒度。
- `torch/distributed/fsdp/api.py` → `ShardingStrategy`、`MixedPrecision`、`BackwardPrefetch` 的枚举定义和 docstring,把本章每个配置项和源码注释对上。
- FSDP2:`torch/distributed/fsdp/_fully_shard/` 目录(`fully_shard` 函数式 API + DTensor 实现),新代码以此为准。

**DeepSpeed 对照(可选):** `deepspeed/runtime/zero/stage_1_and_2.py` 和 `stage3.py`——看另一套独立实现如何处理同一套机制,尤其 stage3 的 `_all_gather_params` 和 partition 逻辑,和 FSDP 的 unshard 是同一回事的不同写法。

## 10. 小结与承上启下

这一章把"数据并行如何拿到模型并行的显存效率"讲透了。三句话收束:

1. **ZeRO = 干掉 DP 的冗余。** DP 下 N 卡存 N 份相同的 参数/梯度/优化器状态;ZeRO 把它们各切成 1/N。ZeRO-1 切优化器状态(显存 16Ψ→4Ψ+12Ψ/N),ZeRO-2 再切梯度(→2Ψ+14Ψ/N),ZeRO-3 连参数也切(→16Ψ/N,正比 1/N)。

2. **ZeRO-3 用"即用即取、用完即弃"换显存。** 前向/反向到某单元时 all-gather 出完整参数、算完立刻 reshard 释放,反向再 reduce-scatter 把梯度归约回属主分片。代价是参数每步被收集两次,通信量 = 1.5× DP;但靠双 stream 预取把通信藏进计算,wall-clock 几乎不变——**前提是计算/通信比够高,跨慢节点会失效**。

3. **FSDP 是它的 PyTorch 原生身。** FlatParameter 把单元内参数拍扁成一根一维张量做高效集合通信;unshard/reshard 挂在 autograd 的前/后向 hook 上;通信预取 + rate limiter 控速度与峰值;`FULL_SHARD`/`SHARD_GRAD_OP`/`HYBRID_SHARD` 对应 ZeRO-3 / ZeRO-2 / 节点内切+节点间复制。

**承上:** 本章的显存账完全建立在第 01 章那本"每参 16 字节"的账上,通信量推导建立在第 02 章 Ring 集合通信的带宽结论上。**启下:** ZeRO 解决了 参数/梯度/优化器 的显存,但留了两个尾巴——(1) **激活**它不管,这是第 08 章激活重计算的领地;(2) 本章混合精度里 `param_dtype` / `reduce_dtype` 的取舍只点到为止,第 07 章会把 FP16/BF16/FP8 的数值稳定性彻底讲清。再往后,第 06 章会把 ZeRO(数据并行轴)和张量并行、流水线并行**正交组合**成 3D 并行,告诉你"给定模型 + 集群,各并行度该怎么配"——那时你会发现,ZeRO-3 通常就是 3D 并行里的"数据并行那一维"的现代实现。
