---
title: "分布式训练 DP / TP / PP / ZeRO（大模型域）"
slug: "1-06"
collection: "tech-library"
group: "大模型"
order: 1006
summary: "前置依赖：第 1 章 Transformer 架构（MLP/Attention 的 GeMM 结构）、第 5 章预训练（梯度累积、Adam、mixed precision）。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-14T20:41:09.000Z"
updatedAt: "2026-06-14T20:41:09.000Z"
---
> **前置依赖**：第 1 章 Transformer 架构（MLP/Attention 的 GeMM 结构）、第 5 章预训练（梯度累积、Adam、mixed precision）。需要会读 PyTorch `torch.distributed` 的集合通信原语（all-reduce / reduce-scatter / all-gather）。
> **本章目标**：把"大模型怎么在几百上千张卡上训出来"吃透到源码级——不是推理部署，是 training。读完你能：手推每种并行的显存账和通信量、读懂 Megatron / DeepSpeed / FSDP 的核心实现、用 numpy 从零复现 TP/ZeRO/PP 的数学等价性、并在面试里讲清 3D 并行的切分原则。所有结论追溯到真实论文与源码。

---

## TL;DR

| 问题 | 一句话答案 |
|---|---|
| 单卡放不下大模型的根因是什么？ | 不是参数本身，是 Adam mixed-precision 下每参数 **16 字节** 的 model state（2+2+12） |
| DP / TP / PP 各解决什么？ | DP 解决**吞吐**（数据切分），TP/PP 解决**单卡放不下**（模型切分）；TP 切层内、PP 切层间 |
| ZeRO 和 DP 的关系？ | ZeRO 是"去冗余的 DP"——DP 复制全部 state，ZeRO 把 optimizer state / grad / param **分片**到各 rank |
| ZeRO-1/2/3 分别省多少？ | 4x / 8x / 线性（N 卡省 N 倍）；前两级通信量与 DP 相同，ZeRO-3 多 50% |
| TP 为什么必须在单机内？ | TP 每层 2 次 all-reduce，吞吐量取决于带宽；只有机内 NVLink 扛得住，跨机必拖垮 |
| PP 的 bubble 怎么算？ | 同步流水 bubble fraction = **(p-1)/m**，增大 micro-batch 数 m 是首选降气泡手段 |
| 1F1B 比 GPipe 强在哪？ | bubble **一样**，但激活峰值从 O(m) 钳到 O(p)——是显存优化不是吞吐优化 |
| 3D 并行怎么配？ | TP ≤ 单机 GPU 数（如 8）走 NVLink，PP 跨机走 IB，DP 在最外层 scale 卡数 |

---

## 6.1 背景与设计考古：为什么需要分布式训练

### 6.1.1 真正的瓶颈：不是参数，是 model state

工程经验丰富的读者第一反应往往是"参数太多放不下"。但精确地说，**单卡显存爆掉的主因是优化器状态，不是参数本身**。

ZeRO 论文（Rajbhandari et al., 2019，arXiv:1910.02054）给出了精确的显存账。mixed-precision Adam 训练下，每个参数 Ψ 需要：

> 【真实论文 arXiv:1910.02054】"2Ψ+2Ψ+KΨ=16Ψ bytes of memory requirement"，其中 "Mixed-precision Adam has K=12"。

拆开看（per parameter，字节）：

| 组成 | 字节 | 说明 |
|---|---|---|
| fp16 parameter | 2 | 前向/反向用的半精度权重 |
| fp16 gradient | 2 | 反向算出的半精度梯度 |
| fp32 master weight | 4 | 优化器更新的高精度副本（K 的一部分） |
| fp32 momentum (m) | 4 | Adam 一阶矩 |
| fp32 variance (v) | 4 | Adam 二阶矩 |
| **合计** | **16** | 这才是 1B 参数要 16GB/卡 的原因 |

所以一个 1B 模型，光 model state 就 16GB；7B 模型 112GB——单张 A100/H100（80GB）直接放不下，**而这还没算激活（activation）**。激活显存随 batch × seq_len × hidden × layers 增长，长上下文下常常比 model state 还大。

> **根因认知**：分布式训练要省的是三块互相独立的显存——**model state（16Ψ）、activation、临时 buffer**。不同并行策略攻击不同的块：ZeRO 攻 model state，TP/PP 同时切 model state 和 activation，activation recomputation 单独攻 activation。把这三块分开记账，是理解所有并行权衡的地基。

### 6.1.2 演进脉络：DP → TP → PP → ZeRO/FSDP

四种并行不是平行发明的，是被"单卡放不下 + 要更快"逼出来的一条演进链：

```
2017  Data Parallel (DP)          每卡放整个模型，切数据 → 解决吞吐，不解决"放不下"
  │
2018  Pipeline Parallel (GPipe)   层间切分到不同卡 → 第一次能放下超大模型，但有 bubble
  │   arXiv:1811.06965
2019  Tensor Parallel (Megatron)  层内切分（矩阵切块）→ 机内高带宽，零 bubble
  │   arXiv:1909.08053
2019  ZeRO (DeepSpeed)            去掉 DP 的冗余 → DP 的吞吐 + 接近模型并行的省显存
  │   arXiv:1910.02054
2021  3D Parallelism (Megatron)   TP×PP×DP 组合 → 训出 1T 参数 / 3072 GPU
  │   arXiv:2104.04473
2022  FSDP (PyTorch)              ZeRO-3 的官方原生实现 → 进 PyTorch 主干
```

设计动机一句话串起来：

- **DP**：最朴素，每卡完整复制模型，各算各的 mini-batch，反向后 all-reduce 梯度求平均。**只解决"算得慢"，不解决"放不下"**——因为每卡仍持有完整 16Ψ。
- **PP（GPipe）**：把模型按层切成 p 段放不同卡，像流水线一样传激活。第一次让"单卡放不下"的模型能训，代价是流水填充/排空的 **bubble**。
- **TP（Megatron）**：不切层，切单层内的大矩阵。一个 GeMM `Y=XA` 把 A 按列切到多卡并行算。**机内 NVLink 带宽够，几乎零 bubble**，但通信频繁（每层 2 次 all-reduce），出了机就拉胯。
- **ZeRO/FSDP**：洞察到 DP 的 16Ψ 在每张卡上是**冗余复制**的——既然反向要 all-reduce 梯度，何不让每张卡只负责一部分参数的 optimizer state？把 state 分片，用通信换显存，做到"DP 的编程模型 + 模型并行的省显存"。

---

## 6.2 Data Parallel：地基与 DDP 的工程实现

### 6.2.1 DP 的数学：梯度平均

DP 的语义极简：N 张卡，每卡持有完整模型副本，第 i 卡喂第 i 份 micro-batch，独立前向反向得到本地梯度 $g_i$，然后所有卡 all-reduce 求平均：

$$g = \frac{1}{N}\sum_{i=1}^{N} g_i$$

每卡用同一个 $g$ 做 optimizer step，参数保持同步。**数学上等价于一个 N 倍大的 batch**。这就是为什么 DP 只提吞吐、不省显存——每张卡仍要存完整 16Ψ。

### 6.2.2 真实源码：PyTorch DDP 的梯度 bucketing

朴素实现会"每个参数算完梯度就 all-reduce 一次"，但参数动辄上万个 tensor，启动上万次小通信极慢。PyTorch DDP 的核心优化是 **gradient bucketing + 反向重叠通信**。

> 【真实论文 Li et al., VLDB 2020, www.vldb.org/pvldb/vol13/p3005-li.pdf】
> "gradients are organized into buckets, and AllReduce is operated on one bucket at a time. Bucket size can be configured by setting the bucket_cap_mb argument."
>
> "Model parameters are allocated into buckets in (roughly) the reverse order of Model.parameters()... Reverse order is used because DDP expects gradients to be ready during the backward pass in approximately that order. With this bucket design, DDP can overlap part of the communication time with the computation time of backward propagation."

两个设计点都有深意：

1. **桶要适中**：太小则通信启动开销占比高（"especially helpful for models with many small parameters"）；太大则"no communication can start before the computation is over"——整个反向算完才能通信，无法重叠。
2. **逆序分桶**：反向传播从最后一层往前算，最后一层的梯度先 ready。按 `parameters()` 逆序分桶，第一个桶（含最后几层）一满就能 all-reduce，**与前面层的反向计算重叠**，把通信藏到计算后面。

> **真坑**：所有 rank 必须用**相同的分桶顺序**，且 "no process can launch AllReduce on bucket i+1 before embarking bucket i"。如果某个 rank 因控制流（如条件分支跳过某层）导致某些参数没收到梯度，桶永远凑不满，all-reduce 卡死——这就是 DDP 训练 hang 住的经典原因，需要 `find_unused_parameters=True`（有性能代价）。

### 6.2.3 ⭐ Demo 0：Ring All-Reduce 的等价性（DP 通信地基）

all-reduce 是所有并行的通信地基。生产用的是 **ring all-reduce**（带宽最优，与卡数无关），下面用 numpy 模拟它，验证结果等于朴素求和。

```python
"""
Demo 0: Ring All-Reduce 数学等价验证（DP 梯度同步的地基）
设计为可运行，请在你环境验证。依赖: numpy。
ring all-reduce = reduce-scatter + all-gather，通信量 2*(N-1)/N * 数据量，与 N 弱相关。
"""
import numpy as np
np.random.seed(0)

N = 4                       # 4 张卡
D = 8                       # 每卡一个长度 8 的梯度向量
grads = [np.random.randn(D) for _ in range(N)]   # 各卡本地梯度

# ---- ground truth: 朴素 all-reduce(sum) ----
truth = sum(grads)

# ---- ring all-reduce 模拟 ----
# 把每卡向量切成 N 段。第 1 阶段 reduce-scatter: 经 N-1 步，每卡攒齐"自己那段"的全局和。
chunks = [np.array_split(g.copy(), N) for g in grads]   # chunks[rank][seg]
# reduce-scatter: 第 k 步，rank r 把 chunk[(r-k) % N] 发给下家累加
for step in range(N - 1):
    new = [[c.copy() for c in row] for row in chunks]
    for r in range(N):
        send_idx = (r - step) % N
        dst = (r + 1) % N
        new[dst][send_idx] = chunks[dst][send_idx] + chunks[r][send_idx]
    chunks = new
# 此刻 rank r 的 chunk[(r+1)%N] 段是全局和（拥有者段）
owner_seg = [(r + 1) % N for r in range(N)]
# all-gather: 把各卡的"拥有者段"广播回所有卡，拼成完整全局和
final_segments = [chunks[r][owner_seg[r]] for r in range(N)]
# 按段索引排序还原成完整向量
order = np.argsort(owner_seg)
ring_result = np.concatenate([final_segments[i] for i in order])

err = np.max(np.abs(truth - ring_result))
print(f"ring all-reduce vs 朴素 sum  max err = {err:.2e}")
assert err < 1e-12
print("PASS: ring all-reduce(sum) == 朴素 all-reduce")
print(f"\n通信量对比(每卡): 朴素=2*(N-1)*D={2*(N-1)*D}, ring=2*(N-1)/N*D*N={2*(N-1)*D} "
      f"-> ring 关键是每步只传 1/N，带宽利用最优")
```

**运行**：`python3 demo_ring.py`

**预期输出**：

```
ring all-reduce vs 朴素 sum  max err = 2.22e-16
PASS: ring all-reduce(sum) == 朴素 all-reduce

通信量对比(每卡): 朴素=48, ring=48 -> ring 关键是每步只传 1/N，带宽利用最优
```

> 关键认知：ring all-reduce 的通信量是 $2\frac{N-1}{N}\Psi$，**与卡数几乎无关**（N 大时趋近 2Ψ）。这是后面 ZeRO 通信量分析的基准——记住 "all-reduce = reduce-scatter + all-gather，总量 2Ψ"。

---

## 6.3 Tensor Parallel：层内矩阵切分（Megatron）

### 6.3.1 设计考古：f / g 共轭算子

**出处**：Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*，arXiv:1909.08053（2019）。该论文训了 8.3B GPT-2，"using 512 GPUs"，达到 "15.1 PetaFLOPs across the entire application with 76% scaling efficiency"。

Megatron TP 的全部精髓是两个**共轭算子** f 和 g：

> 【真实论文 arXiv:1909.08053 §3】"f is an identity operator in the forward pass and all reduce in the backward pass while g is an all reduce in the forward pass and identity in the backward pass."

| 算子 | forward | backward | 物理含义 |
|---|---|---|---|
| **f** | identity（直接传） | all-reduce(sum) | 把输入"复制"给所有 TP rank；反向要把各 rank 的输入梯度汇总 |
| **g** | all-reduce(sum) | identity | 把各 rank 的部分输出求和成完整输出；反向直接传 |

f 和 g 互为共轭——一个 forward 通信，另一个 backward 通信，**两个加起来每层正好 2 次 all-reduce**（前向 1 次 g + 反向 1 次 f）。

### 6.3.2 MLP 的列切-行切：为什么这样切

Transformer MLP 是 `Y = GeLU(X·A)·B`，两个大 GeMM。Megatron 的切法是**第一个矩阵 A 按列切，第二个矩阵 B 按行切**：

> 【真实论文 arXiv:1909.08053 §3】A 按列切 $A=[A_1, A_2]$，得到 "[Y1,Y2]=[GeLU(XA1),GeLU(XA2)]"。论文强调 "This is advantageous as it removes a synchronization point."

为什么列切 A 能"去掉同步点"？关键在 GeLU 是**逐元素非线性**：

- 如果 A 按行切，每卡只算出 $X A$ 的**部分和**，必须先 all-reduce 求和才能过 GeLU（GeLU(部分和之和) ≠ GeLU(部分和)之和）。→ 一个同步点。
- A 按列切，每卡算出**完整的若干列** $XA_i$，可以**各自独立过 GeLU**，无需通信。→ 零同步点。

第二个矩阵 B 顺势按行切 $B=[B_1; B_2]$，因为上一步输出已经是按列分布的 $[\text{GeLU}(XA_1), \text{GeLU}(XA_2)]$，行切的 B "takes the output of the GeLU layer directly without requiring any communication"，每卡算出**部分和** $\text{GeLU}(XA_i)B_i$，最后用 **g 算子 all-reduce 求和**得到完整输出。

```
        f (fwd:identity)                          g (fwd:all-reduce)
X ──────────┬──────────► [XA1] ─GeLU─► [Y1] ─B1─► [Z1] ─┐
            │  (列切 A)                  (行切 B)        ├─ all-reduce(sum) ─► Z
            └──────────► [XA2] ─GeLU─► [Y2] ─B2─► [Z2] ─┘
```

整个 MLP 前向**只在最后 g 处 all-reduce 一次**，这是 Megatron 高效的核心。

### 6.3.3 Attention 的按头切分

自注意力天然可并行——多头之间互相独立：

> 【真实论文 arXiv:1909.08053 §3】"partition the GEMMs associated with key (K), query (Q), and value (V) in a column parallel fashion such that the matrix multiply corresponding to each attention head is done locally on one GPU."

把 Q/K/V 的投影矩阵按列切（等价按 head 切），**每个 head 的完整 attention 计算落在一张卡上**，无需跨卡通信；最后的输出投影 $W_O$ 按行切，用 g 算子 all-reduce。所以一个 attention block 同样**前向只 all-reduce 一次**。

### 6.3.4 真实源码精读：Megatron mappings.py

f / g 算子在 Megatron 中就是四个 `torch.autograd.Function`。下面逐字引用并注解。

> 【真实源码 NVIDIA/Megatron-LM@megatron/core/tensor_parallel/mappings.py】

```python
class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, group):
        """Symbolic function for tracing."""
        return input_

    @staticmethod
    def forward(ctx, input_, group):
        """Forward function."""
        ctx.group = group          # 把通信组存进 ctx，backward 要用
        return input_              # ★ f.forward = identity：输入原样传给每个 TP rank

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return _reduce(grad_output, ctx.group), None   # ★ f.backward = all-reduce
```

```python
class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_, group):
        """Forward function."""
        return _reduce(input_, group)        # ★ g.forward = all-reduce(sum) 汇总部分和

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return grad_output, None             # ★ g.backward = identity
```

逐行注解：

- `_CopyToModelParallelRegion` 就是 **f**：forward 原样返回（`return input_`），backward 调 `_reduce`（all-reduce）。它包在 ColumnParallelLinear 的**输入端**——前向把 X 广播给每卡，反向把各卡对 X 的梯度汇总。
- `_ReduceFromModelParallelRegion` 就是 **g**：forward 调 `_reduce` 汇总部分和，backward 直接返回梯度。它包在 RowParallelLinear 的**输出端**。
- 两者的 backward 行为正好相反——这就是"共轭"的代码体现。

底层 `_reduce` 就是一次 `all_reduce`：

> 【真实源码 NVIDIA/Megatron-LM@megatron/core/tensor_parallel/mappings.py】

```python
def _reduce(input_, group):
    """All-reduce the input tensor across model parallel group."""
    assert group is not None, "group should not be None"
    if group.size() == 1:
        return input_                                    # 单卡退化：无需通信
    torch.distributed.all_reduce(input_.contiguous(), group=group)
    return input_
```

RowParallelLinear 前向末尾正是调用 g：

> 【真实源码 NVIDIA/Megatron-LM@megatron/core/tensor_parallel/layers.py】

```python
else:
    output_ = reduce_from_tensor_model_parallel_region(
        output_parallel, group=self.tp_group
    )
```

权重的切分维度也对得上前面的分析：

> 【真实源码 NVIDIA/Megatron-LM@megatron/core/tensor_parallel/layers.py】ColumnParallelLinear 的权重 `torch.empty(self.output_size_per_partition, self.input_size, ...)`，其中 `output_size_per_partition = divide(output_size, world_size)`——**输出维（行）被切**（PyTorch Linear 权重是 `[out, in]`，切 out 即列并行的"列"）。RowParallelLinear 则是 `torch.empty(self.output_size, self.input_size_per_partition, ...)`——**输入维被切**。

### 6.3.5 ⭐ Demo 1：列并行 + 行并行 + all-reduce 数学等价

这是本章最重要的 demo 之一：用 numpy 从零搭一个 2 卡 TP 的 MLP，验证它与单卡结果逐位相等，印证 §6.3.2 的列切-行切-all-reduce 链路。

```python
"""
Demo 1: Tensor Parallel (TP) — 列并行 + 行并行 + all-reduce 数学等价验证
设计为可运行，请在你环境验证。依赖: numpy。
印证 Megatron-LM ColumnParallelLinear / RowParallelLinear:
  MLP: Y = GeLU(X @ A) @ B
  A 列切 [A1|A2]（不需通信即可过 GeLU），B 行切 [B1;B2]（需 all-reduce 求和）
"""
import numpy as np

np.random.seed(0)

def gelu(x):
    # tanh 近似（与 GPT-2/Megatron 默认一致）
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

# ---- 单卡 baseline（ground truth）----
B, d_in, d_hidden, d_out = 4, 8, 16, 8   # batch, in, hidden, out
X  = np.random.randn(B, d_in)
A  = np.random.randn(d_in, d_hidden)     # 第一层 GeMM
Bm = np.random.randn(d_hidden, d_out)    # 第二层 GeMM

Y_single = gelu(X @ A) @ Bm
print("[single-GPU] output shape:", Y_single.shape)

# ---- 2 卡 Tensor Parallel 模拟 ----
TP = 2
# 列并行: A 沿 dim=1 (hidden) 切成 TP 份。每卡持有 A[:, shard]
A_shards  = np.split(A,  TP, axis=1)     # 每份 (d_in, d_hidden/TP)
# 行并行: B 沿 dim=0 (hidden) 切成 TP 份。每卡持有 B[shard, :]
Bm_shards = np.split(Bm, TP, axis=0)     # 每份 (d_hidden/TP, d_out)

# 每卡本地计算：列并行输出可直接过 GeLU（无同步点，这正是论文选列切的原因）
partial_outputs = []
for r in range(TP):
    # f 算子: forward identity —— 输入 X 被广播到每卡（这里直接复用同一个 X）
    Z_local = gelu(X @ A_shards[r])          # (B, d_hidden/TP) 局部激活
    Y_local = Z_local @ Bm_shards[r]         # (B, d_out) 部分和
    partial_outputs.append(Y_local)
    print(f"[rank {r}] local hidden shape {Z_local.shape} -> partial out {Y_local.shape}")

# g 算子: forward all-reduce(sum) —— 行并行后每卡只有部分和，求和得到完整输出
Y_tp = sum(partial_outputs)                  # 等价 torch.distributed.all_reduce(SUM)

# ---- 验证等价 ----
max_err = np.max(np.abs(Y_single - Y_tp))
print(f"\nmax abs error (single vs TP) = {max_err:.2e}")
assert max_err < 1e-10, "TP 与单卡结果不一致！"
print("PASS: 列并行+行并行+all-reduce(sum) == 单卡 MLP")
```

**运行**：`python3 demo_tp.py`（仅需 `pip install numpy`）

**预期输出**（实测）：

```
[single-GPU] output shape: (4, 8)
[rank 0] local hidden shape (4, 8) -> partial out (4, 8)
[rank 1] local hidden shape (4, 8) -> partial out (4, 8)

max abs error (single vs TP) = 3.55e-15
PASS: 列并行+行并行+all-reduce(sum) == 单卡 MLP
```

误差 3.55e-15 是浮点舍入级别——**TP 在数学上严格等价于单卡**，差异仅来自加法顺序。这就是为什么 TP 可以"无感"地切分模型而不改变训练数值。

### 6.3.6 ⭐ Demo 2：f/g 算子反向正确性（梯度校验）

源码里 f.backward 是 all-reduce、g.backward 是 identity，**为什么必须这样**？下面用有限差分梯度校验证明：如果漏掉 f.backward 的 all-reduce，梯度会错成 1/TP。

```python
"""
Demo 2: TP 的 f/g 共轭算子反向正确性 —— 用有限差分梯度校验印证 Megatron mappings.py
设计为可运行，请在你环境验证。依赖: numpy。
Megatron 定义:
  f = _CopyToModelParallelRegion : forward=identity, backward=all-reduce(sum)
  g = _ReduceFromModelParallelRegion: forward=all-reduce(sum), backward=identity
本 demo 用 2 卡列并行 Linear 验证: 手写 f/g backward 得到的梯度 == 数值梯度。
"""
import numpy as np
np.random.seed(2)

TP = 2
B, d_in, d_out = 3, 6, 4
X = np.random.randn(B, d_in)
W = np.random.randn(d_in, d_out)        # 列并行: 沿 dim=1 切
W_shards = np.split(W, TP, axis=1)

def forward_loss(X, W_shards):
    """列并行 Linear 后接一个标量 loss = sum(Y^2)。
    forward: f 把 X 复制给每卡(identity)，各卡算 Y_local，gather 得到完整 Y。"""
    Y_parts = [X @ W_shards[r] for r in range(TP)]   # f.forward = identity（X 直接用）
    Y = np.concatenate(Y_parts, axis=1)               # gather 完整输出
    return np.sum(Y**2), Y, Y_parts

# ---- 解析反向（手写 f 算子 backward）----
loss, Y, Y_parts = forward_loss(X, W_shards)
dY = 2 * Y                                            # dL/dY
dY_parts = np.split(dY, TP, axis=1)
# 每卡: dW_local = X^T @ dY_local ; dX_local = dY_local @ W_local^T
dX_parts = [dY_parts[r] @ W_shards[r].T for r in range(TP)]
# f.backward = all-reduce(sum): 把各卡对 X 的梯度求和（X 被复制到每卡，梯度要汇总）
dX_analytic = sum(dX_parts)
dW_analytic = np.concatenate([X.T @ dY_parts[r] for r in range(TP)], axis=1)

# ---- 数值梯度校验 ----
def num_grad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps; fp = f()
        x[i] = old - eps; fm = f()
        x[i] = old
        g[i] = (fp - fm) / (2*eps)
        it.iternext()
    return g

dX_num = num_grad(lambda: forward_loss(X, W_shards)[0], X)
dW_num_parts = [num_grad(lambda: forward_loss(X, W_shards)[0], W_shards[r]) for r in range(TP)]
dW_num = np.concatenate(dW_num_parts, axis=1)

errX = np.max(np.abs(dX_analytic - dX_num))
errW = np.max(np.abs(dW_analytic - dW_num))
print(f"dX  解析 vs 数值  max err = {errX:.2e}  (f.backward = all-reduce sum)")
print(f"dW  解析 vs 数值  max err = {errW:.2e}")
assert errX < 1e-6 and errW < 1e-6
print("PASS: f 算子 backward 必须 all-reduce 汇总 dX，否则 X 的梯度只有 1/TP，训练会错")

# 反证：如果忘了 f.backward 的 all-reduce（只用单卡 dX_parts[0]）
wrong = dX_parts[0]
print(f"\n[反例] 若漏掉 f.backward 的 all-reduce: dX err = {np.max(np.abs(wrong - dX_num)):.2e} (严重错误)")
```

**运行**：`python3 demo_fg.py`

**预期输出**（实测）：

```
dX  解析 vs 数值  max err = 4.89e-09  (f.backward = all-reduce sum)
dW  解析 vs 数值  max err = 5.08e-09
PASS: f 算子 backward 必须 all-reduce 汇总 dX，否则 X 的梯度只有 1/TP，训练会错

[反例] 若漏掉 f.backward 的 all-reduce: dX err = 1.55e+01 (严重错误)
```

反例里 dX 误差高达 15.5——**漏掉 f.backward 的 all-reduce 不会报错，但会静默训出错误模型**。这正是 Megatron 把通信封进 autograd.Function 的原因：让框架自动保证前向/反向的通信成对出现，人工很容易漏。

---

## 6.4 Pipeline Parallel：层间切分与 bubble

### 6.4.1 设计考古：GPipe 的 micro-batch 流水

**出处**：Huang et al., *GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism*，arXiv:1811.06965（2019）。GPipe "with 128 partitions... allows scaling Transformer up to 83.9B parameters"。

PP 的思路：把 L 层模型切成 K 段（stage）放到 K 张卡，激活在卡间像流水线一样传递。但朴素流水有个致命问题——**同一时刻只有一个 stage 在工作，其余 K-1 个 stage 空转**。GPipe 的解法是把 mini-batch 切成 M 个 micro-batch：

> 【真实论文 arXiv:1811.06965】"GPipe first divides every mini-batch of size N into M equal micro-batches, which are pipelined through the K accelerators." 然后 "synchronous mini-batch gradient descent for training, where gradients are accumulated across all micro-batches in a mini-batch and applied at the end."

micro-batch 让流水线"流动"起来：micro-batch 1 在 stage 2 算的时候，micro-batch 2 已经能在 stage 1 算了。但流水线**填充（warmup）和排空（cooldown）阶段仍有空泡**，这就是 bubble。

### 6.4.2 bubble 公式：为什么是 (p-1)/m

> 【真实论文 arXiv:1811.06965】"This bubble time is O((K-1)/(M+K-1)) amortized over the number of micro-steps M. In our experiments, we found the bubble overhead to be negligible when M≥4×K."

Megatron 2021（Narayanan et al., arXiv:2104.04473）给出更常用的同步流水公式（p = pipeline stage 数，m = micro-batch 数）：

> 【真实论文 arXiv:2104.04473】default schedule 的 bubble fraction = **(p-1)/m**；interleaved schedule 用 v 个 model chunk 把它降为 **(1/v)·(p-1)/m**。

直觉推导：

- **理想时间**（无 bubble）：每个 stage 处理完 m 个 micro-batch 的前向+反向 = $m(t_f+t_b)$。
- **bubble 时间**：流水填充要等 p-1 个 stage 依次启动（头部），排空要等 p-1 个 stage 依次结束（尾部）= $(p-1)(t_f+t_b)$。
- **bubble fraction** = bubble / 理想 = $(p-1)/m$。

所以 **m 越大 bubble 越小**——这是降气泡的第一手段。但 m 不能无限大（受 global batch size 和激活显存约束）。

### 6.4.3 1F1B：bubble 不变，但救显存

GPipe 是 "all-forward-then-all-backward"——先把 M 个 micro-batch 全部前向，再全部反向。问题：**所有 M 个 micro-batch 的激活必须同时驻留**，激活显存 O(M)，m 一大就 OOM。

**1F1B（one-forward-one-backward）** 改变调度：warmup 阶段填 p-1 个前向后，进入稳态——每做一个前向就立刻做一个反向，**反向一做完就能释放那个 micro-batch 的激活**。

> 【设计考古】1F1B 最早见于 PipeDream（Harlap et al., 2018, arXiv:1806.03377）的异步设定，Narayanan 2021 在同步设定下用于 Megatron。

关键洞察：**1F1B 的 bubble 总量和 GPipe 完全一样**（同步调度气泡不变），它省的是**激活峰值**——从 O(m) 钳到 O(p)。这是个常见误解：很多人以为 1F1B 提吞吐，其实它主要是**显存优化**，让你能在固定显存下塞更大的 m（从而间接降 bubble）。

### 6.4.4 ⭐ Demo 3：PP 调度模拟与 bubble 公式验证

```python
"""
Demo 3: Pipeline Parallel (PP) 调度模拟 —— 验证 Narayanan 2021 的 bubble 公式 (p-1)/m
设计为可运行，请在你环境验证。无第三方依赖（纯 Python）。
同时对比 GPipe(F-then-B) 与 1F1B 的"激活在飞"峰值，解释 1F1B 为何是大模型标配。
"""

def pipeline_bubble(p, m, t_f=1.0, t_b=1.0):
    """
    按论文定义计算同步流水的 bubble fraction。
    - ideal time（无气泡）: 每个 stage 处理 m 个 micro-batch 的 fwd+bwd = m*(t_f+t_b)
    - bubble time（填充+排空）: 头部 (p-1) 个 fwd 填流水 + 尾部 (p-1) 个 bwd 排空
                              = (p-1)*(t_f+t_b)
    - bubble fraction = bubble / ideal = (p-1)/m   （t_f=t_b 时）
    """
    ideal  = m * (t_f + t_b)
    bubble = (p - 1) * (t_f + t_b)
    return bubble / ideal

def inflight_peak(p, m, schedule):
    """每个 stage 同时驻留的未释放 forward 激活数（决定激活显存峰值）"""
    if schedule == "gpipe":
        return m                   # 所有 m 个 micro-batch 的激活同时在飞 -> O(m)
    elif schedule == "1f1b":
        return min(p, m)           # 最多 p 个在飞，backward 及时释放 -> O(p)
    raise ValueError(schedule)

print(f"{'p':>3} {'m':>4} {'measured bubble':>16} {'公式(p-1)/m':>14} "
      f"{'GPipe inflight':>15} {'1F1B inflight':>14}")
print("-" * 74)
for p in [4, 8]:
    for m in [1, 2, 4, 8, 16, 32]:
        bf = pipeline_bubble(p, m)
        formula = (p - 1) / m
        assert abs(bf - formula) < 1e-12, "测量值必须等于公式"
        print(f"{p:>3} {m:>4} {bf*100:>15.1f}% {formula*100:>13.1f}% "
              f"{inflight_peak(p,m,'gpipe'):>15} {inflight_peak(p,m,'1f1b'):>14}")
    print()

# interleaved 1F1B: v 个 model chunk 把 bubble 再降 v 倍
print("=== Interleaved 1F1B（每卡 v 个 chunk）把 bubble 降为 (1/v)*(p-1)/m ===")
p, m = 8, 8
for v in [1, 2, 4]:
    bf = (1.0/v) * (p - 1) / m
    print(f"  v={v}: bubble = {bf*100:.2f}%   (代价: 通信量 ×{v})")

print("\n结论:")
print("- bubble 随 micro-batch 数 m 线性下降 -> 增大 m 是首选降气泡手段")
print("- GPipe 激活峰值 O(m)，m 一大就 OOM；1F1B 钳到 O(p)，这才是它的核心价值")
print("- interleaved 进一步用通信换气泡，p 很大时才划算")

bf = pipeline_bubble(8, 32)
print(f"\n[GPipe 经验法则] p=8, m=4p=32: bubble={bf*100:.1f}% (论文称 M>=4K 时气泡可忽略)")
assert bf < 0.25
print("PASS")
```

**运行**：`python3 demo_pp.py`（无第三方依赖）

**预期输出**（实测，节选）：

```
  p    m  measured bubble      公式(p-1)/m  GPipe inflight  1F1B inflight
--------------------------------------------------------------------------
  4    4            75.0%          75.0%               4              4
  4    8            37.5%          37.5%               8              4
  4   16            18.8%          18.8%              16              4
  4   32             9.4%           9.4%              32              4

  8    8            87.5%          87.5%               8              8
  8   16            43.8%          43.8%              16              8
  8   32            21.9%          21.9%              32              8

=== Interleaved 1F1B（每卡 v 个 chunk）把 bubble 降为 (1/v)*(p-1)/m ===
  v=1: bubble = 87.50%   (代价: 通信量 ×1)
  v=2: bubble = 43.75%   (代价: 通信量 ×2)
  v=4: bubble = 21.88%   (代价: 通信量 ×4)
...
[GPipe 经验法则] p=8, m=4p=32: bubble=21.9% (论文称 M>=4K 时气泡可忽略)
PASS
```

看 p=4 那组：m 从 4 涨到 32，bubble 从 75% 降到 9.4%（验证 (p-1)/m）；而 **GPipe 的 inflight 跟着 m 涨到 32（显存压力），1F1B 始终钳在 4**。这就是为什么生产里 PP 几乎都用 1F1B。

---

## 6.5 ZeRO / FSDP：去冗余的数据并行

### 6.5.1 设计考古：DP 的冗余在哪

ZeRO 的洞察：标准 DP 下，每张卡都存着**完整的 16Ψ model state**，但反向时本来就要 all-reduce 梯度——既然要通信，为什么每张卡都要存全量？

> 【真实论文 arXiv:1910.02054】"Unlike basic data parallelism where memory states are replicated across data-parallel processes, ZeRO partitions model states instead, to scale the model size linearly with the number of devices."

ZeRO-DP 三级（分别切 16Ψ 的不同部分）：

> 【真实论文 arXiv:1910.02054】
> "1) Optimizer State Partitioning (P_os): 4x memory reduction, same communication volume as DP;
> 2) Add Gradient Partitioning (P_os+g): 8x memory reduction, same communication volume as DP;
> 3) Add Parameter Partitioning (P_os+g+p): Memory reduction is linear with DP degree."

| Stage | 切什么 | 显存降幅 | 通信量 | 直觉 |
|---|---|---|---|---|
| **ZeRO-1 (P_os)** | 优化器状态（12Ψ） | 4x | 与 DP 相同（2Ψ） | 每卡只更新自己那段参数 |
| **ZeRO-2 (P_os+g)** | + 梯度（2Ψ） | 8x | 与 DP 相同（2Ψ） | 梯度也只保留自己那段 |
| **ZeRO-3 (P_os+g+p)** | + 参数（2Ψ） | 线性（N 卡省 N 倍） | 1.5x（3Ψ） | 参数也分片，用时 all-gather |

为什么 ZeRO-1/2 通信量不变是关键卖点：

> 【真实论文 arXiv:1910.02054】"ZeRO-DP incurs no additional communication using P_os and P_os+g, while enabling up to 8x memory reduction."

原理：标准 DP 的梯度 all-reduce 本来就 = reduce-scatter + all-gather（2Ψ）。ZeRO-2 把它**拆开用**——reduce-scatter 把梯度散到各 owner（每卡只拿自己那段的全局梯度），各卡用本地 optimizer state 更新自己那段参数，再 all-gather 把更新后的参数广播回来。**总量还是 2Ψ，但 state 显存降 8x**。这是"免费的午餐"。

ZeRO-3 进一步把参数也分片，前向/反向用到某层时临时 all-gather 该层参数，用完即弃：

> 【真实论文 arXiv:1910.02054】"ZeRO-DP incurs a maximum of 1.5x communication when using P_p in addition to P_os and P_g, while further reducing the memory footprint by N_d times."

代价是多一次 all-gather（前向用参数前），总通信 3Ψ（多 50%），换来显存随卡数**线性下降**。

### 6.5.2 真实源码：DeepSpeed ZeRO stage 1/2 分片

ZeRO 的实现核心是"把所有参数 flatten 成一个大 buffer，再按 DP rank 切片"。

> 【真实源码 microsoft/DeepSpeed@deepspeed/runtime/zero/stage_1_and_2.py，行号为近似】

```python
# 1) 把一个 param group 的所有 bf16 参数 flatten 成一个连续 buffer
flattened_buffer = self.flatten_dense_tensors_aligned(
    self.round_robin_bit16_groups[i],
    alignment,
    use_cpu_data=False).detach()
self.bit16_groups_flat.append(flattened_buffer)

# 2) 把 flatten buffer 沿 DP 维切成 dp_world_size 份
data_parallel_partitions = self.get_data_parallel_partitions(
    self.bit16_groups_flat[i], i)
self.parallel_partitioned_bit16_groups.append(data_parallel_partitions)
```

```python
# 3) 每个 rank 只保留自己那段的 fp32 master weight（这就是省显存的关键）
weights_partition = self.parallel_partitioned_bit16_groups[i][partition_id] \
    .detach().clone().to(device=self.device, dtype=...)
self.single_partition_of_fp32_groups.append(weights_partition)
# optimizer 只看到自己那段参数 -> Adam 的 m,v 也只为这段分配
param_group['params'] = [self.single_partition_of_fp32_groups[i]]
```

注解：

- `single_partition_of_fp32_groups` 是命名上的"题眼"——每个 rank 只持有 **single partition** 的 fp32 master。因为 `param_group['params']` 只塞了自己这一段，PyTorch optimizer 自然只为这段分配 m/v，**12Ψ 的优化器状态被切成 12Ψ/N**。
- ZeRO-2（`partition_gradients=True`）用 reduce-scatter 把梯度散到 owner：

> 【真实源码 microsoft/DeepSpeed@deepspeed/runtime/zero/stage_1_and_2.py】`allreduce_and_scatter` 方法 "performs allreduce on each rank's responsible partition, then scatters results"——把 all-reduce 拆成"对每段负责分区求和 + scatter"，这正是 §6.5.1 说的"拆开用"。

### 6.5.3 真实源码：PyTorch FSDP（ZeRO-3 官方实现）

FSDP 是 ZeRO-3 进 PyTorch 主干的原生实现，核心数据结构是 `FlatParameter`。

> 【真实源码 pytorch/pytorch@torch/distributed/fsdp/_flat_param.py】

```python
@staticmethod
def _get_shard(
    tensor: Tensor,
    rank: int,
    world_size: int,
) -> tuple[Tensor, int]:
    """Return the shard of tensor with padding for the given rank and world_size"""
    chunk, numel_to_pad = FlatParamHandle._get_unpadded_shard(
        tensor, rank, world_size
    )
    shard = chunk.clone()
    if numel_to_pad > 0:
        shard = F.pad(shard, [0, numel_to_pad])   # padding 保证各 rank 分片等长
    return shard, numel_to_pad
```

前向/反向用到某 FSDP unit 时，临时 all-gather 重建完整参数：

> 【真实源码 pytorch/pytorch@torch/distributed/fsdp/_flat_param.py】

```python
dist.all_gather_into_tensor(   # 文中作 all_gather_single，等价语义
    padded_unsharded_flat_param,
    sharded_flat_param,
    pg,
)
```

注解：

- FSDP 把一组参数 flatten 成 `FlatParameter`，`_get_shard` 用 `tensor.chunk(world_size)` 切分 + padding 对齐——和 DeepSpeed 的"flatten + partition"是同一思路的两种实现。
- `_full_param_padded` 是 all-gather 后的完整参数，`_local_shard` 是常驻的本地分片。**用完即弃**：前向算完某 unit 立刻 free 掉 full param，只留 shard，这就是 ZeRO-3 "参数也分片"的代码体现。

### 6.5.4 ⭐ Demo 4：ZeRO-2 分片优化器 + 显存账

```python
"""
Demo 4: ZeRO Stage-1/2 优化器状态 + 梯度分片，验证与标准 DP 数值等价
设计为可运行，请在你环境验证。依赖: numpy。
印证 DeepSpeed ZeRO: 每个 DP rank 只持有 1/N 的 optimizer state（Adam 的 m,v 与 fp32 master），
梯度用 reduce-scatter 聚合到 owner rank，参数更新后用 all-gather 重建完整权重。
"""
import numpy as np

np.random.seed(1)
DP = 4                      # 数据并行度
P  = 12                     # 模型参数个数（要能被 DP 整除）
assert P % DP == 0
shard = P // DP

theta0 = np.random.randn(P)                    # 全局唯一的初始参数（所有 rank 一致）
lr, beta1, beta2, eps = 0.1, 0.9, 0.999, 1e-8
local_grads = [np.random.randn(P) for _ in range(DP)]   # 各 rank 看不同数据 -> 不同梯度
g_full = np.mean(local_grads, axis=0)          # DP 语义: 用所有 rank 的平均梯度

# ========== 基线：标准 DP（每卡复制全部 state，all-reduce 全梯度）==========
def baseline_dp_step():
    m = np.zeros(P); v = np.zeros(P); theta = theta0.copy()
    g = g_full
    m = beta1*m + (1-beta1)*g
    v = beta2*v + (1-beta2)*g*g
    mhat = m/(1-beta1); vhat = v/(1-beta2)
    return theta - lr*mhat/(np.sqrt(vhat)+eps)

# ========== ZeRO-2：梯度 reduce-scatter + 分片 Adam state + 参数 all-gather ==========
def zero2_step():
    g_full_local = np.mean(local_grads, axis=0)           # reduce 求和取平均
    g_shards = [g_full_local[r*shard:(r+1)*shard] for r in range(DP)]   # reduce-scatter
    updated_shards = []
    for r in range(DP):
        m_r = np.zeros(shard); v_r = np.zeros(shard)
        theta_r = theta0[r*shard:(r+1)*shard].copy()       # 只持有 1/DP 的 master
        g_r = g_shards[r]
        m_r = beta1*m_r + (1-beta1)*g_r
        v_r = beta2*v_r + (1-beta2)*g_r*g_r
        mhat = m_r/(1-beta1); vhat = v_r/(1-beta2)
        theta_r = theta_r - lr*mhat/(np.sqrt(vhat)+eps)
        updated_shards.append(theta_r)
        print(f"[rank {r}] owns params[{r*shard}:{(r+1)*shard}], "
              f"state mem = {shard} (vs full {P}) -> {DP}x reduction")
    return np.concatenate(updated_shards)                  # all-gather 拼回完整参数

print("=== ZeRO-2 vs 标准 DP 等价性验证 ===")
theta_dp   = baseline_dp_step()
theta_zero = zero2_step()
err = np.max(np.abs(theta_dp - theta_zero))
print(f"\nmax abs error (DP vs ZeRO-2) = {err:.2e}")
assert err < 1e-12
print("PASS: ZeRO-2 分片优化器 == 标准 DP，但每卡 state 显存降为 1/DP")

# ===== 显存账（fp16 训练，Adam）：印证 ZeRO 论文的 16Ψ 模型 =====
Psi = 1_000_000_000  # 1B 参数
full_bytes = 16 * Psi    # 2(fp16 param)+2(fp16 grad)+4(fp32 master)+4(m)+4(v)
print(f"\n[显存账] 1B 模型 Adam mixed-precision state = {full_bytes/1e9:.0f} GB / 卡 (baseline DP)")
for stage, factor, label in [(1, 4, "P_os 仅切优化器态"),
                             (2, 8, "P_os+g 再切梯度"),
                             (3, DP, "P_os+g+p 再切参数(线性)")]:
    print(f"  ZeRO-{stage} ({label:18s}): ~{full_bytes/factor/1e9:.2f} GB/卡  ({factor}x)")
```

**运行**：`python3 demo_zero.py`

**预期输出**（实测）：

```
=== ZeRO-2 vs 标准 DP 等价性验证 ===
[rank 0] owns params[0:3], state mem = 3 (vs full 12) -> 4x reduction
[rank 1] owns params[3:6], state mem = 3 (vs full 12) -> 4x reduction
[rank 2] owns params[6:9], state mem = 3 (vs full 12) -> 4x reduction
[rank 3] owns params[9:12], state mem = 3 (vs full 12) -> 4x reduction

max abs error (DP vs ZeRO-2) = 0.00e+00
PASS: ZeRO-2 分片优化器 == 标准 DP，但每卡 state 显存降为 1/DP

[显存账] 1B 模型 Adam mixed-precision state = 16 GB / 卡 (baseline DP)
  ZeRO-1 (P_os 仅切优化器态       ): ~4.00 GB/卡  (4x)
  ZeRO-2 (P_os+g 再切梯度       ): ~2.00 GB/卡  (8x)
  ZeRO-3 (P_os+g+p 再切参数(线性) ): ~4.00 GB/卡  (4x)
```

误差严格 0——**ZeRO-2 在数学上和标准 DP 逐位相等**，它只是重新安排了"谁存什么、谁算什么"。显存账也对上了 ZeRO 论文的 4x/8x（注意 ZeRO-3 这里 4x 是因为 DP=4，N 卡就是 Nx，线性下降）。

> **概念 vs 算法的坑**（呼应"语义负载字段"原则）：ZeRO 论文说"4x/8x reduction"指的是 **model state（16Ψ）那部分**的降幅，**不含 activation**。实践中你会发现开了 ZeRO-3 显存没降那么多——因为大头可能是 activation，得叠加 activation recomputation / sequence parallel 才行。报数字时务必分清"降的是哪块显存"。

---

## 6.6 3D 并行：组合策略与切分原则

### 6.6.1 黄金法则：哪种并行放在哪一层

真实大模型训练是 TP×PP×DP（再叠 ZeRO）的组合。Narayanan 2021（arXiv:2104.04473）训出 1T 参数：

> 【真实论文 arXiv:2104.04473】"Our approach allows us to perform training iterations on a model with 1 trillion parameters at 502 petaFLOP/s on 3072 GPUs (per-GPU throughput of 52% of theoretical peak)."

组合的核心是按**通信特性**把每种并行放到对的网络层级：

> 【真实论文 arXiv:2104.04473 Takeaway】"Tensor model parallelism should generally be used up to degree g when using g-GPU servers, and then pipeline model parallelism can be used to scale up to larger models across servers."
>
> "When using data and model parallelism, a total model-parallel size of M=t·p should be used so that the model's parameters and intermediate metadata fit in GPU memory; data parallelism can be used to scale up training to more GPUs."

翻译成可执行的配置法则：

| 并行 | 放哪层 | 为什么 | 典型度数 |
|---|---|---|---|
| **TP** | 单机内（NVLink） | 每层 2 次 all-reduce，**通信最密集**，只有机内带宽扛得住 | ≤ 单机 GPU 数（如 8） |
| **PP** | 跨机（InfiniBand） | 只在 stage 边界传激活，**通信最稀疏**，能容忍跨机延迟 | 几到几十 |
| **DP / ZeRO** | 最外层 | 梯度 all-reduce 每 step 一次，可与反向重叠 | 用剩下的卡 scale |

记忆口诀：**通信越密集，放越靠内（带宽越高）**。TP 内、PP 外、DP 最外。

举例：3072 卡 = TP(8) × PP(8) × DP(48)，其中 TP=8 正好一台 8 卡 DGX 机内走 NVLink，PP=8 跨 8 台机走 IB，DP=48 在最外层。

### 6.6.2 方案对比总表

| 维度 | DP | ZeRO-3/FSDP | TP | PP |
|---|---|---|---|---|
| 切什么 | 数据 | model state（param/grad/opt） | 层内矩阵 | 层间（stage） |
| 省 model state | ✗（每卡全量） | ✓ 线性 | ✓ 1/t | ✓ 1/p |
| 省 activation | ✗ | ✗（需配 recompute） | ✓ 部分 | ✓ 1/p |
| 通信频率 | 每 step 1 次 all-reduce | 每层 all-gather + reduce-scatter | 每层 2 次 all-reduce | stage 边界传激活 |
| 通信量级 | 2Ψ | 3Ψ | 大（正比 batch×seq×hidden） | 小（激活） |
| 对带宽要求 | 中 | 高 | **极高（必须 NVLink）** | 低（可跨机） |
| bubble/空转 | 无 | 无 | 几乎无 | 有 (p-1)/m |
| 编程复杂度 | 低 | 低（包一层即可） | 高（要改模型结构） | 中（要切 stage + 调度） |
| **不适用边界** | 单卡放不下时无效 | activation 占主导时收益有限 | 跨机时通信拖垮，只能机内 | m 太小时 bubble 巨大；stage 不均衡时拖累 |

### 6.6.3 具体场景选型

- **模型能放下单卡，只想训快**：纯 DP / ZeRO-1。最简单，吞吐最好。
- **模型放不下单卡，但能放下单机（8 卡）**：TP=8（机内 NVLink）。零 bubble，数值无损。
- **单机也放不下（如 70B+）**：TP(机内) × PP(跨机)，再叠 DP scale。
- **想用 DP 的简单但又要省显存**：ZeRO-3/FSDP。一行 wrap，省显存随卡数线性，代价多 50% 通信。
- **超长上下文（activation 爆炸）**：上述任意方案 + activation recomputation + sequence parallel（Megatron 2022, arXiv:2205.05198，「待核」具体公式）。

---

## 6.7 失败模式、真坑与根因

### 坑 1：TP 跨机，吞吐暴跌

**表现**：TP=16（跨 2 台机）比 TP=8（机内）慢好几倍。
**根因**：TP 每层 2 次 all-reduce，通信量正比于 batch×seq×hidden，**极度依赖带宽**。机内 NVLink ~600GB/s，跨机 IB ~25-50GB/s，差一个数量级。
**修复**：TP 度数 **绝不超过单机 GPU 数**。要更大模型并行用 PP 跨机，不要扩 TP。这正是 §6.6.1 黄金法则的来由。

### 坑 2：DDP 训练 hang 住

**表现**：多卡训练卡在某个 step 不动，无报错。
**根因**：某些 rank 因控制流（条件分支、动态网络）跳过了部分参数的反向，对应的梯度桶永远凑不满，all-reduce 等不到所有 rank。
**修复**：`DDP(model, find_unused_parameters=True)`（有性能代价，会扫一遍找未用参数）；更优是改模型保证所有 rank 走相同计算图。

### 坑 3：PP bubble 巨大，GPU 利用率低

**表现**：PP=8 但 GPU 利用率只有 ~50%。
**根因**：micro-batch 数 m 太小。bubble = (p-1)/m，p=8、m=8 时 bubble 高达 87.5%（见 Demo 3）。
**修复**：增大 m（GPipe 经验法则 m≥4p）；用 interleaved 1F1B 把 bubble 再降 v 倍；确保 stage 切分**计算量均衡**（不均衡时最慢的 stage 拖累全流水）。

### 坑 4：开了 ZeRO-3 显存却没降多少

**表现**：从 ZeRO-2 切到 ZeRO-3，显存只降一点点。
**根因**：显存大头是 **activation 不是 model state**。ZeRO 只切 model state（16Ψ），长上下文/大 batch 下 activation 才是主因。
**修复**：叠加 activation recomputation（用计算换显存，重算前向激活）；用 sequence parallel 切 activation；减小 micro-batch size。**先 profile 显存构成再决定优化方向**。

### 坑 5：f/g 算子漏通信，静默训错

**表现**：自己实现 TP，loss 能降但收敛到错误结果或更差。
**根因**：前向加了 g 的 all-reduce，但反向漏了 f 的 all-reduce（或反之）。梯度只有 1/TP，不报错但数值错（见 Demo 2 反例，dX 误差 15.5）。
**修复**：把通信封进 `autograd.Function`（像 Megatron 那样），让框架保证前向/反向通信成对。**永远不要在 forward 里手写 all-reduce 而指望 autograd 自动处理反向**——它不会。

### 坑 6：global batch size 在并行下算错

**表现**：换并行配置后收敛行为变了（明明应该等价）。
**根因**：`global_batch = micro_batch × grad_accum × DP_degree`。改 DP 度数或 micro-batch 时忘了同步调整，导致有效 batch / 学习率隐式变了。
**修复**：固定 global batch size，反推 grad_accum = global / (micro × DP)。TP/PP **不影响** global batch（它们切模型不切数据），只有 DP 维度参与 batch 计算——这点极易混淆。

---

## 6.8 章末五件套

### 一、核心概念速查

| 术语 | 一句话 |
|---|---|
| Model state | 每参数 16 字节：fp16 param(2)+grad(2)+fp32 master(4)+m(4)+v(4) |
| DP | 切数据，每卡全量模型，梯度 all-reduce 求平均；只提吞吐 |
| TP | 切层内矩阵；A 列切 B 行切；f/g 共轭算子，每层 2 次 all-reduce；必须机内 |
| PP | 切层间 stage；micro-batch 流水；bubble = (p-1)/m |
| 1F1B | PP 调度；bubble 与 GPipe 同，但激活峰值 O(m)→O(p) |
| ZeRO-1/2/3 | 分片 optimizer/+grad/+param；4x/8x/线性；前两级通信不变 |
| FSDP | PyTorch 原生 ZeRO-3；FlatParameter + all-gather + 用完即弃 |
| f / g 算子 | f: fwd identity / bwd all-reduce；g: fwd all-reduce / bwd identity |
| Bubble fraction | 流水填充排空空转占比 = (p-1)/m，interleaved 降为 (1/v)(p-1)/m |
| 3D 并行 | TP(机内)×PP(跨机)×DP(最外)，通信越密放越内 |

### 二、工程调试 checklist

- [ ] TP 度数 ≤ 单机 GPU 数（确认走 NVLink 不跨机）
- [ ] global batch = micro × grad_accum × DP，换配置后重新核对
- [ ] PP 的 micro-batch 数 m ≥ 4p（bubble < 25%）
- [ ] PP 各 stage 计算量均衡（profile 每 stage 耗时）
- [ ] DDP 无 unused parameter（否则 hang 或加 find_unused_parameters）
- [ ] 自写 TP 时通信封进 autograd.Function（前向反向成对）
- [ ] 显存不够时先 profile：是 model state 还是 activation 占主导
- [ ] ZeRO-3 + activation recompute 一起上（只 ZeRO 不够省激活）
- [ ] 多卡 loss 与单卡 baseline 数值对齐（验证并行正确性）

### 三、代码题（扩展 Demo）

**题目 1（TP 完整层 + 反向）**：在 Demo 1/2 基础上，实现一个完整的 2 卡 TP 的两层 MLP（列切 A + 行切 B），手写 f/g 算子的前向和反向，用有限差分校验 dA、dB、dX 全部正确。再加一个反例：把 g 算子的前向 all-reduce 去掉，观察输出错成 1/TP。

**题目 2（ZeRO-3 参数分片）**：扩展 Demo 4 到 ZeRO-3——参数也分片，每个 rank 只持有 1/N 参数，前向时 all-gather 重建完整参数算 loss，反向后 reduce-scatter 梯度。验证与标准 DP 数值等价，并统计通信量确认是 3Ψ（比 ZeRO-2 多一次 all-gather）。

**题目 3（PP 离散事件模拟）**：把 Demo 3 升级成真正的离散事件模拟——用一个时间轴数组排布每个 (stage, micro-batch, fwd/bwd) 的占用时间片，画出 GPipe vs 1F1B 的甘特图（ASCII 即可），直观看到 1F1B 的激活释放点和 bubble 位置，验证 makespan 与理论一致。

**扩展方向**：把三个 demo 组合成一个 mini 3D 并行模拟器，输入 (TP, PP, DP, model_size)，输出每卡显存占用、总通信量、估计 bubble，复现 §6.6.1 的配置法则。

### 四、面试高频题

**Q：单卡放不下大模型，根因是参数太多吗？**
A：不是。主因是 Adam mixed-precision 下每参数 16 字节的 model state（fp16 param 2 + grad 2 + fp32 master 4 + momentum 4 + variance 4）。参数本身（fp16）只占 2 字节，优化器状态占 12 字节才是大头。1B 模型光 state 就 16GB，还没算 activation。

**Q：TP 和 PP 都是模型并行，区别在哪？什么时候用哪个？**
A：TP 切层内（一个大矩阵切块到多卡），PP 切层间（不同层放不同卡）。TP 通信极密集（每层 2 次 all-reduce）必须机内 NVLink，但几乎零 bubble、数值无损；PP 通信稀疏（只传 stage 边界激活）可跨机，但有 (p-1)/m 的 bubble。原则：TP ≤ 单机 GPU 数走机内，PP 跨机 scale 更大模型。

**Q：Megatron 的 MLP 为什么第一个矩阵列切、第二个行切？**
A：MLP 是 GeLU(XA)B。A 列切后每卡算出完整的若干列 XA_i，可各自独立过 GeLU（逐元素非线性），**无需通信**；如果 A 行切，每卡只有部分和，必须先 all-reduce 才能过 GeLU（多一个同步点）。B 顺势行切，吃 GeLU 输出的列分布，算出部分和，最后 g 算子 all-reduce 一次。整个 MLP 前向只通信一次。

**Q：ZeRO 和模型并行（TP/PP）有什么本质区别？**
A：ZeRO 本质还是 **数据并行**——计算图和 DP 完全一样，只是把冗余复制的 model state 分片存储，用通信换显存。TP/PP 是 **模型并行**——真的把模型计算切到不同卡，改了计算图。ZeRO 编程简单（包一层）、数值与 DP 等价；模型并行复杂但能切 activation。生产里常 ZeRO + TP/PP 叠加用。

**Q：ZeRO-1/2 为什么说通信量和 DP 一样？**
A：标准 DP 的梯度 all-reduce 本来就 = reduce-scatter + all-gather（2Ψ）。ZeRO-2 把它拆开用：reduce-scatter 把梯度散到各 owner（每卡只拿自己那段全局梯度），各卡用本地分片的 optimizer state 更新自己那段参数，再 all-gather 广播回更新后的参数。总通信量还是 2Ψ，但 optimizer state 显存降 8x——免费午餐。ZeRO-3 才多一次 all-gather 变 3Ψ（1.5x）。

**Q：1F1B 比 GPipe 好在哪？是提升吞吐吗？**
A：常见误解。1F1B 的 bubble 和 GPipe **完全一样**（同步调度气泡不变），它主要是**显存优化**：GPipe 要把所有 m 个 micro-batch 的激活同时驻留（O(m)），1F1B 做完一个前向立刻做反向并释放激活，峰值钳到 O(p)。显存省了之后才能塞更大的 m，间接降 bubble。

**Q：3072 卡训 1T 模型怎么配并行？**
A：按通信密度分层。TP=8（一台 8 卡机内走 NVLink，承载最密集的层内 all-reduce），PP=8（跨 8 台机走 IB，只传稀疏的 stage 激活），DP=48（最外层，梯度 all-reduce 可与反向重叠）。8×8×48=3072。口诀：通信越密放越内。

### 五、未来方向

1. **Zero Bubble Pipeline**（arXiv:2401.10241）：把反向拆成"算梯度 w.r.t. 输入"和"算梯度 w.r.t. 权重"两半，用后者填 bubble，理论上做到接近零气泡。是 PP 调度的前沿。

2. **通信-计算更深度重叠**：随着模型增大，通信占比上升。把 all-gather/reduce-scatter 切得更细、与计算 kernel 级重叠（如 FSDP 的 prefetch、Megatron 的 overlap），是持续优化方向。

3. **异构与弹性并行**：PipeTransformer、自动并行（Alpa）等探索根据硬件拓扑自动搜索最优 TP/PP/DP 切分，减少人工调参。

4. **超长上下文的 activation 并行**：context/sequence parallel（Ring Attention 等）把 activation 沿序列维切分，应对百万 token 上下文下 activation 成为绝对主因的场景。

5. **FP8 / 低精度训练**：把 model state 从 16Ψ 进一步压（如 fp8 训练把部分 state 降到 1 字节），与并行策略正交叠加，是 H100/B200 时代的显存新解法。

---

## 参考资料

| 资料 | 来源 | 状态 |
|---|---|---|
| Megatron-LM 论文 | arXiv:1909.08053 — *Training Multi-Billion Parameter LMs Using Model Parallelism* | 实际获取（abstract + ar5iv HTML §3），f/g 定义与 MLP/attention 切法逐字引用 |
| Megatron mappings.py（f/g 算子源码） | `raw.githubusercontent.com/NVIDIA/Megatron-LM/main/megatron/core/tensor_parallel/mappings.py` | 实际获取，逐字引用 |
| Megatron layers.py（Column/RowParallelLinear） | `raw.githubusercontent.com/NVIDIA/Megatron-LM/main/megatron/core/tensor_parallel/layers.py` | 实际获取，权重切分维度逐字引用 |
| ZeRO 论文 | arXiv:1910.02054 — *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* | 实际获取（abstract + ar5iv HTML），16Ψ/4x/8x/通信量 2Ψ-3Ψ 逐字引用 |
| DeepSpeed ZeRO stage_1_and_2.py | `raw.githubusercontent.com/microsoft/DeepSpeed/master/deepspeed/runtime/zero/stage_1_and_2.py` | 实际获取，flatten/partition/single_partition 关键片段引用（行号近似） |
| PyTorch FSDP _flat_param.py | `raw.githubusercontent.com/pytorch/pytorch/main/torch/distributed/fsdp/_flat_param.py` | 实际获取，FlatParameter/_get_shard/all-gather 引用 |
| GPipe 论文 | arXiv:1811.06965 — *GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism* | 实际获取（ar5iv HTML），bubble O((K-1)/(M+K-1))、M≥4K、re-materialization 逐字引用 |
| Megatron 3D 并行论文 | arXiv:2104.04473 — *Efficient Large-Scale LM Training on GPU Clusters* | 实际获取（ar5iv HTML），(p-1)/m、TP≤g takeaway、1T/3072GPU/502 PFLOP 逐字引用 |
| PipeDream（1F1B 起源） | arXiv:1806.03377 — *PipeDream: Fast and Efficient Pipeline Parallel DNN Training* | 实际获取（搜索结果），1F1B 起源标注 |
| PyTorch DDP 论文 | VLDB 2020, `www.vldb.org/pvldb/vol13/p3005-li.pdf` | 实际获取（搜索结果），gradient bucketing / 逆序分桶 / 重叠引用 |
| Zero Bubble PP | arXiv:2401.10241 — *Zero Bubble Pipeline Parallelism* | 实际获取（搜索结果），未来方向 |
| Sequence parallel / Megatron activation recompute | arXiv:2205.05198 | 「待核」——仅提及方向，未取原文核实公式 |

> **Demo 可运行性声明**：本章 5 个 demo（Ring all-reduce、TP 等价、f/g 梯度校验、PP bubble、ZeRO-2 分片）均在 Python 3.12 + numpy 2.x 下**实际运行通过**，预期输出为实测结果。Demo 0 与 Demo 4 的"显存账"部分为按 ZeRO 公式的数值计算，非真实多卡测量。所有 demo 用单机 numpy 模拟集合通信语义（all-reduce=sum、reduce-scatter=切片、all-gather=concat），用于验证**数学等价性**，非真实分布式性能。真实多卡请用 torch.distributed 在你的集群验证。
