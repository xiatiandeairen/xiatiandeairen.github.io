---
title: "分布式推理与 CUDA 入门"
slug: "2-07"
collection: "tech-library"
group: "推理引擎"
order: 2007
summary: "推理引擎知识树位置 本章横跨两个根节点：「硬件执行层」(CUDA 线程模型、显存体系) 与「系统并行层」(TP/PP/EP 切分、集合通信)。 前置依赖：第 3 章 Attention 计算原理、第 4 章 KV Cache 管理、第 5 章 连续批处理。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-12T11:29:52.000Z"
updatedAt: "2026-06-12T11:29:52.000Z"
---
> **推理引擎知识树位置**
> 本章横跨两个根节点：「硬件执行层」(CUDA 线程模型、显存体系) 与「系统并行层」(TP/PP/EP 切分、集合通信)。
> 前置依赖：第 3 章 Attention 计算原理、第 4 章 KV Cache 管理、第 5 章 连续批处理。
> 后续章节：第 8 章 Flash Attention / Paged Attention 内核优化会直接用到这里的 CUDA 概念与 TP 通信框架。

---

## TL;DR

| 问题 | 核心答案 |
|------|---------|
| 为什么要分布式推理？ | 单卡装不下 70B+ 权重；单卡吞吐撑不住高并发 |
| TP vs PP 根本区别 | TP 切同一层的矩阵(层内通信)；PP 切不同层(流水线气泡) |
| CUDA 的执行单元是什么？ | Thread → Warp(32 threads) → Block → Grid；SM 是物理执行单元 |
| vLLM 如何实现 TP？ | ColumnParallelLinear + RowParallelLinear + AllReduce，Megatron 切法 |
| 最大的实际坑是什么？ | TP AllReduce 是同步点；PP 的 bubble；两者组合时通信放大 |

---

## 7.1 设计考古：从"单卡放不下"到分布式推理

### 7.1.1 问题的起源

2020 年前，GPT-2 (1.5B) 单卡可跑。GPT-3 (175B) 出现时，BF16 权重占 350 GB，远超单张 A100 的 80 GB。即便量化到 INT4，也需要 ~90 GB。问题彻底不同了：

1. **显存墙(Memory Wall)**：权重放不下
2. **吞吐墙(Throughput Wall)**：即使放得下，单卡的 compute/memory bandwidth 比在推理时是 memory-bound，吞吐受限

解决路径有两条：**模型并行**（切权重，多卡协作跑同一个请求）和 **数据并行**（多卡各自跑不同请求，权重复制）。推理引擎两者都用，但复杂度主要来自前者。

### 7.1.2 Tensor Parallelism 的奠基论文

**Megatron-LM** (2019, Shoeybi et al., [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)) 是 TP 在大模型上的奠基工作。

核心洞察：Transformer 里有两类大矩阵乘法天然可以列切分 / 行切分配对：

**MLP 块**（原论文图 3）：
```
X → [W1_col_split] → 列并行，无需通信 → GeLU → [W2_row_split] → AllReduce → Y
```

**Attention 块**（原论文图 4）：
```
X → QKV列切分(每卡负责部分head) → Local Attention → O行切分 → AllReduce → Y
```

关键性质：两块都只需要 **一次 AllReduce**，不是每个算子都通信——这是 Megatron-TP 通信效率高的根本原因。

论文在 512 张 V100 上训练了 8.3B 参数模型，实测 76% 的线性缩放效率，证明方案可行。

### 7.1.3 Pipeline Parallelism 与 1F1B 调度

PP 把不同 Transformer 层分配给不同 GPU（GPU 0 跑 layer 0-11，GPU 1 跑 layer 12-23，etc.）。核心挑战是 **bubble**：当 GPU 0 在跑 micro-batch 1 的反向时，GPU 1 在等，产生空闲。

Megatron v2 ([arXiv:2104.04473](https://arxiv.org/abs/2104.04473)) 提出 **1F1B (One Forward One Backward) Interleaved Schedule**：

- Bubble fraction（朴素 GPipe）= `(p-1)/m`，p = pipeline stages，m = micro-batches
- 1F1B Interleaved 把 bubble 压到 `(p-1)/(m*v)`，v = virtual stages per device

这使得推理引擎在用 PP 时，通过增加并发 micro-batch 数可以有效隐藏 bubble。

### 7.1.4 vLLM 采纳 Megatron 切法的时间线

vLLM 早期 ([commit 2023-06](https://github.com/vllm-project/vllm)) 专注单卡 KV Cache 管理，TP 支持是后来加入的。`vllm/distributed/` 目录下的代码明显继承了 Megatron-LM 的张量并行切分方式（`ColumnParallelLinear` / `RowParallelLinear` 名字和语义完全一致），通信后端用 NCCL via PyTorch DistributedGroup 封装。

### 7.1.5 Disaggregated Prefill/Decode：最新趋势

Prefill（处理 prompt）是 compute-bound；Decode（逐 token 生成）是 memory-bound。将两阶段部署到不同机器/GPU 上，可分别优化：

- **Splitwise** ([arXiv:2311.18677](https://arxiv.org/abs/2311.18677), Patel et al.): 同 budget 下吞吐提升 2.35x
- **Infinite-LLM** ([arXiv:2401.02669](https://arxiv.org/abs/2401.02669)): 在 32 张 A100 上吞吐提升 1.35–3.4x，支持 2M token 上下文

vLLM v0.6.x 开始实验性支持 disaggregated prefill (`--enable-chunked-prefill`)。

---

## 7.2 CUDA 线程模型：推理 kernel 的底层地基

推理引擎里大量 kernel（矩阵乘、attention、dequant）都跑在 GPU 上。要读懂 kernel 代码，必须先把这套执行模型刻入骨髓。

### 7.2.1 执行层次：Thread → Warp → Block → Grid

```
Grid (整个 kernel 的执行空间)
 └─ Block (共享 Shared Memory，可 __syncthreads())
      └─ Warp (32 threads，SIMT 执行单位，同一时钟周期执行相同指令)
           └─ Thread (基本并行单位，有私有 registers)
```

**物理对应**：每个 Block 运行在一个 **SM (Streaming Multiprocessor)** 上。A100 有 108 个 SM，每个 SM 最多同时跑 4 个 Block（受寄存器/共享内存限制）。

**关键内置变量**（CUDA C）：
```c
threadIdx.x / .y / .z   // 在 block 内的线程坐标
blockIdx.x  / .y / .z   // block 在 grid 内的坐标
blockDim.x  / .y / .z   // block 的维度
gridDim.x   / .y / .z   // grid 的维度
```

计算全局线程 ID（1D 情况）：
```c
int tid = blockIdx.x * blockDim.x + threadIdx.x;
```

### 7.2.2 内存体系（速度从快到慢）

| 内存类型 | 位置 | 延迟 | 容量 | 作用域 |
|---------|------|------|------|--------|
| Registers | on-chip, per-thread | ~1 cycle | ~256KB/SM | private per thread |
| Shared Memory (SMEM) | on-chip, per-block | ~4 cycles | 48-192KB/SM | shared within block |
| L1 Cache | on-chip, per-SM | ~20 cycles | 32-128KB | per-SM |
| L2 Cache | on-chip, global | ~200 cycles | 40MB (A100) | all SMs |
| Global Memory (HBM) | off-chip (DRAM) | ~700 cycles | 40-80GB | all threads |
| Constant Memory | off-chip, cached | ~100 cycles | 64KB | read-only, all |

**推理 kernel 优化的核心**：把热数据放 Shared Memory，减少 Global Memory 访问次数（这就是 Flash Attention 的根本思想）。

### 7.2.3 Warp 发散(Warp Divergence)：推理引擎必须知道的坑

同一 Warp 内的 32 个 thread 必须执行相同指令。如果有 `if/else`：
```c
if (tid % 2 == 0) { /* path A */ }
else              { /* path B */ }
```
SIMT 硬件会先让所有 thread 执行 path A（非 path A 的 thread mask 掉），再执行 path B——吞吐减半。在 attention mask、variable-length sequence 处理里这是高频问题，vLLM 的 PagedAttention kernel 有专门的分支预测优化。

### 7.2.4 真实源码：来自 NVIDIA 官方 blog 的 vectorAdd

以下代码来自 NVIDIA 官方博客《An Even Easier Introduction to CUDA》([developer.nvidia.com](https://developer.nvidia.com/blog/even-easier-introduction-cuda/))，展示最基础的 GPU kernel 结构：

```cuda
// 【真实源码 developer.nvidia.com/blog/even-easier-introduction-cuda】
__global__
void add(int n, float *x, float *y)
{
  int index  = blockIdx.x * blockDim.x + threadIdx.x;  // 全局线程 ID
  int stride = blockDim.x * gridDim.x;                 // grid 总线程数(用于 stride loop)
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];                                // 向量加法
}

int main(void)
{
  int N = 1<<20;   // 1M 元素
  float *x, *y;

  // Unified Memory：CPU 和 GPU 都能访问
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  for (int i = 0; i < N; i++) { x[i] = 1.0f; y[i] = 2.0f; }

  // 预取到 GPU device 0
  cudaMemPrefetchAsync(x, N*sizeof(float), 0, 0);
  cudaMemPrefetchAsync(y, N*sizeof(float), 0, 0);

  int blockSize = 256;                         // 每个 block 256 个线程
  int numBlocks = (N + blockSize - 1) / blockSize;  // 向上取整
  add<<<numBlocks, blockSize>>>(N, x, y);      // 启动 kernel

  cudaDeviceSynchronize();  // 等 GPU 完成

  // 验证结果
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  printf("Max error: %f\n", maxError);  // 预期 0.000000

  cudaFree(x); cudaFree(y);
  return 0;
}
```

**注解**：
- `__global__` 修饰符表示这是 GPU kernel，从 CPU 调用，在 GPU 执行
- `<<<numBlocks, blockSize>>>` 就是 launch config：`numBlocks` 个 block，每个 block `blockSize` 个 thread
- `stride loop` 是一种常见 pattern：让 1M 个元素被 ~4096 个 thread 以步进方式处理，每个 thread 处理 N/(numBlocks*blockSize) ≈ 1M/4096 ≈ 256 个元素
- Unified Memory (`cudaMallocManaged`) 简化 host/device 同步，生产推理代码通常用显式 `cudaMalloc` + `cudaMemcpy` 以获得更细控制

---

## 7.3 vLLM 张量并行实现：真实源码精读

### 7.3.1 进程组初始化

**来源**：`vllm/distributed/parallel_state.py`（vllm-project/vllm main 分支）

```python
# 【真实源码 vllm-project/vllm@main/vllm/distributed/parallel_state.py】
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    prefill_context_model_parallel_size: int = 1,
    decode_context_model_parallel_size: int | None = 1,
    backend: str | None = None,
) -> None:
    # ...
    # 把所有 rank 按 (ExternalDP, DP, PP, prefill_CP, TP) 的顺序排列
    all_ranks = torch.arange(world_size).reshape(
        -1, data_parallel_size, pipeline_model_parallel_size,
        prefill_context_model_parallel_size, tensor_model_parallel_size)

    # 创建 TP 组：在最后一个维度(TP)上 unbind，得到每组 TP rank 列表
    group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    _TP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,
        group_name="tp",
    )
```

**逐行解析**：
- `all_ranks.reshape(...)` 把 rank 0..N-1 按并行维度排成多维张量，这是 Megatron 的 rank 布局惯例
- `view(-1, tensor_model_parallel_size).unbind(0)` 提取出每个 TP 组的成员列表
- 例如 world_size=8, TP=4, PP=2：rank 0-3 是第一个 PP stage 的 TP 组，rank 4-7 是第二个

### 7.3.2 GroupCoordinator.all_reduce

**来源**：`vllm/distributed/parallel_state.py`

```python
# 【真实源码 vllm-project/vllm@main/vllm/distributed/parallel_state.py】
def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
    """
    User-facing all-reduce function before we actually call the
    all-reduce operation.

    We need this because Dynamo does not support passing an arbitrary
    object (`self` in this case) to a custom op. We need to pass the
    group name as a string, and then look up the group coordinator from
    the group name, dispatch the all-reduce operation to the group
    coordinator.

    In addition, PyTorch custom ops do not support mutation or returning
    a new tensor in the same op. So we always make the all-reduce operation
    out-of-place.
    """
    if self.world_size == 1:       # 单卡直接返回，zero overhead
        return input_

    if self.use_custom_op_call:
        # torch.compile / Dynamo 路径：用字符串 group_name 间接查找
        return torch.ops.vllm.all_reduce(input_, group_name=self.unique_name)
    else:
        return self._all_reduce_out_place(input_)
```

**注解**：
- `world_size == 1` 的 early return 是推理引擎里普遍的优化模式：单机单卡时 TP=1，所有通信路径零开销
- `torch.compile` 不支持把 Python 对象传入自定义 op（因为 Dynamo 需要图捕获），所以用字符串 name 作为间接引用
- out-of-place 设计是为了满足 PyTorch custom op 的限制（不能原地修改）

### 7.3.3 NCCL allreduce 底层调用

**来源**：`vllm/distributed/device_communicators/pynccl.py`

```python
# 【真实源码 vllm-project/vllm@main/vllm/distributed/device_communicators/pynccl.py】
def all_reduce(
    self,
    in_tensor: torch.Tensor,
    out_tensor: torch.Tensor = None,
    op: ReduceOp = ReduceOp.SUM,
    stream=None,
) -> torch.Tensor:
    if self.disabled:
        return None
    assert in_tensor.device == self.device

    if out_tensor is None:
        out_tensor = torch.empty_like(in_tensor)

    if stream is None:
        stream = current_stream()

    self.nccl.ncclAllReduce(
        buffer_type(in_tensor.data_ptr()),    # 输入 buffer 原始指针
        buffer_type(out_tensor.data_ptr()),   # 输出 buffer 原始指针
        in_tensor.numel(),                    # 元素数
        ncclDataTypeEnum.from_torch(in_tensor.dtype),   # dtype 映射
        ncclRedOpTypeEnum.from_torch(op),               # 规约类型(默认 SUM)
        self.comm,                            # NCCL communicator
        cudaStream_t(stream.cuda_stream),     # CUDA stream，异步执行
    )
    return out_tensor
```

**注解**：
- `in_tensor.data_ptr()` 获取原始 GPU 指针，直接传给 NCCL C library
- NCCL 在 cuda stream 上异步执行，不立即阻塞 CPU；后续 GPU kernel 依赖这个 stream 自动同步
- FP8 特殊处理：NCCL 不原生支持 FP8，vLLM 将其 cast 成 uint8 传递（`ncclDataTypeEnum.from_torch(torch.uint8)`）

### 7.3.4 ColumnParallelLinear 与 RowParallelLinear

**来源**：`vllm/model_executor/layers/linear.py`

```python
# 【真实源码 vllm-project/vllm@main/vllm/model_executor/layers/linear.py】
class ColumnParallelLinear(LinearBase):
    # 权重按列切分：W = [W_1 | W_2 | ... | W_tp]
    # 每个 rank 只持有 W_i，output_size_per_partition = output_size / tp_size
    
    def forward(self, input_):
        # 1. 本地矩阵乘（无通信）
        output_parallel = self.quant_method.apply(self, input_, bias)
        # 2. 如需聚合（如最后输出层），才 AllGather
        if self.gather_output and self.tp_size > 1:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel   # 大多数情况保持分布式，后续 Row 层消费
        return (output, bias) if self.return_bias else output

class RowParallelLinear(LinearBase):
    # 权重按行切分：W = [W_1; W_2; ...; W_tp]^T
    # 输入按列切分（与前面 Column 层的输出对齐）
    
    def forward(self, input_):
        if not self.input_is_parallel:
            # 输入未切分时，手动按 tp_rank 切分
            split_input = split_tensor_along_last_dim(input_, self.tp_size)
            input_parallel = split_input[self.tp_rank].contiguous()
        else:
            input_parallel = input_   # 接 Column 层输出，已是分布式

        # 只有 rank 0 加 bias（避免多 rank 重复加）
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self, input_parallel, bias_)

        # 关键：AllReduce 合并各 rank 的部分结果
        if self.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel
        return (output, bias) if self.return_bias else output
```

### 7.3.5 Llama 模型如何使用这两个 layer

**来源**：`vllm/model_executor/models/llama.py`

```python
# 【真实源码 vllm-project/vllm@main/vllm/model_executor/models/llama.py】
class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, ...):
        # gate 和 up 合并成 MergedColumnParallelLinear（等价于两个 Column 并排）
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,  # gate, up 各占一半
            ...
        )
        # down 是 Row，完成 AllReduce
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            reduce_results=True,   # 这里触发 AllReduce
            ...
        )

    def forward(self, x):
        x, _ = self.gate_up_proj(x)    # Column：本地矩阵乘，无通信
        x = self.act_fn(x)              # SiluAndMul：本地激活
        x, _ = self.down_proj(x)       # Row：本地矩阵乘 + AllReduce
        return x

class LlamaAttention(nn.Module):
    def __init__(self, ...):
        # Q/K/V 合并为 QKVParallelLinear（Column 变体，按 head 切分）
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            ...
        )
        # O projection 是 Row，做 AllReduce
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            ...
        )

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)   # Column：无通信
        q, k, v = qkv.split([...], dim=-1)
        q, k = self.rotary_emb(positions, q, k)  # 本地 RoPE
        attn_output = self.attn(q, k, v)          # 本地 Attention（每 rank 负责部分 head）
        output, _ = self.o_proj(attn_output)       # Row：AllReduce
        return output
```

**整体数据流（TP=4 时的 MLP）**：
```
rank 0,1,2,3 各持有完整 input X (shape: [seq, hidden])
    ↓ MergedColumnParallelLinear (无通信)
rank 0: [gate_0, up_0]  rank 1: [gate_1, up_1]  ...  (各取 intermediate/4)
    ↓ SiluAndMul (本地)
rank 0: act_0          rank 1: act_1          ...
    ↓ RowParallelLinear (本地矩阵乘，然后 AllReduce)
rank 0..3 各有部分结果 → AllReduce → 所有 rank 得到完整 output
```

整个 MLP 块只有 **1 次 AllReduce**。Attention 块同样只有 1 次（在 o_proj 后）。这是 Megatron-LM 设计最精妙的地方。

---

## 7.4 并行策略对比

### 7.4.1 TP vs PP vs DP vs EP 全景

| 维度 | TP (Tensor Parallel) | PP (Pipeline Parallel) | DP (Data Parallel) | EP (Expert Parallel) |
|------|---------------------|----------------------|-------------------|---------------------|
| 切分粒度 | 层内矩阵 | 层间（stage） | 请求（batch） | MoE expert |
| 通信类型 | AllReduce (同步) | P2P send/recv | AllReduce (梯度) | AllGather + Scatter |
| 通信频率 | 每层 2 次 | 每 stage 边界 1 次 | 每 step 1 次 | 每层 2 次 |
| 适用场景 | 单卡放不下 | 卡多层多 | 高并发 | MoE 模型 |
| 主要开销 | AllReduce 延迟 | Pipeline bubble | 同步等待 | Expert 负载均衡 |
| NVLink 需求 | 强（同节点） | 低（跨节点可用 IB） | 低 | 中 |

### 7.4.2 通信量分析

设 hidden_size = H，sequence_length = S，TP = N_tp：

- **TP AllReduce**：每次通信量 = `2 * S * H * bytes_per_elem`（ring-allreduce 的 send+receive），每个 Transformer 层 2 次（MLP + Attn）
- **PP send/recv**：每个 stage 边界传 activation，通信量 = `S * H * bytes_per_elem`，但只跨 stage，不广播

对于 Llama-70B（H=8192，S=2048，BF16）：
- TP AllReduce per layer：`2 * 2048 * 8192 * 2 = 67 MB`，在 NVLink（600 GB/s）上约 0.1 ms
- 全模型 80 层 × 2 次 = 160 次 AllReduce ≈ 16 ms（纯通信时间）

这也是为什么 TP 要求 GPU 之间有 **NVLink** 高速互联，InfiniBand 的 200 Gbps（25 GB/s）远不够。

### 7.4.3 vLLM 的实际部署组合

```
4 节点 × 8 GPU/节点 = 32 GPU
典型配置：TP=8（节点内，用 NVLink） × PP=4（节点间，用 IB）
```

- 节点内 TP AllReduce 走 NVLink，快
- 节点间 PP 的 activation 传输走 InfiniBand，慢但每 stage 只传一次

### 7.4.4 Prefill 与 Decode 的不同瓶颈

| 阶段 | 计算特征 | 瓶颈 | 最优策略 |
|------|---------|------|---------|
| Prefill | 长 seq，大 GEMM | Compute-bound | 高 TP，利用 Tensor Core |
| Decode | 1 token，小 GEMM | Memory-bound（读权重） | 大 batch，减少权重读次数 |

这是 Splitwise/DistServe 把两个阶段物理分离的根本依据：用计算密集型 GPU（如 H100）跑 prefill，用内存大但计算弱的 GPU 跑 decode。

---

## 7.5 Demo 1：CUDA vectorAdd — 理解 grid/block/thread

**依赖**：CUDA Toolkit 12+，任意 NVIDIA GPU，`nvcc`  
**设计为可运行，请在你的 CUDA 环境验证**

```cuda
// demo_vector_add.cu
// 展示 CUDA kernel 启动配置，与 NVIDIA blog 代码对应
// 来源参考：developer.nvidia.com/blog/even-easier-introduction-cuda

#include <stdio.h>
#include <math.h>

// __global__: GPU kernel，CPU 调用，GPU 执行
__global__
void vector_add(int n, float *a, float *b, float *c) {
    // 计算全局线程 ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // stride loop: 让线程覆盖所有元素（不限于 n <= gridDim*blockDim 的情况）
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int N = 1 << 20;  // 1M 元素
    float *a, *b, *c;
    
    // Unified Memory: CPU 和 GPU 都可访问，CUDA runtime 负责迁移
    cudaMallocManaged(&a, N * sizeof(float));
    cudaMallocManaged(&b, N * sizeof(float));
    cudaMallocManaged(&c, N * sizeof(float));
    
    for (int i = 0; i < N; i++) { a[i] = 1.0f; b[i] = 2.0f; }
    
    // 预取到 GPU 0 以避免 page fault 开销
    int dev = 0;
    cudaMemPrefetchAsync(a, N*sizeof(float), dev, NULL);
    cudaMemPrefetchAsync(b, N*sizeof(float), dev, NULL);
    cudaMemPrefetchAsync(c, N*sizeof(float), dev, NULL);
    
    // Launch config:
    //   blockSize = 256 (每个 block 256 thread)
    //   numBlocks = ceil(N / blockSize) ≈ 4096
    //   总线程 = 256 * 4096 = 1M，恰好覆盖所有元素
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    printf("Launch config: <<<%d blocks, %d threads/block>>>\n", numBlocks, blockSize);
    printf("Total threads: %d, array size: %d\n", numBlocks * blockSize, N);
    
    vector_add<<<numBlocks, blockSize>>>(N, a, b, c);
    
    // CPU 等待 GPU 完成（异步 kernel 需要显式同步）
    cudaDeviceSynchronize();
    
    // 验证
    float maxErr = 0.0f;
    for (int i = 0; i < N; i++)
        maxErr = fmaxf(maxErr, fabsf(c[i] - 3.0f));
    printf("Max error: %e (expected: 0)\n", maxErr);
    
    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}
```

**编译运行**：
```bash
nvcc -o demo_vector_add demo_vector_add.cu
./demo_vector_add
```

**预期输出**：
```
Launch config: <<<4096 blocks, 256 threads/block>>>
Total threads: 1048576, array size: 1048576
Max error: 0.000000e+00 (expected: 0)
```

**与推理引擎的联系**：真实的 dequantize kernel、RoPE kernel 结构与此完全相同——`blockIdx.x * blockDim.x + threadIdx.x` 算出全局 token ID 或元素 ID，然后做对应的计算。

---

## 7.6 Demo 2：NumPy 模拟 TP 张量切分与 AllReduce

**依赖**：Python 3.8+，numpy，无需 GPU  
**设计为可运行，请在你的环境验证**  
**目的**：印证 vLLM 的 ColumnParallelLinear + RowParallelLinear + AllReduce 的数学等价性

```python
# demo_tp_simulation.py
# 用 numpy 模拟 TP=4 时的 MLP 前向（Column + Row + AllReduce）
# 对应 vLLM: ColumnParallelLinear.forward + RowParallelLinear.forward
# 来源对应: vllm/model_executor/layers/linear.py

import numpy as np

np.random.seed(42)

# ── 超参 ──
HIDDEN   = 512    # hidden_size
INTER    = 2048   # intermediate_size
SEQ      = 8      # sequence_length
TP       = 4      # tensor_parallel_size

# ── 初始化权重（模拟单卡"完整"权重，用于基准对照） ──
W1 = np.random.randn(HIDDEN, INTER).astype(np.float32) * 0.02   # gate/up weight (合并)
W2 = np.random.randn(INTER,  HIDDEN).astype(np.float32) * 0.02  # down weight
X  = np.random.randn(SEQ, HIDDEN).astype(np.float32)             # 输入 [seq, hidden]

def silu(x):
    return x / (1.0 + np.exp(-x))

# ── 基准：单卡完整 MLP（无并行） ──
def mlp_single(x, w1, w2):
    h = x @ w1          # [seq, hidden] × [hidden, inter] = [seq, inter]
    h = silu(h)
    out = h @ w2        # [seq, inter]  × [inter, hidden] = [seq, hidden]
    return out

ref = mlp_single(X, W1, W2)
print(f"Single GPU output shape: {ref.shape}")

# ── TP=4 并行模拟 ──
# ColumnParallelLinear: 把 W1 按列切分（output dim / TP）
# 每个 rank 持有 W1_i，形状 [hidden, inter/TP]
inter_per_rank = INTER // TP
W1_shards = [W1[:, i*inter_per_rank:(i+1)*inter_per_rank] for i in range(TP)]

# RowParallelLinear: 把 W2 按行切分（input dim / TP）
# 每个 rank 持有 W2_i，形状 [inter/TP, hidden]
W2_shards = [W2[i*inter_per_rank:(i+1)*inter_per_rank, :] for i in range(TP)]

# 每个 rank 独立前向（无通信）
rank_outputs = []
for rank in range(TP):
    # Column: 本地矩阵乘，no communication
    h_local = X @ W1_shards[rank]             # [seq, inter/TP]
    h_local = silu(h_local)                    # 本地激活
    # Row: 本地矩阵乘，no communication
    out_local = h_local @ W2_shards[rank]     # [seq, hidden]  (部分结果)
    rank_outputs.append(out_local)
    print(f"  Rank {rank}: local partial output shape {out_local.shape}")

# ── AllReduce (SUM): 对应 vLLM 的 tensor_model_parallel_all_reduce ──
# 真实 AllReduce: ring-allreduce，这里用 sum 模拟
tp_output = np.stack(rank_outputs, axis=0).sum(axis=0)  # [TP, seq, hidden] → sum → [seq, hidden]
print(f"\nTP AllReduce output shape: {tp_output.shape}")

# ── 验证等价性 ──
max_diff = np.abs(ref - tp_output).max()
rel_diff = max_diff / (np.abs(ref).max() + 1e-8)
print(f"\n数学等价性验证:")
print(f"  Max absolute diff: {max_diff:.2e}")
print(f"  Max relative diff: {rel_diff:.2e}")
print(f"  数值等价: {'✓ PASS' if rel_diff < 1e-5 else '✗ FAIL'}")

# ── 通信量统计 ──
allreduce_bytes = SEQ * HIDDEN * 4 * 2  # 2x because ring-allreduce: reduce-scatter + all-gather
print(f"\n通信量统计 (TP={TP}):")
print(f"  AllReduce 数据量: {allreduce_bytes / 1024:.1f} KB/layer")
print(f"  MLP 只需 1 次 AllReduce (发生在 RowParallelLinear.forward 末尾)")

# ── 验证 Attention head 切分 ──
print(f"\n--- Attention head 切分验证 ---")
NUM_HEADS = 8
HEAD_DIM  = HIDDEN // NUM_HEADS  # 64
heads_per_rank = NUM_HEADS // TP  # 2

# QKV Column: 每 rank 负责部分 head
W_Q = np.random.randn(HIDDEN, NUM_HEADS * HEAD_DIM).astype(np.float32) * 0.02
W_Q_shards = [W_Q[:, i*heads_per_rank*HEAD_DIM:(i+1)*heads_per_rank*HEAD_DIM] for i in range(TP)]

# 单卡基准
Q_ref = X @ W_Q

# TP 模拟：按 head 切分
Q_local_list = [X @ W_Q_shards[r] for r in range(TP)]
Q_tp = np.concatenate(Q_local_list, axis=-1)  # AllGather 等价（不是 AllReduce）

# 注意：Attention 用 AllGather 而 MLP 用 AllReduce，对应不同切分方向
q_diff = np.abs(Q_ref - Q_tp).max()
print(f"  Attention QKV head-split 等价: {'✓ PASS' if q_diff < 1e-5 else '✗ FAIL'} (diff={q_diff:.2e})")
print(f"  每 rank 负责 {heads_per_rank}/{NUM_HEADS} 个 head，计算完全独立（无需通信直到 O_proj）")
```

**运行方式**：
```bash
python demo_tp_simulation.py
```

**预期输出**：
```
Single GPU output shape: (8, 512)
  Rank 0: local partial output shape (8, 512)
  Rank 1: local partial output shape (8, 512)
  Rank 2: local partial output shape (8, 512)
  Rank 3: local partial output shape (8, 512)

TP AllReduce output shape: (8, 512)

数学等价性验证:
  Max absolute diff: 0.00e+00
  Max relative diff: 0.00e+00
  数值等价: ✓ PASS

通信量统计 (TP=4):
  AllReduce 数据量: 8.0 KB/layer
  MLP 只需 1 次 AllReduce (发生在 RowParallelLinear.forward 末尾)

--- Attention head 切分验证 ---
  Attention QKV head-split 等价: ✓ PASS (diff=0.00e+00)
  每 rank 负责 2/8 个 head，计算完全独立（无需通信直到 O_proj）
```

这个 demo **直接印证了** `vllm/model_executor/models/llama.py` 里 `LlamaMLP.forward` 和 `LlamaAttention.forward` 的数学本质：列切分→激活→行切分→AllReduce 等价于单卡完整计算。

---

## 7.7 Demo 3：模拟 Pipeline Parallel bubble

**依赖**：Python 3.8+，无需 GPU  
**目的**：直观理解 PP 的 bubble overhead 与 micro-batch 的关系

```python
# demo_pp_bubble.py
# 用时间轴模拟 Pipeline Parallelism 的 bubble 比例
# 对应 Megatron-LM 的 GPipe 和 1F1B 调度

def compute_bubble_fraction(pp_stages: int, micro_batches: int, schedule: str = "gpipe") -> float:
    """
    计算 pipeline bubble fraction (空闲时间 / 总时间)
    GPipe:    bubble = (p-1) / (m + p - 1)  ≈ (p-1)/m for large m
    1F1B:     bubble ≈ same as GPipe for full pipeline
    Interleaved 1F1B: bubble = (p-1) / (m*v) where v = virtual stages
    """
    p, m = pp_stages, micro_batches
    if schedule == "gpipe":
        bubble = (p - 1) / (m + p - 1)
    elif schedule == "1f1b":
        bubble = (p - 1) / (m + p - 1)  # 同 GPipe，但显存更少
    elif schedule == "interleaved":
        v = 2  # 假设 virtual stages = 2
        bubble = (p - 1) / (m * v + p - 1)
    return bubble

import sys

print("=" * 60)
print("Pipeline Parallel Bubble Fraction Analysis")
print("=" * 60)
print(f"{'PP Stages':>10} {'Micro-batches':>14} {'GPipe Bubble':>13} {'1F1B Inter':>11}")
print("-" * 60)

for pp in [2, 4, 8]:
    for m in [1, 2, 4, 8, 16, 32]:
        gpipe_bubble = compute_bubble_fraction(pp, m, "gpipe")
        interleaved  = compute_bubble_fraction(pp, m, "interleaved")
        print(f"{pp:>10} {m:>14} {gpipe_bubble:>12.1%} {interleaved:>10.1%}")
    print()

# 计算 vLLM 典型配置下的 bubble
print("=" * 60)
print("vLLM 典型配置 (Llama-70B, PP=4, TP=8, batch=32):")
pp, m = 4, 32
bubble = compute_bubble_fraction(pp, m, "gpipe")
effective_util = 1 - bubble
print(f"  PP stages={pp}, micro-batches={m}")
print(f"  Bubble fraction: {bubble:.1%}")
print(f"  Effective GPU utilization (pipeline only): {effective_util:.1%}")
print()
print("Key insight:")
print("  增大 micro-batch 数（更多并发请求）是掩盖 PP bubble 的主要手段")
print("  这也是推理引擎要做连续批处理(continuous batching)的另一个动机")
```

**运行方式**：
```bash
python demo_pp_bubble.py
```

**预期输出**（片段）：
```
============================================================
Pipeline Parallel Bubble Fraction Analysis
============================================================
 PP Stages  Micro-batches  GPipe Bubble  1F1B Inter
------------------------------------------------------------
         2              1         50.0%       33.3%
         2              4         20.0%       11.1%
         2             32          3.0%        1.5%

         4              1         75.0%       60.0%
         4              4         42.9%       26.7%
         4             32         8.6%         4.5%
         ...

============================================================
vLLM 典型配置 (Llama-70B, PP=4, TP=8, batch=32):
  PP stages=4, micro-batches=32
  Bubble fraction: 8.6%
  Effective GPU utilization (pipeline only): 91.4%

Key insight:
  增大 micro-batch 数（更多并发请求）是掩盖 PP bubble 的主要手段
  这也是推理引擎要做连续批处理(continuous batching)的另一个动机
```

---

## 7.8 失败模式 / 生产真坑 / 底层根因

### 坑 1：TP AllReduce 挂在异构 NVLink 拓扑

**现象**：8 卡 A100 上 TP=8 跑起来，换到某个云厂商的"同等规格"机器，速度下降 3x。  
**根因**：部分云厂商的 8 卡实例由 **两个 4 卡 NVLink 域** 拼接而成，卡 0-3 和卡 4-7 之间走 PCIe，NVLink 带宽从 600 GB/s 骤降到 PCIe 的 64 GB/s。AllReduce 的 ring 经过跨 PCIe 链路时带宽受限。  
**排查**：`nvidia-smi topo -m` 查看 NVLink 拓扑；NCCL debug 日志里看 `RING` 路径。  
**解法**：TP=4 + PP=2，把 TP 限制在单个 NVLink 域内；或使用 `NCCL_TOPO_FILE` 指定拓扑。

### 坑 2：PP 分组切割不均导致 load imbalance

**现象**：PP=4 时，GPU 3（最后 stage）持续 100% 利用率，GPU 0-2 有明显等待。  
**根因**：Transformer 最后几层通常包含 LM Head（大词表 embedding），在不均衡切层时会集中在某个 stage。Llama-70B 的词表 32K × 8192 = 2GB，光 LM Head 就比一个 decoder layer 重得多。  
**解法**：手动配置每个 stage 的层数，如 `--num-layers-per-virtual-pipeline-stage`；或把 embedding/LM head 单独放一个 stage。

### 坑 3：NCCL 版本与 CUDA driver 不匹配

**现象**：多节点推理概率性挂死（hang），没有报错，只是某个 rank 一直在等。  
**根因**：NCCL 2.18 在某些 CUDA 12.2 driver 组合下有 SHARP 集合通信的 deadlock bug。  
**排查**：`NCCL_DEBUG=TRACE NCCL_TRACE_VERBOSITY=2` 找到卡在哪个 collective；检查各节点 NCCL 版本一致性。  
**解法**：固定 NCCL 版本（vLLM 的 Docker 镜像严格 pin NCCL 版本就是这个原因）。

### 坑 4：KV Cache 在 TP 时的 head 切分语义

**现象**：TP=4 时 KV cache 内存估算与实际不符，导致 OOM。  
**根因**：MHA（Multi-Head Attention）在 TP 下每卡只负责 `num_kv_heads / TP` 个 KV head，但 GQA（Grouped Query Attention，如 Llama-70B 用 GQA）中 KV head 数是 8 而非 64，TP=8 时 `8/8=1`，不能再切了——继续设 TP>8 会让某些 rank 没有 KV head。vLLM 有检查：`num_kv_heads % tp_size == 0`，违反时报错。  
**解法**：检查 `num_key_value_heads` 字段，确保 `tp_size <= num_kv_heads`。

### 坑 5：Warp divergence 在 variable-length sequence

**现象**：处理 batch 中 seq_len 差异很大的请求时，GPU 利用率低。  
**根因**：若 kernel 用 `if (seq_idx < seq_len[batch_idx])` 做边界判断，同 warp 内不同 batch 的 seq_len 不同，导致 warp divergence，执行效率可能降到 1/32。  
**解法**：vLLM 的 PagedAttention kernel 用 flash-decoding 风格的分块 + 掩码，或用 `torch.nn.utils.rnn.pack_padded_sequence` 等压缩表示避免 padding。

### 坑 6：Disaggregated prefill 的 KV cache 迁移代价

**现象**：Splitwise 架构下，prefill 完成后 KV cache 要从 prefill 节点传到 decode 节点，高并发时这个传输成为瓶颈。  
**根因**：Llama-70B 处理 1024 token 的 KV cache 约为 `80 layers × 2 × 8 KV-heads × 1024 × 128 × BF16 = ~2.7 GB`，100 Gbps IB 传输需要 ~215 ms，比 decode 一个 token（~30 ms）慢得多。  
**解法**：chunked prefill（边 prefill 边 decode）+ pipeline KV transfer，或选用 400 Gbps IB。

---

## 7.9 架构演进脉络

```
2019  Megatron-LM v1     TP for Transformer（ColumnParallel/RowParallel）
         ↓
2021  Megatron-LM v2     PP 1F1B + Sequence Parallel（减少 activation 显存）
         ↓
2022  Alpa               Auto-parallel（ILP 求最优 TP×PP 切分）
         ↓
2023  vLLM               PagedAttention + TP；连续批处理
         ↓
2023  Splitwise          Disaggregated prefill/decode（同 budget 2.35x 吞吐）
         ↓
2024  Infinite-LLM       Attention 层单独调度；2M context on 32×A100
         ↓
2024  vLLM v0.6+         Chunked prefill；EP for MoE；TP+PP+EP 组合
         ↓
2025  vLLM v0.8+         Disaggregated prefill 实验性支持；Speculative Decoding TP
```

---

## 7.10 章末五件套

### 1. 关键概念速查

| 术语 | 定义 |
|------|------|
| TP (Tensor Parallel) | 层内矩阵按列/行切分，同 TP 组内 AllReduce 合并 |
| PP (Pipeline Parallel) | 层间切分，stage 间传 activation，有 bubble |
| EP (Expert Parallel) | MoE 的 expert 切分到不同 GPU，配合 token routing |
| AllReduce | 所有 rank 的张量求和后广播回所有 rank |
| SMEM (Shared Memory) | SM 上的片内高速内存，latency 约 4 cycles |
| Warp Divergence | 同 warp 的线程走不同分支导致串行化 |
| Bubble Fraction | PP 中 GPU 等待时间 / 总执行时间 |
| NVLink | NVIDIA GPU 间高速互联，A100 单向 300 GB/s |

### 2. 思考题

1. 为什么 vLLM 的 `GroupCoordinator.all_reduce` 要在 `world_size == 1` 时直接返回 input 而不走 AllReduce？这个 early return 能节省多少时间？
2. Llama-3-70B 使用 GQA，num_kv_heads=8。如果你要设 TP=16，会出现什么问题？如何解决？
3. 对比 TP=8 和 PP=8 处理同一个 70B 模型的 prefill 请求：哪种方案单请求延迟更低？为什么？
4. `ColumnParallelLinear` 设 `gather_output=True` 和 `gather_output=False` 各适用什么场景？

### 3. 代码题（扩展 Demo）

**题目 A（CUDA）**：修改 Demo 1 的 `vector_add`，改为用 Shared Memory 做 block-level reduction（求和），输出每个 block 内元素的总和。这是 attention softmax 里 log-sum-exp 计算的原型。

**题目 B（TP 模拟）**：扩展 Demo 2，支持 TP=1,2,4,8 的循环对比，输出不同 TP 下的通信量和理论延迟（假设 NVLink 带宽 300 GB/s），画出 TP vs 通信开销的曲线。

**题目 C（PP 调度）**：扩展 Demo 3，实现 GPipe 和 1F1B 的完整时间轴模拟（用列表记录每个 stage 每个 time-step 的状态），可视化 bubble 分布。

### 4. 关联章节

- **第 4 章** KV Cache 管理 → PagedAttention 在 TP 下 KV head 切分的细节
- **第 5 章** 连续批处理 → PP 下 micro-batch 是如何与请求调度协调的
- **第 8 章** Flash Attention Kernel → 直接使用本章的 SMEM 概念和 warp 执行模型

### 5. 延伸阅读

- [Megatron-LM 原论文 (arXiv:1909.08053)](https://arxiv.org/abs/1909.08053)
- [Megatron v2 PP interleaved (arXiv:2104.04473)](https://arxiv.org/abs/2104.04473)
- [Splitwise (arXiv:2311.18677)](https://arxiv.org/abs/2311.18677)
- [NVIDIA CUDA C Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [vLLM distributed 目录源码](https://github.com/vllm-project/vllm/tree/main/vllm/distributed)
- [NCCL 开发者文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)

---

## 附录：关键文件路径索引

| 内容 | 文件路径（vllm-project/vllm@main） |
|------|----------------------------------|
| 进程组初始化 | `vllm/distributed/parallel_state.py` |
| TP 通信函数 | `vllm/distributed/communication_op.py` |
| NCCL 封装 | `vllm/distributed/device_communicators/pynccl.py` |
| Column/Row 线性层 | `vllm/model_executor/layers/linear.py` |
| Llama 模型 TP 应用 | `vllm/model_executor/models/llama.py` |
| Worker 基类 | `vllm/worker/worker_base.py` |
| 并行配置 | `vllm/config.py` (ParallelConfig) |
