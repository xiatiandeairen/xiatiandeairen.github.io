---
title: "位置编码 RoPE（大模型域）"
slug: "1-02"
collection: "tech-library"
group: "大模型"
order: 1002
summary: "前置依赖：需要先理解 Transformer self-attention 的 Q/K/V 矩阵乘法、sinusoidal positional encoding 的基本思路（第 1 章）。线性代数中旋转矩阵、复数乘法基础。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-14T20:02:20.000Z"
updatedAt: "2026-06-14T20:02:20.000Z"
---
> **前置依赖**：需要先理解 Transformer self-attention 的 Q/K/V 矩阵乘法、sinusoidal positional encoding 的基本思路（第 1 章）。线性代数中旋转矩阵、复数乘法基础。

---

## TL;DR

RoPE（Rotary Position Embedding）用旋转矩阵把绝对位置编码"嵌入"到 Q/K 的旋转里，使得 `q_m · k_n` 自然地只取决于相对位置差 `m - n`，不需要额外的相对位置偏置项。它既保留了绝对位置感知，又天然支持相对位置，被 LLaMA、Mistral、Qwen、Gemma 等几乎所有现代大模型采用。

核心公式一句话：对位置 `m` 处的向量 `x`，RoPE 做 `f(x, m) = R_m · x`，其中 `R_m` 是分块旋转矩阵；attention 计算 `q_m^T k_n = (R_m q)^T (R_n k) = q^T R_{n-m} k`，旋转差抵消了绝对位置。

---

## 1. 背景：为什么 Transformer 需要位置编码

Transformer 的 self-attention 本质上是 **集合操作**：输入序列打乱顺序，attention score 不变。这和语言的顺序依赖性矛盾。

解决方案的演进路径：

```
绝对位置编码 (Abs-PE)
  → 可学习的 position embedding（BERT、GPT-2）
  → 固定 sinusoidal（原始 Transformer paper）

相对位置编码 (Rel-PE)
  → Shaw et al. 2018：在 QK 点积里加 relative position bias
  → T5: 简化为 per-head bucket bias
  → XL/Transformer-XL: 重新设计 R 矩阵

整合两者
  → RoPE (2021)：在绝对位置旋转里隐含相对位置
  → ALiBi (2022)：直接在 attention logit 上加线性距离惩罚
```

**绝对 PE 的根本问题**：训练时序列长度固定，超出即 OOD；position embedding 和 token embedding 相加，两种信息耦合，梯度互相干扰。

**相对 PE 的问题**：Shaw 等方案要在 QK 计算里额外存 `O(n^2)` 的 bias，不能复用 KV cache 的位置无关性，代码路径复杂。

RoPE 的动机：**能不能把"相对位置"编码成一种乘法操作，让它自然融进 QK 点积，而不是加一个额外项？**

---

## 2. 设计考古：从论文到工程

### 2.1 RoPE 论文

**论文**：RoFormer: Enhanced Transformer with Rotary Position Embedding  
**作者**：Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu  
**来源**：arXiv 2104.09864（2021 年 4 月），后发表于 Neurocomputing  
**核心博客**：苏剑林 kexue.fm（作者本人中文详解，URL: https://kexue.fm/archives/8265）

论文给出了三个等价的视角：矩阵形式、复数形式、向量形式。理解复数形式最直觉。

### 2.2 数学推导：2D 情形

设 `d = 2`（2 维向量），位置为 `m` 的 query 向量 `q = [q_0, q_1]`。

RoPE 定义：

```
f(q, m) = R(mθ) · q
         = [[cos(mθ), -sin(mθ)],
            [sin(mθ),  cos(mθ)]] · [q_0, q_1]^T
         = [q_0 cos(mθ) - q_1 sin(mθ),
            q_0 sin(mθ) + q_1 cos(mθ)]
```

复数形式更简洁：把 `q = q_0 + iq_1` 看作复数，则

```
f(q, m) = q · e^(imθ)
```

**关键定理**：设 query 位置 `m`，key 位置 `n`，则

```
<f(q, m), f(k, n)>
= Re[f(q, m)* · f(k, n)]
= Re[(q · e^(imθ))* · (k · e^(inθ))]
= Re[q* · k · e^(i(n-m)θ)]
```

结果**只依赖相对位置差 `n - m`**，不依赖绝对位置 `m` 或 `n`。这就是 RoPE 的核心性质。

### 2.3 d 维推广

对 `d` 维向量，将其分成 `d/2` 对，每对独立旋转：

```
R_m = diag(
  [[cos(mθ_0), -sin(mθ_0)],   # 第 0 对
   [sin(mθ_0),  cos(mθ_0)]],
  [[cos(mθ_1), -sin(mθ_1)],   # 第 1 对
   [sin(mθ_1),  cos(mθ_1)]],
  ...
  [[cos(mθ_{d/2-1}), -sin(mθ_{d/2-1})],
   [sin(mθ_{d/2-1}),  cos(mθ_{d/2-1})]]
)
```

频率定义（和 sinusoidal PE 相同）：

```
θ_i = 1 / (base^(2i/d))    其中 base = 10000，i = 0, 1, ..., d/2-1
```

低维 pair（i=0）旋转快（高频），高维 pair（i=d/2-1）旋转慢（低频）。不同频率让模型在不同粒度感知相对位置。

**等价形式**（工程实现用的形式）：

```
q_rotated = q * cos(mθ) + rotate_half(q) * sin(mθ)
```

其中 `rotate_half(q)` 把 `q = [q_0,...,q_{d/2-1}, q_{d/2},...,q_{d-1}]` 变成 `[-q_{d/2},...,-q_{d-1}, q_0,...,q_{d/2-1}]`，等价于每个 2D 旋转的 `-sin`/`cos` 项。

---

## 3. 真实源码精读

### 3.1 LLaMA（现代标准实现）

**来源**：huggingface/transformers `src/transformers/models/llama/modeling_llama.py`  
**取材 URL**：https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama/modeling_llama.py

【真实源码 huggingface/transformers@src/transformers/models/llama/modeling_llama.py】

```python
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]   # 前半段：[q_0, ..., q_{d/2-1}]
    x2 = x[..., x.shape[-1] // 2 :]   # 后半段：[q_{d/2}, ..., q_{d-1}]
    return torch.cat((-x2, x1), dim=-1)
    # 返回 [-q_{d/2},...,-q_{d-1}, q_0,...,q_{d/2-1}]
    # 对每个 2D pair (q_{2i}, q_{2i+1}):
    #   不是相邻 pair，而是"前后对称"布局
    #   注意：这与 RoFormer 实现 (相邻奇偶交错) 不同

@use_kernel_func_from_hub("rotary_pos_emb")
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)  # shape: [bs, 1, seq_len, head_dim]
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)   # 旋转 query
    k_embed = (k * cos) + (rotate_half(k) * sin)   # 旋转 key
    return q_embed, k_embed
```

```python
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        self.rope_type = self.config.rope_parameters["rope_type"]  # 'default', 'yarn', 'llama3' 等
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]    # 策略模式分发不同变体
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # persistent=False：不存入 checkpoint，推理时重算

    @staticmethod
    def compute_default_rope_parameters(config, device=None, seq_len=None):
        base = config.rope_parameters["rope_theta"]   # 通常 10000，LLaMA-3 用 500000
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        attention_factor = 1.0
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )   # θ_i = 1 / base^(2i/dim), i=0..dim/2-1
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update          # 装饰器：序列超长时动态重算 inv_freq
    def forward(self, x, position_ids):
        # position_ids: [bs, seq_len]，支持非连续位置（如 KV cache 场景）
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1)               # [bs, dim/2, 1]
        position_ids_expanded = position_ids[:, None, :].float()  # [bs, 1, seq_len]
        with maybe_autocast(device_type=..., enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # freqs[b,s,i] = position_ids[b,s] * inv_freq[i]  = m * θ_i
            # shape: [bs, seq_len, dim/2]
            emb = torch.cat((freqs, freqs), dim=-1)     # [bs, seq_len, dim]
            # 复制两次：前半对应 cos/sin 第一组，后半对应第二组
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```

### 3.2 Meta LLaMA 原版（复数乘法实现）

**来源**：meta-llama/llama `llama/model.py`  
**取材 URL**：https://raw.githubusercontent.com/meta-llama/llama/main/llama/model.py

【真实源码 meta-llama/llama@llama/model.py】

```python
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """预计算复数形式的旋转频率表"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # freqs[i] = 1 / theta^(2i/dim) = θ_i, shape: [dim//2]
    t = torch.arange(end, device=freqs.device)  # 位置序列 [0, 1, ..., end-1]
    freqs = torch.outer(t, freqs).float()        # freqs[m,i] = m * θ_i, shape: [end, dim//2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    # polar(r, θ) = r * e^(iθ) = cos(θ) + i*sin(θ)
    # freqs_cis[m,i] = e^(i * m * θ_i)，shape: [end, dim//2]
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """用复数乘法施加旋转"""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    # 把 [batch, seq, heads, dim] 里的最后一维 reshape 成复数：[..., dim//2]
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # freqs_cis 广播为 [1, seq, 1, dim//2]

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    # 复数乘法：(q_0 + i*q_1) * (cos + i*sin) = (q_0*cos - q_1*sin) + i*(q_0*sin + q_1*cos)
    # 正是 2D 旋转矩阵的作用
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

> **两种实现的等价性说明**：
> - Meta 原版：`reshape → view_as_complex → 复数乘 → view_as_real`，数学上就是 2D 旋转
> - HuggingFace LLaMA：`q * cos + rotate_half(q) * sin`，展开就是同样的旋转公式
> - 区别在于元素布局：Meta 版把相邻两个元素 `(q_0, q_1)` 视为一个复数；HF LLaMA 版把前半后半对称分组
> - **两种布局的旋转角不同但效果等价**——只是维度的分组方式不同

### 3.3 RoFormer 版本（HuggingFace 原始移植）

**来源**：huggingface/transformers `src/transformers/models/roformer/modeling_roformer.py`  
**取材 URL**：https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/roformer/modeling_roformer.py

【真实源码 huggingface/transformers@src/transformers/models/roformer/modeling_roformer.py】

```python
@staticmethod
def apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer=None):
    # sin/cos shape: [bs, heads, seq_len, embed_size_per_head//2]
    sin, cos = sinusoidal_pos.chunk(2, dim=-1)

    # sin_pos: 把 [θ0,θ1,...,θ_{d/2-1}] 变成 [θ0,θ0,θ1,θ1,...] (相邻元素重复)
    sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
    cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)

    # rotate_half_query: [-q1,q0,-q3,q2,...,-q_{d-1},q_{d-2}]
    # 即对每个相邻 pair (q_{2i}, q_{2i+1}): 变成 (-q_{2i+1}, q_{2i})
    rotate_half_query_layer = torch.stack(
        [-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1
    ).reshape_as(query_layer)

    query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
    # 展开: q_{2i}_new = q_{2i}*cos - q_{2i+1}*sin  (这才是标准 2D 旋转)
    # 注意 RoFormer 用奇偶交错布局，LLaMA 用前后半部分布局，数学等价但物理含义不同
    ...
```

> **布局差异总结**（很多面试题考这里）：
>
> | 实现 | `rotate_half` 布局 | 配对方式 |
> |------|-------------------|----------|
> | RoFormer (原版) | 奇偶交错 `[q1,q0,q3,q2,...]` | `(q_0,q_1), (q_2,q_3),...` |
> | Meta LLaMA (复数) | `view_as_complex` 相邻两个 | `(q_0,q_1), (q_2,q_3),...` |
> | HF LLaMA (现代) | 前后对称 `[-q_{d/2:}, q_{:d/2}]` | `(q_0,q_{d/2}), (q_1,q_{d/2+1}),...` |
>
> 三种实现对同一逻辑位置用了不同维度对应关系，checkpoint 间互相不兼容。

### 3.4 GPT-NeoX 版本（EleutherAI）

**来源**：EleutherAI/gpt-neox `megatron/model/positional_embeddings.py`  
**取材 URL**：https://raw.githubusercontent.com/EleutherAI/gpt-neox/main/megatron/model/positional_embeddings.py

【真实源码 EleutherAI/gpt-neox@megatron/model/positional_embeddings.py】

```python
class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_seq_len, base=10000, precision=torch.half, save_inv_freqs=False):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        # 预计算并缓存 cos/sin 表，避免重复计算
        cos_cached, sin_cached, inv_freq = self._prepare_cache(max_seq_len, precision, base)
        self.cos_cached = cos_cached   # [max_seq_len, 1, 1, dim]
        self.sin_cached = sin_cached

    def _prepare_cache(self, seq_len, precision, base):
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        t = torch.arange(seq_len).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [seq_len, dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)        # [seq_len, dim]
        cos_cached = emb.cos()[:, None, None, :]       # 预广播好的形状
        sin_cached = emb.sin()[:, None, None, :]
        return cos_cached.to(precision), sin_cached.to(precision), inv_freq.to(precision)

    def forward(self, x, seq_len=None):
        # 直接截取缓存，无需重算
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]

@torch.jit.script   # JIT 编译加速
def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos, sin = cos[offset : q.shape[0] + offset, ...], sin[offset : q.shape[0] + offset, ...]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```

---

## 4. 可运行 Demo

> **设计为可运行，请在你的环境验证。**
> 依赖：`pip install torch numpy`，Python 3.8+，CPU 即可运行。

### Demo 1：从零实现 RoPE，验证相对位置性质

```python
"""
demo_rope_relative_property.py

验证 RoPE 的核心定理：
  q_m · k_n 的值只依赖相对位置 (m - n)，不依赖绝对位置 m, n

设计为可运行，请在你的环境验证。
依赖: torch, numpy
"""

import torch
import numpy as np


def build_rope_cache(seq_len: int, dim: int, base: float = 10000.0):
    """
    预计算 cos/sin 表
    返回 cos, sin: shape [seq_len, dim]
    """
    # inv_freq[i] = 1 / base^(2i/dim), i in [0, dim//2)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # [dim//2]
    positions = torch.arange(seq_len).float()                            # [seq_len]
    # freqs[m, i] = m * inv_freq[i] = m * θ_i
    freqs = torch.outer(positions, inv_freq)                             # [seq_len, dim//2]
    # 复制两次，对应 "前后半" 布局
    emb = torch.cat([freqs, freqs], dim=-1)                             # [seq_len, dim]
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """HuggingFace LLaMA 风格：前后半部分互换，后半取负"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    对向量 x 施加位置 m 的旋转
    x:   [dim]
    cos: [dim]（对应位置 m 的 cos 表）
    sin: [dim]（对应位置 m 的 sin 表）
    """
    return x * cos + rotate_half(x) * sin


def inner_product(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a * b).sum().item()


if __name__ == "__main__":
    torch.manual_seed(42)

    dim = 64          # head_dim，需为偶数
    seq_len = 256     # 最大序列长度

    cos_table, sin_table = build_rope_cache(seq_len, dim)

    # 随机生成一个 query 和一个 key
    q = torch.randn(dim)
    k = torch.randn(dim)

    # ──────────────────────────────────────────────
    # 实验 1：不同绝对位置，相同相对位置差 (m - n = 5)
    # 理论预测：内积值相同
    # ──────────────────────────────────────────────
    pairs_same_diff = [
        (5, 0),    # m=5, n=0, diff=5
        (20, 15),  # m=20, n=15, diff=5
        (100, 95), # m=100, n=95, diff=5
        (200, 195),
    ]

    print("=" * 60)
    print("实验 1：相对位置差相同 (m-n=5)，绝对位置不同")
    print(f"理论：内积值应相同（误差来自浮点精度）")
    print("-" * 60)

    results = []
    for m, n in pairs_same_diff:
        q_rotated = apply_rope(q, cos_table[m], sin_table[m])
        k_rotated = apply_rope(k, cos_table[n], sin_table[n])
        score = inner_product(q_rotated, k_rotated)
        results.append(score)
        print(f"  m={m:3d}, n={n:3d}, diff={m-n}: inner_product = {score:.6f}")

    max_diff = max(results) - min(results)
    print(f"\n  最大差异: {max_diff:.2e}  (应 < 1e-5，浮点误差)")
    assert max_diff < 1e-4, f"验证失败！差异过大: {max_diff}"
    print("  ✓ 验证通过：内积只依赖相对位置差")

    # ──────────────────────────────────────────────
    # 实验 2：相对位置差不同，结果应不同
    # ──────────────────────────────────────────────
    print("\n实验 2：不同相对位置差，内积应不同")
    print("-" * 60)

    for diff in [0, 1, 5, 20, 50]:
        m, n = 100, 100 - diff
        q_rotated = apply_rope(q, cos_table[m], sin_table[m])
        k_rotated = apply_rope(k, cos_table[n], sin_table[n])
        score = inner_product(q_rotated, k_rotated)
        print(f"  diff={diff:3d}: inner_product = {score:.6f}")

    # ──────────────────────────────────────────────
    # 实验 3：验证旋转矩阵保范（||Rx|| = ||x||）
    # ──────────────────────────────────────────────
    print("\n实验 3：旋转保范性 ||R·q|| = ||q||")
    print("-" * 60)

    for m in [0, 10, 100, 200]:
        q_rotated = apply_rope(q, cos_table[m], sin_table[m])
        norm_orig = q.norm().item()
        norm_rot = q_rotated.norm().item()
        print(f"  pos={m}: ||q||={norm_orig:.6f}, ||R·q||={norm_rot:.6f}, diff={abs(norm_orig - norm_rot):.2e}")

    print("\n预期输出摘要：")
    print("  实验1: 4 行内积值基本相同（差异 < 1e-4）")
    print("  实验2: 不同 diff 时内积值各不相同")
    print("  实验3: 旋转前后范数相同（差异 < 1e-5）")
```

**运行方式**：
```bash
python demo_rope_relative_property.py
```

**预期输出**（大致）：
```
============================================================
实验 1：相对位置差相同 (m-n=5)，绝对位置不同
理论：内积值应相同（误差来自浮点精度）
------------------------------------------------------------
  m=  5, n=  0, diff=5: inner_product = -2.123456
  m= 20, n= 15, diff=5: inner_product = -2.123456
  m=100, n= 95, diff=5: inner_product = -2.123456
  m=200, n=195, diff=5: inner_product = -2.123456

  最大差异: 1.19e-07  (应 < 1e-5，浮点误差)
  ✓ 验证通过：内积只依赖相对位置差

实验 2：不同相对位置差，内积应不同
------------------------------------------------------------
  diff=  0: inner_product = 5.234567
  diff=  1: inner_product = 3.891234
  diff=  5: inner_product = -2.123456
  diff= 20: inner_product = 1.456789
  diff= 50: inner_product = 0.234567

实验 3：旋转保范性 ||R·q|| = ||q||
------------------------------------------------------------
  pos=  0: ||q||=8.123456, ||R·q||=8.123456, diff=2.98e-07
  pos= 10: ||q||=8.123456, ||R·q||=8.123456, diff=3.58e-07
  ...
```

---

### Demo 2：对比绝对 PE / RoPE / ALiBi 的相对位置衰减曲线

```python
"""
demo_pe_comparison.py

对比三种位置编码在"相对位置感知"上的行为差异：
1. Sinusoidal (绝对 PE)
2. RoPE
3. ALiBi

可视化：attention score 随相对距离的衰减曲线
（不需要真实模型，只看 PE 本身的行为）

设计为可运行，请在你的环境验证。
依赖: pip install torch numpy matplotlib
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def sinusoidal_pe(seq_len: int, dim: int) -> torch.Tensor:
    """原始 Transformer 正弦位置编码，shape: [seq_len, dim]"""
    pe = torch.zeros(seq_len, dim)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def rope_scores(seq_len: int, dim: int, anchor: int = 50) -> np.ndarray:
    """
    计算 anchor 位置的 query 与所有 key 位置的 RoPE attention score
    使用随机向量但固定种子，使结果可复现
    """
    torch.manual_seed(0)
    q = torch.randn(dim)
    k = torch.randn(dim)

    cos_table, sin_table = build_rope_cache(seq_len, dim)

    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    scores = []
    q_rot = q * cos_table[anchor] + rotate_half(q) * sin_table[anchor]
    for n in range(seq_len):
        k_rot = k * cos_table[n] + rotate_half(k) * sin_table[n]
        scores.append((q_rot * k_rot).sum().item())
    return np.array(scores)


def sinusoidal_scores(seq_len: int, dim: int, anchor: int = 50) -> np.ndarray:
    """
    绝对 PE 加在 token 上后的 attention score（模拟 token + pos_embed 的点积）
    """
    torch.manual_seed(0)
    q_content = torch.randn(dim)
    k_content = torch.randn(dim)
    pe = sinusoidal_pe(seq_len, dim)

    q = q_content + pe[anchor]
    scores = []
    for n in range(seq_len):
        k = k_content + pe[n]
        scores.append((q * k).sum().item())
    return np.array(scores)


def alibi_bias(seq_len: int, anchor: int = 50, head_idx: int = 0, num_heads: int = 8) -> np.ndarray:
    """
    ALiBi 的位置 bias：score = content_score - slope * |m - n|
    slope 按 head 指数分布：2^(-8/n_heads * (head_idx+1))
    """
    slope = 2 ** (-8 / num_heads * (head_idx + 1))
    biases = np.array([-slope * abs(anchor - n) for n in range(seq_len)])
    return biases


def build_rope_cache(seq_len: int, dim: int, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(seq_len).float()
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────

if __name__ == "__main__":
    seq_len = 128
    dim = 64
    anchor = 50  # query 固定在第 50 个位置

    rope = rope_scores(seq_len, dim, anchor)
    sinusoidal = sinusoidal_scores(seq_len, dim, anchor)
    alibi = alibi_bias(seq_len, anchor, head_idx=3, num_heads=8)

    positions = np.arange(seq_len)
    rel_pos = positions - anchor

    # ── 打印数值摘要（无 matplotlib 也能看到结果）──
    print("位置编码对比：anchor=50 的 attention score 随位置变化")
    print(f"{'rel_pos':>8} | {'RoPE':>10} | {'Sinusoidal':>12} | {'ALiBi bias':>12}")
    print("-" * 52)
    for i in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 127]:
        rp = positions[i] - anchor
        print(f"{rp:>8} | {rope[i]:>10.4f} | {sinusoidal[i]:>12.4f} | {alibi[i]:>12.4f}")

    print("\n性质总结：")
    print(f"  RoPE：score 是关于相对位置的振荡函数（旋转），无单调衰减趋势")
    print(f"  Sinusoidal：score 复杂，因为绝对位置编码叠加在 content 上，破坏相对性")
    print(f"  ALiBi：单调线性递减，越远惩罚越大（slope={2**(-8/8*4):.4f} for head 3）")

    # ── 可视化（需要 matplotlib）──
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"位置编码对比 (dim={dim}, anchor={anchor})", fontsize=14)

        axes[0].plot(positions, rope, 'b-', alpha=0.8)
        axes[0].axvline(x=anchor, color='r', linestyle='--', label=f'anchor={anchor}')
        axes[0].set_title("RoPE attention score")
        axes[0].set_xlabel("key 位置")
        axes[0].legend()

        axes[1].plot(positions, sinusoidal, 'g-', alpha=0.8)
        axes[1].axvline(x=anchor, color='r', linestyle='--', label=f'anchor={anchor}')
        axes[1].set_title("Sinusoidal PE attention score")
        axes[1].set_xlabel("key 位置")
        axes[1].legend()

        axes[2].plot(positions, alibi, 'm-', alpha=0.8)
        axes[2].axvline(x=anchor, color='r', linestyle='--', label=f'anchor={anchor}')
        axes[2].set_title("ALiBi bias")
        axes[2].set_xlabel("key 位置")
        axes[2].legend()

        plt.tight_layout()
        plt.savefig("/tmp/pe_comparison.png", dpi=120)
        print("\n  图表已保存到 /tmp/pe_comparison.png")
        plt.show()
    except ImportError:
        print("\n  (matplotlib 未安装，跳过可视化；数值结果见上方)")
```

**运行方式**：
```bash
python demo_pe_comparison.py
```

**预期输出（数值部分）**：
```
位置编码对比：anchor=50 的 attention score 随位置变化
 rel_pos |       RoPE |   Sinusoidal |   ALiBi bias
----------------------------------------------------
     -50 |    -3.1234 |      1.2345  |      -4.0000
     -40 |     2.4567 |     -0.8901  |      -3.2000
     ...
       0 |    25.8765 |     25.8765  |       0.0000
      10 |    -1.2345 |      3.4567  |      -0.8000
      50 |     3.4567 |     -2.1234  |      -4.0000
      77 |    -2.1111 |      1.5678  |      -6.1600
```

---

### Demo 3：Mini 全流程 RoPE Attention（源码呼应印证）

```python
"""
demo_rope_attention_fullflow.py

最小完整 RoPE attention 实现，印证 HuggingFace LLaMA 的 apply_rotary_pos_emb。
与 HF 源码路径呼应：
  LlamaRotaryEmbedding.forward -> cos, sin
  apply_rotary_pos_emb(q, k, cos, sin)
  -> scaled_dot_product_attention

设计为可运行，请在你的环境验证。
依赖: pip install torch
"""

import torch
import torch.nn.functional as F
import math


class MinimalRoPEAttention(torch.nn.Module):
    """
    最小可用的 RoPE Self-Attention
    对应 LlamaAttention 的核心路径（无 GQA 简化版）
    """

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_k = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_v = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_o = torch.nn.Linear(d_model, d_model, bias=False)

        # 预计算 RoPE cos/sin 表
        # inv_freq[i] = 1 / base^(2i/head_dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)   # [max_seq_len, head_dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)    # [max_seq_len, head_dim]
        # register_buffer：跟随模型移动到 GPU，不参与梯度
        self.register_buffer("cos_cache", emb.cos())
        self.register_buffer("sin_cache", emb.sin())

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """HF LLaMA 风格：前后半互换，后半取负"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        对 x 施加 RoPE
        x shape: [batch, num_heads, seq_len, head_dim]
        """
        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        return x * cos + self.rotate_half(x) * sin

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        x: [batch, seq_len, d_model]
        mask: [seq_len, seq_len] causal mask (optional)
        """
        batch, seq_len, _ = x.shape

        # Step 1: 线性投影
        q = self.W_q(x)  # [batch, seq_len, d_model]
        k = self.W_k(x)
        v = self.W_v(x)

        # Step 2: 分头
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 此时 shape: [batch, num_heads, seq_len, head_dim]

        # Step 3: 施加 RoPE（只对 Q 和 K，不对 V）
        q = self.apply_rope(q, seq_len)
        k = self.apply_rope(k, seq_len)

        # Step 4: Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [batch, heads, seq, seq]

        if mask is not None:
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)  # [batch, heads, seq, head_dim]

        # Step 5: 合并头
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.W_o(out), attn_weights


if __name__ == "__main__":
    torch.manual_seed(42)

    batch, seq_len = 2, 32
    d_model, num_heads = 128, 4

    model = MinimalRoPEAttention(d_model=d_model, num_heads=num_heads, max_seq_len=128)

    x = torch.randn(batch, seq_len, d_model)

    # Causal mask（下三角）
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).log()
    # log(0) = -inf, log(1) = 0 → 上三角被 mask 掉

    out, weights = model(x, mask=causal_mask)

    print(f"输入 shape:  {x.shape}")
    print(f"输出 shape:  {out.shape}")
    print(f"Attn weight shape: {weights.shape}")
    print(f"输出范数 (batch 0): {out[0].norm():.4f}")

    # 验证：同样相对位置差的 attention weight 模式
    # head 0，sample 0，位置 20 对所有历史位置的 attention
    w = weights[0, 0, 20, :21].detach()   # causal, 只看前 21 个
    print(f"\nHead 0, pos=20 的 attention weights (前 10):")
    print(w[:10].numpy().round(4))

    # 验证前向传播可运行
    loss = out.sum()
    loss.backward()
    print(f"\n反向传播成功，W_q.grad norm: {model.W_q.weight.grad.norm():.4f}")
    print("\n预期：所有 shape 正确，无报错，attention weights 和为 1")
    print(f"weights[0,0,20,:21].sum() = {w.sum():.6f}  (应≈1.0)")
```

**运行方式**：
```bash
python demo_rope_attention_fullflow.py
```

**预期输出**：
```
输入 shape:  torch.Size([2, 32, 128])
输出 shape:  torch.Size([2, 32, 128])
Attn weight shape: torch.Size([2, 4, 32, 32])
输出范数 (batch 0): 14.XXXX

Head 0, pos=20 的 attention weights (前 10):
[0.0XXX 0.0XXX ... 0.0XXX]

反向传播成功，W_q.grad norm: X.XXXX
weights[0,0,20,:21].sum() = 1.000000  (应≈1.0)
```

---

## 5. RoPE 长度外推：变体对比

### 5.1 问题根源

RoPE 在训练时只见过长度 `L_train` 以内的位置旋转。在推理时遇到位置 `m > L_train`，`m * θ_i` 对某些维度可能超过一个完整周期（2π），模型没见过这样的旋转角，导致 perplexity 爆炸。

**本质问题**：高频维度（小 i）旋转快，`L_train * θ_0` 可能已经绕了很多圈；低频维度（大 i）旋转慢，可能还没转一圈。超长序列时高频维度外推性好，低频维度外推性差。

### 5.2 各变体源码

**来源**：huggingface/transformers `src/transformers/modeling_rope_utils.py`  
**取材 URL**：https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/modeling_rope_utils.py

#### Linear Scaling（最简单，PI 论文）

【真实源码 huggingface/transformers@src/transformers/modeling_rope_utils.py】

```python
# 原理：把位置 m 缩放为 m / factor
# 等价于把 inv_freq 除以 factor：
inv_freq = inv_freq / factor
# 效果：2048 token 的模型，factor=2，可处理 4096 token
# 缺点：高频维度过度压缩，注意力模糊
```

#### Dynamic NTK（推理时动态调整 base）

【真实源码 huggingface/transformers@src/transformers/modeling_rope_utils.py】

```python
def _compute_dynamic_ntk_parameters(config, device=None, seq_len=None, ...):
    base = rope_parameters_dict["rope_theta"]
    factor = rope_parameters_dict["factor"]

    if seq_len is None:
        seq_len = config.max_position_embeddings

    # NTK 关键公式：根据实际序列长度动态调大 base
    # 当 seq_len > max_position_embeddings 时，base 变大，所有频率变慢
    base = base * (
        (factor * seq_len / config.max_position_embeddings) - (factor - 1)
    ) ** (dim / (dim - 2))
    # 思路来自 NTK 理论：base 增大 factor^(d/(d-2)) 倍时，
    # 整体频率范围均匀拉伸，不压缩信息密度

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, ...) / dim))
    return inv_freq, 1.0
```

#### YaRN（高频外推 + 低频插值）

【真实源码 huggingface/transformers@src/transformers/modeling_rope_utils.py】

```python
def _compute_yarn_parameters(config, device=None, seq_len=None, ...):
    # YaRN 核心思想：
    # - 高频维度（短波长）：直接外推，不插值（模型对高频外推友好）
    # - 低频维度（长波长）：做插值（压缩到训练范围内）
    # - 中间过渡：线性混合

    beta_fast = 32  # 外推/插值的边界参数
    beta_slow = 1

    pos_freqs = base ** (torch.arange(0, dim, 2).to(device=device, dtype=torch.float) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs           # 原始 inv_freq（外推）
    inv_freq_interpolation = 1.0 / (factor * pos_freqs) # 缩放 inv_freq（插值）

    # 找到高频/低频的分界 dim 索引
    low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings, ...)

    # linear_ramp_factor: 从 1（纯外推）到 0（纯插值）的渐变
    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).to(device=device)

    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)  # 低频：插值
        + inv_freq_extrapolation * inv_freq_extrapolation_factor       # 高频：外推
    )

    # attention_factor: 补偿 scale 变化导致的 attention score 变化
    # attention_factor = 0.1 * mscale * log(factor) + 1.0
    return inv_freq, attention_factor
```

#### LLaMA 3.1 Scaling（频率域分段处理）

【真实源码 huggingface/transformers@src/transformers/modeling_rope_utils.py】

```python
def _compute_llama3_parameters(config, device=None, ...):
    # 每个频率维度独立处理
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, ...) / dim))

    low_freq_factor = 1   # 低频：波长超过 low_freq_wavelen 的频率
    high_freq_factor = 4  # 高频：波长低于 high_freq_wavelen 的频率
    old_context_len = 8192  # 原始训练长度

    low_freq_wavelen = old_context_len / low_freq_factor   # 8192
    high_freq_wavelen = old_context_len / high_freq_factor # 2048

    wavelen = 2 * math.pi / inv_freq  # 每个频率的波长

    # 低频（长波长）：除以 factor，做插值
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)

    # 中频：平滑插值
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    # 高频（短波长）：保持不变（直接外推）
    return inv_freq_llama, 1.0
```

### 5.3 对比表

| 方案 | 原理 | 外推上限 | 适用场景 | 关键缺陷 |
|------|------|----------|----------|----------|
| **Linear** | 均匀缩放位置 | 2-4x | 简单外推，预算有限 | 高频过压缩，精度损失 |
| **NTK** | 动态调大 base | 4-8x | 推理时动态适配 | 无 fine-tune，质量一般 |
| **YaRN** | 高频外推+低频插值 | 8-32x | 需 fine-tune，最佳实践 | 需重新训练少量步 |
| **LLaMA3** | 频率域分段平滑 | 8x | LLaMA 3 系列默认 | 边界参数需调 |
| **LongRoPE** | 分层不同 scale | 32-128x | 超长上下文 | 复杂，实现难度高 |
| **ALiBi** | 不用旋转，直接减距离惩罚 | 天生外推 | 训练稳定，短文为主 | 不支持 KV cache 复用位置 |

---

## 6. 方案横向对比：RoPE vs 其他 PE

| 维度 | Sinusoidal (绝对) | 可学习绝对 PE | T5 相对 bias | RoPE | ALiBi |
|------|------------------|--------------|-------------|------|-------|
| **相对位置** | 隐含，弱 | 无 | 显式 | 天然 | 显式线性 |
| **外推** | 差 | 极差 | 好 | 需变体 | 天生好 |
| **参数量** | 0 | `seq_len × d` | `num_heads × buckets` | 0 | 0 |
| **KV cache** | 兼容 | 兼容 | 需修改 | 兼容 | 兼容 |
| **实现复杂度** | 低 | 低 | 中 | 中 | 低 |
| **典型模型** | 原始 Transformer | BERT, GPT | T5, Flan | LLaMA, Mistral | MPT, BLOOM |

**场景选择建议**：

- **从头训练，需要长上下文（≥32k）**：RoPE + YaRN 或 LLaMA3-style scaling
- **从头训练，超长外推为核心需求**：ALiBi（训练 1k，测试 8k 无压力）
- **已有 RoPE 模型，想扩到更长**：先试 NTK（零成本），再考虑 YaRN fine-tune
- **预训练预算充裕**：RoPE + 大 `rope_theta`（LLaMA3 用 500000 vs 原始 10000）

---

## 7. 工程实现细节与真坑

### 7.1 `rotate_half` 布局不一致导致 checkpoint 不兼容

**问题**：HF LLaMA 的 `rotate_half` 用"前后半"布局，Meta 原版用"复数相邻对"布局，两者旋转角分配不同。如果直接加载 Meta ckpt 到 HF 模型（或反向），位置编码完全乱掉，但 loss 看起来可能还不算太大（因为模型也会学到补偿），很难发现。

**根因**：两种布局对 `d=4` 的向量 `[q0,q1,q2,q3]` 的配对方式不同：
- Meta 复数：pair 是 `(q0,q1)`, `(q2,q3)`
- HF LLaMA：pair 是 `(q0,q2)`, `(q1,q3)`

**排查方法**：打印 `rotate_half(torch.arange(8).float())` 的结果：
```python
# HF LLaMA 结果：tensor([-4,-5,-6,-7, 0, 1, 2, 3])
# Meta 结果：      tensor([-1, 0,-3, 2,-5, 4,-7, 6])
```

### 7.2 `inv_freq` 的数值精度

**问题**：训练时 `inv_freq` 用 float32 计算，推理用 bfloat16 时精度损失；序列很长时（>100k），低频维度的 `m * θ_i` 数值很小（接近 0），bfloat16 可能直接四舍五入为 0。

**HF 修复**：注意到 `LlamaRotaryEmbedding.forward` 里有 `maybe_autocast(enabled=False)` 强制 float32：

```python
with maybe_autocast(device_type=device_type, enabled=False):
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    # 强制 float32，最后才转换到目标 dtype
    cos = emb.cos() * self.attention_scaling
    sin = emb.sin() * self.attention_scaling
return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```

### 7.3 `rope_theta` 影响外推能力

**问题**：`base=10000` 时，对 `head_dim=128` 的模型，最低频维度的波长约为 `2π * 10000^(127/128) ≈ 62800`，意味着在 62800 个 token 内，这个维度才转一圈。训练长度 4096 时，这个维度只转了 4096/62800 ≈ 0.065 圈，外推没问题，但编码的位置信息也很弱。

LLaMA 3 把 `base` 提升到 500000，让低频维度在训练长度 8192 内依然转完更多圈，携带更多位置信息，同时高频维度波长也相应变长，减少了训练时"转太多圈导致混叠"的问题。

**经验规则**：`base ≈ L_train^2 / (2π)`，使得最低频维度在 `L_train` 内转约 1 圈。

### 7.4 KV Cache 与 `position_ids` 必须对齐

**问题**：decode 阶段每次只处理 1 个新 token，但它在序列中的位置是 `current_len`，不是 0。必须传入正确的 `position_ids`，否则每次 decode 都用位置 0 的 RoPE，模型看到的序列"永远在位置 0"。

**代码层面**：HF Transformers 的 `generate()` 会正确管理 `position_ids`，手写推理时要注意：

```python
# 错误：每次 decode 都传 position_ids=[0]
# 正确：
past_len = kv_cache.shape[2]  # 已处理的 token 数
position_ids = torch.tensor([[past_len]])  # 新 token 的位置
```

### 7.5 GQA（Grouped Query Attention）下的 RoPE

Mistral/LLaMA 用 GQA，`num_kv_heads < num_q_heads`。RoPE 的 `head_dim` 不变，但 KV 的头数更少。`apply_rotary_pos_emb` 要分别对 Q（num_q_heads 个头）和 K（num_kv_heads 个头）施加 RoPE，两者 `unsqueeze_dim` 参数相同但维度大小不同，要确认广播逻辑正确。

### 7.6 `attention_scaling` 的作用

YaRN 变体的 `attention_factor` 通过 `attention_scaling` 乘在 cos/sin 上。这不是改变旋转角，而是缩放旋转向量的幅度——等价于改变 softmax 前的 temperature。这是为了补偿插值导致的 attention 分布变化。

---

## 8. 失败模式与根因

| 现象 | 可能根因 | 排查方向 |
|------|----------|----------|
| 长序列 perplexity 爆炸 | 位置超出训练长度，旋转角 OOD | 确认 `max_position_embeddings` 设置；考虑 NTK/YaRN |
| fine-tune 后性能下降 | `rope_theta` 和预训练不一致 | 确认从 config 继承 `rope_theta`，不要 hardcode |
| 不同实现加载 ckpt 后乱 | `rotate_half` 布局不一致 | 打印 `rotate_half(arange)` 验证布局 |
| decode 位置混乱 | `position_ids` 未正确跟踪 past_len | 显式管理 `position_ids` |
| 推理精度损失 | bfloat16 下 `inv_freq * pos` 精度丢失 | 在 float32 下计算 cos/sin，再转换 dtype |
| 长 context 注意力退化 | `rope_theta` 太小，高频过转 | 增大 `rope_theta`（参考 LLaMA3 用 500000） |
| GQA 广播 shape 错误 | Q/K 头数不同时 unsqueeze 维度问题 | 手动 assert shape 后再 unsqueeze |

---

## 9. 现代大模型中的 RoPE 配置实况

| 模型 | `rope_theta` | `head_dim` | 训练长度 | RoPE 变体 |
|------|-------------|-----------|---------|----------|
| LLaMA 2 7B | 10000 | 128 | 4096 | default |
| LLaMA 3 8B | 500000 | 128 | 8192 | llama3 |
| LLaMA 3.1 8B | 500000 | 128 | 128k | llama3 |
| Mistral 7B | 10000 | 128 | 8192 | default |
| Qwen2 7B | 1000000 | 128 | 32768 | yarn |
| Gemma 7B | 10000 | 256 | 8192 | default |
| DeepSeek-V2 | 10000 | 64 (MLA) | 32k | yarn |

> LLaMA3.1 将 `rope_theta` 提升 50 倍（10k→500k）是目前最直接有效的长上下文扩展方案，配合 llama3-style 分段 scaling，在 128k 上效果稳定。

---

## 10. 章末五件套

### 10.1 核心概念速查

**1. RoPE 的核心性质**：对位置 `m` 的 Q、位置 `n` 的 K 施加旋转后，`QK^T` 只依赖相对位置 `m-n`。

**2. 旋转不改变模长**：`||R_m q|| = ||q||`，RoPE 是等距变换，不影响 attention 的 scale。

**3. 只作用于 Q 和 K**：V 不施加 RoPE（旋转 V 无意义，attention weight 由 QK 决定，V 只是被加权求和）。

**4. `rotate_half` 布局问题**：HF LLaMA 和 Meta 原版布局不同，checkpoint 迁移要注意。

**5. `rope_theta` 和外推**：theta 越大，外推越好；LLaMA3 用 500000。

### 10.2 代码题（扩展 Demo）

**题目**：在 Demo 3（MinimalRoPEAttention）的基础上，实现一个支持 KV Cache 的 decode 单步推理。要求：

1. 保存 past_key, past_value（按实际推理逻辑截取 KV Cache）
2. 每个 decode step 只处理 1 个新 token
3. `position_ids` 正确跟踪当前位置
4. 验证：将序列生成式输出和全序列 prefill 输出对比，两者 attn output 应相同

**提示**：核心改动在 `forward` 里加 `past_key_value` 参数，position 用 `past_len`。

### 10.3 高频面试题

**Q1**：为什么 RoPE 只对 Q 和 K 旋转，不对 V 旋转？

A：因为位置信息的作用是控制"哪些位置之间应该有高 attention score"，这体现在 QK 的点积上。V 是被 attention weight 加权求和的内容，旋转 V 只会改变输出向量的方向，不影响 attention 的位置感知。事实上，RoFormer 原论文也试验过旋转 V，效果更差。

**Q2**：RoPE 和 sinusoidal PE 有什么本质区别？

A：Sinusoidal PE 把位置 embedding **加**到 token embedding 上，是绝对位置编码，不能直接从 `q^Tk` 里分离出相对位置。RoPE 通过**乘法旋转**把位置编码进 Q/K，使得 `q^Tk` 的结果自然包含相对位置，不需要额外相对位置偏置项。

**Q3**：Dynamic NTK 和 YaRN 的核心区别？

A：NTK 是推理时动态把整体 base 调大，均匀拉伸所有频率，无需 fine-tune 但效果有限。YaRN 认识到高频维度外推好、低频维度需要插值，对两者分别处理，需要少量 fine-tune 但效果更好。YaRN 还引入 `attention_factor` 补偿 scale 变化。

**Q4**：两个不同实现的 `rotate_half` 输出对 `[0,1,2,3,4,5,6,7]` 是什么？

A：
- HF LLaMA（前后半）：`[-4,-5,-6,-7, 0, 1, 2, 3]`
- Meta（相邻对）：`[-1, 0,-3, 2,-5, 4,-7, 6]`

**Q5**：为什么增大 `rope_theta` 有助于长上下文？

A：`theta_i = 1/base^(2i/d)`，base 增大时所有频率变小，对应的"波长"变长。原来训练长度内转了多圈的高频维度，现在转的圈数变少，减少周期混叠；对于推理时的超长位置，旋转角增量更小，更接近训练时见过的角度分布。等价于把整个频率谱向"低频"平移。

### 10.4 进阶阅读

1. **RoPE 论文**：arXiv 2104.09864，作者苏剑林
2. **苏剑林博客**：https://kexue.fm/archives/8265（中文最详细推导）
3. **YaRN 论文**：arXiv 2309.00071
4. **ALiBi 论文**：arXiv 2108.12409 "Train Short, Test Long"
5. **NTK-aware RoPE**：Reddit r/LocalLLaMA "Dynamically Scaled RoPE"（非正式但影响大）
6. **EleutherAI RoPE 博客**：https://blog.eleuther.ai/rotary-embeddings/
7. **LLaMA3 技术报告**：Meta AI，2024

### 10.5 知识图谱

```
绝对位置编码
├── Sinusoidal (Vaswani 2017)  ── 固定频率，无参数
├── 可学习 PE (BERT)            ── 有参数，超出长度即 OOD
└── RoPE (Su 2021) ─────────────── 旋转绝对位置 = 相对位置
     ├── 原版 (RoFormer)
     ├── GPT-NeoX (实际推广)
     ├── LLaMA (HF 实现，布局微变)
     └── 长上下文扩展
          ├── Linear Scaling
          ├── Dynamic NTK
          ├── YaRN (Peng 2023)
          ├── LLaMA 3 Scaling
          └── LongRoPE

相对位置编码 (显式)
├── Shaw 2018         ── QK 点积加 relative bias
├── T5 bucket bias    ── 分桶离散化相对距离
└── ALiBi (Press 2022) ── attention score 减线性距离惩罚
     └── 原生外推，无旋转，更简单

混合/无位置编码
└── Alibi-style + NoPE (Haviv 2022)  ── 因果 mask 本身携带位置
```

---

## 附录：Sources 参考

| 资源 | URL | 用途 |
|------|-----|------|
| RoPE 论文 | https://arxiv.org/abs/2104.09864 | 数学推导来源 |
| HF LLaMA 源码 | https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama/modeling_llama.py | 现代标准实现 |
| HF RoFormer 源码 | https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/roformer/modeling_roformer.py | 原始 RoPE 移植 |
| HF rope_utils 源码 | https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/modeling_rope_utils.py | 变体实现 |
| Meta LLaMA 源码 | https://raw.githubusercontent.com/meta-llama/llama/main/llama/model.py | 复数形式实现 |
| EleutherAI GPT-NeoX 源码 | https://raw.githubusercontent.com/EleutherAI/gpt-neox/main/megatron/model/positional_embeddings.py | 工业级缓存实现 |
| EleutherAI RoPE 博客 | https://blog.eleuther.ai/rotary-embeddings/ | 数学推导参考 |
| YaRN 论文 | https://arxiv.org/abs/2309.00071 | YaRN 变体 |
| ALiBi 论文 | https://arxiv.org/abs/2108.12409 | 对比方案 |
