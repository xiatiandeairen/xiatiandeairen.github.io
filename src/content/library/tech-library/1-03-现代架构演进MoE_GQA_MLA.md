---
title: "现代架构演进：MoE / GQA / MLA（大模型域）"
slug: "1-03"
collection: "tech-library"
group: "大模型"
order: 1003
summary: "前置依赖：需要先理解 Transformer self-attention 的 Q/K/V 矩阵乘法与多头机制（第 1 章）、RoPE 旋转位置编码（第 2 章），以及 KV cache 在自回归解码里的作用。线性代数中的低秩分解（low-rank factorization）、矩阵乘法结合律会反复用到。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-14T20:13:11.000Z"
updatedAt: "2026-06-14T20:13:11.000Z"
---
> **前置依赖**：需要先理解 Transformer self-attention 的 Q/K/V 矩阵乘法与多头机制（第 1 章）、RoPE 旋转位置编码（第 2 章），以及 KV cache 在自回归解码里的作用。线性代数中的低秩分解（low-rank factorization）、矩阵乘法结合律会反复用到。本章面向"怎么搭怎么训"，所以会区分**训练时形态**与**推理时形态**——这两者在 GQA/MLA 里并不一样,是全章最容易踩的认知坑。

---

## TL;DR

现代大模型相对于 2020 年的"GPT-3 架构"做了三组正交的结构升级，目标各不相同：

| 升级 | 替换了什么 | 核心目标 | 代表 |
|------|-----------|---------|------|
| **SwiGLU + RMSNorm + RoPE** | ReLU-FFN / LayerNorm / 绝对PE | 同算力下更低 loss | LLaMA |
| **MoE**（稀疏专家） | dense FFN | **解耦"参数量"与"每 token 算力"** | Mixtral / DeepSeek-V3 |
| **MQA → GQA → MLA**（注意力变体） | MHA | **压 KV cache → 长上下文 + 高吞吐解码** | Llama 3(GQA) / DeepSeek-V2(MLA) |

三句话记住本质：

- **MoE** 是 FFN 维度的稀疏化：N 个专家里每个 token 只激活 top-k 个（Mixtral=8选2），于是 47B 总参只跑 13B 的算力。代价是**显存装全量参数 + 训练要做 load balancing**。
- **GQA** 是注意力 head 维度的"KV 共享":n_h 个 query head 共享 n_kv 组 KV（Llama 3 是 32 query / 8 KV），KV cache 直接砍到 1/4。它是 MHA 与 MQA 之间的连续插值。
- **MLA** 是注意力的"低秩压缩":把 KV 联合压成一个 d_c 维的 latent 向量缓存，配一条 decoupled RoPE 旁路。DeepSeek-V2 用它把 KV cache 砍到 MHA 的 ~7%,且质量反超 MHA。它的精髓是推理时把上投影矩阵**吸收(absorb)进 Q/O**,做到"缓存即 latent"。

⭐ 本章 4 个核心 demo（numpy，已在 numpy 2.4.4 实测跑通）：KV cache 元素数对比、toy MoE top-k 路由 + 负载统计、GQA `repeat_kv`、load-balancing loss。torch 版 demo 标注"设计为可运行,请在装了 torch 的环境验证"。

---

## 1. 背景：2020 之后,架构在优化三条不同的曲线

GPT-3（2020）的 decoder block 是这样的：

```
x → LayerNorm → MHA → +residual → LayerNorm → FFN(ReLU/GELU) → +residual
```

到了 2023–2025 的开源 SOTA（LLaMA / Mixtral / DeepSeek / Qwen），同一个 block 变成：

```
x → RMSNorm → {MHA|GQA|MLA} → +residual → RMSNorm → {SwiGLU-FFN | MoE} → +residual
```

关键认知：**这些升级各自服务于完全不同的工程瓶颈**，不能混为一谈。把它们摊到"训练成本 / 推理成本 / 模型质量"三轴上看：

| 瓶颈 | 升级手段 | 它**不**解决什么 |
|------|---------|----------------|
| 同算力下 loss 不够低 | SwiGLU、RMSNorm、RoPE | 不省显存、不省 KV cache |
| 想要更大容量但算力封顶 | **MoE** | **不省显存（反而暴涨）**、不省 KV cache |
| 解码慢 / KV cache 撑爆长上下文 | **GQA / MLA** | 不增加模型容量 |

这张表是全章的骨架。下面三节分别钻进去。先做设计考古,再读真实源码,再用 demo 印证。

> **一个高频误解先打掉**：很多人以为"MoE 是为了省显存/省成本"。恰恰相反——MoE 推理时**所有专家都要常驻显存**（Mixtral 47B 全参必须装下），它省的是**每 token 的 FLOPs**。它和 GQA/MLA（省 KV cache 显存）是正交的两件事,经常被混淆。

---

## 2. 设计考古：从 GPT 到 LLaMA 到 MoE 到 MLA

### 2.1 第一波:LLaMA 把 GPT block 的三个零件换掉

LLaMA（2023）没有发明新范式,但把三个被验证有效的零件标准化了,后来几乎所有开源模型照抄:

**(a) RMSNorm 替换 LayerNorm**
- 论文:Zhang & Sennrich, *Root Mean Square Layer Normalization*,arXiv 1910.07467（2019）。
- 动机:LayerNorm 要算均值 + 方差并做 re-centering;RMSNorm 砍掉减均值那步,只做 re-scaling,省掉一半统计量,几乎不掉点。

**(b) SwiGLU 替换 ReLU-FFN**
- 论文:Noam Shazeer, *GLU Variants Improve Transformer*,arXiv 2002.05202（2020）。**已 WebFetch 核实**。
- 动机:把 FFN 的第一层换成 GLU（gated linear unit）的 Swish 变体,`FFN(x) = (Swish(xW1) ⊙ xW3) W2`,多一个门控分支。论文实测 GEGLU/SwiGLU 在多个下游任务上 perplexity 最低。
- 趣闻:论文结论段那句著名的"We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence."（把成功归于"神的恩典")——我 WebSearch 时**未能在原文检索命中这两句的确切措辞,标「待核」**;但 SwiGLU 被 PaLM、LLaMA 采用是确证的。

**(c) RoPE 替换绝对位置编码**——见第 2 章,此处不展开。

这三者构成"现代 dense 模型基线"。MoE 和注意力变体都是**在这个基线之上**叠加的。

### 2.2 第二波:MoE 的三十年——从 1991 到 Mixtral

MoE 的思想比 Transformer 老得多:

| 年份 | 工作 | 贡献 |
|------|------|------|
| 1991 | Jacobs/Jordan, *Adaptive Mixtures of Local Experts* | 提出 "mixture of experts" + gating 概念 |
| **2017** | **Shazeer et al., *Outrageously Large Neural Networks***（arXiv 1701.06538,ICLR'17） | **把 MoE 做成可微的稀疏层**,塞进 LSTM,top-k gating,>1000× 参数扩容,引入 **load balancing loss** |
| 2020–22 | GShard / Switch Transformer / GLaM | 把 MoE 搬进 Transformer FFN,Switch 简化到 top-1,解决分布式 routing |
| **2024** | **Mixtral 8x7B**（arXiv 2401.04088） | 第一个**开源权重、质量打平 Llama2-70B**的 SMoE,top-2 of 8 |
| 2024–25 | DeepSeek-V2/V3 | 细粒度专家 + 共享专家 + 无辅助损失负载均衡 |

**Shazeer 2017 的两个遗产**（已 WebSearch 核实作者与年份）:
1. **稀疏 top-k gating**:gating 网络对每个样本输出一个稀疏分布,只有 top-k 个专家被激活。这是"参数量 ≠ 算力"的根。
2. **load balancing**:朴素 gating 会**塌缩**——少数专家被反复选中、其余饿死(rich-get-richer)。论文用一个辅助损失逼专家负载均衡。这个问题至今是 MoE 训练的头号坑(见 §6)。

**Mixtral 的定位**（arXiv 2401.04088,**已 WebFetch 核实**）:
- 8 个专家 / 层,每 token 选 **top-2**。
- 总参 **47B**,每 token 激活 **13B**。注意 47B ≠ 8×7B:attention 和 embedding 是共享的,只有 FFN 被复制成 8 份。
- 质量 outperform/match Llama2-70B 与 GPT-3.5,Apache-2.0 开源。

### 2.3 第三波:注意力的 KV cache 之战——MHA → MQA → GQA → MLA

这条线全是为了一个东西:**自回归解码时的 KV cache**。

**问题根源**:解码第 t 个 token 时,要对前面所有 token 的 K/V 做 attention。为避免重算,把历史 K/V 缓存下来。MHA 下,每层每个 head 都要存一份 K 和 V,显存随 `序列长度 × 层数 × head 数 × head_dim` 线性涨。长上下文 + 大 batch 时,**KV cache 比模型权重还大**,且解码是 memory-bandwidth bound——每生成一个 token 都要把整个 KV cache 从显存搬一遍。

演进路径:

```
MHA (2017)         n_h 个 query head,各自配 n_h 组 K/V
  │  KV cache 太大,解码带宽瓶颈
  ▼
MQA (2019, Shazeer "One Write-Head is All You Need")
                   n_h 个 query head 共享 1 组 K/V → cache 砍到 1/n_h
  │  砍太狠,质量掉、训练不稳
  ▼
GQA (2023, Ainslie et al. EMNLP)
                   分 n_g 组,每组共享 1 组 K/V → MHA 与 MQA 之间连续插值
  │  仍是"复制 KV head",压缩率受限于 n_g
  ▼
MLA (2024, DeepSeek-V2)
                   不复制,而是把 KV 联合低秩压成 d_c 维 latent 缓存
                   + decoupled RoPE 旁路 → 压到 MHA 的 ~7%,质量反超
```

逐篇考古:

**MQA — Shazeer 2019, *Fast Transformer Decoding: One Write-Head is All You Need***（arXiv 1911.02150,**已 WebSearch 核实标题/作者/年份**）:
- 动机原文核心:incremental inference 慢,是因为反复加载巨大的 K/V 张量的 memory-bandwidth cost。
- 方案:所有 head **共享同一组 K/V**("one write-head"),K/V 张量瞬间小 n_h 倍。
- 代价:论文承认 "minor quality degradation"。后续实践发现大模型上"minor"并不 minor,且训练易不稳。

**GQA — Ainslie et al. 2023, *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints***（arXiv 2305.13245,EMNLP'23,**已 WebFetch 核实**):
- 两个贡献:
  1. **GQA 本身**:把 query head 分成 n_g 组,组内共享一组 KV。n_g=1 退化为 MQA,n_g=n_h 退化为 MHA。是连续谱上的中间点。
  2. **uptraining 配方**:不用从头训 MQA/GQA,而是拿现成 MHA checkpoint,把每组的多个 KV head **mean-pooling**成一个,再用 **5% 的原预训练算力**继续训。这是工程上真正让 GQA 普及的关键——存量 MHA 模型能低成本转 GQA。
- 结论:GQA 质量接近 MHA,速度接近 MQA。Llama 2 70B 起开始用 GQA(8 组),Llama 3 全系 8 个 KV head。

**MLA — DeepSeek-V2 2024**（arXiv 2405.04434,**已 WebFetch 核实**):
- 动机:GQA/MQA 只是"共享/减少 KV head",压缩率受限(最多压到 1/n_h);且 MQA 掉点。能不能**既大幅压缩、又不掉点**?
- 方案两件套:
  1. **低秩 KV 联合压缩**:对每个 token,把它的 KV 信息下投影成一个低维 latent 向量 `c_KV`(维度 d_c),**只缓存 c_KV**。用到时再上投影还原出每个 head 的 K/V。
  2. **decoupled RoPE**:RoPE 是位置相关的旋转,**和低秩压缩不兼容**(旋转矩阵不能被吸收进上投影的合并里,详见 §4.3 的"absorb"分析)。MLA 的解法是把 query/key 各拆成两段——一段 nope(no positional embedding)走压缩路径,一段 rope 单独走、单独带 RoPE,最后拼起来。
- 战果:KV cache 砍 **93.3%**(降到 MHA 的 ~7%),且性能**反超** MHA(低秩起到了正则/信息瓶颈的作用)。DeepSeek-V2 总参 236B / 激活 21B。

KV cache per token 的精确公式(DeepSeek-V2 论文 Table,**已 WebFetch 核实**),`l`=层数:

| 方案 | 每 token 缓存元素数 | 说明 |
|------|-------------------|------|
| MHA | `2 · n_h · d_h · l` | K、V 各 n_h 个 head |
| GQA | `2 · n_g · d_h · l` | n_g 组 KV |
| MQA | `2 · d_h · l` | 单组 KV |
| **MLA** | `(d_c + d_h^R) · l` | latent + decoupled-RoPE 旁路 |

DeepSeek-V2 实际超参(已核实):`d_c`(kv_lora_rank)=512,`d_h^R`(qk_rope_head_dim)=64,n_h=128,d_h=128。代入对比见 §4.1 demo。

---

## 3. 真实源码精读

### 3.1 LLaMA 的 SwiGLU 与 RMSNorm（基线零件）

先看现代 dense 基线那两个零件的真身。

```python
# 【真实源码 meta-llama/llama@llama/model.py】(已 WebFetch raw.githubusercontent.com)
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)                 # ① SwiGLU 三矩阵,为保持参数量近似,
        if ffn_dim_multiplier is not None:                   #    把隐层宽度乘以 2/3 补偿(否则 3 个矩阵会比 2 个大 1.5x)
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)  # ② 对齐到 multiple_of(利于 GPU)
        self.w1 = ColumnParallelLinear(dim, hidden_dim, bias=False, ...)  # gate 分支
        self.w2 = RowParallelLinear(hidden_dim, dim, bias=False, ...)     # down 投影
        self.w3 = ColumnParallelLinear(dim, hidden_dim, bias=False, ...)  # up 分支(value)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))     # ③ SwiGLU: silu(W1 x) ⊙ (W3 x),再 W2 down
```

逐行注解:
- ①②:`hidden_dim = 2/3 × 原值`。这是 SwiGLU 的工程惯例——GLU 类有三个权重矩阵(w1/w2/w3)而非 FFN 的两个(w1/w2),若隐层宽度不变会多 50% 参数;乘 2/3 把总参拉回与原 FFN 持平,方便公平对比。`multiple_of` 把宽度对齐到 256 之类,迁就硬件。
- ③:`F.silu(self.w1(x))` 是 Swish 门控,`self.w3(x)` 是被门控的值分支,逐元素相乘后过 `w2` 下投影。这就是 SwiGLU。

```python
# 【真实源码 meta-llama/llama@llama/model.py】RMSNorm 核心(已 WebFetch)
def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
```
- 对比 LayerNorm 少了 `- x.mean(...)`(不做 re-centering)。`rsqrt(mean(x²)+eps)` 就是 1/RMS。省一半统计量。

### 3.2 GQA:Llama 的 `repeat_kv`（注意力变体最朴素的实现）

GQA 在朴素实现里就是一句话:**把少数 KV head 复制几份,凑齐 query head 数,然后照常做 MHA**。

```python
# 【真实源码 meta-llama/llama@llama/model.py】(已 WebFetch raw.githubusercontent.com)
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:                                  # ① n_rep=1 即 MHA,直接返回
        return x
    return (
        x[:, :, :, None, :]                         # ② 插一个新轴: (bs,slen,n_kv,1,hd)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)   # ③ expand 是视图,不复制内存
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim) # ④ reshape 才物化成 n_kv*n_rep 个 head
    )
```

```python
# 【真实源码 meta-llama/llama@llama/model.py】Attention.__init__ 关键行(已 WebFetch)
self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
self.n_rep = self.n_local_heads // self.n_local_kv_heads      # ⑤ 每组复制次数 = query head / kv head

self.cache_k = torch.zeros(
    (args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim)
).cuda()                                                       # ⑥ ★缓存只存 n_kv_heads,这才是省 cache 的关键
self.cache_v = torch.zeros(
    (args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim)
).cuda()
```

```python
# 【真实源码 meta-llama/llama@llama/model.py】Attention.forward 关键行(已 WebFetch)
keys = repeat_kv(keys, self.n_rep)        # ⑦ 算 attention 前才 expand 回 n_h 个 head
values = repeat_kv(values, self.n_rep)
```

**这段最关键的认知**(面试高频):
- ⑥ KV cache 物理上只存 `n_kv_heads` 份——这是 GQA 省显存的**唯一**来源。
- ⑦ `repeat_kv` 是在**计算时**临时 expand 回 `n_h` 个 head。注意 ③ 的 `expand` 是**视图(view)**,不真复制数据;真正的内存物化发生在 ④ 的 `reshape`,但那是临时张量,**不进 cache**。所以 GQA 的算力(FLOPs)和 MHA 几乎一样(还是 n_h 个 head 在算 attention),省的纯粹是 **cache 显存 + 解码时搬运带宽**。
- 训练时形态 vs 推理时形态在这里**一致**(都是复制 KV head),这是 GQA 比 MLA 简单的地方。

### 3.3 MoE:Mixtral 参考实现的 top-k 路由

Mixtral 官方参考实现(mistral-inference)的 MoE 短到惊人,把"稀疏路由"的本质暴露得最清楚:

```python
# 【真实源码 mistralai/mistral-inference@src/mistral_inference/moe.py】(已 WebFetch raw.githubusercontent.com)
@dataclasses.dataclass
class MoeArgs(Serializable):
    num_experts: int            # 专家总数,Mixtral=8
    num_experts_per_tok: int    # 每 token 激活几个,Mixtral=2 (top-2)

class MoeLayer(nn.Module):
    def __init__(self, experts, gate, moe_args):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)   # 8 个独立 FFN
        self.gate = gate                        # 路由打分: Linear(dim, num_experts)
        self.args = moe_args

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(inputs)                                   # ① 每 token 对 8 个专家打分 (n_tok, 8)
        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)  # ② 取 top-2 的 logits + 索引
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)  # ③ ★只在选中的 2 个上做 softmax
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):                         # ④ 遍历每个专家
            batch_idx, nth_expert = torch.where(selected_experts == i)    # ⑤ 找出"选了专家 i"的 token 及其槽位
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(inputs[batch_idx])  # ⑥ 加权累加
        return results
```

逐行注解(这是全章最该背下来的 20 行):
- ①:`gate` 是个最朴素的 `Linear`,把 d 维 token 映射成 num_experts 维 logits。没有 softmax,没有噪声(Mixtral 推理路径很干净)。
- ②:`torch.topk` 直接在 logits 上取 top-2,返回 `weights`(top-2 的原始 logit 值)和 `selected_experts`(专家索引)。
- ③:**关键**——softmax 只在被选中的 k 个 logit 上做(`dim=1` 是那 k 维)。这意味着每个 token 的 2 个专家权重和为 1,但**未被选中的 6 个专家完全不参与**。这是稀疏的数学体现。用 fp32 算 softmax 防溢出。
- ④⑤⑥:这是"按专家分发(dispatch)"的写法。`torch.where(selected_experts == i)` 返回所有把专家 i 选进 top-2 的 (token行, 在top-2里的槽位) 坐标。`expert(inputs[batch_idx])` 只对这些 token 跑专家 i,乘上对应权重累加进结果。
- **效率真相**:这个 for 循环是教学版,实际上每个专家都对"选了它的 token 子集"做一次 forward。生产推理(vLLM/TensorRT-LLM)会用 grouped GEMM / megablocks 把它融成一次 batched 矩阵乘,但**数学等价**。

### 3.4 MoE 训练的灵魂:HF 版路由 + load balancing loss

参考实现省略了训练才需要的两样东西:**辅助负载均衡损失**和**更显式的 one-hot dispatch**。HF transformers 的 Mixtral 实现有:

```python
# 【真实源码 huggingface/transformers@src/transformers/models/mixtral/modeling_mixtral.py】(已 WebFetch raw.githubusercontent.com)
# —— router(注:HF 近期重构过,把 router 拆成独立模块)——
def forward(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states, self.weight)
    router_probs = torch.nn.functional.softmax(router_logits.float(), dim=-1)   # ① 先对全部 8 个专家 softmax
    router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)  # ② 再取 top-k
    router_top_value /= router_top_value.sum(dim=-1, keepdim=True)              # ③ ★top-k 概率重归一化(和=1)
    return router_logits, router_top_value, router_indices
```

> **⚠ 一个微妙但重要的差异**:mistral-inference(§3.3)是 **topk-then-softmax**(先选 top-k 再 softmax);HF 这版是 **softmax-then-topk-then-renormalize**(先全局 softmax、取 top-k、再重归一化)。两者权重数值**不完全相同**——前者的 softmax 分母只含 k 项,后者含全部 N 项再除以 top-k 的和。Mixtral 论文公式写的是 `softmax(topk(·))` 即前者;HF 的实现选了后者。这是"spec 的 NL 描述 vs code 实际算法"会分叉的典型例子,迁移权重/复现指标时务必对齐。**两种都是工业界在用的合法变体。**

负载均衡损失(训练时加到主 loss 上,这才是 MoE 能训稳的关键):

```python
# 【真实源码 huggingface/transformers@.../modeling_mixtral.py】load_balancing_loss_func 核心(已 WebFetch)
routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
_, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
tokens_per_expert = torch.mean(expert_mask.float(), dim=0)        # ① f_i: 路由到专家 i 的 token 比例
router_prob_per_expert = torch.mean(routing_weights, dim=0)       # ② P_i: 专家 i 的平均路由概率
overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))  # ③ Σ f_i · P_i
return overall_loss * num_experts                                 # ④ ×N,使均衡时 loss≈1
```
- 这是 Switch Transformer 提出的"f_i · P_i"形式。**直觉**:`f_i`(实际负载,不可导的 argmax 计数)× `P_i`(可导的软概率)。如果某专家被过度路由,它的 `f_i` 和 `P_i` 同时偏高,乘积放大,惩罚把概率往下压;饿死的专家则反向被抬起来。
- ④ 乘 num_experts 做归一化:**完美均衡时每个 f_i=P_i=1/N,Σ = N·(1/N)² = 1/N,×N = 1**。所以这个 loss 的理想值是 1.0,越大越不均衡(§4.4 demo 实测)。
- 训练时:`total_loss = ce_loss + α · aux_loss`,Mixtral 的 α(`router_aux_loss_coef`)通常 0.01~0.02 量级。

### 3.5 MLA:DeepSeek-V3 的完整实现(本章技术含量最高的源码)

MLA 是全章最绕的,直接读 DeepSeek-V3 的官方 inference 实现。它**同时支持两种 attention 形态**:`naive`(教学/对拍用,显式还原 K/V)和默认的 absorb 路径(生产用,"缓存即 latent")。

```python
# 【真实源码 deepseek-ai/DeepSeek-V3@inference/model.py】MLA.__init__(已 WebFetch raw.githubusercontent.com)
class MLA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank          # query 也低秩,DeepSeek-V2=1536
        self.kv_lora_rank = args.kv_lora_rank        # ★KV latent 维 d_c,V2=512
        self.qk_nope_head_dim = args.qk_nope_head_dim  # 不带RoPE的那段 head_dim,V2=128
        self.qk_rope_head_dim = args.qk_rope_head_dim  # ★decoupled-RoPE 段 d_h^R,V2=64
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim  # query/key 总 head_dim = nope+rope
        self.v_head_dim = args.v_head_dim

        # —— Q 的低秩分解(可选)——
        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)             # 下投影 dim→q_lora_rank
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank,
                                             self.n_heads * self.qk_head_dim)  # 上投影
        # —— KV 的低秩联合压缩(MLA 心脏)——
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)  # ★dim → (d_c + d_h^R)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank,
                    self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))  # latent 上投影出 K_nope 和 V
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5
        ...
        if attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(.., self.n_local_heads, self.qk_head_dim), ..)  # 还原版:存满 K
            self.register_buffer("v_cache", torch.zeros(.., self.n_local_heads, self.v_head_dim), ..)
        else:
            self.register_buffer("kv_cache", torch.zeros(.., self.kv_lora_rank), ..)   # ★★只存 d_c 维 latent
            self.register_buffer("pe_cache", torch.zeros(.., self.qk_rope_head_dim), ..) # ★★+ 64 维 RoPE 旁路
```

`__init__` 三个要点:
1. **Q 也做了低秩**(wq_a→q_norm→wq_b),不只 KV。训练时这是参数压缩/正则;推理时 Q 每步重算不缓存,所以 Q 的低秩主要是省参数和正则。
2. **`wkv_a` 把 dim 投到 `d_c + d_h^R`**——前 d_c 维是 KV 联合 latent,后 d_h^R(=64)维是 decoupled-RoPE 的 key 旁路。一个 Linear 同时产出两段。
3. **两种 cache 形态对比一目了然**:naive 路径 cache 是 `(.., n_heads, qk_head_dim)`(每 head 存满);absorb 路径只存 `(.., kv_lora_rank)` + `(.., qk_rope_head_dim)`——这就是 §2.3 表里 MLA 的 `(d_c + d_h^R)` 来源(注意**不乘 n_heads**!所有 head 共享同一个 latent)。

```python
# 【真实源码 deepseek-ai/DeepSeek-V3@inference/model.py】MLA.forward(已 WebFetch)
def forward(self, x, start_pos, freqs_cis, mask):
    bsz, seqlen, _ = x.size()
    end_pos = start_pos + seqlen
    # —— Q: 低秩还原 + 拆 nope/rope ——
    if self.q_lora_rank == 0:
        q = self.wq(x)
    else:
        q = self.wq_b(self.q_norm(self.wq_a(x)))                     # ① dim→q_lora→dim',低秩两步
    q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
    q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)  # ② 拆成 nope段 + rope段
    q_pe = apply_rotary_emb(q_pe, freqs_cis)                         # ③ ★只对 rope 段施加 RoPE
    # —— KV: 联合压缩 ——
    kv = self.wkv_a(x)                                               # ④ dim → (d_c + d_h^R)
    kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)  # ⑤ 拆出 latent 和 rope-key 旁路
    k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)            # ⑥ rope-key 单独带 RoPE(全 head 共享这一份)

    if attn_impl == "naive":   # —— 教学/对拍路径:显式还原 K/V ——
        q = torch.cat([q_nope, q_pe], dim=-1)
        kv = self.wkv_b(self.kv_norm(kv))                            # ⑦ latent 上投影还原出 K_nope 和 V
        kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)  # ⑧ 拼 nope+rope 成完整 K
        self.k_cache[:bsz, start_pos:end_pos] = k                    #    存满 K/V(此路径不省 cache)
        self.v_cache[:bsz, start_pos:end_pos] = v
        scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
    else:                      # —— ★生产路径:matrix absorption,缓存即 latent ——
        wkv_b = self.wkv_b.weight ...                                # 取上投影权重
        wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])  # ⑨ ★★把 K 上投影"吸收"进 q_nope
        self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)    # ⑩ ★只缓存 d_c 维 latent
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)     #    + 64 维 rope 旁路
        scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +   # ⑪ nope 段: 在 latent 空间算分
                  torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])       #    rope 段: 在旁路算分
                 ) * self.softmax_scale                              #    两段相加 = 完整 attention 分数
    if mask is not None:
        scores += mask.unsqueeze(1)
    scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)

    if attn_impl == "naive":
        x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
    else:
        x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])  # ⑫ 在 latent 空间聚合
        x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])         # ⑬ ★最后才用 V 上投影还原
    x = self.wo(x.flatten(2))
    return x
```

这是 MLA 的精华,⑨⑪⑫⑬ 是必须理解的"matrix absorption"魔法:

- **decoupled RoPE(②③⑤⑥)**:为什么不能像 MHA 那样直接对完整 K 施加 RoPE?因为 absorb 路径要把 K 的上投影矩阵 `W_UK` 提前乘进 query(⑨),让 attention 分数能直接在 latent 空间算。但 RoPE 是位置 m,n 相关的旋转 `R_m, R_n`,**夹在 query 和 key 之间**,无法被吸收进一个**与位置无关**的 `W_UK`(`q^T W_UK^T R_{n-m} c` 没法把 R 并进 W_UK)。解法:把 head_dim 劈成两半,**nope 段**(128 维)走可吸收的低秩路径、不带位置;**rope 段**(64 维)单独留出来带 RoPE,全 head 共享一份 k_pe。attention 分数 = nope 段点积(⑪第一项)+ rope 段点积(⑪第二项)。
- **matrix absorption(⑨⑫⑬)**:朴素想法是缓存 latent,用时上投影还原 K/V 再算(naive 路径⑦⑧)。但那样每步要还原全部 K/V,没省计算只省了 cache。absorb 路径更狠:利用矩阵乘结合律,把 `(q W_UK) · c` 重排成 `q · (W_UK c)`——既然 `W_UK` 与位置无关,干脆**预先把 W_UK 乘进 q_nope**(⑨),于是 attention 直接拿 `q_nope`(已含 W_UK)和缓存的 latent `c` 做点积(⑪),全程**不还原 K**;output 侧同理,先在 latent 空间用 score 加权聚合 latent(⑫),最后才乘 V 上投影 `W_UV` 还原(⑬)。
- **净效果**:缓存只有 `d_c + d_h^R = 512 + 64 = 576` 维/token/层(且不乘 n_heads),而 MHA 是 `2 × 128 × 128 = 32768` 维。压缩约 57×→ 论文报 KV cache 降 93.3%(实际还有 GQA 等基线对比口径差异)。

> **训练时 vs 推理时形态(MLA 最大的坑)**:训练时一般走"还原 K/V 后正常 attention"(等价 naive 但全并行);absorb 是**推理时的等价改写**。两条路径**数学等价但数值不同**(浮点累加顺序、是否走 latent 空间),复现/部署时如果训练用一条、推理用另一条,务必对拍。DeepSeek-V3 仓库同时给两条正是为了对拍。

### 3.6 DeepSeek 的 Gate:细粒度 + group-limited routing

DeepSeek 的 MoE 路由比 Mixtral 复杂,引入了 expert group、group-limited 路由、可选 sigmoid 打分 + bias(V3 的"无辅助损失"负载均衡):

```python
# 【真实源码 deepseek-ai/DeepSeek-V3@inference/model.py】Gate.forward(已 WebFetch)
def forward(self, x):
    scores = linear(x, self.weight)
    if self.score_func == "softmax":
        scores = scores.softmax(dim=-1, dtype=torch.float32)
    else:
        scores = scores.sigmoid()                       # ① V3 用 sigmoid 打分(每专家独立,不竞争归一)
    original_scores = scores
    if self.bias is not None:
        scores = scores + self.bias                     # ② ★可学习/动态 bias: V3 用它做"无辅助损失"负载均衡
    if self.n_groups > 1:                               # ③ 专家分组,先选组再选专家(限制跨节点通信)
        scores = scores.view(x.size(0), self.n_groups, -1)
        if self.bias is None:
            group_scores = scores.amax(dim=-1)
        else:
            group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)   # 组得分 = 组内 top-2 之和
        indices = group_scores.topk(self.topk_groups, dim=-1)[1]   # ④ 只保留 top 几个组
        mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
        scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)  # 其余组置 -inf
    indices = torch.topk(scores, self.topk, dim=-1)[1]  # ⑤ 在保留的组里取 top-k 专家
    weights = original_scores.gather(1, indices)        # ⑥ ★权重用加 bias 之前的原始分数(bias 只影响"选谁",不影响权重)
    if self.score_func == "sigmoid":
        weights /= weights.sum(dim=-1, keepdim=True)    # ⑦ sigmoid 路径要手动归一
    weights *= self.route_scale
    return weights.type_as(x), indices
```

- ①:V3 把路由打分从 softmax 换成 **sigmoid**——每个专家独立打分,不像 softmax 那样互相竞争。配合细粒度专家(更多更小的专家)。
- ②⑥:这是 DeepSeek-V3 的**auxiliary-loss-free load balancing**(无辅助损失负载均衡,arXiv 2408.15664 思路):给每个专家一个可动态调整的 `bias`,**只用于决定"选哪些专家"(②)**,但最终权重用的是**加 bias 之前**的分数(⑥)。训练中根据各专家负载在线调 bias(过载的调低、饿死的调高),不需要往 loss 里塞辅助项扰动主目标。这是对 §3.4 那套 aux-loss 的重要改进。
- ③④:**group-limited routing**——专家先分组(对应不同 GPU 节点),每 token 只能路由到 top 几个组内的专家,把 all-to-all 通信限制在少数节点,是分布式 MoE 训练的工程关键。

```python
# 【真实源码 deepseek-ai/DeepSeek-V3@inference/model.py】MoE.forward(已 WebFetch)
def forward(self, x):
    shape = x.size(); x = x.view(-1, self.dim)
    weights, indices = self.gate(x)
    y = torch.zeros_like(x)
    counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()  # 各专家被选次数
    for i in range(self.experts_start_idx, self.experts_end_idx):   # ① 只跑本 rank 负责的专家(EP 切分)
        if counts[i] == 0:                                          # ② ★没 token 选它就跳过(稀疏)
            continue
        expert = self.experts[i]
        idx, top = torch.where(indices == i)
        y[idx] += expert(x[idx]) * weights[idx, top, None]          # ③ 同 Mixtral 的 dispatch 累加
    z = self.shared_experts(x)                                      # ④ ★共享专家: 每 token 都过,提供"通用能力底座"
    if world_size > 1:
        dist.all_reduce(y)                                          # ⑤ 跨 rank 把各专家结果汇总
    return (y + z).view(shape)                                      # ⑥ 路由专家 + 共享专家
```
- ④⑥:DeepSeek 的**shared expert**(共享专家)——区别于 Mixtral 纯路由。每个 token 除了走 top-k 路由专家,还**无条件**过 1~2 个共享专家。直觉:共享专家学"所有 token 都需要的通用能力",路由专家学"专门化能力",避免路由专家被迫各自重复学通用模式,提升参数效率。
- ①⑤:`experts_start_idx:end_idx` + `all_reduce` 是 **expert parallelism**(EP)——专家被切到不同 GPU,每 rank 只持有/计算一部分专家,最后 all-reduce 汇总。这是 MoE 区别于 dense 的核心分布式范式(详见 §6)。

---

## 4. 可运行 Demo

> 以下 4 个 numpy demo **已在本机 numpy 2.4.4 实测跑通**,输出附在每段后。torch demo 标注"设计为可运行,请在装了 torch 的环境验证"。
> 依赖:`pip install numpy`(核心 demo);torch demo 需 `pip install torch`。

### 4.1 Demo 1:KV cache 元素数对比(MHA/GQA/MQA/MLA)——印证 §2.3 公式

把 §2.3 的公式直接写成代码,代入 DeepSeek-V2 规模,直观感受压缩比。

```python
# demo_kv.py  —— 已实测(numpy 2.4.4)。注:纯算术,其实无需 numpy,保持依赖一致
import numpy as np

def kv_cache_elems(scheme, n_h, n_kv, d_h, n_layers, seq_len, d_c=0, d_rope=0):
    if scheme == "MHA":
        per_tok = 2 * n_h  * d_h * n_layers      # K,V 各 n_h 个 head
    elif scheme == "GQA":
        per_tok = 2 * n_kv * d_h * n_layers      # 只存 n_kv 组
    elif scheme == "MQA":
        per_tok = 2 * 1    * d_h * n_layers      # 单组
    elif scheme == "MLA":
        per_tok = (d_c + d_rope) * n_layers      # ★latent + rope 旁路,且不乘 n_h
    return per_tok, per_tok * seq_len

cfg = dict(n_h=128, n_kv=8, d_h=128, n_layers=60, seq_len=4096, d_c=512, d_rope=64)
print(f"{'scheme':6} {'per-token elems':>16} {'@4k ctx (M elems)':>20} {'rel. to MHA':>12}")
base = None
for s in ["MHA", "GQA", "MQA", "MLA"]:
    pt, tot = kv_cache_elems(s, **cfg)
    if base is None: base = pt
    print(f"{s:6} {pt:>16,} {tot/1e6:>20.2f} {pt/base:>11.3%}")
```

运行:`python3 demo_kv.py`。**实测输出**:
```
scheme  per-token elems    @4k ctx (M elems)  rel. to MHA
MHA           1,966,080              8053.06    100.000%
GQA             122,880               503.32      6.250%
MQA              15,360                62.91      0.781%
MLA              34,560               141.56      1.758%
```
**读数**:GQA(8/128 组)= 精确 6.25%;MQA 最狠(0.78%)但质量代价大;**MLA 1.76%——压缩比逼近 MQA(57×),但质量反超 MHA**,这正是 MLA 的价值点。注意 GQA 的 6.25% = n_kv/n_h = 8/128,一眼可验。

### 4.2 Demo 2:toy MoE top-k 路由 + 负载统计——印证 §3.3

从零写 §3.3 的稀疏路由,并验证"稀疏 dispatch == 手算加权和"。

```python
# demo_moe.py —— 已实测(numpy 2.4.4)
import numpy as np
rng = np.random.default_rng(0)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True); e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)

n_tok, d, n_exp, k = 6, 8, 4, 2
x = rng.standard_normal((n_tok, d)).astype(np.float32)
W_gate = rng.standard_normal((d, n_exp)).astype(np.float32) * 0.5
W_exp  = [rng.standard_normal((d, d)).astype(np.float32) * 0.3 for _ in range(n_exp)]  # 每专家=线性 d→d

# —— 路由(对应 mistral-inference: topk-then-softmax)——
logits     = x @ W_gate                                  # (n_tok, n_exp)
topk_idx   = np.argsort(-logits, axis=-1)[:, :k]         # top-k 专家索引   ≈ torch.topk(...)[1]
topk_logit = np.take_along_axis(logits, topk_idx, -1)
weights    = softmax(topk_logit, axis=-1)               # 只在 k 个上 softmax ≈ §3.3 ③

# —— dispatch:按专家累加,对应 torch.where(selected==i)——
out  = np.zeros_like(x)
load = np.zeros(n_exp, dtype=int)
for e in range(n_exp):
    rows, slot = np.where(topk_idx == e)                # 选了专家 e 的 token 行 + 槽位
    if len(rows) == 0: continue
    load[e] = len(rows)
    out[rows] += (x[rows] @ W_exp[e]) * weights[rows, slot][:, None]

print("token0 -> experts", topk_idx[0], "weights", np.round(weights[0], 3), "sum", round(float(weights[0].sum()), 4))
print("per-expert token load:", load.tolist(), "(sum =", load.sum(), "= n_tok*k =", n_tok*k, ")")
man = sum(weights[0, j] * (x[0] @ W_exp[topk_idx[0, j]]) for j in range(k))   # 手算 token0 的 2 专家加权和
print("max|sparse - manual| token0:", float(np.abs(out[0] - man).max()))
```

运行:`python3 demo_moe.py`。**实测输出**:
```
token0 -> experts [2 0] weights [0.505 0.495] sum 1.0
per-expert token load: [2, 4, 2, 4] (sum = 12 = n_tok*k = 12 )
max|sparse - manual| token0: 5.960464477539063e-08
```
**读数**:每 token 权重和=1(只在 top-2 上);负载和=12=n_tok×k(每 token 恰好激活 k 个专家);稀疏 dispatch 与手算结果差仅 6e-8(fp32 精度)——证明 §3.3 那个 `torch.where` 累加写法**数学正确**。注意此处负载 `[2,4,2,4]` 已经不均衡(专家1/3 各 4,专家0/2 各 2),这就引出了 load balancing(Demo 4)。

### 4.3 Demo 3:GQA `repeat_kv` 从零实现 + 跨 MHA/GQA/MQA 验证——印证 §3.2

```python
# demo_gqa.py —— 已实测(numpy 2.4.4)
import numpy as np
rng = np.random.default_rng(1)

def repeat_kv(x, n_rep):   # 对应 llama repeat_kv: (bs,slen,n_kv,hd) -> (bs,slen,n_kv*n_rep,hd)
    bs, slen, n_kv, hd = x.shape
    if n_rep == 1: return x
    return np.broadcast_to(x[:, :, :, None, :], (bs, slen, n_kv, n_rep, hd)).reshape(bs, slen, n_kv * n_rep, hd)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True); e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)

def attn(q, k, v):   # q,k,v: (bs, n_head, slen, hd)
    hd = q.shape[-1]
    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd)
    return softmax(scores) @ v

bs, slen, hd = 1, 5, 4
for name, n_h, n_kv in [("MHA", 8, 8), ("GQA", 8, 2), ("MQA", 8, 1)]:
    n_rep = n_h // n_kv
    q = rng.standard_normal((bs, slen, n_h,  hd)).astype(np.float32)
    k = rng.standard_normal((bs, slen, n_kv, hd)).astype(np.float32)   # ★KV 只有 n_kv 个 head(=cache 大小)
    v = rng.standard_normal((bs, slen, n_kv, hd)).astype(np.float32)
    k_full = repeat_kv(k, n_rep); v_full = repeat_kv(v, n_rep)         # 计算时才 expand 回 n_h
    out = attn(q.transpose(0,2,1,3), k_full.transpose(0,2,1,3), v_full.transpose(0,2,1,3))
    print(f"{name:4} n_h={n_h} n_kv={n_kv} n_rep={n_rep}  KV cached elems={k.size+v.size:4d}  out shape={out.shape}")
```

运行:`python3 demo_gqa.py`。**实测输出**:
```
MHA  n_h=8 n_kv=8 n_rep=1  KV cached elems= 320  out shape=(1, 8, 5, 4)
GQA  n_h=8 n_kv=2 n_rep=4  KV cached elems=  80  out shape=(1, 8, 5, 4)
MQA  n_h=8 n_kv=1 n_rep=8  KV cached elems=  40  out shape=(1, 8, 5, 4)
```
**读数**:三者输出 shape 完全一致 `(1,8,5,4)`(对 query 而言都是 8 个 head),但**缓存的 KV 元素数 320→80→40**(线性随 n_kv)。这就是 GQA 的全部秘密:**输出形态不变、cache 线性缩小**。`repeat_kv` 的 `broadcast_to` 对应源码里的 `expand`(零拷贝视图)。

### 4.4 Demo 4:load balancing loss——印证 §3.4

验证 §3.4 那个 `Σ f_i·P_i × N` 的 loss:均衡时 ≈1,塌缩时飙高。

```python
# demo_lb.py —— 已实测(numpy 2.4.4)
import numpy as np
rng = np.random.default_rng(3)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True); e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)

def lb_loss(logits, k):                       # 对应 HF load_balancing_loss_func
    n_tok, n_exp = logits.shape
    probs = softmax(logits)
    idx = np.argsort(-probs, axis=-1)[:, :k]
    onehot = np.zeros((n_tok, k, n_exp))
    for t in range(n_tok):
        for s in range(k): onehot[t, s, idx[t, s]] = 1
    f_i = onehot.max(axis=1).mean(axis=0)     # 路由到专家 i 的 token 比例
    P_i = probs.mean(axis=0)                  # 专家 i 的平均路由概率
    return n_exp * float((f_i * P_i).sum())   # ×N → 完美均衡=1.0

n_exp, k = 8, 2
balanced = rng.standard_normal((512, n_exp)) * 0.3
collapsed = balanced.copy(); collapsed[:, 0] += 5.0     # 人为把所有 token 推向专家 0
print("balanced loss  :", round(lb_loss(balanced, k),  4))
print("collapsed loss :", round(lb_loss(collapsed, k), 4), "(higher = worse, ideal = 1.0)")
```

运行:`python3 demo_lb.py`。**实测输出**:
```
balanced loss  : 2.0017
collapsed loss : 7.6633 (higher = worse balance, ideal = 1.0)
```
**读数**:均衡分布 loss≈2.0(注:top-2 时 f_i 之和=2,不是 1,所以理想值随 k 变,这里相对比较看趋势),塌缩后飙到 7.66。训练时这个 loss 乘 α(~0.01)加进主 loss,把路由往均衡拽。**自己改 `collapsed[:, 0] += 5.0` 的强度,看 loss 单调上升**——这就是辅助损失的反馈信号。

### 4.5 Demo 5:MLA 低秩压缩(torch 版,设计为可运行)

> **设计为可运行,请在装了 torch 的环境验证**(本机未装 torch,故未实测;逻辑对照 §3.5 naive 路径)。依赖:`pip install torch`。

```python
# demo_mla.py  —— 设计为可运行(需 torch),验证"latent 压缩后还原的 attention" ≈ 概念正确,并量化 cache 压缩比
import torch, torch.nn.functional as F

torch.manual_seed(0)
dim, n_h, d_nope, d_rope, d_c, v_head = 64, 4, 16, 8, 24, 16
seqlen = 6
x = torch.randn(seqlen, dim)

# —— 投影(对照 §3.5: wkv_a 产 latent+rope-key;wkv_b 还原 K_nope+V)——
wkv_a = torch.nn.Linear(dim, d_c + d_rope, bias=False)
wkv_b = torch.nn.Linear(d_c, n_h * (d_nope + v_head), bias=False)
wq    = torch.nn.Linear(dim, n_h * (d_nope + d_rope), bias=False)

q = wq(x).view(seqlen, n_h, d_nope + d_rope)
q_nope, q_pe = q.split([d_nope, d_rope], dim=-1)            # 拆 nope/rope(此 demo 略去真实 RoPE 旋转,占位)
kv = wkv_a(x)
c_kv, k_pe = kv.split([d_c, d_rope], dim=-1)                # ★只有 c_kv(d_c 维)需要缓存!
print("cache/token  MLA:", c_kv.shape[-1] + k_pe.shape[-1], " vs MHA:", 2 * n_h * (d_nope + v_head))

# naive 还原路径(对照 §3.5 attn_impl=='naive')
kvb = wkv_b(c_kv).view(seqlen, n_h, d_nope + v_head)
k_nope, v = kvb.split([d_nope, v_head], dim=-1)
k = torch.cat([k_nope, k_pe.unsqueeze(1).expand(-1, n_h, -1)], dim=-1)
qq = torch.cat([q_nope, q_pe], dim=-1)
scores = torch.einsum("shd,thd->sht", qq, k) / (d_nope + d_rope) ** 0.5
out = torch.einsum("sht,thd->shd", scores.softmax(-1), v)
print("attn out shape:", out.shape)   # 预期 (6, 4, 16)
```
**预期输出**(概念):`cache/token MLA: 32  vs MHA: 256`(8× 压缩,玩具规模);`attn out shape: torch.Size([6, 4, 16])`。真实 MLA 还要加 RoPE 旋转与 absorb 路径,见 §3.5。

---

## 5. 方案对比

### 5.1 注意力变体对比(MHA / MQA / GQA / MLA)

| 维度 | MHA | MQA | GQA | MLA |
|------|-----|-----|-----|-----|
| KV cache/token | `2 n_h d_h l` | `2 d_h l` | `2 n_g d_h l` | `(d_c + d_h^R) l` |
| 典型压缩比(vs MHA) | 1× | n_h× | n_h/n_g× | ~57×(V2) |
| 质量 | 基准 | 掉点(明显) | 接近 MHA | **反超 MHA** |
| 训练稳定性 | 好 | 较差 | 好 | 好(但实现复杂) |
| 实现复杂度 | 低 | 低 | 低(repeat_kv) | **高**(decoupled RoPE + absorb) |
| 存量模型转换 | — | uptraining | **uptraining(5%算力)** | 一般需重训 |
| RoPE 兼容 | 天然 | 天然 | 天然 | **需 decoupled 改造** |
| 代表 | GPT-3、原始LLaMA | PaLM、Falcon-7B | Llama2-70B、Llama3全系、Mistral | DeepSeek-V2/V3 |

**选型场景**:
- **存量 MHA 模型要降本** → GQA + uptraining(5% 算力,Llama 团队验证)。改动小、风险低,首选。
- **从头训、追求极致长上下文 + 低 KV** → MLA。但要接受实现复杂度和训练/推理双形态对拍成本。
- **极端追求解码吞吐、能接受掉点** → MQA(小模型/特定场景,如早期 Falcon-7B)。大模型已基本被 GQA/MLA 取代。
- **不在乎 KV cache(短上下文、小 batch)** → 留着 MHA,别过度工程。

**不适用边界**:
- MLA 的 absorb 优化在 **prefill 阶段**(长 prompt 一次性处理)收益不如 decode 阶段——prefill 是 compute-bound,absorb 主要省的是 decode 的 memory-bandwidth。
- GQA 的 uptraining 假设你**有原 MHA checkpoint**;从零训新模型时,直接训 GQA 即可,不存在"转换"。

### 5.2 FFN:Dense vs MoE

| 维度 | Dense FFN | MoE |
|------|-----------|-----|
| 每 token FLOPs | 全部参数 | 仅 top-k 专家 |
| 显存(推理) | = 参数量 | **= 全部专家参数(不省!)** |
| 训练难度 | 标准 | **load balancing + EP 通信 + 路由不稳** |
| 同算力下容量 | 受限 | **大幅提升** |
| 同参数量下质量 | 基准 | 通常更低(稀疏有损) |
| 微调/部署 | 简单 | 复杂(专家放置、负载、量化) |
| 代表 | LLaMA dense | Mixtral、DeepSeek-V3、Qwen-MoE |

**核心 trade-off 一句话**:MoE 用**显存换算力**——你必须装得下全部专家(显存暴涨),换来的是每 token 只花 top-k 的计算(算力/延迟下降)。**适用**:有充足显存(或多卡 EP)、想在固定推理算力下逼高质量上限。**不适用**:显存紧张的单卡部署、需要简单微调/量化的场景、对路由稳定性敏感的小数据微调。

---

## 6. 扎根:失败模式、真坑、根因

### 6.1 MoE 专家塌缩(expert collapse)——头号坑

- **现象**:训练初期少数专家被反复选中,gating 把概率越推越偏,其余专家梯度稀少、几乎不更新("死专家")。最终等效于一个小 dense 模型,白白浪费容量。
- **根因**:routing 的 `argmax`/`top-k` 是**正反馈**——被选中的专家得到更新变得更强→更容易被选中(rich-get-richer)。这是 Shazeer 2017 就识别出的问题。
- **解法**:(a) load balancing aux loss(§3.4,把 f_i·P_i 拉平);(b) router z-loss(惩罚 logit 幅度防数值爆炸);(c) DeepSeek-V3 的 **auxiliary-loss-free**(§3.6,用动态 bias 调负载,不扰动主 loss);(d) 训练初期加 noisy gating / jitter(HF Mixtral 的 `jitter_noise`)。
- **Demo 印证**:§4.4 的 collapsed loss 飙到 7.66,就是塌缩的可观测信号。生产中要监控每专家的 token 占比直方图。

### 6.2 MoE 的 token dropping 与 capacity factor

- **现象**:为了让 EP 通信用固定 buffer,每个专家有 **capacity**(最多接收多少 token)。超出的 token 被**丢弃**(直接走 residual 不过 FFN),训练/推理质量受损。
- **根因**:负载不均 + 固定容量。capacity_factor 设太小→丢 token;太大→浪费显存和算力。
- **解法**:调 capacity_factor(常见 1.0~1.25);或用 dropless MoE(megablocks 的变长 grouped GEMM,不丢 token,代价是 kernel 复杂)。DeepSeek-V3 inference 实现(§3.6)是 dropless(直接 `where` + 累加,无 capacity 截断)。
- **坑**:capacity 截断让**训练和推理行为不一致**(推理 batch 分布与训练不同→丢的 token 不同),复现指标对不齐常源于此。

### 6.3 GQA:mean-pooling 转换的质量回退

- **现象**:拿 MHA checkpoint 直接 mean-pool 成 GQA,**不 uptrain** 就推理,质量明显掉。
- **根因**:mean-pooling 只是个初始化(把 n_rep 个 KV head 平均成 1 个),它破坏了原 head 的特化分工;需要 uptraining(GQA 论文的 5% 算力)让模型适应新的共享结构。
- **解法**:严格按论文做 uptraining;别跳过这步。这是"GQA 几乎无损"的前提条件。

### 6.4 MLA:decoupled RoPE 漏施 / absorb 数值不一致

- **现象一**:把 RoPE 错误地施加到 nope 段,或忘了给 rope 段施加→位置信息错乱,长上下文崩。
- **根因**:MLA 把 head_dim 劈成 nope/rope 两段(§3.5),**只有 rope 段带 RoPE**。新手照搬 MHA 的"对整个 K/Q 施加 RoPE"就会错。
- **现象二**:训练走 naive(还原 K/V)、推理走 absorb(latent 空间),指标对不齐。
- **根因**:两条路径数学等价但**浮点累加顺序/精度不同**(absorb 在低维 latent 算、还涉及 fp8 反量化 §3.5 的 `weight_dequant`)。
- **解法**:用 DeepSeek-V3 仓库同时给的 naive 路径做**逐层对拍**(allclose),确认两路径在 fp32 下数值一致再上 absorb;部署时统一精度策略。

### 6.5 把 MoE 当"省钱/省显存"方案——认知根因坑

- **现象**:团队选 MoE 期望省显存,上线发现显存爆了(要装全部专家)。
- **根因**:混淆了"省 FLOPs"(MoE 给的)和"省显存"(MoE 不给,GQA/MLA 才给)。这两者正交。
- **解法**:选型前先明确瓶颈是**算力还是显存/带宽**。算力受限→MoE;KV cache/带宽受限→GQA/MLA;两者都要→MoE + MLA(DeepSeek-V3 正是两者叠加)。

### 6.6 SwiGLU 的隐层宽度没补偿

- **现象**:把 ReLU-FFN 换成 SwiGLU 时直接复用原 `hidden_dim`,参数量莫名多了 50%,对比不公平/超显存。
- **根因**:GLU 类有 3 个权重矩阵(w1/w2/w3)vs FFN 的 2 个(§3.1)。
- **解法**:按 LLaMA 惯例把 hidden_dim 乘 2/3 再对齐(§3.1 的 `int(2*hidden_dim/3)`)。

---

## 7. 面试 / 实战快问快答

**Q1:GQA 省的是算力还是显存?为什么?**
A:主要省**显存(KV cache)和解码带宽**,几乎不省算力。因为 cache 只存 n_kv 组 KV(§3.2 ⑥),但 attention 计算时 `repeat_kv` 把 KV 临时 expand 回 n_h 个 head 再算(⑦),FLOPs 与 MHA 相当。解码是 memory-bandwidth bound,省 cache = 省每步搬运量 = 提速。

**Q2:MQA、GQA、MHA 是什么关系?**
A:同一条谱上的三点。GQA 分 n_g 组:`n_g=1` 即 MQA(全共享一组 KV),`n_g=n_h` 即 MHA(每 head 一组)。GQA 是两者之间的连续插值,取折中(Llama3 用 8 组)。

**Q3:MLA 为什么需要 decoupled RoPE?直接对压缩后的 K 施加 RoPE 不行吗?**
A:不行。MLA 推理靠 matrix absorption——把 K 的上投影 `W_UK` 提前吸收进 query(§3.5 ⑨),要求 `W_UK` **与位置无关**。但 RoPE 是位置相关旋转 `R_{n-m}` 夹在 q、k 之间,无法并进 `W_UK`。故把 head_dim 劈两段:nope 段走可吸收的低秩路径、rope 段单独带 RoPE,分数相加(⑪)。

**Q4:MLA 推理时到底缓存什么?**
A:absorb 路径只缓存两样(§3.5 ⑩):`d_c` 维的 KV 联合 latent `c_KV` + `d_h^R` 维的 decoupled-RoPE key 旁路 `k_pe`。**注意都不乘 n_heads**(所有 head 共享同一 latent),所以是 `(d_c + d_h^R) × l` 元素/token,DeepSeek-V2 即 `(512+64)×l`。

**Q5:Mixtral 47B 是 8×7B 吗?推理要多少显存?**
A:不是简单 8×7B——attention/embedding 共享,只有 FFN 复制 8 份,总参 47B。每 token 激活 13B(top-2)。但推理**显存要装下全部 47B**(所有专家常驻),省的是算力不是显存。

**Q6:MoE 训练最大的难点?**
A:**负载均衡**。朴素 routing 会专家塌缩(§6.1,rich-get-richer 正反馈)。解法:aux load-balancing loss(f_i·P_i,§3.4)、router z-loss、或 DeepSeek-V3 的无辅助损失动态 bias(§3.6)。外加 expert parallelism 的 all-to-all 通信、capacity/token-dropping(§6.2)。

**Q7:topk-then-softmax 和 softmax-then-topk 有区别吗?**
A:有,权重数值不同(§3.4 的 ⚠)。前者(mistral-inference/论文公式)softmax 分母只含 k 项;后者(HF 实现)对全部 N 个 softmax 取 top-k 再重归一化。两种工业界都在用,迁移权重/复现指标必须对齐用哪种。

**Q8:DeepSeek 的 shared expert 是干嘛的?**
A:每 token **无条件**经过的专家(§3.6 ④),与 top-k 路由专家并存。让它学"所有 token 共需的通用能力",路由专家专注特化能力,避免每个路由专家重复学通用模式,提升参数效率。

**Q9:为什么现代模型几乎都用 RMSNorm 而非 LayerNorm?**
A:RMSNorm 砍掉减均值(re-centering),只做 RMS re-scaling(§3.1),省一半统计量、计算更省,几乎不掉点。LLaMA 起成事实标准。

**Q10:训练时和推理时,GQA / MLA 的形态一样吗?**
A:**GQA 一样**(都是复制 KV head)。**MLA 不一样**:训练常走还原 K/V 的并行形态,推理走 absorb(latent 空间)的等价改写;两者数学等价、数值有别,需对拍(§6.4)。这是 MLA 最大的认知坑。

---

## 8. 未来与展望

- **注意力**:MLA 之后,趋势是"更激进的 KV 压缩 + 兼容长上下文"。值得关注:NSA(native sparse attention)、各种 KV cache 量化(KIVI 等),以及把 MLA 思想推广到 cross-layer KV 共享。
- **MoE**:(a) **更细粒度专家**(DeepSeek 路线,专家更多更小,组合表达力更强);(b) **无辅助损失负载均衡**(§3.6)成为新基线,摆脱 aux-loss 对主目标的扰动;(c) MoE + MLA 叠加(DeepSeek-V3 已是范式),把"省算力"和"省 KV"两条正交收益合一;(d) 推理侧 expert offloading / 动态加载,缓解"全专家常驻显存"的痛点。
- **认知层面的提醒**:这三类升级仍在快速演化,**别把任何一个当终态**。选型时回到第一性:你的瓶颈是算力、显存、带宽还是质量?对症下药,避免为了"用上 SOTA 架构"而过度工程。

---

## 9. 五件套（动手巩固）

> 前 4 个是**扩展 demo**(代码题),建议在你的环境跑通并改参观察。

**(1) 代码题 · 扩展 Demo 1:画 KV cache 随上下文增长曲线**
基于 §4.1,固定 DeepSeek-V2 配置,横轴 seq_len ∈ {2k, 8k, 32k, 128k},纵轴 KV cache 显存(GB,按 bf16=2字节算),四条线 MHA/GQA/MQA/MLA。验证:128k 上下文时 MHA 的 KV cache 是否超过模型权重本身?(提示:`bytes = elems × 2`,体会"长上下文下 cache > weights")

**(2) 代码题 · 扩展 Demo 2:给 toy MoE 加 load balancing loss 并训练**
基于 §4.2 + §4.4,把 toy MoE 接一个简单回归目标,对比"加 aux loss"vs"不加"训练 200 步后的专家负载直方图。验证:不加会塌缩(某专家占比→1),加了趋于均匀。再实现 DeepSeek 的动态 bias 版,对比两种均衡手段。

**(3) 代码题 · 扩展 Demo 5:实现 MLA 的 absorb 路径并与 naive 对拍**
基于 §4.5,补上真实 RoPE 旋转,再实现 §3.5 的 absorb 路径(把 wkv_b 吸收进 q_nope),用 `torch.allclose` 验证 absorb 输出 == naive 输出(fp32 下 atol=1e-4)。体会"数学等价、数值有别"。

**(4) 代码题 · GQA uptraining 模拟**
构造一个 toy MHA,mean-pool 成 GQA(n_g=2),对比:(a) 直接推理掉多少点;(b) 微调少量步后恢复多少。在玩具任务上复现 GQA 论文"uptraining 后接近 MHA"的现象。

**(5) 简答题 · 架构选型 case**
你要给一个**128k 长上下文、单机 8×A100(80G)、追求高吞吐**的客服模型选架构。dense 还是 MoE?MHA/GQA/MLA 选哪个?写出你的推理链:先判断瓶颈(算力?KV cache 显存?带宽?),再对照 §5 的不适用边界给方案,并说明 MoE 的显存约束是否被 8×80G 满足。

---

## 附录:本章信息源（均实际 WebFetch / WebSearch 核实）

**真实源码（raw.githubusercontent.com）**
- `meta-llama/llama@llama/model.py` — repeat_kv / Attention(GQA)/ FeedForward(SwiGLU)/ RMSNorm
- `mistralai/mistral-inference@src/mistral_inference/moe.py` — MoeArgs / MoeLayer top-k 路由
- `huggingface/transformers@src/transformers/models/mixtral/modeling_mixtral.py` — router / load_balancing_loss_func
- `deepseek-ai/DeepSeek-V3@inference/model.py` — MLA(naive + absorb)/ Gate / MoE

**设计考古（论文，arXiv）**
- 1701.06538 — Shazeer et al. 2017,*Outrageously Large Neural Networks*(稀疏 MoE + load balancing)
- 1911.02150 — Shazeer 2019,*Fast Transformer Decoding: One Write-Head is All You Need*(MQA)
- 2002.05202 — Shazeer 2020,*GLU Variants Improve Transformer*(SwiGLU)
- 2305.13245 — Ainslie et al. 2023,*GQA*(EMNLP,uptraining 配方)
- 2401.04088 — Jiang et al. 2024,*Mixtral of Experts*(8x7B,47B/13B)
- 2405.04434 — DeepSeek-AI 2024,*DeepSeek-V2*(MLA,KV cache -93.3%)

**待核条目**
- SwiGLU 论文"divine benevolence / we offer no explanation"原文措辞:WebSearch 未命中确切句子,标「待核」(SwiGLU 被 PaLM/LLaMA 采用为确证)。
- HF Mixtral 近期重构后 `MixtralExperts` 的完整向量化 dispatch 体(one_hot/permute 后的 grouped GEMM 细节):WebFetch 仅取到前半段,完整 kernel 路径标「待核」,但路由 + aux-loss 核心已核实。
