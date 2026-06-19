---
title: "Capstone:剖一个真实推理服务并把它调快"
slug: "2-16"
collection: "ai-research-compass"
group: "MLSys专家课程"
order: 2016
summary: "这章把你从\"学过 PagedAttention、投机解码、量化的原理\"带到\"能对一个真实的 LLM 推理服务,走完一次严谨的测—改—测闭环:用 profiler 抓 trace 定位瓶颈、用 Roofline 判断该往哪打、依次施加优化并各自量化收益、最后产出一份能给老板和同事看的优化报告\"。"
topics:
  - "AI 研究"
tags: []
createdAt: "2026-06-17T14:01:39.000Z"
updatedAt: "2026-06-17T14:01:39.000Z"
---
> 这章把你从"学过 PagedAttention、投机解码、量化的原理"带到"能对一个真实的 LLM 推理服务,走完一次严谨的测—改—测闭环:用 profiler 抓 trace 定位瓶颈、用 Roofline 判断该往哪打、依次施加优化并各自量化收益、最后产出一份能给老板和同事看的优化报告"。读完你应该能拿任意一个 vLLM 服务,独立完成"基线 → 瓶颈 → 逐步措施与收益 → 结论"的完整工程动作,而不是凭感觉调参。

## 一、动机:为什么收官要做一次"动手"

前面 15 章我们把现代推理引擎的内核拆了个底朝天:第 9 章证明注意力是**访存受限**、第 10 章把 KV cache 当虚拟内存分页、第 11 章用并行验证无损加速 decode、第 12 章把权重压成 INT4 抬高算术强度。每一章你都能在白板上推导。但**白板上推导对**和**线上调得快**之间,隔着一道很多人跨不过去的坎——这道坎不是知识,是**方法论纪律**。

工业界优化推理服务最常见的失败模式有三种,我见过无数次:

- **凭感觉优化**:"听说量化能加速,上!"——上完发现延迟反而涨了(小 batch 下 INT4 反量化开销盖过了带宽收益),因为没先测、不知道自己的瓶颈根本不在权重带宽。
- **优化错地方**:花两周把一个只占总耗时 5% 的 kernel 调快了 3 倍,整体提升 1.5%(这正是 **Amdahl 定律** 要锤你的)。
- **改完不量化**:上了三个优化,吞吐"感觉快了",但说不清每个优化各贡献多少、有没有哪个其实是负优化、P99 延迟有没有被某个优化偷偷拉高。

这一章的全部价值,就是给你一套**可复现的纪律**来对抗这三种失败。纪律可以浓缩成四句话,贯穿全章:

> **(1) 先测后改**:没有基线数字,任何"变快了"都是错觉。
> **(2) 抓大头(Amdahl)**:只优化占比最大的那一段,小头先放着。
> **(3) Roofline 指导**:先判断瓶颈是 compute-bound 还是 memory-bound,再决定用哪类优化,别拿"省算力"的手段去打"缺带宽"的瓶颈。
> **(4) 每步独立量化**:一次只改一个变量,前后对照,把收益钉死到具体措施上。

注意一个**反直觉但必须接受的前提**:推理服务的优化目标不是单一数字。**吞吐(throughput)、延迟(P50/P99)、显存** 三者经常互相打架——大 batch 提吞吐却拉高单请求延迟,量化省显存却可能增延迟。所以全章每一步都要同时记三个量,任何"只报吞吐不报延迟"的优化结论都是耍流氓。

本章用一个具体场景把闭环跑通:**单卡(假设一张 80GB A100【待核,以你手头卡为准】),vLLM 起一个 Llama 系开源模型,先定基线负载,再依次施加 PagedAttention 验证 → AWQ 权重量化 → 投机解码 → batch/chunked-prefill 调参,每步出前后对照表。** 你照着能在自己的服务上原样复刻。

## 二、第 0 步:把"快慢"定义清楚——指标与负载

优化之前,先定义"什么叫快"。这一步看似废话,却是 90% 的人跳过然后翻车的地方。

### 2.1 推理服务的核心指标(必须区分清楚)

```
吞吐(throughput)
  · 请求吞吐  req/s        —— 每秒完成多少个请求(受输出长度影响大)
  · 输出吞吐  out_tok/s    —— 每秒产出多少个生成 token(更公平的吞吐口径)
  · 总吞吐    tot_tok/s    —— 含 prefill 处理的输入 token(衡量算力利用)

延迟(latency,必须看分位数,不是均值)
  · TTFT  Time To First Token   —— 首 token 延迟,prefill 主导,交互体验关键
  · TPOT  Time Per Output Token —— 每个后续 token 的平均间隔,decode 主导
  · E2E   端到端延迟 = TTFT + (输出长度−1) × TPOT
  · P50 / P99 —— 中位数 / 99 分位;P99 才暴露长尾,SLO 通常压 P99

显存
  · 权重占用 + KV cache 占用 + 激活/临时缓冲
  · KV cache 可用容量 = 决定能并发多少请求 = 决定吞吐上限(回扣第10章)
```

**为什么延迟一定要看分位数而不是均值?** 因为推理延迟分布是**重尾**的:大多数请求很快,但少数请求(撞上抢占、长输出、调度排队)会慢很多。均值会被少数极端值带偏,也会把"大多数还行但尾巴很烂"这种致命情况掩盖掉。SLO(服务等级目标)几乎总是写成"P99 < X ms",所以你优化时盯的是 P99。**一个优化让 P50 降了但 P99 涨了,在延迟敏感场景里是负优化**——这种事在开大 batch 时经常发生。

**TTFT 和 TPOT 为什么要分开?** 因为它们由两个不同阶段主导:TTFT 由 **prefill**(compute-bound,第 10/11 章)决定,TPOT 由 **decode**(memory-bound)决定。**分不开就找不准瓶颈**——你以为服务慢,可能 TTFT 很好只是 TPOT 烂(decode 带宽不够),也可能反过来(prefill 太长阻塞)。这个区分直接决定第三步往哪打。

### 2.2 定基线负载:把实验条件钉死

负载不定死,数字没法比。一个最小可复现负载至少要固定这五个量:

```
模型          : 如 Llama-3-8B-Instruct(权重精度 BF16)【模型名以你实际用的为准】
输入长度分布  : 如固定 512 token(或用真实 trace 的长度分布)
输出长度      : 如固定生成 128 token(用 ignore_eos 强制生成满,见下)
并发           : 如 64 个并发请求(closed-loop)或 10 req/s(open-loop)
采样配置       : temperature / top_p / seed —— 必须固定,否则输出长度会抖
```

**两种压测口径,别混用:**

- **Closed-loop(闭环并发)**:维持固定数量 N 的"在飞"请求,一个完成立刻补一个。测的是**饱和吞吐**(系统能跑多满)。调吞吐时用这个。
- **Open-loop(开环到达率)**:按固定 QPS 泊松到达地发请求,不管系统能不能消化。测的是**给定负载下的延迟分布**,能暴露排队延迟。测 SLO 时用这个。

> ⚠ **坑(极其高频):测输出吞吐时一定要 `ignore_eos=True` 让每个请求都生成满固定长度。** 否则不同优化(尤其是带采样随机性或投机解码)会改变实际生成长度,你测到的"吞吐变化"里混进了"长度变化",根本不是同一个实验。把输出长度变成受控常量,是公平对比的前提。

vLLM 自带 benchmark 脚本(`benchmarks/benchmark_serving.py` 和 `benchmark_throughput.py`【路径以你 clone 的版本为准,待核】),也可以自己写一个最小骨架(见第八节)。先把基线数字记下来,后面每一步都跟它比。

## 三、第 1 步:起服务、跑基线

### 3.1 起一个 vLLM OpenAI 兼容服务

```bash
# 启动 vLLM 的 OpenAI 兼容 server(基线:BF16 权重,PagedAttention 默认开启)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --port 8000
# 关键参数解释:
#   --gpu-memory-utilization 0.90  指 vLLM 可用 GPU 显存的比例;
#       它先放权重,剩下的几乎全部切成 KV cache 物理块(回扣第10章 block pool)
#   --max-model-len 4096           单请求最大 (prompt+输出) 长度上限
# 此版本 PagedAttention 是默认且唯一的 KV 管理方式,无需额外开关
```

### 3.2 跑基线 benchmark

用 closed-loop 压满,固定输入 512 / 输出 128 / 并发 64:

```bash
# 用 vLLM 自带的 serving benchmark(对着上面起的 server 打)
python benchmarks/benchmark_serving.py \
    --backend vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset-name random \
    --random-input-len 512 \
    --random-output-len 128 \
    --max-concurrency 64 \
    --num-prompts 512 \
    --ignore-eos \
    --percentile-metrics ttft,tpot,e2el       # 让它同时报 P50/P99
# 产出:输出吞吐 (tok/s)、TTFT P50/P99、TPOT P50/P99、E2E P50/P99
```

把结果填进基线行(数字是**示意占位,你要填自己实测的**):

```
=== 基线(BF16,PagedAttention 默认开,并发64,in512/out128)===
输出吞吐 : 2,800 tok/s   【示意,待你实测】
TTFT P50 : 180 ms        TTFT P99 : 420 ms
TPOT P50 : 18 ms         TPOT P99 :  35 ms
KV 显存  : 已用 ~48 GB(权重 ~16GB,KV pool ~50GB 中用了大部分)
```

**这一行是整章的锚点。后面每个优化都跟它比,任何"变快了"都要相对这个数字。**

## 四、第 2 步:抓 trace 定位瓶颈(测的核心)

有了基线数字,下一步是**找瓶颈**——别急着上优化。这一步是全章方法论的灵魂:**先看清楚时间花在哪,再决定打哪。**

### 4.1 两种 profiler,各看一层

```
torch profiler(PyTorch 自带)
  · 看什么:Python/算子级时间线,每个 op(matmul、attention、layernorm…)的耗时占比、
            CPU 下发(launch)与 GPU 执行的关系、是否有 CPU-bound 的下发瓶颈
  · 适合:回答"哪个 op / 哪段代码吃了最多时间""是不是 kernel launch 太碎"
  · 入口:torch.profiler.profile(...)  导出 chrome trace 或 tensorboard

Nsight Systems(nsys,NVIDIA 系统级)
  · 看什么:整机时间线 —— CUDA kernel、内存拷贝(H2D/D2H)、NCCL 通信、CPU 线程,
            统一时间轴对齐;能看到 GPU 是否有空泡(gap)、通信和计算是否重叠
  · 适合:回答"GPU 利用率高不高""有没有等数据的空泡""是不是通信受限(多卡时)"
  · 入口:nsys profile -o report ...   再用 Nsight Systems GUI / nsys stats 看
```

经验:**先用 nsys 看"GPU 忙不忙、有没有空泡、是不是在等拷贝/通信",定性判断瓶颈大类;再用 torch profiler 钻进去看"具体哪个 op 占比最大"。** 两者互补,一个看系统、一个看算子。

### 4.2 抓 trace 的命令

```bash
# --- 用 nsys 抓系统级 trace(包住一段稳定负载) ---
nsys profile \
    --trace=cuda,nvtx,osrt \
    --gpu-metrics-device=all \
    --output=baseline_report \
    --force-overwrite=true \
    python run_fixed_load.py        # 你的脚本:发固定一批请求、跑稳态
# 看结果(命令行汇总,不用开 GUI):
nsys stats --report cuda_gpu_kern_sum baseline_report.nsys-rep | head -30
#   → 列出耗时最高的 kernel 及其总耗时占比 —— 这就是 Amdahl 的"大头"清单
```

```python
# --- 用 torch profiler 抓算子级 trace(对一段 decode 循环) ---
import torch
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=2, active=5),   # 跳过预热,只测稳态5步
    record_shapes=True,
    with_stack=True,
) as prof:
    for _ in range(8):
        run_one_decode_iteration()       # 触发一次 forward(替换成你的调用)
        prof.step()
# 打印按 GPU 自身耗时排序的 op top-N —— 看谁占大头
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
# 也可导出 chrome trace 用 chrome://tracing 或 perfetto 可视化看时间线:
prof.export_chrome_trace("decode_trace.json")
```

### 4.3 读 trace 的清单:四个要回答的问题

抓到 trace,带着这四个问题去读(这是把"一堆时间线"变成"瓶颈结论"的关键)：

```
Q1. prefill 还是 decode 占大头?
    · 看时间线里 prefill kernel(大矩阵乘,处理整段 prompt)与
      decode kernel(逐 token 的瘦长前向)各占多少总时长。
    · 短 prompt + 长输出 → decode 主导(memory-bound,该打第11/12章的手段)
    · 长 prompt + 短输出 → prefill 主导(compute-bound,该打 chunked prefill / 并行)

Q2. 哪个 kernel 最贵?(Amdahl 的大头)
    · nsys 的 cuda_gpu_kern_sum 直接给排序。通常是:
      attention kernel、各个 GEMM(qkv/o/gate/up/down proj)、可能还有 sampling。
    · 把 top-3 kernel 的占比加起来 —— 这决定了你优化的天花板。

Q3. GPU 有没有空泡(是不是没喂饱)?
    · nsys 时间线上 GPU 行有大片空白 = GPU 在等(等 CPU 下发?等拷贝?等通信?)。
    · decode 阶段如果空泡多,常是 batch 太小 / kernel launch 太碎 / CPU 调度跟不上。

Q4. 是不是通信受限 / KV 吃满显存?(多卡 or 大并发)
    · 多卡:看 NCCL all-reduce/all-gather 是否和计算重叠,还是串行等通信(回扣第14章)。
    · 单卡:看 KV pool 是否接近占满、有没有频繁的 preempt/swap(回扣第10章抢占)。
      vLLM 日志会打印 "Preemption" 警告 / KV cache 使用率,这是 KV 吃满的直接信号。
```

**这一步的产出,是一句明确的瓶颈判断**,例如:"基线负载下 decode 占总时长 ~80%,最贵的三个 kernel 是 attention + qkv_proj + down_proj 共占 decode 的 ~65%,GPU 在 decode 阶段约有 20% 空泡(batch 偏小),KV pool 用了 80% 未触发抢占。" 有了这句话,才知道该上哪些优化、上的顺序怎么排。

### 4.4 用 Roofline 把瓶颈"翻译"成优化方向

光知道"哪个 kernel 贵"还不够,还要知道**它为什么贵**——是算不过来(compute-bound)还是搬不过来(memory-bound)。这正是 **Roofline 模型** 的用处(第 9、11 章已建立)。复述一遍判据:

```
算术强度 I = 该 kernel 的 FLOP 数 / 它读写 HBM 的字节数   (FLOP/Byte)
脊点      I* = 硬件峰值算力 P / 峰值带宽 B
   · I < I*  → memory-bound：瓶颈是带宽。优化方向 = 减少要搬的字节
              (量化权重↓字节、KV 量化、增大 batch 提升数据复用、投机解码摊薄读权重)
   · I > I*  → compute-bound：瓶颈是算力。优化方向 = 减少要算的 FLOP 或提升算力利用
              (更好的 GEMM tiling、低精度算力、减少冗余计算、chunked prefill 平滑算力)
```

把第三步抓到的瓶颈往 Roofline 上一放,优化方向**几乎自动浮现**:

```
观察:decode 主导、attention+GEMM 占大头、GPU 有空泡、batch=64 但单序列算术强度低
推断:decode 单序列 I≈1 FLOP/Byte ≪ I*(脊点几十~上百)→ 重度 memory-bound
     → 该打"减少搬运字节 / 提升数据复用"这条线:
        · 增大有效 batch(让一次读权重服务更多序列)→ 提算术强度
        · AWQ 量化权重(W16→W4)→ 直接把读权重的字节砍到 ~1/4(第12章)
        · 投机解码 → 把 K 次串行读权重折叠成 ~1 次(第11章)
     注意:这些都是"打 memory-bound"的手段。如果瓶颈是 prefill(compute-bound),
           这几招里只有"增大 batch 提利用"和 chunked prefill 才对路,量化收益会小很多。
```

**这一步是全章最值钱的方法论**:不是"把所有优化都试一遍看哪个快",而是**先用 Roofline 判断瓶颈性质,只施加与瓶颈对路的优化**。盲目穷举既慢又会得出误导结论(在错误制度下测一个优化,可能得出它"没用"的假象)。

## 五、第 3 步:逐个施加优化,每步前后对照

现在开始"改",纪律是**一次只改一个变量,改完立刻用第三节同一套 benchmark 复测,填进对照表**。顺序按 Roofline 推断的"性价比 × 风险"排:先验证已有的(PagedAttention),再上低风险高收益的(量化、调 batch),最后上高风险需验证无损的(投机解码)。

### 5.1 优化 A:验证 PagedAttention 确实生效(第 10 章)

vLLM 默认就开了 PagedAttention,所以这里不是"加上它",而是**验证它在起作用、并理解它给了你多大 KV 容量**——这是后面所有吞吐优化的地基。

**怎么验证生效(三个独立证据):**

```
证据1:KV 容量公式对照。vLLM 启动日志会打印类似
       "# GPU blocks: 12000, # CPU blocks: 2048"(数字示意【待核】)。
       自己用第10章公式 block_bytes = block_size · 2 · L · d_model · P 算一遍,
       num_gpu_blocks × block_bytes 应 ≈ (可用显存 − 权重)。对得上 = 分页在按块管 KV。

证据2:碎片消失。把 max_model_len 从 4096 调到 8192,基线方案(连续预留)会因为
       单请求预留翻倍而骤降可并发数;PagedAttention 下,只要请求实际短,
       可并发数几乎不变(因为按需分配、不预留)。这个"改 max_len 不掉并发"的现象
       就是内部碎片被锁死在一个块以内的直接体现(回扣第10章 4.2/5.2)。

证据3:前缀共享(若开 enable_prefix_caching)。发两批共享同一长 system prompt 的请求,
       第二批的 TTFT 应显著低于第一批(前缀 KV 命中、prefill 被复用,COW 红利)。
```

PagedAttention 不是一个"前后对照吞吐"的旋钮(没有它 vLLM 跑不起来),它的收益体现在**KV 容量 = 你能开多大并发**。所以它的"对照"放在下一项(调 batch/并发)里一起体现——你能把并发开到 64、128,本身就是分页给的。

### 5.2 优化 B:AWQ 权重量化(第 12 章)

decode 是 memory-bound,瓶颈在"读权重的字节数"。**AWQ(Activation-aware Weight Quantization)** 把权重从 16-bit 压到 4-bit,读权重字节直接 ~÷4,理论上 decode(读权重主导那部分)能接近 ~4× 带宽收益。回扣第 12 章:AWQ 通过"按激活分布保护重要权重通道的缩放"在 INT4 下保住精度。

```bash
# 用 AWQ 量化版权重起服务(社区常有现成 AWQ 量化好的 checkpoint)
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-3-8B-Instruct-AWQ \   # 模型名示意【待核,用你能拿到的AWQ权重】
    --quantization awq \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --port 8000
# 量化后权重从 ~16GB 降到 ~5GB → 省出的显存全变成更多 KV 块 → 可并发更高(双重收益!)
```

复测、填表。**关键提醒:量化收益高度依赖 batch 制度,必须分别测**:

```
=== 优化B:AWQ INT4 权重 vs 基线BF16(同负载 in512/out128)===
                     基线BF16      AWQ-INT4     变化
输出吞吐(并发64)   2,800 tok/s   ~4,600 tok/s  +64%   【示意,待实测】
TPOT P50            18 ms         ~12 ms        −33%   ← decode 读权重字节↓,带宽收益
TTFT P50            180 ms        ~165 ms       −8%    ← prefill compute-bound,量化帮助小
权重显存            ~16 GB        ~5 GB         省11GB → 多出的全给 KV
KV 可并发上限        ~X            ~更高         ← 权重省下的显存转成 KV 容量
```

> ⚠ **量化的两个坑**:(1) **小 batch / 极低延迟场景,INT4 反量化(dequant)开销可能盖过带宽收益**,TPOT 不降反升——务必在你的真实 batch 下测,别信"量化必快"。(2) **精度可能掉**:量化是 lossy(对比第 11 章投机解码的无损),上线前要在你的评测集上比一下质量,别只看速度。这正是"先测后改、用数据说话"的纪律——量化对不对路,Roofline + 实测说了算。

### 5.3 优化 C:投机解码(第 11 章)

decode memory-bound 还有算力闲置,投机解码用小模型草拟 K 个 token、大模型一次前向并行验证,**无损**地把多个 token 折叠进一次大模型前向。回扣第 11 章:加速比 ≈ E[tokens/轮] / (1 + K·c),净赚条件是 α(1−α^K)/(1−α) > K·c。

```bash
# vLLM 开投机解码:指定草稿模型 + 草稿 token 数 K(配置项名以版本为准【待核】)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \  # 同系列小模型当草稿,词表一致!
    --num-speculative-tokens 4 \        # 即第11章的 K,常取 3~5
    --dtype bfloat16 --port 8000
```

复测、填表,并**额外记录实测接受率 α 和接受长度**(vLLM 通常会暴露 spec decode 的接受统计【待核】):

```
=== 优化C:投机解码(K=4)vs 不开(同模型,小batch低延迟场景)===
                     不开           开spec(K=4)   变化
TPOT P50(并发8)    18 ms          ~10 ms         −44%   ← 一次前向出多个token
输出吞吐(并发8)    低             更高           +显著
实测接受率 α        —              ~0.75          ← 决定收益;α高=收益大
实测接受长度 E[n]   —              ~2.8 tok/轮    与公式(◆)对照验算
--- 但在大 batch 下 ---
输出吞吐(并发64)  4,600(已AWQ)  可能↓!         ← 大batch已接近compute-bound,
                                                    "多算K个token"不再免费(第11章边界)
```

> ⚠ **投机解码最大的坑(第 11 章重点)**:它的红利来自**单序列 memory-bound 时算力闲置**。**大 batch 下大模型已接近 compute-bound,"多算 K 个 token"不再免费,加速比会掉到甚至 < 1(负优化)。** 所以投机解码主要利好**低延迟/小 batch** 制度。这就是为什么我把它和"调 batch"分开测、且明确标注 batch 制度——**同一个优化在不同 batch 下结论相反**,这是本章"分制度测"纪律最锋利的体现。

### 5.4 优化 D:调 batch / 并发 / chunked prefill(第 10 章)

最后是调度旋钮,它们不改算法,只改"怎么喂"。核心是在 Roofline 上**把工作点往脊点推**(提算术强度),同时守住延迟 SLO。

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ... --quantization awq \
    --max-num-seqs 256 \             # 调度器一个 step 最多并发多少序列(decode 宽度)
    --max-num-batched-tokens 8192 \  # 一个 step 最多处理多少 token(prefill+decode 合计)
    --enable-chunked-prefill \       # 把长 prefill 切块,与 decode 交错,避免阻塞 decode
    --port 8000
```

三个旋钮的作用与权衡:

```
max-num-seqs(并发序列上限):
   ↑ 提高 → 一次读权重服务更多序列 → 算术强度↑ → 吞吐↑(直到撞算力上限或KV不够)
   代价:单步计算量↑ → 单步延迟↑ → TPOT/P99 可能恶化。吞吐与延迟在此拔河。

max-num-batched-tokens(单步 token 预算):
   控制一个 forward 处理的总 token,直接影响单步耗时上限,是延迟的闸门。

chunked prefill(回扣第10章 7.4 / 第11章):
   问题:长 prompt 的 prefill 是个大 compute 块,会霸占一整步,把正在 decode 的请求
         憋住 → 这些请求的 TPOT/TTFT 尖刺(P99 长尾恶化)。
   解法:把长 prefill 切成小块,和 decode 在同一步交错跑 → prefill 不再独占、
         decode 不被长 prefill 阻塞 → P99 平滑。
   收益主要体现在【混合负载下的 P99 延迟】,不是平均吞吐——所以要看 P99 才看得见它的价值。
```

**做一条 batch 扫描曲线**(这是调度调参的标准动作),固定其它、只扫 `max-num-seqs`:

```
=== 优化D:并发扫描(AWQ权重,in512/out128,closed-loop)===
max-num-seqs   输出吞吐(tok/s)   TPOT P50   TPOT P99   KV是否吃满
   16            ~2,000            9 ms       14 ms      宽松
   64            ~4,600            12 ms      20 ms      宽松
  128            ~6,200            18 ms      40 ms      接近满
  256            ~6,800            31 ms      90 ms      满,偶发抢占
  → 吞吐随并发先快涨后饱和(撞算力上限/KV满);延迟单调上升。
  → 最优工作点 = 在你的 P99 SLO 约束下,选吞吐最高的那个并发(如 SLO=P99<50ms → 选128)。
```

**这条曲线把"吞吐 vs 延迟"的拔河画成了可决策的图**:不是越大越好,而是**在 SLO 红线内取吞吐最大点**。这是整个调参的终局动作。

## 六、把每步收益拼起来:Amdahl 与归因纪律

逐步优化做完,最后要回答一个老板必问的问题:**总共快了多少?每个优化各贡献多少?** 这里有两条铁律。

### 6.1 Amdahl 定律:优化收益被"没优化的部分"封顶

第 1 步抓 trace 时我们看到"decode 占 80%、prefill 占 20%"。Amdahl 定律告诉你:**只优化 decode,整体加速比的上限被那 20% 的 prefill 死死封住**。

```
设某部分占总耗时比例 f,你把它加速 s 倍,整体加速比:
   Speedup_total = 1 / ((1 − f) + f/s)
当 s → ∞(把这部分优化到瞬间完成):
   Speedup_max = 1 / (1 − f)               ← 上限只取决于 f
例:decode 占 f=0.8,即使把 decode 优化到无穷快:
   Speedup_max = 1/(1−0.8) = 5×            ← 整体最多快 5 倍,prefill 那 20% 是天花板
   若 decode 实际只加速 s=3:
   Speedup_total = 1/(0.2 + 0.8/3) = 1/0.467 ≈ 2.14×
```

**推论(直接指导排优先级)**:(1) 先优化占比最大的段(decode),回报最高;(2) 当 decode 被压到很小后,prefill 那 20% 反而变成新的大头,届时该回头优化 prefill(chunked prefill、prefill 并行)——**瓶颈是会转移的,每轮优化后要重新抓 trace 看新大头在哪**。这就是为什么优化是**迭代闭环**而非一次性:测→改→再测→发现新瓶颈→再改。

### 6.2 归因纪律:增量叠加,不是各自孤立相乘

报告里写"AWQ +64%、投机 +40%、调 batch +48%"然后声称总共 +200% 是**错的**——这些优化互相不独立(AWQ 省的显存被调 batch 吃掉、投机在大 batch 下收益缩水)。正确做法:

```
按"逐步叠加"记录,每一步都是"在上一步基础上再开这一个":
   step0 基线BF16,并发64                          : 2,800 tok/s   (1.00×)
   step1 +AWQ(仍并发64)                           : 4,600 tok/s   (1.64×, 累计1.64×)
   step2 +调并发到128(AWQ已开)                    : 6,200 tok/s   (1.35×, 累计2.21×)
   step3 +chunked prefill(看P99,吞吐基本不变)     : 6,200 tok/s   (P99 90→55ms ✓)
   (投机解码在并发128这种大batch下经测为负优化,故最终配置不采用 —— 这也是一条结论!)
最终:吞吐 2,800 → 6,200 tok/s(2.21×),P99 在 SLO 内,显存从量化省出转为KV容量。
```

**注意 step3 和"投机不采用":一个优化吞吐没涨但 P99 改善了,是正收益;一个优化经测是负优化,把"不采用"明确写进结论同样有价值。** 报告的诚实度,体现在你敢不敢写"我测了 X,在我的负载下没用/反而变慢"。

## 七、推理优化报告模板(可直接套用)

把全章方法论固化成一份可交付模板。每次优化一个服务,填满它,就是一份合格的工程报告:

```
# 推理优化报告:<服务名 / 模型 / 日期>

## 1. 目标与约束(先定义"快")
- 优化目标(主):  例)在 P99 E2E < 500ms 约束下最大化输出吞吐
- 硬约束:        SLO(P99=__ms)、单卡显存=__GB、模型/精度要求
- 基线负载:      模型=__ 输入=__ 输出=__(ignore_eos) 并发=__ 采样=__(seed固定)
- 压测口径:      closed-loop / open-loop(写清,别混)

## 2. 基线(先测后改的锚点)
| 指标 | 值 |
|---|---|
| 输出吞吐 (tok/s) | __ |
| TTFT P50 / P99 (ms) | __ / __ |
| TPOT P50 / P99 (ms) | __ / __ |
| E2E  P50 / P99 (ms) | __ / __ |
| 显存:权重 / KV / 余量 | __ / __ / __ |

## 3. 瓶颈定位(抓大头 + Roofline)
- 工具:nsys(系统级)+ torch profiler(算子级)
- prefill vs decode 占比:        __% / __%        → 主导阶段 = __
- Top-3 最贵 kernel 及占比:       __, __, __(合计 __%)
- GPU 空泡 / KV 是否吃满 / 是否通信受限: __
- Roofline 判定:算术强度 I≈__,脊点 I*≈__ → 瓶颈性质 = memory/compute-bound
- **一句话瓶颈结论**:____________________
- **据此选定的优化方向(只选对路的)**:____________________

## 4. 逐步措施与收益(一次改一个变量,前后对照)
| 步骤 | 措施 | 出处章 | 吞吐 | TPOT P99 | TTFT P99 | 显存 | 累计加速 | 备注/制度 |
|---|---|---|---|---|---|---|---|---|
| 0 | 基线 | — | __ | __ | __ | __ | 1.00× | |
| 1 | __ | 第_章 | __ | __ | __ | __ | __× | 并发=__ |
| 2 | __ | 第_章 | __ | __ | __ | __ | __× | |
| ... | | | | | | | | |
- 每步必须注明:测试时的 batch/并发制度、是否同一负载、是否 ignore_eos。
- 负优化也要记:例)"投机解码在并发≥64 下加速比<1,不采用"。

## 5. 结论
- 最终配置(可直接上线的启动参数):____________________
- 最终 vs 基线:吞吐 __→__(__×),P99 __→__(是否满足 SLO:是/否)
- 仍存在的瓶颈 / 下一轮该打哪(瓶颈转移到哪了):____________________
- 风险提示:量化导致的质量变化(评测集结果)、稳定性(抢占频率)等。
```

这份模板的每一栏都对应本章一条纪律:第 1 节"先定义快"、第 2 节"先测"、第 3 节"抓大头+Roofline"、第 4 节"逐步量化+分制度+记负优化"、第 5 节"结论+瓶颈转移"。**填满它,你就完成了一次有纪律的优化,而不是一次凭感觉的瞎调。**

## 八、代码:最小 benchmark 骨架(测吞吐/延迟分位数)

如果不想用 vLLM 自带脚本,这是一个最小的 closed-loop 压测骨架,把"测吞吐 + P50/P99 延迟"这件事讲透。机制核心:**维持固定并发、记录每个请求的 TTFT 与 E2E、最后算分位数**。

```python
import asyncio, time, statistics, aiohttp

API = "http://localhost:8000/v1/completions"
MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
CONCURRENCY = 64          # closed-loop 并发数
TOTAL_REQUESTS = 512
PROMPT = "Once upon a time " * 100   # 凑到 ~固定输入长度(实际应按 token 数控制)
OUTPUT_LEN = 128

async def one_request(session, sem, records):
    async with sem:                              # 信号量维持固定并发
        t0 = time.perf_counter()
        first_token_t = None
        payload = {
            "model": MODEL, "prompt": PROMPT,
            "max_tokens": OUTPUT_LEN,
            "ignore_eos": True,                  # ★ 强制生成满,保证输出长度受控(第2节坑)
            "temperature": 0.0, "stream": True,  # 流式才能测 TTFT(首 token 到达时刻)
        }
        n_out = 0
        async with session.post(API, json=payload) as resp:
            async for line in resp.content:      # 逐 chunk 读流式输出
                if not line.strip():
                    continue
                if first_token_t is None:        # 记录首 token 到达 → TTFT
                    first_token_t = time.perf_counter()
                n_out += 1                        # 粗略计 token(精确应解析每个 chunk)
        t_end = time.perf_counter()
        records.append({
            "ttft": (first_token_t - t0) * 1e3,           # ms
            "e2e":  (t_end - t0) * 1e3,                    # ms
            "tpot": (t_end - first_token_t) * 1e3 / max(n_out - 1, 1),  # ms/token
            "n_out": n_out,
        })

async def main():
    sem = asyncio.Semaphore(CONCURRENCY)
    records = []
    wall_start = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.create_task(one_request(session, sem, records))
                 for _ in range(TOTAL_REQUESTS)]
        await asyncio.gather(*tasks)
    wall = time.perf_counter() - wall_start

    # ---- 汇总:吞吐 + 延迟分位数(P50/P99 才看得见长尾)----
    total_out = sum(r["n_out"] for r in records)
    def pct(xs, p):                              # 简单分位数(线性插值可换 numpy)
        xs = sorted(xs); k = int(round((len(xs) - 1) * p))
        return xs[k]
    ttfts = [r["ttft"] for r in records]
    tpots = [r["tpot"] for r in records]
    e2es  = [r["e2e"]  for r in records]
    print(f"输出吞吐      : {total_out / wall:8.1f} tok/s")
    print(f"TTFT P50/P99 : {pct(ttfts,0.5):7.1f} / {pct(ttfts,0.99):7.1f} ms")
    print(f"TPOT P50/P99 : {pct(tpots,0.5):7.1f} / {pct(tpots,0.99):7.1f} ms")
    print(f"E2E  P50/P99 : {pct(e2es,0.5):7.1f} / {pct(e2es,0.99):7.1f} ms")

if __name__ == "__main__":
    asyncio.run(main())
```

读这段要抓住四个"测准"的要点(每个都对应一个高频翻车点)：

1. **流式(`stream=True`)才能测 TTFT**:首个 chunk 到达的时刻就是首 token 时刻;非流式只能等整个响应回来,测不到 TTFT。
2. **`ignore_eos=True` 锁死输出长度**:否则不同配置生成长度不同,吞吐对比不公平(第 2 节的坑)。
3. **算分位数而非均值**:`pct(...,0.99)` 才暴露长尾,均值会骗你。
4. **要有 warmup**:正式测前先发几十个请求把服务"热"起来(CUDA graph/缓存/JIT 预热),否则前几个请求的冷启动会污染数字——上面骨架为简洁省略了,实测务必加。

## 九、设计权衡与常见坑(全章纪律的反面清单)

**纪律性的坑(最致命,因为它们让你的结论本身是错的)：**

- **不设基线就改**:没有 step0 的锚点数字,任何"变快了"都无法证伪。**先测,永远先测。**
- **均值掩盖长尾**:只报 P50 或均值,P99 烂了不知道。SLO 是 P99,优化也盯 P99。
- **一次改多个变量**:同时上量化 + 调 batch + 投机,出了变化归不了因,也分不出谁是负优化。**一次一个变量。**
- **不控输出长度**:忘了 `ignore_eos`,吞吐变化里混进长度变化,实验不可比。
- **冷启动污染**:没 warmup,前几个请求的 JIT/缓存冷启动把延迟拉高,数字虚高。
- **不分 batch 制度报结论**:投机解码、量化的收益**强依赖 batch 大小**,只在一个 batch 下测就下"它有用/没用"的结论,是本章最隐蔽的错误。

**技术性的坑(对应前面各章)：**

- **拿 memory-bound 的手段打 compute-bound 的瓶颈**(或反之):瓶颈是 prefill(compute-bound)却猛上 KV 量化,收效甚微;瓶颈是 decode 带宽却去优化 GEMM tiling,打偏了。**先 Roofline 判性质。**
- **量化在小 batch 反而变慢**:INT4 反量化开销 > 带宽收益(第 12 章/5.2)。
- **投机解码在大 batch 负优化**:大 batch 已 compute-bound,多算 K 个不再免费(第 11 章/5.3)。
- **KV 吃满频繁抢占**:`gpu-memory-utilization` 调太高或并发太大,触发 preempt/swap,P99 尖刺(第 10 章/7.4)。看 vLLM 日志里的 Preemption 警告。
- **忽视瓶颈转移**:优化完 decode 不回头重新抓 trace,没发现 prefill 已变成新大头(Amdahl)。

**一条贯穿全章的元纪律**:**任何优化的"有用/没用",都是相对"特定模型 + 特定负载 + 特定 batch 制度"成立的。** 换了负载和制度,结论可能翻转。所以报告里**永远要带上你测试的完整条件**——离开条件谈加速比,等于没说。

## 十、动手练习

**练习 1(完整闭环,核心)。** 在你能拿到的任意一张 GPU 上,用 vLLM 起一个开源模型(Llama 系或任意 ≤8B 模型),按本章流程走一遍完整闭环并**填满第七节报告模板**:(a) 定基线负载(固定 in/out/并发/seed,`ignore_eos`),用第八节骨架或 vLLM 自带脚本测基线吞吐 + TTFT/TPOT 的 P50/P99;(b) 用 nsys 或 torch profiler 抓 trace,回答 4.3 的四个问题,给出一句话瓶颈结论 + Roofline 判定;(c) 至少施加两个优化(从 AWQ / 调并发 / chunked prefill / 投机解码里选),每步前后对照填表;(d) 写结论,指出最终加速比、是否满足你设的 SLO、瓶颈转移到了哪。
- 提示:从"调并发扫描曲线"开始最容易出图(5.4);AWQ 需要现成量化权重;投机解码记得在**小 batch** 下测才看得到收益。

**练习 2(估算题 · Amdahl 排序)。** 你抓 trace 发现:decode 占 75%、prefill 占 20%、sampling 占 5%。手头有两个优化:优化 X 能把 decode 加速 2.5×,优化 Y 能把 prefill 加速 4×。(a) 分别算单独上 X、单独上 Y 的整体加速比;(b) 两个都上的整体加速比;(c) 如果只能先做一个,先做哪个?为什么?(d) 做完 X 之后,重新算各阶段占比,指出新的大头是谁。
- 提示:用 `Speedup = 1/((1−f)+f/s)`;(d) 注意优化后总时间变了,占比要按新的绝对时间重算。

**练习 3(估算题 · Roofline 判方向)。** 某卡峰值算力 P、峰值带宽 B 给定(用你卡的真实值,拿不准就标【待核】并用一组假设值)。(a) 算脊点 I* = P/B;(b) decode 单序列算术强度 ≈ 1 FLOP/Byte(第 11 章),它落在脊点哪侧?该用哪类优化?(c) 若把 batch 增大到 64,decode 的有效算术强度大致变成多少(提示:权重读一次服务 64 个序列)?它还在 memory-bound 区吗?这解释了为什么"增大 batch"本身就是一种打 memory-bound 的优化。
- 提示:大 batch 下算术强度 ≈ batch × 单序列强度(权重复用),据此判断它是否逼近 I*。

**练习 4(分析题 · 设计一个会翻转结论的实验)。** 设计一组实验,证明"投机解码的收益随 batch 增大而衰减直至变负"。要求:(a) 写出固定哪些变量、扫描哪个变量(batch);(b) 每个 batch 点记录哪些指标才能说明问题(至少吞吐 + TPOT + 是否 compute-bound 的判据);(c) 预测曲线形状并用第 11 章的"大 batch→compute-bound→多算 K 个不再免费"机制解释;(d) 说明这个实验对应本章哪条元纪律。
- 提示:判 compute-bound 可用"继续增大 batch 时吞吐是否还涨"——涨说明还没撞算力墙(仍 memory-bound),饱和说明已 compute-bound。

## 十一、源码 / 论文导读

**压测与 profiling 工具:**
- **vLLM benchmark 脚本** —— 仓库 `benchmarks/` 下的 `benchmark_serving.py`(open/closed-loop serving 压测,带分位数)和 `benchmark_throughput.py`(离线吞吐)。**重点读它怎么控输入/输出长度、怎么算 TTFT/TPOT、怎么实现并发控制**——和第八节骨架对照,看工业级压测多了哪些严谨处理(warmup、token 精确计数、采样可复现)。具体路径/参数名以你 clone 的版本为准【待核】。
- **PyTorch Profiler 文档** —— `torch.profiler` 的 `schedule`(wait/warmup/active 分段)、`key_averages().table()` 排序、`export_chrome_trace`。重点理解 `self_cuda_time`(kernel 自身)与 `cuda_time`(含子调用)的区别——读 trace 排大头时别搞反。
- **Nsight Systems 用户手册** —— `nsys profile` 的 `--trace`、`--gpu-metrics-device`,以及 `nsys stats` 的 `cuda_gpu_kern_sum`(kernel 耗时排序,Amdahl 大头清单的直接来源)、`cuda_gpu_trace`。GUI 里重点看时间线上的 GPU 空泡和 NCCL 行(多卡时判通信受限)。

**方法论与回扣各章:**
- **Roofline 模型 —— Williams, Waterman, Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures", CACM 2009。** 本章 4.4 节判瓶颈性质的理论根。读它怎么用"算术强度 vs 脊点"一张图统一表达 compute/memory-bound——这是你一辈子受用的性能直觉。
- **Amdahl 定律** —— 任意体系结构教材(如 Hennessy & Patterson)的开篇。本章 6.1 的"优化收益被未优化部分封顶"就是它,理解"瓶颈会转移"这一推论。
- **回扣前面各章在本章被实测**:第 9 章 FlashAttention(decode attention kernel 的实现,trace 里它通常在 top kernel)、**第 10 章 PagedAttention + 连续批处理 + chunked prefill**(5.1/5.4 实测验证 KV 容量与 P99)、**第 11 章投机解码**(5.3 实测 α、接受长度、大 batch 负优化边界)、**第 12 章量化**(5.2 实测 AWQ 的带宽收益与小 batch 反例)。**这一章就是把前四章的理论拉到示波器下看真实波形**——理论说"应该快 X 倍",实测告诉你"在你的负载下实际快了多少、为什么没到理论值"。

读源码/调服务的方法,和第 10/11 章一样:**先用本章方法论在脑子里建好"该测什么、瓶颈大概在哪、该上哪类优化"的模型,再去 vLLM 的启动参数和 benchmark 脚本里找对应旋钮**。你会发现工业系统的复杂度几乎都在工程细节(参数组合、边界、稳定性),而**指导你拧哪个旋钮的判断力,全部来自前面 15 章的第一性原理**。

## 十二、小结:全课在此收束

这一章没有引入新算法,它做的是把全课的"知识"淬炼成"能力"——**一套对抗"凭感觉优化"的工程纪律**:

1. **先测后改**:基线是一切对比的锚点,没有数字就没有优化。
2. **抓大头(Amdahl)**:只打占比最大的段,收益被未优化部分封顶,且瓶颈会随优化转移——所以是迭代闭环。
3. **Roofline 指导方向**:先判瓶颈是 compute- 还是 memory-bound,只施加对路的优化,不拿省算力的手段打缺带宽的瓶颈。
4. **每步独立量化、分制度、记负优化**:一次一个变量,前后对照,同一个优化在不同 batch 制度下结论可能相反,负优化也是有价值的结论。
5. **报告模板** 把这五条固化成可交付物:目标→基线→瓶颈→逐步措施与收益→结论。

**在整门课里的位置——这是终点站。** 前 8 章(GPU 体系结构、训练迭代、各种并行、混合精度、重计算)给了你训练侧的系统底座;第 9–12 章(FlashAttention、PagedAttention、投机解码、量化)是推理侧的四大内核;第 13–15 章(MoE、通信与编译、前沿专题)拓宽到稀疏、多卡与前沿。**而这一章 capstone,把推理侧的四大内核拉到一个真实服务上,用 profiler 和 benchmark 验证它们各自到底打的哪个瓶颈、实际收益多少、彼此怎么叠加、在什么制度下失效。**

当你能独立完成这个闭环——拿到一个陌生服务,抓 trace 定位瓶颈,用 Roofline 判方向,逐步施加优化并诚实地量化每步收益,最后产出一份说得清"快在哪、为什么、代价是什么"的报告——**你就具备了在工业界独当一面的 MLSys 推理优化能力**。这不是会背几个优化名词,而是掌握了把任意推理服务"测明白、改对地方、说清楚收益"的方法论。

记住全课最后这句话:**所有优化的价值,都相对"特定模型 + 特定负载 + 特定制度"成立;离开实测条件谈加速比,等于没说。先测,后改,再测——这是 MLSys 工程师对抗自我欺骗的唯一武器。**
