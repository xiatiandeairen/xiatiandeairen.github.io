---
title: "API 工程"
slug: "02"
collection: "ai-app-engineering"
order: 2
summary: "第 1 章把 LLM 拆成\"无状态、按 token 收费、会胡说、有窗口上限\"的远程函数。这一章把它真正接进你的系统——不是\"能跑通一次 demo\",而是\"在 429、超时、断流、限流、成本失控这些真实工况下，它还得是一个你敢挂在生产链路上的依赖\"。"
topics:
  - "AI 应用工程"
tags: []
createdAt: "2026-06-11T06:53:40.000Z"
updatedAt: "2026-06-11T06:53:40.000Z"
---
> 阶段〇 · 底座 · 第 2 章
>
> 第 1 章把 LLM 拆成"无状态、按 token 收费、会胡说、有窗口上限"的远程函数。这一章把它**真正接进你的系统**——不是"能跑通一次 demo",而是"在 429、超时、断流、限流、成本失控这些真实工况下，它还得是一个你敢挂在生产链路上的依赖"。
>
> 一句话定位：**第 1 章告诉你这个组件的物理特性，第 2 章教你把它当一个不可靠的远程 RPC 来治理。** 你过去十年对付下游服务抖动的全部手艺——超时、重试、退避、限流、连接池、幂等、降级——一条都没浪费，全都要重新焊到这个又贵又慢还偶尔说谎的新依赖上。这一章是主线项目从 P0 走向 P1 的地方：给裸 API 助手装上基础设施层。

读这章前，先合上文档问自己三个问题：

1. 多轮对话在 HTTP 层到底发生了什么？为什么说"模型不记得上一句"？
2. 一个 LLM 请求可能以多少种方式失败？哪些该重试，哪些重试就是烧钱？
3. 同样一段 50KB 的 system prompt，发 100 次，你为它付了 100 遍的钱吗？

答得上来，这章你只需要查漏补缺；答不上来，这章就是你和"只会 `client.messages.create()` 的人"之间的护城河。

---

## 一、messages 结构与角色：先把数据模型刻进脑子

### 背景：为什么是 messages，而不是一个 prompt 字符串

早期的 completion API（GPT-3 时代）是一个纯字符串进、纯字符串出的接口：你拼一大段文本，模型续写。这个模型有个致命问题——**没有结构**。你没法清晰区分"这是系统指令""这是用户说的""这是模型上一轮的回复""这是工具返回的结果"，全靠你自己用 `\n\nHuman:` `\n\nAssistant:` 这种约定俗成的分隔符硬拼。约定一旦写错一个字节，模型行为就漂。

Chat / Messages API 把这层结构**显式化**了。现在一次请求的核心是一个 `messages` 数组，每个元素是一个带 `role` 的对象。这不是 API 设计的审美选择，是**把"对话状态"这个概念提升为一等公民**——它直接决定了你后面所有的上下文管理、工具调用、缓存策略怎么写。

### 现状：三种角色，各管一摊

以 Anthropic Messages API 为例（OpenAI 结构几乎同构，差异后面单列）：

| 角色 | 谁在说话 | 权威级别 | 典型内容 |
|---|---|---|---|
| `system` | 你（开发者） | 最高，定义模型的人格/规则/约束 | "你是一个严谨的代码审查助手，只输出 JSON" |
| `user` | 终端用户（或你代用户注入的上下文） | 中，是任务输入 | "帮我审查这段代码" + 检索到的文档 |
| `assistant` | 模型自己 | 模型的历史输出，多轮时回填 | 上一轮模型说的话、tool_use 块 |

注意一个**容易被新手忽略的结构细节**：在 Anthropic API 里，`system` **不是 `messages` 数组里的一个元素**，而是请求体顶层的一个独立字段。

```python
client.messages.create(
    model="claude-opus-4-8",
    max_tokens=16000,
    system="你是一个严谨的代码审查助手。",   # ← 顶层字段，独立于 messages
    messages=[
        {"role": "user", "content": "审查这段代码"},
    ],
)
```

这跟 OpenAI 不同——OpenAI 把 system 当作 `messages[0]`（`{"role": "system", ...}`）。**这个差异会直接坑到你两个地方**：

1. **跨 provider 抽象层**：如果你写一个统一网关（P4 会做），system 在两家的位置不同，转换层必须处理。
2. **prompt caching 的渲染顺序**：Anthropic 的缓存前缀渲染顺序是 `tools → system → messages`，system 在 messages 之前。理解这个顺序，是后面缓存断点放置的前提（见本章第六节）。

> **工程推论（每条都能独立成面试题）：**
>
> 1. **system prompt 是"程序"，user/assistant 是"运行时输入"。** 把可复用的指令、人格、输出格式约束放 system；把每次都变的任务数据放 user。混了，缓存命中率和行为稳定性同时崩——因为 system 在缓存前缀的最前面，动一个字节，后面全失效。
> 2. **`content` 可以是字符串，也可以是 block 数组。** 纯文本时是 `"content": "..."`；多模态、工具结果、需要精细化缓存控制时是 `"content": [{"type": "text", ...}, {"type": "image", ...}]`。生产代码里建议**统一用 block 数组**，因为你迟早要往里塞图片、tool_result、cache_control，提前用数组省得后面大改。
> 3. **consecutive same-role 是允许的，但首条必须是 user。** Anthropic 会把连续的同 role 消息合并成一轮。但 `messages[0]` 必须是 `user`——拿 assistant 开头直接 400。这个约束在你做"对话历史截断"时是个暗坑：截断后如果第一条变成了 assistant，请求会炸。
> 4. **最后一轮 assistant 的 prefill 在新模型上已被废弃。** 老模型可以用"预填 assistant 开头"来强制输出格式（比如塞个 `{` 逼它吐 JSON）。但在 Fable 5 / Opus 4.6/4.7/4.8 / Sonnet 4.6 上，最后一个 assistant turn 做 prefill 直接 **400**。要强制结构化输出，改用 `output_config.format`（见第 4 章）。这是个高频踩坑点——老教程里的 prefill 技巧在新模型上全废了。

### 演进视角：为什么"多轮"在 HTTP 层是个幻觉

这是第 1 章"无状态"那句话的具体兑现，也是 **P0 验收标准第 1 条**要你能讲清的东西。

服务端**不保存任何对话状态**。你以为的"连续对话"，在 HTTP 层是这样的：

```
第 1 轮：POST /messages  body: [user:"我叫Alice"]
        ← resp: assistant:"你好Alice"

第 2 轮：POST /messages  body: [user:"我叫Alice", assistant:"你好Alice", user:"我叫什么?"]
        ← resp: assistant:"你叫Alice"
```

模型第 2 轮"记得"你叫 Alice，**不是因为它有记忆**，而是因为你把整段历史又重发了一遍。每一轮，你都在把全部历史塞进请求。这件事有三个直接的工程后果，每一个都是 P0/P1 必须处理的：

- **成本随轮数线性甚至超线性增长**：第 N 轮的输入 token ≈ 前 N−1 轮所有内容之和。10 轮对话，第 10 轮可能要重发前面 9 轮几千个 token。这就是为什么"长对话烧钱"——你不是付了 10 次钱，是付了 `1+2+3+...+10` 次钱。
- **迟早撞上下文窗口上限**：历史无限增长，1M 窗口也有撑爆的一天。所以 **P0 验收第 4 条**要求超窗有降级策略（截断或摘要），不能直接崩。
- **prompt caching 成为刚需而非优化**：既然每轮都重发大量相同前缀（system + 早期历史），不缓存就是纯烧钱。这是第六节的核心。

> **心法：** LLM 的 API 是 RESTful 的——**无状态、每次自包含**。"会话"是你在客户端用一个数组维护出来的幻觉。谁维护这个数组、怎么截断、怎么缓存，就是 AI 应用工程师的核心工作之一。模型只是个纯函数：`f(完整历史) → 下一句`。

---

## 二、Streaming / SSE：长输出为什么必须流式

### 背景：一个 10 分钟的 HTTP 请求意味着什么

LLM 是**自回归**的：一个 token 一个 token 地生成，前一个 token 是后一个的输入（第 1 章讲过）。生成 4000 个 token，就是串行跑 4000 次前向。这个过程**慢**——几十秒到几分钟很正常。

如果你用非流式（一次性等完整响应），你的 HTTP 连接就得**挂着等这几分钟**。问题来了：

- **用户体验**：用户盯着转圈几十秒，什么都看不到，体感等于卡死。
- **连接超时**：HTTP 客户端、反向代理、负载均衡器都有空闲超时。一个挂着不传数据的连接，很容易被中间某一跳掐断。Anthropic SDK 默认请求超时是 **10 分钟**，而且——**这是个关键护栏**——当你设了一个很大的 `max_tokens`（SDK 估算可能超过 ~10 分钟）却没开流式时，**SDK 会直接抛 `ValueError` 拒绝发请求**，因为它知道这个请求大概率会因为连接空闲被掐断。

### 现状：SSE 是怎么工作的

流式用的是 **Server-Sent Events（SSE）**——一个基于 HTTP 的单向服务器推送协议。服务端不等生成完，而是**边生成边把 token 一块块推过来**。一次流式响应是一串事件：

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_...","usage":{...}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"你"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"好"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":12}}

event: message_stop
data: {"type":"message_stop"}
```

把这串事件类型刻进脑子，这是 **P0 验收第 2 条（streaming + 断流处理）**的考点：

| 事件类型 | 何时触发 | 你要从里面拿什么 |
|---|---|---|
| `message_start` | 开头一次 | 消息元数据、**初始 usage（input_tokens）** |
| `content_block_start` | 每个内容块开始 | 块类型：是 text？thinking？tool_use？ |
| `content_block_delta` | 每个 token/片段 | **增量文本**（`text_delta` / `thinking_delta` / `input_json_delta`） |
| `content_block_stop` | 块结束 | — |
| `message_delta` | 消息级更新 | **`stop_reason` 和最终 output_tokens** |
| `message_stop` | 结尾一次 | 完成信号 |

注意两个新手第一误区：

- **token 计数分散在事件里**：`input_tokens` 在 `message_start`，`output_tokens` 在 `message_delta`。要做成本归因，这两个地方都得捞。
- **一次响应可能有多种 block**：开了 thinking（见下文）就会先来 `thinking` 块再来 `text` 块；带工具就会有 `tool_use` 块。`content_block_delta` 里的 `delta.type` 告诉你这是哪种增量，别只处理 `text_delta`。

### 落地：别手写 SSE 解析，用 SDK 的兜底

手写 SSE 解析（split `\n\n`、解析 `event:` / `data:` 行）是新手最容易翻车的地方——JSON 跨行、心跳包、半个 chunk，处理不好就乱。**生产里直接用 SDK 的流式 helper**，它替你累积状态。

Python 的推荐写法，注意最后的 `get_final_message()` 兜底：

```python
with client.messages.stream(
    model="claude-opus-4-8",
    max_tokens=64000,
    messages=[{"role": "user", "content": "写一个长故事"}],
) as stream:
    for text in stream.text_stream:        # 边到边打印，给用户即时反馈
        print(text, end="", flush=True)    # flush=True 是关键，否则缓冲区憋着不显示

    # 即使是流式，也能在结束后拿到完整 Message 对象
    final = stream.get_final_message()      # 兜底：拿完整内容 + usage
    print(f"\n用了 {final.usage.output_tokens} 个输出 token")
```

`get_final_message()`（TS 是 `finalMessage()`）是**默认流式 + 兜底**这个最佳实践的核心：你既享受了流式的"不超时、有即时反馈"，又不用自己处理一堆增量事件——SDK 把碎片帮你拼回完整对象。**如果你不需要处理单个 token，只想要最终结果还要规避超时，就用这个模式。**

> **工程推论：**
>
> 1. **流式不只是 UX，是超时防护。** 很多人以为流式是为了"打字机效果好看"。错。它最硬的价值是**保持连接活跃、规避 idle timeout**。一个跑 3 分钟的非流式请求，随时可能被某一跳代理掐断；流式因为持续有字节流动，连接不会被判定为 idle。**`max_tokens` 大（>16K）就必须流式，这是规则不是建议。**
> 2. **断流必须可恢复。** P0 验收明确要求"处理中途断流"。SSE 没有内置重放——连接断了，你已经收到的是半截内容。最低限度：捕获 `APIConnectionError`，决定是重试整个请求，还是把已收到的部分交给用户并标注"已截断"。**永远不要让一次断流把整个进程带崩。**
> 3. **`flush=True` / `process.stdout.write` 是必须的。** 不刷新缓冲区，token 会憋在标准输出缓冲里，用户看到的还是"先卡半天，然后唰一下全出来"——流式效果归零。这是个一行代码的暗坑。
> 4. **Web 场景要做 token 缓冲。** 浏览器里如果每个 token 都触发一次 DOM 更新，会把渲染打爆。攒几个 token 再渲染一次，平衡即时感和性能。
> 5. **`thinking` 内容在新模型上默认不显示。** Fable 5 / Opus 4.7 / 4.8 默认 `thinking.display` 是 `"omitted"`——`thinking` 块照样在流里出现，但文本是空的。如果你在 UI 里给用户展示"思考过程"，会发现输出前有一段莫名其妙的长暂停（模型在想，但你没收到文本）。要恢复可见进度，显式设 `thinking={"type": "adaptive", "display": "summarized"}`。这是个静默变化，不报错，排查起来很费时间。

---

## 三、同步 / 异步 / 并发：把吞吐做起来

### 背景：单条请求慢，那批量怎么办

单条 LLM 请求慢是物理事实，改不了。但你的系统经常要**同时处理很多条**：批量给 1000 篇文档打标签、给一批用户消息并行生成回复。这时候性能瓶颈不是单条多快，而是**并发度**——同一时刻能压多少条在飞。这正是 **P1 增量**要给 P0 加上的能力之一：并发批处理。

### 演进：从串行 for 循环到 asyncio

最朴素的写法是同步串行：

```python
results = []
for doc in docs:                          # 1000 篇，一篇一篇来
    results.append(summarize(doc))        # 每篇等 30 秒 → 总共 30000 秒 ≈ 8.3 小时
```

这慢得离谱，但慢的原因很关键：**瓶颈是 I/O 等待，不是 CPU**。每条请求的 30 秒里，你的进程绝大部分时间在干等网络响应，CPU 闲着。这正是 **async I/O 的主场**——一个线程内，趁一条请求在等网络，去发下一条。

```python
import asyncio
from anthropic import AsyncAnthropic

client = AsyncAnthropic()

async def summarize(doc):
    resp = await client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"总结：{doc}"}],
    )
    return next((b.text for b in resp.content if b.type == "text"), "")

async def main(docs):
    return await asyncio.gather(*[summarize(d) for d in docs])  # 全部并发
```

理论上 1000 条同时飞，总耗时 ≈ 最慢那一条 ≈ 几十秒。**但你绝不能真这么写**——下一节解释为什么。

### 现状：并发的两个必加约束——信号量限流 + 连接池

直接 `asyncio.gather` 一千条会立刻撞墙：

- **撞服务端限流**：瞬间一千个请求，分分钟 429（见第五节）。
- **撞本地资源**：一千个并发连接，把本地文件描述符、内存、连接池打爆。

所以**生产级并发 = 受控并发**。锚点类比你肯定熟：这跟你给下游服务做并发调用时加**信号量 / 限流器 / 连接池**是一模一样的工程，只是被调方换成了 LLM。

```python
import asyncio
from anthropic import AsyncAnthropic

client = AsyncAnthropic()
SEM = asyncio.Semaphore(10)               # ← 最多 10 条在飞，这是闸门

async def summarize(doc):
    async with SEM:                        # 进闸：超过 10 条的在这里排队等
        resp = await client.messages.create(
            model="claude-opus-4-8",
            max_tokens=1024,
            messages=[{"role": "user", "content": f"总结：{doc}"}],
        )
        return next((b.text for b in resp.content if b.type == "text"), "")

async def main(docs):
    return await asyncio.gather(*[summarize(d) for d in docs])
```

`Semaphore(10)` 把并发死死压在 10 条以内。这个数字怎么定？看你的 **RPM/TPM 配额**（第五节）和**单条请求的 token 量**，不是拍脑袋。配额低就调小，配额高、请求小就调大。

关于**连接池**：Anthropic Python SDK 底层用 `httpx`，自带连接池和合理的默认超时。高并发 async 场景，SDK 还支持换 `aiohttp` 后端提升性能（`pip install anthropic[aiohttp]`，传 `DefaultAioHttpClient`）。**关键纪律：永远用 SDK 的 `DefaultHttpxClient` / `DefaultAioHttpClient`，不要自己 new 一个裸 `httpx.Client`**——裸客户端会丢掉 SDK 调好的超时和连接数默认值，埋下连接泄漏的雷。

> **工程推论：**
>
> 1. **CPU 密集用多进程，I/O 密集用 async——LLM 调用是纯 I/O 密集。** 别用多线程去并发 LLM 请求（Python 的 GIL 让多线程在 CPU 上没意义，而这里 CPU 又不是瓶颈）；asyncio 单线程 + 信号量就是最优解。这条能区分"懂并发模型"和"会写 `gather`"的人。
> 2. **裸 `gather` = 自杀。** 不加信号量的全并发，要么被服务端 429 打回来重试到天荒地老，要么把本地资源打爆。**受控并发是底线，不是优化。**
> 3. **信号量的值 = min(配额允许, 本地资源允许)。** 这个数要根据 RPM/TPM 反推，并留 buffer。压满配额是危险的——突发流量没有余量就直接 429。
> 4. **per-request override 比改全局 client 更安全。** SDK 支持 `client.with_options(timeout=5.0, max_retries=5).messages.create(...)`，给单次调用临时改超时/重试，不污染全局 client。批处理里不同任务要不同超时时，这个比 new 多个 client 干净。

---

## 四、重试与指数退避：哪些错误该重试

### 背景：分布式系统的老道理，换个被调方

LLM API 是网络服务，网络服务会**瞬时失败**——限流、服务端抖动、网络闪断。这些大多是**暂时的**，重试一下就好。这套逻辑你在治理下游服务时早就烂熟，现在原样搬过来，只是要搞清楚**LLM API 特有的"哪些该重试"**。

### 现状：错误码体系决定重试策略

这是面试高频考点——**给一个错误码，能不能立刻说出该不该重试、为什么**：

| HTTP 码 | 错误类型 | 该重试？ | 原因 |
|---|---|---|---|
| 400 | `invalid_request_error` | ❌ | 请求本身错了（格式/参数），重试 100 次还是错 |
| 401 | `authentication_error` | ❌ | API key 无效，重试无意义 |
| 403 | `permission_error` | ❌ | 没权限，换 key 或申请权限，不是重试 |
| 404 | `not_found_error` | ❌ | 模型 ID 拼错/端点错（如 `claude-sonnet-4.6` 写成带点的） |
| 413 | `request_too_large` | ❌ | 请求体超限，得缩小输入，不是重试 |
| 429 | `rate_limit_error` | ✅ | 限流——退避后重试，**读 `retry-after`** |
| 500 | `api_error` | ✅ | 服务端内部错误，重试 |
| 529 | `overloaded_error` | ✅ | 服务过载，退避后重试 |

**心法一句话：4xx（除 429）是你的错，别重试；429 和 5xx 是暂时的，退避重试。** 把 400 这种确定性错误拿去重试，纯属浪费配额和时间，还可能掩盖真正的 bug。

### 落地：指数退避 + jitter，以及"SDK 已经替你做了"

**第一个反常识的事实：Anthropic SDK 默认已经自动重试** 连接错误、408、409、429、≥500，用指数退避，默认重试 2 次。也就是说——**大多数情况你什么都不用写**，只在需要超出 SDK 默认行为时才自定义。

什么时候要自定义？比如你想要更多重试次数、想接入自己的监控、想对不同错误用不同策略。手写的标准形态长这样：

```python
import time, random, anthropic

def call_with_retry(client, max_retries=5, base_delay=1.0, max_delay=60.0, **kwargs):
    last_exc = None
    for attempt in range(max_retries):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            last_exc = e
            # 429 优先读服务端给的 retry-after，它比你的退避公式更准
            retry_after = e.response.headers.get("retry-after")
            if retry_after:
                time.sleep(int(retry_after))
                continue
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                last_exc = e                          # 5xx 重试
            else:
                raise                                 # 其他 4xx 直接抛，不重试
        # 指数退避 + jitter：2^attempt 秒，再加 0~1 秒随机抖动，封顶 max_delay
        delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
        time.sleep(delay)
    raise last_exc
```

两个**必须讲清的点**：

**为什么要 jitter（随机抖动）？** 如果 100 个客户端同时被 429，又同时按 `1s, 2s, 4s` 退避，它们会在同一时刻**整齐划一地重试**——形成"惊群"，把刚恢复的服务端再次打死，然后再次整齐退避，进入死循环。加一个随机 jitter，把重试时间打散，避免同步惊群。**这是退避算法里最容易被漏掉、面试官最爱问的细节。**

**为什么 429 要优先读 `retry-after`？** 服务端通过 `retry-after` 头明确告诉你"等 N 秒再来"。这个值比你本地的 `2^attempt` 公式更权威——它知道自己什么时候恢复。盲目用本地退避公式，要么等太久（浪费），要么等太短（又被打回）。

### 落地：幂等性——重试的前提

重试有个**沉默的前提**：被重试的操作必须**幂等**——重复执行多次，效果跟执行一次相同。

- **纯文本生成**（总结、翻译、问答）天然幂等：重发一次，无非再生成一遍，没有副作用，重试无害。
- **带副作用的工具调用**（发邮件、扣款、写数据库）**不幂等**：盲目重试可能发两封邮件、扣两次款。

所以**重试逻辑要包在哪一层，是个架构决策**：包在"纯 LLM 调用"这层安全；一旦工具调用产生了真实副作用（第 3 章 Function Calling 会做），重试就必须配合幂等键（idempotency key）或去重机制，否则重试本身会制造数据灾难。

> **工程推论：**
>
> 1. **错误分类是第一性的，重试策略从分类推导。** 不先把错误分成"我的错 / 暂时的错"，重试逻辑就是瞎写。给个错误码能秒答该不该重试——这是 AI 工程师的基本功，也是面试官区分"读过文档"和"上过线"的第一刀。
> 2. **永远用 SDK 的 typed exception，不要字符串匹配错误信息。** `except anthropic.RateLimitError` 而不是 `if "429" in str(e)`。字符串匹配脆弱、会随文案变化失效，是新手代码的典型异味。每个 SDK 都有 typed exception 类（`BadRequestError` / `RateLimitError` / `APIStatusError` …），还有 `.type` 字段做更细粒度分类（比如区分 `billing_error` 和 `permission_error`，两者都是 403）。
> 3. **jitter 不是可选项。** 没有 jitter 的指数退避在高并发下会制造惊群，把退避变成集体自杀。这条几乎是退避算法的必考题。
> 4. **重试的前提是幂等。** 纯生成可以无脑重试；带副作用的操作必须先解决幂等再谈重试。把这两者混为一谈，会在 Agent 阶段（工具有副作用）十倍奉还。
> 5. **`max_retries` 要有上限，退避要封顶。** 无限重试会在服务端长时间故障时把你自己拖死，还可能持续烧配额。给次数上限、给单次延迟封顶（`max_delay`），失败了就降级而不是死磕。

---

## 五、限流：RPM / TPM 与 429 的正确姿势

### 背景：配额是双维度的

LLM API 的限流不是单一维度。它至少卡两个量，这是和普通 Web API 限流最不一样的地方：

- **RPM**（Requests Per Minute）：每分钟请求数。
- **TPM**（Tokens Per Minute）：每分钟 token 数（输入+输出）。还可能有 **TPD**（每天 token 数）。

**为什么 TPM 这个维度容易被忽略、又最容易先撞？** 因为 LLM 请求的 token 量差异巨大——一条带 50KB 上下文的请求，顶得上几百条短请求的 token。你可能 RPM 还很宽裕（请求数不多），但 TPM 已经爆了（每条都很大）。**新手只盯着 RPM，结果被 TPM 限流打懵。**

### 现状：429 响应里的信息金矿

撞限流时返回 429，响应头里有一组关键信息，**会读这些头 = 会做精细限流控制**：

- `retry-after`：等多少秒再重试（第四节讲过，优先用它）。
- `x-ratelimit-limit-*`：你的配额上限（requests / tokens 分别有）。
- `x-ratelimit-remaining-*`：剩余配额。

读 `remaining`，你甚至能在**撞墙之前**主动降速——剩余配额快见底了，主动把并发降下来，而不是等 429 了再被动退避。

### 落地：限流控制的三层做法

1. **被动层（最低保障）**：429 来了，读 `retry-after`，退避重试。SDK 默认就做这层。
2. **主动层（信号量限流）**：用第三节的 `Semaphore` 把并发压在配额能承受的范围内，从源头减少 429。信号量的值要根据 RPM **和** TPM 综合反推。
3. **预测层（高级）**：读 `x-ratelimit-remaining-*`，动态调整发送速率——配额充裕时压满，临近上限时主动降速。这是网关层（P4）才需要的精细度。

国内厂商语境补一句：豆包 / Qwen / DeepSeek / Kimi / GLM / 混元 / 文心这些 API，限流维度和配额申请方式各有差异（有的还有 QPS、并发数单独限制），但**"双维度配额 + 429 退避 + 主动降速"这套方法论是通用的**。换 provider 时，先去查它的限流文档把 RPM/TPM/QPS/并发数四个数搞清楚，方法不变。

> **工程推论：**
>
> 1. **TPM 通常比 RPM 先爆——盯错维度会误判。** 大上下文请求吃 token 凶。监控和限流必须同时看 RPM 和 TPM，只盯一个会在另一个维度上猝死。
> 2. **限流是"分层防御"，不是单点。** 信号量从源头削峰（主动），retry-after 退避兜底（被动），读 remaining 预测降速（高级）。三层叠起来才稳。只靠退避是被动挨打。
> 3. **`retry-after` > 本地退避公式。** 服务端最知道自己何时恢复。能读到 `retry-after` 就别用 `2^attempt`。
> 4. **配额是可申请的资源，要纳入容量规划。** 上线前按预估 QPS × 平均 token 量算出需要的 TPM，对照当前 tier 配额，不够就提前申请升配——别等线上 429 了才发现配额不够（这是 P5 容量估算要做的事）。

---

## 六、成本模型与 Prompt Caching：这章最值钱的一节

成本思维贯穿全课（这是课程主线之一）。这一节是你作为后端老手最能甩开"只会调 API 的人"的地方——**模型能力是别人的，成本治理是你的工程**。

### 背景：token 计费与那个致命的不对称

LLM 按 **token** 计费，输入和输出**分开定价**，而且——记住这个**贯穿全课的关键不对称**——**输出价约等于输入价的 5 倍**。

锚点价格（每百万 token，按 2026 初版图，名单会变，请学方法论）：

| 梯队 | 代表模型 | 输入 $/1M | 输出 $/1M | 备注 |
|---|---|---|---|---|
| Frontier | claude-opus-4-8 | $5 | $25 | 输出是输入 5 倍 |
| Frontier | claude-fable-5 | $10 | $50 | 最强，更贵 |
| Balanced | claude-sonnet-4-6 | ~$3 | ~$15 | 高并发生产主力 |
| Fast | claude-haiku-4-5 | ~$1 | ~$5 | 简单/低延迟任务 |

**这个 5 倍不对称直接决定你的优化优先级：省输出比省输入划算得多。** 让模型"少废话、直接给结论"不只是 UX，是省钱。一个让模型啰嗦的 prompt，是在按 5 倍价格烧钱。

**怎么数 token？** 别用 `tiktoken`——那是 OpenAI 的 tokenizer，估 Claude 会**系统性偏差 15-20%**（代码和非英文偏得更多）。Claude 没有公开的 tokenizer，要准确计数必须调 `count_tokens` API：

```python
resp = client.messages.count_tokens(
    model="claude-opus-4-8",
    system=system,
    messages=messages,
)
print(resp.input_tokens)        # 发请求前先估成本，能算账才能控成本
```

### 现状：Prompt Caching——前缀复用，省钱杀手锏

回到第一节那个事实：多轮对话每轮都重发大量相同前缀（system + 早期历史）；批处理里每条请求共享同一大段 system。**这些重复的前缀，不缓存就是一遍遍全价付费。** Prompt caching 就是来解决这个的，最高能省 **90%**。

**一个不变量统摄一切（把这句刻进脑子）：Prompt caching 是前缀匹配。前缀里任何一个字节变化，都会让它之后的全部缓存失效。**

缓存 key 由"渲染后的 prompt 到每个 `cache_control` 断点为止的精确字节"决定。渲染顺序固定是 **`tools → system → messages`**。所以设计 prompt 组装代码时，必须**把稳定内容放前面、易变内容放后面**：

```python
response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=16000,
    system=[{
        "type": "text",
        "text": LARGE_STABLE_SYSTEM_PROMPT,        # 大段稳定前缀
        "cache_control": {"type": "ephemeral"},    # ← 断点：缓存到这里为止
    }],
    messages=[{"role": "user", "content": user_question}],  # 易变内容在断点之后
)
```

**经济学账要算清**（这是面试加分点）：

- 缓存**读**：约 **0.1×** 基础输入价（省 90%）。
- 缓存**写**：**1.25×**（5 分钟 TTL）/ **2×**（1 小时 TTL）——写比正常贵。
- **盈亏平衡**：5 分钟 TTL 下，**两次请求就回本**（1.25× 写 + 0.1× 读 = 1.35×，对比不缓存的 2×）；1 小时 TTL 下，需要**至少三次**才回本（2× + 0.2× = 2.2×，对比 3×）。

所以缓存不是无脑开——**前缀太短会静默不缓存**（不报错，`cache_creation_input_tokens` 直接是 0）；前缀每次都变的，开了缓存只付写的钱、零读，纯亏。这个"最小可缓存前缀"**因模型而异，是个真会坑人的细节**：Opus 4.8/4.7/4.6/4.5 和 Haiku 4.5 是 **4096 token**，Fable 5 和 Sonnet 4.6 是 **2048**，Sonnet 4.5 这档是 **1024**。同一段 3K token 的前缀，在 Sonnet 4.5 上能缓存、切到 Opus 4.8 就**静默失效**——迁移模型时这是个隐形回退点。

**怎么验证缓存命中？** 看响应的 `usage`：

```python
print(response.usage.cache_creation_input_tokens)  # 写入缓存的 token（付 ~1.25×）
print(response.usage.cache_read_input_tokens)      # 命中缓存的 token（付 ~0.1×）
print(response.usage.input_tokens)                 # 未缓存的全价 token
```

**如果重复发相同前缀，`cache_read_input_tokens` 始终是 0，就一定有"静默击穿"。**

### 落地：静默击穿排查清单——这是真正的暗坑区

缓存最阴的地方是**击穿了不报错**——它只是默默不命中，你的账单悄悄翻倍，还以为缓存生效了。在喂进 prompt 前缀的代码里，grep 这些杀手：

| 模式 | 为什么击穿缓存 |
|---|---|
| system prompt 里嵌 `datetime.now()` / `Date.now()` | 每次请求前缀都变，整个缓存失效 |
| 早期内容里有 `uuid4()` / 请求 ID | 同理，每次都唯一 |
| `json.dumps(d)` 没加 `sort_keys=True` / 迭代 `set` | 序列化顺序不确定，前缀字节不同 |
| system 里 f-string 插了 session/user ID | 每个用户一个前缀，跨用户不共享缓存 |
| 条件式 system 段落（`if flag: system += ...`） | 每种 flag 组合是一个独立前缀 |
| `tools=build_tools(user)` 工具集随用户变 | tools 在位置 0，工具一变全员不缓存 |

> **新手第一误区：在 system prompt 里塞时间戳/UUID。** "当前时间是 X"看起来无害，但它在前缀最前面，每秒都变，让后面所有缓存全失效。动态内容（时间、模式、用户名）要往后放——放进 `messages` 的后段，而不是 system 的开头。

几条**架构层面比断点位置更重要**的纪律：

1. **冻结 system prompt。** 别往 system 里插任何每次都变的东西。
2. **别中途换 tools 或 model。** tools 渲染在位置 0，增删改一个工具，整个缓存崩；切模型缓存也作废（缓存是 model-scoped 的）。要"模式切换"别换工具集，用消息内容传模式。
3. **序列化要确定性。** JSON `sort_keys=True`，别迭代无序集合。
4. **fork 操作要复用父请求的精确前缀。** 摘要、压缩、子 agent 这些副链路如果重建 system/tools，会完全错过父请求的缓存。

### 落地：三个"前缀字节没变也命中不了"的进阶坑

前面讲的击穿都是"前缀字节变了"。但还有三类坑——**前缀字节一个没动，缓存照样不命中**。这三条是区分"读过缓存文档"和"线上调过缓存命中率"的分水岭，新手几乎必栽：

**1. 失效是分层的——不是任何参数一变就全崩。** 缓存有三个 tier：`tools` / `system` / `messages`，改动只作废它**自己这一层及之后**，前面的层照旧命中。这意味着一批你以为"会毁缓存"的操作其实是安全的：

| 改了什么 | tools 缓存 | system 缓存 | messages 缓存 |
|---|:---:|:---:|:---:|
| tools 定义（增删/改/重排） | ❌ | ❌ | ❌ |
| 切 model | ❌ | ❌ | ❌ |
| system prompt 内容 | ✅ 命中 | ❌ | ❌ |
| `tool_choice`、`thinking` 开关、加图片 | ✅ | ✅ | ❌ |
| message 内容（追加新一轮） | ✅ | ✅ | ❌ |

> 实战含义：**你可以每次请求改 `tool_choice`、开关 `thinking`，而不丢 tools+system 缓存。** 别为这些杯弓蛇影。真正会逼全量重建的只有"动 tools 定义"和"切 model"两件事。

**2. 20-block 回溯窗口——长 turn 的隐形杀手。** 每个断点向前**最多回溯 20 个 content block** 去找上一次的缓存条目。在 Agent 循环里，一轮如果塞了超过 20 个 block（大量 `tool_use` / `tool_result` 成对出现，几轮工具调用就破 20），下一次请求的断点就**够不着上一轮缓存了，静默 miss**——前缀一字节没变，命中率却崩。

- 排查特征：`cache_read_input_tokens` 在短对话里正常，一进入多工具 Agent 循环就掉到 0。
- 修法：长 turn 里每 ~15 个 block 补一个**中间断点**（最多 4 个断点，省着用），让断点始终落在上一个缓存点的 20 block 之内。这条 PRD 没写、文档藏得深，但 Agent 阶段（P3+）必踩。

**3. 并发同前缀互相打不到缓存——fan-out 的隐藏全价。** 缓存条目要等**第一个响应开始 streaming 之后**才可读。所以你 `gather` 出去的 N 条同前缀请求，**谁也读不到别人正在写的缓存，全部按全价付写入**——这跟第三节那个"批量打标共享 system"的场景直接撞车：你以为缓存会省钱，结果第一批全价。

- 正解（fan-out 预热）：先发 **1 条**，**等它流出第一个 token**（不是等整条响应完），再放出剩下的 N−1 条——它们就能读到第一条刚写好的缓存。
- 更狠的（启动预热）：用 **`max_tokens: 0`** 发一条占位请求，API 只跑 prefill、把缓存写在你的 `cache_control` 断点上，然后立刻返回 `content: []`（不计输出 token，只付一次写入费）。用户的**第一条真请求**就直接命中，省掉冷启动那一下的 TTFT。

```python
# 启动时预热缓存：max_tokens=0，只写缓存不出文，断点放在与真实请求共享的 system 上
client.messages.create(
    model="claude-opus-4-8",
    max_tokens=0,
    system=[{"type": "text", "text": SYSTEM_PROMPT,
             "cache_control": {"type": "ephemeral"}}],
    messages=[{"role": "user", "content": "warmup"}],  # 占位，prefill 时读、永不作答
)
```

> **什么时候值得预热？** 三个条件同时成立才划算：(a) 首请求延迟用户可感知（聊天/语音/交互，不是后台批处理）；(b) 共享前缀够大，冷写明显慢；(c) 真流量到来**之前**有个空窗能发这一发（启动、worker 拉起、部署后、定时窗口开头）。流量连续（请求间隔 < TTL）时别预热——第一条真请求自己就把缓存焐热了，额外发一发纯属多付一次写入费。注意 `max_tokens: 0` 与 `stream=True`、`thinking.type:"enabled"`、`output_config.format`、`tool_choice:{type:"tool"/"any"}` 互斥（会 400）。

> **工程推论：**
>
> 1. **输出比输入贵 5 倍——优化顺序由此决定。** 先压输出（让模型简洁、直接给结论），再谈压输入。这个不对称是成本直觉的地基。
> 2. **缓存是前缀匹配，一个字节定生死。** 这是 prompt caching 唯一需要刻进骨子里的不变量，所有断点放置、静默击穿都从它推导。
> 3. **`cache_read_input_tokens == 0` 是击穿的唯一可靠信号。** 不要凭感觉判断缓存生效，用 usage 验证。重复相同前缀还读不到缓存，就 diff 两次请求的渲染字节找差异。
> 4. **缓存写更贵，少量重复别开。** 两次（5min）/三次（1h）才回本。一次性请求、前缀每次都变的，开缓存是负优化。
> 5. **成本归因要落到每一次调用。** 把每轮的 input / output / cache_read / cache_write token 都记下来，乘以价目表，按用户/功能维度汇总——这就是 P1 要做的成本归因，也是 P4 网关治理的基础。**算不出每次调用花了多少，就谈不上控成本。**
> 6. **静默击穿是高频生产事故。** 团队里有人在 system prompt 里加了个时间戳，全公司缓存命中率归零，账单翻倍，还查不出来——这种事真实发生。排查清单要背下来。
> 7. **失效是分层的，别杞人忧天。** 只有动 tools 定义和切 model 会逼全量重建；改 `tool_choice`、开关 thinking、加图片都只动下层，tools+system 缓存照命中。分不清哪层失效，就会为了"保缓存"把能改的也不敢改。
> 8. **`input_tokens` 不是总输入量。** 它只是"全价未缓存"那部分。总前缀 = `input_tokens + cache_creation + cache_read`。看到 Agent 跑了俩小时 `input_tokens` 才 4K 别惊讶——剩下全被缓存读了，要看三者之和。
> 9. **并发同前缀≠自动省钱。** N 条 `gather` 出去的同前缀请求互相读不到对方正在写的缓存，全按全价写。fan-out 要先发一条、等它流出首 token 再放剩下的；交互式首请求延迟敏感就用 `max_tokens:0` 启动预热。这条最能体现"在生产里真调过缓存命中率"。

### 落地补充：Batch API——非实时场景直接砍半

如果你的任务**不在乎实时性**（离线批量打标、夜间报表、大规模数据清洗），用 **Batch API**：异步提交，**50% 价格**，单批最多 10 万请求 / 256MB，多数 1 小时内完成（上限 24 小时），结果保留 29 天。

```python
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

batch = client.messages.batches.create(requests=[
    Request(custom_id=f"doc-{i}", params=MessageCreateParamsNonStreaming(
        model="claude-opus-4-8", max_tokens=1024,
        messages=[{"role": "user", "content": f"总结：{doc}"}],
    )) for i, doc in enumerate(docs)
])
# 轮询 batch.processing_status 直到 "ended"，再取结果
```

而且 Batch **支持 prompt caching**——批量请求共享同一大段 system 时，缓存 + 批量五折**叠加省钱**。

**判断规则：能接受最长 24 小时延迟的批量任务，一律走 Batch，白省 50%。** 需要实时交互的（聊天、Agent 循环）才用同步/流式。这是个一眼就能下的成本判断。

---

## 七、超时与 max_tokens：两个容易拍脑袋的默认值

### max_tokens：不是越大越好，也别太小

`max_tokens` 是**单次响应输出的硬上限**。两个方向都有坑：

- **太小**：模型还没说完就被截断（`stop_reason: "max_tokens"`），输出半截，得加大重试，白烧一次钱。
- **太大且非流式**：撞 SDK 的超时护栏。

推荐默认值（**记住这组数**）：

- **非流式**：默认 `~16000`，保证响应在 SDK HTTP 超时内。
- **流式**：默认 `~64000`，超时不是问题，给模型充分空间。
- **分类等确定短输出**：`~256`，省钱。
- **128K 超长输出**：Fable 5 / Opus 4.6/4.7/4.8 支持 128K，但这么大**必须流式**（否则 HTTP 超时），用 `.stream()` + `get_final_message()`。

注意 `max_tokens` 和**新模型的 `task_budget`（beta）**是两回事：`max_tokens` 是强制的、模型不知道的每响应天花板；`task_budget` 是告诉模型"整个 agentic 循环你有这么多 token 预算"，模型能看到倒计时并自我节制（Agent 阶段才用，最小 2 万 token）。

### 超时：默认 10 分钟，按场景收紧

Anthropic SDK 默认请求超时 **10 分钟**。可以传 float（秒）或 `httpx.Timeout` 做精细控制（连接/读/写分开设）。超时会抛 `APITimeoutError`，并按 `max_retries` 自动重试。

```python
import httpx
client = anthropic.Anthropic(timeout=20.0)                       # 整体 20 秒
client = anthropic.Anthropic(
    timeout=httpx.Timeout(60.0, read=5.0, connect=2.0),          # 精细：连接 2s，读 5s
)
```

**一个要命的认知误区（面试加分点）：`httpx` 的 `read` 超时是"每个 chunk 之间"的超时，不是"整个请求"的墙钟超时。** 流式响应只要持续有字节流动，`read` 超时永远不触发——一个细水长流的连接可以挂到天荒地老。要硬性的总时长上限，得在循环层面自己用 `time.monotonic()` 记时并主动 break，或者用 `asyncio.wait_for()` 包一层。**很多人以为设了 `timeout` 就有总时长保护，这是错的。**

> **工程推论：**
>
> 1. **`max_tokens` 按场景给，不要全局一个值。** 分类 256、对话 16K、流式长文 64K——按预期输出形态定。全局拍一个值，要么截断要么浪费。
> 2. **`stop_reason` 必须检查。** `max_tokens` 表示被截断（加大重试），`end_turn` 是正常结束，`refusal` 是模型拒答（查 `stop_details`，别同 prompt 重试），`tool_use` 是要调工具，`pause_turn` 是服务端工具循环到上限需续。不看 stop_reason，你不知道这次响应是完整的还是半截的。
> 3. **HTTP 库的 timeout 不是墙钟超时。** `read`/`connect` 超时是 per-chunk 的。要总时长硬上限，自己在循环层记时。这条几乎是个陷阱题。
> 4. **超时设置要分场景。** 交互式聊天容忍不了 10 分钟，收紧到几十秒并降级；后台批处理可以放宽。一刀切的超时不合理。

---

## 八、错误码体系与 SDK typed exceptions：收个尾

前面零散讲了错误处理，这里系统化收口。**把错误当一等公民处理，是后端老手的本能**，照搬过来即可。

完整的 Python 错误处理骨架：

```python
import anthropic

try:
    response = client.messages.create(...)
except anthropic.BadRequestError as e:            # 400：请求错了
    print(f"Bad request: {e.message}")
except anthropic.AuthenticationError:             # 401：key 无效
    print("Invalid API key")
except anthropic.PermissionDeniedError:           # 403：没权限
    print("API key lacks permission")
except anthropic.NotFoundError:                   # 404：模型/端点错
    print("Invalid model or endpoint")
except anthropic.RateLimitError as e:             # 429：限流
    retry_after = int(e.response.headers.get("retry-after", "60"))
    print(f"Rate limited. Retry after {retry_after}s.")
except anthropic.APIStatusError as e:             # 其他状态错误，按码细分
    if e.status_code >= 500:
        print(f"Server error ({e.status_code}). Retry later.")
    else:
        print(f"API error: {e.message}")
except anthropic.APIConnectionError:              # 网络层错误
    print("Network error.")
```

几条纪律：

- **从具体到宽泛排列 `except`。** `RateLimitError` 在 `APIStatusError` 之前，否则被宽的先捕获了。所有类都继承 `APIError`（带 `.status`），还有 `.type` 字段做更细分类。
- **记录 `_request_id`。** 每个响应有 `message._request_id`（来自 `request-id` 头），出问题向 Anthropic 报告时带上它，能端到端追踪。
- **`ANTHROPIC_LOG=debug`** 开 SDK 日志，排查时有用。

> **工程推论：**
>
> 1. **typed exception > 字符串匹配，没有例外。** `except anthropic.RateLimitError` 永远优于 `if "rate_limit" in str(e)`。后者是脆弱代码的红旗。
> 2. **错误处理的颗粒度决定系统的可运维性。** 能区分 400（别重试）/ 429（退避）/ 500（重试）/ 网络错（重连），才能在每种情况下做对的事。粗暴地 `except Exception` 一把抓，等于放弃了所有针对性恢复手段。
> 3. **`request_id` 是你和 Anthropic 之间的追踪凭证。** 日志里没记 request_id，线上出了诡异问题就只能干瞪眼。

---

## 九、未来演进：API 工程这一层正在往哪走

前面教的全是"此刻怎么把裸 API 焊成生产依赖"。但这层的接口语义正在快速漂移——**看清漂移方向，你写的封装才不会半年作废**。这一节给趋势判断，不给可背的名单（名单按季度过期）。

**趋势一：行为控制的主通道，从"采样超参"迁移到"自然语言 + 思考/任务预算"。** 这是过去一年 API 最大的语义变化，也是最容易让老代码 400 的地方。前沿模型（Fable 5 / Opus 4.7 / 4.8 这一档）已经**移除了 `temperature` / `top_p` / `top_k`**——传了直接 400；`thinking` 也不再吃 `budget_tokens`，改成 `thinking:{type:"adaptive"}` 让模型自己决定想多深，配 `output_config.effort`（`low`/`medium`/`high`/`xhigh`/`max`）控总花费。

- **对你的封装意味着什么**：那些靠 `temperature=0` 求"确定性"的老调用，迁移时要么删掉（4.7+ 上是死代码），要么换成 `effort:"low"` + 更紧的 prompt。**"控行为"这件事正在从"调旋钮"变成"写清楚 + 给预算"。**
- **埋的坑**：`adaptive` 在新模型上**默认不开**（不传 `thinking` 字段 = 不思考），且 thinking 文本默认 `omitted`（块还在、文本是空）。你 UI 上给用户看"思考过程"会发现先卡一段长暂停——要显式 `display:"summarized"` 才有可见进度。这是个静默变化，不报错，排查极费时间（第二节工程推论 5 讲过）。

**趋势二：从"我兜底"到"服务端兜底"——重试、压上下文、控总预算正在内化进 API。** 你这章手写的很多东西，平台在一层层吃掉：

- **重试**：SDK 默认已自动退避重试 429/5xx（第四节）。
- **压上下文**：server-side compaction（beta）——逼近窗口时 API 自动把早期历史摘要成一个 `compaction` 块，你只要把 `response.content` 整个回传（不是只回文本，否则静默丢压缩态）。这把第一节"超窗要降级"那个 P0 验收，从"你写截断/摘要"变成"开个 flag"。
- **控总花费**：`task_budget`（beta，新模型，最小 2 万 token）——告诉模型整个 agentic 循环有多少 token 预算，模型看得到倒计时、自己节制。注意它跟 `max_tokens` 是两回事：`max_tokens` 是模型不知道的硬天花板，`task_budget` 是模型能看到的软预算。

> **这给你的判断**：这一层"通用基础设施"会持续被平台收编。你的护城河**不在"我会手写退避/截断"**（这些会变成 flag），而在**跨 provider、跨会话、跨工具的治理与成本归因**——那是平台短期不会替你做、但你业务必须有的（第六节成本归因、P4 网关）。"平台不会做用户层"是个危险假设，差异化要建在"平台做不到的跨边界"上，不是"平台暂时没做"。

**趋势三：缓存安全的"算子通道"正在标准化。** 第六节反复强调"别动 system prompt，否则缓存全崩"。但生产里你**确实**需要中途注入算子指令（切到简洁模式、塞入异步到达的上下文、报告剩余预算）。新出的 mid-conversation system message（beta）让你把 `{"role":"system", ...}` 追加进 `messages` 尾部——**既不动被缓存的前缀，又带算子权威**（比把指令塞进 user turn 更抗 prompt injection）。这是"缓存友好"和"可注入算子指令"两个诉求第一次被同时满足。把它当作第六节"冻结 system"纪律的官方逃生口。

> **一句话总结这层的演进方向**：**接口语义在从"机械旋钮"退场、向"自然语言意图 + 预算 + 服务端自动化"收敛。** 你越早把封装写成"对意图和预算建模"而非"对旋钮和手写循环建模"，迁移成本越低。**按 2026 初版图，具体参数名和 beta header 会变，但这个方向不会。**

---

## 本章心法

**把 LLM API 当成一个又贵、又慢、还偶尔说谎的远程 RPC，然后用你治理过所有不可靠下游的老手艺去包它。** 流式防超时、退避抗抖动、信号量控并发、缓存省成本、幂等保安全、typed exception 做分类——这一章没有一个概念是 AI 独有的，全是分布式系统的老道理换了层皮。区别只在于：这个下游按 token 收费（所以成本要归因到每次调用），输出价是输入价 5 倍（所以省输出优先），前缀缓存一个字节定生死（所以 prompt 组装要把稳定内容焊在前面）。**谁能把这个概率组件包装成一个 SLA 可控、成本可算、故障可恢复的确定性服务，谁就是这个时代的高级 AI 工程师。**

---

## 章末五件套

### 一、高频面试题（附答题框架）

**Q1：多轮对话在 HTTP 层是怎么实现的？为什么说模型"不记得"上一句？**

- 框架：API 无状态 → 每轮把全部历史重发 → 模型"记得"是因为历史在请求里，不是真有记忆 → 三个后果（成本随轮数超线性增长、迟早撞窗口、缓存成刚需）。
- 错误答案陷阱：说"服务端维护了会话 session"——大错，这正好把无状态的本质讲反了。
- 区分点：面试官想看你是否真理解"会话是客户端维护的幻觉"，以及能否推出成本/窗口/缓存三个工程后果。

**Q2：给你 400 / 429 / 500 / 503，分别该不该重试？为什么？**

- 框架：先分类"我的错（4xx 除 429）/ 暂时的错（429+5xx）"→ 400/401/403/404/413 不重试（重试也是错）→ 429 退避重试且读 `retry-after` → 500/529 退避重试。
- 加分点：主动提"重试的前提是幂等"，并指出 429 要优先读 `retry-after` 而非本地退避公式。
- 区分点：能不能秒答 + 给出原因，区分"读过文档"和"上过线"。

**Q3：指数退避为什么要加 jitter？不加会怎样？**

- 框架：不加 jitter → 大量客户端同步退避 → 同时重试形成惊群 → 把刚恢复的服务端再次打死 → 再次同步退避，死循环。jitter 把重试时间打散，破除同步。
- 错误答案陷阱：只说"退避就是等指数倍时间"，漏掉 jitter——这是最常被考的细节。
- 区分点：是否理解"分布式系统里同步行为本身是灾难"。

**Q4：Prompt caching 的核心不变量是什么？举三个会"静默击穿"缓存的例子。**

- 框架：不变量=前缀匹配，任何字节变化使其后全部失效，渲染顺序 tools→system→messages。三个击穿例子：system 里嵌时间戳/UUID、JSON 没 sort_keys、tools 随用户变。验证方法=看 `cache_read_input_tokens` 是否为 0。
- 加分点：讲清经济学（读 0.1×、写 1.25×/2×、两次/三次回本），指出击穿不报错只是账单悄悄翻倍。
- 区分点：是否真正理解"前缀匹配"以及它对 prompt 组装架构的约束。

**Q5：输入和输出 token 的定价关系是什么？这对你的优化策略有什么影响？**

- 框架：输出价 ≈ 输入价 5 倍 → 省输出比省输入划算 → 优化优先级：先让模型简洁/直接给结论（压输出），再谈压输入（缓存）。
- 加分点：联系到"让模型啰嗦 = 按 5 倍价烧钱"，把成本和 prompt 设计挂钩。
- 区分点：是否有成本直觉，能否把定价不对称转化为具体的工程优先级。

**Q6：为什么长输出必须用流式？流式只是为了打字机效果吗？**

- 框架：自回归生成慢（几十秒到几分钟）→ 非流式连接挂着等容易被 idle timeout 掐断 → 流式持续有字节流动，连接不被判 idle。SDK 在 max_tokens 大且非流式时直接抛 ValueError 拒绝。
- 错误答案陷阱：只说"流式是为了 UX 好看"——漏掉最硬的"超时防护"价值。
- 区分点：是否理解流式的工程本质（连接保活）而非仅 UX。

**Q7：RPM 和 TPM 有什么区别？为什么 TPM 容易先爆？**

- 框架：RPM=每分钟请求数，TPM=每分钟 token 数 → LLM 请求 token 量差异巨大，大上下文请求吃 token 凶 → 可能 RPM 还宽裕但 TPM 已爆 → 监控必须双维度。
- 加分点：提到 429 响应头里的 `x-ratelimit-remaining-*`，可以做"撞墙前主动降速"。
- 区分点：是否知道 LLM 限流是双维度，区分于普通 Web API 的单维度 QPS 限流。

**Q8：`httpx` 的 timeout 能保证一个请求不超过 N 秒吗？**

- 框架：不能。`read`/`connect` timeout 是 per-chunk 的（每收到一个字节就重置），不是墙钟总超时。流式响应只要持续有字节就永不触发。要总时长硬上限，循环层用 `time.monotonic()` 记时主动 break 或 `asyncio.wait_for()`。
- 错误答案陷阱：以为"设了 timeout=N 就有 N 秒总超时保护"——这是个经典陷阱。
- 区分点：是否真懂 HTTP 库超时语义，能否区分 per-chunk 和 wall-clock。

**Q9（拔高题，筛"真在生产里调过缓存"）：前缀一个字节没变，缓存为什么还可能不命中？至少说两种。**

- 框架：(1) **20-block 回溯窗口**——断点只向前找 20 个 content block，Agent 循环里一轮塞超过 20 个 `tool_use`/`tool_result` 就够不着上一轮缓存，静默 miss；修法是每 ~15 block 补中间断点。(2) **并发同前缀互斥**——缓存要等第一个响应开始 streaming 才可读，`gather` 出去的 N 条同前缀全按全价写；修法是先发一条等首 token 再放剩下的，或 `max_tokens:0` 启动预热。(3) 加分：失效是分层的，改 `tool_choice`/开关 thinking 不动 tools+system 缓存，只有动 tools 定义或切 model 才全量重建。
- 错误答案陷阱：只会背"system 别塞时间戳"——那是"字节变了"那一类，答不出"字节没变也 miss"就露馅。
- 区分点：能不能讲出"短对话命中、一进多工具 Agent 循环就掉 0"这个排查特征，是读过文档 vs 上过线的分水岭。

**Q10：流式默认开 + `get_final_message()` 兜底，这个组合到底解决了什么？为什么不是"要么纯流式要么纯非流式"？**

- 框架：流式的硬价值是**连接保活、规避 idle timeout**（不是打字机效果）；但纯流式你得自己拼一堆增量事件、还要处理断流半截。`get_final_message()`（TS 是 `finalMessage()`）让你**既享受流式不超时，又拿到 SDK 帮你拼回的完整 Message + usage**——不需要处理单 token 时，这就是最佳实践。`max_tokens` 大（>16K）SDK 估算会超时，非流式直接抛 `ValueError` 拒发，等于强制你走这条路。
- 加分点：指出 usage（含 output_tokens、cache_read）也从 final message 拿，所以成本归因不丢；断流要捕 `APIConnectionError`，决定重发整请求还是把半截标"已截断"交付。
- 区分点：是否理解"默认流式 + 兜底"是一个组合最佳实践，而非二选一。

### 二、实战项目增量（P1 基础设施层）

在 P0（裸 API 终端助手）基础上，给它套一层**生产级基础设施**，对应本章四块能力。

**验收标准（缺一不算过）：**

1. **重试退避**：所有 API 调用走统一的重试封装，正确区分可重试（429/5xx/网络错）与不可重试（4xx 其余）错误；指数退避**带 jitter**；429 时**优先读 `retry-after`**；有最大重试次数和延迟封顶。能演示：故意触发一次 429（或 mock），看到它正确退避而不是立刻打死或无限重试。
2. **并发批处理**：实现一个批量接口（一次处理 N 条输入），用 **asyncio + 信号量**控制并发，并发度可配置且根据 RPM/TPM 反推。能讲清你的信号量值是怎么定的。
3. **Prompt caching**：把可复用的大段 system prompt 加 `cache_control` 断点；**用 `usage.cache_read_input_tokens` 验证命中**（第二次相同前缀请求必须读到缓存 > 0）。能演示一次"静默击穿"：在 system 里故意插个时间戳，看到命中归零，然后修复。
4. **成本归因**：每次调用记录 input / output / cache_read / cache_write 四类 token，按价目表算出每次调用的成本，并能按"会话"或"功能"维度汇总输出累计花费。

**坑要写进复盘**（至少记 1 个真实踩坑）：缓存为什么没命中、信号量设多大被 429、退避为什么没生效……一个真坑的价值 > 十句正确的废话。

### 三、设计题（考权衡判断）

> 你要给一个"客服对话系统"设计 LLM 调用层。已知：每个会话平均 15 轮，system prompt 是 8KB 的固定客服规则，用户消息平均 200 token，高峰期 QPS 200。请设计这个调用层，并对至少 3 个关键决策给出**带 trade-off 的方案对照表并下判断**：(a) 用哪个梯队的模型，(b) prompt caching 怎么放断点、用 5min 还是 1h TTL，(c) 多轮历史增长怎么控（截断 vs 摘要 vs 滑窗），(d) 并发和限流怎么设。最后回答：这个系统**最大的成本黑洞**在哪、你怎么堵。

考点：能不能把本章的成本/缓存/限流/历史管理串成一个连贯的系统决策，而不是孤立背知识点。注意"15 轮 × 8KB 固定前缀"是个明显的缓存信号，"高峰 QPS 200 × 多轮"是限流和容量信号——能不能识别出来并下判断。

### 四、系统设计题（含量级估算）

> 设计一个**离线文档打标系统**：要给 100 万篇文档（平均每篇 2000 token）各打一组结构化标签（输出约 100 token）。要求 24 小时内完成，成本尽量低。
>
> 请估算：(1) 用 Batch API + Haiku，总 token 量和总成本大致是多少量级？(2) 如果改用同步 + Opus，成本和时间各差多少？(3) 这些文档如果共享一段 2KB 的打标 system prompt，prompt caching 能省多少？(4) 给出你的最终方案（模型 + 同步/Batch + 是否缓存）和理由。

参考量级（教方法，不是背数字；按 2026 初版图，以官方为准）：输入 100 万 × 2000 = 20 亿 token，输出 100 万 × 100 = 1 亿 token。Haiku 输入 ~$1/1M → 输入约 $2000，输出 ~$5/1M → 约 $500，合计 ~$2500 量级；走 **Batch 再砍半** → ~$1250。换 Opus（输入 $5 / 输出 $25）成本约涨 5 倍且不缓存的话更高。共享 system 前缀走缓存，命中部分按 0.1× 计，能在输入侧再省一大块。**最终方案大概率是：Haiku + Batch + 共享前缀缓存**——这题考的就是"能不能把模型选型、Batch、缓存三个成本杠杆叠起来算账并下判断"。

### 五、代码题（带 TODO 骨架 + 测试要求 + 暗坑提示）

实现一个**带成本归因的并发批处理器** `BatchProcessor`。

```python
import asyncio, time, random
from dataclasses import dataclass, field
import anthropic

# 价目表：每 1M token 的美元价（按 2026 初，opus-4-8）
PRICE = {"input": 5.0, "output": 25.0, "cache_read": 0.5, "cache_write": 6.25}

@dataclass
class CostLedger:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, usage):
        # TODO 1: 从 usage 累加四类 token
        #   注意 input_tokens 是"未缓存的全价部分"，不含 cache_read/cache_write
        ...

    @property
    def total_cost(self) -> float:
        # TODO 2: 按 PRICE 算总成本（四类 token 分别乘各自单价 / 1e6 后求和）
        ...

class BatchProcessor:
    def __init__(self, client, concurrency: int = 10, max_retries: int = 5):
        self.client = client
        self.sem = asyncio.Semaphore(concurrency)   # 暗坑A：信号量必须在事件循环内创建/使用
        self.max_retries = max_retries
        self.ledger = CostLedger()

    async def _call_one(self, messages, system) -> str:
        async with self.sem:                         # 进闸控并发
            for attempt in range(self.max_retries):
                try:
                    resp = await self.client.messages.create(
                        model="claude-opus-4-8", max_tokens=1024,
                        system=system, messages=messages,
                    )
                    self.ledger.add(resp.usage)      # 成功才记账
                    return next((b.text for b in resp.content if b.type == "text"), "")
                except anthropic.RateLimitError as e:
                    # TODO 3: 优先读 retry-after；读不到则指数退避 + jitter
                    ...
                except anthropic.APIStatusError as e:
                    if e.status_code >= 500:
                        # TODO 4: 5xx 退避重试；其余 4xx 直接 raise（不可重试！）
                        ...
                    else:
                        raise
            raise RuntimeError("max retries exceeded")

    async def run(self, items: list, system) -> list:
        # TODO 5: 用 asyncio.gather 并发跑所有 item，返回结果列表
        #   每个 item 是一段文本，包成 [{"role":"user","content":item}]
        ...
```

**测试要求：**

1. 用一个 mock client（不真打网络）验证：连续两条相同 system 前缀的请求，第二条的 `cache_read_tokens` 被正确累加。
2. 验证 mock 一个 429（带 `retry-after: 1`）后能正确退避并最终成功，且重试次数符合预期。
3. 验证 mock 一个 400 时**立刻 raise，不重试**（断言总调用次数为 1）。
4. 验证 `total_cost` 计算正确：手算一组已知 token 量的成本，对比 `ledger.total_cost`。

**暗坑提示：**

- **暗坑 A**：`asyncio.Semaphore` 不能在没有事件循环时创建后跨循环用——如果你在 `__init__` 里创建、又在不同 event loop 里跑，会报错。理解它和事件循环的绑定关系。
- **暗坑 B**：`ledger.add(resp.usage)` 里，`usage.input_tokens` 是**未缓存的全价部分**，不包含 `cache_read_input_tokens` 和 `cache_creation_input_tokens`。算总输入量要三者相加，但**算成本时四类分开乘各自单价**——别把 cache_read 也按全价算了，那就把"缓存省钱"算没了。
- **暗坑 C**：400 一定要立刻 raise。如果你的 `except` 顺序写反（宽的 `APIStatusError` 在前把 400 也吞了去重试），就会把不可重试错误拿去重试 5 次，浪费时间还掩盖 bug。`except` 从具体到宽泛排。
- **暗坑 D**：429 优先读 `retry-after`，读不到才用本地退避公式；退避公式记得加 jitter，否则批量请求会同步惊群。
