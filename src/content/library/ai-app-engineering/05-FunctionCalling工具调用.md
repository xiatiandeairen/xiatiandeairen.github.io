---
title: "Function Calling 工具调用"
slug: "05"
collection: "ai-app-engineering"
order: 5
summary: "这章排在哪、替你解决什么：前四章我们把 LLM 当成一个\"输入文本、输出文本\"的黑盒在调教——能控制它的格式、思考深度、稳定性。但纯文本模型有个根本残疾：它只会说，不会做。它不知道今天的天气，查不了你的数据库，连 `17 * 384` 都可能算错。"
topics:
  - "AI 应用工程"
tags: []
createdAt: "2026-06-11T06:51:24.000Z"
updatedAt: "2026-06-11T06:51:24.000Z"
---
> **这章排在哪、替你解决什么**：前四章我们把 LLM 当成一个"输入文本、输出文本"的黑盒在调教——能控制它的格式、思考深度、稳定性。但纯文本模型有个根本残疾：它**只会说，不会做**。它不知道今天的天气，查不了你的数据库，连 `17 * 384` 都可能算错。Function Calling 是给这个"大脑"接上"手脚"的协议层——它把"模型想做什么"和"系统替它做什么"在结构上彻底分开。这是从"聊天机器人"跨到"Agent"的第一道、也是最关键的一道工程门槛。第 6 章的 MCP、第 12 章的 agent，全部建立在本章这个 `tool_use → 执行 → tool_result` 的循环之上。学不透这一章，后面全是空中楼阁。

用一个你熟的锚点先把心智模型立住：**Function Calling 之于 LLM，相当于系统调用（syscall）之于用户态进程**。进程（模型）没有权限直接碰硬件（外部世界），它只能发起一个结构化的请求（tool_use ≈ syscall number + 参数），陷入内核（你的 harness），由内核执行真正的特权操作，再把结果（tool_result ≈ 返回值/errno）填回去。模型全程待在自己的沙箱里，**一行你的代码都不能直接执行**。把这条记牢，本章 80% 的设计决策你都能自己推出来。

---

## 一、背景：为什么需要工具调用

LLM 的训练产物是一个固定权重的函数：给定 token 序列，预测下一个 token。这个函数有三个治不好的病：

1. **知识截止（knowledge cutoff）**：权重冻结在某个时间点，之后的世界它一无所知。问它"现在 BTC 多少钱"，它要么瞎编，要么告诉你截止日期的旧价。
2. **无副作用能力**：它不能发邮件、不能写数据库、不能下单。它输出的永远是字符串，不是动作。
3. **不可靠的确定性计算**：算术、日期推算、精确字符串处理——这些本该 100% 准确的事，概率模型会以非零概率算错。

这三个病的共同解法是同一个：**让模型把"需要外部能力的部分"外包出去**。模型负责它擅长的——理解意图、规划步骤、组织语言；外部系统负责它做不到的——取实时数据、执行副作用、做精确计算。Function Calling 就是这个外包的**标准化协议**。

> **新手第一误区**：以为 Function Calling 是"模型自己去调用了 API"。**不是**。模型永远不联网、不执行代码。它只是输出一个"我想调用 `get_weather(location='Paris')`"的结构化意图，真正的 HTTP 请求是**你的代码**发的。模型连工具到底有没有被执行都不知道，直到你把结果填回去。这个边界是整个 Agent 安全模型的基石——想清楚"谁在执行"，比记住 API 字段重要得多。

---

## 二、演进：以前怎么做、为什么不够

### 阶段 0：纯 prompt 硬解析（2022 及更早）

最早没有原生工具支持，大家用 prompt 硬凑。典型是 ReAct 范式：让模型按固定格式输出 `Thought: ... / Action: search / Action Input: ...`，然后用正则把 `Action` 和 `Action Input` 抠出来。

```
Thought: 我需要查巴黎的天气
Action: get_weather
Action Input: Paris
```

**为什么不够**：

- **解析极脆**。模型多打一个换行、把参数写成 JSON、加一句解释，正则就崩。你在写一个永远打不完补丁的字符串解析器。
- **参数无 schema 约束**。`Action Input: Paris` 还是 `{"city": "Paris", "unit": "C"}`？没有契约，全靠 prompt 里举例子求模型配合。
- **多工具、多参数时指数级劣化**。工具一多，模型选错、格式跑偏的概率飙升。

### 阶段 1：原生 Function Calling（2023 至今）

各家把工具调用做进了模型训练和 API 协议。你用 **JSON Schema** 声明工具，模型被训练成**输出结构化的 `tool_use` 块**而非自由文本，API 层保证你拿到的是合法 JSON（配合 `strict` 还能保证符合 schema）。解析器死了，契约立住了。

这一步的本质跃迁不是"省了正则"，而是**把工具调用从 prompt 工程问题变成了协议工程问题**。协议是稳定的、可版本化的、可被 SDK 封装的。你的后端直觉在这里完全适用——这就是从"约定俗成的文本格式"升级到"带 schema 的 RPC"。

### 阶段 2：Agent 循环与生态标准化（2024 至今）

单次工具调用不够用了。真实任务需要"调用 A → 看结果 → 决定调 B → 再看 → 收尾"的多轮循环——这就是 agent loop 的雏形（第 12 章展开）。同时出现了：

- **并行工具调用**：一轮里同时发起多个独立调用。
- **server-side 工具**：代码执行、网页搜索这类由模型厂商托管执行的工具。
- **MCP（Model Context Protocol）**：把工具/资源/prompt 的暴露方式标准化，让工具可以跨应用复用（第 6 章主题）。
- **programmatic tool calling**：把多次工具调用压成一段脚本，减少 round trip。

本章覆盖的就是阶段 1 和阶段 2 的核心机制。

> **工程推论（背景与演进）**
> 1. 任何还在用正则解析模型输出来"实现工具调用"的生产系统，都应优先迁移到原生 Function Calling——这不是优化，是消除一整类线上故障。
> 2. 工具调用是协议而非 prompt，意味着它可以被 SDK、被 MCP、被 gateway 统一封装。你做架构选型时，应把"工具协议"当成和 HTTP/gRPC 同级的稳定接口来对待。
> 3. ReAct 的 `Thought` 没有消失——它变成了模型的 thinking（见第 3 章）。原生工具调用 + adaptive thinking ≈ 训练进权重里的、不会被正则解析的 ReAct。

---

## 三、现状：工具循环机制（核心中的核心）

这一节是整章的地基，必须讲到骨子里。

### 3.1 一次完整的工具循环

以 Anthropic Messages API 为例（其他家形态一致，字段名不同）。整个循环围绕 `stop_reason` 这个状态机驱动。

**第一步**：你带着工具定义发起请求。

```python
import anthropic

client = anthropic.Anthropic()

tools = [{
    "name": "get_weather",
    "description": "Get the current weather in a given location. "
                   "Call this whenever the user asks about weather, "
                   "temperature, or conditions in a specific place.",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name, e.g. Paris"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    },
}]

messages = [{"role": "user", "content": "What's the weather in Paris?"}]

response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=16000,
    tools=tools,
    messages=messages,
)
```

**第二步**：模型决定调工具，`stop_reason` 变成 `tool_use`，响应的 `content` 里出现一个 `tool_use` 块：

```json
{
  "stop_reason": "tool_use",
  "content": [
    {"type": "text", "text": "I'll check the weather in Paris for you."},
    {"type": "tool_use", "id": "toolu_01A...", "name": "get_weather",
     "input": {"location": "Paris"}}
  ]
}
```

注意三个关键点：
- `id`（`toolu_...`）是这次调用的**关联键**，回填结果时必须原样带回，否则 API 报错。
- `input` 已经是**解析好的对象**，不是字符串——但若你走原始 JSON，仍要用 `json.loads` 解析，绝不能裸字符串匹配（4.x 模型的 Unicode/斜杠转义可能和你预期不同）。
- `content` 里可能**同时有 text 和 tool_use**。模型可以一边说话一边调工具。

**第三步**：你执行工具，把结果作为 `tool_result` 塞进一个 **user 消息**回传。**关键纪律：先把模型的整个 assistant 响应（含 tool_use 块）追加进历史，再追加 tool_result。**

```python
# 把 assistant 的响应原样追加（必须保留 tool_use 块）
messages.append({"role": "assistant", "content": response.content})

# 执行工具，收集结果
tool_results = []
for block in response.content:
    if block.type == "tool_use":
        result = run_my_tool(block.name, block.input)   # 你的真实实现
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,     # 必须匹配
            "content": result,
        })

# tool_result 放进 user 消息回传
messages.append({"role": "user", "content": tool_results})
```

**第四步**：再次请求，模型拿到工具结果，生成最终回答，`stop_reason` 变回 `end_turn`，循环结束。

把这四步抽象成循环，就是最小 agent loop：

```python
messages = [{"role": "user", "content": user_input}]

while True:
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=16000,
        tools=tools,
        messages=messages,
    )

    if response.stop_reason == "end_turn":
        break   # 模型不再要工具，收工

    # （server-side 工具达到迭代上限时会给 pause_turn，需回填续跑，见 6.4）

    messages.append({"role": "assistant", "content": response.content})

    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            result = run_my_tool(block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })
    messages.append({"role": "user", "content": tool_results})

final = next(b.text for b in response.content if b.type == "text")
```

### 3.2 `stop_reason` 状态机：你必须处理的每一个分支

`stop_reason` 是驱动整个循环的状态字段。生产代码里漏处理任何一个分支都是隐患：

| stop_reason | 含义 | 你该做什么 |
|---|---|---|
| `end_turn` | 模型自然说完了 | 退出循环，取最终文本 |
| `tool_use` | 模型要调工具 | 执行工具，回填 tool_result，继续循环 |
| `max_tokens` | 撞到 `max_tokens` 上限 | 输出被截断了！调大上限或改流式 |
| `pause_turn` | server-side 工具的服务端循环达到迭代上限 | 把当前对话原样回传续跑，**别加"continue"** |
| `stop_sequence` | 撞到自定义停止序列 | 按业务处理 |
| `refusal` | 模型出于安全拒绝 | 看 `stop_details`，别用同样的 prompt 重试 |

> **回填第 1 章的伏笔——截断不能砍断工具链**
> 第 1 章讲过"`max_tokens` 截断会留下半截输出"。在工具调用语境下，这个坑会**升级成结构性破坏**。如果模型正在生成一个 `tool_use` 块时撞上 `max_tokens`，你拿到的是一个 `input` JSON 残缺、甚至 `tool_use` 块本身不完整的响应。此时：
> - 你**不能**把这个残块当正常 tool_use 执行——参数是烂的。
> - 你**更不能**把它原样追加进历史然后继续——API 会因为"有 tool_use 但无匹配的 tool_result"或"tool_use 块格式非法"而 400。
> 正确做法：循环里**显式检查 `stop_reason == "max_tokens"`**，要么调大 `max_tokens` 重来，要么对这一轮做特殊回收。工具链一旦被截断砍断，整个 agent 状态机就进了非法态。这就是为什么"给工具调用预留充足的 `max_tokens`"不是省钱问题，是正确性问题。

> **工程推论（工具循环）**
> 4. **每一个 `tool_use` 块都必须有一个 `tool_use_id` 严格匹配的 `tool_result` 回填**，缺一个就 400。这是面试高频陷阱：问"模型一轮发了 3 个工具调用，你执行成功 2 个、第 3 个抛异常，怎么回填？"——答案是 3 个 tool_result 一个都不能少，失败那个用 `is_error: true` 回填（见 3.6），而不是省略它。
> 5. **追加历史的顺序是死的**：先 assistant（含完整 content），后 user（含全部 tool_result）。把 tool_result 放进 assistant 消息、或漏掉 assistant 的 text 块，都会让后续轮次行为错乱。
> 6. **API 是无状态的**——每一轮你都要把完整对话历史发回去。这意味着工具调用的 token 成本是累积的、随轮次线性增长的。长 agent loop 的成本主要烧在这里，prompt caching（第 4 章）是这里的关键 lever。

### 3.3 `tool_choice`：谁来决定调不调、调哪个

在讲并行之前，先把 `tool_choice` 这个旋钮立住——它决定了"工具的调用决策权在模型还是在你手里"，是本章"可控性"主题的直接体现。默认情况下模型**自己决定**要不要调工具（auto），但很多生产场景你需要夺回这个决策权。

| `tool_choice` | 行为 | 典型用途 |
|---|---|---|
| `{"type": "auto"}` | 模型自己决定调不调、调哪个（**默认**） | 通用 agent，绝大多数场景 |
| `{"type": "any"}` | **必须**调至少一个工具，但调哪个由模型定 | 强制走工具路径（比如"所有查询都必须落到数据库工具，不许凭记忆答"） |
| `{"type": "tool", "name": "extract"}` | **强制**调指定的那个工具 | 把 LLM 当结构化抽取器用——见 5.4 |
| `{"type": "none"}` | **禁止**调任何工具 | 临时关掉工具，让模型纯文本回答 |

```python
# 强制模型必须调某个工具（常用于结构化抽取）
response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=16000,
    tools=tools,
    tool_choice={"type": "tool", "name": "extract_contact"},
    messages=messages,
)
```

注意一个**反直觉的耦合**：`disable_parallel_tool_use` 不是独立参数，它是挂在 `tool_choice` 对象里的一个布尔字段，对 `auto` / `any` / `tool` 都能加。下一节就用到它。

> **坑点提示**：`tool_choice` 设成 `any` 或 `tool`（强制调用）时，**和 thinking 有冲突**——强制立刻调工具，就没给模型留"先想再调"的空间。需要模型先推理再调工具时，别用强制档；强制档适合"我已经确定要调，只想要结构化参数"的场景（典型就是把工具当抽取 schema 用）。

> **工程推论（tool_choice）**
> 8a. `tool_choice` 是"工具调用决策权"的开关：`auto` 把权交给模型，`any`/`tool` 把权收回到你手里。面试被问"怎么保证模型一定走工具而不是凭记忆瞎答"，答案是 `tool_choice: any`（强制走工具）+ description 写清触发条件，而不是在 prompt 里喊"你必须用工具"。

### 3.4 并行工具调用与 `disable_parallel_tool_use`

模型可以在**一轮响应里发起多个 `tool_use` 块**。比如"对比巴黎和伦敦的天气"，它会同时发 `get_weather(Paris)` 和 `get_weather(London)`。这是默认行为。

并行的价值是**省 round trip**：两个独立查询不必串行两轮，一轮发出、你这边并发执行、一次性回填。延迟近乎减半。

但并行有个**致命前提：这些工具调用必须互相独立、且并行执行安全**。问题在于——**模型不知道你的执行环境是否支持并行**。它只是发出意图，能不能并发跑、跑了会不会出竞态，是你 harness 的事。

什么时候要关掉并行（`tool_choice` 里设 `disable_parallel_tool_use: true`，强制一轮最多一个工具）：

- **工具间有数据依赖**：B 的参数依赖 A 的结果。模型若并行发了 A、B，B 的参数就是它瞎猜的。
- **工具有副作用且非幂等**：并行发两个 `transfer_money` 你敢并发执行吗？
- **下游资源不支持并发**：数据库连接、外部 API 限流。
- **你想要严格的、可审计的串行执行顺序**。

```python
response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=16000,
    tools=tools,
    tool_choice={"type": "auto", "disable_parallel_tool_use": True},
    messages=messages,
)
```

> **坑点提示**：很多人以为"关掉并行 = 让 agent 变慢"，于是默认开着。但在**有副作用的写操作场景**里，默认开并行是在埋竞态地雷。安全的默认是：**只读、幂等、无依赖的工具放开并行；任何写操作、有依赖的工具，要么关并行，要么在 harness 层自己串行化**。这和你做分布式系统时"读可以并发、写要串行/加锁"的直觉一模一样。

> **工程推论（并行）**
> 7. 并行工具调用的安全性是 **harness 的责任，不是模型的责任**。模型只管发意图，"能不能并发执行"必须由你根据工具的可逆性、幂等性、资源约束来裁决。
> 8. `disable_parallel_tool_use` 是个**粗粒度开关**——它一刀切关掉所有并行。更细的控制（比如"只读工具并行、写工具串行")要在 harness 里自己实现：按工具名给每个工具打"parallel-safe"标记，并行跑安全的，串行跑不安全的。这正是下一节"专用工具"价值的来源之一。

---

## 四、工具设计哲学：bash 广度 vs 专用工具

这是本章最能拉开"会用 API"和"懂 Agent 架构"差距的一节，也是大厂面试官最爱深挖的地方。

### 4.1 核心张力

给 agent 配工具，有两个极端：

- **一个 `bash` 工具打天下**：给模型一个执行 shell 命令的工具，它几乎无所不能——读文件、调 API（`curl`）、跑脚本、操作 git。**广度极大**。
- **一堆专用工具（dedicated tools）**：`read_file`、`send_email`、`query_db`、`edit_file`……每个动作一个带 typed schema 的专用工具。**广度小，但每个动作可被精确管控**。

为什么这是个真问题？因为**模型只发出工具调用，harness 处理它们。工具调用的"形状"决定了 harness 能做什么**。

- `bash` 给 harness 的，是一个**不透明的命令字符串**——每个动作都长一个样（都是个 string）。harness 无法分辨 `bash("grep foo")`（并行安全的只读）和 `bash("git push")`（绝不能并发的写）。
- 把一个动作**提升为专用工具**，给 harness 的是一个**带 typed 参数、动作专属的 hook**——它可以拦截、可以审批、可以渲染、可以审计、可以调度。

### 4.2 何时把动作提升为专用工具——五条判据

这是可以直接当面试答题框架背的五条，每条都对应一个 harness 能力：

| 判据 | 含义 | 为什么 bash 做不到 | 例子 |
|---|---|---|---|
| **安全边界（security boundary）** | 这个动作需要被门控/审批吗？可逆性是好判据：难以撤销的动作（发消息、删数据、调外部写 API）应门控。 | `send_email` 工具一眼能 gate；`bash -c "curl -X POST ..."` 藏在字符串里，harness 看不出这是个不可逆的外发请求。 | 发邮件、转账、删除资源 |
| **可逆性 / 失效检查（staleness）** | 需要在执行前校验不变量吗？ | 专用 `edit` 工具能在写入前检查"文件自模型上次读取后是否被改动"，不一致就拒写。bash 无法强制这个不变量。 | 文件编辑的乐观锁 |
| **渲染（rendering）** | 这个动作需要专属 UI 吗？ | Claude Code 把"向用户提问"做成工具，于是能渲染成 modal、给出选项、阻塞 agent loop 直到用户回答。bash 输出没有这种结构。 | 向用户提问、展示 diff、确认对话框 |
| **调度 / 并行安全（scheduling）** | 这个动作能并行跑吗？ | 只读工具（`glob`、`grep`）能被标为 parallel-safe。同样的操作走 bash，harness 分不清 parallel-safe 的 `grep` 和 parallel-unsafe 的 `git push`，只能全部串行。 | 区分只读检索 vs 写操作 |
| **审计 / 可观测** | 需要结构化日志吗？ | 专用工具的调用是 `{name, typed_input}`，天然可结构化记录、可统计、可回放。bash 只有一坨命令字符串。 | 合规审计、用量分析 |

### 4.3 拇指法则

> **从 bash 起步求广度，当你需要 gate（门控）、render（渲染）、audit（审计）、parallelize（并行）某个动作时，把它提升为专用工具。**

这条法则背后是一个深刻的权衡：**广度 vs 可控性**。`bash` 给你最大的能力面，但把所有动作压成同一个不可区分的形状，让 harness 丧失了管控的抓手。专用工具牺牲广度，换回 harness 对该动作的全部控制权。

国内大厂的实践语境也一致：豆包/Qwen/DeepSeek 等的 agent 框架里，对"危险动作"（外发、写库、执行任意代码）几乎都会做成专用的、带审批钩子的工具，而不是放任 agent 用一个万能执行器。原因不是模型不行，是**生产系统对"可逆性"和"可审计"的硬要求**。

> **工程推论（工具设计）**
> 9. 工具的粒度选择，本质是在"模型的能力广度"和"harness 的管控能力"之间做权衡。这不是模型问题，是**系统设计问题**——和你设计 API 时"给一个万能 endpoint 还是一堆 RESTful 资源"是同构的。
> 10. **可逆性是提升为专用工具的第一判据**。一个动作越难撤销，越应该有专属的、可门控的工具，而不是藏在 bash 命令里。面试时被问"什么动作该做成专用工具"，先答可逆性/安全边界，再补渲染、调度、审计——这个顺序体现你抓住了主要矛盾。
> 11. server-side 的 code execution 工具（见第六节）某种意义上是"受控的 bash"——它给了你 bash 的广度，但跑在厂商托管的隔离沙箱里、无网络、有资源限额。当你需要广度又怕自己 host bash 的安全风险时，它是一个折中点。

---

## 五、工具描述写法：prescriptive when-to-call 对召回率的影响

这一节是"小改动、大收益"的典型，也是最容易被忽视的工程杠杆。

### 5.1 description 是模型的唯一决策依据

模型决定"要不要调这个工具、怎么填参数"，**几乎完全依赖 `description` 字段**。工具名和 schema 是辅助，description 是主导。一个写得烂的 description，会让模型该调的时候不调（召回率低），或不该调的时候乱调（精确率低）。

### 5.2 描述"什么时候调"，而不只是"做什么"

最关键的一条写法：**description 要 prescriptive about *when* to call，不能只 descriptive about *what* it does。**

对比：

```python
# 弱：只说做什么
"description": "Get weather data."

# 强：明确什么时候该调
"description": "Get the current weather in a given location. "
               "Call this whenever the user asks about weather, temperature, "
               "or conditions in a specific city or place. Do not call it for "
               "general climate questions that don't need live data."
```

第二种写法把"触发条件"写进了 description。这在**近期的 Opus 模型上有可测量的召回率提升**——这些模型默认对工具更"克制"（reach for tools more conservatively），不给明确的触发条件，它们倾向于自己用知识回答而不调工具。把"call this when..."写明，是把工具的调用率拉回来的直接手段。

### 5.3 这条规律和模型版本强相关（知识时效）

> **知识时效标注（版图类，按 2026 初，模型行为会随版本变，请学方法论）**：不同代模型对工具的"积极性"不同。比如较早的模型可能过度调用工具，而较新的 Opus（4.7/4.8 一线）默认更保守、更倾向先推理后调工具。这意味着**同一套工具描述，换个模型表现可能反向**：在"过度调用"的模型上你要写得克制，在"保守"的模型上你要写得 prescriptive、鼓励它多调。**方法论是固定的——根据模型的默认倾向反向调 description；具体哪个模型偏哪边，会随季度变化，以实测为准。**

落地建议：
- **召回率低（该调不调）**：在 description 里加明确的触发条件（"Call this when..."），必要时在 system prompt 里也强调。可调高 effort（第 3 章）——高 effort 下模型的工具使用率显著上升。
- **精确率低（乱调）**：description 里加排除条件（"Do not call for..."），dial back system prompt 里 `CRITICAL: YOU MUST` 这类过激措辞——近期模型对 system prompt 跟得很紧，过激指令会导致 overtriggering。

> **面试加分点**：能说出"工具描述要 prescriptive 触发条件，且最优写法依赖模型对工具的默认倾向，需要在 eval 集上实测召回/精确率来调"——这一句话同时体现了你懂 prompt 工程、懂模型代际差异、懂用数据驱动调优。比单纯说"description 要写清楚"高一个段位。

> **工程推论（工具描述）**
> 12. 工具描述的触发条件（when-to-call）是召回率的直接 lever。把它当成可 A/B 测试的产品参数，而不是一次写完就不管的文档。
> 13. effort 参数和工具召回率耦合：调高 effort，模型更舍得调工具、更舍得多轮探索。当 agent"偷懒不调工具"时，先试 effort 再试改 prompt。
> 14. 在**多工具**场景下，每个工具的 description 还要承担"和其他工具区分"的职责——描述要让模型能在相似工具间正确选择。工具一多，描述的"边界清晰度"比"内容详尽度"更重要。

### 5.4 `strict` 与结构化工具参数：让 schema 从"提示"变成"保证"

description 解决"调不调"，但还有一个独立问题：**调了之后，参数填得对不对？** 默认情况下，`input_schema` 对模型只是一个**强烈建议**——模型大概率会遵守，但不保证。它可能漏掉 required 字段、把 `enum` 填成范围外的值、把 `integer` 填成字符串。在"可控性"这个阶段，这是必须堵上的洞。

机制是给工具加 `strict: true`（近期模型支持，配合结构化输出能力）：

```python
tools=[{
    "name": "book_flight",
    "description": "Book a flight. Call this only when the user confirms a booking.",
    "strict": True,                       # 开启严格 schema 约束
    "input_schema": {
        "type": "object",
        "properties": {
            "destination": {"type": "string"},
            "date": {"type": "string", "format": "date"},
            "passengers": {"type": "integer", "enum": [1, 2, 3, 4, 5, 6, 7, 8]},
        },
        "required": ["destination", "date", "passengers"],
        "additionalProperties": False,    # strict 下必须显式禁止额外字段
    },
}]
```

`strict: true` 把 schema 从"模型尽量遵守"升级成"API 层保证产出合法"。代价/约束要知道：
- **`additionalProperties: false` 是必须的**，且 schema 有限制（不支持递归、数值范围 `minimum/maximum`、字符串长度 `minLength` 等约束——这些要在你代码里二次校验）。
- **首次用某个新 schema 有一次性编译延迟**，之后 24 小时命中缓存。
- 和 `tool_choice: {type: "tool", name: ...}` 组合，就是"强制调这个工具 + 保证参数合法"——这正是**把 LLM 当结构化抽取器**的标准配方：输入一段自由文本，输出一个你敢直接 `json.loads` 喂给下游的对象，不用写一行防御性解析。

> **面试加分点**：能区分"description 影响召回率（调不调）"和"strict 影响参数合法性（填得对不对）"是两个正交的控制维度——前者是 prompt 层的软约束，后者是 schema 层的硬约束。再补一句"`tool_choice: tool` + `strict` + 单工具 = 把对话模型降维成一个带类型保证的抽取函数"，直接体现你把 Function Calling 用出了 RPC 的工程味道。

> **新手第一误区**：以为写了 `input_schema` 模型就一定按 schema 填参数。不是——不加 `strict`，schema 只是给模型看的"建议格式"，模型有非零概率违反它。要么开 `strict`，要么在 harness 里对 `block.input` 做一次显式校验（Pydantic / zod），**永远不要假设 `input` 一定合法**。SDK 还提供 `messages.parse()`（配合 Pydantic/zod）自动校验响应——能用就用，省掉手写校验。

> **工程推论（strict）**
> 14a. 工具参数的可靠性有两道闸：description（软，影响召回）和 strict（硬，影响合法性）。生产里凡是"工具参数错了会造成副作用"的场景（下单、转账、写库），都应该上 strict 或等价的 harness 层校验，不能只靠模型自觉。
> 14b. `tool_choice: {type:"tool"}` + `strict: true` + 单一工具，是"用 LLM 做结构化抽取"的最稳形态——比让模型直接吐 JSON 文本再解析可靠得多，因为吐文本那条路没有 schema 强制、还要自己处理 markdown 代码块/前后缀废话。

---

## 六、错误恢复、循环管理、server-side 工具

### 6.1 错误恢复：`is_error` 回传

工具执行失败是常态——网络抖、参数非法、资源不存在。**关键原则：失败不要静默吞掉，也不要直接抛给用户，而是结构化地告诉模型，让它自己适应。**

机制是在 `tool_result` 里设 `is_error: true`，content 写清楚错在哪：

```python
tool_result = {
    "type": "tool_result",
    "tool_use_id": tool_use_id,
    "content": "Error: Location 'xyz' not found. Provide a valid city name.",
    "is_error": True,
}
```

模型拿到这个 error，通常会**自我修正**：换个参数重试、或回头问用户澄清。这把"错误处理"从你的硬编码 if-else，变成了模型的自适应行为。

写好 error content 的要点（和你写好 API 错误信息的直觉一致）：
- **可操作**：告诉模型怎么修（"provide a valid city name"），而不只是"failed"。
- **不泄露敏感信息**：error 会进模型上下文，别把内部栈、密钥、连接串塞进去。
- **区分错误类型**：参数错（模型能自己改）vs 系统错（重试也没用，该升级）——用不同的 content 引导模型走不同的路。

> **新手第一误区**：工具一抛异常就让整个 agent loop 崩掉，或者把 Python traceback 原样塞回 tool_result。前者放弃了模型的自愈能力；后者既浪费 token 又泄露内部实现。正确姿势是 catch 住异常，翻译成一句模型能理解、能据此调整的错误描述。

### 6.2 tool runner vs 手写循环：何时必须手写

SDK 提供了 **tool runner**（beta）——你用装饰器/schema 定义工具，runner 自动跑完整个"调 API → 执行工具 → 回填 → 循环到 end_turn"。

```python
from anthropic import beta_tool

@beta_tool
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get current weather for a location.

    Args:
        location: City and state, e.g., San Francisco, CA.
        unit: Temperature unit, "celsius" or "fahrenheit".
    """
    return f"72°F and sunny in {location}"

runner = client.beta.messages.tool_runner(
    model="claude-opus-4-8",
    max_tokens=16000,
    tools=[get_weather],
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
)

for message in runner:   # 循环被 SDK 接管，迭代到模型不再调工具
    print(message)
```

tool runner 的好处：少写循环样板、schema 从函数签名/类型自动生成、类型安全。**简单场景、内部工具、无审批需求时，用它。**

但有几类场景**必须手写循环**——因为你需要在"模型发起调用"和"真正执行"之间插一脚：

| 何时手写循环 | 原因 |
|---|---|
| **审批 / human-in-the-loop** | 危险动作执行前要等人点"同意"。runner 会自动执行，没给你插入审批的缝。 |
| **自定义日志 / 审计** | 你要在每次工具调用前后记结构化日志、埋点、做合规留痕。 |
| **条件执行** | 根据运行时状态决定某个工具调不调、改参数、跳过。 |
| **per-token 流式 + 工具** | 需要边流式输出边处理工具时，手写循环里用 `stream()` 更可控。 |
| **复杂状态机 / 自定义重试** | 你的 agent 有超出 runner 默认行为的循环逻辑。 |

> **判断口诀**：**runner 管"自动化"，手写管"插一脚"**。只要你需要在工具执行的循环里插入审批、日志、条件判断这三件事的任何一件，就手写。这正是上面那段最小 agent loop 代码的价值所在——它是你能完全掌控的循环。

### 6.3 human-in-the-loop（HITL）：审批门控的标准形态

HITL 是手写循环的头号用例，也是生产 agent 处理"危险动作"的标准模式。结构很清晰：

```python
# 手写循环里，执行工具前插入审批
for block in response.content:
    if block.type == "tool_use":
        if is_dangerous(block.name):           # 你定义的危险动作判定
            approved = ask_human_to_approve(block.name, block.input)  # 阻塞等人
            if not approved:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "User denied this action. "
                               "Suggest an alternative or ask for clarification.",
                    "is_error": True,        # 把"被拒"当成一种错误反馈给模型
                })
                continue
        result = run_my_tool(block.name, block.input)
        tool_results.append({
            "type": "tool_result", "tool_use_id": block.id, "content": result,
        })
```

注意一个精妙处：**用户拒绝时，把"被拒绝 + 为什么"作为 tool_result 回填**（甚至带上拒绝理由），模型会据此调整策略——换个安全的做法、或问用户到底想怎样。这比直接中断 agent 体验好得多。

> **HITL 的工程化要点**：审批不该是"每个工具都弹窗"——那会把用户烦死。正确做法是**按可逆性分级**：只读/幂等动作自动放行，不可逆/高风险动作才门控（回到第四节"可逆性是第一判据"）。Claude Code 的 permission 模型就是这个思路——危险命令问、安全命令直接跑。这也呼应了 Managed Agents 里的 `always_ask` / `always_allow` 权限策略：平台级的 agent 把这套门控做成了配置项。

> **工程推论（错误与循环）**
> 15. `is_error: true` 把错误处理从"你的代码分支"变成"模型的自适应行为"。这是 agent 鲁棒性的关键设计——但前提是 error content 写得可操作、不泄密。
> 16. tool runner 和手写循环不是二选一的信仰问题，是**场景问题**：要审批/日志/条件执行就手写，否则用 runner 省事。能清晰说出这条边界，是区分"调过 API"和"做过生产 agent"的分水岭。
> 17. HITL 的审批粒度要按可逆性分级，不能一刀切全审批。把"被拒绝"作为带理由的 tool_result 回填，让模型自适应，比硬中断体验好。

### 6.4 server-side 工具 vs client-side 工具

到这里要引入一个关键的架构分类，它决定了"谁执行工具"。

| | **client-side 工具** | **server-side 工具** |
|---|---|---|
| 谁定义 | 你（name/schema/实现全是你的） | 厂商定义 name/schema/模型用法 |
| 谁执行 | **你的 harness** | **厂商基础设施** |
| 典型 | `get_weather`、`query_db`、你的一切业务工具；以及厂商给了参考实现但你执行的 bash/text editor | **code execution**（厂商托管沙箱跑代码）、**web search / web fetch**（厂商执行搜索） |
| 你要做什么 | 实现 + 执行 + 回填 tool_result | **只需在 `tools` 里声明，模型自动用，结果直接回到上下文** |
| 控制力 | 完全控制（能审批、能拦截） | 厂商托管，你管不到执行细节 |

server-side 工具的声明极简——比如 code execution 和 web search，连 schema 都不用写：

```python
response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=16000,
    messages=[{"role": "user", "content": "查一下最新的 BTC 价格并算出涨跌幅"}],
    tools=[
        {"type": "web_search_20260209", "name": "web_search"},
        {"type": "code_execution_20260120", "name": "code_execution"},
    ],
)
```

声明完，模型自己发搜索、自己跑代码，你**什么循环都不用写**，结果直接出现在响应里。

**两者的本质区别和选型**：

- **client-side**：要执行业务逻辑、碰你的数据/系统、需要审批和管控时用。控制力最大。
- **server-side**：要"开箱即用"的通用能力（跑沙箱代码、搜网）、又不想自己 host 沙箱/搜索基础设施时用。代价是执行在厂商那边，你管不到细节。

> **坑点提示**：server-side 工具的服务端有自己的循环上限。当 code execution / web search 在服务端连续跑到上限时，响应 `stop_reason` 是 `pause_turn`，你要把"当前 user 消息 + assistant 响应"原样回传让它续跑，**且千万别额外加一句"continue"**——API 检测到末尾的 `server_tool_use` 块会自动接着跑，加多余的话反而干扰它。生产里要设 `max_continuations`（比如 5）防无限循环。

> **工程推论（server vs client）**
> 18. "谁执行工具"是 agent 架构的一级分类。server-side 工具用 host 控制力换开发便利和零运维；client-side 工具用开发成本换完全控制。涉及业务副作用和数据安全的，必须 client-side。
> 19. server-side code execution 是"受控 bash"的最佳落地——给你 bash 的广度，跑在隔离/无网/限额的沙箱里。需要让 agent 跑任意计算又怕自己 host 的安全风险时，优先它。
> 20. `pause_turn` 是 server-side 工具特有的续跑信号，处理方式和 `tool_use` 不同（不回填 tool_result，而是原样续传）。漏处理会让 server-side 工具任务半途而废。

### 6.5 programmatic tool calling：把多次调用压成脚本

标准工具调用里，每次调用都是一次 round trip：模型调 → 结果进上下文 → 模型推理 → 调下一个。三个串行动作（查用户 → 查订单 → 查库存）= 三次 round trip，每次都加延迟、加 token，而且大量中间数据其实后面再没用过。

**programmatic tool calling（PTC）** 让模型把这些调用**编排成一段脚本**，脚本跑在 code execution 容器里。脚本调用工具时，容器暂停、调用被执行（client 或 server 侧）、结果返回给**运行中的代码**（不是模型上下文），脚本用正常控制流（循环、过滤、分支）处理它。**只有脚本的最终输出才回到模型上下文。**

什么时候用 PTC：
- **大量串行工具调用**：把 N 次 round trip 压成 1 段脚本。
- **中间结果巨大**：比如一个工具返回上万行数据，你只要其中几行——让脚本在容器里过滤完再把精简结果给模型，避免把上万行灌进上下文窗口。

PTC 的本质收益是：**token 成本和延迟从"随调用次数增长"变成"只和最终输出相关"**。这对重 agent、大中间结果的场景是数量级的优化。

> **工程推论（PTC）**
> 21. PTC 是用"代码编排"替代"模型逐次编排工具"。当你的 agent 有很多串行调用、或中间结果很大需要在进上下文前过滤时，PTC 把成本从 O(调用次数) 降到 O(最终输出)。这是高级 agent 的成本优化手筋，面试时能点出"用脚本编排压 round trip + 过滤中间结果"是加分项。

---

## 五件套

### 高频面试题（12 道，每道附答题框架）

**1. 完整描述一次工具调用的循环，以及 `stop_reason` 在其中的作用。**
框架：四步——带工具发请求 → 模型回 `tool_use`（stop_reason=tool_use）→ 执行并回填 tool_result（顺序：先 assistant 含 tool_use，后 user 含 tool_result，id 严格匹配）→ 再请求拿最终答案（stop_reason=end_turn）。强调 `stop_reason` 是驱动循环的状态机，要处理 end_turn / tool_use / max_tokens / pause_turn / refusal 每个分支。**陷阱**：很多人漏掉"先追加 assistant 响应再追加 tool_result"的顺序，或漏说 API 无状态、每轮要发全量历史。

**2. 模型一轮发了 3 个工具调用，第 2 个执行抛异常了，你怎么回填？**
框架：3 个 `tool_result` 一个都不能少（缺任一个 → API 400）；失败那个用 `is_error: true` + 可操作的错误描述回填，让模型自适应。**陷阱**：答"跳过失败的那个"是错的——会导致 tool_use 无匹配 result 而 400。

**3. 什么时候该把一个动作从 bash 提升为专用工具？**
框架：五条判据，按重要性——可逆性/安全边界（第一）、失效检查（乐观锁）、渲染（专属 UI）、调度/并行安全、审计。核心逻辑：模型只发意图，工具的"形状"决定 harness 能不能 gate/render/audit/parallelize。**陷阱**：只答"复杂操作做成专用工具"太浅，要扣住"harness 的管控抓手"这个本质。

**4. 并行工具调用的安全性由谁负责？什么时候关掉它？**
框架：harness 负责，不是模型。模型不知道你的执行环境是否支持并发。关并行的场景：工具间有依赖、有副作用且非幂等、下游不支持并发、需严格审计顺序。用 `disable_parallel_tool_use`。**加分**：指出这是个粗粒度开关，细粒度要在 harness 里按工具打 parallel-safe 标记。

**5. 工具的 description 怎么写才能提高召回率？这和模型版本有什么关系？**
框架：prescriptive about when-to-call（"Call this when..."）而非只 descriptive about what。近期保守的 Opus 模型上这能显著提升召回。**关键**：最优写法依赖模型默认倾向——过度调用的模型要写克制，保守的模型要写鼓励，需在 eval 集实测调。**陷阱**：只说"写清楚"没抓住"触发条件 + 模型代际差异"。

**5b. `tool_choice` 有哪几档？什么时候强制调用？它和 thinking 有什么冲突？**
框架：四档——`auto`（模型自决，默认）、`any`（必须调至少一个、调哪个由模型定）、`tool`（强制调指定工具）、`none`（禁止调工具）。强制档用途：把模型当结构化抽取器（`tool`），或保证查询一定落到工具而非凭记忆（`any`）。**冲突点**：强制档（any/tool）让模型立刻调工具，挤掉了"先推理后调用"的空间，和需要 thinking 的场景互斥——要先想再调就别用强制档。**陷阱**：用 prompt 里喊"你必须用工具"来代替 `tool_choice: any`——前者是软约束、会漏，后者是协议层硬保证。

**5c. 写了 `input_schema`，模型就一定按 schema 填参数吗？怎么保证参数合法？**
框架：不一定。不加 `strict`，`input_schema` 对模型只是建议，有非零概率违反（漏 required、enum 越界、类型不符）。保证合法的两条路：(1) 工具加 `strict: true`（需 `additionalProperties: false`，schema 有限制，首次有编译延迟），把合法性提升为 API 层保证；(2) harness 里对 `block.input` 做显式校验（Pydantic/zod），SDK 的 `messages.parse()` 可自动做。**关键区分**：description 管"调不调"（召回率，软），strict 管"填得对不对"（合法性，硬），两个正交维度。**加分**：`tool_choice: tool` + `strict` + 单工具 = 最稳的 LLM 结构化抽取形态。

**6. `is_error` 的作用是什么？error content 该怎么写？**
框架：把错误从你的代码分支变成模型的自适应行为——模型拿到 error 会自我修正（换参数/问用户）。content 要可操作（告诉怎么修）、不泄密（不塞栈/密钥）、区分参数错 vs 系统错。**陷阱**：答"直接把 traceback 回填"是错的——泄密 + 浪费 token。

**7. tool runner 和手写循环怎么选？**
框架：runner 管自动化（少样板、类型安全），手写管"插一脚"。需要审批/HITL、自定义日志审计、条件执行这三件事任一件，就必须手写——因为 runner 会自动执行，没给你插入的缝。**加分**：点出 per-token 流式 + 工具、复杂状态机也倾向手写。

**8. client-side 工具和 server-side 工具有什么区别？怎么选？**
框架：区别在"谁执行"——client 你的 harness 执行（完全控制，能审批），server 厂商执行（声明即用，零运维，管不到细节）。选型：碰业务数据/有副作用/要审批 → client；要开箱即用的通用能力（跑代码、搜网）又不想 host → server。**加分**：提 server-side 的 `pause_turn` 续跑机制。

**9. `max_tokens` 截断会对工具调用造成什么特殊影响？**
框架：普通文本截断只是半截输出，但若截在 `tool_use` 块生成中，会得到残缺的 input JSON、不完整的 tool_use 块——既不能当正常调用执行，也不能原样续传（会因非法块或缺 result 而 400）。要显式检查 `stop_reason==max_tokens` 做回收。**本质**：工具链被截断 = agent 状态机进非法态。这是第 1 章伏笔的升级版。

**10. programmatic tool calling 解决什么问题？**
框架：标准工具调用每次是一次 round trip，N 次串行调用 = N 次延迟 + 大量没用的中间数据进上下文。PTC 让模型把调用编排成脚本跑在沙箱里，中间结果留在运行的代码里、只有最终输出回上下文。收益：成本从 O(调用次数) 降到 O(最终输出)，尤其适合多串行调用 + 大中间结果。

### 实战项目（P1 增量）：工具化助手

**目标**：在 P0 纯对话助手基础上，建立工具循环与错误恢复，接入三类工具——计算器、天气、数据库查询。

**增量任务**：
1. 用 client-side 工具实现 `calculator(expression)`、`get_weather(location, unit)`、`query_db(sql)` 三个工具，写好 prescriptive 的 description。
2. **手写** agent loop（不用 tool runner），正确处理 `end_turn` / `tool_use` / `max_tokens` / `refusal` 四个 `stop_reason` 分支。
3. 实现 `is_error` 错误回传：`query_db` 收到非法 SQL 时返回可操作 error，验证模型会自我修正。
4. 给 `query_db`（写操作风险）加 HITL 审批门控；`calculator` / `get_weather`（只读）自动放行。
5. 让"对比两个城市天气"触发并行工具调用，观察一轮内多个 tool_use 块；再对 `query_db` 用 `disable_parallel_tool_use` 强制串行。

**验收标准**：
- [ ] "巴黎和伦敦哪个更热"能在一轮内并行发起两个 `get_weather`，且 harness 正确并发执行、一次性回填两个 tool_result（id 严格匹配）。
- [ ] 给 `query_db` 喂一个语法错的 SQL，模型收到 `is_error` 后能自己改正重试，而非 agent 崩溃。
- [ ] 触发 `query_db` 时弹出审批，拒绝后模型收到"被拒 + 理由"并给出替代方案，而非硬中断。
- [ ] 故意把 `max_tokens` 设到很小，构造工具调用被截断，验证你的循环能识别 `stop_reason==max_tokens` 并优雅回收（不 400、不死循环）。
- [ ] 计算 `17 * 384 + 9001` 时模型走 `calculator` 而非自己心算，结果精确正确。

### 设计题（开放式，考权衡）

你要为一个"运维助手 agent"设计工具集。它需要能：查日志、查监控指标、重启服务、改配置、给值班群发告警。请设计这套工具——哪些做成专用工具、哪些可以共用一个通用执行器、每个工具的并行策略和审批策略如何定，并说明你的权衡。

考点：能否按可逆性/安全边界给动作分级（查日志/查指标是只读可并行可自动放行；重启服务/改配置是不可逆必须专用工具 + 审批；发告警是有副作用要门控）；能否说清"为什么不全用一个 bash 执行器"（丧失 harness 的 gate/audit/调度抓手）;能否给出并行策略（只读并行、写串行）。好答案会主动提"按可逆性分级审批，避免每个动作都弹窗烦死值班"。

### 系统题（含量级估算）

设计一个面向 1000 家企业客户的"数据分析 agent"服务。用户用自然语言提问，agent 通过工具查客户自己的数据仓库（client-side `query_warehouse` 工具）、跑统计分析（server-side code execution）、生成图表。假设峰值 QPS=50，平均每个问题触发 4 轮工具调用，每轮上下文约 8K token，输出约 1K token。请估算：(a) 峰值的 token 吞吐量级；(b) 在不加优化时单问题的成本量级（用 Frontier 档输入 $5 / 输出 $25 每百万 token）；(c) prompt caching 能省多少；(d) 哪些工具该并行、该用 PTC。

框架：
- **吞吐**：50 QPS × 4 轮 = 200 次模型调用/秒；输入约 200 × 8K = 1.6M token/s 输入侧（注意：无状态 API 每轮发全量历史，4 轮的输入是累积的，实际更高）。
- **单问题成本**：4 轮输入 ≈ (8+16+24+32)K ≈ 80K token（历史累积）× $5/M = **$0.40**；输出 4×1K=4K token × $25/M = **$0.10**；合计 **约 $0.5/问题**量级。**关键洞察**：成本主要烧在累积的输入历史，不是输出——输入侧 80K vs 输出侧 4K，差 20 倍，尽管输出单价是输入的 5 倍，输入仍是大头。（顺手记一下量级换算：1M token 的 $5 即 $5e-6/token；80K × $5e-6 = $0.4。面试现场最容易翻车的就是这个 1e6 的量级，别把 $0.4 算成 $0.0004。）
- **caching**：稳定前缀（system prompt + 工具定义 + 早期历史）缓存命中后约 0.1× 读取价。对上面这个 $0.5/问题：若 80K 输入里有 ~64K 是可缓存的稳定前缀，命中后这部分从 $0.32 降到约 $0.032，单问题成本砍到 ~$0.2 量级——长 agent loop 里这是这类服务的命门优化。注意缓存写入是 1.25×（5min TTL），所以要复用才划算。
- **并行/PTC**：多个独立的 `query_warehouse` 应并行；"查多张表 → 过滤 → 聚合"这类大中间结果场景用 PTC，避免把全表灌进上下文。
- **追问**：rate limit、客户数据隔离（每个企业的工具凭据隔离）、code execution 的容器复用。

### 代码题（带 TODO 骨架 + 测试要求 + 暗坑提示）

实现一个**带审批门控和错误恢复的手写 agent loop**。

```python
import anthropic
import json

client = anthropic.Anthropic()

# 工具实现（calculator 只读，transfer_money 危险）
def calculator(expression: str) -> str:
    # 暗坑：别用 eval！这里仅示意，生产要用安全的表达式解析器
    return str(eval(expression, {"__builtins__": {}}))

def transfer_money(to: str, amount: float) -> str:
    return f"Transferred ${amount} to {to}."

TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression. "
                       "Call this whenever the user needs a precise calculation.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "transfer_money",
        "description": "Transfer money to a recipient. "
                       "Call this only when the user explicitly requests a transfer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "amount": {"type": "number"},
            },
            "required": ["to", "amount"],
        },
    },
]

DANGEROUS = {"transfer_money"}   # 需要审批的工具集

def execute_tool(name: str, tool_input: dict) -> str:
    if name == "calculator":
        return calculator(**tool_input)
    if name == "transfer_money":
        return transfer_money(**tool_input)
    raise ValueError(f"Unknown tool: {name}")

def run_agent(user_input: str, approve_fn) -> str:
    """
    approve_fn(name, input) -> bool : 危险工具的审批回调（测试时可 mock）
    返回模型最终文本。

    TODO 1: 写主循环，正确处理 stop_reason 的 end_turn / tool_use / max_tokens 分支。
    TODO 2: 追加历史时遵守顺序——先 assistant(含完整 content)，后 user(含全部 tool_result)。
    TODO 3: 对 DANGEROUS 里的工具，执行前调 approve_fn；被拒则用 is_error 回填
            "被拒 + 提示模型给替代方案"，而不是中断循环。
    TODO 4: execute_tool 抛异常时，catch 住并用 is_error 回填可操作的错误描述，
            不要让循环崩溃，也不要把原始 traceback 塞回去。
    TODO 5: 每个 tool_use 块都必须有一个 tool_use_id 匹配的 tool_result——
            即使被拒、即使报错，也必须回填，不能省略。
    """
    messages = [{"role": "user", "content": user_input}]
    # ... 你的实现 ...
    raise NotImplementedError
```

**测试要求**：
1. `run_agent("17 * 384 + 9001", approve_fn=lambda n, i: True)` —— 断言走了 calculator，结果含 `15529`。
2. `run_agent("转 100 块给 Alice", approve_fn=lambda n, i: False)` —— 断言审批被拒后，模型收到 is_error 并给出替代/澄清，**且没抛异常、没 400**。
3. 构造 `execute_tool` 内部抛异常（比如喂 calculator 一个非法表达式），断言循环用 is_error 回填且最终正常返回，而非崩溃。
4. （进阶）把 `max_tokens` 设极小触发截断，断言循环识别 `stop_reason==max_tokens` 并优雅处理。

**进阶（可选）**：给 `transfer_money` 加 `"strict": True` + `"additionalProperties": False`，验证模型产出的 `amount` 一定是 number、`to` 一定在；再写一个 `run_extractor(text)` 用 `tool_choice={"type":"tool","name":"transfer_money"}` 把一句自由文本（"给 Bob 打两百块"）强制抽成 `{to, amount}` 结构——体会"`tool_choice: tool` + `strict` = 结构化抽取器"这一形态，对比直接让模型吐 JSON 文本再 `json.loads` 的脆弱。

**暗坑提示**：
- 别忘了 `model="claude-opus-4-8"`（精确 ID，**不加日期后缀**）。
- 追加 `response.content` 时要追加**整个 content 列表**（含 text 块），不能只挑 tool_use。
- 一轮多个 tool_use 时，所有 tool_result 要放进**同一个** user 消息一次性回填，不是每个工具发一条 user 消息。
- 被拒/报错的 tool_result 也算"已回填"——漏掉任何一个 tool_use 的 result 就会 400。
- 用 `strict: True` 时**必须**同时写 `additionalProperties: False`，否则报错；且 strict 不支持数值范围/字符串长度约束，这类校验仍要你自己在 harness 里补。
- 不加 `strict` 时**绝不能假设 `block.input` 一定合法**——`transfer_money` 的 `amount` 可能是字符串甚至缺失，执行前要么开 strict、要么自己校验。
- `eval` 是教学示意，真要上生产请换安全的表达式求值（这本身也是"为什么危险动作要专用工具 + 沙箱"的活例子）。

---

## 本章心法

> **模型只发意图，系统才执行——工具的"形状"决定了你能管控它多少。** 把工具调用想成 syscall：模型陷入你的内核，你决定放行、门控、还是拒绝。会写循环只是入门；懂得按"可逆性"给动作分级、按"谁执行"选 client/server、用 `is_error` 把错误交还给模型自愈、知道何时必须手写循环插一脚——这才是从"调 API"跨到"造 Agent"的那一步。下一章的 MCP，是把这一章的工具"标准化、可复用"；第 12 章的 agent，是把这一章的循环"自主化、长程化"。地基在这里，打牢它。
