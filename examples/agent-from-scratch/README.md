# agent-from-scratch

从零用 TypeScript 拆解一个生产级 Agent 的核心能力。配套书库合集《Agent 工程》——每个 stage 对应一章，章里讲「为什么」，这里是「能跑的怎么做」。

不用任何 agent 框架（没有 LangChain / LlamaIndex）。唯一外部依赖是 `@anthropic-ai/sdk`，而且是可选的：**没有 API key 也能跑**——每个 stage 都带一个确定性的离线 mock，机制是真的，只有模型被脚本化。这样你读代码、改代码、看输出，都不需要花钱或联网。

## 跑起来

```bash
npm install
npm run stage01          # 控制循环
npm run stage02:tools    # 工具调用往返
npm run stage02:mcp      # 手写 MCP server + client（stdio JSON-RPC）
npm run stage03          # 执行沙箱（macOS sandbox-exec）
npm run stage04          # 权限：allow / ask / deny + 注入演示
npm run stage05          # 上下文工程：压缩 / 卸载 / KV cache
npm run stage06          # 记忆系统：分层 + 召回 + 投毒演示
npm run typecheck        # 全量类型检查
```

想用真模型驱动同样的循环：

```bash
cp .env.example .env     # 填入 ANTHROPIC_API_KEY
```

之后所有 stage 自动走真 SDK，机制代码一行不变（见 `src/core/llm.ts::createLLM`）。

## 结构

```
src/
  core/
    types.ts     # agent 在 wire 上传的最小类型（Anthropic Messages API 的子集）
    llm.ts       # LLM 抽象：真 SDK 适配器 + 离线 mock，二者同一接口
  stage01-loop.ts        # 第 1 章：循环 + 停止条件 + token 累积曲线
  stage02-tools.ts       # 第 2 章：function calling 往返 + 工具粒度
  stage02-mcp.ts         # 第 2 章：最小 MCP 协议（initialize/tools/list/tools/call）
  stage03-sandbox.ts     # 第 3 章：隔离执行，攻击用例验证拦截
  stage04-permissions.ts # 第 4 章：权限三态 + lethal trifecta 演示
  stage05-context.ts     # 第 5 章：压缩 / 卸载 / 缓存前缀
  stage06-memory.ts      # 第 6 章：工作/情景/语义记忆 + 投毒
```

## 设计约定

- **一个 LLM 接口** (`core/types.ts::LLM`)：agent 就是绕这一个 `generate` 调用转的循环。真适配器和 mock 都实现它。
- **诚实数字**：token 用 `~4 字符/token` 估算（不是真 tokenizer，代码里标注了），有 key 时换成 API 实测 usage。
- **机制离线可验证**：沙箱真的拦截、MCP 真的握手、权限真的拒绝——这些不依赖模型，纯 Node 内置即可复现。
