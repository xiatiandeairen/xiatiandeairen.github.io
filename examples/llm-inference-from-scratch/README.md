# LLM 推理引擎从零

配套书《LLM 推理引擎从零：用 TypeScript 手写一个会算真实 tok/s 的推理栈》的可跑代码。

**纯算法、离线、零运行时依赖。** 不需要任何 LLM、API key 或网络。所有权重由种子 PRNG 合成（不是训练出来的），所以模型输出是"结构正确的胡言"——这本书讲的是**推理引擎**（KV cache、批处理、分页、投机解码、量化），不是模型质量。引擎的正确性（缓存路径是否等于无缓存参考？批处理是否改变输出？int8 是否漂移 logits？）与模型好不好完全无关。

## 诚实数字约定

- **能 wall-clock 的真测**：tok/s、TTFT、per-token latency 全部用 `performance.now()` 实测。
- **内存按 IEEE-754 float64 payload 估算并标 `(est.)`**：不是测 RSS（GC 噪声大），而是对 model config 做精确算术——这正是真实引擎容量规划用的数。
- **等价性用 perplexity / logit-drift 证明**，不靠嘴说"没变质"。
- **toy 数据偏乐观**：合成模型小、kernel 是未优化的 float64 循环，所以**绝对** tok/s 偏悲观；能迁移到真引擎的是**相对**趋势与**加速比**。

## 跑法

先安装一次（只装 tsx / typescript / @types/node）：

```bash
npm install
```

每个 stage 一条命令（加载即跑，自带 `main()`）：

```bash
npm run stage01   # forward pass：无缓存参考前向，建立 baseline
npm run stage02   # KV cache：缓存路径 vs 无缓存，证明等价并测加速比
npm run stage03   # sampling：greedy / temperature / top-k / top-p / repetition penalty
npm run stage04   # continuous batching：连续批处理，多序列共享前向
npm run stage05   # paged KV：分页 KV cache，消除按步重分配的拷贝开销
npm run stage06   # speculative decoding：投机解码，draft+verify，perplexity 对拍
npm run stage07   # quantization：fp64→int8 权重量化，测 logit L∞ 漂移
npm run stage08   # benchmark：跨配置/优化的诚实记分卡汇总
```

类型检查：

```bash
npm run typecheck   # tsc --noEmit
```

## 共享底座 `src/core/`

所有 stage 复用 core，不重造算子与计时：

- `core/tensor.ts` — toy 张量与算子内核（matmul / addBias / rmsNorm / softmax / silu / rope / causalMask），row-major Float64Array，无 BLAS，慢但确定、可读、可数 FLOP。
- `core/model.ts` — toy LLaMA 风格 decoder（RMSNorm + RoPE + GQA + SiLU-gated FFN + 权重绑定）。`buildModel(cfg, seed)` 纯函数确定性；`forwardNoCache`（参考路径）与 `forwardStep`（缓存路径）两条前向供对照。
- `core/tokenizer.ts` — byte 级确定性 tokenizer（vocab=256，零词表文件），`encode` / `decode` 保证离线 round-trip，附 `PROMPTS`（≥3 个不同长度）。
- `core/metrics.ts` — 推理诚实记分卡（timeIt / tokensPerSecond / ttft / interTokenLatency / estimateKVBytes / formatBytes / speedup / perplexity / maxLogitDrift / argmax）。

底座自检（非 stage，验证 core 不变量）：

```bash
npx tsx src/core/_smoke.ts
```
