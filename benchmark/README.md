# Content-Quality Benchmark（中文长文盲评基准）

> **与 `article/benchmark/` 的区别**：`article/benchmark/` 测 skill **pipeline 步骤完整性**（单元测试性质）；本 benchmark 测 skill **最终产出文章的质量**（相对位次）。两者不替代。

## 核心立场

这是 **反自我欺骗** 基准：

- 评审器不是 Claude（避免 self-enhancement bias；skill 跑在 Claude）
- 外部锚点 = 阮一峰 / 王垠 / 云风，**skill 必须在这些人中间排出具体位置**，否则不叫"高质量"
- 禁分维度打分（历史已证在本项目饱和）
- 每次 run 必须先跑 **calibration**，评审器通过不了就整 run 作废
- 禁"部分通过重试" — 是自欺入口

## Pilot 硬门槛（阻断式）

**Pilot 不通过 → 整个 benchmark 不上线，sprint 终止**。不得以"通过率差一点"为由降低门槛。

Pilot 规格：

- 4 对 golden pairs = 阮/王/云 各 1 对（本人"代表作 vs 一般水位文"）+ 博主本人 1 对
- 每对 × 3 judge × 2 位置（交换 A/B）= 24 次 pairwise 判决
- **通过标准**：≥3/4 对判对（"好"排在前）+ 位置交换内部一致性 >70%
- 通不过时的 exit 1 报告会指出：哪对判错 / 哪 judge 不可靠 / 需要换 judge 还是重塑方向

## 目录

```
benchmark/
├── corpus/              # 文本语料
│   ├── ruanyifeng/      # 阮一峰锚点（5-8 篇）
│   ├── wangyin/         # 王垠锚点（5-8 篇）
│   ├── yunfeng/         # 云风锚点（5-8 篇）
│   ├── self/            # 博主本人历史文（5 篇）
│   └── escape/          # 逃逸用例：故意写坏的文章（3-5 篇）
├── specs/               # 15 条测试 spec（3 题材组 × 5）
├── pilot/               # 4 对 golden pairs + 运行脚本
├── judge/               # Judge prompt + ensemble 配置（GPT-4o + Gemini + DeepSeek/Qwen）
├── scripts/             # 主 benchmark、call-judge、Elo 聚合
└── runs/                # 每次 run 的产出（gitignored）
```

## 一次 run 流程

```
run-benchmark.sh
  1. 重跑 calibration（Pilot 的 4 对）  →  <75% 则本次 run 作废
  2. article skill 按 15 条 spec 产文   →  runs/{ts}/generated/
  3. 组 pair 矩阵：skill vs 每个对照组
  4. 3 judge × 2 位置 并行 pairwise 判决
  5. Elo 聚合 + 1000 次 bootstrap CI
  6. 逃逸 sanity：escape rating 必须 < 1250，否则 run 作废
  7. 生成 runs/{ts}/report.md
```

## 成本估算

- 单 run ≈ 500 次 pairwise 判决
- 每次 ~3-5k tokens input + 500 tokens output
- 综合 ≈ $8-15/run（GPT-4o + Gemini + DeepSeek 加总）

## 运行

```bash
# 一次性：设置 3 个 judge 的 API key（见 judge/judges.yaml）
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
export DEEPSEEK_API_KEY=...

# 先跑 Pilot 校准评审器（必须通过才能进下一步）
./pilot/run-pilot.sh

# Pilot 通过后，跑完整 benchmark
./scripts/run-benchmark.sh
```

## 输出解读

`runs/{ts}/report.md` 包含：

- **Rating 表**：阮/王/云/self/skill/baseline/escape 各自 Elo ± 95% CI
- **Skill 位次**：X/7
- **Per-spec 胜率**：15 行 × 6 对手
- **Judge 分歧率**：3 judge 两两 tie 率
- **Red flags**：calibration 未通过的 pair / verdict 方差异常的 spec

**Rank 变化 1 位** 才算改进信号；CI 重叠则视为"无显著改进"。

## 不做（v2）

- 公开 benchmark / 排行榜
- 跨 LLM 对比（换底层模型跑同 spec）
- Rubric 多维展开
- 英文锚点

见 sprint handoff：`.sprint/{sprint-id}/handoffs/design.md` 的 Decision Register 与 Constraints。
