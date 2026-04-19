# 多约束布局直接打补丁必失败

## 现象

首页 hero + sidebar 布局迭代 18 轮未收敛。每轮满足用户最新一条反馈，打破上轮建立的约束。

## 根因

1. 约束散在多轮对话中，从未汇总成表
2. 约束之间物理冲突（评论吸底 + 摘要撑满 + 固定高度 + 变长文字）
3. Edge case（评论 0 条 / 摘要过短 / 标签 3 行）直到末期才被提起
4. 验证只看当前 viewport 视觉，不测数值 → 列宽被 chips nowrap 撑反了才发现

## Guard 规则

**信号**（满足任一立即停手）：
- 用户反馈"又不对了" / "变形了" ≥2 次
- 同区域连续改动 >3 轮未收敛
- 本次改动影响 >1 个视觉维度（宽/高/间距/字号）
- 布局用 flex/grid + ≥3 条约束

**动作**：跑 [Constraint Protocol](../../docs/methodology/design-iteration.md) 5 步
1. 约束清单（表格列全部约束，不只最新）
2. 矛盾检查（两两配对）
3. Edge case 枚举（变量最小/正常/最大）
4. 锁变量（line-clamp / max-height / 占位补齐）
5. 数值验证（playwright getBoundingClientRect，不用肉眼）

## 反例

v9 给 chips 加 `nowrap + flex-shrink: 0` 防标签换行溢出。结果 chips 的 min-content = 所有 chip 总宽，通过 sidebar 撑爆 grid `1fr` 的 auto min-width，导致左右列宽反向（main 321, side 497）。

## 经验

设计迭代的本质矛盾：用户也不知道所有约束，要通过迭代暴露。AI 的任务不是一次做对，而是**每轮迭代时先把所有已知约束列出来，识别冲突，明确本轮打破了哪条，让用户拍板取舍**。
