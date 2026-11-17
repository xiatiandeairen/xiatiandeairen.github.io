# 阮一峰 锚点语料

**主力题材**：信息聚合 / 周刊体 / 技术观察

## 选文标准（博主执行）

- 5-8 篇代表作，覆盖至少 3 个 spec 可能的题材
- 优先"科技爱好者周刊"里被他本人反复引用/精选的期
- 不选纯工具介绍类（过于短平快，风格对照弱）

## 文件命名约定

```
{年份}-{slug}.md
```

例：`2024-asymmetric-returns.md`

## 文件内容模板

```markdown
<!--
source_url: https://www.ruanyifeng.com/blog/2024/XX/...
captured_at: 2026-04-22
author_rating: high | reference | low     # 博主对该篇的评级（Pilot 会用）
genre: 信息聚合 | 思辨随笔 | 工程深度
word_count: 3200
-->

# 原文标题

原文正文（纯文本，去掉图片/代码块以外的 HTML）...
```

## TODO（博主）

- [ ] 收集 5-8 篇（建议至少 1 篇标为 `high`、1 篇标为 `low`，供 Pilot 作 golden pair）
- [ ] 每篇字数 >1500，长度与 skill 产出可比
- [ ] 确认版权上不发布到公开（本目录仅 git 内部使用，且非 public build 依赖）
