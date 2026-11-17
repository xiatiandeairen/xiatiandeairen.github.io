# Specs 数据集（15 条）

每个 spec 是一个独立 `.yaml`，对应一次 skill 产文任务。

## Schema

```yaml
id: essay-slug                    # 唯一，文件名须一致
title: 文章标题
group: 思辨随笔 | 信息聚合 | 工程深度
expected_length: 2500-4000        # 字数区间（中文字符）
anchor_refs:                      # 同题材的锚点文章，benchmark 对照用
  - ruanyifeng/2024-XXX
  - self/2024-YYY
prompt_for_skill: |               # 交给 article skill 的完整 brief
  （博主填入。包含：题材、角度、禁忌、目标读者、"必须回答的核心问题"）
author_rating_target: high        # skill 理应达到的档位
```

## 分布（硬约束）

| 题材组 | 数量 | 锚点主力 |
|--------|------|---------|
| 思辨随笔 | 5 | 王垠 + 博主本人 |
| 信息聚合 | 5 | 阮一峰 |
| 工程深度 | 5 | 云风 |

## 选题原则

**由锚点现有文章反推题目**，不凭空出题。锚点不存在同题材文 = 对照失效 = spec 无效。

## TODO（博主）

- [ ] 每个 stub yaml 的 prompt_for_skill 填入真实 brief
- [ ] anchor_refs 指向 corpus/ 下实际存在的 .md
