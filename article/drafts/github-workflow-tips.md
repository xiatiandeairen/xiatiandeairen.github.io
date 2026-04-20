---
title: "AI 项目里我靠 GitHub 工作流兜底的三件事"
slug: "github-workflow-tips"
createdAt: "2026-04-20T11:54:59Z"
updatedAt: "2026-04-20T11:54:59Z"
date: "2026-04-20T11:54:59Z"
question:
  type: "工程实践"
  subType: "CI/CD"
quality:
  overall: 7
  coverage: 7
  depth: 7
  specificity: 8
  reviewer: "ai"
analysis:
  objectivity:
    factRatio: 0.6
    inferenceRatio: 0.25
    opinionRatio: 0.15
  assumptions:
    - "读者用过 GitHub Actions"
    - "读者在维护一个有 build 步骤的内容/代码项目"
  limitations:
    - "示例都来自 Astro + pnpm 项目，具体脚本名对其他技术栈需改写"
    - "未覆盖 matrix build、cache 调优等进阶场景"
review:
  status: "draft"
tags:
  - name: "github"
  - name: "actions"
    parent: "github"
  - name: "ci"
    parent: "engineering"
  - name: "workflow"
topics:
  - name: "工程实践"
  - name: "AI 工程化"

template: skill-experience
gate_passed: true
quality_gate:
  contrarian_view: done
  concrete_example: done
  takeaway: done
  real_skill: done
---

## 起源: 遇到的真实问题

我维护一个个人内容站 `xiatiandeairen.github.io`——Astro + Tailwind，GitHub Pages 部署。文章是 markdown 文件，frontmatter 要对齐 20 多个字段：title、date（必须 ISO8601 带 Z 后缀）、quality 四项 0-10 数字、objectivity 三个比例和必须等于 1.0、tags 必须是对象数组不是字符串数组。

起初我靠"写的时候小心点"。结果每隔两三篇就会炸 build：漏一个字段、比例打成 0.5/0.3/0.3（和是 1.1）、tag 写成 `["rust"]` 而不是 `[{name: "rust"}]`。本地看着没事，push 到 GitHub 后 Pages 构建失败，文章发不出去——而我已经关掉 IDE 去干别的了，晚上才收到邮件告警。

更烦的是，和我合作写草稿的 AI 也记不住。哪怕我在 CLAUDE.md 里写了"记得 factRatio + inferenceRatio + opinionRatio 必须等于 1.0"，AI 还是会偶尔给出 0.5 + 0.3 + 0.3。不是它不看规则，是规则太多、AI 不会每次都把所有 rules 完整 apply 一遍。

## 尝试过的方案

最直觉的方案是**加强提交前校验**。装 husky 或原生 git hook，在 pre-commit 里跑 frontmatter 校验，不通过就拦住 commit。

我没选这个。两个理由：

一是**本地 hook 有绕过成本**——`git commit --no-verify` 一键跳过，AI 偶尔自己就会跳（尤其在 rebase 或 cherry-pick 时）。二是**本地 hook 和 CI 会分叉**——本地装了一版校验脚本，CI 装了另一版，哪天规则改了忘同步，两边不一致，bug 只在其中一边暴露。

另一个方案是**把规则塞进 CLAUDE.md 里**，让 AI 写文章时自觉遵守。这个我试过——有用，但不够。上面说了，AI 在长 session 里会漏字段；人也一样，一周不写文章再回来，就要重新读 CLAUDE.md 找规则。靠记忆的东西不可靠。

## 踩的坑

真正的坑不是 build 失败本身，是**我以为"build 成功"就等于"内容合规"**。

站点早期我没上 zod schema。那时 frontmatter 是"软约定"——Astro content collection 没配 schema，字段随便填都能构建。结果站点上线后，某篇文章的 date 写成了 `"2024-03-01"`（缺时间和 Z 后缀），前端渲染逻辑按 ISO8601 解析，fallback 到了 `Invalid Date`，排序炸了但不报错——最新文章跑到了倒数第二页。

我以为这类问题会"一跑就发现"。没有。**不校验的字段等于没约定**。它只会在某个你想不到的地方静默地错。

后来加了 `src/utils/schema.ts` 的 zod schema，build 时对每篇文章跑一遍 `NoteFrontmatterSchema.parse`。这一加就抓出三个字段历史遗留的脏数据——都是"看起来对、其实错了 3 个月"的那种。

## 最终设计

现在我靠三个独立的 GitHub 工作流兜底，每个管一件事。

**第一件：build 期用 zod 校验 frontmatter**。`src/utils/schema.ts` 里定义 schema，Astro content collection 引用它。每次 `pnpm build` 都跑一遍。字段缺失、类型错、objectivity 三比例和不等于 1.0（允差 0.01），构建失败：

```ts
const ObjectivitySchema = z.object({
  factRatio: z.number().min(0).max(1),
  inferenceRatio: z.number().min(0).max(1),
  opinionRatio: z.number().min(0).max(1)
}).refine(
  (data) => {
    const sum = data.factRatio + data.inferenceRatio + data.opinionRatio;
    return Math.abs(sum - 1.0) < 0.01;
  },
  { message: 'factRatio + inferenceRatio + opinionRatio must equal 1.0' }
);
```

**第二件：CI 里扫死代码**。`.github/workflows/lint.yml` 在每次 push 和 PR 上跑两个脚本：`lint-unused-css.sh` 扫出 `src/styles/` 里定义了但 grep 不到任何引用的 class，`lint-unused-components.sh` 扫 `src/components/` 里没被任何页面 import 的组件。重构后留下的僵尸代码、改名忘了删旧的 class，都在这一步被拦。

**第三件：部署和 release 解耦**。两个独立 workflow：

```yaml
# .github/workflows/pages.yml
on:
  push:
    branches: [main]
# .github/workflows/release.yml
on:
  push:
    tags: ['v*']
```

日常改文章 → push main → Pages 自动部署（读者 5 分钟内看到）。达到一个里程碑 → 打 `v1.2` tag → release.yml 调 `action-gh-release@v2` 自动生成 release notes。两者互不依赖。release 失败不会阻塞部署，部署有问题也不会污染 release 历史。

取舍：我放弃了"每次 commit 都跑完整校验"的即时反馈。本地只跑 typecheck，zod schema 校验和 lint:unused 留给 CI。推到 main 失败就 revert 一下，比本地每次都等 30 秒要值。放弃本地 hook，换来的是"规则只有一份、在 CI 里"。

## 反共识 takeaway

**CI 是 skill 的一部分**。把规则写进 CLAUDE.md / README / 文档，你是在指望人（或 AI）下次记得。把规则写成 CI 脚本，你是在让系统替你执行。两者对可靠性的要求差一个数量级。

推论一：**"公开值"也不要硬编码**。这个仓库的 Cloudflare Analytics beacon token 本来是硬编码的——我想"反正是公开的，前端 render 后就暴露了"。但后来想轮换 token，发现要改代码提 PR，成本意外地高。改成走 `secrets.CF_ANALYTICS_TOKEN` 注入后，轮换只要在 repo Settings 里改一次。不是为了安全，是为了**轮换成本**。

推论二：**部署和 release 一定要分开**。Pages 持续部署（main push），release 是人为标记的稳定点（tag push）。如果合并成一个 workflow——"打 tag 才部署"——你会因为"这次改动太小不值得打 tag"而推迟部署；如果另一个极端——"每次 push 都当 release"——你会失去"这个版本是稳定的"这一层语义。

**何时不要这样做**：项目特别小（<10 个文件，零贡献者）、规则只有 1-2 条、你每天都在动它——这种情况下本地记忆足够，CI 是过度工程。CI 兜底的价值随项目寿命、规则数量、协作者数量指数级上升。

<!-- ARTICLE_GATE_CHECKLIST -->
# 发布前 Checklist

> `/article gate <slug>` 会把以下内容追加到 `drafts/<slug>.md` 末尾。
> 逐条勾选；任一未勾 → 不要 publish。
> 全部勾选 → 在 frontmatter 把 `gate_passed: false` 改为 `true`。

## 4 条硬标准

- [ ] **反共识观点** — 文章中至少有 1 条"与直觉/主流做法相悖"的结论。位置：__________
- [ ] **具体例子** — 至少 1 个非抽象论述（命令 / 代码 / 数字 / 对话片段 / 截图）。位置：__________
- [ ] **可带走的 takeaway** — 读者读完能在自己工作里立即用上的一条建议。位置：__________
- [ ] **对应真实 skill 或实践** — 不是纯理论；写明对应的 skill 名 / 工具 / 项目。对应：__________
- [ ] **frontmatter 补齐** — title 非空；question.type 填具体分类；quality 四项均为 1-10 的数字（非 0 非 pending）；analysis.objectivity 三个比例和 = 1.0；tags/topics 填为 `[{name: ...}]` 对象数组。

## 自检问题（不打分，逼自己回答）

- 这篇文章只有我能写吗？还是网上随便搜搜就能拼出来？
- 把第一段删掉，还能读得下去吗？（开头是不是在凑字）
- 反共识 takeaway 那条，半年前的我会同意吗？（如果会，说明不够反共识）
- 推给一个完全不认识我的人，他读完会不会"哦，然后呢？"
