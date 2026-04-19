# Design Governance 架构设计

## 1. 定位与边界

### 职责

让 [ui-architecture](ui-architecture.md) 描述的静态架构**持续保持有效**。提供 4 个治理机制（思考工具 / 资产白名单 / 预测工具 / 自动化扫描），让"改动半径可预测、抽象不过早、死代码不堆积"成为可执行的纪律。

### 不负责

- UI 静态架构本身（→ [ui-architecture](ui-architecture.md)）
- token / pattern 内容（→ [design-tokens](design-tokens.md), [design-patterns](design-patterns.md)）
- 单次设计探索流程（→ [design-archive](design-archive.md)）

## 2. 结构与交互

### 组件图

```
[触发场景]                    [治理机制]                       [输出]

新写共享样式            ──→  Rule of Three (思考)         ──→  内联 / 提升 决策
                                    │
要改共享层文件          ──→  变更前 grep (预测)            ──→  影响半径报告
                                    │
增删共享 class          ──→  Primitive 登记表 (白名单)     ──→  登记同步 / 删除
                                    │
push / PR              ──→  lint:unused (CI 自动)         ──→  阻断 / 放行
```

### 组件表

| 组件 | 性质 | 触发时机 | 强制方式 |
|------|------|---------|---------|
| **Rule of Three** | 思考工具 | 新写共享样式时 | 人工纪律 + 文档约定 |
| **变更前 grep** | 预测工具 | 改 tokens / base / layout / components / global / components/*.astro 前 | 人工纪律 |
| **Primitive 登记表** | 资产白名单 | 增删 components.css 共享 class 时 | 人工同步 + 偶发审计；不在表 = 待删 |
| **lint:unused (CI)** | 自动化扫描 | 每次 push / PR | 退出码 1 → CI 阻断合并 |

#### Rule of Three（抽象门槛）

| 使用次数 | 放哪 |
|---------|------|
| 1 | 页面内联 `<style>` |
| 2 | 页面内联（仍可复制） |
| **3+** | 提升到 `components.css` 或 `components/*.astro`，必须登记 §2.3 |

**禁止防御性抽象**：不为"可能以后用到"写共享代码。回滚错误抽象的成本远高于暂时重复。

#### 变更前 grep（影响半径预测）

改任一共享层文件前运行：

```bash
grep -rlE "\b<target-class-or-component>\b" src/
```

输出 = 所有受影响的文件。几秒钟就知道波及面。页面私有内联样式改动免 grep。

#### 2.3 Primitive 登记表

所有 `components.css` 共享 class 必须出现在此表。**不在表里 = 待删除**。

| Class | 用途 | 使用方 | 修改影响面 |
|-------|------|-------|----------|
| `.section-label` | section 标题（小字母距大写 + 底线） | Sidebar, index, about, archive, [slug] (8 处) | 全站所有 section 标题 |
| `.cat-label` | 分类小标签 | NoteCard, archive, search, [slug] (6 处) | 所有文章展示位 |
| `.archive-item` | 归档/搜索结果列表项 | archive, index, search (6 处) | 列表页 hover 行为 |
| `.layout-featured-secondary` | 双栏 featured 区布局 | index, layout.css (3 处) | 首页 featured 区 |
| `.filter-btn` | 归档筛选按钮 | archive, layout.css (3 处) | archive 页筛选 |
| `.reading-progress` | 阅读进度条 | [slug], global.css | 文章页顶部 1px 进度 |
| `.back-to-top` | 回到顶部链接 | [slug], global.css | 文章页 |
| `.card` | 基础卡片容器 | BaseLayout (1 处) | 保留：v1 统一卡片表面样式，即使目前只有 1 处消费也作为 primitive 锚点 |
| `.remaining-section` | 首页 "继续阅读" 区块容器 | index, en/index (2 处) | 首页 featured 区外的次要列表容器 |

**单文件私用但放在 components.css 的类**（v1 保留；post-launch backlog 再评估是否下沉到 scoped）：
- `.article-card-*` 9 个 → 主要消费 `ArticleCard.astro` + index.astro 模板
- `.about-*`, `.value-card`, `.avatar-box`, `.contact-*`, `.colophon`, `.reader-letter-*` → 主要消费 `about.astro`
- `.post-card*`, `.post-list` → 主要消费 `NoteCard.astro`

保留理由：迁移需要 scoped CSS 回归验证 + 打破多页面共享路径；对上线无影响，作为架构 backlog 不阻塞。

#### lint:unused (CI 自动扫描)

```bash
npm run lint:unused                       # 跑两个扫描（本地随手跑）
./scripts/lint-unused-css.sh              # 扫描 src/styles/*.css 中 0 引用 class
./scripts/lint-unused-components.sh       # 扫描 components/*.astro 中 0 import
```

CI 入口：`.github/workflows/lint.yml`（push + PR 触发，任一脚本失败 → 整个 workflow 红 → 合并被拦）。

报告的死代码要么立即删除，要么在 §2.3 表中登记合法用途。

### 数据流

```
[源码状态]                       [扫描脚本]                       [CI 门]
src/styles/*.css     ─┐
src/components/*.astro─┼──→  lint-unused-css       ─┐
src/pages/*.astro    ─┘                            ├──→  lint.yml workflow ──→ 阻断 push/PR
                          lint-unused-components ──┘
                                    │
                                    │ 输入参考
                                    ▼
                          §2.3 Primitive 登记表
                          （白名单：在表里 = 合法死代码）
```

| 来源 | 目标 | 数据格式 | 类型 | 说明 |
|------|------|---------|------|------|
| `src/styles/*.css` 定义 | `lint:unused-css` | class 名集合 | 强 | 缺失检查 → 死代码积累 |
| `src/**/*.{astro,ts,js}` 引用 | `lint:unused-css` | grep 匹配 | 强 | 用法判定依据 |
| `src/components/*.astro` | `lint:unused-components` | import 语句 | 强 | 0 import 即死 |
| 两脚本退出码 | `.github/workflows/lint.yml` | 0 / 1 | 强 | 非 0 → CI 红 |
| §2.3 登记表 | 人工 / 未来 audit 工具 | Markdown 表 | 弱 | 暂无机器对账，靠纪律同步 |

## 3. 设计决策

### 驱动因素

| 因素 | 类型 | 对架构的影响 |
|------|------|------------|
| 综合迭代成本最低 | 业务需求 | "改动半径可预测 + 抽象延迟 + 死代码机械化清理" 三根支柱 |
| 个人项目无 code review | 技术约束 | 治理必须自动化（CI 强制），不能靠人工纪律 |
| 设计探索频繁 | 业务需求 | Rule of Three 防过早抽象；登记表防共享层膨胀 |
| 改动影响难直观判断 | 质量要求 | 提供 grep 工具替代心算 |

### 关键选择

| 决策 | 选择 | 被拒方案 | 为什么 |
|------|------|---------|--------|
| 抽象门槛 | Rule of Three（≥3 次才提升） | 立即抽象 / 永远内联 | 立即抽象错误率高、回滚贵；永远内联让设计漂移 |
| 共享 class 治理 | 白名单登记表 | 黑名单 / 仅靠 grep | 白名单防共享层膨胀；登记需列使用方强制反查 |
| 死代码检测 | grep + 脚本 + CI 阻断 | 人工 review / 定期清理 | 个人项目无 review；机械化才能不积累 |
| 检测时机 | push + PR 触发 | 每天定时跑 / 仅 PR | push 即时反馈；PR 兜底；定时浪费 actions 额度 |
| 扫描粒度 | 全部 `src/styles/*.css`（不只 components.css） | 仅 components.css | 死代码可能藏在任何 css 文件 |

### 约束

- **禁止**共享层（components.css / global.css / components/*.astro）新增样式不登记（违反"不在表里 = 待删"）
- **禁止**绕过 lint.yml CI 强行合并（死代码会积累）
- **必须**改共享层前 `grep -rlE "\b<target>\b" src/` 预判波及面
- **必须**删除共享 class 同时更新 §2.3 登记表（否则文档骗人）

## 4. 质量要求

| 属性 | 指标 | 目标 |
|------|------|------|
| 死代码积累 | `lint:unused` 报告的 class / 组件数 | 0（CI 强制） |
| Primitive 登记完整 | §2.3 登记表 vs 实际共享 class 一致率 | 100% |
| 改动半径可预测 | 共享层改动前 grep 覆盖率 | 100%（人工纪律） |
| CI 反馈速度 | lint.yml 从 push 到结果 | <30s（实测约 10s） |
| Rule of Three 遵循 | 共享层新增前 inline 阶段比例 | ≥80%（人工估算） |
