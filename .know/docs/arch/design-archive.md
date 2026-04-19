# Design Archive 架构设计

## 1. 定位与边界

### 职责

负责候选设计方向的 mockup 归档与本地预览机制，让设计探索期产生的多套候选方案可以保留在 repo 内、按需拉起 dev server 对比，但不参与生产构建、不污染站点路由。

### 不负责

- 已沉淀的设计 token / pattern / 治理规则（→ [design-tokens](design-tokens.md), [design-patterns](design-patterns.md), [design-governance](design-governance.md)）
- 选定方案在生产页面的应用（→ `src/pages/` / `src/components/`）
- 未来 PR 流程的设计 review 工具（暂未做）

## 2. 结构与交互

### 组件图

```
[design-archive/]              [scripts/]                  [src/pages/preview/]
  README.md                     view-design.sh             （gitignored，临时存在）
  YYYY-MM-DD-<topic>/                │
   ├── *.astro     ───复制───────────┘─────────────────→     (动态填充)
   └── notes.md (可选)             启动 npm run dev                │
                                  trap EXIT 清理                  ▼
                                                          [Astro dev server]
                                                          /preview/<filename>
```

### 组件表

| 组件 | 职责 | 边界规则 |
|------|------|---------|
| `design-archive/` | 永久保存历次设计探索的 mockup（含 README） | 必须按 `YYYY-MM-DD-<topic>/` 命名；mockup 须能独立渲染；不参与生产构建 |
| `scripts/view-design.sh` | 复制归档到 gitignored 临时目录 + 启动 dev + 退出清理 | 必须 trap EXIT/INT/TERM 清理；无参数运行须列出归档目录 |
| `src/pages/preview/` (gitignored) | 临时承载预览路由 | 必须 `.gitignore` 排除；不能被任何业务代码 import |
| `.gitignore` 条目 | 阻止 preview 目录入库 | 必须保留 `src/pages/preview/` 行，不可删除 |
| `package.json design:view` 脚本 | 暴露用户友好命令 | 别名指向 view-design.sh，不重复实现 |

### 数据流

```
[人] 新建 design-archive/<dated-folder>/<name>.astro
       │
       ▼
[CLI] ./scripts/view-design.sh <folder>     # = npm run design:view <folder>
       │
       ├── cp design-archive/<folder>/*.astro src/pages/preview/
       ├── echo 预览 URL 列表
       ├── npm run dev (前台)
       └── trap EXIT/INT/TERM → rm -rf src/pages/preview/
       │
       ▼
[浏览器] http://localhost:4321/preview/<filename>
```

| 来源 | 目标 | 数据格式 | 类型 | 说明 |
|------|------|---------|------|------|
| `design-archive/<folder>/*.astro` | `src/pages/preview/` | Astro 文件 | 强 | 复制即预览源 |
| `view-design.sh` | dev server | 通过启动 `npm run dev` | 强 | 前台运行直到 Ctrl+C |
| 退出信号 | preview 目录 | 删除操作 | 强 | trap 失效会留垃圾 |
| `design-archive/` | 生产 build | **无**（被 Astro 忽略，因不在 src/pages/） | 弱 | 不变量 |

## 3. 设计决策

### 驱动因素

| 因素 | 类型 | 对架构的影响 |
|------|------|------------|
| 设计需要并排比较 4-5 个候选方向 | 业务需求 | 必须能挂在真站点的 token / layout 下，截图对比不够 |
| 候选方案不该污染生产 | 业务需求 + 质量要求 | 路径放 `design-archive/`（不在 `src/`），脚本中转目录 gitignored |
| 历次决策需要可追溯 | 业务需求 | 归档不删；按日期命名便于检索 |
| 启动成本必须低 | 质量要求 | 一条命令完成（复制 + dev + 退出清理），不要求人工记忆 |

### 关键选择

| 决策 | 选择 | 被拒方案 | 为什么 |
|------|------|---------|--------|
| 归档位置 | repo 根 `design-archive/` 入库 | 单独 fork / draft 分支 / 截图群里 | 入库可追溯 + 与代码同行；fork/分支重；截图脱离 token |
| 预览机制 | 临时复制到 gitignored `src/pages/preview/` | Astro config glob 增加扫描路径 / 独立 Astro 子项目 | 临时复制对 build 零影响；config 扩展易出意外；子项目重 |
| 退出清理 | bash `trap EXIT/INT/TERM` | 提示用户手动删 / cron 定期清 | 自动 + 即时；手动易忘；cron 重 |
| 命名规范 | `YYYY-MM-DD-<kebab-topic>/` | 自由命名 / `<topic>-vN/` | 日期排序便于回溯；topic 易识别 |

### 约束

- **禁止** `design-archive/` 内代码被 `src/` 任何文件 import（违反隔离）
- **禁止**删除已归档目录（破坏决策追溯）
- **必须** mockup 引用站点真 token / layout，不写死 magic 数字（否则对比失真）
- **必须**新归档目录在 [design-archive/README.md](../../../design-archive/README.md) "现有归档"段同步登记

## 4. 质量要求

| 属性 | 指标 | 目标 |
|------|------|------|
| 生产构建隔离 | `dist/` 中含 preview 路由数 | 0（实测：32 页生产构建无 preview） |
| 退出清理可靠性 | Ctrl+C 后 `src/pages/preview/` 残留文件数 | 0（trap 兜底） |
| 启动延迟 | `view-design.sh` 到 dev server ready | <8s（实测约 5s） |
| 归档保留率 | 历史归档保留比例 | 100%（不删除政策） |
