# Design Archive

设计方向归档。这些文件**不会**打包到生产站点。

## 目录结构

```
design-archive/
└── YYYY-MM-DD-<主题>/
    └── <name>.astro          # 独立可渲染的设计 mockup 页面
```

每个归档目录对应一次设计探索（例如一次重设计决策前的多方向对比）。

## 查看归档

```bash
./scripts/view-design.sh <folder-name>
```

脚本会：
1. 复制 `design-archive/<folder-name>/` 下所有 `.astro` 到 `src/pages/preview/`
2. 启动 dev server
3. Ctrl+C 退出时自动清理 `src/pages/preview/`

访问：`http://localhost:4321/preview/<filename>`（去掉 `.astro`）。

## 归档规范

- 新增归档用 `YYYY-MM-DD-<kebab-case-主题>` 命名目录
- 每个 `.astro` 文件要能独立渲染（引用站点内 layout / token 没问题，但不要依赖业务数据）
- 在目录内加一个 `notes.md` 记录背景（可选）：为什么做这次对比、最终选了哪个方向

## 现有归档

- `2026-04-17-article-header/` — 文章 header 重设计，4 个方向（极简文艺 / 古典书籍 / 现代科技博客 / 报纸专栏）。最终选定 C 方向后又切到 A。
