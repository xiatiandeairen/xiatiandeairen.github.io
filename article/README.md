# article

把 AI 对话中的碎片知识 → 高质量文章 → 发布到个人站。

v1 仅服务 **AI skill 经验类文章**。

## 安装

本目录位于站点仓库内（`<site_repo>/article/`），跟随站点仓库 git 管理。

1. Claude Code skill 识别：在 `~/.claude/skills/` 下建软链
   ```bash
   ln -s "$(pwd)" ~/.claude/skills/article
   ```
2. 站点路径：**无需手动配置**。脚本优先级：`$ARTICLE_SITE_REPO` > `config.json` > `git rev-parse --show-toplevel`（即本仓库根）。
3. 站点结构假设：Astro，文章在 `<site_repo>/src/content/notes/{N}-{slug}.md`。

## 用法

```
/article capture "<tag>" "<一句话观点>"   # 在对话中即时保存素材
/article draft   "<topic>" "<观点>"       # AI 起草
/article gate    <slug>                   # 追加发布前 checklist
/article publish <slug>                   # 拷贝到站点 notes/，待手动 commit
```

## 流程

```
对话中产生想法
   └─ /article capture           → inbox/{ts}-{slug}.md
聚到几条素材后
   └─ /article draft <topic>     → drafts/{slug}.md (五节骨架填充)
草稿打磨后
   └─ /article gate              → 草稿末尾追加 checklist
逐条勾选 + 改 frontmatter gate_passed: true
   └─ /article publish           → 站点 notes/，提示手动 commit
```

## 设计取舍

- **半自动**：AI 起草 + 用户审校；不自动 push。
- **不调 API**：AI 扩写 = 当前 Claude 主会话直接生成，零额外 token 成本。
- **质量闸门 = 人工 checklist**，不假装 AI 自判。
- **v1 单一 blueprint**：不预留模板系统，未来 v2 加第二个时再抽象。

## 文件

- `SKILL.md` — Claude Code skill 入口
- `scripts/article-ctl.sh` — bash 总控
- `blueprint.md` — 文章骨架 + AI 扩写 prompt
- `quality-checklist.md` — 发布前 4 条 checklist
- `config.json` — 本地配置（gitignored）
- `inbox/` `drafts/` — 运行时目录（inbox gitignored）
