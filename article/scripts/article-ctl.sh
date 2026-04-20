#!/usr/bin/env bash
# article-ctl.sh — article skill 总控
set -euo pipefail

ROOT="${ARTICLE_ROOT:-/Users/taoxia/Workspace/self/xiatiandeairen.github.io/article}"
INBOX="$ROOT/inbox"
DRAFTS="$ROOT/drafts"
CONFIG="$ROOT/config.json"
CHECKLIST="$ROOT/quality-checklist.md"

mkdir -p "$INBOX" "$DRAFTS"

ts() { date +%Y%m%d-%H%M%S; }
iso() { date -u +%Y-%m-%dT%H:%M:%SZ; }

site_repo() {
  # priority: env var > config.json > git auto-detect (skill dir inside site repo)
  if [ -n "${ARTICLE_SITE_REPO:-}" ]; then
    echo "$ARTICLE_SITE_REPO"; return
  fi
  if [ -f "$CONFIG" ]; then
    local v
    v="$(grep -o '"site_repo"[[:space:]]*:[[:space:]]*"[^"]*"' "$CONFIG" \
      | sed 's/.*"site_repo"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')"
    [ -n "$v" ] && { echo "$v"; return; }
  fi
  if git -C "$ROOT" rev-parse --show-toplevel >/dev/null 2>&1; then
    git -C "$ROOT" rev-parse --show-toplevel; return
  fi
  echo ""
}

cmd_help() {
  cat <<'EOF'
article-ctl.sh — article skill 总控

子命令:
  help                                    显示本帮助
  capture <slug> <tag> <opinion>          创建 inbox 素材文件框架
  draft   <slug>                          创建 drafts 草稿文件框架
  gate    <slug>                          追加发布前 checklist 到草稿
  publish <slug>                          拷贝 draft 到站点 notes/，需 gate_passed: true

环境:
  ARTICLE_SITE_REPO  站点仓库绝对路径 (优先于 config.json)

文件:
  inbox/   素材 (gitignored)
  drafts/  草稿 (进 git)
  config.json  本地配置 (gitignored)
EOF
}

cmd_capture() {
  local slug="${1:?slug required}" tag="${2:?tag required}" opinion="${3:?opinion required}"
  local file="$INBOX/$(ts)-${slug}.md"
  cat > "$file" <<EOF
---
ts: $(iso)
tag: [${tag}]
source: claude-code-conv
opinion: "${opinion}"
---

<!-- 由 AI 在会话中填充原始片段 / 关键观点 -->
EOF
  echo "$file"
}

cmd_draft() {
  local slug="${1:?slug required}"
  shift || true
  # Default template: skill-experience. Use --template tutorial-intro for intro/tutorial articles.
  local template="skill-experience"
  while [ $# -gt 0 ]; do
    case "$1" in
      --template) template="${2:?template value required}"; shift 2 ;;
      --template=*) template="${1#--template=}"; shift ;;
      *) echo "unknown draft arg: $1" >&2; exit 1 ;;
    esac
  done

  local blueprint_hint sections_hint default_question_type default_objectivity
  case "$template" in
    skill-experience)
      blueprint_hint="blueprint.md"
      sections_hint="起源 / 尝试方案 / 踩的坑 / 最终设计 / 反共识 takeaway"
      default_question_type="方法论 / 系统设计 / 工程实践 / AI 工程化"
      default_objectivity="0.6 / 0.25 / 0.15"
      ;;
    tutorial-intro)
      blueprint_hint="blueprint-tutorial-intro.md"
      sections_hint="这是什么 / 什么时候用 / 核心能力 / 典型用法 / 进阶建议"
      default_question_type="工具使用 / 工程实践"
      default_objectivity="0.7 / 0.15 / 0.15"
      ;;
    *)
      echo "unknown template: $template (expected: skill-experience | tutorial-intro)" >&2
      exit 1
      ;;
  esac

  local file="$DRAFTS/${slug}.md"
  if [ -e "$file" ]; then
    echo "draft already exists: $file" >&2
    exit 1
  fi
  local now
  now="$(iso)"
  cat > "$file" <<EOF
---
# ⚠ AI DRAFT CONTRACT: 本框架生成后，AI 必须用 Edit 改下列字段为实值：
#   title / question.type / quality.{overall,coverage,depth,specificity}
#   tags (≥1 条对象) / topics (≥1 条对象)
# analysis.objectivity 三个 ratio 之和必须 = 1.0（允差 0.01）
# 不得留: "" / 0 / pending / [] (tags/topics)
# ── Site schema required fields (src/utils/schema.ts) ─────────────
title: ""
slug: "${slug}"
createdAt: "${now}"
updatedAt: "${now}"
date: "${now}"
question:
  type: ""            # e.g. ${default_question_type}
  subType: ""
quality:
  overall: 0          # 0-10; fill before publish (will block on 0)
  coverage: 0
  depth: 0
  specificity: 0
  reviewer: ai        # ai | human | hybrid
analysis:
  objectivity:
    factRatio: 0.5    # must sum to 1.0 across the three ratios
    inferenceRatio: 0.3
    opinionRatio: 0.2
  assumptions: []
  limitations: []
review:
  status: draft       # draft | reviewed | deprecated
tags: []              # [{ name: "rust", parent?: "lang", alias?: ["rs"] }]
topics: []            # [{ name: "AI 工程化", alias?: [] }]

# ── Skill workflow metadata (passthrough; not consumed by site) ───
template: ${template}
gate_passed: false
quality_gate:
  contrarian_view: pending
  concrete_example: pending
  takeaway: pending
  real_skill: pending
---

<!-- 由 AI 依 ${blueprint_hint} 五节骨架填充: ${sections_hint} -->
<!-- analysis.objectivity 典型默认 (${template}): ${default_objectivity} -->
EOF
  echo "$file"
}

cmd_gate() {
  local slug="${1:?slug required}"
  local file="$DRAFTS/${slug}.md"
  [ -f "$file" ] || { echo "draft not found: $file" >&2; exit 1; }
  [ -f "$CHECKLIST" ] || { echo "checklist not found: $CHECKLIST" >&2; exit 1; }
  if grep -q "<!-- ARTICLE_GATE_CHECKLIST -->" "$file"; then
    echo "checklist already appended" >&2
    exit 0
  fi
  {
    echo ""
    echo "<!-- ARTICLE_GATE_CHECKLIST -->"
    cat "$CHECKLIST"
  } >> "$file"
  echo "appended checklist to $file"
}

cmd_publish() {
  local slug="${1:?slug required}"
  local file="$DRAFTS/${slug}.md"
  [ -f "$file" ] || { echo "draft not found: $file" >&2; exit 1; }

  if ! grep -q '^gate_passed:[[:space:]]*true' "$file"; then
    echo "gate not passed (frontmatter must contain 'gate_passed: true')" >&2
    exit 1
  fi

  # Schema pre-check: site zod requires numeric quality + filled question.type.
  # Scan only the frontmatter (between the first two `---` lines) to avoid
  # false positives from body text or appended gate checklist.
  local fm
  fm="$(awk '/^---$/{c++; next} c==1{print} c>=2{exit}' "$file")"
  local issues=()
  echo "$fm" | grep -q '^title:[[:space:]]*""' && issues+=("title empty")
  echo "$fm" | grep -q '^  type:[[:space:]]*""' && issues+=("question.type empty")
  echo "$fm" | grep -qE '^  (overall|coverage|depth|specificity):[[:space:]]*0$' && issues+=("quality score still 0 — fill 1-10")
  echo "$fm" | grep -q 'pending' && issues+=("'pending' values remain in frontmatter — replace with real data")
  echo "$fm" | grep -qE '^tags:[[:space:]]*\[\][[:space:]]*$' && issues+=("tags empty — add ≥1 object like [{name: \"github\"}]")
  echo "$fm" | grep -qE '^topics:[[:space:]]*\[\][[:space:]]*$' && issues+=("topics empty — add ≥1 object like [{name: \"AI 工程化\"}]")
  if [ ${#issues[@]} -gt 0 ]; then
    echo "schema pre-check failed:" >&2
    printf '  - %s\n' "${issues[@]}" >&2
    exit 1
  fi

  local repo
  repo="$(site_repo)"
  if [ -z "$repo" ]; then
    echo "site repo not configured (set ARTICLE_SITE_REPO or config.json site_repo)" >&2
    exit 1
  fi
  local notes_dir="$repo/src/content/notes"
  [ -d "$notes_dir" ] || { echo "notes dir missing: $notes_dir" >&2; exit 1; }

  # next number
  local max_n=0 n
  for f in "$notes_dir"/*.md; do
    [ -e "$f" ] || continue
    n="$(basename "$f" | sed -n 's/^\([0-9][0-9]*\)-.*/\1/p')"
    [ -n "$n" ] && [ "$n" -gt "$max_n" ] && max_n="$n"
  done
  local next=$((max_n + 1))
  local target="$notes_dir/${next}-${slug}.md"

  # Strip gate checklist (everything from ARTICLE_GATE_CHECKLIST marker) and
  # workflow-only frontmatter fields (template / gate_passed / quality_gate)
  # before publishing to the site.
  awk '
    /^<!-- ARTICLE_GATE_CHECKLIST -->/ { exit }
    /^template:[[:space:]]/ { next }
    /^gate_passed:[[:space:]]/ { next }
    /^quality_gate:[[:space:]]*$/ { skip = 1; next }
    skip && /^[[:space:]]+/ { next }
    { skip = 0; print }
  ' "$file" > "$target"
  echo "published: $target"
  echo ""
  echo "━━ 下一步（手动，按顺序执行） ━━"
  echo ""
  echo "1. 本地 dry-run 构建，失败则回滚新文件："
  echo "   cd \"$repo\""
  echo "   npm run build"
  echo "   # 若 build 失败: git rm src/content/notes/${next}-${slug}.md  → 回 draft 修补"
  echo ""
  echo "2. lint 死代码扫描："
  echo "   npm run lint:unused"
  echo ""
  echo "3. 确认 diff 只含预期新文件："
  echo "   git status"
  echo "   git diff --stat"
  echo ""
  echo "4. commit + push（push 后 GitHub Actions 会自动部署 Pages）："
  echo "   git add src/content/notes/${next}-${slug}.md"
  echo "   git commit -m \"feat(notes): ${slug}\""
  echo "   git push origin main"
}

main() {
  local sub="${1:-help}"
  shift || true
  case "$sub" in
    help|-h|--help) cmd_help ;;
    capture) cmd_capture "$@" ;;
    draft)   cmd_draft "$@" ;;
    gate)    cmd_gate "$@" ;;
    publish) cmd_publish "$@" ;;
    *) echo "unknown subcommand: $sub" >&2; cmd_help; exit 1 ;;
  esac
}

main "$@"
