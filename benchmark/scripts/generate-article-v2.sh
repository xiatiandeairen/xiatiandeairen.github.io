#!/usr/bin/env bash
#
# generate-article-v2.sh — 按 article-v2 5 步流程产文
#
# Usage: generate-article-v2.sh [--dry-run] <spec.yaml> <output.md>

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; shift; fi

SPEC="${1:?usage: generate-article-v2.sh [--dry-run] <spec.yaml> <output.md>}"
OUT="${2:?output path required}"
mkdir -p "$(dirname "$OUT")"

if [[ "$DRY_RUN" == "1" ]]; then
  SPEC_ID="$(basename "$SPEC" .yaml)"
  cat > "$OUT" <<EOF
<!--
generated_by: article-v2 (dry-run placeholder)
spec: $SPEC
flow: spark → angle → evidence → voice → revise
-->

# [v2 dry-run placeholder] $SPEC_ID

按 article-v2 5 步流程走, 每步 Gate 通过后产出 final.md。dry-run 不实际跑 skill, 只产占位。

真实产出需:
1. 博主提供 voice_references (历史高质文)
2. OPENAI_API_KEY 可用
3. 手工按 stages/*.md 串 5 步 (或等封装 CLI 的 sprint)
EOF
  echo "wrote v2 dry-run stub to $OUT"
  exit 0
fi

# Real mode: not wired yet
echo "[ERROR] article-v2 real invocation not wired." >&2
echo "        Need: (a) 5-stage orchestrator CLI OR (b) 博主手工按 stages/*.md 串流程" >&2
exit 3
