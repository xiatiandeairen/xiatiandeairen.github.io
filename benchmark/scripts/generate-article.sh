#!/usr/bin/env bash
#
# generate-article.sh — 调 article skill 按给定 spec 产文。
#
# Usage:
#   ./scripts/generate-article.sh <spec.yaml> <output.md>
#   ./scripts/generate-article.sh --baseline <spec.yaml> <output.md>   # 裸 prompt, 无 skill
#   ./scripts/generate-article.sh --dry-run <spec.yaml> <output.md>    # 只写 placeholder
#
# article skill 的实际调用协议由博主/环境决定（Claude Code 调 slash command / MCP / 或本地 Python
# wrapper）。本脚本先预留契约：读 spec.yaml 的 prompt_for_skill 字段，写到 output.md。
# TODO(博主): 填入真实调用。

set -euo pipefail

MODE=skill
if [[ "${1:-}" == "--baseline" ]]; then MODE=baseline; shift; fi
if [[ "${1:-}" == "--dry-run" ]]; then MODE=dry; shift; fi

SPEC="${1:?usage: generate-article.sh [--baseline|--dry-run] <spec.yaml> <output.md>}"
OUT="${2:?output path required}"

if [[ ! -f "$SPEC" ]]; then echo "spec not found: $SPEC" >&2; exit 2; fi

mkdir -p "$(dirname "$OUT")"

PROMPT_BODY="$(python3 -c "
import yaml, sys
s = yaml.safe_load(open('$SPEC'))
print(s.get('prompt_for_skill', '') or 'TODO: missing prompt_for_skill')
")"

case "$MODE" in
  skill)
    # TODO(博主): 填入真实 article skill 调用。可能形态:
    #   claude --skill article --input "$PROMPT_BODY" > "$OUT"
    # 或通过 MCP / Python wrapper。
    echo "[ERROR] article skill invocation not wired yet." >&2
    echo "        edit benchmark/scripts/generate-article.sh to add your actual call." >&2
    exit 3
    ;;
  baseline)
    # TODO(博主): 裸 Claude prompt baseline
    echo "[ERROR] baseline invocation not wired yet." >&2
    exit 3
    ;;
  dry)
    cat > "$OUT" <<EOF
<!--
generated_by: dry-run placeholder
spec: $SPEC
captured_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)
-->

# dry-run placeholder

Spec prompt was:

$PROMPT_BODY

（这里应该是 article skill 产出的真实文章。dry-run 模式仅用于 pipeline 连通性测试。）
EOF
    echo "wrote dry-run stub to $OUT"
    ;;
esac
