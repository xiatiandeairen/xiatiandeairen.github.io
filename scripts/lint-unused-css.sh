#!/usr/bin/env bash
# Scan all src/styles/*.css files for top-level class selectors that have
# 0 references anywhere in src/ outside the stylesheets themselves and
# design-governance.md.
#
# Exit codes:
#   0 — no dead classes
#   1 — dead classes found (listed)

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STYLES_DIR="$REPO_ROOT/src/styles"

if [ ! -d "$STYLES_DIR" ]; then
  echo "lint-unused-css: $STYLES_DIR not found"
  exit 2
fi

# Extract top-level class selectors from a CSS file.
# Matches lines like:
#   .foo {
#     .foo {        (inside @layer)
#     .foo,
#     .foo:hover {
# Captures the bare class name (no dot, no pseudo, no comma).
extract_classes() {
  grep -oE '^[[:space:]]*\.[a-z_][a-zA-Z0-9_-]*' "$1" \
    | sed -E 's/^[[:space:]]*\.//' \
    | sort -u
}

dead_total=0
declare -a report

for css in "$STYLES_DIR"/*.css; do
  [ -e "$css" ] || continue
  fname=$(basename "$css")
  classes=$(extract_classes "$css")
  [ -z "$classes" ] && continue

  for cls in $classes; do
    # Word-boundary match (hyphen-aware), exclude all stylesheets and design-governance.md
    usage=$(grep -rlE "(^|[^a-zA-Z0-9_-])${cls}([^a-zA-Z0-9_-]|$)" "$REPO_ROOT/src/" 2>/dev/null \
            | grep -vE '/styles/[^/]+\.css$|design-(tokens|patterns|governance)\.md$|/styles/README\.md$' || true)
    if [ -z "$usage" ]; then
      report+=("  ${fname}  .${cls}")
      dead_total=$((dead_total + 1))
    fi
  done
done

if [ "$dead_total" -eq 0 ]; then
  echo "lint-unused-css: no dead classes ✓"
  exit 0
fi

echo "lint-unused-css: ${dead_total} dead class(es):"
printf '%s\n' "${report[@]}"
echo ""
echo "Fix: delete the rule, or register usage in design-governance.md §2.3"
exit 1
