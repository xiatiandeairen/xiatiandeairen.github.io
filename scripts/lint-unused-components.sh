#!/usr/bin/env bash
# Scan src/components/*.astro for components that are never imported.
#
# Exit codes:
#   0 — no dead components
#   1 — dead components found (listed)

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPONENTS_DIR="$REPO_ROOT/src/components"

if [ ! -d "$COMPONENTS_DIR" ]; then
  echo "lint-unused-components: $COMPONENTS_DIR not found"
  exit 2
fi

dead=()
for f in "$COMPONENTS_DIR"/*.astro; do
  [ -e "$f" ] || continue
  name=$(basename "$f" .astro)
  # Find imports of this component elsewhere in src/
  hits=$(grep -rlE "import[^;]*${name}([^a-zA-Z0-9-]|$)" "$REPO_ROOT/src/" 2>/dev/null \
         | grep -v "$(basename "$f")" || true)
  if [ -z "$hits" ]; then
    dead+=("$name")
  fi
done

if [ ${#dead[@]} -eq 0 ]; then
  echo "lint-unused-components: no dead components ✓"
  exit 0
fi

echo "lint-unused-components: ${#dead[@]} dead component(s):"
for c in "${dead[@]}"; do
  echo "  src/components/${c}.astro"
done
echo ""
echo "Fix: delete the file, or import it somewhere"
exit 1
