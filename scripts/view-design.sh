#!/usr/bin/env bash
# View an archived design mockup in the dev server.
# Usage: ./scripts/view-design.sh <folder-name>
# Example: ./scripts/view-design.sh 2026-04-17-article-header

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARCHIVE_DIR="$REPO_ROOT/design-archive"
PREVIEW_DIR="$REPO_ROOT/src/pages/preview"

if [ -z "$1" ]; then
  echo "Usage: $0 <folder-name>"
  echo ""
  echo "Available archives:"
  ls -1 "$ARCHIVE_DIR" 2>/dev/null | grep -v '^README' || echo "  (none)"
  exit 1
fi

SRC="$ARCHIVE_DIR/$1"
if [ ! -d "$SRC" ]; then
  echo "Error: archive not found: $SRC"
  exit 1
fi

cleanup() {
  echo ""
  echo "Cleaning up preview..."
  rm -rf "$PREVIEW_DIR"
}
trap cleanup EXIT INT TERM

echo "Copying $1 → src/pages/preview/"
mkdir -p "$PREVIEW_DIR"
cp "$SRC"/*.astro "$PREVIEW_DIR/" 2>/dev/null || {
  echo "Error: no .astro files in $SRC"
  exit 1
}

echo ""
echo "Preview routes:"
for f in "$PREVIEW_DIR"/*.astro; do
  name=$(basename "$f" .astro)
  echo "  http://localhost:4321/preview/$name"
done
echo ""

cd "$REPO_ROOT" && npm run dev
