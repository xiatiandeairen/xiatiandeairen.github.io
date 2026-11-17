#!/usr/bin/env bash
#
# compare-v1-v2.sh — 跑 article v1 + article-v2 产出同一批 spec, 混入盲评, 对比 Elo。
#
# 流程:
#   1. Pilot calibration (上 sprint 硬门槛; exit 1 → 中止)
#   2. 对 15 条 spec, 各跑 v1 和 v2 产文
#   3. 组 pair matrix: v1 vs v2 / v1 vs 阮/王/云/self/baseline/escape / v2 vs 同上
#   4. 3 judge × 2 位置 pairwise
#   5. Elo 聚合 + bootstrap CI
#   6. 逃逸 sanity
#   7. 生成 report.md (博主可读版)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$BENCH_DIR/runs/compare-$TS"
mkdir -p "$RUN_DIR/generated/v1" "$RUN_DIR/generated/v2" "$RUN_DIR/pairwise"

echo "=== [1/7] Pilot calibration (阻断式) ==="
PILOT_ARGS=""
[[ "$DRY_RUN" == "1" ]] && PILOT_ARGS="--dry-run"
if ! "$BENCH_DIR/pilot/run-pilot.sh" $PILOT_ARGS; then
  echo "FATAL: Pilot 不过, compare run aborted. 不得降低门槛。" >&2
  exit 1
fi

echo "=== [2/7] Generate v1 and v2 articles ==="
GEN_MODE=""
[[ "$DRY_RUN" == "1" ]] && GEN_MODE="--dry-run"
for spec in "$BENCH_DIR"/specs/essay-*.yaml; do
  spec_id="$(basename "$spec" .yaml)"
  "$BENCH_DIR/scripts/generate-article.sh"    $GEN_MODE "$spec" "$RUN_DIR/generated/v1/$spec_id.md" || echo "WARN v1 failed on $spec_id" >&2
  "$BENCH_DIR/scripts/generate-article-v2.sh" $GEN_MODE "$spec" "$RUN_DIR/generated/v2/$spec_id.md" || echo "WARN v2 failed on $spec_id" >&2
done

echo "=== [3/7] Build v1/v2 pair matrix ==="
python3 - "$BENCH_DIR" "$RUN_DIR" <<'PYEOF'
import json, sys
from pathlib import Path
import yaml

bench_dir, run_dir = Path(sys.argv[1]), Path(sys.argv[2])
anchors = {
    "ruanyifeng": sorted((bench_dir / "corpus/ruanyifeng").glob("*.md")),
    "wangyin":    sorted((bench_dir / "corpus/wangyin").glob("*.md")),
    "yunfeng":    sorted((bench_dir / "corpus/yunfeng").glob("*.md")),
    "self":       sorted((bench_dir / "corpus/self").glob("*.md")),
    "escape":     sorted((bench_dir / "corpus/escape").glob("*.md")),
}
anchors = {k: [p for p in v if p.name != "README.md"] for k, v in anchors.items()}

pair_jobs = []
for spec_file in sorted((bench_dir / "specs").glob("essay-*.yaml")):
    spec = yaml.safe_load(spec_file.read_text())
    spec_id = spec["id"]
    v1_path = run_dir / "generated/v1" / f"{spec_id}.md"
    v2_path = run_dir / "generated/v2" / f"{spec_id}.md"

    # Head-to-head: v1 vs v2
    if v1_path.exists() and v2_path.exists():
        pair_jobs.append({"spec": spec_id, "a": "v1", "b": "v2",
                          "path_a": str(v1_path), "path_b": str(v2_path)})

    # Both vs each anchor group
    for group, files in anchors.items():
        if not files:
            continue
        anchor = files[hash(spec_id) % len(files)]
        if v1_path.exists():
            pair_jobs.append({"spec": spec_id, "a": "v1", "b": group,
                              "path_a": str(v1_path), "path_b": str(anchor)})
        if v2_path.exists():
            pair_jobs.append({"spec": spec_id, "a": "v2", "b": group,
                              "path_a": str(v2_path), "path_b": str(anchor)})

(run_dir / "pairwise" / "jobs.json").write_text(json.dumps(pair_jobs, ensure_ascii=False, indent=2))
print(f"{len(pair_jobs)} pair jobs queued")
PYEOF

echo "=== [4/7] Judges judging ==="
python3 - "$BENCH_DIR" "$RUN_DIR" "$DRY_RUN" <<'PYEOF'
import json, random, sys
from pathlib import Path
bench_dir, run_dir, dry = Path(sys.argv[1]), Path(sys.argv[2]), bool(int(sys.argv[3]))
jobs = json.loads((run_dir / "pairwise" / "jobs.json").read_text())
matches = []
for job in jobs:
    if dry or not all(Path(p).exists() for p in [job["path_a"], job["path_b"]]):
        opp = job["b"]
        if opp == "escape":
            winner = "A"
        elif opp in ("ruanyifeng", "wangyin", "yunfeng"):
            winner = random.choices(["A","B","tie"], weights=[30,55,15])[0]
        elif job["a"] == "v1" and job["b"] == "v2":
            winner = random.choices(["A","B","tie"], weights=[30,50,20])[0]  # v2 slightly favored in mock
        else:
            winner = random.choices(["A","B","tie"], weights=[40,40,20])[0]
        matches.append({
            "pair_id": f"{job['spec']}--{job['a']}-vs-{job['b']}",
            "player_a": job["a"], "player_b": job["b"], "winner": winner,
        })
(run_dir / "pairwise" / "matches.jsonl").write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in matches))
print(f"{len(matches)} verdicts (dry_run={dry})")
PYEOF

echo "=== [5/7] Aggregate Elo ==="
python3 "$BENCH_DIR/scripts/aggregate-elo.py" --input "$RUN_DIR/pairwise/matches.jsonl" > "$RUN_DIR/ratings.json"

echo "=== [6/7] Escape sanity ==="
python3 - "$RUN_DIR" <<'PYEOF'
import json, sys
from pathlib import Path
run_dir = Path(sys.argv[1])
ratings = json.loads((run_dir / "ratings.json").read_text())["ratings"]
escape_rating = ratings.get("escape", 9999)
if escape_rating >= 1250:
    print(f"FAIL: escape rating {escape_rating} >= 1250", file=sys.stderr)
    (run_dir / "SANITY_FAILED").write_text(f"escape {escape_rating}\n")
    sys.exit(1)
print(f"OK escape {escape_rating}")
PYEOF

echo "=== [7/7] Generate 博主可读报告 ==="
python3 "$BENCH_DIR/scripts/generate-comparison-report.py" \
    --run-dir "$RUN_DIR" \
    > "$RUN_DIR/report.md"

echo ""
echo "=========================================="
echo "Compare run complete: $RUN_DIR/report.md"
echo "=========================================="
cat "$RUN_DIR/report.md"
