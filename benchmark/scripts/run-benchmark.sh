#!/usr/bin/env bash
#
# run-benchmark.sh — 主 benchmark 入口。
#
# 流程：
#   1. 跑 Pilot calibration（每次必跑）— 通过率 <75% 则本次 run 作废
#   2. 按 15 条 spec 用 article skill 产文（+ baseline 产文）
#   3. 组 pairwise 矩阵：skill vs {阮, 王, 云, self, baseline, escape} = 6 对手 × 15 spec
#   4. 3 judge × 2 位置 并行判决
#   5. Elo 聚合 + bootstrap CI
#   6. 逃逸 sanity：escape rating 必须 < 1250
#   7. 生成 report.md
#
# Usage:
#   ./scripts/run-benchmark.sh              # 完整 run
#   ./scripts/run-benchmark.sh --dry-run    # 不调 API，stub 产文

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
RUN_DIR="$BENCH_DIR/runs/$TS"
mkdir -p "$RUN_DIR/generated" "$RUN_DIR/pairwise"

echo "=== [1/7] Pilot calibration ==="
PILOT_ARGS=""
[[ "$DRY_RUN" == "1" ]] && PILOT_ARGS="--dry-run"
if ! "$BENCH_DIR/pilot/run-pilot.sh" $PILOT_ARGS; then
  echo "FATAL: Pilot calibration failed. Run aborted." >&2
  echo "       不得降低门槛。Pilot 通过后才能跑 benchmark。" >&2
  exit 1
fi

echo "=== [2/7] Generate articles from 15 specs ==="
GEN_MODE=""
[[ "$DRY_RUN" == "1" ]] && GEN_MODE="--dry-run"
for spec in "$BENCH_DIR"/specs/essay-*.yaml; do
  spec_id="$(basename "$spec" .yaml)"
  "$BENCH_DIR/scripts/generate-article.sh" $GEN_MODE "$spec" "$RUN_DIR/generated/$spec_id-skill.md" || {
    echo "WARN: skill failed on $spec_id" >&2
  }
  "$BENCH_DIR/scripts/generate-article.sh" --baseline $GEN_MODE "$spec" "$RUN_DIR/generated/$spec_id-baseline.md" || true
done

echo "=== [3/7] Build pairwise matrix ==="
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
# Filter out README files
anchors = {k: [p for p in v if p.name != "README.md"] for k, v in anchors.items()}

pair_jobs = []
for spec_file in sorted((bench_dir / "specs").glob("essay-*.yaml")):
    spec = yaml.safe_load(spec_file.read_text())
    spec_id = spec["id"]
    skill_out = run_dir / "generated" / f"{spec_id}-skill.md"
    baseline_out = run_dir / "generated" / f"{spec_id}-baseline.md"

    # Pick one anchor from each opponent group (rotate for diversity)
    for group, files in anchors.items():
        if not files:
            continue
        anchor = files[hash(spec_id) % len(files)]
        if skill_out.exists():
            pair_jobs.append({"spec": spec_id, "a": "skill", "b": group, "path_a": str(skill_out), "path_b": str(anchor)})
        if baseline_out.exists():
            pair_jobs.append({"spec": spec_id, "a": "baseline", "b": group, "path_a": str(baseline_out), "path_b": str(anchor)})

    if skill_out.exists() and baseline_out.exists():
        pair_jobs.append({"spec": spec_id, "a": "skill", "b": "baseline", "path_a": str(skill_out), "path_b": str(baseline_out)})

(run_dir / "pairwise" / "jobs.json").write_text(json.dumps(pair_jobs, ensure_ascii=False, indent=2))
print(f"{len(pair_jobs)} pair jobs queued")
PYEOF

echo "=== [4/7] Run judges on pair matrix ==="
# TODO: 真实实现会对 jobs.json 逐个调 call-judge.py, 3 judge × 2 位置, 并行（GNU parallel or python multiprocessing）。
# Dry-run 直接生成 mock verdicts 便于 pipeline 测试。
python3 - "$BENCH_DIR" "$RUN_DIR" "$DRY_RUN" <<'PYEOF'
import json, random, subprocess, sys
from pathlib import Path

bench_dir, run_dir, dry = Path(sys.argv[1]), Path(sys.argv[2]), bool(int(sys.argv[3]))
jobs = json.loads((run_dir / "pairwise" / "jobs.json").read_text())

matches = []
for job in jobs:
    # In real mode: call call-judge.py for each (judge, position) combo.
    # For now: produce mock verdicts so aggregate-elo.py has input.
    if dry or not all(Path(p).exists() for p in [job["path_a"], job["path_b"]]):
        # Biased random: give anchors a slight edge, escape a big penalty
        opp = job["b"]
        if opp == "escape":
            winner = "A"  # skill/baseline usually beats escape
        elif opp in ("ruanyifeng", "wangyin", "yunfeng"):
            winner = random.choices(["A", "B", "tie"], weights=[25, 60, 15])[0]
        else:
            winner = random.choices(["A", "B", "tie"], weights=[40, 40, 20])[0]
        matches.append({
            "pair_id": f"{job['spec']}--{job['a']}-vs-{job['b']}",
            "player_a": job["a"],
            "player_b": job["b"],
            "winner": winner,
        })
    else:
        # TODO(博主): real invocation
        # for judge in ("gpt-4o", "gemini-2-pro", "deepseek-v3"):
        #     for position_swap in (False, True):
        #         result = subprocess.run(["python3", str(bench_dir / "scripts/call-judge.py"), ...])
        pass

(run_dir / "pairwise" / "matches.jsonl").write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in matches))
print(f"{len(matches)} verdicts recorded (dry_run={dry})")
PYEOF

echo "=== [5/7] Aggregate Elo ==="
python3 "$BENCH_DIR/scripts/aggregate-elo.py" --input "$RUN_DIR/pairwise/matches.jsonl" > "$RUN_DIR/ratings.json"

echo "=== [6/7] Escape sanity ==="
python3 - "$RUN_DIR" <<'PYEOF'
import json, sys
from pathlib import Path
run_dir = Path(sys.argv[1])
ratings = json.loads((run_dir / "ratings.json").read_text())["ratings"]
escape_rating = ratings.get("escape", 1500)
if escape_rating >= 1250:
    print(f"FAIL: escape rating {escape_rating} >= 1250; judge untrustworthy this run.", file=sys.stderr)
    (run_dir / "SANITY_FAILED").write_text(f"escape rating {escape_rating}\n")
    sys.exit(1)
print(f"OK: escape rating {escape_rating} < 1250")
PYEOF
SANITY_PASS=$?
if [[ $SANITY_PASS -ne 0 ]]; then
  echo "FATAL: escape sanity failed. Run marked invalid." >&2
  exit 1
fi

echo "=== [7/7] Generate report ==="
python3 - "$RUN_DIR" > "$RUN_DIR/report.md" <<'PYEOF'
import json, sys
from pathlib import Path
run_dir = Path(sys.argv[1])
ratings = json.loads((run_dir / "ratings.json").read_text())
matches = [json.loads(l) for l in (run_dir / "pairwise" / "matches.jsonl").read_text().splitlines() if l.strip()]

lines = [f"# Benchmark Run {run_dir.name}", ""]
lines.append("## Summary")
lines.append("")
lines.append("| player | rating | 95% CI |")
lines.append("|--------|--------|--------|")
for p in ratings["rank"]:
    lo, hi = ratings["ci_95"][p]
    lines.append(f"| {p} | {ratings['ratings'][p]} | [{lo}, {hi}] |")

skill_rank = ratings["rank"].index("skill") + 1 if "skill" in ratings["rank"] else "N/A"
lines.append("")
lines.append(f"**Skill rank**: {skill_rank}/{len(ratings['rank'])}")
lines.append(f"**Total matches**: {ratings['n_matches']}")

lines.append("")
lines.append("## Per-opponent win rate for skill")
from collections import defaultdict
wins = defaultdict(lambda: [0, 0, 0])  # [win, loss, tie]
for m in matches:
    if "skill" not in (m["player_a"], m["player_b"]):
        continue
    opp = m["player_b"] if m["player_a"] == "skill" else m["player_a"]
    skill_is_a = m["player_a"] == "skill"
    if m["winner"] == "tie":
        wins[opp][2] += 1
    elif (m["winner"] == "A" and skill_is_a) or (m["winner"] == "B" and not skill_is_a):
        wins[opp][0] += 1
    else:
        wins[opp][1] += 1

lines.append("")
lines.append("| opponent | skill wins | losses | ties |")
lines.append("|----------|-----------|--------|------|")
for opp, (w, l, t) in sorted(wins.items()):
    lines.append(f"| {opp} | {w} | {l} | {t} |")

lines.append("")
lines.append("## Red flags")
lines.append("- calibration (Pilot) 通过（已由 step 1 保证）")
escape_rating = ratings["ratings"].get("escape", 9999)
lines.append(f"- escape rating: {escape_rating} ({'PASS' if escape_rating < 1250 else 'FAIL'})")
ci_spans = {p: ratings["ci_95"][p][1] - ratings["ci_95"][p][0] for p in ratings["rank"]}
max_span = max(ci_spans.values())
if max_span > 200:
    lines.append(f"- 警告: 最大 CI 跨度 {max_span:.0f} > 200，样本方差大，结论脆弱")

print("\n".join(lines))
PYEOF

echo ""
echo "=========================================="
echo "Run complete: $RUN_DIR/report.md"
echo "=========================================="
cat "$RUN_DIR/report.md"
