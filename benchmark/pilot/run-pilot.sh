#!/usr/bin/env bash
#
# run-pilot.sh — 跑 Pilot 校准评审器。
#
# 阻断式：exit 0 = 通过，exit 1 = 不通过（sprint 终止，不得以"差一点"为由继续）。
#
# Usage:
#   ./pilot/run-pilot.sh                # 真实调 3 个 judge API
#   ./pilot/run-pilot.sh --dry-run      # 不调 API，生成模拟结果便于 debug pipeline

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

PAIRS_YAML="$BENCH_DIR/pilot/golden-pairs.yaml"
JUDGES_YAML="$BENCH_DIR/judge/judges.yaml"
PROMPT="$BENCH_DIR/judge/prompts/pairwise.txt"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$BENCH_DIR/runs/pilot-$TS"
mkdir -p "$OUT_DIR"

python3 - "$PAIRS_YAML" "$JUDGES_YAML" "$PROMPT" "$OUT_DIR" "$BENCH_DIR" "$DRY_RUN" <<'PYEOF'
import json
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("pyyaml required: pip install pyyaml")

pairs_yaml, judges_yaml, prompt, out_dir, bench_dir, dry_run = sys.argv[1:]
dry_run = bool(int(dry_run))
out_dir = Path(out_dir)
bench_dir = Path(bench_dir)

pairs_cfg = yaml.safe_load(Path(pairs_yaml).read_text())
thresholds = pairs_cfg["thresholds"]
pairs = pairs_cfg["pairs"]

call_judge = str(bench_dir / "scripts" / "call-judge.py")

all_results = []
correct = 0
swap_consistent = 0
total_swaps = 0

for pair in pairs:
    path_a = bench_dir / pair["path_a"]
    path_b = bench_dir / pair["path_b"]

    if not path_a.exists() or not path_b.exists():
        if dry_run:
            # In dry-run, synthesize both files so call-judge can render prompts
            tmp_dir = out_dir / "stub-inputs" / pair["id"]
            tmp_dir.mkdir(parents=True, exist_ok=True)
            (tmp_dir / "a.md").write_text(f"（{pair['author']} 高水位 stub）\n这是占位文本，真实语料由博主补充。" * 40)
            (tmp_dir / "b.md").write_text(f"（{pair['author']} 一般水位 stub）\n占位占位占位。" * 40)
            path_a, path_b = tmp_dir / "a.md", tmp_dir / "b.md"
        else:
            sys.exit(f"FATAL: pair {pair['id']} missing corpus: {path_a} / {path_b}")

    # Position forward: A=path_a, B=path_b (expected A win)
    # Position swap:    A=path_b, B=path_a (expected B win = still the "good" file)
    for replicate, (p_a, p_b, expected_good_pos) in enumerate(
        [(path_a, path_b, "A"), (path_b, path_a, "B")]
    ):
        cmd = [
            "python3", call_judge,
            "--judges", judges_yaml,
            "--prompt", prompt,
            "--text-a", str(p_a),
            "--text-b", str(p_b),
            "--output", "json",
        ]
        if dry_run:
            cmd.append("--dry-run")
            cmd.extend(["--seed", str(hash(pair["id"] + str(replicate)) % 10000)])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"WARN: call-judge failed on {pair['id']} rep {replicate}: {result.stderr}", file=sys.stderr)
            continue
        parsed = json.loads(result.stdout)
        verdicts = [r.get("verdict") for r in parsed["results"]]

        # Majority verdict
        counts = {"A": 0, "B": 0, "tie": 0}
        for v in verdicts:
            if v in counts:
                counts[v] += 1
        if counts["A"] >= 2:
            majority = "A"
        elif counts["B"] >= 2:
            majority = "B"
        else:
            majority = "tie"

        pair_outcome = {
            "pair_id": pair["id"],
            "replicate": replicate,
            "position_good": expected_good_pos,
            "majority": majority,
            "per_judge": verdicts,
            "pair_correct": majority == expected_good_pos,
        }
        all_results.append(pair_outcome)

    # Swap consistency: forward says good-at-A, swap says good-at-B → both correct
    forward = next(r for r in all_results if r["pair_id"] == pair["id"] and r["replicate"] == 0)
    swap = next(r for r in all_results if r["pair_id"] == pair["id"] and r["replicate"] == 1)
    total_swaps += 1
    if forward["pair_correct"] == swap["pair_correct"]:
        swap_consistent += 1

    # Count pair as "correct" only if BOTH forward and swap agree on the good article
    if forward["pair_correct"] and swap["pair_correct"]:
        correct += 1

consistency = swap_consistent / total_swaps if total_swaps else 0.0

report = {
    "timestamp_utc": out_dir.name.replace("pilot-", ""),
    "dry_run": dry_run,
    "correct_pairs": correct,
    "total_pairs": len(pairs),
    "position_consistency": consistency,
    "thresholds": thresholds,
    "pass": correct >= thresholds["min_correct_pairs"] and consistency >= thresholds["min_position_consistency"],
    "per_trial": all_results,
}

(out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

# Human-readable summary
lines = [
    f"# Pilot Report ({out_dir.name})",
    "",
    f"- dry_run: {dry_run}",
    f"- correct pairs: {correct}/{len(pairs)} (threshold ≥{thresholds['min_correct_pairs']})",
    f"- position consistency: {consistency:.0%} (threshold ≥{thresholds['min_position_consistency']:.0%})",
    f"- verdict: {'PASS ✓' if report['pass'] else 'FAIL ✗'}",
    "",
    "## Per-trial",
    "| pair | rep | good@ | majority | correct | per-judge |",
    "|------|-----|-------|----------|---------|-----------|",
]
for r in all_results:
    lines.append(
        f"| {r['pair_id']} | {r['replicate']} | {r['position_good']} | {r['majority']} | {'✓' if r['pair_correct'] else '✗'} | {r['per_judge']} |"
    )

if not report["pass"]:
    lines.extend([
        "",
        "## Failure guidance",
        "- 判错的 pair: 检查博主标注是否合理",
        "- 位置交换不一致: judge 对该对文章判断脆弱，考虑换 judge 或换 pair",
        "- 整体判对率 <75%: sprint 方向重塑（换评审方式 / 加人工混合 / 收回 benchmark 设想）",
        "- **不得调低 threshold 重跑**",
    ])

(out_dir / "report.md").write_text("\n".join(lines) + "\n")

print((out_dir / "report.md").read_text())
sys.exit(0 if report["pass"] else 1)
PYEOF
