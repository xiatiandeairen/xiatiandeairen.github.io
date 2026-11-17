#!/usr/bin/env python3
"""
generate-comparison-report.py — 生成博主可读的 v1 vs v2 对比报告。

读 runs/compare-{ts}/ratings.json + pairwise/matches.jsonl, 输出 markdown。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    ratings = json.loads((run_dir / "ratings.json").read_text())
    matches = [
        json.loads(l) for l in (run_dir / "pairwise" / "matches.jsonl").read_text().splitlines() if l.strip()
    ]

    # Head-to-head v1 vs v2
    h2h = {"v1_wins": 0, "v2_wins": 0, "ties": 0}
    for m in matches:
        if {m["player_a"], m["player_b"]} == {"v1", "v2"}:
            a_is_v1 = m["player_a"] == "v1"
            if m["winner"] == "tie":
                h2h["ties"] += 1
            elif (m["winner"] == "A" and a_is_v1) or (m["winner"] == "B" and not a_is_v1):
                h2h["v1_wins"] += 1
            else:
                h2h["v2_wins"] += 1

    v1 = ratings["ratings"].get("v1")
    v2 = ratings["ratings"].get("v2")
    v1_ci = ratings["ci_95"].get("v1", [0, 0])
    v2_ci = ratings["ci_95"].get("v2", [0, 0])

    gap = (v2 - v1) if (v1 is not None and v2 is not None) else None
    ci_overlap = v1 is not None and v2 is not None and (v1_ci[1] >= v2_ci[0] and v2_ci[1] >= v1_ci[0])

    # Verdict
    if gap is None:
        verdict = "数据不足, 无法判断"
    elif ci_overlap:
        verdict = "v2 未显著优于 v1 (CI 重叠, 看不出真差别)"
    elif gap > 0:
        verdict = f"**v2 显著优于 v1** (差 {gap:.0f} Elo 点, CI 不重叠)"
    else:
        verdict = f"**v2 显著劣于 v1** (差 {abs(gap):.0f} Elo 点, CI 不重叠) — 要紧事, 检查流程是否带来了反效果"

    lines = [
        f"# Benchmark 对比报告 — {run_dir.name}",
        "",
        "## 一句话结论",
        "",
        verdict,
        "",
        "## 头对头 (v1 vs v2 同题材直接对比)",
        "",
        f"- v1 赢: {h2h['v1_wins']}",
        f"- v2 赢: {h2h['v2_wins']}",
        f"- 平: {h2h['ties']}",
        "",
        "## 相对锚点的位置",
        "",
        "| 玩家 | Elo 评分 | 95% CI |",
        "|------|---------|--------|",
    ]
    for p in ratings["rank"]:
        lo, hi = ratings["ci_95"][p]
        r = ratings["ratings"][p]
        marker = " 👈" if p in ("v1", "v2") else ""
        lines.append(f"| {p}{marker} | {r} | [{lo}, {hi}] |")

    lines += [
        "",
        "## 读这张表的方法",
        "",
        "- **排第几就是水平在哪里**。阮/王/云 应该在最上, escape 应该在最下, v1/v2 在中间",
        "- **CI 重叠就是没区别**。v1 = 1520, v2 = 1535 看起来 v2 高, 但如果它们的 CI 都 ±50, 就是测不出差别, 别兴奋",
        "- **如果 escape 不在最后** → 整个 run 作废, Pilot 应该已经阻断; 如果没阻断, judge 失效",
        "",
        "## Per-spec 胜率 (v2 对 v1)",
        "",
    ]
    per_spec = defaultdict(lambda: {"v1_wins": 0, "v2_wins": 0, "ties": 0})
    for m in matches:
        if {m["player_a"], m["player_b"]} != {"v1", "v2"}:
            continue
        spec = m["pair_id"].split("--")[0]
        a_is_v1 = m["player_a"] == "v1"
        if m["winner"] == "tie":
            per_spec[spec]["ties"] += 1
        elif (m["winner"] == "A" and a_is_v1) or (m["winner"] == "B" and not a_is_v1):
            per_spec[spec]["v1_wins"] += 1
        else:
            per_spec[spec]["v2_wins"] += 1
    lines += ["| spec | v1 win | v2 win | tie |", "|------|--------|--------|-----|"]
    for spec, cnt in sorted(per_spec.items()):
        lines.append(f"| {spec} | {cnt['v1_wins']} | {cnt['v2_wins']} | {cnt['ties']} |")

    lines += [
        "",
        "## Red flags (要小心的)",
        "",
    ]
    escape_rating = ratings["ratings"].get("escape", 9999)
    if escape_rating < 1250:
        lines.append(f"- escape sanity: PASS ({escape_rating})")
    else:
        lines.append(f"- **escape sanity FAIL**: rating {escape_rating} >= 1250, 本次 run 不可信, 整个报告作废")

    max_span = max((ratings["ci_95"][p][1] - ratings["ci_95"][p][0]) for p in ratings["rank"])
    if max_span > 200:
        lines.append(f"- 最大 CI 跨度 {max_span:.0f} 点 > 200, 样本不足或 judge 方差大, 结论脆弱")

    lines += [
        "",
        "## 如果你只看一眼要知道什么",
        "",
        "看 **\"一句话结论\"** + **\"相对锚点的位置\"** 表。其他都是证据。",
        "",
        "如果 v2 没显著好, 原因几种可能:",
        "1. Gate 阈值错 (太宽没过滤 AI 味, 太严博主风格被误杀)",
        "2. Gate 实现 stub 或未校准, 产文本身没改进",
        "3. 5 步流程本身错了, 没抓住关键瓶颈",
        "4. Benchmark 数据不足, 看不出差别",
        "",
        "按 1-4 顺序排查。",
    ]

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
