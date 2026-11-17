#!/usr/bin/env python3
"""
aggregate-elo.py — 从 pairwise 判决 JSON 列表聚合成 Elo rating + bootstrap CI。

输入（JSON lines 或单个 JSON 数组）：
    {"pair_id": ..., "player_a": "skill", "player_b": "ruanyifeng", "winner": "A" | "B" | "tie"}

输出（stdout JSON）：
    {
      "ratings": {"skill": 1523, "ruanyifeng": 1680, ...},
      "ci_95": {"skill": [1478, 1568], ...},
      "rank": ["ruanyifeng", "wangyin", ...],
      "n_matches": 423
    }

K=32, init=1500, bootstrap=1000. 纯 stdlib。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

INIT_RATING = 1500
K = 32


def expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def run_elo(matches: list[dict[str, Any]], shuffle: bool = False) -> dict[str, float]:
    ratings: dict[str, float] = defaultdict(lambda: float(INIT_RATING))
    order = list(matches)
    if shuffle:
        random.shuffle(order)
    for m in order:
        a, b = m["player_a"], m["player_b"]
        w = m["winner"]
        if w == "tie":
            score_a, score_b = 0.5, 0.5
        elif w == "A":
            score_a, score_b = 1.0, 0.0
        elif w == "B":
            score_a, score_b = 0.0, 1.0
        else:
            continue
        ea = expected(ratings[a], ratings[b])
        eb = 1 - ea
        ratings[a] += K * (score_a - ea)
        ratings[b] += K * (score_b - eb)
    return dict(ratings)


def bootstrap_ci(matches: list[dict[str, Any]], n_boot: int, seed: int) -> dict[str, tuple[float, float]]:
    random.seed(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_boot):
        resampled = [random.choice(matches) for _ in range(len(matches))]
        ratings = run_elo(resampled)
        for player, r in ratings.items():
            samples[player].append(r)
    ci: dict[str, tuple[float, float]] = {}
    for player, rs in samples.items():
        rs_sorted = sorted(rs)
        lo = rs_sorted[int(0.025 * len(rs_sorted))]
        hi = rs_sorted[int(0.975 * len(rs_sorted))]
        ci[player] = (lo, hi)
    return ci


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="matches JSON file (array or JSONL)")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = Path(args.input).read_text(encoding="utf-8").strip()
    if raw.startswith("["):
        matches = json.loads(raw)
    else:
        matches = [json.loads(line) for line in raw.splitlines() if line.strip()]

    point_ratings = run_elo(matches)
    ci = bootstrap_ci(matches, args.bootstrap, args.seed)

    rank = sorted(point_ratings.keys(), key=lambda p: -point_ratings[p])

    out = {
        "ratings": {p: round(point_ratings[p], 1) for p in rank},
        "ci_95": {p: [round(ci[p][0], 1), round(ci[p][1], 1)] for p in rank},
        "rank": rank,
        "n_matches": len(matches),
        "k": K,
        "init": INIT_RATING,
        "bootstrap_iters": args.bootstrap,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
