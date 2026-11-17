#!/usr/bin/env python3
"""
call-judge.py — 统一封装三家 judge API（OpenAI / Google / DeepSeek），
对一对文章输出 {verdict, analysis, judge_id, raw_response, cost_usd}。

Usage:
    python3 call-judge.py \
        --judges benchmark/judge/judges.yaml \
        --prompt  benchmark/judge/prompts/pairwise.txt \
        --text-a  path/to/a.md \
        --text-b  path/to/b.md \
        [--judge gpt-4o]      # 只跑指定 judge；默认全跑
        [--dry-run]           # 不调 API，打印将要发送的 prompt
        [--output json|tsv]   # 输出格式，默认 json

Exit:
    0 on any successful verdict (even tie)
    1 on total failure (all judges 3 次重试仍失败)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:
    print("ERROR: pyyaml required. pip install pyyaml", file=sys.stderr)
    sys.exit(2)


VERDICT_RE = re.compile(r"<verdict>\s*(A|B|tie)\s*</verdict>", re.IGNORECASE)
ANALYSIS_RE = re.compile(r"<analysis>(.*?)</analysis>", re.DOTALL | re.IGNORECASE)


def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    # Strip html comment metadata block if present
    return re.sub(r"<!--.*?-->\s*", "", raw, count=1, flags=re.DOTALL).strip()


def build_prompt(template: str, text_a: str, text_b: str) -> str:
    return template.replace("{text_a}", text_a).replace("{text_b}", text_b)


def parse_response(raw: str) -> tuple[str, str]:
    m = VERDICT_RE.search(raw)
    if not m:
        raise ValueError(f"No <verdict> tag in response: {raw[:200]!r}")
    verdict = m.group(1).lower()
    if verdict not in ("a", "b", "tie"):
        raise ValueError(f"Invalid verdict: {verdict}")
    m2 = ANALYSIS_RE.search(raw)
    analysis = m2.group(1).strip() if m2 else ""
    return verdict.upper() if verdict != "tie" else "tie", analysis


def call_openai(judge: dict[str, Any], prompt: str) -> tuple[str, float]:
    import urllib.request

    key = os.environ.get(judge["api_key_env"])
    if not key:
        raise RuntimeError(f"env {judge['api_key_env']} not set")

    body = json.dumps(
        {
            "model": judge["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": judge["temperature"],
            "max_tokens": judge["max_tokens"],
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        judge["endpoint"],
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=judge["timeout_s"]) as r:
        data = json.loads(r.read())
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    # Rough cost estimate (judge-specific rates would go in judges.yaml v2)
    cost = (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)) * 5e-6
    return content, cost


def call_gemini(judge: dict[str, Any], prompt: str) -> tuple[str, float]:
    import urllib.request

    key = os.environ.get(judge["api_key_env"])
    if not key:
        raise RuntimeError(f"env {judge['api_key_env']} not set")

    body = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": judge["temperature"],
                "maxOutputTokens": judge["max_tokens"],
            },
        }
    ).encode("utf-8")

    url = f"{judge['endpoint']}?key={key}"
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=judge["timeout_s"]) as r:
        data = json.loads(r.read())
    content = data["candidates"][0]["content"]["parts"][0]["text"]
    usage = data.get("usageMetadata", {})
    cost = (
        usage.get("promptTokenCount", 0) + usage.get("candidatesTokenCount", 0)
    ) * 2e-6
    return content, cost


def call_deepseek(judge: dict[str, Any], prompt: str) -> tuple[str, float]:
    return call_openai(judge, prompt)  # DeepSeek is OpenAI-compatible


PROVIDER_DISPATCH = {
    "openai": call_openai,
    "google": call_gemini,
    "deepseek": call_deepseek,
}


def invoke_judge(judge: dict[str, Any], prompt: str, retries: dict[str, Any]) -> dict[str, Any]:
    fn = PROVIDER_DISPATCH[judge["provider"]]
    last_err = None
    for attempt in range(retries["max_attempts"]):
        try:
            raw, cost = fn(judge, prompt)
            verdict, analysis = parse_response(raw)
            if cost > judge["cost_cap_usd"]:
                raise RuntimeError(f"cost {cost:.4f} exceeds cap {judge['cost_cap_usd']}")
            return {
                "judge_id": judge["id"],
                "verdict": verdict,
                "analysis": analysis,
                "cost_usd": cost,
                "raw": raw,
                "attempts": attempt + 1,
            }
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(retries["backoff_s"])
    return {
        "judge_id": judge["id"],
        "verdict": None,
        "error": str(last_err),
        "attempts": retries["max_attempts"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--text-a", required=True)
    ap.add_argument("--text-b", required=True)
    ap.add_argument("--judge", help="only run this judge id")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output", choices=["json", "tsv"], default="json")
    ap.add_argument("--seed", type=int, default=0, help="for reproducible dry-run output")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.judges).read_text(encoding="utf-8"))
    prompt_tpl = load_prompt_template(Path(args.prompt))
    text_a = load_text(Path(args.text_a))
    text_b = load_text(Path(args.text_b))
    prompt = build_prompt(prompt_tpl, text_a, text_b)

    judges = cfg["judges"]
    if args.judge:
        judges = [j for j in judges if j["id"] == args.judge]
        if not judges:
            print(f"ERROR: judge {args.judge} not found", file=sys.stderr)
            return 2

    if args.dry_run:
        random.seed(args.seed)
        verdicts = ["A", "B", "tie"]
        fake = [
            {
                "judge_id": j["id"],
                "verdict": random.choice(verdicts),
                "analysis": f"[dry-run fake analysis for {j['id']}]",
                "cost_usd": 0.0,
                "dry_run": True,
            }
            for j in judges
        ]
        print(json.dumps({"prompt_preview": prompt[:300], "results": fake}, ensure_ascii=False, indent=2))
        return 0

    results = [invoke_judge(j, prompt, cfg["retries"]) for j in judges]

    if args.output == "json":
        print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    else:
        print("judge_id\tverdict\tcost_usd\tattempts")
        for r in results:
            print(f"{r['judge_id']}\t{r.get('verdict')}\t{r.get('cost_usd', 0):.4f}\t{r['attempts']}")

    return 0 if any(r.get("verdict") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
