#!/usr/bin/env python3
"""Rebuild a head-to-head report by picking, for each question in a question
file, the LATEST entry from backtests/h2h-store.jsonl. Useful for splicing
retries into a previous run as if they never failed.

Usage:
    python scripts/rebuild_h2h_report.py data/h2h-10q.txt [--out backtests/h2h-rebuilt.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from surrogate.head_to_head import write_h2h_md, H2H_JSONL  # noqa: E402


def _read_questions(path: Path) -> list[str]:
    qs = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        qs.append(s)
    return qs


def _latest_entry_per_question(questions: list[str], store: Path) -> list[dict]:
    """For each question, return the most recent matching entry by ts."""
    all_entries = []
    for line in store.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            all_entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    out: list[dict] = []
    misses: list[str] = []
    for q in questions:
        matches = [e for e in all_entries if e.get("question", "").strip() == q.strip()]
        if not matches:
            misses.append(q)
            continue
        # Pick the latest by ts string (ISO8601 lex order works for UTC).
        latest = max(matches, key=lambda e: e.get("ts", ""))
        out.append(latest)
    if misses:
        print(f"WARNING: no entry found for {len(misses)} question(s):", file=sys.stderr)
        for q in misses:
            print(f"  - {q}", file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("questions_file", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output .md path. Default: backtests/h2h-rebuilt-<ts>.md")
    args = ap.parse_args()

    qs = _read_questions(args.questions_file)
    if not qs:
        print(f"No questions in {args.questions_file}", file=sys.stderr)
        return 2

    entries = _latest_entry_per_question(qs, H2H_JSONL)
    if not entries:
        print("No entries to render", file=sys.stderr)
        return 1

    out_path = args.out
    if out_path is None:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_path = Path("backtests") / f"h2h-rebuilt-{ts}.md"
    out_path.parent.mkdir(exist_ok=True, parents=True)

    write_h2h_md(entries, out_path)
    print(f"Wrote {out_path} ({len(entries)}/{len(qs)} questions resolved)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
