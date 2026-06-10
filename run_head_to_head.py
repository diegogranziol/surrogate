#!/usr/bin/env python3
"""Run the head-to-head benchmark: surrogate vs Claude, each with own tools.

Usage:
    python run_head_to_head.py <questions_file> [--limit N] [--k K]

Environment:
    ANTHROPIC_API_KEY            (required)        — for Claude
    FRONTIER_CLAUDE_MODEL        default claude-sonnet-4-6
    STAGE1_BASE_URL / STAGE2_*   surrogate vLLM box (must be reachable)

Outputs:
    backtests/h2h-store.jsonl    appended (one line per question)
    backtests/h2h-run-<ts>.md    side-by-side, verbatim, no curation
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from surrogate.head_to_head import (
    run_one_h2h, append_h2h, write_h2h_md, STORE_DIR,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("questions_file", help="Path to a .txt with one question per line.")
    ap.add_argument("--limit", type=int, default=0, help="Stop after N questions (0 = all).")
    ap.add_argument(
        "--k", type=int, default=None,
        help=("Top-N for soft-match. Default: auto-detected from each question "
              "(explicit 'top N' -> N; otherwise 5). Pass an int to force one "
              "value across the whole set."),
    )
    ap.add_argument(
        "--mode", choices=["structured", "natural", "auto"], default="auto",
        help=("Answer-shape mode for the frontier. 'auto' (default) infers from "
              "the question; 'structured' forces numbered-list mode; 'natural' "
              "lets the frontier answer freely."),
    )
    ap.add_argument(
        "--frontier", choices=["claude", "openai"], default="claude",
        help=("Which frontier model to compare against. 'claude' = Anthropic "
              "Claude + web_search tool. 'openai' = OpenAI gpt-5 (or "
              "FRONTIER_OPENAI_MODEL) via Responses API + web_search."),
    )
    args = ap.parse_args()

    qs = [
        q.strip()
        for q in Path(args.questions_file).read_text().splitlines()
        if q.strip() and not q.lstrip().startswith("#")
    ]
    if args.limit:
        qs = qs[: args.limit]

    if not qs:
        print(f"No questions in {args.questions_file}", file=sys.stderr)
        return 2

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    STORE_DIR.mkdir(exist_ok=True, parents=True)
    md_path = STORE_DIR / f"h2h-run-{ts}.md"

    entries: list[dict] = []
    forced_mode = None if args.mode == "auto" else args.mode
    print(f"\nFrontier: {args.frontier}\n", flush=True)
    for i, q in enumerate(qs, 1):
        print(f"\n[{i}/{len(qs)}] {q}", flush=True)
        try:
            e = run_one_h2h(q, k=args.k, mode=forced_mode, frontier=args.frontier)
            append_h2h(e)
            entries.append(e)
            m = e["match"]
            print(
                f"  mode={e.get('mode')} k={e['k']}  "
                f"overlap={m['overlap']}/{len(m['a'])}  "
                f"sur={e['surrogate']['ranked']}  fr={e['frontier']['ranked']}",
                flush=True,
            )
        except Exception as ex:
            print(f"  ERROR: {ex!r}", flush=True)

    write_h2h_md(entries, md_path)
    if entries:
        print(f"\nWrote {md_path}")
        # Per-mode summary (denominators differ between modes).
        by_mode: dict[str, list[dict]] = {}
        for e in entries:
            by_mode.setdefault(e.get("mode", "structured"), []).append(e)
        for mode, group in by_mode.items():
            avg = sum(e["match"]["overlap"] for e in group) / len(group)
            mean_k = sum(len(e["match"]["a"]) or e["k"] for e in group) / len(group)
            print(f"  {mode}: mean overlap {avg:.2f} / {mean_k:.1f}  ({len(group)} q)")
    else:
        print(f"\nNo successful runs; nothing written to {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
