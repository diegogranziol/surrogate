"""Query Claude + gpt-5 with a list of questions, capture every URL they
cite or consult, append to backtests/frontier_mining.jsonl.

This is a frontier-only data-gathering script — no surrogate calls. Used to
expand the citation pool that feeds `mine_frontier_domains.py` so the
trusted-domain list has more signal per category.

Usage:
    python scripts/mine_with_frontiers.py data/h2h-mining-supplements.txt
    python scripts/mine_with_frontiers.py data/h2h-mining-supplements.txt --frontier openai
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from surrogate.frontier_claude import ask_claude
from surrogate.frontier_openai import ask_openai


def claude_urls(resp: dict) -> list[str]:
    out: list[str] = []
    for tc in resp.get("tool_calls") or []:
        if tc.get("kind") != "tool_result":
            continue
        content = tc.get("content") or []
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "web_search_result":
                    u = item.get("url")
                    if u:
                        out.append(u)
    return out


def openai_urls(resp: dict) -> list[str]:
    out: list[str] = []
    for blk in resp.get("blocks_raw") or []:
        if not isinstance(blk, dict) or blk.get("type") != "message":
            continue
        for c in blk.get("content") or []:
            if not isinstance(c, dict):
                continue
            for a in c.get("annotations") or []:
                if isinstance(a, dict) and a.get("type") == "url_citation":
                    u = a.get("url")
                    if u:
                        out.append(u)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("questions_file", type=Path)
    ap.add_argument("--frontier", choices=["claude", "openai", "both"], default="both")
    ap.add_argument("--out", type=Path, default=ROOT / "backtests/frontier_mining.jsonl")
    ap.add_argument("--mode", choices=["structured", "natural"], default="structured",
                    help="System-prompt mode. 'structured' surfaces more URLs (default).")
    args = ap.parse_args()

    qs = [
        l.strip()
        for l in args.questions_file.read_text().splitlines()
        if l.strip() and not l.lstrip().startswith("#")
    ]
    if not qs:
        print(f"No questions in {args.questions_file}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frontiers = ["claude", "openai"] if args.frontier == "both" else [args.frontier]

    print(f"Mining {len(qs)} questions × {len(frontiers)} frontiers → {args.out}\n", flush=True)
    for i, q in enumerate(qs, 1):
        print(f"[{i}/{len(qs)}] {q}", flush=True)
        for fr in frontiers:
            t0 = time.time()
            try:
                if fr == "claude":
                    resp = ask_claude(q, mode=args.mode)
                    urls = claude_urls(resp)
                else:
                    resp = ask_openai(q, mode=args.mode)
                    urls = openai_urls(resp)
                entry = {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "frontier": fr,
                    "model": resp.get("model"),
                    "question": q,
                    "mode": args.mode,
                    "urls": urls,
                    "duration_s": round(time.time() - t0, 1),
                }
                with args.out.open("a") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                print(f"  {fr}: {len(urls)} URLs ({entry['duration_s']}s)", flush=True)
            except Exception as e:
                print(f"  {fr}: ERROR {type(e).__name__}: {e!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
