"""CLI: backtest the surrogate against the GLM reference.

  python run_backtest.py data/starter8.txt
  python run_backtest.py "single question in quotes"

For each question: run the surrogate two-stage pipeline, call GLM with the
SAME evidence and bare, extract top picks, append to backtests/store.jsonl,
and write a verbatim side-by-side backtests/run-<ts>.md.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from surrogate.backtest import run_one, append_store, write_run_md, STORE_DIR


def _load(arg: str) -> list[str]:
    p = Path(arg)
    if p.is_file():
        return [l.strip() for l in p.read_text().splitlines() if l.strip()]
    return [arg]


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    questions = _load(argv[1])
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    entries: list[dict] = []
    for i, q in enumerate(questions, 1):
        print(f"\n########## [{i}/{len(questions)}] {q}")
        try:
            e = run_one(q)
        except Exception as ex:
            print(f"  !! question failed: {ex!r}")
            continue
        append_store(e)
        entries.append(e)
        sr = e["surrogate"]["pick"].get("ranked") or []
        er = e["reference_evidence"]["pick"].get("ranked") or []
        ev = (e.get("top3") or {}).get("surrogate_vs_reference_evidence") or {}
        print(f"  sur top3={sr}")
        print(f"  GLM top3={er}  -> overlap={ev.get('overlap',0)}/{len(sr)} "
              f"({e['duration_s']}s)")
    if entries:
        out = STORE_DIR / f"run-{ts}.md"
        write_run_md(entries, out)
        any_match = sum(
            1 for e in entries
            if ((e.get("top3") or {}).get("surrogate_vs_reference_evidence") or {}).get("overlap", 0) >= 1
        )
        total = sum(((e.get("top3") or {}).get("surrogate_vs_reference_evidence") or {}).get("overlap", 0) for e in entries)
        denom = sum(len(e["surrogate"]["pick"].get("ranked") or []) for e in entries)
        print(f"\n=== top-3 overlap: {total}/{denom} items  | {any_match}/{len(entries)} questions ≥1 match ===")
        print(f"[store] {STORE_DIR/'store.jsonl'}")
        print(f"[run md] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
