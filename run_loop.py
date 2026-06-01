"""CLI for the new single-stage ReAct loop (Tongyi-shaped).

  python run_loop.py "which restaurant is the best for italian food in Tashkent"

Writes a verbatim bundle under logs/<ts>-<slug>/ (trace.jsonl + transcript.md)
and prints the final answer + termination reason.
"""
from __future__ import annotations

import sys

from surrogate.loop import run
from surrogate.loop_tools import default_tools


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    question = " ".join(argv[1:])
    res = run(question, tools=default_tools())
    print("\n=" * 1, "ANSWER", "=" * 60)
    print(res.final_answer or "(no answer)")
    print()
    print(f"termination : {res.termination}")
    print(f"steps       : {res.steps}")
    print(f"duration_s  : {res.duration_s}")
    print(f"bundle      : {res.bundle_dir}")
    return 0 if res.termination == "answer" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
