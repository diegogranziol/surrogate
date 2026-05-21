"""CLI for the two-stage pipeline.

  python run_two_stage.py "Which restaurant is the best in Oxford for steak?"

Stage 1 does the tool calling; Stage 2 reasons over the raw tool outputs.
Models come from .env (STAGE{1,2}_MODEL). Our setup: Qwen3-8B for both
stages on one vLLM server, SURROGATE_SKIP_SWAP=1 (no model swapping, no
reload overhead between stages).
"""
from __future__ import annotations

import sys

from surrogate.two_stage import run_two_stage


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    question = " ".join(argv[1:])
    run_two_stage(question)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
