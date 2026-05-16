"""CLI for the two-stage pipeline.

  python run_two_stage.py "Which restaurant is the best in Oxford for steak?"

Stage 1 (Qwen2.5-7B) does the tool calling.
Stage 2 (Qwen3-32B bf16) thinks over the same raw tool outputs and answers.
Both vLLM models are swapped sequentially on the remote box; expect ~60-90s
of model-reload overhead between stages.
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
