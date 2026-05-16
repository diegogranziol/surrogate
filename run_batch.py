"""Run every question in data/questions.txt through the two-stage pipeline and
aggregate the full trace of each into one master markdown file.

Reuses surrogate.two_stage.run_two_stage (no model swap; dual endpoint).
"""
from __future__ import annotations

import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from aggregate_traces import collect_bundles, render_stage1, render_stage2, load
from surrogate.two_stage import run_two_stage


QUESTIONS_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/questions.txt")
RUN_DIR = Path("runs") / f"batch-{datetime.now():%Y%m%d-%H%M%S}"
MASTER = RUN_DIR / "all-answers.md"


def _section_for(i: int, q: str, bundle: Path) -> str:
    s1_dir = next(p for p in bundle.iterdir() if p.is_dir() and p.name.endswith("-stage1"))
    s2_dir = next(p for p in bundle.iterdir() if p.is_dir() and p.name.endswith("-stage2"))
    s1_events = load(s1_dir / "trace.jsonl")
    s2_events = load(s2_dir / "trace.jsonl")
    s1_start = next((e for e in s1_events if e["kind"] == "session_start"), {})
    s2_start = next((e for e in s2_events if e["kind"] == "session_start"), {})
    s1_resp = [e for e in s1_events if e["kind"] == "llm_response"]
    s2_resp = [e for e in s2_events if e["kind"] == "llm_response"]
    s1_tool = [e for e in s1_events if e["kind"] == "tool_call"]
    s1_total = sum(e.get("duration_s", 0) for e in s1_resp) + sum(
        e.get("duration_s", 0) for e in s1_events if e["kind"] in ("tool_result", "tool_error")
    )
    s2_total = sum(e.get("duration_s", 0) for e in s2_resp)

    out = []
    out.append(f"## Q{i:02d}: {q}  <a id=\"q{i:02d}\"></a>\n")
    out.append(f"**Bundle**: `{bundle}`\n")
    out.append(
        f"**Stage 1**: `{s1_start.get('model','?')}` @ `{s1_start.get('base_url','?')}` — "
        f"{s1_total:.1f}s — {len(s1_tool)} tool calls\n"
    )
    out.append(
        f"**Stage 2**: `{s2_start.get('model','?')}` @ `{s2_start.get('base_url','?')}` — "
        f"{s2_total:.1f}s\n\n"
    )
    out.append("### Stage 1 — full trace\n\n")
    out.append(render_stage1(s1_events))
    out.append("\n### Stage 2 — full trace\n\n")
    out.append(render_stage2(s2_events, bundle))
    out.append("\n---\n\n")
    return "".join(out)


def _err_section(i: int, q: str, exc: BaseException) -> str:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return (
        f"## Q{i:02d}: {q}  <a id=\"q{i:02d}\"></a>\n\n"
        f"**Status**: ERROR\n\n```\n{tb}\n```\n\n---\n\n"
    )


def main() -> int:
    questions = [l.strip() for l in QUESTIONS_FILE.read_text().splitlines() if l.strip()]
    print(f"loaded {len(questions)} questions from {QUESTIONS_FILE}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    bundles_before = set(b.name for b in collect_bundles())

    with MASTER.open("w") as f:
        f.write(f"# Batch run — {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(
            f"{len(questions)} questions • "
            "Qwen2.5-7B (stage 1, tools) → Nemotron-Super-49B-FP8 (stage 2, reasoning)\n\n"
        )
        f.write("## Table of contents\n\n")
        for i, q in enumerate(questions, 1):
            f.write(f"- [Q{i:02d}: {q}](#q{i:02d})\n")
        f.write("\n---\n\n")

    t_batch = time.time()
    for i, q in enumerate(questions, 1):
        print(f"\n========== Q{i:02d}/{len(questions)}: {q} ==========")
        t0 = time.time()
        try:
            result = run_two_stage(q)
            section = _section_for(i, q, result.bundle_dir)
            print(f"[Q{i:02d}] OK in {time.time()-t0:.1f}s  bundle={result.bundle_dir.name}")
        except Exception as e:
            section = _err_section(i, q, e)
            print(f"[Q{i:02d}] ERROR in {time.time()-t0:.1f}s: {e!r}")

        with MASTER.open("a") as f:
            f.write(section)

    print(f"\n=== batch complete in {time.time()-t_batch:.0f}s ===")
    print(f"master: {MASTER}  ({MASTER.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
