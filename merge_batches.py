"""Merge two batch master MDs into one Q01-Q35 ordered file.

For each canonical question in data/questions.txt, find its section in either
input MD (the original Mac batch with Q01-Q19 OK, or the box-resident batch
with the missing 16 ordered differently) and emit it under its canonical number.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path


SOURCES = [
    Path("runs/batch-20260515-005853/all-answers.md"),
    Path("runs/batch-20260515-091445-from-box/all-answers.md"),
]
QUESTIONS = Path("data/questions.txt")
OUT = Path("runs") / f"merged-{datetime.now():%Y%m%d-%H%M%S}" / "all-answers.md"


SECTION_RE = re.compile(
    r"^## Q\d+:\s*(?P<q>.+?)\s*<a id=\"q\d+\"></a>\s*$",
    re.MULTILINE,
)


def parse_sections(md: str) -> dict[str, str]:
    """Return {question_text: section_body_without_header}. Bodies end at next
    '## Q' or end of file. Skip ERROR sections (no useful body)."""
    out: dict[str, str] = {}
    matches = list(SECTION_RE.finditer(md))
    for i, m in enumerate(matches):
        q = m.group("q").strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        body = md[start:end].strip()
        # skip Q??/ERROR placeholders that have no real trace
        if body.startswith("**Status**: ERROR") or "**Status**: ERROR" in body[:200]:
            continue
        out[q] = body
    return out


def main() -> int:
    pool: dict[str, str] = {}
    for src in SOURCES:
        if not src.exists():
            print(f"missing: {src}", file=sys.stderr)
            continue
        sections = parse_sections(src.read_text())
        # newer wins on duplicate question text (last source listed)
        pool.update(sections)
        print(f"[{src}] {len(sections)} good sections", file=sys.stderr)

    questions = [l.strip() for l in QUESTIONS.read_text().splitlines() if l.strip()]
    OUT.parent.mkdir(parents=True, exist_ok=True)

    matched = 0
    missing: list[tuple[int, str]] = []
    with OUT.open("w") as f:
        f.write(f"# Merged batch — Q01-Q{len(questions):02d}  ({datetime.now():%Y-%m-%d %H:%M})\n\n")
        f.write(f"{len(questions)} questions · Qwen2.5-7B → Nemotron-Super-49B-FP8\n\n")
        f.write("## Table of contents\n\n")
        for i, q in enumerate(questions, 1):
            mark = "" if q in pool else "  *(missing)*"
            f.write(f"- [Q{i:02d}: {q}](#q{i:02d}){mark}\n")
        f.write("\n---\n\n")

        for i, q in enumerate(questions, 1):
            f.write(f"## Q{i:02d}: {q}  <a id=\"q{i:02d}\"></a>\n")
            body = pool.get(q)
            if body is None:
                missing.append((i, q))
                f.write("\n**Status**: MISSING — no trace found in either source\n\n---\n\n")
                continue
            matched += 1
            f.write(body.rstrip() + "\n\n---\n\n")

    print(f"wrote {OUT}  ({OUT.stat().st_size:,} bytes)  matched={matched}  missing={len(missing)}")
    for i, q in missing:
        print(f"  missing Q{i:02d}: {q}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
