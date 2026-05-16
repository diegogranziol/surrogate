from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def load(log_dir: str | Path) -> list[dict]:
    """Load events from a trace.jsonl file."""
    p = Path(log_dir)
    if p.is_dir():
        p = p / "trace.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def replay(log_dir: str | Path) -> str:
    """Return the rendered transcript for a past session."""
    p = Path(log_dir)
    if p.is_dir():
        p = p / "transcript.md"
    return p.read_text(encoding="utf-8")


def iter_events(log_dir: str | Path) -> Iterator[dict]:
    for ev in load(log_dir):
        yield ev
