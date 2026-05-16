"""Minimal terminal REPL for the surrogate.

Usage:
    source .venv/bin/activate
    python run_chat.py

After each turn, the path to the full trace is printed.
"""
from __future__ import annotations

import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from surrogate.agent import chat
from surrogate.logger import TraceLogger


console = Console()


def main() -> int:
    console.print(Panel.fit(
        "[bold]Surrogate REPL[/bold]\n"
        "Connected to qwen2.5-7b via http://localhost:8000/v1\n"
        "Ctrl-D or empty line to quit.",
        border_style="cyan",
    ))
    history = None
    log: TraceLogger | None = None
    try:
        while True:
            try:
                q = console.input("[bold green]you>[/bold green] ").strip()
            except EOFError:
                console.print()
                break
            if not q:
                break

            if log is None:
                log = TraceLogger(q)
            answer, history, log = chat(q, history, log=log)
            console.print()
            console.print(Markdown(answer or "_(empty)_"))
            console.print(f"[dim]trace: {log.dir}/transcript.md[/dim]\n")
    finally:
        if log is not None:
            log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
