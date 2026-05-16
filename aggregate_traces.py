"""Aggregate two-stage bundles in `logs/` into a single markdown file.

PRIME DIRECTIVE (see CLAUDE.md): every field of every event is dumped verbatim.
Do NOT add slicing like `text[:N]`, do NOT strip `<think>...</think>`, do NOT
elide tool specs / tool calls / tool results / system prompts. The user is
studying the model's thinking and any curation is a bug.

Usage:
    python aggregate_traces.py                       # aggregate all bundles
    python aggregate_traces.py 20260515              # only bundles whose ts starts with this
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


LOGS = Path("logs")
OUT_DIR = Path("runs") / f"aggregated-{datetime.now():%Y%m%d-%H%M%S}"


def fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def render_stage1(events: list[dict]) -> str:
    out: list[str] = []
    sstart = next((e for e in events if e["kind"] == "session_start"), None)
    if sstart:
        out.append("#### System prompt\n```\n" + (sstart.get("system") or "") + "\n```\n")
        tools = sstart.get("tools") or []
        out.append(f"#### Tool specs ({len(tools)})\n```json\n" +
                   json.dumps(tools, indent=2, ensure_ascii=False) + "\n```\n")
        sampling = sstart.get("sampling") or {}
        out.append(f"#### Sampling\n`{json.dumps(sampling)}`\n")
        out.append(f"#### User question\n```\n{sstart.get('user_question', '')}\n```\n")

    for e in events:
        k = e["kind"]
        if k == "llm_request":
            out.append(f"\n##### llm_request #{e['idx']} step={e.get('step')}")
            out.append("Full `messages` array sent to model:")
            out.append("```json\n" + json.dumps(e.get("messages", []), indent=2, ensure_ascii=False, default=str) + "\n```")
        elif k == "llm_response":
            u = e.get("usage") or {}
            tcs = e.get("tool_calls") or []
            out.append(
                f"\n##### llm_response #{e['idx']} step={e.get('step')}  "
                f"took={e.get('duration_s', 0):.2f}s  "
                f"tokens={u.get('prompt_tokens','?')}+{u.get('completion_tokens','?')}  "
                f"finish={e.get('finish_reason')}"
            )
            rc = e.get("reasoning_content")
            if rc:
                out.append(f"reasoning_content ({len(rc)} chars):")
                out.append("```\n" + str(rc) + "\n```")
            c = e.get("content")
            if c is not None:
                out.append(f"content ({len(c)} chars):")
                out.append("```\n" + str(c) + "\n```")
            for tc in tcs:
                fn = tc.get("function") or {}
                out.append(f"tool_call → `{fn.get('name')}` args={fn.get('arguments')!r}  id=`{tc.get('id')}`")
        elif k == "tool_call":
            out.append(f"\n##### tool_call #{e['idx']} **{e.get('name')}** id=`{e.get('id')}`")
            out.append("args:")
            out.append("```json\n" + json.dumps(e.get("args", {}), indent=2, ensure_ascii=False) + "\n```")
        elif k == "tool_result":
            out.append(f"\n##### tool_result #{e['idx']} **{e.get('name')}** took={e.get('duration_s', 0):.2f}s")
            out.append("```\n" + str(e.get("result", "")) + "\n```")
        elif k == "tool_error":
            out.append(f"\n##### tool_error #{e['idx']} **{e.get('name')}**")
            out.append("```\n" + str(e.get("error", "")) + "\n```")
        elif k == "final_answer":
            out.append("\n##### final_answer")
            out.append("```\n" + str(e.get("content", "")) + "\n```")
    return "\n".join(out) + "\n"


def render_stage2(events: list[dict], bundle: Path) -> str:
    out: list[str] = []
    sstart = next((e for e in events if e["kind"] == "session_start"), None)
    if sstart:
        out.append("#### System prompt\n```\n" + (sstart.get("system") or "") + "\n```\n")

    # The full user message sent to stage 2 is stored on disk as stage2-input.md
    stage2_input = bundle / "stage2-input.md"
    if stage2_input.exists():
        out.append("#### Stage-2 input (system + user message, verbatim)\n")
        out.append("```\n" + stage2_input.read_text() + "\n```\n")

    # If multi-sample (sample_index present on responses), render each sample
    # as its own labelled sub-section.
    resps = [e for e in events if e["kind"] == "llm_response"]
    errs = [e for e in events if e["kind"] == "llm_error"]
    multi = any("sample_index" in e for e in resps + errs)
    if multi:
        out.append("\n#### Stage-2 samples\n")
        # Combine responses + errors keyed by sample_index for ordered render
        by_idx: dict[int, dict] = {}
        for e in resps + errs:
            by_idx[e.get("sample_index", e.get("step", 0))] = e
        for si in sorted(by_idx):
            e = by_idx[si]
            T = e.get("temperature", "?")
            tag = "greedy" if T == 0 or T == 0.0 else f"T={T}"
            out.append(f"\n##### sample {si}  ({tag})  took={e.get('duration_s', 0):.2f}s")
            if e["kind"] == "llm_error":
                out.append(f"ERROR: {e.get('error')}")
                continue
            u = e.get("usage") or {}
            out.append(
                f"tokens={u.get('prompt_tokens','?')}+{u.get('completion_tokens','?')}  "
                f"finish={e.get('finish_reason')}"
            )
            rc = e.get("reasoning_content")
            if rc:
                out.append(f"reasoning_content ({len(rc)} chars):")
                out.append("```\n" + str(rc) + "\n```")
            c = e.get("content")
            if c is not None:
                out.append(f"content ({len(c)} chars):")
                out.append("```\n" + str(c) + "\n```")
        return "\n".join(out) + "\n"

    # Legacy single-sample path
    for e in events:
        if e["kind"] == "llm_request":
            out.append(f"\n##### llm_request #{e['idx']}")
            out.append("Full payload sent (already shown above via stage2-input.md). Sampling:")
            out.append(f"`temperature={e.get('temperature')} top_p={e.get('top_p')} max_tokens={e.get('max_tokens')}`")
        elif e["kind"] == "llm_response":
            u = e.get("usage") or {}
            out.append(
                f"\n##### llm_response #{e['idx']}  "
                f"took={e.get('duration_s', 0):.2f}s  "
                f"tokens={u.get('prompt_tokens','?')}+{u.get('completion_tokens','?')}  "
                f"finish={e.get('finish_reason')}"
            )
            rc = e.get("reasoning_content")
            if rc:
                out.append(f"reasoning_content ({len(rc)} chars):")
                out.append("```\n" + str(rc) + "\n```")
            c = e.get("content")
            if c is not None:
                out.append(f"content ({len(c)} chars):")
                out.append("```\n" + str(c) + "\n```")
        elif e["kind"] == "final_answer":
            out.append("\n##### final_answer")
            out.append("```\n" + str(e.get("content", "")) + "\n```")
    return "\n".join(out) + "\n"


def load(jsonl: Path) -> list[dict]:
    return [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]


def collect_bundles(prefix: str = "") -> list[Path]:
    bundles = []
    for d in sorted(LOGS.glob("two-stage-*")):
        if not d.is_dir():
            continue
        if prefix and prefix not in d.name:
            continue
        # need at least one stage1 + one stage2 subdir
        s1 = [p for p in d.iterdir() if p.is_dir() and p.name.endswith("-stage1")]
        s2 = [p for p in d.iterdir() if p.is_dir() and p.name.endswith("-stage2")]
        if s1 and s2:
            bundles.append(d)
    return bundles


def main() -> int:
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    bundles = collect_bundles(prefix)
    if not bundles:
        print("no bundles found", file=sys.stderr)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    master = OUT_DIR / "all-answers.md"

    sections: list[tuple[int, str, Path, dict | None, dict | None]] = []

    for i, b in enumerate(bundles, 1):
        s1_dir = next(p for p in b.iterdir() if p.is_dir() and p.name.endswith("-stage1"))
        s2_dir = next(p for p in b.iterdir() if p.is_dir() and p.name.endswith("-stage2"))
        s1_events = load(s1_dir / "trace.jsonl")
        s2_events = load(s2_dir / "trace.jsonl")
        s1_start = next((e for e in s1_events if e["kind"] == "session_start"), None)
        question = (s1_start or {}).get("user_question") or s1_dir.name.split("-", 1)[-1]
        sections.append((i, question, b, s1_events, s2_events))

    with master.open("w") as f:
        f.write(f"# Aggregated two-stage traces — {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(f"{len(bundles)} bundles • Qwen2.5-7B (stage 1, tools) → "
                f"Nemotron-Super-49B-FP8 / Qwen3-32B (stage 2, reasoning)\n\n")
        f.write("## Table of contents\n\n")
        for i, q, *_ in sections:
            f.write(f"- [Q{i:02d}: {q}](#q{i:02d})\n")
        f.write("\n---\n\n")

        for i, q, b, s1_events, s2_events in sections:
            s1_start = next((e for e in s1_events if e["kind"] == "session_start"), {})
            s2_start = next((e for e in s2_events if e["kind"] == "session_start"), {})
            s1_resp = [e for e in s1_events if e["kind"] == "llm_response"]
            s2_resp = [e for e in s2_events if e["kind"] == "llm_response"]
            s1_tool = [e for e in s1_events if e["kind"] == "tool_call"]
            s1_total = sum(e.get("duration_s", 0) for e in s1_resp) + sum(
                e.get("duration_s", 0) for e in s1_events if e["kind"] in ("tool_result","tool_error"))
            s2_total = sum(e.get("duration_s", 0) for e in s2_resp)

            f.write(f"## Q{i:02d}: {q}  <a id=\"q{i:02d}\"></a>\n\n")
            f.write(f"**Bundle**: `{b}`\n\n")
            f.write(f"**Stage 1**: `{s1_start.get('model','?')}` @ `{s1_start.get('base_url','?')}` — "
                    f"{s1_total:.1f}s — {len(s1_tool)} tool calls\n")
            f.write(f"**Stage 2**: `{s2_start.get('model','?')}` @ `{s2_start.get('base_url','?')}` — "
                    f"{s2_total:.1f}s\n\n")
            f.write("### Stage 1 — full trace\n\n")
            f.write(render_stage1(s1_events))
            f.write("\n### Stage 2 — full trace\n\n")
            f.write(render_stage2(s2_events, b))
            f.write("\n---\n\n")

    print(f"wrote {master}  ({master.stat().st_size:,} bytes, {len(sections)} sections)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
