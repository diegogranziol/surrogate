"""Trace logger for the surrogate pipeline.

PRIME DIRECTIVE (see CLAUDE.md at repo root): every event is written verbatim.
Do NOT add length caps, do NOT strip `<think>...</think>` blocks, do NOT redact
tool specs, do NOT summarise. The slug helper here trims at 40 chars for the
filesystem directory name only; the full question text is preserved in the
`session_start` event under `user_question`.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _default(o: Any):
    if hasattr(o, "model_dump"):
        return o.model_dump()
    if isinstance(o, (set, tuple)):
        return list(o)
    return repr(o)


def _slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:n] or "session"


class TraceLogger:
    """Per-session trace logger.

    Writes one directory per conversation:
      logs/<ts>-<slug>/
        ├── trace.jsonl     one JSON record per event (machine-readable)
        ├── transcript.md   human-readable rendering
        └── meta.json       written on close()
    """

    def __init__(self, question: str, log_root: str | Path = "logs"):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = Path(log_root) / f"{ts}-{_slug(question)}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = (self.dir / "trace.jsonl").open("a", encoding="utf-8")
        self._md = (self.dir / "transcript.md").open("a", encoding="utf-8")
        self._counts: dict[str, int] = {}
        self._start = time.time()
        self._md.write(f"# Trace — {question}\n\n_started {datetime.now().isoformat(timespec='seconds')}_\n\n")
        self._md.flush()

    def event(self, kind: str, **payload):
        self._counts[kind] = self._counts.get(kind, 0) + 1
        idx = self._counts[kind]
        rec = {"ts": time.time(), "kind": kind, "idx": idx, **payload}
        self._jsonl.write(json.dumps(rec, default=_default, ensure_ascii=False) + "\n")
        self._jsonl.flush()
        self._md.write(self._render(rec))
        self._md.flush()
        return rec

    def close(self, **meta):
        meta = {
            "started": self._start,
            "ended": time.time(),
            "duration_s": time.time() - self._start,
            "counts": self._counts,
            **meta,
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2, default=_default))
        self._jsonl.close()
        self._md.close()

    # ------------------------------------------------------------------ render

    def _render(self, rec: dict) -> str:
        k = rec["kind"]
        idx = rec["idx"]
        ts = datetime.fromtimestamp(rec["ts"]).strftime("%H:%M:%S")
        out = []
        if k == "session_start":
            out.append(f"## session_start ({ts})\n")
            out.append(f"- model: `{rec.get('model')}`\n")
            out.append(f"- base_url: `{rec.get('base_url')}`\n")
            out.append(f"- sampling: `{json.dumps(rec.get('sampling', {}))}`\n\n")
            out.append("### system prompt\n```\n" + (rec.get("system") or "") + "\n```\n\n")
            tools = rec.get("tools") or []
            out.append(f"### tools registered ({len(tools)})\n")
            out.append("```json\n" + json.dumps(tools, indent=2, ensure_ascii=False) + "\n```\n\n")
        elif k == "llm_request":
            out.append(f"## llm_request #{idx}  ({ts})  step={rec.get('step')}\n")
            out.append("### full messages array sent to model\n")
            out.append("```json\n" + json.dumps(rec.get("messages", []), indent=2, ensure_ascii=False, default=_default) + "\n```\n")
            out.append(f"### sampling\n`{json.dumps({k: rec.get(k) for k in ('temperature','top_p','max_tokens') if rec.get(k) is not None})}`\n\n")
        elif k == "llm_response":
            usage = rec.get("usage") or {}
            out.append(
                f"## llm_response #{idx}  ({ts})  "
                f"step={rec.get('step')}  "
                f"took={rec.get('duration_s', 0):.2f}s  "
                f"tokens={usage.get('prompt_tokens', '?')}+{usage.get('completion_tokens', '?')}  "
                f"finish={rec.get('finish_reason')}\n"
            )
            if rec.get("reasoning_content"):
                out.append("### <think> reasoning_content\n```\n" + str(rec["reasoning_content"]) + "\n```\n")
            if rec.get("content"):
                out.append("### content\n```\n" + str(rec["content"]) + "\n```\n")
            tcs = rec.get("tool_calls") or []
            if tcs:
                out.append(f"### tool_calls ({len(tcs)})\n")
                for tc in tcs:
                    fn = (tc.get("function") or {})
                    out.append(f"- `{tc.get('id')}`: **{fn.get('name')}**(`{fn.get('arguments')}`)\n")
            out.append("\n")
        elif k == "tool_call":
            out.append(f"## tool_call #{idx}  ({ts})  **{rec.get('name')}**  id=`{rec.get('id')}`\n")
            out.append("### args\n```json\n" + json.dumps(rec.get("args", {}), indent=2, ensure_ascii=False) + "\n```\n\n")
        elif k == "tool_result":
            out.append(
                f"## tool_result #{idx}  ({ts})  **{rec.get('name')}**  "
                f"took={rec.get('duration_s', 0):.2f}s  id=`{rec.get('id')}`\n"
            )
            out.append("### result\n```\n" + str(rec.get("result", "")) + "\n```\n\n")
        elif k == "tool_error":
            out.append(f"## tool_error #{idx}  ({ts})  **{rec.get('name')}**\n")
            out.append("```\n" + str(rec.get("error")) + "\n```\n\n")
        elif k == "final_answer":
            out.append(f"## final_answer  ({ts})\n")
            out.append("```\n" + str(rec.get("content", "")) + "\n```\n\n")
        elif k == "max_iters_reached":
            out.append(f"## max_iters_reached  ({ts})  iters={rec.get('iters')}\n\n")
        else:
            out.append(f"## {k}  ({ts})\n```json\n{json.dumps(rec, indent=2, default=_default, ensure_ascii=False)}\n```\n\n")
        return "".join(out)
