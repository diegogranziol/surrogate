"""Frontier comparator: OpenAI (gpt-5 / gpt-4.1) via Responses API + web_search.

Mirror of `frontier_claude.py` for the OpenAI side. Uses the modern Responses
API (not Chat Completions) because it supports the native `web_search` tool
and the reasoning output shape we want.

Returns the SAME dict shape as `ask_claude`, so the head-to-head harness can
swap frontiers via env var without code changes:

    {
      "model":        str,
      "answer":       str,         # final text content
      "thinking":     str,         # concatenated reasoning summaries (verbatim)
      "tool_calls":   list[dict],  # web_search invocations
      "blocks_raw":   list[dict],  # every output item dict, verbatim
      "usage":        dict,
      "stop_reason":  str,
    }

Config via .env:
    OPENAI_API_KEY               (required)
    FRONTIER_OPENAI_MODEL        default gpt-5
    FRONTIER_OPENAI_EFFORT       default 'medium' — reasoning effort (low|medium|high)
    FRONTIER_OPENAI_MAX_TOKENS   default 8000
"""
from __future__ import annotations

import os
import time
from typing import Any

from openai import OpenAI
from openai import APIConnectionError, APIError, APIStatusError, RateLimitError

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def current_openai_model() -> str:
    return os.environ.get("FRONTIER_OPENAI_MODEL", "gpt-5")


def _effort() -> str:
    return os.environ.get("FRONTIER_OPENAI_EFFORT", "medium")


def _max_tokens() -> int:
    return int(os.environ.get("FRONTIER_OPENAI_MAX_TOKENS", "8000"))


# Same shape of instruction as the Claude side, structured/natural switch.
SYSTEM_STRUCTURED = (
    "You are a careful research assistant answering purchase-intent questions "
    '("best X in Y", "top N products for ..."). Use web_search liberally to '
    "gather current evidence — do not rely on memory. The user asked for a "
    "specific number of items; present a clearly NUMBERED ranked list of that "
    "many items, best first. Each entry: entity name as the first thing on the "
    "line, one concrete reason, source URL. Be specific and current."
)

SYSTEM_NATURAL = (
    "You are a careful research assistant answering purchase-intent questions. "
    "Use web_search liberally to gather current evidence — do not rely on "
    "memory. Answer in the style the user asked: if they asked 'what X should "
    "I use', a single confident pick (with strong reasoning + a couple of "
    "alternatives) is appropriate; if they asked 'best X' without naming a "
    "count, give 3–10 ranked items. Each item should have a name, a concrete "
    "evidence-grounded reason, and a source URL. Be specific and current."
)


def ask_openai(question: str, *, mode: str = "structured") -> dict[str, Any]:
    """One call to OpenAI via Responses API with web_search + reasoning."""
    client = OpenAI(max_retries=5)

    sys_prompt = SYSTEM_STRUCTURED if mode == "structured" else SYSTEM_NATURAL

    # Responses API takes `instructions` (system) + `input` (user) separately.
    kwargs: dict[str, Any] = {
        "model": current_openai_model(),
        "instructions": sys_prompt,
        "input": question,
        "tools": [{"type": "web_search"}],
        "max_output_tokens": _max_tokens(),
        # `summary: auto` asks the API to emit human-readable reasoning
        # summaries (otherwise only encrypted_content is returned). This is
        # the OpenAI analog of Claude's extended-thinking blocks.
        "reasoning": {"effort": _effort(), "summary": "auto"},
    }

    # Backoff schedule mirrors the Claude client: 5/15/45/135/300s.
    delays = [5, 15, 45, 135, 300]
    resp = None
    last_err: Exception | None = None
    for attempt, delay in enumerate([0] + delays):
        if delay:
            time.sleep(delay)
        try:
            resp = client.responses.create(**kwargs)
            break
        except (RateLimitError, APIConnectionError) as e:
            last_err = e
            print(f"  [openai retry] attempt {attempt + 1}: {type(e).__name__} — "
                  f"sleeping {delays[attempt] if attempt < len(delays) else 0}s")
            continue
        except APIStatusError as e:
            if getattr(e, "status_code", None) in (500, 502, 503, 504, 529):
                last_err = e
                next_d = delays[attempt] if attempt < len(delays) else 0
                print(f"  [openai retry] attempt {attempt + 1}: "
                      f"{type(e).__name__} {e.status_code} — sleeping {next_d}s")
                continue
            raise
        except APIError as e:
            last_err = e
            continue
    if resp is None:
        raise RuntimeError(f"OpenAI call failed after {len(delays) + 1} attempts: {last_err!r}")

    # Walk output items, preserving each verbatim per CLAUDE.md (no truncation).
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict] = []
    raw_blocks: list[dict] = []
    for item in resp.output:
        d = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        raw_blocks.append(d)
        t = d.get("type")
        if t == "message":
            # message.content is a list of {type, text, annotations}
            for c in d.get("content") or []:
                if isinstance(c, dict) and c.get("type") == "output_text":
                    text_parts.append(c.get("text", ""))
                elif isinstance(c, dict) and "text" in c:
                    text_parts.append(c.get("text", ""))
        elif t == "reasoning":
            # `summary` is the visible reasoning; `content` is sometimes used.
            # `encrypted_content` is opaque — store as-is, don't try to read.
            summary = d.get("summary") or []
            for s in summary if isinstance(summary, list) else [summary]:
                if isinstance(s, dict):
                    reasoning_parts.append(s.get("text", str(s)))
                elif s:
                    reasoning_parts.append(str(s))
            content = d.get("content") or []
            for c in content if isinstance(content, list) else [content]:
                if isinstance(c, dict):
                    reasoning_parts.append(c.get("text", str(c)))
                elif c:
                    reasoning_parts.append(str(c))
        elif t == "web_search_call":
            tool_calls.append({
                "kind": "web_search_call",
                "id": d.get("id"),
                "status": d.get("status"),
                "action": d.get("action"),
            })

    # Prefer the SDK's pre-joined output_text if our walk produced nothing.
    answer = "\n".join(p for p in text_parts if p).strip()
    if not answer:
        answer = (getattr(resp, "output_text", "") or "").strip()

    usage_d = {}
    if resp.usage is not None:
        usage_d = resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else dict(resp.usage)

    return {
        "model": resp.model,
        "answer": answer,
        "thinking": "\n\n---\n\n".join(p for p in reasoning_parts if p).strip(),
        "tool_calls": tool_calls,
        "blocks_raw": raw_blocks,
        "usage": usage_d,
        "stop_reason": getattr(resp, "status", None) or "completed",
    }
