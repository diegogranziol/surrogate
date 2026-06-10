"""Frontier comparator: Claude with built-in web_search + extended thinking.

Single-shot call to Anthropic's Claude with the server-side `web_search` tool
enabled and extended thinking on. Returns a structured dict so the head-to-head
harness can store EVERY byte the model produced (per CLAUDE.md — full stack
trace, no curation).

Config via .env:
    ANTHROPIC_API_KEY            (required)
    FRONTIER_CLAUDE_MODEL        default claude-sonnet-4-6
    FRONTIER_CLAUDE_THINKING     default 4096        — extended-thinking budget; 0 disables
    FRONTIER_CLAUDE_SEARCH_USES  default 10          — max web_search calls per turn
    FRONTIER_CLAUDE_MAX_TOKENS   default 8000        — output cap
"""
from __future__ import annotations

import os
import time
from typing import Any

from anthropic import Anthropic
from anthropic import APIConnectionError, APIError, APIStatusError, RateLimitError

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def current_claude_model() -> str:
    return os.environ.get("FRONTIER_CLAUDE_MODEL", "claude-sonnet-4-6")


def _thinking_budget() -> int:
    return int(os.environ.get("FRONTIER_CLAUDE_THINKING", "4096"))


def _search_uses() -> int:
    return int(os.environ.get("FRONTIER_CLAUDE_SEARCH_USES", "10"))


def _max_tokens() -> int:
    return int(os.environ.get("FRONTIER_CLAUDE_MAX_TOKENS", "8000"))


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


def ask_claude(question: str, *, mode: str = "structured") -> dict[str, Any]:
    """One call to Claude with web_search + extended thinking.

    Returns:
        {
          "model":        str,                 # actual model id returned by API
          "answer":       str,                 # concatenated `text` blocks
          "thinking":     str,                 # concatenated `thinking` blocks (verbatim)
          "tool_calls":   list[dict],          # web_search invocations + results, verbatim
          "blocks_raw":   list[dict],          # every content block, dict-form, verbatim
          "usage":        dict,                # input/output token counts
          "stop_reason":  str,
        }

    Per CLAUDE.md prime directive: nothing in here gets truncated or summarised.
    """
    # max_retries=5 in the SDK covers connect/429, but 529 (overloaded) isn't
    # always retried automatically. We still wrap with our own backoff loop so
    # transient 529s and connection blips don't kill a benchmark mid-flight.
    client = Anthropic(max_retries=5)

    sys_prompt = SYSTEM_STRUCTURED if mode == "structured" else SYSTEM_NATURAL
    kwargs: dict[str, Any] = {
        "model": current_claude_model(),
        "max_tokens": _max_tokens(),
        "system": sys_prompt,
        "messages": [{"role": "user", "content": question}],
        "tools": [{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": _search_uses(),
        }],
    }
    tb = _thinking_budget()
    if tb > 0:
        # Extended thinking needs temperature=1 (Anthropic's constraint).
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": tb}
        kwargs["temperature"] = 1.0

    # Backoff: 5s, 15s, 45s, 135s, 300s — covers ~10 min of Anthropic outage.
    delays = [5, 15, 45, 135, 300]
    resp = None
    last_err: Exception | None = None
    for attempt, delay in enumerate([0] + delays):
        if delay:
            time.sleep(delay)
        try:
            resp = client.messages.create(**kwargs)
            break
        except (RateLimitError, APIConnectionError) as e:
            last_err = e
            print(f"  [claude retry] attempt {attempt + 1}: {type(e).__name__} — sleeping {delays[attempt] if attempt < len(delays) else 0}s")
            continue
        except APIStatusError as e:
            # 529 overloaded, 503 service unavailable, 500/502 — retryable.
            if getattr(e, "status_code", None) in (500, 502, 503, 529):
                last_err = e
                next_d = delays[attempt] if attempt < len(delays) else 0
                print(f"  [claude retry] attempt {attempt + 1}: {type(e).__name__} {e.status_code} — sleeping {next_d}s")
                continue
            raise  # non-retryable status
        except APIError as e:
            last_err = e
            continue
    if resp is None:
        raise RuntimeError(f"Claude call failed after {len(delays) + 1} attempts: {last_err!r}")

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict] = []
    raw_blocks: list[dict] = []
    for block in resp.content:
        d = block.model_dump() if hasattr(block, "model_dump") else dict(block)
        raw_blocks.append(d)
        t = d.get("type")
        if t == "text":
            text_parts.append(d.get("text", ""))
        elif t == "thinking":
            thinking_parts.append(d.get("thinking", ""))
        elif t == "server_tool_use":
            tool_calls.append({
                "kind": "tool_use",
                "id": d.get("id"),
                "name": d.get("name"),
                "input": d.get("input"),
            })
        elif t == "web_search_tool_result":
            tool_calls.append({
                "kind": "tool_result",
                "tool_use_id": d.get("tool_use_id"),
                "content": d.get("content"),
            })

    usage = {}
    if resp.usage is not None:
        usage = resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else dict(resp.usage)

    return {
        "model": resp.model,
        "answer": "\n".join(text_parts).strip(),
        "thinking": "\n\n---\n\n".join(p for p in thinking_parts if p).strip(),
        "tool_calls": tool_calls,
        "blocks_raw": raw_blocks,
        "usage": usage,
        "stop_reason": resp.stop_reason,
    }
