"""Reference-model client (Phase 1).

A clean SINGLE-SHOT call to a frontier reference model, used to backtest the
self-hosted surrogate against. We deliberately do NOT route through an agent
harness (e.g. Claude Code) — that would measure the wrapper, not the model,
and would not be reproducible. We want: question (+ optional evidence) in,
one answer + its thinking out.

Backend: z.ai's Anthropic-compatible endpoint, driven by the plain `anthropic`
SDK. Config via .env:
    ZAI_API_KEY        (required)            — z.ai key
    ZAI_BASE_URL       default https://api.z.ai/api/anthropic
    REFERENCE_MODEL    default glm-4.6       — e.g. glm-4.6 / glm-5.1
    REFERENCE_THINKING_BUDGET  default 2048  — Anthropic extended-thinking budget

Returns a dict with the model id, the answer text, the thinking text (if the
backend exposed any), token usage, and the raw response, so callers can log
everything verbatim (CLAUDE.md prime directive).
"""
from __future__ import annotations

import os
from typing import Any

from anthropic import Anthropic

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


ZAI_BASE_URL = os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/anthropic")
# REFERENCE_MODEL / THINKING_BUDGET are re-read from env at every call (see
# `current_reference_model()` and `ask_reference`) so the Settings tab can
# change them without forcing a process restart. The module-level constants
# stay as the import-time snapshot for backward compat / display.
REFERENCE_MODEL = os.environ.get("REFERENCE_MODEL", "glm-4.6")
THINKING_BUDGET = int(os.environ.get("REFERENCE_THINKING_BUDGET", "2048"))


def current_reference_model() -> str:
    """Live value of REFERENCE_MODEL, re-read from env on each call."""
    return os.environ.get("REFERENCE_MODEL", REFERENCE_MODEL)


def current_thinking_budget() -> int:
    try:
        return int(os.environ.get("REFERENCE_THINKING_BUDGET", THINKING_BUDGET))
    except ValueError:
        return THINKING_BUDGET


def _client() -> Anthropic:
    key = os.environ.get("ZAI_API_KEY")
    if not key:
        raise RuntimeError(
            "ZAI_API_KEY not set. Put it in surrogate/.env (gitignored)."
        )
    # z.ai's coding plan authenticates via ANTHROPIC_AUTH_TOKEN (Bearer).
    # The anthropic SDK's auth_token= sets `Authorization: Bearer ...` and
    # omits x-api-key, which matches z.ai's expectation.
    return Anthropic(base_url=ZAI_BASE_URL, auth_token=key)


def _split_blocks(content: list[Any]) -> tuple[str, str]:
    """Return (thinking_text, answer_text) from a messages-API content list."""
    thinking_parts, text_parts = [], []
    for block in content or []:
        btype = getattr(block, "type", None)
        if btype == "thinking":
            thinking_parts.append(getattr(block, "thinking", "") or "")
        elif btype == "redacted_thinking":
            thinking_parts.append("[redacted_thinking]")
        elif btype == "text":
            text_parts.append(getattr(block, "text", "") or "")
    return "\n".join(thinking_parts), "\n".join(text_parts)


def ask_reference(
    question: str,
    *,
    system: str | None = None,
    evidence: str | None = None,
    model: str | None = None,
    max_tokens: int = 4096,
    thinking: bool = True,
) -> dict:
    """One single-shot reference answer.

    `evidence`, if given, is appended to the user message (Phase 2 will pass
    the surrogate's gathered tool outputs here so the comparison is apples-to-
    apples). Phase 1 just passes the bare question.
    """
    client = _client()
    mdl = model or current_reference_model()

    user = question if not evidence else f"{question}\n\nEVIDENCE:\n{evidence}"
    msgs = [{"role": "user", "content": user}]

    base_kwargs: dict[str, Any] = {"model": mdl, "max_tokens": max_tokens, "messages": msgs}
    if system:
        base_kwargs["system"] = system

    used_thinking = False
    try:
        if thinking:
            resp = client.messages.create(
                **base_kwargs,
                thinking={"type": "enabled", "budget_tokens": current_thinking_budget()},
            )
            used_thinking = True
        else:
            resp = client.messages.create(**base_kwargs)
    except Exception as e:
        # Backend may reject the Anthropic `thinking` param. Retry plain so we
        # still get an answer; record that thinking was unavailable this way.
        if thinking:
            resp = client.messages.create(**base_kwargs)
        else:
            raise RuntimeError(f"reference call failed: {e!r}") from e

    thinking_text, answer_text = _split_blocks(resp.content)
    usage = getattr(resp, "usage", None)
    return {
        "model": mdl,
        "base_url": ZAI_BASE_URL,
        "thinking_requested": thinking,
        "thinking_param_accepted": used_thinking,
        "thinking": thinking_text,
        "answer": answer_text,
        "usage": usage.model_dump() if usage and hasattr(usage, "model_dump") else None,
        "stop_reason": getattr(resp, "stop_reason", None),
        "raw": resp.model_dump() if hasattr(resp, "model_dump") else None,
    }
