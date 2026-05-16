from __future__ import annotations

import json
import time
from typing import Any

from surrogate.llm import MODEL, BASE_URL, make_client
from surrogate.logger import TraceLogger
from surrogate.tools import TOOL_IMPLS, TOOL_SPECS


SYSTEM = (
    "You are a careful research assistant with access to web_search and fetch_url. "
    "When a question depends on current, local, or time-sensitive information "
    "(restaurants, prices, news, weather, recent events, places), you MUST:\n"
    "  1. Call web_search with a specific query (include location/country if relevant).\n"
    "  2. Pick the most authoritative result (TripAdvisor, Time Out, Guardian, "
    "BBC, the venue's own site — NOT TikTok, NOT generic global lists).\n"
    "  3. Call fetch_url on that result to read its actual content.\n"
    "  4. If the fetched page is not useful, search again with a better query OR "
    "fetch a different result. Do not give up after one search.\n"
    "  5. Only then answer the user, citing the specific source URL(s) you read.\n"
    "Never fabricate a venue name or invent details you did not see in tool output. "
    "If after 2-3 searches you cannot find a good source, say so honestly."
)

SAMPLING: dict[str, Any] = {"temperature": 0.3, "top_p": 1.0, "max_tokens": 1024}


def _tc_to_dict(tc) -> dict:
    if hasattr(tc, "model_dump"):
        return tc.model_dump()
    return dict(tc)


def chat(
    user_msg: str,
    history: list[dict] | None = None,
    max_iters: int = 6,
    log: TraceLogger | None = None,
):
    """Run one user turn through the agent loop.

    Returns: (final_text, updated_messages, log)
    The full prompt context sent to the model is captured on every llm_request
    event in `log.dir/trace.jsonl` and rendered in `log.dir/transcript.md`.
    """
    client = make_client()
    close_log = False
    if log is None:
        log = TraceLogger(user_msg)
        close_log = True

    msgs: list[dict] = list(history or [{"role": "system", "content": SYSTEM}])
    msgs.append({"role": "user", "content": user_msg})

    log.event(
        "session_start",
        model=MODEL,
        base_url=BASE_URL,
        system=SYSTEM,
        tools=TOOL_SPECS,
        sampling=SAMPLING,
        user_question=user_msg,
    )

    final_text = "[no answer]"
    try:
        for step in range(max_iters):
            log.event(
                "llm_request",
                step=step,
                messages=msgs,
                tools=TOOL_SPECS,
                **SAMPLING,
            )
            t0 = time.time()
            resp = client.chat.completions.create(
                model=MODEL,
                messages=msgs,
                tools=TOOL_SPECS,
                tool_choice="auto",
                # Qwen3's chat template is hybrid: thinking is OFF by default.
                # extra_body is passed through to vLLM which forwards
                # chat_template_kwargs into apply_chat_template. Non-Qwen3
                # models ignore this kwarg harmlessly.
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                **SAMPLING,
            )
            duration = time.time() - t0
            choice = resp.choices[0]
            m = choice.message
            tool_calls = [_tc_to_dict(tc) for tc in (m.tool_calls or [])]
            # vLLM's --reasoning-parser splits <think>…</think> into a separate
            # field on the message. vLLM 0.20.x names it `reasoning`; older
            # versions used `reasoning_content`. Check both. We log it so the
            # trace shows what the model thought before answering.
            reasoning = (
                getattr(m, "reasoning", None)
                or getattr(m, "reasoning_content", None)
                or (m.model_extra or {}).get("reasoning")
                or (m.model_extra or {}).get("reasoning_content")
            )
            log.event(
                "llm_response",
                step=step,
                duration_s=duration,
                reasoning_content=reasoning,
                content=m.content,
                tool_calls=tool_calls,
                usage=resp.usage.model_dump() if resp.usage else None,
                finish_reason=choice.finish_reason,
            )

            assistant_msg = m.model_dump(exclude_none=True)
            msgs.append(assistant_msg)

            if not m.tool_calls:
                final_text = m.content or ""
                log.event("final_answer", content=final_text)
                return final_text, msgs, log

            for tc in m.tool_calls:
                name = tc.function.name
                raw_args = tc.function.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError as e:
                    log.event("tool_error", id=tc.id, name=name, error=f"bad JSON args: {e!r}; raw={raw_args!r}")
                    msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": name,
                            "content": f"[tool error: malformed JSON arguments: {e}]",
                        }
                    )
                    continue

                log.event("tool_call", id=tc.id, name=name, args=args)
                impl = TOOL_IMPLS.get(name)
                t0 = time.time()
                if impl is None:
                    result = f"[unknown tool: {name}]"
                    log.event(
                        "tool_error",
                        id=tc.id,
                        name=name,
                        duration_s=time.time() - t0,
                        error=result,
                    )
                else:
                    try:
                        result = impl(**args)
                        log.event(
                            "tool_result",
                            id=tc.id,
                            name=name,
                            duration_s=time.time() - t0,
                            result=result,
                        )
                    except Exception as e:
                        result = f"[tool error: {e!r}]"
                        log.event(
                            "tool_error",
                            id=tc.id,
                            name=name,
                            duration_s=time.time() - t0,
                            error=repr(e),
                        )

                msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": result,
                    }
                )

        log.event("max_iters_reached", iters=max_iters)
        final_text = "[max iterations reached]"
        return final_text, msgs, log
    finally:
        if close_log:
            log.close(final_text=final_text, n_messages=len(msgs))
