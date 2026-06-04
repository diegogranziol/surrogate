"""Single-stage ReAct loop modeled on Tongyi DeepResearch's `inference/react_agent.py`.

Why this loop, not our old two-stage agent:
- Single LLM context — the model plans, calls tools, reads observations, and
  emits the final answer in one continuous trajectory. Visible reasoning lives
  *between* tool calls as raw `<think>...</think>` text.
- Hermes-format tool calls (XML-wrapped JSON), per Qwen3's recommended format —
  ReAct stop-word templates collide with thinking tokens on Qwen3.
- Three budgets enforced: max_steps, max_wall_seconds, soft max_chars (token
  surrogate). Tongyi proves these are sufficient at SOTA.
- Stop sequences `<tool_response>` / `\n<tool_response>` prevent the model
  hallucinating observation content — same as Tongyi.
- Reasoning preserved verbatim: if vLLM's `--reasoning-parser` splits
  `<think>...</think>` into a separate field, we splice it back into `content`
  so the raw bytes are in the trace exactly as Diego wants.

Tool surface:
    Each tool is a `Tool(name, description, parameters, call)` object.
    `parameters` is OpenAPI-style JSON schema; rendered into the system prompt's
    `<tools></tools>` block. `call(args: dict) -> str` returns the observation
    text that gets wrapped in `<tool_response>...</tool_response>` and re-fed
    as a user message.

Output:
    LoopResult with the trajectory, the final answer (if any), termination
    reason, and the bundle path (logs/<ts>-<slug>/) carrying the verbatim
    trace.jsonl + transcript.md.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

try:
    import json5
    _JSON5 = True
except ImportError:
    _JSON5 = False

from openai import OpenAI

from surrogate.logger import TraceLogger


# ---------------------------------------------------------------------------
# Tool registry primitive — a function + its OpenAPI-style schema.
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict                # OpenAPI-style JSON schema
    call: Callable[[dict], str]     # (args_dict) -> observation_str

    def spec(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name,
                             "description": self.description,
                             "parameters": self.parameters}}


# ---------------------------------------------------------------------------
# System prompt — adapted from Tongyi's, generalised for purchase-intent +
# our eventual 7-tool spec. The tool block is built at runtime from whatever
# tools the caller registers.
# ---------------------------------------------------------------------------

BASE_SYSTEM_PROMPT = """You are a research assistant that answers purchase-intent questions \
(\"best X in Y for Z\") through grounded, multi-source investigation. Your reasoning is \
visible to the user inside <think>...</think> blocks — show your work.

## Process

1. **Plan first.** Begin with a <think>...</think> block containing a 2–3 step plan: \
what items will you search for, what constraints will you verify (rating, price, location, \
key features), and what counts as "done".

2. **Search & gather.** Use `search` with 2–4 complementary queries in ONE batched call \
(fan out — don't serial-search). Then call `extract_entity` on the most promising URLs \
for clean structured fields (name, rating, review_count, price, address).

3. **Verify before citing.** Before including any specific factual claim (a rating, a \
price, a review count, a venue name), call `verify_fact(claim, evidence_url)` on it. \
Cite only what verifies.

4. **Reflect & check completeness.** After every 2 `search`/`extract_entity` calls, call \
`think` to verbalize what you've learned, what's still uncertain, and the next best action. \
Before finalising, call `check_missing_fields` on each top candidate; if required fields \
are missing, run ONE targeted search/extract for those fields, then re-check.

5. **Finalise with `stop_and_answer`** — structured payload with ranked `top_picks` (each \
with name + one-line evidence-grounded reasoning + rating + price + source_url), a summary \
paragraph, and a citations list.

**Count rule — STRICT, non-negotiable:**
- If the question specifies a number ("top 10", "top 5", "best 3"), `top_picks` MUST \
contain EXACTLY that many entries. Never under-shoot. If you have N-1 verified candidates, \
run ONE more `search` round and find one more — do NOT stop early.
- "what X should I use" / "which X is best" → 1 confident pick or 2–3 alternatives is fine.
- "best X" with no number → 3–10 picks, your judgment.

## Hard limits

- Simple queries: **2–3 `search` calls maximum**.
- Complex queries (including any top-10): **up to 6 `search` calls**.
- **Do NOT call `stop_and_answer` until you have the EXACT count required by the question** \
(see Process step 5). A "top 10" question with only 7 verified candidates is NOT done — \
keep searching. A "best X" with no number is done when you have 3+ verified.
- **Force-stop ceiling** (call `stop_and_answer` with what you have, even if short): when \
ALL of:
  • Last 2 `search` calls returned no new candidates.
  • Total tool calls ≥ 28.

## Output rules

- `stop_and_answer.top_picks` contains EXACTLY the number specified by the question (see \
Process step 5 — for "top N", N picks, no fewer). Ranked best-first.
- Each `reasoning` field is concrete and evidence-grounded — "rated 4.8 with 175 reviews on \
TripAdvisor, praised for wood-fired pizza" — NOT vague praise like "highly recommended".
- Use inline markdown citations `([title](url))` in the `summary` text.
- The `<answer>` tag is a fallback only; **prefer `stop_and_answer`** so the payload is \
validated against the schema."""

_TOOL_INSTRUCTIONS = """
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_block}
</tools>

For each function call, return a json object with function name and arguments within \
<tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

After each tool call, you will receive the result as a user message wrapped in \
<tool_response>...</tool_response>. Read it carefully, write your reasoning inside \
<think>...</think>, then decide whether to call another tool or to finalize with \
<answer>...</answer>.

Current date: {date}
"""


def build_system_prompt(base: str, tools: list[Tool], *, date: str | None = None) -> str:
    """Render the full system prompt, embedding tool schemas Hermes-style."""
    schemas = "\n".join(json.dumps(t.spec(), ensure_ascii=False) for t in tools)
    suffix = _TOOL_INSTRUCTIONS.format(
        tools_block=schemas,
        date=date or datetime.now().strftime("%Y-%m-%d"),
    )
    return base.rstrip() + "\n" + suffix


# ---------------------------------------------------------------------------
# Stop sequences — model must not generate the observation tag itself.
# ---------------------------------------------------------------------------

STOP_SEQUENCES = ["\n<tool_response>", "<tool_response>"]


# ---------------------------------------------------------------------------
# Parsing helpers.
# ---------------------------------------------------------------------------

_RE_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_RE_ANSWER    = re.compile(r"<answer>\s*(.*?)\s*</answer>",       re.DOTALL)


def _parse_json_lenient(text: str) -> dict | None:
    """LLMs sometimes emit single-quoted / trailing-comma / py-style JSON.
    Try strict first, then json5, then ast.literal_eval as last resort."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    if _JSON5:
        try:
            return json5.loads(text)
        except Exception:
            pass
    try:
        import ast
        v = ast.literal_eval(text)
        if isinstance(v, dict):
            return v
    except Exception:
        pass
    return None


def extract_tool_call(content: str) -> dict | None:
    """Pull the first <tool_call>{...}</tool_call> block; return parsed dict
    with .name + .arguments, or None if absent/malformed."""
    m = _RE_TOOL_CALL.search(content)
    if not m:
        return None
    return _parse_json_lenient(m.group(1))


def extract_answer(content: str) -> str | None:
    m = _RE_ANSWER.search(content)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Loop result.
# ---------------------------------------------------------------------------

@dataclass
class LoopResult:
    question: str
    final_answer: str | None
    termination: str                        # "answer" | "max_steps" | "wall_clock" | "ctx_overflow" | "error"
    messages: list[dict] = field(default_factory=list)
    steps: int = 0
    duration_s: float = 0.0
    bundle_dir: Path | None = None


# ---------------------------------------------------------------------------
# The loop.
# ---------------------------------------------------------------------------

def run(
    question: str,
    *,
    tools: list[Tool],
    base_system_prompt: str = BASE_SYSTEM_PROMPT,
    base_url: str | None = None,
    model: str | None = None,
    max_steps: int = 48,
    max_wall_seconds: int = 600,
    max_chars: int = 350_000,        # ~85k tokens at ~4 chars/token — soft cap
    temperature: float = 0.6,
    top_p: float = 0.95,
    sampling_max_tokens: int = 4096,
    enable_thinking: bool = True,
    log_root: str = "logs",
) -> LoopResult:
    """Run the Tongyi-shaped ReAct loop. Blocking; returns when the model
    emits `<answer>...</answer>` or a budget fires."""

    base_url = base_url or os.environ.get("STAGE2_BASE_URL", "http://localhost:8000/v1")
    model    = model    or os.environ.get("STAGE2_MODEL",    "qwen3-8b")
    client = OpenAI(base_url=base_url, api_key=os.environ.get("SURROGATE_API_KEY", "EMPTY"))

    tool_map = {t.name: t for t in tools}
    system_prompt = build_system_prompt(base_system_prompt, tools)
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question},
    ]

    log = TraceLogger(question, log_root=log_root)
    log.event(
        "session_start",
        model=model, base_url=base_url, system=system_prompt,
        tools=[t.spec() for t in tools],
        sampling={"temperature": temperature, "top_p": top_p,
                  "max_tokens": sampling_max_tokens,
                  "stop": STOP_SEQUENCES, "enable_thinking": enable_thinking},
        budgets={"max_steps": max_steps, "max_wall_seconds": max_wall_seconds,
                 "max_chars": max_chars},
        user_question=question,
    )

    t0 = time.time()
    termination = "max_steps"
    final_answer: str | None = None

    for step in range(max_steps):
        # Budget checks before each LLM call
        if time.time() - t0 > max_wall_seconds:
            termination = "wall_clock"
            break
        cur_chars = sum(len(m.get("content") or "") for m in messages)
        if cur_chars > max_chars:
            # Force final-answer attempt — same recovery as Tongyi
            messages.append({
                "role": "user",
                "content": ("You have reached the context limit. Stop calling tools "
                            "and provide your best answer now inside <answer>...</answer>."),
            })
            termination = "ctx_overflow"

        log.event("llm_request", step=step, messages=messages,
                  temperature=temperature, top_p=top_p,
                  max_tokens=sampling_max_tokens, stop=STOP_SEQUENCES)

        try:
            t_call = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=sampling_max_tokens,
                stop=STOP_SEQUENCES,
                extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
            )
        except Exception as e:
            log.event("llm_error", step=step, error=repr(e))
            termination = "error"
            break

        m = resp.choices[0].message
        content = m.content or ""
        # vLLM's qwen3 reasoning parser pulls <think>...</think> into a separate
        # field. Splice it back so the trace and the next-turn prompt contain
        # the literal raw bytes (Diego: "output the entire thinking tags").
        reasoning = (
            getattr(m, "reasoning", None)
            or getattr(m, "reasoning_content", None)
            or (m.model_extra or {}).get("reasoning")
            or (m.model_extra or {}).get("reasoning_content")
        )
        if reasoning and "<think>" not in content:
            content = f"<think>\n{reasoning}\n</think>\n{content}".rstrip()

        # If the model still managed to emit a fake <tool_response> after our
        # stop sequence, defensively chop everything from there onwards.
        if "<tool_response>" in content:
            content = content.split("<tool_response>", 1)[0].rstrip()

        log.event(
            "llm_response", step=step, duration_s=round(time.time() - t_call, 2),
            content=content, reasoning_content=reasoning,
            usage=resp.usage.model_dump() if resp.usage else None,
            finish_reason=resp.choices[0].finish_reason,
        )
        messages.append({"role": "assistant", "content": content})

        # Final answer wins, even if a tool_call is also present.
        ans = extract_answer(content)
        if ans is not None:
            final_answer = ans
            termination = "answer"
            log.event("final_answer", step=step, content=ans)
            break

        # If ctx_overflow has been raised, give up after the forced attempt.
        if termination == "ctx_overflow":
            break

        # Otherwise try to dispatch a tool call.
        tc = extract_tool_call(content)
        if tc is None:
            # Model produced neither answer nor tool call — nudge and continue.
            log.event("no_action", step=step,
                      note="no <answer> and no <tool_call> in response")
            messages.append({
                "role": "user",
                "content": ("Please either call a tool via "
                            "`<tool_call>{...}</tool_call>` or provide your final "
                            "answer inside `<answer>...</answer>`."),
            })
            continue

        tool_name = tc.get("name", "")
        tool_args = tc.get("arguments", {}) or {}
        log.event("tool_call", step=step, name=tool_name, args=tool_args, id=f"step{step}")

        # ---- stop_and_answer is the contractual terminator ---------------
        if tool_name == "stop_and_answer":
            from surrogate.tools.stop import validate, render_markdown
            payload, err = validate(tool_args)
            if err is None:
                rendered = render_markdown(payload)
                final_answer = rendered
                termination = "answer"
                log.event("tool_result", step=step, name=tool_name,
                          result=rendered, validated_payload=payload.model_dump())
                log.event("final_answer", step=step, content=rendered)
                break
            # Validation failed → return the error as observation and let
            # the model try once more.
            log.event("tool_error", step=step, name=tool_name, error=err)
            messages.append({
                "role": "user",
                "content": f"<tool_response>\n{err}\n</tool_response>",
            })
            continue
        # -------------------------------------------------------------------

        tool = tool_map.get(tool_name)
        if tool is None:
            obs = (f"Error: tool '{tool_name}' not found. "
                   f"Available tools: {sorted(tool_map)}.")
            log.event("tool_error", step=step, name=tool_name, error=obs)
        else:
            try:
                t_tool = time.time()
                obs = tool.call(tool_args)
                log.event("tool_result", step=step, name=tool_name,
                          duration_s=round(time.time() - t_tool, 2), result=obs)
            except Exception as e:
                obs = f"Error calling {tool_name}: {e!r}"
                log.event("tool_error", step=step, name=tool_name, error=obs)

        messages.append({
            "role": "user",
            "content": f"<tool_response>\n{obs}\n</tool_response>",
        })

    duration_s = round(time.time() - t0, 2)
    if final_answer is None and termination == "max_steps":
        log.event("max_steps_reached", steps=max_steps)
    log.close(final_text=final_answer or f"[{termination}]", n_messages=len(messages))

    return LoopResult(
        question=question,
        final_answer=final_answer,
        termination=termination,
        messages=messages,
        steps=step + 1 if "step" in locals() else 0,
        duration_s=duration_s,
        bundle_dir=log.dir,
    )
