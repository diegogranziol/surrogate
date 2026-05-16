"""Two-stage pipeline.

Stage 1: a tool-using model (Qwen2.5-7B) gathers web evidence.
Stage 2: a larger reasoning model (Nemotron-Super-49B or Qwen3-32B) sees ONLY
         the raw tool outputs (not stage 1's reasoning) and produces its own
         answer with a <think> trace.

Two modes:
- DUAL-ENDPOINT (preferred, no model swap): each stage talks to a different
  vLLM server simultaneously. Set STAGE1_BASE_URL/STAGE1_MODEL and
  STAGE2_BASE_URL/STAGE2_MODEL.
- SINGLE-ENDPOINT (legacy): one server, swap models between stages via
  surrogate.swap.swap_to().
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from surrogate import agent as agent_mod
from surrogate.agent import chat
from surrogate.logger import TraceLogger


STAGE2_SYSTEM = (
    "detailed thinking on\n"
    "You are an expert reviewer. The user asked a question, and a separate "
    "research assistant has already gathered web evidence using tools. You see "
    "the raw tool outputs but not the previous assistant's reasoning or answer. "
    "Think carefully using ONLY the evidence below, then give your best answer "
    "with specific citations (source URLs). If the evidence is insufficient, "
    "say so honestly rather than fabricating details."
)


# Stage-2 sampling matrix: greedy reference + 4 high-temperature samples.
# All samples see the same evidence pack; only the sampling config varies.
STAGE2_SAMPLES: list[dict] = [
    {"temperature": 0.0, "top_p": 1.0, "max_tokens": 4096},   # sample 0: greedy
    {"temperature": 1.0, "top_p": 1.0, "max_tokens": 4096},   # sample 1: hot
    {"temperature": 1.0, "top_p": 1.0, "max_tokens": 4096},   # sample 2: hot
    {"temperature": 1.0, "top_p": 1.0, "max_tokens": 4096},   # sample 3: hot
    {"temperature": 1.0, "top_p": 1.0, "max_tokens": 4096},   # sample 4: hot
]


def _extract_tool_outputs(log_dir: Path) -> list[dict]:
    events = [json.loads(l) for l in (log_dir / "trace.jsonl").read_text().splitlines() if l.strip()]
    by_id, order = {}, []
    for e in events:
        if e["kind"] == "tool_call":
            by_id.setdefault(e["id"], {})["call"] = e
            order.append(e["id"])
        elif e["kind"] in ("tool_result", "tool_error"):
            by_id.setdefault(e["id"], {})["result"] = e
    return [by_id[i] for i in order if "call" in by_id[i]]


def _build_stage2_user_message(question: str, tool_pairs: list[dict]) -> str:
    if not tool_pairs:
        return (
            f"QUESTION: {question}\n\n"
            "EVIDENCE: (no tools were called; nothing to review)\n\n"
            "Answer the question and say clearly that no web evidence was available."
        )
    parts = [f"QUESTION: {question}", "", "EVIDENCE GATHERED FROM TOOL CALLS:"]
    for i, pair in enumerate(tool_pairs, 1):
        call = pair["call"]
        result = pair.get("result", {})
        parts.append("")
        parts.append(f"---- Source {i}: {call['name']}({json.dumps(call.get('args', {}), ensure_ascii=False)}) ----")
        parts.append(str(result.get("result") or result.get("error") or "(no result)"))
    parts.append("")
    parts.append("Now, using ONLY the evidence above, think step by step and provide your best answer to the QUESTION. Cite specific source URLs.")
    return "\n".join(parts)


@dataclass
class StageResult:
    model: str
    answer: str
    reasoning: str | None
    log_dir: Path
    duration_s: float


@dataclass
class TwoStageResult:
    question: str
    stage1: StageResult
    stage2: StageResult
    bundle_dir: Path


def _resolve_endpoint(stage: str) -> tuple[str, str]:
    """Return (base_url, model) for stage='1' or '2'.

    Reads STAGE{1,2}_BASE_URL and STAGE{1,2}_MODEL from env. Falls back to
    SURROGATE_BASE_URL / SURROGATE_MODEL for backward compatibility.
    """
    url = os.environ.get(f"STAGE{stage}_BASE_URL") or os.environ.get("SURROGATE_BASE_URL", "http://localhost:8000/v1")
    model = os.environ.get(f"STAGE{stage}_MODEL")
    if model:
        return url, model
    # legacy single-server defaults
    if stage == "1":
        return url, "qwen2.5-7b"
    return url, "qwen3-32b"


def run_two_stage(
    question: str,
    *,
    stage1_model: str | None = None,
    stage2_model: str | None = None,
    log_root: str = "logs",
) -> TwoStageResult:
    s1_url, s1_default = _resolve_endpoint("1")
    s2_url, s2_default = _resolve_endpoint("2")
    s1_model = stage1_model or s1_default
    s2_model = stage2_model or s2_default

    dual = (s1_url != s2_url)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    bundle = Path(log_root) / f"two-stage-{ts}"
    bundle.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1 -----------------------------------------------------------
    print(f"\n========== STAGE 1: {s1_model} @ {s1_url} ==========")
    if not dual:
        # legacy single-endpoint mode: swap models on the same server
        from surrogate.swap import swap_to
        swap_to(s1_model)

    # Point agent at stage-1 endpoint
    os.environ["SURROGATE_BASE_URL"] = s1_url
    os.environ["SURROGATE_MODEL"] = s1_model
    agent_mod.BASE_URL = s1_url
    agent_mod.MODEL = s1_model

    s1_log = TraceLogger(question, log_root=str(bundle))
    s1_log.dir = s1_log.dir.with_name(s1_log.dir.name + "-stage1")
    s1_log.dir.mkdir(parents=True, exist_ok=True)
    s1_log._jsonl.close(); s1_log._md.close()
    s1_log._jsonl = (s1_log.dir / "trace.jsonl").open("a", encoding="utf-8")
    s1_log._md = (s1_log.dir / "transcript.md").open("a", encoding="utf-8")

    t0 = time.time()
    s1_answer, _msgs, _ = chat(question, log=s1_log)
    s1_dur = time.time() - t0
    s1_log.close()
    print(f"\n--- stage 1 answer ({s1_dur:.1f}s) ---\n{s1_answer}\n")

    tool_pairs = _extract_tool_outputs(s1_log.dir)
    print(f"[stage1] extracted {len(tool_pairs)} tool-call/result pair(s)")

    # ---- Stage 2 -----------------------------------------------------------
    print(f"\n========== STAGE 2: {s2_model} @ {s2_url} ==========")
    if not dual:
        from surrogate.swap import swap_to
        swap_to(s2_model)

    user_msg = _build_stage2_user_message(question, tool_pairs)
    (bundle / "stage2-input.md").write_text(
        f"# Stage 2 input\n\n## system\n```\n{STAGE2_SYSTEM}\n```\n\n## user\n```\n{user_msg}\n```\n"
    )

    s2_log = TraceLogger(question, log_root=str(bundle))
    s2_log.dir = s2_log.dir.with_name(s2_log.dir.name + "-stage2")
    s2_log.dir.mkdir(parents=True, exist_ok=True)
    s2_log._jsonl.close(); s2_log._md.close()
    s2_log._jsonl = (s2_log.dir / "trace.jsonl").open("a", encoding="utf-8")
    s2_log._md = (s2_log.dir / "transcript.md").open("a", encoding="utf-8")

    client = OpenAI(base_url=s2_url, api_key="EMPTY")
    msgs = [{"role": "system", "content": STAGE2_SYSTEM},
            {"role": "user", "content": user_msg}]
    s2_log.event(
        "session_start",
        model=s2_model, base_url=s2_url,
        system=STAGE2_SYSTEM, tools=[],
        sampling={"samples": STAGE2_SAMPLES},
        user_question=question,
        evidence_chars=sum(len(str(p.get("result", {}).get("result") or "")) for p in tool_pairs),
    )

    # Loop over sampling configs: sample 0 = greedy reference; 1..4 = high-T.
    s2_samples: list[dict] = []  # captured (idx, reasoning, content, dur) for return summary
    s2_dur_total = 0.0
    for sample_idx, cfg in enumerate(STAGE2_SAMPLES):
        s2_log.event(
            "llm_request",
            step=sample_idx,
            sample_index=sample_idx,
            messages=msgs,
            tools=[],
            **cfg,
        )
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=s2_model, messages=msgs,
                temperature=cfg["temperature"],
                top_p=cfg["top_p"],
                max_tokens=cfg["max_tokens"],
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
            dur = time.time() - t0
            s2_dur_total += dur
            m = resp.choices[0].message
            reasoning = (
                getattr(m, "reasoning", None)
                or getattr(m, "reasoning_content", None)
                or (m.model_extra or {}).get("reasoning")
                or (m.model_extra or {}).get("reasoning_content")
            )
            content = m.content or ""
            s2_log.event(
                "llm_response",
                step=sample_idx,
                sample_index=sample_idx,
                temperature=cfg["temperature"],
                duration_s=dur,
                reasoning_content=reasoning,
                content=content,
                tool_calls=[],
                usage=resp.usage.model_dump() if resp.usage else None,
                finish_reason=resp.choices[0].finish_reason,
            )
            s2_samples.append({
                "sample_index": sample_idx,
                "temperature": cfg["temperature"],
                "duration_s": dur,
                "reasoning": reasoning,
                "content": content,
            })
            # FULL STACK TRACE rule (see CLAUDE.md): dump the model's entire
            # content (including <think> blocks) to stdout AFTER it has been
            # logged. Do NOT replace this with a preview/summary.
            print(f"  [sample {sample_idx}  T={cfg['temperature']}  {dur:.1f}s  "
                  f"tokens {(resp.usage.prompt_tokens if resp.usage else '?')}+"
                  f"{(resp.usage.completion_tokens if resp.usage else '?')}]")
            print(content or "")
            print()
        except Exception as e:
            dur = time.time() - t0
            s2_dur_total += dur
            s2_log.event(
                "llm_error",
                step=sample_idx,
                sample_index=sample_idx,
                temperature=cfg["temperature"],
                duration_s=dur,
                error=repr(e),
            )
            s2_samples.append({
                "sample_index": sample_idx,
                "temperature": cfg["temperature"],
                "duration_s": dur,
                "reasoning": None,
                "content": f"[error: {e!r}]",
            })
            print(f"  [sample {sample_idx}  T={cfg['temperature']}  {dur:.1f}s]  ERROR {e!r}")

    # Return-value semantics: sample 0 (greedy) is the canonical answer.
    s2_answer = s2_samples[0]["content"] if s2_samples else ""
    reasoning = s2_samples[0]["reasoning"] if s2_samples else None
    s2_dur = s2_dur_total
    s2_log.event("final_answer", content=s2_answer, source_sample=0)
    s2_log.close()

    print(f"\n--- stage 2 total time across {len(s2_samples)} samples: {s2_dur:.1f}s ---")

    (bundle / "compare.md").write_text(
        f"# Two-stage comparison\n\n**Question:** {question}\n\n"
        f"## Stage 1 — {s1_model} ({s1_dur:.1f}s)\n\n{s1_answer}\n\n"
        f"## Stage 2 — {s2_model} ({s2_dur:.1f}s)\n\n"
        f"### Thinking\n\n```\n{reasoning or '(none)'}\n```\n\n"
        f"### Answer\n\n{s2_answer}\n"
    )
    print(f"[bundle] {bundle}")

    return TwoStageResult(
        question=question,
        stage1=StageResult(s1_model, s1_answer, None, s1_log.dir, s1_dur),
        stage2=StageResult(s2_model, s2_answer, reasoning, s2_log.dir, s2_dur),
        bundle_dir=bundle,
    )
