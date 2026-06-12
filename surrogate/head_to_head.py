"""head_to_head — surrogate vs frontier model, each using their own tools.

NO shared evidence. Each side does its own retrieval (surrogate via its
7-tool ReAct loop; Claude via Anthropic's server-side web_search). The metric
is top-N soft-match overlap between the two final ranked lists.

This complements (does not replace) `backtest.py`:
  - backtest.py        : surrogate vs GLM-on-same-evidence  -> reasoning fidelity
  - head_to_head.py    : surrogate vs Claude with own tools  -> output similarity

The new headline number per the 2026-06-01 framing shift (Diego's reframing).

Per CLAUDE.md prime directive: every byte (surrogate thinking, surrogate tool
trace, Claude thinking blocks, Claude web_search calls, full answers) is
preserved verbatim in the storage layer.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import date, datetime
from pathlib import Path

from surrogate.loop import run as loop_run
from surrogate.loop_tools import default_tools
from surrogate.frontier_claude import ask_claude
from surrogate.frontier_openai import ask_openai
from surrogate.reference import ask_reference
from surrogate.backtest import (
    _extract_client,
    _clean_ranked,
    _concat_thinking,
    _SOFT_MATCH_SYSTEM,
)


# Registry of frontier comparators. The harness picks one via the `frontier`
# argument to run_one_h2h(). Both functions take (question, mode=...) and
# return the same dict shape (see frontier_claude.ask_claude docstring).
FRONTIERS = {
    "claude": ask_claude,
    "openai": ask_openai,
}

STORE_DIR = Path("backtests")
H2H_JSONL = STORE_DIR / "h2h-store.jsonl"


# ---- question shape inference ----------------------------------------------
# Auto-detect requested top-N and the "mode" from the wording of the question.
# - Explicit "top 10 / top-5 / top 3" -> structured mode, k = N
# - Otherwise                          -> natural mode,    k = 5 (sensible default)
#
# Mode is passed to Claude's system prompt so the frontier answers in the same
# shape; we don't force the surrogate (its stop_and_answer schema now allows
# 1..10 picks, so the model can match the question naturally).

_TOPN_RX = re.compile(r"\btop[\s\-_]?(\d{1,2})\b", re.I)


def infer_question_shape(question: str, *, default_natural_k: int = 5) -> tuple[int, str]:
    """Return (k, mode). mode in {'structured', 'natural'}."""
    m = _TOPN_RX.search(question)
    if m:
        n = int(m.group(1))
        n = max(1, min(n, 10))
        return n, "structured"
    return default_natural_k, "natural"


# ---- top-N extractor (free: local qwen3-8b, same backend as Stage 1) -------

_EXTRACT_SYS_TOPN = """You extract a clean ranked top-N list from a free-text
purchase-intent answer. Return ONLY compact JSON, no prose, no markdown:
{"ranked": ["<item 1>", "<item 2>", ...]}

Rules:
- Up to N entries, in the order the answer ranks them.
- Each entry = entity name ONLY (product / restaurant / place / book / etc.).
  No description, no review count, no URL, no markdown formatting.
- Strip ranking prefixes ("1.", "#1", "Top 1", etc.).
- If the answer has no clear ranked list, return {"ranked": []}."""


def extract_pick_topN(answer: str, *, k: int = 10) -> dict:
    """Extract up to k ranked picks from a free-text answer."""
    if not answer or not answer.strip():
        return {"ranked": [], "_raw": "", "_ok": False}
    client, model = _extract_client()
    sys = _EXTRACT_SYS_TOPN.replace("top-N", f"top-{k}").replace(
        "Up to N entries", f"Up to {k} entries"
    )
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": answer[:20000]},
            ],
            temperature=0.0,
            max_tokens=800,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = r.choices[0].message.content or ""
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {}
        ranked = _clean_ranked(data.get("ranked") or [], k=k)
        return {"ranked": ranked, "_raw": raw, "_ok": True}
    except Exception as e:
        return {"ranked": [], "_raw": f"[extract error: {e!r}]", "_ok": False}


# ---- soft-match overlap at arbitrary N --------------------------------------

def _judge_via_claude(a: list[str], b: list[str]) -> dict:
    """Fallback judge using Claude (used when GLM is out of balance / fails).
    Uses Haiku for cost — judge task is simple."""
    import os
    from anthropic import Anthropic
    client = Anthropic(max_retries=3)
    model = os.environ.get("JUDGE_CLAUDE_MODEL", "claude-haiku-4-5-20251001")
    user = json.dumps({"A": a, "B": b}, ensure_ascii=False)
    resp = client.messages.create(
        model=model,
        max_tokens=2000,
        system=_SOFT_MATCH_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    return {"answer": text, "model": resp.model}


def soft_match_topN(a_list: list[str], b_list: list[str], *, k: int = 10) -> dict:
    """Soft-match overlap with brand-level rules (see backtest._SOFT_MATCH_SYSTEM).

    Judge chain: GLM (z.ai) → Claude Haiku → normalized-equality fallback.
    """
    a = _clean_ranked(a_list, k=k)
    b = _clean_ranked(b_list, k=k)
    if not a or not b:
        return {"overlap": 0, "matched_pairs": [], "a": a, "b": b,
                "judge_raw": "", "_ok": False, "_fallback": False, "_judge": "none"}

    user = json.dumps({"A": a, "B": b}, ensure_ascii=False)

    # 1. GLM (cheapest when available).
    try:
        r = ask_reference(question=user, system=_SOFT_MATCH_SYSTEM,
                          max_tokens=2000, thinking=False)
        raw = r["answer"] or ""
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {}
        pairs = data.get("matched_pairs") or []
        overlap = int(data.get("overlap", 0))
        overlap = max(0, min(overlap, len(a)))
        return {"overlap": overlap, "matched_pairs": pairs, "a": a, "b": b,
                "judge_raw": raw, "_ok": True, "_fallback": False, "_judge": "glm"}
    except Exception as e_glm:
        glm_err = e_glm

    # 2. Claude Haiku — handles JSON cleanly and we already have the key.
    try:
        r = _judge_via_claude(a, b)
        raw = r["answer"] or ""
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {}
        pairs = data.get("matched_pairs") or []
        overlap = int(data.get("overlap", 0))
        overlap = max(0, min(overlap, len(a)))
        return {"overlap": overlap, "matched_pairs": pairs, "a": a, "b": b,
                "judge_raw": raw, "_ok": True, "_fallback": False,
                "_judge": f"claude:{r.get('model','')}"}
    except Exception as e_claude:
        claude_err = e_claude

    # 3. Normalized substring fallback.
    def n(s):
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", "", s.lower())).strip()
    nb = [n(x) for x in b]
    pairs = []
    for x in a:
        nx = n(x)
        for j, y in enumerate(b):
            if nx == nb[j] or nx in nb[j] or nb[j] in nx:
                pairs.append([x, y, "fallback: normalized match"])
                break
    return {"overlap": len(pairs), "matched_pairs": pairs, "a": a, "b": b,
            "judge_raw": f"[glm error: {glm_err!r} | claude error: {claude_err!r}]",
            "_ok": False, "_fallback": True, "_judge": "regex"}


# ---- one question ----------------------------------------------------------

def run_one_h2h(
    question: str,
    *,
    k: int | None = None,
    mode: str | None = None,
    frontier: str = "claude",
    log_root: str = "logs",
) -> dict:
    """Run the head-to-head for ONE question. Returns a fully-loaded entry
    suitable for append_h2h() / render_h2h_section().

    If k or mode are None, infers them from the question wording via
    `infer_question_shape`.

    `frontier` selects which comparator to use ("claude" or "openai" — see
    FRONTIERS registry).
    """
    inferred_k, inferred_mode = infer_question_shape(question)
    if k is None:
        k = inferred_k
    if mode is None:
        mode = inferred_mode

    if frontier not in FRONTIERS:
        raise ValueError(f"Unknown frontier {frontier!r}. Options: {list(FRONTIERS)}")
    ask_frontier = FRONTIERS[frontier]

    t0 = time.time()

    # 1. surrogate (own tools, full ReAct loop)
    sur = loop_run(question, tools=default_tools(), log_root=log_root)
    sur_answer = sur.final_answer or ""
    sur_pick = extract_pick_topN(sur_answer, k=k)

    # 2. frontier (own server-side web_search + extended thinking/reasoning)
    fr = ask_frontier(question, mode=mode)
    fr_pick = extract_pick_topN(fr["answer"], k=k)

    # 3. soft-match
    match = soft_match_topN(sur_pick["ranked"], fr_pick["ranked"], k=k)

    elapsed = time.time() - t0
    return {
        "ts": datetime.utcnow().isoformat() + "Z",
        "date": date.today().isoformat(),
        "question": question,
        "k": k,
        "mode": mode,
        "frontier": frontier,
        "surrogate": {
            "model": os.environ.get("STAGE2_MODEL", "qwen3-8b"),
            "answer": sur_answer,
            "thinking": _concat_thinking(sur.messages),
            "ranked": sur_pick["ranked"],
            "ranked_raw": sur_pick["_raw"],
            "termination": sur.termination,
            "steps": sur.steps,
            "duration_s": sur.duration_s,
            "bundle": str(sur.bundle_dir) if sur.bundle_dir else None,
        },
        "frontier": {
            "model": fr["model"],
            "answer": fr["answer"],
            "thinking": fr["thinking"],
            "ranked": fr_pick["ranked"],
            "ranked_raw": fr_pick["_raw"],
            "tool_calls": fr["tool_calls"],
            "blocks_raw": fr["blocks_raw"],
            "usage": fr["usage"],
            "stop_reason": fr["stop_reason"],
        },
        "match": match,
        "elapsed_s": elapsed,
    }


# ---- storage ---------------------------------------------------------------

def append_h2h(entry: dict) -> None:
    STORE_DIR.mkdir(exist_ok=True, parents=True)
    with H2H_JSONL.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---- rendering helpers -----------------------------------------------------

# Emoji-tagged tool name → readable header for the report.
_TOOL_ICON = {
    "search": "🔎",
    "fetch_url": "📄",
    "extract_entity": "🏷️",
    "verify_fact": "✓",
    "check_missing_fields": "📋",
    "think": "💭",
    "stop_and_answer": "✋",
}


def render_surrogate_trace(bundle_dir: str | Path | None) -> str:
    """Walk the bundle's trace.jsonl and produce a step-by-step tool trace
    rendered as readable markdown. Includes every tool call + result verbatim
    per CLAUDE.md — no truncation."""
    if not bundle_dir:
        return "_(no bundle dir — surrogate produced no trace)_"
    p = Path(bundle_dir) / "trace.jsonl"
    if not p.exists():
        return f"_(no trace.jsonl in {bundle_dir})_"

    events = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    # Group by step where present; events without a step (session_start etc.)
    # are emitted at the top.
    pre: list[dict] = []
    by_step: dict[int, list[dict]] = {}
    for e in events:
        step = e.get("step")
        if step is None:
            pre.append(e)
        else:
            by_step.setdefault(step, []).append(e)

    out: list[str] = []
    # Optional session_start / system info up front.
    for e in pre:
        kind = e.get("kind")
        if kind == "session_start":
            q = e.get("user_question") or e.get("question") or ""
            out.append(f"_Session opened with question:_ {q!r}")
            out.append("")

    for step in sorted(by_step):
        evs = by_step[step]
        call = next((x for x in evs if x.get("kind") == "tool_call"), None)
        result = next((x for x in evs if x.get("kind") in ("tool_result", "tool_error")), None)
        if not call:
            continue
        name = call.get("name", "(unknown)")
        icon = _TOOL_ICON.get(name, "•")
        args = call.get("args", {})
        out.append(f"#### Step {step} — {icon} `{name}`")
        out.append("")
        out.append("**Args:**")
        out.append("```json")
        out.append(json.dumps(args, indent=2, ensure_ascii=False))
        out.append("```")
        if result:
            kind = result.get("kind")
            body = result.get("result") if kind == "tool_result" else result.get("error")
            body_str = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False, indent=2)
            out.append("")
            out.append(f"**Result ({kind}):**")
            out.append("```")
            out.append(body_str or "(empty)")
            out.append("```")
        out.append("")
    return "\n".join(out)


def render_picks_compare(a: list[str], b: list[str], pairs: list[list]) -> str:
    """Side-by-side picks table. Matches (from soft-match judge) flagged with ✓."""
    matched_a = {str(p[0]).lower() for p in pairs if len(p) >= 1}
    matched_b = {str(p[1]).lower() for p in pairs if len(p) >= 2}
    n = max(len(a), len(b))
    lines = ["| # | surrogate | frontier |", "|---|-----------|----------|"]
    for i in range(n):
        ai = a[i] if i < len(a) else ""
        bi = b[i] if i < len(b) else ""
        ai_disp = f"**{ai}** ✓" if ai and ai.lower() in matched_a else ai
        bi_disp = f"**{bi}** ✓" if bi and bi.lower() in matched_b else bi
        lines.append(f"| {i+1} | {ai_disp} | {bi_disp} |")
    return "\n".join(lines)


# ---- rendering (verbatim — no curation, per CLAUDE.md) ---------------------

def render_h2h_section(e: dict) -> str:
    s = e["surrogate"]
    f = e["frontier"]
    m = e["match"]
    out: list[str] = []
    out.append(f"## {e['question']}")
    out.append("")
    out.append(
        f"_mode={e.get('mode','?')}, k={e['k']}, "
        f"overlap={m['overlap']}/{len(m['a'])}, elapsed={e['elapsed_s']:.1f}s_"
    )
    out.append("")

    out.append(f"### 🟢 Surrogate — {s['model']} "
               f"({s['steps']} steps, {s['duration_s']:.1f}s, term={s['termination']})")
    out.append("")
    out.append(f"_Bundle: `{s.get('bundle','?')}`_")
    out.append("")
    # Side-by-side picks first — easy at-a-glance comparison.
    out.append("**Picks (side-by-side, ✓ = soft-matched):**")
    out.append("")
    out.append(render_picks_compare(s["ranked"], f["ranked"], m.get("matched_pairs") or []))
    out.append("")
    # Surrogate's tool trace — the differentiator.
    out.append("**Tool trace (verbatim, every step):**")
    out.append("")
    out.append(render_surrogate_trace(s.get("bundle")))
    out.append("")
    if s["thinking"]:
        out.append("**Surrogate thinking (concatenated `<think>` blocks, verbatim):**")
        out.append("")
        out.append("```")
        out.append(s["thinking"])
        out.append("```")
        out.append("")
    out.append("**Surrogate final answer:**")
    out.append("")
    out.append(s["answer"] or "_(no final answer)_")
    out.append("")

    out.append(f"### 🔵 Frontier — {f['model']} (stop={f['stop_reason']}, "
               f"usage={f['usage']})")
    out.append("")
    if f["thinking"]:
        out.append("**Frontier thinking (verbatim):**")
        out.append("")
        out.append("```")
        out.append(f["thinking"])
        out.append("```")
        out.append("")
    out.append("**Frontier answer:**")
    out.append("")
    out.append(f["answer"] or "_(no final answer)_")
    out.append("")
    out.append(f"_Frontier ranked (extracted): {f['ranked']}_")
    out.append("")
    if f["tool_calls"]:
        out.append("**Frontier tool calls (verbatim):**")
        out.append("")
        out.append("```json")
        out.append(json.dumps(f["tool_calls"], indent=2, ensure_ascii=False))
        out.append("```")
        out.append("")

    out.append(f"### 🎯 Soft-match (k={e['k']})")
    out.append("")
    out.append(f"- A (surrogate): `{m['a']}`")
    out.append(f"- B (frontier):  `{m['b']}`")
    out.append(f"- **Overlap: {m['overlap']}/{len(m['a'])}**"
               + ("  _(fallback judge)_" if m.get("_fallback") else ""))
    if m["matched_pairs"]:
        out.append("- Matched pairs:")
        for p in m["matched_pairs"]:
            reason = p[2] if len(p) > 2 else ""
            out.append(f"  - `{p[0]}` ↔ `{p[1]}` — {reason}")
    out.append("")
    return "\n".join(out)


# ---- brand-visibility analytics (used by the executive summary) ------------

def _contains_brand(items: list[str], brand: str) -> str | None:
    """Return the matching item (verbatim) if `brand` appears as a substring in
    any of `items`, case-insensitive. None otherwise."""
    needle = brand.lower()
    for x in items or []:
        if needle in str(x).lower():
            return x
    return None


# Heuristic: which questions are about supplements / Avea's playing field.
# Only used to decide whether to add the "brand visibility" subsection.
_SUPPLEMENT_KEYWORDS = (
    "supplement", "nmn", "nad+", "spermidine", "collagen", "longevity",
    "healthy aging", "anti-aging", "vitamin",
)


def _is_brand_question(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in _SUPPLEMENT_KEYWORDS)


def write_h2h_md(entries: list[dict], path: Path, *, brand: str = "Avea") -> None:
    """Write a presentable head-to-head report.

    Structure:
      1. Executive summary (mode-split overlap, per-Q table)
      2. Brand visibility (if any supplement questions present) — for the pitch
      3. Per-question detail (tool trace, thinking, answer, soft-match)
    """
    out: list[str] = []
    out.append(f"# Head-to-head — Surrogate vs Claude ({len(entries)} questions)")
    out.append("")
    out.append("_Each side answers independently using its own tools "
               "(Surrogate: 7-tool ReAct loop + Tavily; Frontier: Claude with "
               "built-in web_search). Comparison is on the final ranked answer "
               "lists — top-N soft-match overlap, judged by GLM-4.6._")
    out.append("")

    if not entries:
        out.append("_(no entries)_")
        path.write_text("\n".join(out))
        return

    # ---- 1. Executive summary -------------------------------------------------
    out.append("## 📊 Executive summary")
    out.append("")
    by_mode: dict[str, list[dict]] = {}
    for e in entries:
        by_mode.setdefault(e.get("mode", "structured"), []).append(e)
    for mode, group in by_mode.items():
        overlaps = [e["match"]["overlap"] for e in group]
        ks = [len(e["match"]["a"]) or e["k"] for e in group]
        mean_overlap = sum(overlaps) / len(overlaps)
        mean_k = sum(ks) / len(ks)
        out.append(
            f"- **{mode}** ({len(group)} q): mean overlap "
            f"**{mean_overlap:.2f} / {mean_k:.1f}**  "
            f"_({mean_overlap / max(mean_k, 1):.0%} of the surrogate's picks "
            f"matched the frontier)_"
        )
    out.append("")
    out.append("**Per-question overlap:**")
    out.append("")
    out.append("| # | mode | k | overlap | question |")
    out.append("|---|------|---|---------|----------|")
    for i, e in enumerate(entries, 1):
        mm = e["match"]
        out.append(
            f"| {i} | {e.get('mode','?')} | {e['k']} | "
            f"**{mm['overlap']}/{len(mm['a'])}** | {e['question']} |"
        )
    out.append("")

    # ---- 2. Brand visibility section -----------------------------------------
    brand_qs = [e for e in entries if _is_brand_question(e["question"])]
    if brand_qs:
        out.append(f"## 🎯 {brand} visibility")
        out.append("")
        out.append(
            f"_For each supplement-relevant question, did **{brand}** appear in "
            f"the surrogate's ranked list? In Claude's? This is the GEO signal — "
            f"\"how does the frontier perceive my brand right now\"._"
        )
        out.append("")
        out.append("| # | Question | In Surrogate | In Claude | Verdict |")
        out.append("|---|----------|--------------|-----------|---------|")
        for i, e in enumerate(brand_qs, 1):
            s_hit = _contains_brand(e["surrogate"]["ranked"], brand)
            f_hit = _contains_brand(e["frontier"]["ranked"], brand)
            s_cell = f"✅ `{s_hit}`" if s_hit else "❌ —"
            f_cell = f"✅ `{f_hit}`" if f_hit else "❌ —"
            if s_hit and f_hit:
                verdict = "🟢 Both — credible proxy + good GEO"
            elif f_hit and not s_hit:
                verdict = "🟡 Frontier yes, surrogate no — tool/search gap"
            elif s_hit and not f_hit:
                verdict = "🟠 Surrogate yes, frontier no — invisible to AI mainstream"
            else:
                verdict = "🔴 Neither — invisible to both → GEO opportunity"
            out.append(f"| {i} | {e['question']} | {s_cell} | {f_cell} | {verdict} |")
        out.append("")

    # ---- 3. Per-question detail ----------------------------------------------
    out.append("---")
    out.append("")
    out.append("## 📋 Per-question detail")
    out.append("")
    out.append("_(Each section: side-by-side picks, full surrogate tool trace, "
               "thinking blocks, final answers from both sides, soft-match judgment.)_")
    out.append("")
    for e in entries:
        out.append(render_h2h_section(e))
        out.append("---")
        out.append("")
    path.write_text("\n".join(out))
