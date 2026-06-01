"""Phase 2 — backtest harness.

For each question:
  1. Run the self-hosted surrogate two-stage pipeline (Qwen3-8B) -> answer +
     thinking + the evidence Stage 1 gathered.
  2. Call the GLM reference TWICE:
       - "evidence": fed the EXACT stage-2 user message the surrogate saw
         (true apples-to-apples — isolates model behaviour from evidence).
       - "bare": just the question (documents the memory-vs-web gap).
  3. Extract each answer's top-3 ranked picks (free local qwen3-8b call).
  4. Score with the TOP-3 SOFT-MATCH metric:
       overlap = how many of surrogate's top-3 soft-match ANY of reference's
       top-3, where soft-match is judged by GLM with these rules baked in:
         - Same product line + adjacent version  => MATCH
           (Saucony Endorphin Speed 3 ~= Speed 4)
         - Same product + variant tier           => MATCH
           (iPhone 16 Pro ~= iPhone 16 Pro Max; Sette ~= Sette Restaurant)
         - Different major generation/brand      => NO MATCH
           (iPhone 15 != iPhone 16; Burj Al Arab != Mandarin Oriental)
       Score is 0..3. Thinking is captured and dumped verbatim per CLAUDE.md
       but NOT used in the score — most production assistant APIs don't expose
       reasoning content, so a thinking-level comparison would be one-sided.
  5. Append entries to backtests/store.jsonl (date-stamped) + write a verbatim
     side-by-side backtests/run-<ts>.md.

NDCG / strict top-1 are parked — fine-grained ranking only matters once top-3
overlap is well-tracked.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import date, datetime
from pathlib import Path

from openai import OpenAI

from surrogate.two_stage import STAGE2_SYSTEM  # reused as reference's strict prompt
from surrogate.reference import ask_reference, REFERENCE_MODEL
from surrogate.loop import run as loop_run
from surrogate.loop_tools import default_tools

STORE_DIR = Path("backtests")
STORE_JSONL = STORE_DIR / "store.jsonl"


# ---- structured top-pick extraction (free: local qwen3-8b) -----------------

_EXTRACT_SYS = (
    "You extract the recommended choice from an assistant's answer. "
    "Reply ONLY with compact JSON: "
    '{"top_pick": "<single best item or null>", '
    '"ranked": ["<item>", ...]}. '
    "Use the exact name(s) the answer endorses. No prose, no markdown."
)


def _extract_client() -> tuple[OpenAI, str]:
    url = os.environ.get("STAGE1_BASE_URL", "http://localhost:8000/v1")
    model = os.environ.get("STAGE1_MODEL", "qwen3-8b")
    return OpenAI(base_url=url, api_key="EMPTY"), model


def _clean_ranked(items: list, k: int = 3) -> list[str]:
    """Keep up to k non-empty, de-duped, stripped strings."""
    out, seen = [], set()
    for x in items or []:
        if not isinstance(x, str):
            continue
        x = x.strip()
        if not x or x.lower() in ("null", "none"):
            continue
        key = x.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
        if len(out) >= k:
            break
    return out


def extract_pick(answer: str) -> dict:
    """Return {'top_pick': str|None, 'ranked': [str up to 3]} from free text.

    `ranked` is the primary signal (top-3 metric). `top_pick` = ranked[0] for
    back-compat. We deliberately overrule a separately-emitted top_pick if the
    model picked something not in its own ranked list (that's how the Q8
    "junk Reddit string" artifact happened).
    """
    if not answer or not answer.strip():
        return {"top_pick": None, "ranked": [], "_raw": "", "_ok": False}
    client, model = _extract_client()
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _EXTRACT_SYS},
                {"role": "user", "content": answer[:6000]},
            ],
            temperature=0.0,
            max_tokens=300,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = r.choices[0].message.content or ""
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {}
        tp_raw = data.get("top_pick")
        ranked = _clean_ranked(data.get("ranked") or [], k=3)
        # If declared top_pick isn't in its own ranked, treat it as suspect
        # (the Q8 "Unsloths Gemma 4 31b UD q5_xl" artifact). Prefer ranked[0].
        if isinstance(tp_raw, str) and tp_raw.strip() and tp_raw.lower() not in ("null", "none"):
            if not ranked:
                ranked = [tp_raw.strip()]
            elif tp_raw.strip().lower() != ranked[0].lower():
                # keep ranked[0] as top_pick; tp_raw appended if not present
                if not any(tp_raw.strip().lower() == x.lower() for x in ranked):
                    ranked = (_clean_ranked(ranked + [tp_raw.strip()], k=3))
        top_pick = ranked[0] if ranked else None
        return {"top_pick": top_pick, "ranked": ranked, "_raw": raw, "_ok": True}
    except Exception as e:
        return {"top_pick": None, "ranked": [], "_raw": f"[extract error: {e!r}]", "_ok": False}


# ---- TOP-3 SOFT-MATCH metric (decided 2026-05-20 — see module docstring) ---

_SOFT_MATCH_SYSTEM = """You judge whether two product/place names refer to the
SAME thing for a purchase-intent recommendation. Apply these rules:

MATCH if any of these hold:
- Same product/venue with naming variation
    "Sette Restaurant & Bar" <-> "Sette Restaurant"
    "iPhone 16 Pro" <-> "Apple iPhone 16 Pro"
    "Llama 3.3 70B Instruct" <-> "Llama 3.3 70B"
- Same product line, adjacent/incremental versions
    "Saucony Endorphin Speed 3" <-> "Saucony Endorphin Speed 4"
    "Nike Vaporfly 3" <-> "Nike Vaporfly 4"
- Same generation, same line, variant tier difference
    "iPhone 16 Pro" <-> "iPhone 16 Pro Max"
    "MacBook Air M3" <-> "MacBook Air M3 (13-inch)"

NO MATCH if:
- Different major generation
    "iPhone 15" <-> "iPhone 16"
    "Llama 3.3" <-> "Llama 4"
- Different brand / different line / different venue
    "Burj Al Arab Jumeirah" <-> "Mandarin Oriental Jumeirah"
    "Nike Vaporfly" <-> "Saucony Endorphin"

Compare each item in list A to EVERY item in list B. An item in A counts as
matched if it matches ANY item in B.

Return ONLY compact JSON, no prose, no markdown:
{"matched_pairs": [["<a_item>", "<b_item>", "<one-line reason>"]],
 "overlap": <int 0..len(A)>}"""


def soft_match_top3(a_list: list[str], b_list: list[str]) -> dict:
    """Judge top-3 overlap with the soft rules above. 0..3 score.

    Uses GLM via z.ai (no GPU needed). Cheap: one judge call per question.
    Falls back to a strict normalized-equality check if the judge errors.
    """
    a = _clean_ranked(a_list, k=3)
    b = _clean_ranked(b_list, k=3)
    if not a or not b:
        return {"overlap": 0, "matched_pairs": [], "a": a, "b": b,
                "judge_raw": "", "_ok": False, "_fallback": False}

    user = json.dumps({"A": a, "B": b}, ensure_ascii=False)
    try:
        r = ask_reference(question=user, system=_SOFT_MATCH_SYSTEM,
                          max_tokens=600, thinking=False)
        raw = r["answer"] or ""
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {}
        pairs = data.get("matched_pairs") or []
        overlap = int(data.get("overlap", 0))
        # sanity clamp
        overlap = max(0, min(overlap, len(a)))
        return {"overlap": overlap, "matched_pairs": pairs, "a": a, "b": b,
                "judge_raw": raw, "_ok": True, "_fallback": False}
    except Exception as e:
        # fallback: strict normalized equality
        def n(s): return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", "", s.lower())).strip()
        nb = [n(x) for x in b]
        pairs = []
        for x in a:
            nx = n(x)
            for j, y in enumerate(b):
                if nx == nb[j] or nx in nb[j] or nb[j] in nx:
                    pairs.append([x, y, "fallback: normalized match"])
                    break
        return {"overlap": len(pairs), "matched_pairs": pairs, "a": a, "b": b,
                "judge_raw": f"[judge error: {e!r}]", "_ok": False, "_fallback": True}


# ---- evidence-pack reconstruction from the new loop's trace ----------------
# The new ReAct loop is single-stage: tool calls and observations are scattered
# through one trajectory. To keep apples-to-apples with the reference, we
# walk the bundle's trace.jsonl, pair each tool_call with its tool_result,
# and format them into the same Stage-2-style EVIDENCE block we used before.

# Tools that don't carry external evidence — exclude from the reference's
# evidence pack so we're only comparing what was actually fetched from the
# world, not the agent's internal scaffolding.
_NON_EVIDENCE_TOOLS = {"stop_and_answer", "think", "check_missing_fields"}


def _evidence_pack_from_bundle(question: str, bundle_dir) -> str:
    """Reconstruct a Stage-2-style EVIDENCE block from the loop's trace."""
    trace = Path(bundle_dir) / "trace.jsonl"
    if not trace.exists():
        return f"QUESTION: {question}\n\nEVIDENCE: (no trace.jsonl found)"

    events = [json.loads(l) for l in trace.read_text().splitlines() if l.strip()]
    pairs: dict[int, dict] = {}
    for e in events:
        step = e.get("step")
        if step is None:
            continue
        if e["kind"] == "tool_call":
            pairs.setdefault(step, {})["call"] = e
        elif e["kind"] in ("tool_result", "tool_error"):
            pairs.setdefault(step, {})["result"] = e

    parts = [f"QUESTION: {question}", "", "EVIDENCE GATHERED FROM TOOL CALLS:"]
    has_any = False
    for step in sorted(pairs):
        p = pairs[step]
        call = p.get("call")
        if not call:
            continue
        if call.get("name") in _NON_EVIDENCE_TOOLS:
            continue
        result = p.get("result", {})
        body = str(result.get("result") or result.get("error") or "(no result)")
        parts.append("")
        parts.append(
            f"---- Source (step {step}): "
            f"{call['name']}({json.dumps(call.get('args', {}), ensure_ascii=False)}) ----"
        )
        parts.append(body)
        has_any = True

    if not has_any:
        return (
            f"QUESTION: {question}\n\n"
            "EVIDENCE: (no external tools were called; nothing to review)\n\n"
            "Answer the question and say clearly that no web evidence was available."
        )
    parts.append("")
    parts.append(
        "Now, using ONLY the evidence above, think step by step and provide your "
        "best answer to the QUESTION. Cite specific source URLs."
    )
    return "\n".join(parts)


def _concat_thinking(messages: list[dict]) -> str:
    """Pull every <think>...</think> block out of assistant messages and
    concatenate them — for storage in the entry's `thinking` field."""
    parts: list[str] = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        content = m.get("content", "") or ""
        for match in re.finditer(r"<think>\s*(.*?)\s*</think>", content, re.DOTALL):
            parts.append(match.group(1).strip())
    return "\n\n---\n\n".join(parts)


def _count_evidence_tool_calls(bundle_dir) -> int:
    """Count tool_call events that are not scaffolding (think / stop / check)."""
    trace = Path(bundle_dir) / "trace.jsonl"
    if not trace.exists():
        return 0
    n = 0
    for line in trace.read_text().splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("kind") == "tool_call" and e.get("name") not in _NON_EVIDENCE_TOOLS:
            n += 1
    return n


# ---- one question end-to-end ----------------------------------------------

def run_one(question: str, *, log_root: str = "logs") -> dict:
    t0 = time.time()
    # New: single-stage ReAct loop with the 7-tool engineered workflow.
    res = loop_run(question, tools=default_tools(), log_root=log_root)

    # Reconstruct apples-to-apples evidence pack from the loop's trace and
    # feed it verbatim to the reference (same fairness contract as before).
    stage2_user = _evidence_pack_from_bundle(question, res.bundle_dir)

    ref_ev = ask_reference(question=stage2_user, system=STAGE2_SYSTEM)
    ref_bare = ask_reference(question=question)

    sur_answer = res.final_answer or ""
    sur_pick = extract_pick(sur_answer)
    refev_pick = extract_pick(ref_ev["answer"])
    refbare_pick = extract_pick(ref_bare["answer"])

    entry = {
        "question": question,
        "capture_date": date.today().isoformat(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "bundle_dir": str(res.bundle_dir),
        "termination": res.termination,
        "surrogate": {
            "model": os.environ.get("STAGE2_MODEL", "qwen3-8b"),
            "answer": sur_answer,
            "thinking": _concat_thinking(res.messages),
            "pick": sur_pick,
            "n_tool_calls": _count_evidence_tool_calls(res.bundle_dir),
            "steps": res.steps,
        },
        "reference_evidence": {
            "model": REFERENCE_MODEL,
            "answer": ref_ev["answer"],
            "thinking": ref_ev["thinking"],
            "thinking_param_accepted": ref_ev["thinking_param_accepted"],
            "pick": refev_pick,
            "usage": ref_ev["usage"],
        },
        "reference_bare": {
            "model": REFERENCE_MODEL,
            "answer": ref_bare["answer"],
            "thinking": ref_bare["thinking"],
            "pick": refbare_pick,
            "usage": ref_bare["usage"],
        },
        "duration_s": round(time.time() - t0, 1),
    }
    # Top-3 soft-match scoring (the agreed metric, unchanged).
    entry["top3"] = {
        "_metric": "top-3 soft-match overlap (decided 2026-05-20; NDCG parked)",
        "surrogate_vs_reference_evidence": soft_match_top3(
            sur_pick["ranked"], refev_pick["ranked"]
        ),
        "surrogate_vs_reference_bare": soft_match_top3(
            sur_pick["ranked"], refbare_pick["ranked"]
        ),
        "reference_evidence_vs_bare": soft_match_top3(
            refev_pick["ranked"], refbare_pick["ranked"]
        ),
    }
    return entry


# ---- store + verbatim render ----------------------------------------------

def append_store(entry: dict) -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    with STORE_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _section(e: dict) -> str:
    sr = e["surrogate"]["pick"].get("ranked") or []
    er = e["reference_evidence"]["pick"].get("ranked") or []
    br = e["reference_bare"]["pick"].get("ranked") or []
    t3 = e.get("top3") or {}
    ev = (t3.get("surrogate_vs_reference_evidence") or {})
    ba = (t3.get("surrogate_vs_reference_bare") or {})
    L = []
    L.append(f"## {e['question']}")
    L.append("")
    L.append(f"- capture_date: {e['capture_date']}  | bundle: `{e['bundle_dir']}`")
    L.append(f"- **surrogate top-3:** {sr}  ({e['surrogate']['n_tool_calls']} tool calls)")
    L.append(f"- **GLM (same evidence) top-3:** {er}  "
             f"-> overlap={ev.get('overlap', 0)}/{len(sr) if sr else 0}")
    L.append(f"- **GLM (bare/memory) top-3:** {br}  "
             f"-> overlap={ba.get('overlap', 0)}/{len(sr) if sr else 0}")
    if ev.get("matched_pairs"):
        L.append("  - matches (surrogate ↔ GLM/ev): " +
                 "; ".join(f"{p[0]} ↔ {p[1]}" for p in ev["matched_pairs"]))
    L.append("")
    for label, key in (
        ("SURROGATE (qwen3-8b)", "surrogate"),
        (f"REFERENCE {e['reference_evidence']['model']} — SAME EVIDENCE", "reference_evidence"),
        (f"REFERENCE {e['reference_bare']['model']} — BARE (memory)", "reference_bare"),
    ):
        blk = e[key]
        L.append(f"### {label}")
        L.append("")
        L.append("#### thinking (verbatim)")
        L.append("```")
        L.append((blk.get("thinking") or "(none)"))
        L.append("```")
        L.append("#### answer (verbatim)")
        L.append("```")
        L.append((blk.get("answer") or "(empty)"))
        L.append("```")
        L.append("")
    L.append("---")
    return "\n".join(L)


def write_run_md(entries: list[dict], path: Path) -> None:
    n = len(entries)

    def sum_overlap(key: str) -> tuple[int, int]:
        total = 0
        denom = 0
        for e in entries:
            ev = ((e.get("top3") or {}).get(key) or {})
            total += ev.get("overlap", 0)
            denom += len(ev.get("a") or e["surrogate"]["pick"].get("ranked") or [])
        return total, denom

    ev_o, ev_d = sum_overlap("surrogate_vs_reference_evidence")
    ba_o, ba_d = sum_overlap("surrogate_vs_reference_bare")
    perfect_ev = sum(
        1 for e in entries
        if ((e.get("top3") or {}).get("surrogate_vs_reference_evidence") or {}).get("overlap", 0)
        >= len(e["surrogate"]["pick"].get("ranked") or [])
        and (e["surrogate"]["pick"].get("ranked") or [])
    )
    any_ev = sum(
        1 for e in entries
        if ((e.get("top3") or {}).get("surrogate_vs_reference_evidence") or {}).get("overlap", 0) >= 1
    )
    head = [
        f"# Backtest run — {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- questions: {n}",
        f"- metric: top-3 soft-match overlap",
        f"- **surrogate vs GLM (SAME evidence):** total overlap {ev_o}/{ev_d}  "
        f"| questions w/ ≥1 match: **{any_ev}/{n}**  | full top-3 match: {perfect_ev}/{n}",
        f"- surrogate vs GLM (bare/memory): total overlap {ba_o}/{ba_d}  "
        f"(expected lower — different info source)",
        "",
        "> Verbatim per CLAUDE.md: full thinking + answers below, no truncation.",
        "> Thinking is captured but NOT scored (SOTA APIs typically don't expose it).",
        "",
        "---",
        "",
    ]
    path.write_text("\n".join(head) + "\n".join(_section(e) for e in entries))
