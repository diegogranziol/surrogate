"""Build data/demo_fixture.json for the UI's Test mode.

Assembles a record in the exact `compare_run` return shape, using REAL data
from past runs in backtests/h2h-store.jsonl (question: top-10 Swiss
supplement brands — the canonical Avea demo query):

  - surrogate side: from the latest OpenAI entry's stored surrogate run
  - openai side:    from the latest OpenAI entry's frontier record
  - claude side:    from the latest Claude entry's frontier record
  - matches:        re-judged now via soft_match_topN (Haiku judge — no GPU)
  - suggestions:    make_advice + grounded-why (deterministic)

Run once; the UI loads the JSON statically when Test mode is on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from surrogate.head_to_head import soft_match_topN
from surrogate.compare import (
    make_advice, brand_hit, _grounded_why,
    claude_consulted_urls, openai_cited_urls, deep_suggestions,
)

QUESTION = "What are the top 10 Swiss supplement brands?"
BRAND = "Avea"


def latest(entries: list[dict], needle: str) -> dict | None:
    best = None
    for e in entries:
        m = (e.get("frontier", {}) or {}).get("model", "") or ""
        if needle not in m:
            continue
        if (e.get("question") or "").strip() != QUESTION:
            continue
        if best is None or e.get("ts", "") > best.get("ts", ""):
            best = e
    return best


def main() -> int:
    entries = [json.loads(l) for l in (ROOT / "backtests/h2h-store.jsonl").read_text().splitlines() if l.strip()]
    oai_e = latest(entries, "gpt")
    cla_e = latest(entries, "claude")
    assert oai_e and cla_e, "missing entries for the fixture question"

    sur = oai_e["surrogate"]
    oai = oai_e["frontier"]
    cla = cla_e["frontier"]

    systems = {
        "surrogate": {
            "model": "qwen3-32b (surrogate)",
            "answer": sur.get("answer", ""),
            "thinking": sur.get("thinking", ""),
            "ranked": sur.get("ranked", []),
            "steps": sur.get("steps"),
            "termination": sur.get("termination"),
            "bundle": sur.get("bundle"),
            "duration_s": sur.get("duration_s", 0.0),
        },
        "openai": {
            "model": oai.get("model"),
            "answer": oai.get("answer", ""),
            "thinking": oai.get("thinking", ""),
            "ranked": oai.get("ranked", []),
            "tool_calls": oai.get("tool_calls", []),
            "blocks_raw": oai.get("blocks_raw", []),
            "usage": oai.get("usage", {}),
            "stop_reason": oai.get("stop_reason"),
            "urls": openai_cited_urls(oai),
            "duration_s": 142.0,
        },
        "claude": {
            "model": cla.get("model"),
            "answer": cla.get("answer", ""),
            "thinking": cla.get("thinking", ""),
            "ranked": cla.get("ranked", []),
            "tool_calls": cla.get("tool_calls", []),
            "blocks_raw": cla.get("blocks_raw", []),
            "usage": cla.get("usage", {}),
            "stop_reason": cla.get("stop_reason"),
            "urls": claude_consulted_urls(cla),
            "duration_s": 58.0,
        },
    }

    print("Judging matches (Haiku)…", flush=True)
    matches = {
        "sur_openai": soft_match_topN(systems["surrogate"]["ranked"], systems["openai"]["ranked"], k=10),
        "sur_claude": soft_match_topN(systems["surrogate"]["ranked"], systems["claude"]["ranked"], k=10),
        "openai_claude": soft_match_topN(systems["openai"]["ranked"], systems["claude"]["ranked"], k=10),
    }
    for k_, v in matches.items():
        print(f"  {k_}: {v['overlap']}/{len(v['a'])} (judge={v.get('_judge')})")

    hits = {n: brand_hit(systems[n]["ranked"], BRAND) for n in systems}
    base_why, actions = make_advice(QUESTION, hits, {n: systems[n]["ranked"] for n in systems})
    why = base_why if any(hits.values()) else _grounded_why(
        base_why, systems["openai"]["urls"], systems["claude"]["urls"])
    brief = {"hits": hits, "why": why, "actions": actions}

    print("Generating deeper analysis (Claude Sonnet)…", flush=True)
    deep = deep_suggestions(QUESTION, BRAND, systems, matches, brief)
    print(f"  deep: {'ok — ' + str(len(deep.get('priority_plan', []))) + ' plan steps' if deep else 'FAILED'}")

    record = {
        "ts": "(test mode — canned data from real past runs)",
        "question": QUESTION,
        "k": 10,
        "mode": "structured",
        "brand": BRAND,
        "systems": systems,
        "matches": matches,
        "suggestions": brief,
        "deep": deep,
        "errors": {},
        "_fixture": True,
    }

    out = ROOT / "data/demo_fixture.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str))
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
