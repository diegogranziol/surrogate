"""Run a single question through surrogate + Claude + gpt-5 and produce a
3-way brand-visibility comparison.

Usage:
    python scripts/threeway_oneq.py "What are the best supplement brands in Switzerland?"

Writes backtests/threeway-<ts>.md with side-by-side picks and pairwise overlap.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from surrogate.loop import run as loop_run
from surrogate.loop_tools import default_tools
from surrogate.frontier_claude import ask_claude, current_claude_model
from surrogate.frontier_openai import ask_openai, current_openai_model
from surrogate.head_to_head import (
    extract_pick_topN, soft_match_topN, infer_question_shape, _concat_thinking,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", type=str)
    ap.add_argument("--k", type=int, default=None,
                    help="Top-N (auto-detect from question if omitted).")
    ap.add_argument("--mode", choices=["structured", "natural", "auto"], default="auto")
    args = ap.parse_args()

    q = args.question
    inferred_k, inferred_mode = infer_question_shape(q)
    k = args.k if args.k else inferred_k
    mode = args.mode if args.mode != "auto" else inferred_mode

    print(f"\nQuestion: {q!r}")
    print(f"Mode: {mode}, k: {k}\n")

    # 1. Surrogate
    print("=== Surrogate (Qwen3-32B + multi-engine + bias) ===", flush=True)
    t0 = time.time()
    sur = loop_run(q, tools=default_tools())
    sur_dur = time.time() - t0
    sur_answer = sur.final_answer or ""
    sur_pick = extract_pick_topN(sur_answer, k=k)
    print(f"  steps: {sur.steps}, term: {sur.termination}, {sur_dur:.1f}s")
    print(f"  picks ({len(sur_pick['ranked'])}): {sur_pick['ranked']}\n", flush=True)

    # 2. Claude
    print(f"=== Claude ({current_claude_model()}) ===", flush=True)
    t0 = time.time()
    cla = ask_claude(q, mode=mode)
    cla_dur = time.time() - t0
    cla_pick = extract_pick_topN(cla["answer"], k=k)
    print(f"  stop: {cla['stop_reason']}, {cla_dur:.1f}s")
    print(f"  picks ({len(cla_pick['ranked'])}): {cla_pick['ranked']}\n", flush=True)

    # 3. OpenAI
    print(f"=== gpt-5 ({current_openai_model()}) ===", flush=True)
    t0 = time.time()
    oai = ask_openai(q, mode=mode)
    oai_dur = time.time() - t0
    oai_pick = extract_pick_topN(oai["answer"], k=k)
    print(f"  status: {oai['stop_reason']}, {oai_dur:.1f}s")
    print(f"  picks ({len(oai_pick['ranked'])}): {oai_pick['ranked']}\n", flush=True)

    # 3-way pairwise soft-match (brand-level rule baked into the judge prompt)
    print("=== Pairwise soft-match (brand-level) ===", flush=True)
    sur_cla = soft_match_topN(sur_pick["ranked"], cla_pick["ranked"], k=max(k, 10))
    sur_oai = soft_match_topN(sur_pick["ranked"], oai_pick["ranked"], k=max(k, 10))
    cla_oai = soft_match_topN(cla_pick["ranked"], oai_pick["ranked"], k=max(k, 10))
    print(f"  Surrogate ↔ Claude:  {sur_cla['overlap']}/{len(sur_cla['a'])}")
    print(f"  Surrogate ↔ gpt-5:   {sur_oai['overlap']}/{len(sur_oai['a'])}")
    print(f"  Claude    ↔ gpt-5:   {cla_oai['overlap']}/{len(cla_oai['a'])}")

    # Save raw bundle + markdown
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    raw = {
        "ts": ts, "question": q, "mode": mode, "k": k,
        "surrogate": {
            "model": "qwen3-32b", "answer": sur_answer, "ranked": sur_pick["ranked"],
            "thinking": _concat_thinking(sur.messages), "duration_s": sur_dur,
            "steps": sur.steps, "termination": sur.termination,
            "bundle": str(sur.bundle_dir) if sur.bundle_dir else None,
        },
        "claude": {
            "model": cla["model"], "answer": cla["answer"], "ranked": cla_pick["ranked"],
            "thinking": cla["thinking"], "tool_calls": cla["tool_calls"],
            "duration_s": cla_dur, "usage": cla["usage"], "stop_reason": cla["stop_reason"],
        },
        "openai": {
            "model": oai["model"], "answer": oai["answer"], "ranked": oai_pick["ranked"],
            "thinking": oai["thinking"], "tool_calls": oai["tool_calls"],
            "duration_s": oai_dur, "usage": oai["usage"], "stop_reason": oai["stop_reason"],
        },
        "matches": {
            "sur_cla": sur_cla, "sur_oai": sur_oai, "cla_oai": cla_oai,
        },
    }
    raw_path = ROOT / f"backtests/threeway-{ts}.json"
    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2))
    print(f"\nWrote raw JSON → {raw_path}")

    md = _render_md(raw)
    md_path = ROOT / f"backtests/threeway-{ts}.md"
    md_path.write_text(md)
    print(f"Wrote markdown   → {md_path}")
    return 0


def _render_md(d: dict) -> str:
    sur = d["surrogate"]; cla = d["claude"]; oai = d["openai"]
    m = d["matches"]
    out: list[str] = []
    out.append(f"# 3-way brand-visibility test")
    out.append(f"## _\"{d['question']}\"_")
    out.append("")
    out.append(f"_Mode: **{d['mode']}**, k={d['k']}. Each system answers independently "
               f"using its own tools and search backend. Brand-level overlap "
               f"(judged by Claude Haiku): \"did the same brand appear in both lists, "
               f"regardless of which specific product\"._")
    out.append("")

    out.append("## Pairwise overlap")
    out.append("")
    out.append("| Pair | Overlap |")
    out.append("|------|---------|")
    out.append(f"| Surrogate ↔ Claude | **{m['sur_cla']['overlap']}/{len(m['sur_cla']['a'])}** |")
    out.append(f"| Surrogate ↔ gpt-5 | **{m['sur_oai']['overlap']}/{len(m['sur_oai']['a'])}** |")
    out.append(f"| Claude ↔ gpt-5    | **{m['cla_oai']['overlap']}/{len(m['cla_oai']['a'])}** |")
    out.append("")

    out.append("## Picks side-by-side")
    out.append("")
    n = max(len(sur["ranked"]), len(cla["ranked"]), len(oai["ranked"]))
    out.append("| # | Surrogate (Qwen3-32B) | Claude | gpt-5 |")
    out.append("|---|------------------------|--------|-------|")
    for i in range(n):
        a = sur["ranked"][i] if i < len(sur["ranked"]) else ""
        b = cla["ranked"][i] if i < len(cla["ranked"]) else ""
        c = oai["ranked"][i] if i < len(oai["ranked"]) else ""
        out.append(f"| {i + 1} | {a} | {b} | {c} |")
    out.append("")

    out.append("## Matched-pair details")
    out.append("")
    for label, match in (("Surrogate ↔ Claude", m["sur_cla"]),
                        ("Surrogate ↔ gpt-5", m["sur_oai"]),
                        ("Claude ↔ gpt-5", m["cla_oai"])):
        out.append(f"### {label}  — overlap **{match['overlap']}/{len(match['a'])}**")
        out.append("")
        if not match.get("matched_pairs"):
            out.append("_No matches._")
            out.append("")
            continue
        for p in match["matched_pairs"]:
            reason = p[2] if len(p) > 2 else ""
            out.append(f"- `{p[0]}` ↔ `{p[1]}` — {reason}")
        out.append("")

    out.append("## Brand-visibility check (Avea Life)")
    out.append("")
    for name, lst in (("Surrogate", sur["ranked"]),
                      ("Claude", cla["ranked"]),
                      ("gpt-5", oai["ranked"])):
        hits = [x for x in lst if "avea" in str(x).lower()]
        cell = f"✅ `{hits[0]}`" if hits else "❌ absent"
        out.append(f"- **{name}**: {cell}")
    out.append("")

    out.append("## Per-system timings & stats")
    out.append("")
    out.append("| System | Model | Time | Picks returned |")
    out.append("|--------|-------|------|----------------|")
    out.append(f"| Surrogate | qwen3-32b ({sur['steps']} loop steps, term={sur['termination']}) | "
               f"{sur['duration_s']:.1f}s | {len(sur['ranked'])} |")
    out.append(f"| Claude | {cla['model']} (stop={cla['stop_reason']}) | "
               f"{cla['duration_s']:.1f}s | {len(cla['ranked'])} |")
    out.append(f"| gpt-5 | {oai['model']} (stop={oai['stop_reason']}) | "
               f"{oai['duration_s']:.1f}s | {len(oai['ranked'])} |")
    out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
