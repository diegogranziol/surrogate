"""Generic re-judge + presentation summary builder.

Reads the latest entries for a given question file and frontier filter from
h2h-store.jsonl, re-runs soft_match_topN with the current (brand-level) rules,
and writes a presentation-quality markdown summary.

Usage:
    python scripts/rejudge_and_summarize.py data/h2h-10q.txt --frontier claude --label "General 10-Q vs Claude"
    python scripts/rejudge_and_summarize.py data/h2h-10q.txt --frontier openai --label "General 10-Q vs ChatGPT"
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from surrogate.head_to_head import soft_match_topN


def latest_per_question(qs: list[str], entries: list[dict], frontier: str) -> dict[str, dict]:
    """Latest entry per question for the requested frontier ('claude' or 'openai')."""
    needle = "gpt" if frontier == "openai" else "claude"
    out: dict[str, dict] = {}
    for e in entries:
        m = (e.get("frontier", {}) or {}).get("model", "") or ""
        if needle not in m:
            continue
        q = (e.get("question") or "").strip()
        if q not in qs:
            continue
        prev = out.get(q)
        if prev is None or e.get("ts", "") > prev.get("ts", ""):
            out[q] = e
    return out


def short_q(q: str, n: int = 65) -> str:
    return q if len(q) <= n else q[: n - 1] + "…"


def render_summary(rows: list[dict], label: str, frontier_label: str) -> str:
    structured = [r for r in rows if r["mode"] == "structured"]
    natural = [r for r in rows if r["mode"] == "natural"]
    s_mean = sum(r["new_overlap"] for r in structured) / max(len(structured), 1)
    s_k_mean = sum(len(r["a"]) for r in structured) / max(len(structured), 1)
    n_mean = sum(r["new_overlap"] for r in natural) / max(len(natural), 1)
    n_k_mean = sum(len(r["a"]) for r in natural) / max(len(natural), 1)

    out: list[str] = []
    out.append(f"# {label}")
    out.append("")
    out.append(f"_Surrogate (Qwen3-32B with 7-tool ReAct loop, multi-engine "
               f"search, curated trusted-domain bias) compared independently "
               f"against **{frontier_label}** on the same 10 questions. "
               f"Brand-level soft-match: \"did the same brand appear in both "
               f"lists, regardless of which specific product\". Judge: Claude "
               f"Haiku with our brand-identity rules._")
    out.append("")

    out.append("## 📊 Overall headline")
    out.append("")
    if structured:
        out.append(f"- **Structured (top-N)** ({len(structured)} q): mean overlap "
                   f"**{s_mean:.2f} / {s_k_mean:.1f}**  "
                   f"_({s_mean / max(s_k_mean, 1):.0%} of surrogate's picks brand-matched in {frontier_label}'s list)_")
    if natural:
        out.append(f"- **Natural mode** ({len(natural)} q): mean overlap "
                   f"**{n_mean:.2f} / {n_k_mean:.1f}**  _({n_mean / max(n_k_mean, 1):.0%})_")
    out.append("")

    # ---- 1. Per-question summary -------------------------------------------
    out.append("## 1. Per-question summary")
    out.append("")
    out.append(f"| # | Question | Mode | Overlap (brand-level) | Surrogate returned | {frontier_label} returned | Δ vs old judge |")
    out.append("|---|----------|------|-----------------------|--------------------|------------|----------|")
    for r in rows:
        delta = r["new_overlap"] - r["old_overlap"]
        sign = "+" if delta > 0 else ("" if delta == 0 else "")
        delta_str = f"{sign}{delta}" if delta else "—"
        out.append(
            f"| {r['i']} | {short_q(r['question'])} | {r['mode']} | "
            f"**{r['new_overlap']}/{len(r['a'])}** | {len(r['a'])} | {len(r['b'])} | "
            f"{delta_str} (was {r['old_overlap']}) |"
        )
    out.append("")

    # ---- 2. Per-question detailed tables ----------------------------------
    out.append("## 2. Per-question side-by-side picks")
    out.append("")
    out.append(f"_✓ next to a pick means it brand-matched at least one pick from the other side._")
    out.append("")

    for r in rows:
        a = r["a"]; b = r["b"]
        pairs = r.get("matched_pairs") or []
        matched_a = {str(p[0]).lower() for p in pairs if len(p) >= 1}
        matched_b = {str(p[1]).lower() for p in pairs if len(p) >= 2}

        out.append(f"### Q{r['i']}. {r['question']}")
        out.append("")
        out.append(f"_Mode: **{r['mode']}**, k={r['k']}, overlap: **{r['new_overlap']}/{len(a)}** (brand-level)_")
        out.append("")
        out.append(f"| Rank | Surrogate (Qwen3-32B) | {frontier_label} |")
        out.append(f"|------|-----------------------|-------|")
        n = max(len(a), len(b))
        for i in range(n):
            ai = a[i] if i < len(a) else ""
            bi = b[i] if i < len(b) else ""
            ai_disp = f"**{ai}** ✓" if ai and ai.lower() in matched_a else ai
            bi_disp = f"**{bi}** ✓" if bi and bi.lower() in matched_b else bi
            out.append(f"| {i + 1} | {ai_disp} | {bi_disp} |")
        out.append("")

        if pairs:
            out.append("**Matched brand pairs:**")
            out.append("")
            for p in pairs:
                reason = p[2] if len(p) > 2 else ""
                out.append(f"- `{p[0]}` ↔ `{p[1]}` — {reason}")
            out.append("")
        else:
            out.append("_No brand matches._")
            out.append("")

    # ---- 3. Insights -------------------------------------------------------
    out.append("---")
    out.append("")
    out.append("## 3. Notes on the methodology")
    out.append("")
    out.append(f"- **What we measure**: brand-level overlap on the final answer "
               f"lists. \"Did the same brand appear in both?\" — not exact-SKU match.")
    out.append(f"- **What each side gets**: the same question, independently. "
               f"Surrogate uses its own 7-tool ReAct loop with Tavily + DDG "
               f"multi-engine search and curated trusted-domain bias. "
               f"{frontier_label} uses its own native web_search.")
    out.append(f"- **Judge**: Claude Haiku 4.5, with explicit brand-level rules. "
               f"Same brand on both sides = MATCH regardless of which specific "
               f"product each side picked. Different brand, same category = NO "
               f"MATCH (avoids over-counting).")
    out.append(f"- **Re-judge**: every overlap number above is the result of "
               f"re-running the brand-level judge on the stored picks. The "
               f"\"old\" column reflects scores from the earlier strict-product "
               f"judge or the regex fallback (when Zhipu was out of balance).")
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("questions_file", type=Path)
    ap.add_argument("--frontier", choices=["claude", "openai"], required=True)
    ap.add_argument("--label", type=str, default=None,
                    help="Display title for the summary, e.g. 'General 10-Q vs ChatGPT'")
    args = ap.parse_args()

    qs = [
        l.strip() for l in args.questions_file.read_text().splitlines()
        if l.strip() and not l.lstrip().startswith("#")
    ]
    entries = [
        json.loads(l)
        for l in (ROOT / "backtests/h2h-store.jsonl").read_text().splitlines()
        if l.strip()
    ]
    latest = latest_per_question(qs, entries, args.frontier)
    print(f"Found {len(latest)}/{len(qs)} entries for frontier={args.frontier}\n")

    rows = []
    for i, q in enumerate(qs, 1):
        e = latest.get(q)
        if e is None:
            print(f"[{i}/{len(qs)}] (no entry) {q}")
            continue
        a = e["match"]["a"]; b = e["match"]["b"]
        k = e.get("k", 10)
        old_overlap = e["match"]["overlap"]
        new_m = soft_match_topN(a, b, k=k)
        new_overlap = new_m["overlap"]
        delta = new_overlap - old_overlap
        sign = "+" if delta > 0 else ""
        print(f"[{i}/{len(qs)}] {q[:55]!r}: old={old_overlap}/{len(a)} -> new=**{new_overlap}**/{len(a)} ({sign}{delta})")
        rows.append({
            "i": i, "question": q, "mode": e.get("mode"), "k": k,
            "a": a, "b": b,
            "old_overlap": old_overlap, "new_overlap": new_overlap,
            "matched_pairs": new_m.get("matched_pairs"),
            "judge_raw": new_m.get("judge_raw"),
            "entry_ts": e.get("ts"),
        })

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    label = args.label or f"{args.questions_file.stem} vs {args.frontier}"
    frontier_label = "Claude" if args.frontier == "claude" else "ChatGPT (gpt-5)"
    md = render_summary(rows, label, frontier_label)
    md_path = ROOT / f"backtests/summary-{args.questions_file.stem}-{args.frontier}-{ts}.md"
    md_path.write_text(md)
    print(f"\nWrote {md_path}")

    # Headline
    structured = [r for r in rows if r["mode"] == "structured"]
    natural = [r for r in rows if r["mode"] == "natural"]
    if structured:
        new_mean = sum(r["new_overlap"] for r in structured) / len(structured)
        old_mean = sum(r["old_overlap"] for r in structured) / len(structured)
        print(f"structured: old={old_mean:.2f} -> new=**{new_mean:.2f}**  ({len(structured)} q)")
    if natural:
        new_mean = sum(r["new_overlap"] for r in natural) / len(natural)
        old_mean = sum(r["old_overlap"] for r in natural) / len(natural)
        print(f"natural:    old={old_mean:.2f} -> new=**{new_mean:.2f}**  ({len(natural)} q)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
