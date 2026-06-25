"""Re-judge the supplement test entries with the loosened soft-match rule.

Reads the latest OpenAI entry per question from data/h2h-10q-supplements.txt,
re-runs soft_match_topN (which now uses the brand-level rule), and writes a
rejudged report.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from surrogate.head_to_head import soft_match_topN


def latest_openai_per_question(qs: list[str], entries: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for e in entries:
        m = (e.get("frontier", {}) or {}).get("model", "") or ""
        if "gpt" not in m:
            continue
        q = (e.get("question") or "").strip()
        if q not in qs:
            continue
        prev = out.get(q)
        if prev is None or e.get("ts", "") > prev.get("ts", ""):
            out[q] = e
    return out


def main() -> int:
    qs = [
        l.strip() for l in (ROOT / "data/h2h-10q-supplements.txt").read_text().splitlines()
        if l.strip() and not l.lstrip().startswith("#")
    ]
    entries = [
        json.loads(l)
        for l in (ROOT / "backtests/h2h-store.jsonl").read_text().splitlines()
        if l.strip()
    ]
    latest = latest_openai_per_question(qs, entries)

    print(f"Found OpenAI entries for {len(latest)}/{len(qs)} questions\n")

    results = []
    for i, q in enumerate(qs, 1):
        e = latest.get(q)
        if e is None:
            print(f"[{i}/{len(qs)}] (no entry) {q}")
            continue
        a = e["match"]["a"]   # surrogate ranked
        b = e["match"]["b"]   # frontier ranked
        k = e.get("k", 10)
        old_overlap = e["match"]["overlap"]
        new_m = soft_match_topN(a, b, k=k)
        new_overlap = new_m["overlap"]
        delta = new_overlap - old_overlap
        sign = "+" if delta >= 0 else ""
        print(f"[{i}/{len(qs)}] {q[:55]!r}: old={old_overlap}/{len(a)} -> new=**{new_overlap}**/{len(a)} ({sign}{delta})")
        results.append({
            "i": i,
            "question": q,
            "mode": e.get("mode"),
            "k": k,
            "a": a,
            "b": b,
            "old_overlap": old_overlap,
            "new_overlap": new_overlap,
            "matched_pairs": new_m.get("matched_pairs"),
            "judge_raw": new_m.get("judge_raw"),
            "entry_ts": e.get("ts"),
        })

    out_path = ROOT / f"backtests/rejudge-supplements-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nWrote {out_path}")

    # Headline aggregates
    structured = [r for r in results if r["mode"] == "structured"]
    natural = [r for r in results if r["mode"] == "natural"]
    if structured:
        new_mean = sum(r["new_overlap"] for r in structured) / len(structured)
        old_mean = sum(r["old_overlap"] for r in structured) / len(structured)
        print(f"\nstructured: mean overlap  old={old_mean:.2f}  new=**{new_mean:.2f}**  ({len(structured)} q)")
    if natural:
        new_mean = sum(r["new_overlap"] for r in natural) / len(natural)
        old_mean = sum(r["old_overlap"] for r in natural) / len(natural)
        print(f"natural:    mean overlap  old={old_mean:.2f}  new=**{new_mean:.2f}**  ({len(natural)} q)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
