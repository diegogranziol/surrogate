"""Re-score existing backtest entries with the top-3 soft-match metric.

Reads backtests/store.jsonl, takes the last N entries (default 8 = starter8),
runs soft_match_top3() on the already-stored ranked lists, and writes:
  - backtests/rescore-<ts>.jsonl  (one row per question, structured)
  - backtests/rescore-<ts>.md     (human-readable summary)

No GPU needed. Each judge call hits z.ai (GLM); one call per (sur vs ev,
sur vs bare) pair per question.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from surrogate.backtest import STORE_DIR, STORE_JSONL, soft_match_top3


def main(argv: list[str]) -> int:
    n = int(argv[1]) if len(argv) > 1 else 8
    if not STORE_JSONL.exists():
        print(f"no store at {STORE_JSONL}", file=sys.stderr)
        return 2
    rows = [json.loads(l) for l in STORE_JSONL.read_text().splitlines() if l.strip()]
    rows = rows[-n:]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_jsonl = STORE_DIR / f"rescore-{ts}.jsonl"
    out_md = STORE_DIR / f"rescore-{ts}.md"

    md = [f"# Rescore (top-3 soft-match) — {datetime.now().isoformat(timespec='seconds')}",
          "",
          f"- entries: {len(rows)}  (last {n} from store.jsonl)",
          "- metric: top-3 soft-match overlap; thinking NOT scored",
          "",
          "---", ""]
    total_ev = total_ba = denom = 0
    any_ev = 0
    perfect_ev = 0
    lines = []
    with out_jsonl.open("w", encoding="utf-8") as f:
        for i, e in enumerate(rows, 1):
            sr = (e.get("surrogate") or {}).get("pick", {}).get("ranked") or []
            er = (e.get("reference_evidence") or {}).get("pick", {}).get("ranked") or []
            br = (e.get("reference_bare") or {}).get("pick", {}).get("ranked") or []
            q = e.get("question", "")
            print(f"[{i}/{len(rows)}] {q}")
            ev = soft_match_top3(sr, er)
            ba = soft_match_top3(sr, br)
            row = {
                "question": q,
                "capture_date": e.get("capture_date"),
                "bundle_dir": e.get("bundle_dir"),
                "surrogate_top3": sr,
                "reference_evidence_top3": er,
                "reference_bare_top3": br,
                "top3_surrogate_vs_reference_evidence": ev,
                "top3_surrogate_vs_reference_bare": ba,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_ev += ev.get("overlap", 0)
            total_ba += ba.get("overlap", 0)
            denom += len(sr)
            if ev.get("overlap", 0) >= 1:
                any_ev += 1
            if sr and ev.get("overlap", 0) >= len(sr):
                perfect_ev += 1
            print(f"  surrogate top3: {sr}")
            print(f"  GLM/ev   top3: {er}  -> overlap={ev.get('overlap',0)}/{len(sr)}")
            print(f"  GLM/bare top3: {br}  -> overlap={ba.get('overlap',0)}/{len(sr)}")
            if ev.get("matched_pairs"):
                print("  matches: " + "; ".join(f"{p[0]} ↔ {p[1]}" for p in ev["matched_pairs"]))
            lines.append(f"## {q}\n")
            lines.append(f"- surrogate top-3: {sr}")
            lines.append(f"- GLM (same evidence) top-3: {er}  → **overlap {ev.get('overlap',0)}/{len(sr)}**")
            if ev.get("matched_pairs"):
                lines.append("  - matches: " + "; ".join(f"`{p[0]}` ↔ `{p[1]}` ({p[2] if len(p)>2 else ''})" for p in ev["matched_pairs"]))
            lines.append(f"- GLM (bare) top-3: {br}  → overlap {ba.get('overlap',0)}/{len(sr)}")
            lines.append("")

    head_summary = [
        f"- **surrogate vs GLM (SAME evidence):** total overlap {total_ev}/{denom}  "
        f"| questions ≥1 match: **{any_ev}/{len(rows)}**  | full top-3 match: {perfect_ev}/{len(rows)}",
        f"- surrogate vs GLM (bare):  total overlap {total_ba}/{denom}",
        "", "---", "",
    ]
    out_md.write_text("\n".join(md + head_summary + lines))
    print(f"\n=== rescore done ===")
    print(f"surrogate vs GLM (same evidence): overlap {total_ev}/{denom} items  "
          f"| {any_ev}/{len(rows)} questions ≥1 match  | {perfect_ev}/{len(rows)} full")
    print(f"surrogate vs GLM (bare):          overlap {total_ba}/{denom}")
    print(f"[jsonl] {out_jsonl}")
    print(f"[md]    {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
