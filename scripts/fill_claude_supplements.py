"""Fill missing Claude entries for the supplements test set.

The supplements 10-Q test set was originally run only vs OpenAI. The audit
exposes "(not captured)" for Claude on those questions. This script calls
Claude on each question, extracts picks, and appends Claude-tagged entries
to backtests/h2h-store.jsonl — reusing the existing surrogate-side data
from the OpenAI entries so we don't re-run the surrogate.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from surrogate.frontier_claude import ask_claude
from surrogate.head_to_head import extract_pick_topN, soft_match_topN, infer_question_shape

STORE = ROOT / "backtests/h2h-store.jsonl"


def main() -> int:
    qs = [
        l.strip()
        for l in (ROOT / "data/h2h-10q-supplements.txt").read_text().splitlines()
        if l.strip() and not l.lstrip().startswith("#")
    ]
    entries = [json.loads(l) for l in STORE.read_text().splitlines() if l.strip()]

    # Find latest OpenAI entry per question to reuse the surrogate side
    openai_latest: dict[str, dict] = {}
    claude_latest: dict[str, dict] = {}
    for e in entries:
        m = (e.get("frontier", {}) or {}).get("model", "") or ""
        q = (e.get("question") or "").strip()
        if not q:
            continue
        if "gpt" in m:
            prev = openai_latest.get(q)
            if prev is None or e.get("ts", "") > prev.get("ts", ""):
                openai_latest[q] = e
        elif "claude" in m:
            prev = claude_latest.get(q)
            if prev is None or e.get("ts", "") > prev.get("ts", ""):
                claude_latest[q] = e

    pending = []
    for q in qs:
        if q in claude_latest:
            print(f"skip (already has Claude entry): {q!r}")
        elif q not in openai_latest:
            print(f"skip (no OpenAI entry to reuse surrogate from): {q!r}")
        else:
            pending.append(q)

    print(f"\nPending: {len(pending)} questions\n")

    for i, q in enumerate(pending, 1):
        oai = openai_latest[q]
        inferred_k, inferred_mode = infer_question_shape(q)
        k = oai.get("k", inferred_k)
        mode = oai.get("mode", inferred_mode)
        print(f"[{i}/{len(pending)}] {q!r}", flush=True)

        t0 = time.time()
        try:
            cla = ask_claude(q, mode=mode)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e!r}", flush=True)
            continue
        elapsed = time.time() - t0
        cla_pick = extract_pick_topN(cla["answer"], k=k)

        sur = oai["surrogate"]  # reuse surrogate-side verbatim
        match = soft_match_topN(sur["ranked"], cla_pick["ranked"], k=k)

        entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "date": datetime.utcnow().date().isoformat(),
            "question": q,
            "k": k,
            "mode": mode,
            "surrogate": sur,
            "frontier": {
                "model": cla["model"],
                "answer": cla["answer"],
                "thinking": cla["thinking"],
                "ranked": cla_pick["ranked"],
                "ranked_raw": cla_pick["_raw"],
                "tool_calls": cla["tool_calls"],
                "blocks_raw": cla["blocks_raw"],
                "usage": cla["usage"],
                "stop_reason": cla["stop_reason"],
            },
            "match": match,
            "elapsed_s": elapsed,
            "_note": "Claude-only run; surrogate side reused from prior OpenAI entry.",
        }

        with STORE.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        picks_short = cla_pick["ranked"][:5]
        more = f" (+{len(cla_pick['ranked']) - 5} more)" if len(cla_pick["ranked"]) > 5 else ""
        print(f"  Claude picks: {picks_short}{more}")
        print(f"  Surrogate ↔ Claude overlap: {match['overlap']}/{len(match['a'])}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
