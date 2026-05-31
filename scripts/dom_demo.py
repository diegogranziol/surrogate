"""DOM-pair demo runner. Run the surrogate over two user-specified URLs,
write a verbatim dump under reasoning_dumps/ using the try-NN naming
convention so the artifact lives next to try01/try02 for direct comparison.

  python scripts/dom_demo.py "QUESTION" URL_A URL_B [try-number]
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

# Make `surrogate` importable when run from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surrogate.two_stage import run_with_dom_pair  # noqa: E402


def _slug(s: str, n: int = 50) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s.strip().lower()).strip("-")
    return s[:n].rstrip("-")


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    question = argv[1]
    url_a, url_b = argv[2], argv[3]
    try_n = argv[4] if len(argv) > 4 else "03"

    res = run_with_dom_pair(question, url_a, url_b)

    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = Path("reasoning_dumps")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"try{try_n}_{ts}_qwen3-8b_dom-pair_{_slug(question)}.md"

    pair = res["dom_pair"]
    L = []
    L.append(f"# Reasoning dump — try {try_n} (DOM-pair flow)")
    L.append("")
    L.append(f"- **Question:** {question}")
    L.append(f"- **Date/time:** {ts}")
    L.append(f"- **Mode:** `run_with_dom_pair` — Stage 1 bypassed, two user-supplied URLs DOM-crawled and packed as Stage 2 evidence")
    L.append(f"- **Surrogate model:** qwen3-8b (self-hosted, vLLM, Mithril B200)")
    L.append(f"- **URL A:** {url_a}  (crawl ok={pair['ok_a']}, title={pair['a']['title']!r})")
    L.append(f"- **URL B:** {url_b}  (crawl ok={pair['ok_b']}, title={pair['b']['title']!r})")
    L.append(f"- **Source bundle:** {res['bundle_dir']}")
    L.append(f"- **Stage 2 samples:** {len(res['samples'])} (sample 0 = greedy T=0.0; 1-4 = T=1.0)  | total {res['duration_s']:.1f}s")
    L.append("")
    L.append("> Verbatim per project rule (CLAUDE.md): full reasoning_content + content, no truncation.")
    L.append("> Thinking is dumped but not used in any score (per the metric decision).")
    L.append("")

    # DOM extracts (verbatim, as Stage 2 saw them)
    for tag, sec in (("A", pair["a"]), ("B", pair["b"])):
        L.append("=" * 80)
        L.append(f"## DOM extract — Website {tag} ({sec['url']})")
        L.append("=" * 80)
        L.append("")
        L.append("```")
        L.append(sec["as_text"])
        L.append("```")
        L.append("")

    # Stage 2 samples (verbatim)
    for s in res["samples"]:
        L.append("=" * 80)
        L.append(f"## Sample {s['sample_index']} — T={s['temperature']} — {s['duration_s']:.1f}s")
        L.append("=" * 80)
        L.append("")
        L.append("### reasoning_content (thinking tokens, verbatim)")
        L.append("")
        L.append("```")
        L.append(s.get("reasoning") or "(empty)")
        L.append("```")
        L.append("")
        L.append("### content (answer, verbatim)")
        L.append("")
        L.append("```")
        L.append(s.get("content") or "(empty)")
        L.append("```")
        L.append("")

    out.write_text("\n".join(L))
    print(f"\nWROTE {out}  ({out.stat().st_size} bytes)")
    print(f"answer (sample 0, first 300 chars):\n{(res['answer'] or '')[:300]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
