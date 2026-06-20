"""Run a full demo example end-to-end and save it as a fixture.

One command produces everything the demo renders for a question:
  - 3-way compare (surrogate + ChatGPT + Claude), picks, matches, suggestions,
    deep analysis, graph panels, and classic Google rank (all via compare_run);
  - the multi-scenario counterfactual (listed-on-source / own-site-in-search /
    Wikipedia-page) spliced in, with the brand's real grounding blurb.

Needs the surrogate (B200/vLLM) up, since the surrogate is one of the models.

Examples:
  # German brands, default Avea blurb, write a separate fixture
  python scripts/run_example_fixture.py \
      -q "What are the top 10 German supplement brands?" \
      --out data/fixtures/german.json

  # skip the counterfactual (faster: 3 model calls instead of 12)
  python scripts/run_example_fixture.py -q "…" --no-counterfactual
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv()

from surrogate.compare import (
    compare_run, pick_platform_anchor, counterfactual_scenarios, brand_hit,
)

# Default grounding blurb (the client). Override with --blurb-file for other
# brands. Taken from avea-life.com — real, not invented.
DEFAULT_BLURB = (
    "Avea Life is a Swiss longevity-supplement brand. Its products — including "
    "NMN, spermidine, and collagen formulations — are developed and "
    "manufactured in GMP-certified facilities in Switzerland, with every batch "
    "independently tested by a Swiss laboratory for purity. Formulations are "
    "science-led, overseen by Chief Science Officer Sophie Chabloz in "
    "collaboration with longevity researchers."
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-q", "--question", required=True, help="the demo question")
    ap.add_argument("-b", "--brand", default="Avea", help="brand to track")
    ap.add_argument("--out", default="data/demo_fixture.json",
                    help="output fixture path (default: data/demo_fixture.json)")
    ap.add_argument("--blurb-file", default=None,
                    help="text file with the brand's grounding blurb")
    ap.add_argument("--no-counterfactual", action="store_true",
                    help="skip the (expensive) counterfactual scenarios")
    args = ap.parse_args()

    blurb = DEFAULT_BLURB
    if args.blurb_file:
        blurb = Path(args.blurb_file).read_text().strip()

    def cb(system, state):
        print(f"  [{system}] {state}", flush=True)

    print(f"Running compare_run for: {args.question!r}  (brand={args.brand})\n",
          flush=True)
    rec = compare_run(args.question, brand=args.brand, status_cb=cb)
    rec["_fixture"] = True

    if not args.no_counterfactual:
        anchor, platforms = pick_platform_anchor(rec)
        print(f"\nanchor: {anchor}  platforms: {platforms[:5]}", flush=True)
        if anchor:
            print("Running counterfactual scenarios…", flush=True)
            cf = counterfactual_scenarios(
                args.question, args.brand, blurb, anchor,
                k=rec.get("k", 10), mode=rec.get("mode", "structured"),
                status_cb=lambda sc, st: print(f"  [{sc}] {st}", flush=True),
            )
            cf["baseline"] = {
                n: {"ranked": rec["systems"][n].get("ranked", []),
                    "hit": brand_hit(rec["systems"][n].get("ranked", []), args.brand)}
                for n in ("surrogate", "openai", "claude")
            }
            rec["counterfactual"] = cf
        else:
            print("no platform anchor found; skipping counterfactual", flush=True)

    # defensive: fix the recurring Sonnet 'Aveva' typo
    s = json.dumps(rec, ensure_ascii=False, indent=2)
    n = s.count("Aveva")
    if n:
        s = s.replace("Aveva", "Avea")
        print(f"fixed {n} 'Aveva' typo(s)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(s)

    # summary
    sur = rec["systems"]["surrogate"]
    print("\n--- summary ---")
    print("surrogate picks:", sur.get("ranked"))
    print("chatgpt picks:  ", rec["systems"]["openai"].get("ranked"))
    print("claude picks:   ", rec["systems"]["claude"].get("ranked"))
    cr = rec.get("classic_rank") or {}
    ranked_in_google = [r for r in (cr.get("ranks") or []) if r.get("position")]
    print(f"classic search: {cr.get('engine')} scanned {cr.get('results_count')}, "
          f"{len(ranked_in_google)} brand sites ranked")
    if rec.get("counterfactual", {}).get("scenarios"):
        print("counterfactual scenarios:",
              [s["id"] for s in rec["counterfactual"]["scenarios"]])
    print(f"\nWrote {out}")
    print(f"Render it:  python scripts/build_static_demo.py --fixture {out} "
          f"--out static_demo/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
