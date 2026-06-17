"""Build the counterfactual ('what if Avea were on the sources') block and
splice it into data/demo_fixture.json.

Picks a third-party platform anchor from the run's cited domains (skips rival
homepages), then re-asks all three models under the grounded assumption that
Avea is listed there. Stores baseline-vs-counterfactual for the before/after
panel."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv()

from surrogate.compare import pick_platform_anchor, counterfactual_run, brand_hit

# Grounded factual blurb, taken from avea-life.com (not invented).
AVEA_BLURB = (
    "Avea Life is a Swiss longevity-supplement brand. Its products — including "
    "NMN, spermidine, and collagen formulations — are developed and "
    "manufactured in GMP-certified facilities in Switzerland, with every batch "
    "independently tested by a Swiss laboratory for purity. Formulations are "
    "science-led, overseen by Chief Science Officer Sophie Chabloz in "
    "collaboration with longevity researchers."
)


def main() -> int:
    p = ROOT / "data/demo_fixture.json"
    rec = json.loads(p.read_text())
    brand = rec.get("brand", "Avea")
    question = rec["question"]
    k = rec.get("k", 10)
    mode = rec.get("mode", "structured")

    anchor, platforms = pick_platform_anchor(rec)
    print(f"anchor: {anchor}\nplatforms: {platforms[:6]}\n")
    if not anchor:
        print("no platform anchor found; aborting")
        return 1

    print("Running counterfactual across all three models…", flush=True)
    cf = counterfactual_run(question, brand, AVEA_BLURB, anchor, k=k, mode=mode)

    # baseline hits (from the current fixture picks) for the before/after panel
    cf["baseline"] = {
        name: {
            "ranked": rec["systems"][name].get("ranked", []),
            "hit": brand_hit(rec["systems"][name].get("ranked", []), brand),
        }
        for name in ("surrogate", "openai", "claude")
    }
    rec["counterfactual"] = cf

    # fix the recurring 'Aveva' typo defensively
    s = json.dumps(rec, ensure_ascii=False, indent=2)
    if "Aveva" in s:
        s = s.replace("Aveva", "Avea")
    p.write_text(s)

    print("\n--- before / after (brand appears?) ---")
    for name in ("surrogate", "openai", "claude"):
        b = bool(cf["baseline"][name]["hit"])
        a = bool(cf["systems"][name]["hit"])
        hitname = cf["systems"][name]["hit"] or ""
        print(f"  {name:9s}: baseline={b}  ->  counterfactual={a}  {hitname}")
    print("\nWrote data/demo_fixture.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
