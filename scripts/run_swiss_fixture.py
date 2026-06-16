"""Run a live 3-way compare for the Swiss-supplements demo question and save
the result as data/demo_fixture.json. The surrogate now runs under the
mandatory-verify prompt, so the trajectory should include verify_fact steps
(verdict pills) — and everything is fresh (no stitched-together pieces)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv()

from surrogate.compare import compare_run

Q = "What are the top 10 Swiss supplement brands?"


def main() -> int:
    def cb(system, state):
        print(f"  [{system}] {state}", flush=True)

    print(f"Running compare_run for: {Q!r}\n", flush=True)
    rec = compare_run(Q, brand="Avea", status_cb=cb)
    rec["_fixture"] = True

    # Fix the recurring Sonnet 'Aveva' typo if present.
    s = json.dumps(rec, ensure_ascii=False, indent=2)
    n = s.count("Aveva")
    if n:
        s = s.replace("Aveva", "Avea")
        print(f"fixed {n} 'Aveva' typo(s)")
    (ROOT / "data/demo_fixture.json").write_text(s)

    traj = rec["systems"]["surrogate"].get("trajectory") or []
    nverify = sum(1 for t in traj if t.get("tool") == "verify_fact")
    nfacts = sum(1 for t in traj if t.get("facts"))
    print("\n--- summary ---")
    print("surrogate steps:", len(traj),
          "| verify_fact steps:", nverify, "| extract-with-facts:", nfacts)
    print("surrogate picks:", rec["systems"]["surrogate"]["ranked"])
    print("overlap sur/openai:", rec["matches"]["sur_openai"]["overlap"],
          "sur/claude:", rec["matches"]["sur_claude"]["overlap"])
    print("deep plan steps:", len((rec.get("deep") or {}).get("priority_plan") or []))
    print("Wrote data/demo_fixture.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
