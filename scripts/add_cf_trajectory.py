"""Backfill the surrogate's step-by-step trajectory into a fixture's existing
counterfactual block.

Re-runs ONLY the surrogate for each scenario (the augmented question), so the
stored ChatGPT/Claude answers are preserved, and splices the fresh
trajectory/answer/ranked/hit into cf['scenarios'][i]['systems']['surrogate'].
Needs the surrogate (B200/vLLM) up.

  python scripts/add_cf_trajectory.py data/fixtures/nmn.json data/demo_fixture.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv()

from surrogate.loop import run as loop_run
from surrogate.loop_tools import default_tools
from surrogate.compare import (
    _counterfactual_suffix, surrogate_trajectory, brand_hit,
)
from surrogate.head_to_head import extract_pick_topN


def backfill(path: str) -> None:
    p = Path(path)
    rec = json.loads(p.read_text())
    cf = rec.get("counterfactual") or {}
    scenarios = cf.get("scenarios")
    if not scenarios:
        print(f"  no multi-scenario counterfactual — skipping")
        return
    brand = cf.get("brand", "Avea")
    blurb = cf.get("blurb", "")
    anchor = cf.get("anchor", "")
    question = rec["question"]
    k = rec.get("k", 10)

    for sc in scenarios:
        sid = sc.get("id", "")
        aug = question + _counterfactual_suffix(brand, blurb, sid, anchor=anchor)
        print(f"  [{sid}] running surrogate…", flush=True)
        res = loop_run(aug, tools=default_tools())
        ans = res.final_answer or ""
        bundle = str(res.bundle_dir) if res.bundle_dir else None
        ranked = extract_pick_topN(ans, k=k)["ranked"]
        traj = surrogate_trajectory(bundle)
        sc.setdefault("systems", {})["surrogate"] = {
            "model": "qwen3-32b (surrogate)", "ranked": ranked,
            "answer": ans, "trajectory": traj, "hit": brand_hit(ranked, brand),
        }
        print(f"  [{sid}] done: {len(traj)} steps, hit={sc['systems']['surrogate']['hit']!r}")

    s = json.dumps(rec, ensure_ascii=False, indent=2)
    if "Aveva" in s:
        s = s.replace("Aveva", "Avea")
    p.write_text(s)
    print(f"  updated {path}")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: add_cf_trajectory.py <fixture.json> [<fixture.json> ...]")
        return 1
    for path in sys.argv[1:]:
        print(f"=== {path} ===")
        backfill(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
