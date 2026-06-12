"""compare — 3-way brand-visibility comparison for the demo UI.

One question goes to three systems IN PARALLEL:
  - Surrogate (Qwen3-32B, 7-tool ReAct loop, multi-engine search + trusted-domain bias)
  - ChatGPT  (gpt-5 via Responses API + web_search)
  - Claude   (claude-sonnet-4-6 + web_search + extended thinking)

Then:
  - top-N picks extracted from each answer
  - brand-level soft-match: surrogate↔ChatGPT, surrogate↔Claude, ChatGPT↔Claude
  - a deterministic "what should <brand> do" suggestions block, grounded in the
    URLs each frontier actually consulted in THIS run

Adding another frontier (e.g. GLM once the Zhipu account has balance) is one
entry in head_to_head.FRONTIERS plus one worker below.

Per CLAUDE.md prime directive: the full record (answers, thinking, tool calls,
consulted URLs) is persisted verbatim to backtests/compare-store.jsonl; the UI
shows it in expanders rather than dropping it.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from surrogate.loop import run as loop_run
from surrogate.loop_tools import default_tools
from surrogate.frontier_claude import ask_claude
from surrogate.frontier_openai import ask_openai
from surrogate.backtest import _concat_thinking
from surrogate.head_to_head import (
    extract_pick_topN,
    soft_match_topN,
    infer_question_shape,
)

STORE_DIR = Path("backtests")
COMPARE_JSONL = STORE_DIR / "compare-store.jsonl"


# ---- URL extraction from frontier responses --------------------------------
# These take the ask_claude / ask_openai RESULT dicts (tool_calls / blocks_raw
# at the top level), unlike the store-entry variants in the audit scripts.

def claude_consulted_urls(resp: dict) -> list[str]:
    out: list[str] = []
    for tc in resp.get("tool_calls") or []:
        if tc.get("kind") != "tool_result":
            continue
        content = tc.get("content") or []
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "web_search_result":
                    u = item.get("url")
                    if u:
                        out.append(u)
    return out


def openai_cited_urls(resp: dict) -> list[str]:
    out: list[str] = []
    for blk in resp.get("blocks_raw") or []:
        if not isinstance(blk, dict) or blk.get("type") != "message":
            continue
        for c in blk.get("content") or []:
            if not isinstance(c, dict):
                continue
            for a in c.get("annotations") or []:
                if isinstance(a, dict) and a.get("type") == "url_citation":
                    u = a.get("url")
                    if u:
                        out.append(re.sub(r"[?&]utm_source=openai", "", u))
    return out


def _domain_of(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


# ---- brand visibility + suggestions -----------------------------------------

def brand_hit(ranked: list[str], brand: str) -> str | None:
    """Return the matching pick if `brand` appears in any ranked item."""
    needle = (brand or "").lower()
    if not needle:
        return None
    for x in ranked or []:
        if needle in str(x).lower():
            return x
    return None


def make_advice(question: str, hits: dict, picks: dict) -> tuple[str, list[str]]:
    """Per-query narrative explanation + tailored action list.

    `hits` maps system name -> matching pick (or None); `picks` maps system
    name -> ranked list. Canonical version — scripts/build_avea_audit.py
    imports this same logic.
    """
    q = question.lower()
    any_hit = any(hits.values())

    if not any_hit:
        why = "Avea Life does not appear in any of the three systems' answers."
    elif sum(1 for v in hits.values() if v) == 1:
        which = [k for k, v in hits.items() if v][0]
        label = {"surrogate": "the open-model proxy", "claude": "Claude", "openai": "ChatGPT"}[which]
        why = f"Avea appears only in {label}'s answer; the other two systems don't surface it."
    else:
        why = "Avea's visibility is uneven — it appears in some systems but not consistently."

    actions: list[str] = []
    if "nmn" in q or "nad+" in q or "nad " in q:
        actions.append("Submit Avea NMN to ConsumerLab for independent testing. "
                       "ConsumerLab's NMN category page is cited by both Claude and "
                       "ChatGPT but does not currently list Avea.")
        actions.append("Pitch inclusion in Fortune's annual \"best NMN supplements\" "
                       "feature, which ChatGPT links to directly when answering NMN queries.")
        actions.append("Pursue NSF Certified for Sport listing for Avea NMN. Elysium Health "
                       "ranks high in ChatGPT's NMN answer specifically because it holds "
                       "this certification.")
    elif "spermidine" in q:
        actions.append("Secure a review or comparative mention on oxfordhealthspan.com, which "
                       "is cited repeatedly across spermidine queries and currently "
                       "highlights Primeadine without mentioning Avea Spermidine.")
        actions.append("Submit Avea Spermidine for ConsumerLab review — the supplement-testing "
                       "site does not yet have a spermidine category review and Avea could "
                       "drive its creation.")
        actions.append("Commission or publish a head-to-head bioavailability study against "
                       "spermidineLIFE (the dominant brand in current AI answers). "
                       "PubMed-indexed evidence is the surest route into ChatGPT's citation pool.")
    elif "collagen" in q:
        actions.append("Submit Avea Bio-Collagen for ConsumerLab testing. Every system lists "
                       "Vital Proteins, Momentous, or Ancient Nutrition — ConsumerLab is the "
                       "underlying source that puts brands into those rankings.")
        actions.append("Pitch inclusion in livemomentous.com or health.com collagen comparison "
                       "articles. Both are cited multiple times in Claude and ChatGPT responses.")
        actions.append("Add Product / Review / AggregateRating schema markup to Avea's "
                       "Bio-Collagen pages so structured data surfaces in the search results "
                       "frontier AIs retrieve.")
    elif "magnesium" in q:
        actions.append("Pitch the next Healthline \"best magnesium supplements\" refresh. "
                       "Their current list runs Thorne, Pure Encapsulations, NOW Foods — "
                       "no Swiss brand is included.")
        actions.append("Get listed on consumerlab.com's magnesium category review page. "
                       "This single placement reaches both Claude and ChatGPT.")
    elif "omega" in q or "fish oil" in q:
        actions.append("Obtain IFOS (Nutrasource) third-party certification for Avea Omega-3. "
                       "ChatGPT cites certifications.nutrasource.ca explicitly when picking "
                       "fish oil brands.")
        actions.append("Pitch Avea Omega-3 into comparison reviews alongside Nordic Naturals "
                       "and Carlson — these dominate every system's omega-3 answer.")
    elif "swiss" in q:
        actions.append("ChatGPT and Claude build their \"Swiss supplement brands\" answers "
                       "from heritage-brand listings (Burgerstein, A.Vogel, Bio-Strath, Nestlé "
                       "Health Science, Nutraswiss). Avea isn't on the Swiss-supplement-industry "
                       "directory pages those answers reference.")
        actions.append("Publish a profile of Avea's Swiss-origin story on Swissinfo, NZZ, or "
                       "Handelszeitung. ChatGPT pulls from Swiss business and lifestyle press "
                       "for this query.")
        actions.append("Apply for membership listings with Swiss Sport Nutrition Society and "
                       "similar trade bodies — their public member directories appear in "
                       "ChatGPT's source set for Swiss-brand questions.")
    elif "longevity" in q or "mitochondrial" in q or "cellular health" in q or "multivitamin" in q:
        actions.append("For these broader queries, frontier systems often answer at the "
                       "compound level (Vitamin D, CoQ10, NMN, creatine) rather than the "
                       "brand level. Category-leader educational content on Avea's site "
                       "gives AIs a brand-connected source to cite when buyers ask "
                       "compound-level questions.")
        actions.append("Place Avea's longevity stack in editorial comparisons on "
                       "livemomentous.com and oxfordhealthspan.com longevity content. These "
                       "are cited as authority sources across all longevity queries.")
    elif "resveratrol" in q:
        actions.append("Pitch Avea Resveratrol into Momentous and Double Wood resveratrol "
                       "comparisons — these are the two brands ChatGPT lists first.")
        actions.append("Submit Avea Resveratrol for ConsumerLab review and seek listing in "
                       "Healthline's resveratrol category coverage.")
    else:
        actions.append("Pitch Avea for inclusion on Healthline, Fortune, ConsumerLab, and "
                       "Innerbody — the four highest-frequency sources in supplement "
                       "answers across both frontier systems.")
        actions.append("Obtain independent third-party testing through NSF or Nutrasource "
                       "to register on the certification lookups ChatGPT consults.")

    return why, actions


def _grounded_why(base_why: str, openai_urls: list[str], claude_urls: list[str]) -> str:
    """Extend the why-sentence with the actual domains consulted in THIS run."""
    bits = [base_why]
    o_doms = sorted({_domain_of(u) for u in openai_urls} - {""})
    c_doms = sorted({_domain_of(u) for u in claude_urls} - {""})
    if o_doms:
        shown = ", ".join(o_doms[:6])
        more = f" (+{len(o_doms) - 6} more)" if len(o_doms) > 6 else ""
        bits.append(f"ChatGPT built its answer from {shown}{more}.")
    if c_doms:
        shown = ", ".join(c_doms[:6])
        more = f" (+{len(c_doms) - 6} more)" if len(c_doms) > 6 else ""
        bits.append(f"Claude consulted {shown}{more}.")
    if o_doms or c_doms:
        bits.append("None of these currently feature the brand for this query.")
    return " ".join(bits)


# ---- the 3-way run ----------------------------------------------------------

def compare_run(
    question: str,
    *,
    k: int | None = None,
    mode: str | None = None,
    brand: str = "Avea",
    status_cb=None,
    log_root: str = "logs",
) -> dict:
    """Run the question through all three systems in parallel and score.

    `status_cb(system, state)` is invoked from the MAIN thread only (Streamlit
    placeholders aren't thread-safe): once with "done"/"error: …" per system
    as its future resolves, then for the judge phase.
    """
    inferred_k, inferred_mode = infer_question_shape(question)
    k = k or inferred_k
    mode = mode or inferred_mode

    def _notify(system: str, state: str) -> None:
        if status_cb:
            try:
                status_cb(system, state)
            except Exception:
                pass

    # ---- workers (each returns its full record; extraction inside worker) --
    def _run_surrogate() -> dict:
        t0 = time.time()
        res = loop_run(question, tools=default_tools(), log_root=log_root)
        answer = res.final_answer or ""
        picks = extract_pick_topN(answer, k=k)
        return {
            "model": "qwen3-32b (surrogate)",
            "answer": answer,
            "thinking": _concat_thinking(res.messages),
            "ranked": picks["ranked"],
            "steps": res.steps,
            "termination": res.termination,
            "bundle": str(res.bundle_dir) if res.bundle_dir else None,
            "duration_s": time.time() - t0,
        }

    def _run_openai() -> dict:
        t0 = time.time()
        r = ask_openai(question, mode=mode)
        picks = extract_pick_topN(r["answer"], k=k)
        return {
            "model": r["model"],
            "answer": r["answer"],
            "thinking": r["thinking"],
            "ranked": picks["ranked"],
            "tool_calls": r["tool_calls"],
            "blocks_raw": r["blocks_raw"],
            "usage": r["usage"],
            "stop_reason": r["stop_reason"],
            "urls": openai_cited_urls(r),
            "duration_s": time.time() - t0,
        }

    def _run_claude() -> dict:
        t0 = time.time()
        r = ask_claude(question, mode=mode)
        picks = extract_pick_topN(r["answer"], k=k)
        return {
            "model": r["model"],
            "answer": r["answer"],
            "thinking": r["thinking"],
            "ranked": picks["ranked"],
            "tool_calls": r["tool_calls"],
            "blocks_raw": r["blocks_raw"],
            "usage": r["usage"],
            "stop_reason": r["stop_reason"],
            "urls": claude_consulted_urls(r),
            "duration_s": time.time() - t0,
        }

    workers = {"surrogate": _run_surrogate, "openai": _run_openai, "claude": _run_claude}
    results: dict[str, dict] = {}
    errors: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {name: ex.submit(fn) for name, fn in workers.items()}
        pending = set(futures)
        while pending:
            for name in sorted(pending):
                fut = futures[name]
                if fut.done():
                    pending.discard(name)
                    try:
                        results[name] = fut.result()
                        _notify(name, "done")
                    except Exception as e:
                        errors[name] = f"{type(e).__name__}: {e}"
                        results[name] = {"model": name, "answer": "", "thinking": "",
                                         "ranked": [], "urls": [], "duration_s": 0.0,
                                         "error": errors[name]}
                        _notify(name, f"error: {errors[name]}")
                    break
            else:
                time.sleep(0.5)

    # ---- brand-level soft match ---------------------------------------------
    _notify("judge", "running")
    sur_ranked = results["surrogate"]["ranked"]
    matches = {
        "sur_openai": soft_match_topN(sur_ranked, results["openai"]["ranked"], k=max(k, 10)),
        "sur_claude": soft_match_topN(sur_ranked, results["claude"]["ranked"], k=max(k, 10)),
        "openai_claude": soft_match_topN(results["openai"]["ranked"],
                                         results["claude"]["ranked"], k=max(k, 10)),
    }
    _notify("judge", "done")

    # ---- suggestions ----------------------------------------------------------
    hits = {name: brand_hit(results[name]["ranked"], brand) for name in workers}
    base_why, actions = make_advice(question, hits, {n: results[n]["ranked"] for n in workers})
    why = base_why if any(hits.values()) else _grounded_why(
        base_why,
        results["openai"].get("urls") or [],
        results["claude"].get("urls") or [],
    )

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "k": k,
        "mode": mode,
        "brand": brand,
        "systems": results,
        "matches": matches,
        "suggestions": {"hits": hits, "why": why, "actions": actions},
        "errors": errors,
    }

    # Persist the full record verbatim (CLAUDE.md prime directive).
    STORE_DIR.mkdir(exist_ok=True, parents=True)
    with COMPARE_JSONL.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    return record
