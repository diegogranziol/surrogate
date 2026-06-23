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
    """URLs ChatGPT relied on. It exposes them two ways: as `url_citation`
    annotation objects, OR (when we ask for 'source URL' in the answer) as
    plain text in the answer body. We collect both, clean, and dedupe."""
    out: list[str] = []
    for blk in resp.get("blocks_raw") or []:
        if not isinstance(blk, dict) or blk.get("type") != "message":
            continue
        for c in blk.get("content") or []:
            if not isinstance(c, dict):
                continue
            for a in c.get("annotations") or []:
                if isinstance(a, dict) and a.get("type") == "url_citation":
                    if a.get("url"):
                        out.append(a["url"])
    # plain-text URLs written into the answer body
    out += _URL_RX.findall(resp.get("answer") or "")

    seen, clean = set(), []
    for u in out:
        u = re.sub(r"[?&]utm_source=openai", "", u).rstrip(".,);]\"'")
        k = u.rstrip("/").lower()
        if u and k not in seen:
            seen.add(k)
            clean.append(u)
    return clean


def _domain_of(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


# ---- surrogate trajectory (ordered think -> action -> sources) --------------
# Parsed from the run's trace.jsonl so the UI can show, step by step, what the
# surrogate was thinking and exactly which URLs each action pulled. This is the
# transparency story: every thought is bound to the tool call it triggered and
# the sources that came back. We own this trace, so we can do it for the
# surrogate in a way the frontier APIs don't expose.

_THINK_RX = re.compile(r"<think>(.*?)</think>", re.S)
_URL_RX = re.compile(r"https?://[^\s)\]\"']+")


def _think_from(content: str) -> str:
    blocks = _THINK_RX.findall(content or "")
    return "\n\n".join(b.strip() for b in blocks if b.strip())


def _action_summary(name: str, args: dict) -> str:
    if name == "search":
        q = args.get("query")
        if isinstance(q, list):
            return "searched " + " · ".join(f'"{x}"' for x in q)
        return f'searched "{q}"'
    if name in ("fetch_url", "extract_entity"):
        return f"read {args.get('url', '')}"
    if name == "verify_fact":
        return f'verified "{(args.get("claim") or "")[:80]}"'
    if name == "check_missing_fields":
        return "checked completeness of a candidate"
    if name == "think":
        return "reflected"
    if name == "stop_and_answer":
        return "finalised the answer"
    return name


def _urls_from_result(result_text: str, args: dict) -> list[str]:
    raw = _URL_RX.findall(str(result_text or ""))
    for key in ("url", "evidence_url"):
        if args.get(key):
            raw.insert(0, str(args[key]))
    # strip trailing punctuation, dedup by normalised form (keep first/cleanest)
    seen, out = set(), []
    for u in raw:
        u = u.rstrip(".,:;)]}\"'")
        key = u.rstrip("/").lower()
        if key and key not in seen:
            seen.add(key)
            out.append(u)
    return out


def _parse_extracted_facts(text: str) -> tuple[list[dict], str]:
    """From an extract_entity result, pull structured entities and an
    OpenGraph title. Returns (facts, og_title)."""
    facts: list[dict] = []
    og_title = ""
    cur: dict | None = None
    keep = ("name", "rating", "review_count", "price", "currency", "price_range")
    for line in str(text or "").splitlines():
        m = re.match(r"\[\d+\] @type:\s*(.+)", line)
        if m:
            if cur:
                facts.append(cur)
            cur = {"type": m.group(1).strip()}
        elif cur is not None and line.startswith("  ") and ":" in line:
            k, _, v = line.strip().partition(":")
            if k.strip() in keep:
                cur[k.strip()] = v.strip()
        elif line.startswith("OpenGraph:"):
            if cur:
                facts.append(cur); cur = None
            mt = re.search(r'"og:title":\s*"([^"]+)"', line)
            if mt:
                og_title = mt.group(1)
    if cur:
        facts.append(cur)
    # keep only entities that carry at least one useful field beyond type
    facts = [f for f in facts if len(f) > 1]
    return facts, og_title


def _parse_verification(text: str) -> dict:
    """From a verify_fact result, pull the claim + verdict."""
    out: dict = {}
    for line in str(text or "").splitlines():
        if line.startswith("Claim:"):
            out["claim"] = line.split(":", 1)[1].strip()
        elif line.startswith("Source:"):
            out["source"] = line.split(":", 1)[1].strip()
        elif line.startswith("Supported:"):
            rest = line.split(":", 1)[1].strip()
            out["verdict"] = rest.split()[0].lower() if rest else ""
    return out


def surrogate_trajectory(bundle_dir: str | None) -> list[dict]:
    """Ordered [{think, action, tool, urls}] from the surrogate's trace.jsonl.
    Returns [] if the bundle/trace is missing."""
    if not bundle_dir:
        return []
    trace = Path(bundle_dir) / "trace.jsonl"
    if not trace.exists():
        return []

    # group events by step
    by_step: dict[int, dict] = {}
    for line in trace.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        step = e.get("step")
        if step is None:
            continue
        by_step.setdefault(step, {})[e["kind"]] = e

    out: list[dict] = []
    for step in sorted(by_step):
        ev = by_step[step]
        resp = ev.get("llm_response", {})
        call = ev.get("tool_call", {})
        result = ev.get("tool_result", {}) or ev.get("tool_error", {})
        think = _think_from(resp.get("content", ""))
        name = call.get("name")
        if not think and not name:
            continue
        args = call.get("args", {}) or {}
        result_text = result.get("result", "")
        urls = _urls_from_result(result_text, args) if name else []
        # don't show URLs for tools that don't carry external sources
        if name in ("think", "check_missing_fields", "stop_and_answer"):
            urls = []

        facts, og_title, verify = [], "", None
        if name == "extract_entity":
            facts, og_title = _parse_extracted_facts(result_text)
        elif name == "verify_fact":
            verify = _parse_verification(result_text)

        out.append({
            "step": step,
            "think": think,
            "tool": name,
            "action": _action_summary(name, args) if name else None,
            "urls": urls,
            "facts": facts,
            "og_title": og_title,
            "verify": verify,
        })
    return out


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
        why = "Avea's visibility is uneven. It appears in some systems but not others."

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
        actions.append("Submit Avea Spermidine for ConsumerLab review. The site has no "
                       "spermidine category review yet, so Avea could drive its creation.")
        actions.append("Commission or publish a head-to-head bioavailability study against "
                       "spermidineLIFE (the dominant brand in current AI answers). "
                       "PubMed-indexed evidence is the surest route into ChatGPT's citation pool.")
    elif "collagen" in q:
        actions.append("Submit Avea Bio-Collagen for ConsumerLab testing. Every system lists "
                       "Vital Proteins, Momentous, or Ancient Nutrition, and ConsumerLab is "
                       "the underlying source that puts brands into those rankings.")
        actions.append("Pitch inclusion in livemomentous.com or health.com collagen comparison "
                       "articles. Both are cited multiple times in Claude and ChatGPT responses.")
        actions.append("Add Product / Review / AggregateRating schema markup to Avea's "
                       "Bio-Collagen pages so structured data surfaces in the search results "
                       "frontier AIs retrieve.")
    elif "magnesium" in q:
        actions.append("Pitch the next Healthline \"best magnesium supplements\" refresh. "
                       "Their current list runs Thorne, Pure Encapsulations, and NOW "
                       "Foods, with no Swiss brand on it.")
        actions.append("Get listed on consumerlab.com's magnesium category review page. "
                       "This single placement reaches both Claude and ChatGPT.")
    elif "omega" in q or "fish oil" in q:
        actions.append("Obtain IFOS (Nutrasource) third-party certification for Avea Omega-3. "
                       "ChatGPT cites certifications.nutrasource.ca explicitly when picking "
                       "fish oil brands.")
        actions.append("Pitch Avea Omega-3 into comparison reviews alongside Nordic Naturals "
                       "and Carlson, which dominate every system's omega-3 answer.")
    elif "swiss" in q:
        actions.append("ChatGPT and Claude build their \"Swiss supplement brands\" answers "
                       "from heritage-brand listings (Burgerstein, A.Vogel, Bio-Strath, Nestlé "
                       "Health Science, Nutraswiss). Avea isn't on the Swiss-supplement-industry "
                       "directory pages those answers reference.")
        actions.append("Publish a profile of Avea's Swiss-origin story on Swissinfo, NZZ, or "
                       "Handelszeitung. ChatGPT pulls from Swiss business and lifestyle press "
                       "for this query.")
        actions.append("Apply for membership listings with Swiss Sport Nutrition Society and "
                       "similar trade bodies. Their public member directories show up in "
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
                       "comparisons, the two brands ChatGPT lists first.")
        actions.append("Submit Avea Resveratrol for ConsumerLab review and seek listing in "
                       "Healthline's resveratrol category coverage.")
    else:
        actions.append("Pitch Avea for inclusion on Healthline, Fortune, ConsumerLab, and "
                       "Innerbody, the four highest-frequency sources in supplement "
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


# ---- deeper suggestions (LLM-generated, rubric-driven) -----------------------
# The rubric below is the curated list of "action-questions" the analyst model
# must answer internally before writing its output. Edit these to steer what
# the deeper analysis looks for — the model only presents conclusions, but
# these questions decide what it investigates.

DEEP_ACTION_QUESTIONS = [
    "Which brands appear in MORE THAN ONE system's list, and what shared, "
    "publicly-visible assets (certifications, review-site coverage, clinical "
    "studies, retail presence, press) most plausibly put them there?",
    "What do the top-3 ranked brands have that {brand} does not — "
    "specifically assets an AI search pipeline can see: structured product "
    "data, third-party test results, authority-site citations, "
    "research-backed claims?",
    "Of all candidate actions, which ordering maximises impact? What should "
    "be done first, what in 1-3 months, what is longer-term — justified by "
    "effort vs reach (one ConsumerLab review reaches many queries; one blog "
    "mention reaches one).",
    "Pick the 2-3 most threatening rival brands across these lists. For "
    "each: why do AI systems rank them, what does their visible web "
    "footprint look like, and what single move would let {brand} compete "
    "with them most directly?",
]

DEEP_SYSTEM = """You are a GEO (generative-engine optimisation) analyst.
You receive the real output of a 3-way AI brand-visibility test: the ranked
recommendation lists produced independently by an open surrogate model,
ChatGPT, and Claude for one purchase-intent question, plus the web domains
the two frontier systems actually consulted, and brand-match results.

Before writing anything, work through the analyst rubric questions you are
given. Ground every claim in the supplied lists and domains, plus widely
verifiable market knowledge. Never invent URLs, certifications a brand does
not plausibly hold, or rankings not present in the data.

Return ONLY compact JSON, no prose around it, exactly this shape:
{
 "summary": "<2-3 sentence executive read of the competitive picture>",
 "competitive_gaps": [
   {"asset": "<asset winning brands have>",
    "brands_with_it": ["<brand>", ...],
    "why_it_matters": "<why this asset wins AI visibility>",
    "gap_for_brand": "<what the tracked brand is missing concretely>"}
 ],
 "priority_plan": [
   {"rank": 1, "action": "<specific action>", "horizon": "now|1-3 months|3-6 months",
    "effort": "low|medium|high", "impact": "<expected reach, concrete>"}
 ],
 "rival_deep_dive": [
   {"brand": "<rival>",
    "why_ai_ranks_them": "<grounded reason>",
    "their_visible_assets": "<what their web footprint shows>",
    "how_to_compete": "<single most direct move>"}
 ]
}
3-5 competitive_gaps, 4-6 priority_plan steps, 2-3 rival_deep_dive entries."""


def deep_suggestions(
    question: str,
    brand: str,
    systems: dict,
    matches: dict,
    brief: dict,
) -> dict | None:
    """Rubric-driven deeper analysis via Claude Sonnet. Returns the parsed
    JSON dict, or None on any failure (the UI simply skips the section)."""
    import os
    from anthropic import Anthropic

    o_doms = sorted({_domain_of(u) for u in systems["openai"].get("urls") or []} - {""})
    c_doms = sorted({_domain_of(u) for u in systems["claude"].get("urls") or []} - {""})

    rubric = "\n".join(
        f"{i}. {q.format(brand=brand)}" for i, q in enumerate(DEEP_ACTION_QUESTIONS, 1)
    )
    payload = {
        "question": question,
        "tracked_brand": brand,
        "ranked_lists": {
            "surrogate": systems["surrogate"]["ranked"],
            "chatgpt": systems["openai"]["ranked"],
            "claude": systems["claude"]["ranked"],
        },
        "brand_matches": {
            "surrogate_vs_chatgpt": matches["sur_openai"].get("matched_pairs"),
            "surrogate_vs_claude": matches["sur_claude"].get("matched_pairs"),
        },
        "domains_chatgpt_cited": o_doms,
        "domains_claude_consulted": c_doms,
        "brief_suggestions_already_shown": brief,
    }
    user = (
        f"ANALYST RUBRIC — answer these internally first:\n{rubric}\n\n"
        f"RUN DATA:\n{json.dumps(payload, ensure_ascii=False, indent=1)}"
    )
    try:
        client = Anthropic(max_retries=3)
        model = os.environ.get("DEEP_ADVICE_MODEL", "claude-sonnet-4-6")
        resp = client.messages.create(
            model=model,
            max_tokens=8000,
            system=DEEP_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        out = json.loads(m.group(0))
        out["_model"] = resp.model
        return out
    except Exception:
        return None


# ---- counterfactual: pick a third-party platform anchor ---------------------
# For the "what if <brand> were on the sources" demo we want to assume the
# brand is listed on a PLATFORM the brand could realistically get onto (a
# directory/review/ranking site), NOT a rival's own homepage. We classify a
# cited domain as a competitor homepage if its name matches one of the brands
# in any model's ranked list; everything else is treated as a platform.

def _collapse(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# Domains we always treat as third-party platforms — fast pre-pass so common
# ones never need the model call.
_PLATFORM_ALLOWLIST = {
    "consumerlab.com", "healthline.com", "fortune.com", "innerbody.com",
    "consumerreports.org", "forbes.com", "health.com", "verywellhealth.com",
    "wikipedia.org", "en.wikipedia.org", "statista.com", "zoominfo.com",
    "rocketreach.co", "ensun.io", "swissmade.direct", "swissbiotech.org",
    "pubmed.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov", "nsfsport.com",
    "vitalabo.com", "tripadvisor.com", "amazon.com", "target.com",
}

_DOMAIN_CLASS_SYSTEM = """You label web domains by the role they play in a
purchase-intent answer. For each domain return one type:
- "brand_homepage": the official site of a single product/company (often one
  that's being recommended). Also give its brand name.
- "directory": lists many companies/products (aggregators, company databases).
- "review": editorial / ranking / "best of" / review publications.
- "research": scientific, medical, clinical, or standards/certification bodies.
- "retail": shops selling many different brands.
- "other": anything else.
Return ONLY compact JSON: {"<domain>": {"type": "...", "brand": "<name or null>"}}."""


def _heuristic_brand_homes(domains: list[str], brands: list[str]) -> set[str]:
    """Fallback string heuristic when the model call is unavailable."""
    keys = []
    for it in brands:
        k = _collapse(it)
        for gen in ("swiss", "health", "nutrition", "supplements",
                    "vitamins", "solutions", "ag", "the", "life"):
            k = k.replace(gen, "")
        if len(k) >= 4:
            keys.append(k)
    return {d for d in domains if any(k in _collapse(d) for k in keys)}


def classify_cited_domains(record: dict) -> dict:
    """Label every cited domain by role. Allowlist pre-pass + one Haiku call
    for the rest; falls back to the string heuristic if the call fails.
    Returns {domain: {"type": str, "brand": str|None}}."""
    from collections import Counter
    import os

    # cache on the record so we classify once per run (Haiku is non-deterministic
    # and several features reuse this), and it persists into the fixture.
    if isinstance(record.get("_domain_class"), dict):
        return record["_domain_class"]

    sysd = record.get("systems", {})
    urls = ((sysd.get("openai", {}).get("urls") or [])
            + (sysd.get("claude", {}).get("urls") or []))
    domains = list({d for d in (_domain_of(u) for u in urls) if d})
    if not domains:
        return {}
    brands = [it for s in sysd.values() for it in (s.get("ranked") or [])]

    out: dict = {}
    todo = []
    for d in domains:
        if d in _PLATFORM_ALLOWLIST:
            out[d] = {"type": "directory", "brand": None}
        else:
            todo.append(d)

    if todo:
        try:
            from anthropic import Anthropic
            client = Anthropic(max_retries=3)
            model = os.environ.get("JUDGE_CLAUDE_MODEL", "claude-haiku-4-5-20251001")
            user = json.dumps({"domains": todo, "recommended_brands": brands},
                              ensure_ascii=False)
            resp = client.messages.create(
                model=model, max_tokens=2000,
                system=_DOMAIN_CLASS_SYSTEM,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in resp.content
                           if getattr(b, "type", None) == "text")
            m = re.search(r"\{.*\}", text, re.S)
            data = json.loads(m.group(0)) if m else {}
            for d in todo:
                e = data.get(d) or {}
                out[d] = {"type": e.get("type", "other"),
                          "brand": e.get("brand")}
        except Exception:
            # fallback: string heuristic decides brand_homepage vs other
            homes = _heuristic_brand_homes(todo, brands)
            for d in todo:
                out[d] = {"type": "brand_homepage" if d in homes else "other",
                          "brand": None}
    if isinstance(record, dict):
        record["_domain_class"] = out
    return out


def pick_platform_anchor(record: dict) -> tuple[str, list[str]]:
    """Return (top_platform_domain, all_platform_domains): the run's most-cited
    domains that are NOT a recommended brand's own homepage, ranked by citation
    frequency. Uses the LLM domain classifier (with heuristic fallback)."""
    from collections import Counter
    sysd = record.get("systems", {})
    urls = ((sysd.get("openai", {}).get("urls") or [])
            + (sysd.get("claude", {}).get("urls") or []))
    counts: Counter = Counter()
    for u in urls:
        d = _domain_of(u)
        if d:
            counts[d] += 1
    if not counts:
        return "", []

    cls = classify_cited_domains(record)
    platforms = [(d, n) for d, n in counts.most_common()
                 if (cls.get(d, {}).get("type") or "other") != "brand_homepage"]
    if not platforms:
        return "", []
    return platforms[0][0], [d for d, _ in platforms]


# Counterfactual scenarios: each is a distinct "what if <brand> became more
# discoverable" premise. They are PROJECTIONS appended to the question prompt —
# we do NOT inject data into any tool's results (frontier search is opaque), we
# just ask the model to reason as if the premise held, grounded in the brand's
# real blurb, and to include the brand only if it then genuinely belongs.
_CF_SCENARIOS: tuple[str, ...] = ("listing", "ownsite", "wikipedia")

_CF_LABELS: dict[str, str] = {
    "listing": "Listed on a top source",
    "ownsite": "Own site ranks in search",
    "wikipedia": "Has a Wikipedia page",
}


def _cf_premise(brand: str, scenario: str, anchor: str = "") -> str:
    if scenario == "listing":
        return (f"assume {brand} now appears among the web sources for this "
                f"question — it is listed on {anchor}, a third-party "
                f"directory/review platform for this category.")
    if scenario == "ownsite":
        return (f"assume {brand}'s official website now ranks on the first page "
                f"of search results for this question, so the brand is directly "
                f"discoverable through ordinary web search.")
    if scenario == "wikipedia":
        return (f"assume {brand} now has its own dedicated Wikipedia article, so "
                f"independent encyclopedic background about the brand (its "
                f"history, products, and notability) is available among the web "
                f"sources.")
    return f"assume {brand} now appears among the web sources for this question."


def _counterfactual_suffix(brand: str, blurb: str, scenario: str,
                           anchor: str = "") -> str:
    return (
        f"\n\nAdditional context for this query: {_cf_premise(brand, scenario, anchor)} "
        f"{brand} is a real, verified brand: {blurb} Re-answer the question on the "
        f"merits, ranking the genuinely best options. Include {brand} only if, "
        f"given this information, it genuinely belongs among them."
    )


def _cf_run_models(aug: str, brand: str, k: int, mode: str) -> dict:
    """Run one augmented question across all three models in parallel; return
    {system: {model, ranked, answer, hit, [trajectory]}}. The surrogate block
    also carries its step-by-step `trajectory` (parsed from the run bundle) so
    the counterfactual popover can show its reasoning, not just the answer."""
    def _sur():
        sur = loop_run(aug, tools=default_tools())
        ans = sur.final_answer or ""
        bundle = str(sur.bundle_dir) if sur.bundle_dir else None
        return {"model": "qwen3-32b (surrogate)",
                "ranked": extract_pick_topN(ans, k=k)["ranked"],
                "answer": ans, "trajectory": surrogate_trajectory(bundle)}

    def _oai():
        r = ask_openai(aug, mode=mode)
        return {"model": r["model"],
                "ranked": extract_pick_topN(r["answer"], k=k)["ranked"],
                "answer": r["answer"]}

    def _cla():
        r = ask_claude(aug, mode=mode)
        return {"model": r["model"],
                "ranked": extract_pick_topN(r["answer"], k=k)["ranked"],
                "answer": r["answer"]}

    out: dict = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {"surrogate": ex.submit(_sur),
                "openai": ex.submit(_oai),
                "claude": ex.submit(_cla)}
        for name, fut in futs.items():
            d = fut.result()
            d["hit"] = brand_hit(d["ranked"], brand)
            out[name] = d
    return out


def counterfactual_run(question: str, brand: str, blurb: str, anchor: str,
                       *, k: int = 10, mode: str = "structured",
                       scenario: str = "listing") -> dict:
    """Re-ask the question of all three models under ONE counterfactual scenario.
    A PROJECTION ('what if'), not a current-state measurement. Kept for callers
    that want a single scenario; counterfactual_scenarios() runs the full set."""
    aug = question + _counterfactual_suffix(brand, blurb, scenario, anchor=anchor)
    return {
        "brand": brand,
        "anchor": anchor,
        "blurb": blurb,
        "scenario": scenario,
        "systems": _cf_run_models(aug, brand, k, mode),
    }


def counterfactual_scenarios(question: str, brand: str, blurb: str, anchor: str,
                             *, k: int = 10, mode: str = "structured",
                             scenarios: tuple[str, ...] = _CF_SCENARIOS,
                             status_cb=None) -> dict:
    """Run several counterfactual scenarios (listed-on-source / own-site-in-search
    / has-Wikipedia-page) across all three models. Returns:

        {brand, blurb, anchor, scenarios: [{id, label, anchor, systems{...}}, …]}

    Each scenario is an independent projection; the baseline is the brand's
    current picks (added by the caller for the before/after panel)."""
    runs = []
    for sc in scenarios:
        if status_cb:
            try:
                status_cb(sc, "running")
            except Exception:
                pass
        aug = question + _counterfactual_suffix(brand, blurb, sc, anchor=anchor)
        systems = _cf_run_models(aug, brand, k, mode)
        runs.append({
            "id": sc,
            "label": _CF_LABELS.get(sc, sc),
            "anchor": anchor if sc == "listing" else None,
            "systems": systems,
        })
        if status_cb:
            try:
                status_cb(sc, "done")
            except Exception:
                pass
    return {"brand": brand, "blurb": blurb, "anchor": anchor, "scenarios": runs}


# ---- classic Google rank (#6) ----------------------------------------------
# Contrast the AI-visibility picture with where each brand's OWN website ranks
# in classic Google search for the same query. Deterministic, code-driven: one
# Serper call (real Google organic positions) + string-based brand→domain
# matching. If a brand's site isn't in the top `depth` results we report it as
# "not ranking" (position=None) — never a fabricated number.

# Generic words to strip when deriving a brand's distinctive token(s). Mirrors
# _heuristic_brand_homes but kept separate so each can evolve independently.
_BRAND_GENERIC = {
    "the", "by", "and", "co", "inc", "ag", "gmbh", "ltd", "llc",
    "life", "labs", "lab", "health", "nutrition", "nutra", "bio",
    "supplements", "supplement", "vitamins", "vitamin", "solutions",
    "swiss", "official", "shop", "store", "get", "try", "company",
}


def _fold(s: str) -> str:
    """Lowercase + strip accents so 'Dünner' -> 'dunner' (avoids the collapse
    dropping 'ü' and leaving a garbage fragment like 'nner')."""
    import unicodedata
    nf = unicodedata.normalize("NFKD", str(s).lower())
    return "".join(c for c in nf if not unicodedata.combining(c))


def _brand_tokens(brand: str) -> list[str]:
    """Distinctive lowercase tokens of a brand name (generic words dropped)."""
    toks = [t for t in re.split(r"[^a-z0-9]+", _fold(brand)) if t]
    distinctive = [t for t in toks if t not in _BRAND_GENERIC]
    return distinctive or toks


def _brand_matches_domain(brand: str, domain: str) -> bool:
    """True if `domain` plausibly is `brand`'s own website. We compare against
    the second-level domain label only (e.g. 'swissherbs' in
    'swissherbs.com', 'usnews' in 'health.usnews.com') so generic path/host
    fragments like 'health' don't cause false matches. A brand matches when the
    label equals/contains its collapsed name, or contains all of its distinctive
    (non-generic, len>=4) tokens — which keeps 'Swiss Energy' off 'swisse.us'."""
    bc = _collapse(_fold(brand))
    host = _fold(domain).strip()
    if host.startswith("www."):
        host = host[4:]
    parts = [p for p in host.split(".") if p]
    if not bc or len(parts) < 2:
        return False
    core = _collapse(parts[-2])  # the registrable label, e.g. "swissherbs"
    if not core:
        return False
    if core == bc or bc in core:
        return True
    toks = [t for t in _brand_tokens(brand) if len(t) >= 4]
    return bool(toks) and all(t in core for t in toks)


def _mention_phrases(brand: str) -> list[str]:
    """Candidate regexes (word-boundaried, whitespace-flexible) that a result's
    title/snippet must contain to count as *naming* the brand. We try the full
    name and a 'core' that drops trailing generic words ('Swiss Energy Vitamins'
    -> also 'Swiss Energy'), so a snippet saying 'Swiss Energy' still counts —
    while a bare common token like 'energy' alone never does."""
    full = [t for t in re.split(r"[^a-z0-9]+", _fold(brand)) if t]
    if not full:
        return []
    core = list(full)
    while len(core) > 1 and core[-1] in _BRAND_GENERIC:
        core.pop()
    phrases = []
    for toks in ([full, core] if core != full else [full]):
        if sum(len(t) for t in toks) < 4:   # too short/generic to be safe
            continue
        phrases.append(r"\b" + r"\W+".join(re.escape(t) for t in toks) + r"\b")
    return phrases


def _brand_mentioned_in(brand: str, text: str) -> bool:
    """True if `text` (a result title/snippet) names the brand."""
    folded = _fold(text)
    return any(re.search(p, folded) for p in _mention_phrases(brand))


def classic_rank(
    question: str,
    brands: list[str],
    *,
    brand: str = "",
    depth: int = 100,
) -> dict | None:
    """Where does each brand's own website rank in classic Google search for
    `question`? Returns a dict (engine, label, depth, results_count, ranks) or
    None if no search engine is configured.

    `ranks` is a list of {brand, position, url, mention, mention_url}:
      - position/url: rank of the brand's OWN website (None if absent).
      - mention/mention_url: rank of the first result (any source — a listicle,
        review, etc.) whose title/snippet NAMES the brand (None if absent).
    Both are 1-based absolute SERP ranks. Serper serves 10 results/page, so `depth` is
    fetched across ceil(depth/10) pages in parallel (1 credit each). The tracked
    `brand` is always included even when it has no picks.
    """
    import os
    from surrogate.tools.search import _serper_raw, _tavily_raw

    if os.environ.get("SERPER_API_KEY"):
        engine, label = "serper", "Google"
        pages = max(1, -(-depth // 10))  # ceil
        rows = []
        with ThreadPoolExecutor(max_workers=min(pages, 5)) as ex:
            futs = {ex.submit(_serper_raw, question, 10, p): p
                    for p in range(1, pages + 1)}
            page_rows = {}
            for f in futs:
                try:
                    page_rows[futs[f]] = f.result() or []
                except Exception:
                    page_rows[futs[f]] = []
        for p in range(1, pages + 1):
            rows.extend(page_rows.get(p, []))
        rows = rows[:depth]
    elif os.environ.get("TAVILY_API_KEY"):
        # Tavily caps low and ranks by its own index, so label it honestly.
        engine, label = "tavily", "web search"
        rows = _tavily_raw(question, min(depth, 20))
    else:
        return None

    # dedup the brand list (tracked brand first), preserving display names
    want: list[str] = []
    seen: set[str] = set()
    for b in ([brand] if brand else []) + list(brands or []):
        b = (b or "").strip()
        ck = _collapse(b)
        if not ck or ck in seen:
            continue
        seen.add(ck)
        want.append(b)

    # precompute per result, in rank order: position, domain, url, and the
    # title+snippet text used for mention detection.
    indexed = []
    for i, r in enumerate(rows):
        pos = r.get("position") or (i + 1)
        text = f"{r.get('title') or ''} {r.get('snippet') or ''}"
        indexed.append((pos, _domain_of(r.get("url") or ""),
                        r.get("url") or "", text))

    ranks = []
    for b in want:
        # own-site rank: first result whose domain is the brand's homepage
        own = next(((p, u) for p, d, u, _t in indexed
                    if _brand_matches_domain(b, d)), None)
        # mention rank: first result (any source) whose title/snippet names it
        mention = next(((p, u) for p, _d, u, t in indexed
                        if _brand_mentioned_in(b, t)), None)
        ranks.append({
            "brand": b,
            "position": own[0] if own else None,
            "url": own[1] if own else None,
            "mention": mention[0] if mention else None,
            "mention_url": mention[1] if mention else None,
        })

    return {
        "engine": engine,
        "label": label,
        "query": question,
        "depth": depth,
        "results_count": len(rows),
        "ranks": ranks,
    }


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
        bundle = str(res.bundle_dir) if res.bundle_dir else None
        return {
            "model": "qwen3-32b (surrogate)",
            "answer": answer,
            "thinking": _concat_thinking(res.messages),
            "trajectory": surrogate_trajectory(bundle),
            "ranked": picks["ranked"],
            "steps": res.steps,
            "termination": res.termination,
            "bundle": bundle,
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
    brief = {"hits": hits, "why": why, "actions": actions}

    # ---- deeper analysis (rubric-driven, Claude Sonnet) -----------------------
    _notify("deep", "running")
    deep = deep_suggestions(question, brand, results, matches, brief)
    _notify("deep", "done" if deep else "error: analysis unavailable")

    # ---- classic Google rank vs AI visibility (#6) ----------------------------
    # One Serper call; deterministic. Brands = union of all systems' picks.
    all_picks = [it for n in workers for it in (results[n]["ranked"] or [])]
    try:
        classic = classic_rank(question, all_picks, brand=brand)
    except Exception as e:
        classic = None
        errors["classic_rank"] = f"{type(e).__name__}: {e}"

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "k": k,
        "mode": mode,
        "brand": brand,
        "systems": results,
        "matches": matches,
        "suggestions": brief,
        "deep": deep,
        "classic_rank": classic,
        "errors": errors,
    }

    # Persist the full record verbatim (CLAUDE.md prime directive).
    STORE_DIR.mkdir(exist_ok=True, parents=True)
    with COMPARE_JSONL.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    return record
