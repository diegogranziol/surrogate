"""Build a presentation-quality 'frontier sources' report.

Shows the client which URLs and domains frontier AI systems (Claude,
ChatGPT) consult when answering purchase-intent questions. This is the
visible-first-stage artifact: the retrieval layer the frontier hides
from end users.

Reads `backtests/h2h-store.jsonl` and produces
`backtests/frontier-sources.md`.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "backtests/h2h-store.jsonl"
OUT = ROOT / "backtests/frontier-sources.md"


def kind(e: dict) -> str | None:
    m = e.get("frontier", {}).get("model", "") or ""
    if "gpt" in m:
        return "openai"
    if "claude" in m:
        return "claude"
    return None


_CATEGORY_RULES = [
    ("supplements", ("supplement", "nmn", "nad+", "spermidine", "collagen",
                     "longevity", "healthy aging", "vitamin")),
    ("phones",      ("phone", "smartphone")),
    ("audio",       ("headphone", "earbud", "noise-cancel")),
    ("places",      ("restaurant", "italian", "tashkent", "dubai", "hotel")),
    ("appliances",  ("espresso", "machine")),
]


def category(q: str) -> str:
    q = q.lower()
    for name, kws in _CATEGORY_RULES:
        if any(k in q for k in kws):
            return name
    return "other"


def domain_of(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def _strip_openai_tracker(url: str) -> str:
    return re.sub(r"[?&]utm_source=openai", "", url)


# ---- URL extraction per frontier --------------------------------------------

def claude_urls(entry: dict) -> list[str]:
    out: list[str] = []
    for tc in entry.get("frontier", {}).get("tool_calls") or []:
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


def openai_urls(entry: dict) -> list[str]:
    """OpenAI exposes cited URLs in `message.content[*].annotations[*].url`."""
    out: list[str] = []
    for blk in entry.get("frontier", {}).get("blocks_raw") or []:
        if not isinstance(blk, dict) or blk.get("type") != "message":
            continue
        for c in blk.get("content") or []:
            if not isinstance(c, dict):
                continue
            for a in c.get("annotations") or []:
                if isinstance(a, dict) and a.get("type") == "url_citation":
                    u = a.get("url")
                    if u:
                        out.append(_strip_openai_tracker(u))
    return out


def openai_queries(entry: dict) -> list[str]:
    """gpt-5 exposes its actual search queries in `web_search_call.action.queries`."""
    out: list[str] = []
    for tc in entry.get("frontier", {}).get("tool_calls") or []:
        action = tc.get("action") or {}
        qs = action.get("queries") or []
        for q in qs if isinstance(qs, list) else []:
            if isinstance(q, str):
                out.append(q)
        single = action.get("query")
        if isinstance(single, str):
            out.append(single)
    return out


# ---- Aggregations -----------------------------------------------------------

def latest_per_question(entries: list[dict], fr_kind: str) -> dict[str, dict]:
    """Keep only the most recent entry per question for one frontier."""
    out: dict[str, dict] = {}
    for e in entries:
        if kind(e) != fr_kind:
            continue
        q = (e.get("question") or "").strip()
        if not q:
            continue
        prev = out.get(q)
        if prev is None or e.get("ts", "") > prev.get("ts", ""):
            out[q] = e
    return out


# ---- Rendering --------------------------------------------------------------

def render_report() -> str:
    entries = [json.loads(l) for l in STORE.read_text().splitlines() if l.strip()]
    claude_by_q = latest_per_question(entries, "claude")
    openai_by_q = latest_per_question(entries, "openai")
    questions = sorted(set(claude_by_q) | set(openai_by_q))

    # 1. global per-category domain frequency
    cat_domains: dict[str, dict[str, Counter]] = defaultdict(lambda: {"claude": Counter(), "openai": Counter()})
    for q in questions:
        cat = category(q)
        c = claude_by_q.get(q)
        o = openai_by_q.get(q)
        if c:
            for u in claude_urls(c):
                d = domain_of(u)
                if d:
                    cat_domains[cat]["claude"][d] += 1
        if o:
            for u in openai_urls(o):
                d = domain_of(u)
                if d:
                    cat_domains[cat]["openai"][d] += 1

    out: list[str] = []
    out.append("# What sources do frontier AI systems consult?")
    out.append("")
    out.append("_This report shows the URLs and domains Anthropic's Claude (with "
               "`web_search`) and OpenAI's gpt-5 (Responses API + `web_search`) "
               "actually visit when answering purchase-intent questions. "
               "Captured from real API calls in the head-to-head benchmark. "
               "For each question we record every URL returned to / cited by "
               "the model — the retrieval layer that's normally hidden from end users._")
    out.append("")

    # ---- Section A: trusted domains by category -----------------------------
    out.append("## 1. Trusted domain hierarchy by category")
    out.append("")
    out.append("Per question category, the domains most frequently cited or "
               "consulted across the benchmark. **Both frontiers** column "
               "= domains that appear in BOTH Claude's and gpt-5's source set "
               "(the strongest GEO targets — these are the sites you must be "
               "mentioned on to appear in AI answers for the category).")
    out.append("")

    for cat in sorted(cat_domains):
        c_counter = cat_domains[cat]["claude"]
        o_counter = cat_domains[cat]["openai"]
        all_doms = set(c_counter) | set(o_counter)
        both_doms = sorted(set(c_counter) & set(o_counter),
                           key=lambda d: -(c_counter[d] + o_counter[d]))
        claude_only = sorted(set(c_counter) - set(o_counter),
                             key=lambda d: -c_counter[d])
        openai_only = sorted(set(o_counter) - set(c_counter),
                             key=lambda d: -o_counter[d])

        out.append(f"### {cat}")
        out.append("")
        out.append(f"_Total citations: Claude {sum(c_counter.values())}, "
                   f"gpt-5 {sum(o_counter.values())}. "
                   f"Unique domains: Claude {len(c_counter)}, gpt-5 {len(o_counter)}._")
        out.append("")
        if both_doms:
            out.append("**Cited by BOTH frontiers** (high-priority GEO targets):")
            out.append("")
            out.append("| Domain | Claude citations | gpt-5 citations | Total |")
            out.append("|---|---|---|---|")
            for d in both_doms[:15]:
                cc = c_counter[d]; oc = o_counter[d]
                out.append(f"| `{d}` | {cc} | {oc} | **{cc+oc}** |")
            out.append("")
        if claude_only:
            out.append(f"**Claude-only sources** (top {min(10, len(claude_only))}):")
            out.append("")
            out.append("| Domain | Claude citations |")
            out.append("|---|---|")
            for d in claude_only[:10]:
                out.append(f"| `{d}` | {c_counter[d]} |")
            out.append("")
        if openai_only:
            out.append(f"**gpt-5-only sources** (top {min(10, len(openai_only))}):")
            out.append("")
            out.append("| Domain | gpt-5 citations |")
            out.append("|---|---|")
            for d in openai_only[:10]:
                out.append(f"| `{d}` | {o_counter[d]} |")
            out.append("")

    # ---- Section B: per-question source lists --------------------------------
    out.append("## 2. Per-question source lists (verbatim URLs)")
    out.append("")
    out.append("For each benchmark question, the exact URLs each frontier model "
               "consulted. This is what 'first-stage retrieval' looks like under "
               "the hood. URLs are deduplicated; gpt-5's `utm_source=openai` "
               "tracker is stripped.")
    out.append("")
    for q in questions:
        c = claude_by_q.get(q)
        o = openai_by_q.get(q)
        out.append(f"### {q}")
        out.append("")
        out.append(f"_Category: **{category(q)}**_")
        out.append("")

        if o:
            qs = openai_queries(o)
            if qs:
                out.append("**gpt-5 actually ran these search queries:**")
                out.append("")
                for qstr in qs[:20]:
                    out.append(f"- `{qstr}`")
                out.append("")

        c_urls = sorted(set(claude_urls(c) or [])) if c else []
        o_urls = sorted(set(openai_urls(o) or [])) if o else []

        if c_urls:
            out.append(f"**Claude consulted {len(c_urls)} URL(s):**")
            out.append("")
            for u in c_urls:
                out.append(f"- {u}")
            out.append("")
        else:
            out.append("_(Claude: no URLs recorded for this question)_")
            out.append("")

        if o_urls:
            out.append(f"**gpt-5 cited {len(o_urls)} URL(s):**")
            out.append("")
            for u in o_urls:
                out.append(f"- {u}")
            out.append("")
        else:
            out.append("_(gpt-5: no URLs recorded for this question)_")
            out.append("")

        out.append("---")
        out.append("")

    # ---- Section C: source-overlap signal -----------------------------------
    out.append("## 3. Aggregate source overlap")
    out.append("")
    overall_c = Counter()
    overall_o = Counter()
    for cat_data in cat_domains.values():
        overall_c.update(cat_data["claude"])
        overall_o.update(cat_data["openai"])
    both = set(overall_c) & set(overall_o)
    c_only = set(overall_c) - set(overall_o)
    o_only = set(overall_o) - set(overall_c)
    out.append(f"- **Domains cited by BOTH frontiers:** {len(both)} "
               f"(of {len(overall_c | overall_o)} unique total — "
               f"{len(both) / max(len(overall_c | overall_o), 1):.0%})")
    out.append(f"- **Claude-only:** {len(c_only)}")
    out.append(f"- **gpt-5-only:** {len(o_only)}")
    out.append("")
    out.append("_The two frontiers cite mostly different webs. If a brand "
               "wants visibility across AI surfaces, it needs presence on "
               "the high-overlap domains in section 1, plus targeted "
               "presence on the frontier-specific lists._")
    out.append("")

    return "\n".join(out)


def main() -> int:
    out = render_report()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(out)
    print(f"Wrote {OUT} ({len(out):,} chars, {out.count(chr(10)):,} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
