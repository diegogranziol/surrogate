"""Mine frontier-cited URLs from h2h-store.jsonl and build per-category
domain frequency tables.

For each entry:
- Claude entries: frontier.tool_calls has tool_result blocks with content
  listing every URL returned by Anthropic's web_search.
- OpenAI entries: frontier.blocks_raw has message.content[*].annotations
  with url_citation entries (the URLs gpt-5 actually cited).

We extract domains, group by question category (heuristic), and rank by
frequency. The top domains per category become our "trusted sources" list.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def kind(e: dict) -> str:
    m = e.get("frontier", {}).get("model", "") or ""
    if "gpt" in m:
        return "openai"
    if "claude" in m:
        return "claude"
    return "other"


def category(q: str) -> str:
    """Heuristic category. Order matters — more specific rules first so
    'headphone' isn't caught by the broader 'phone' rule."""
    q = q.lower()
    if any(k in q for k in ("headphone", "earbud", "noise-cancel", "noise cancel")):
        return "audio"
    if any(k in q for k in ("supplement", "nmn", "nad+", "spermidine", "collagen",
                            "longevity", "healthy aging", "vitamin")):
        return "supplements"
    if any(k in q for k in ("restaurant", "italian", "tashkent", "dubai", "hotel")):
        return "places"
    if "espresso" in q:
        return "appliances"
    if any(k in q for k in ("phone", "smartphone")):
        return "phones"
    return "other"


def domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


# ---- URL extraction by frontier ----------------------------------------------

def claude_urls(entry: dict) -> list[str]:
    out = []
    for tc in entry.get("frontier", {}).get("tool_calls") or []:
        if tc.get("kind") != "tool_result":
            continue
        content = tc.get("content") or []
        for item in content if isinstance(content, list) else []:
            if isinstance(item, dict) and item.get("type") == "web_search_result":
                u = item.get("url")
                if u:
                    out.append(u)
    return out


def openai_urls(entry: dict) -> list[str]:
    out = []
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
                        out.append(u)
    return out


# ---- Main --------------------------------------------------------------------

def main():
    by_cat: dict[tuple, Counter] = {}

    # Source 1: h2h-store.jsonl — every benchmark entry.
    store_path = ROOT / "backtests/h2h-store.jsonl"
    if store_path.exists():
        entries = [json.loads(l) for l in store_path.read_text().splitlines() if l.strip()]
        print(f"h2h-store entries: {len(entries)}")
        for e in entries:
            fr = kind(e)
            if fr == "other":
                continue
            cat = category(e.get("question", ""))
            urls = claude_urls(e) if fr == "claude" else openai_urls(e)
            urls = [re.sub(r"[?&]utm_source=openai", "", u) for u in urls]
            domains = [d for d in (domain(u) for u in urls) if d]
            by_cat.setdefault((fr, cat), Counter()).update(domains)

    # Source 2: frontier_mining.jsonl — frontier-only mining runs (no surrogate).
    mining_path = ROOT / "backtests/frontier_mining.jsonl"
    if mining_path.exists():
        mining = [json.loads(l) for l in mining_path.read_text().splitlines() if l.strip()]
        print(f"frontier_mining entries: {len(mining)}\n")
        for m in mining:
            fr = m.get("frontier")
            if fr not in ("claude", "openai"):
                continue
            cat = category(m.get("question", ""))
            urls = m.get("urls") or []
            urls = [re.sub(r"[?&]utm_source=openai", "", u) for u in urls]
            domains = [d for d in (domain(u) for u in urls) if d]
            by_cat.setdefault((fr, cat), Counter()).update(domains)
    else:
        print()

    # Print top domains per (frontier, category)
    for (fr, cat), counter in sorted(by_cat.items()):
        top = counter.most_common(10)
        total = sum(counter.values())
        print(f"### {fr} / {cat}  ({total} citations, {len(counter)} unique domains)")
        for dom, n in top:
            print(f"  {n:3d}  {dom}")
        print()

    # Also: a flat trusted-domain whitelist by category (union of frontiers).
    print("=== Combined trusted-domain whitelist by category ===\n")
    cats: dict[str, Counter] = {}
    for (fr, cat), counter in by_cat.items():
        cats.setdefault(cat, Counter()).update(counter)
    out_table = {}
    for cat, counter in cats.items():
        # Take domains that appear at least 2x — drops one-off noise.
        whitelist = [d for d, n in counter.most_common(15) if n >= 2]
        out_table[cat] = whitelist
        print(f"  {cat:12s}: {whitelist}")

    out_path = ROOT / "surrogate/tools/trusted_domains.json"
    out_path.write_text(json.dumps(out_table, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
