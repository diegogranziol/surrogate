"""Multi-engine web search with fan-out + URL dedup.

Default behaviour (WEB_SEARCH_MODE=multi, the default): query Tavily and
DuckDuckGo in parallel, merge results, dedup by URL, interleave so each
engine gets fair representation in the top of the list the LLM sees. This
broadens the evidence pool — Tavily indexes Google-class sources, DDG
indexes Bing-class sources, so the surrogate sees URLs the frontiers see
on either side.

Backward-compat: WEB_SEARCH_MODE=single picks the highest-priority engine,
matching the old behaviour (SerpAPI > Tavily > DDG).

Each engine adapter returns `list[dict]` with keys {title, url, snippet}.
The public `web_search()` returns the same numbered-list string format the
rest of the loop expects.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse


# ---- trusted-domain authority filter ---------------------------------------
# Loaded from surrogate/tools/trusted_domains.json (built by
# scripts/build_frontier_sources_report.py / mine_frontier_domains.py).
# Per-category whitelist of domains the frontiers actually cite. We use it
# to re-rank multi-engine merged results so the surrogate sees the same
# authoritative sources first.

_TRUSTED_CACHE: dict | None = None


def _load_trusted_domains() -> dict[str, list[str]]:
    global _TRUSTED_CACHE
    if _TRUSTED_CACHE is None:
        path = Path(__file__).parent / "trusted_domains.json"
        if path.exists():
            try:
                _TRUSTED_CACHE = json.loads(path.read_text())
            except Exception:
                _TRUSTED_CACHE = {}
        else:
            _TRUSTED_CACHE = {}
    return _TRUSTED_CACHE


_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Order matters: rules with more specific keywords come first so
    # "headphone" doesn't get caught by the "phone" rule below.
    ("audio",      ("headphone", "earbud", "noise-cancel", "noise cancel",
                    "wireless audio", "speaker", "soundbar")),
    ("supplements", ("supplement", "nmn", "nad+", "spermidine", "collagen",
                     "longevity", "healthy aging", "vitamin", "magnesium",
                     "omega-3", "resveratrol")),
    ("places",     ("restaurant", "italian", "tashkent", "dubai", "hotel",
                    "ramen", "bar ", "pizzeria")),
    ("appliances", ("espresso", "coffee maker", "blender", "stand mixer",
                    "vacuum", "washing machine")),
    ("phones",     ("phone", "smartphone", "iphone", "android")),
)


def _infer_category(query: str) -> str:
    q = (query or "").lower()
    for name, kws in _CATEGORY_RULES:
        if any(k in q for k in kws):
            return name
    return "other"


def _domain_of(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def _rerank_by_authority(items: list[dict], trusted: list[str]) -> list[dict]:
    """Move trusted-domain results to the top; preserve relative order in each
    group. Tag the trusted ones with `trusted=True` so the formatter can mark
    them visibly in the LLM's view."""
    if not trusted:
        return items
    trusted_set = {d.lower() for d in trusted}
    boosted: list[dict] = []
    rest: list[dict] = []
    for r in items:
        d = _domain_of(r.get("url", ""))
        if d in trusted_set:
            r = dict(r); r["trusted"] = True
            boosted.append(r)
        else:
            rest.append(r)
    return boosted + rest


# ---- shared format & dedup -------------------------------------------------

def _format_results(items: list[dict]) -> str:
    """Render a list of result dicts as the numbered text block the LLM reads."""
    if not items:
        return "(no results)"
    lines = []
    for i, r in enumerate(items, 1):
        title = (r.get("title") or "(no title)").strip()
        url = (r.get("url") or "").strip()
        snippet = (r.get("snippet") or "").strip().replace("\n", " ")
        engine = r.get("engine")
        trusted = r.get("trusted", False)
        parts = []
        if engine:
            parts.append(f"via {engine}")
        if trusted:
            parts.append("trusted-source")
        tag = f" [{', '.join(parts)}]" if parts else ""
        lines.append(f"{i}. {title} — {url}{tag}\n   {snippet}")
    return "\n".join(lines)


def _normalize_url(u: str) -> str:
    """Strip protocol-trail noise for dedup."""
    u = (u or "").strip().rstrip("/").lower()
    # protocol-insensitive comparison
    for p in ("https://", "http://"):
        if u.startswith(p):
            u = u[len(p):]
            break
    if u.startswith("www."):
        u = u[4:]
    return u


def _dedup_and_interleave(per_engine: dict[str, list[dict]]) -> list[dict]:
    """Take {engine_name: [results]} and produce a single list. Round-robin
    across engines (engine A's #1, engine B's #1, engine A's #2, …) then
    dedup by normalised URL, keeping the first occurrence (round-robin order
    ensures every engine gets one shot near the top)."""
    if not per_engine:
        return []
    seen: set[str] = set()
    out: list[dict] = []
    # Sort engines by name for stable order (mainly for tests).
    engines = sorted(per_engine.keys())
    max_len = max((len(per_engine[e]) for e in engines), default=0)
    for i in range(max_len):
        for engine in engines:
            items = per_engine[engine]
            if i >= len(items):
                continue
            r = items[i]
            key = _normalize_url(r.get("url", ""))
            if not key or key in seen:
                continue
            seen.add(key)
            # Tag the engine of origin for visibility in the trace.
            tagged = dict(r)
            tagged["engine"] = engine
            out.append(tagged)
    return out


# ---- public entry ----------------------------------------------------------

def web_search(query: str, max_results: int = 5) -> str:
    """Search the web. Returns a numbered list of `title — url — snippet` lines.

    `max_results` is per-engine in multi mode; the merged list is typically
    1.3–1.7× larger after dedup.
    """
    max_results = max(1, min(int(max_results or 5), 10))
    mode = os.environ.get("WEB_SEARCH_MODE", "multi").lower()

    if mode == "single":
        # Old behaviour: pick the highest-priority engine.
        if os.environ.get("SERPAPI_API_KEY"):
            return _format_results(_serpapi_raw(query, max_results))
        if os.environ.get("TAVILY_API_KEY"):
            return _format_results(_tavily_raw(query, max_results))
        return _format_results(_ddg_raw(query, max_results))

    # multi mode: fan-out, merge, dedup, interleave.
    engines: list[tuple[str, callable]] = []
    if os.environ.get("TAVILY_API_KEY"):
        engines.append(("tavily", _tavily_raw))
    if os.environ.get("SERPAPI_API_KEY"):
        engines.append(("serpapi", _serpapi_raw))
    # DDG is always available (no key needed).
    engines.append(("ddg", _ddg_raw))

    per_engine: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=len(engines)) as ex:
        futures = {ex.submit(fn, query, max_results): name for name, fn in engines}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                per_engine[name] = fut.result() or []
            except Exception as e:
                # One engine failing is fine; we keep the others.
                per_engine[name] = []
                # Note: we don't print here; the LLM doesn't need engine errors
                # in its context. The bundle's trace will show the empty result.

    merged = _dedup_and_interleave(per_engine)

    # Authority filter: re-rank so trusted-domain results come first. Toggle
    # via WEB_SEARCH_DOMAIN_BIAS env (default on). Detects category from the
    # query text; falls through to no-op for unknown categories.
    if os.environ.get("WEB_SEARCH_DOMAIN_BIAS", "on").lower() == "on":
        cat = _infer_category(query)
        trusted = _load_trusted_domains().get(cat, [])
        if trusted:
            merged = _rerank_by_authority(merged, trusted)

    if not merged:
        return f"(no results for {query!r})"
    return _format_results(merged)


# ---- per-engine adapters (return list[dict]) -------------------------------

def _ddg_raw(query: str, n: int) -> list[dict]:
    """DuckDuckGo (no key). Mostly Bing-derived results."""
    from ddgs import DDGS
    region = os.environ.get("SURROGATE_DDG_REGION", "uk-en")
    items: list[dict] = []
    for r in DDGS().text(query, max_results=n, region=region):
        items.append({
            "title": r.get("title") or "",
            "url": r.get("href") or r.get("url") or "",
            "snippet": (r.get("body") or "").strip().replace("\n", " "),
        })
    return items


def _tavily_raw(query: str, n: int) -> list[dict]:
    """Tavily (Google-class). Needs TAVILY_API_KEY."""
    import httpx
    r = httpx.post(
        "https://api.tavily.com/search",
        json={
            "api_key": os.environ["TAVILY_API_KEY"],
            "query": query,
            "max_results": n,
            "search_depth": "basic",
        },
        timeout=20.0,
    )
    r.raise_for_status()
    data = r.json()
    items = []
    for res in data.get("results") or []:
        items.append({
            "title": res.get("title") or "",
            "url": res.get("url") or "",
            "snippet": (res.get("content") or "").strip().replace("\n", " "),
        })
    return items


def _serpapi_raw(query: str, n: int) -> list[dict]:
    """SerpAPI Google engine. Needs SERPAPI_API_KEY."""
    import httpx
    r = httpx.get(
        "https://serpapi.com/search.json",
        params={
            "engine": "google",
            "q": query,
            "num": n,
            "api_key": os.environ["SERPAPI_API_KEY"],
            "hl": os.environ.get("SERPAPI_HL", "en"),
            "gl": os.environ.get("SERPAPI_GL", "us"),
        },
        timeout=20.0,
    )
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        return []
    items = []
    for res in (data.get("organic_results") or [])[:n]:
        items.append({
            "title": res.get("title") or "",
            "url": res.get("link") or "",
            "snippet": (res.get("snippet") or "").strip().replace("\n", " "),
        })
    return items
