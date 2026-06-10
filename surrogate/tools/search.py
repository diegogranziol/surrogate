from __future__ import annotations

import os


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web. Precedence: SerpAPI (Google) > Tavily > DuckDuckGo.

    Returns a numbered list of `title — url — snippet` lines.
    """
    max_results = max(1, min(int(max_results or 5), 10))
    if os.environ.get("SERPAPI_API_KEY"):
        return _serpapi(query, max_results)
    if os.environ.get("TAVILY_API_KEY"):
        return _tavily(query, max_results)
    return _ddg(query, max_results)


def _serpapi(query: str, n: int) -> str:
    """Google search via SerpAPI. Free tier: 100 searches/month."""
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
        return f"(serpapi error: {data['error']})"
    results = data.get("organic_results") or []
    if not results:
        return f"(no results for {query!r})"
    lines = []
    for i, res in enumerate(results[:n], 1):
        title = res.get("title") or "(no title)"
        url = res.get("link") or ""
        snippet = (res.get("snippet") or "").strip().replace("\n", " ")
        lines.append(f"{i}. {title} — {url}\n   {snippet}")
    return "\n".join(lines)


def _ddg(query: str, n: int) -> str:
    from ddgs import DDGS

    region = os.environ.get("SURROGATE_DDG_REGION", "uk-en")
    results = list(DDGS().text(query, max_results=n, region=region))
    if not results:
        return f"(no results for {query!r})"
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title") or "(no title)"
        url = r.get("href") or r.get("url") or ""
        body = (r.get("body") or "").strip().replace("\n", " ")
        lines.append(f"{i}. {title} — {url}\n   {body}")
    return "\n".join(lines)


def _tavily(query: str, n: int) -> str:
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
    results = data.get("results") or []
    if not results:
        return f"(no results for {query!r})"
    lines = []
    for i, res in enumerate(results, 1):
        title = res.get("title") or "(no title)"
        url = res.get("url") or ""
        content = (res.get("content") or "").strip().replace("\n", " ")
        lines.append(f"{i}. {title} — {url}\n   {content}")
    return "\n".join(lines)
