# Agent C — Tool layer (full report)

**Date:** 2026-05-29
**Companion file:** `2026-05-29_workflow_synthesis.md` (the integrated golden findings)
**Brief:** Surveyed seven open-source browser/crawl/extraction tools (browser-use, Firecrawl, Crawl4AI, Playwright-MCP, extruct, trafilatura, SearXNG) to decide which to adopt, which to learn from, and which to skip for our `fetch_url` / `extract_entity` / `search` tools.

---

# Tool-Layer Research Report: Browser/Crawl/Extraction Primitives for the Surrogate Project

## 1. Executive Summary (the TL;DR)

- **Adopt extruct** as a third extraction tier inside `fetch_url` (or as a new `extract_entity` tool). It's a tiny, BSD-3 dependency that gives us JSON-LD/microdata/OpenGraph parsing — exactly what's missing today, and the dominant signal source on review/product pages (TripAdvisor, Amazon, Yelp, IMDB all ship rich JSON-LD).
- **Adopt Crawl4AI as an opt-in fallback** ("hard mode" branch) for pages where `curl_cffi + trafilatura` returns <500 chars or where the URL is on a known SPA-list (tripadvisor, expedia, yelp, etc.). It's Apache-2.0, self-hosted, Playwright-backed, and has built-in undetected-browser/stealth.
- **Lift these patterns without depending on them**: Playwright-MCP's *accessibility-tree snapshot* (better LLM-friendly than raw HTML); Crawl4AI's *PruningContentFilter* concept (token-budget filter before LLM sees text); Firecrawl's *output-format menu* (markdown + json + summary in one call); SearXNG's *JSON-API + format=json* pattern (a smart upgrade target for our DDGS-only `web_search`).
- **Skip**: browser-use (LLM-as-driver is the wrong abstraction for our ReAct loop where Qwen3-8B *is* the driver), Firecrawl (AGPL + cloud-best, the OSS version loses the moat we'd actually want), Playwright-MCP (great for Claude Desktop, wrong runtime model for vLLM-hosted Qwen).
- **Net change**: extruct + a `render=true` flag on `fetch_url` that switches to a Crawl4AI subprocess, plus a SearXNG container option for `web_search`. No browser-use, no Firecrawl.

---

## 2. Per-Tool Analysis

### 2.1 browser-use (`browser-use/browser-use`)

**Verbs exposed to the LLM:** the default controller registers a high-level action set: `done`, `click_element`, `input_text`, `scroll`, `scroll_to_text`, `switch_tab`, `open_tab`, `close_tab`, `go_back`, `extract_content`, `send_keys`, `save_pdf`, `wait`, `drag_drop`. It's essentially "Playwright wrapped in named actions the LLM picks from."

**JS handling:** Full Playwright (Chromium by default). Real browser rendering.

**Output to LLM:** A DOM-tree summary (elements indexed with `[1]`, `[2]`...) plus optional screenshot. The agent's loop is: snapshot → LLM picks action → execute → snapshot again. Each step is ~2–5s wall-clock.

**Cost/speed:** ~1.5–4s per agent step, depending on page weight + LLM latency. An end-to-end task is usually 10–30s.

**License + self-host:** MIT, fully self-hostable, Dockerfile present. **96.2k stars**, 9.2k commits, latest release 0.12.9 (May 26).

**Anti-bot:** Their FAQ explicitly punts: "For CAPTCHA handling, you need better browser fingerprinting and proxies. Use Browser Use Cloud."

**Verdict: ❌ Skip.** browser-use's whole abstraction is *the LLM drives the browser turn-by-turn*. We already have an LLM driving a ReAct loop with text tools. Bolting browser-use under our `fetch_url` would give us a multi-step sub-agent inside a single tool call, which (a) eats tokens, (b) doubles our debugging surface, and (c) Qwen3-8B is not yet strong enough to drive a multi-action visual browser well. The right pieces to lift from them — Playwright + a DOM summarizer — exist in Crawl4AI without the agent overhead.

---

### 2.2 Firecrawl (`mendableai/firecrawl`)

**Verbs:** `/scrape`, `/crawl`, `/map`, `/search`, `/batch_scrape`, `/extract` (LLM-powered schema extraction), plus `actions` (pre-scrape click/scroll), `/interact`, `/agent` (FIRE-1).

**JS handling:** Playwright under the hood with stealth plugins on the cloud side.

**Output:** `formats: ["markdown", "html", "rawHtml", "json", "screenshot", "links", "summary", "images", "branding"]` — one call returns whatever combination you ask for. JSON output accepts either a JSON schema or a natural-language prompt; this is the closest commercial analog to our `extract_entity`.

**Cost/speed:** ~2–4s/page on cloud. Self-hosted depends on your box. Cloud cost is 1 credit/scrape; free tier 1k credits/mo, Hobby $16/mo for 5k, Standard $83/mo for 100k.

**License + self-host:** **AGPL-3.0 core** (SDKs MIT). Self-hostable via Docker but requires Redis + Playwright + Chromium (1–2 GB RAM minimum, 500 MB images). 126k stars, very active.

**Anti-bot:** Cloud has rotating proxies + "Fire-engine" stealth; **self-hosted version is materially weaker**. Spider's benchmark put cloud Firecrawl at 95.3% success vs Crawl4AI's 89.7%, but that gap collapses for self-hosted Firecrawl.

**Verdict: ❌ Skip (but lift the output-format API design).** Three reasons: (1) AGPL-3.0 propagates if we vendor anything; we're shipping an open-source project but AGPL is sticky and will scare off any downstream wrapper. (2) Self-hosted Firecrawl loses the very anti-bot moat that would justify adopting it. (3) We'd be running Redis + Chromium + Playwright just to wrap trafilatura — same outcome as adopting Crawl4AI directly with one fewer process. **What we should steal**: the multi-format response shape (`{markdown, json, summary, links}` from a single call) is a clean ergonomic for ReAct tools.

---

### 2.3 Crawl4AI (`unclecode/crawl4ai`)

**Verbs:** Library API, not a network protocol. Core surface: `AsyncWebCrawler.arun(url, config)`, `arun_many(urls, dispatcher=...)`. Extraction strategies are pluggable: `JsonCssExtractionStrategy` (CSS-selector, LLM-free, fast), `JsonXPathExtractionStrategy`, `LLMExtractionStrategy` (Pydantic schema + OpenAI/Ollama/etc.), `RegexExtractionStrategy`. Content filters: `PruningContentFilter`, `BM25ContentFilter`, `LLMContentFilter`.

**JS handling:** Playwright. Has scroll simulation for infinite-scroll, lazy-load support, shadow-DOM flattening (v0.8.5+), session reuse, custom JS injection.

**Output:** Markdown by default (clean, with citations), structured JSON via extraction strategies, raw HTML available. `result.markdown` is the headline accessor.

**Cost/speed:** Playwright cold start + nav ~2s/page for a clean HTML site. With `LLMExtractionStrategy` + GPT-4o on top: ~25s/page in the Spider benchmark (LLM is the slow part, not Crawl4AI). With `JsonCssExtractionStrategy` it stays sub-3s.

**License + self-host:** **Apache-2.0**, 67.2k stars, 1,468 commits, v0.8.6 (recent). Self-hostable as a library *and* as a Docker-served FastAPI with JWT auth at `/dashboard`.

**Anti-bot:** Has an "undetected browser" mode (v0.8.5 added a 3-tier escalation with proxy fallback). Not as strong as commercial residential-proxy services but better than vanilla Playwright.

**Concurrency:** `MemoryAdaptiveDispatcher` (default, monitors RAM and throttles) and `SemaphoreDispatcher` (fixed concurrency cap), with built-in `RateLimiter` (random delay + exp backoff on 429/503).

**Verdict: ✅ Adopt (as opt-in fallback).** This is the right tool for the SPA hole. Apache-2.0 is friendly, the library imports cleanly, and we can keep the fast path (curl_cffi + trafilatura, ~200ms) and only spin up Playwright when we detect a thin result. Crawl4AI exposes its strategies as plain Python objects, so we don't need to swallow their whole stack — we can call `AsyncWebCrawler` ourselves and pipe the resulting markdown through our existing 4 KB cap.

---

### 2.4 Playwright MCP (`microsoft/playwright-mcp`)

**Verbs:** 70+ MCP tools across families: `browser_navigate`, `browser_click`, `browser_type`, `browser_fill_form`, `browser_hover`, `browser_drag`, `browser_press_key`, `browser_wait_for`, `browser_select_option`, `browser_evaluate`, `browser_snapshot`, `browser_take_screenshot`, `browser_console_messages`, `browser_network_requests`, `browser_tabs`, opt-in `browser_route` (request interception), storage ops, `browser_pdf_save`, locator generation, assertions.

**JS handling:** Real Playwright, headless or headed.

**Output:** Critically, **the snapshot uses Playwright's accessibility tree, not pixels.** Returns a structured, addressable view of the page where every interactable element has an ID the LLM can reference. This is meaningfully better than raw HTML for LLM consumption — fewer tokens, deterministic refs.

**Cost/speed:** ~1–3s per tool call (Playwright nav + a11y dump). Browser stays warm across calls in a session.

**License + self-host:** **Apache-2.0**, **33.2k stars**, v0.0.75 (May 7), npx + Docker.

**Anti-bot:** Vanilla Playwright — no built-in stealth. You'd need to pair with `playwright-extra` + stealth plugin.

**Verdict: ⚠️ Adapt patterns, don't adopt.** MCP is the wrong runtime contract for our system: Qwen3-8B + vLLM speaks ReAct text, not MCP-over-stdio. Wiring an MCP client into our vLLM agent loop is non-trivial and gains us nothing over calling Playwright directly. **What we lift**: the *accessibility-snapshot* concept. When/if we add a "render mode" via Crawl4AI, we should produce an a11y-style structured outline (headings + interactable elements + their refs) rather than dumping raw rendered HTML to the model. Our existing `crawl_dom` is conceptually adjacent — formalize it as an a11y-style snapshot.

---

### 2.5 extruct (`scrapinghub/extruct`)

**Verbs:** Single function. `extruct.extract(html, base_url=url, syntaxes=[...], uniform=True)`. Syntaxes: `'json-ld'`, `'microdata'`, `'microformat'`, `'rdfa'`, `'opengraph'`, `'dublincore'`. Each extractor is also available standalone (`JsonLdExtractor().extract(html)`).

**JS handling:** None — pure HTML parsing. Pipe rendered HTML in if you need post-JS data.

**Output:** Nested dict keyed by syntax. With `uniform=True`, normalizes shapes across formats. Schema.org JSON-LD on a typical product page looks like `{"@type": "Product", "name": ..., "aggregateRating": {"ratingValue": ..., "reviewCount": ...}, "offers": [...]}` — this is *literally* what `extract_entity` exists to produce.

**Cost/speed:** Pure Python parsing, ~10–50ms per page.

**License + maturity:** BSD-3, 966 stars, 25 releases, actively maintained. Lower star count is misleading — extruct is the standard structured-data extractor in the Python scraping ecosystem (it's a Scrapy/Scrapinghub project).

**Anti-bot:** N/A, parser-only.

**Verdict: ✅ Adopt now.** Smallest, cheapest, biggest signal upgrade in the whole report. Most "best X in Y" target pages — TripAdvisor listings, Yelp, Booking, Amazon, IMDB, restaurant pages, hotel pages, product reviews — embed JSON-LD with names, ratings, prices, review counts. Today we regex them out of visible text; with extruct we get them structured. Cost: one ~500-line dependency.

---

### 2.6 trafilatura (current dep) — config we likely aren't using

Looking at our current `fetch_url` (curl_cffi → trafilatura), options worth flipping on:

- `output_format='json'` (or `'markdown'`) instead of plain text — JSON gives us `title`, `author`, `date`, `description`, `text`, `comments`, `excerpt` as fields.
- `with_metadata=True` — adds title/date/author into the output. Free signal.
- `include_links=True` — keeps `<a href>` targets inline; useful for follow-up `fetch_url` decisions by the agent.
- `include_tables=True` — TripAdvisor-style comparison tables, hotel amenity tables, etc.
- `favor_precision=True` (vs. recall) — for our "best X in Y" use case we want the central content, not boilerplate. Trafilatura defaults to recall.
- `include_formatting=True` — preserves bold/italic so list items survive intact in markdown mode.
- `deduplicate=True` — drops repeated paragraphs (common on commerce pages with sidebar duplication).

**Recommended trafilatura call** for our case (sketch):

```python
trafilatura.extract(
    html,
    url=url,
    output_format="json",
    with_metadata=True,
    include_tables=True,
    include_links=True,
    include_formatting=True,
    favor_precision=True,
    deduplicate=True,
)
```

This alone will likely improve our content quality on ~30% of pages with no other change.

---

### 2.7 SearXNG (`searxng/searxng`)

**Verbs:** HTTP search API. `GET /search?q=<query>&format=json&categories=general&engines=google,bing,duckduckgo` returns federated results as JSON. Also `format=rss|csv|html`.

**JS handling:** N/A — it's a search aggregator; engines are called server-side.

**Output:** JSON with `results[]` (title, url, content snippet, engine, score), `infoboxes`, `suggestions`, `corrections`, `answers`. **The structured `engine` field per result is gold for ranking** — we can prefer Google-sourced hits, demote DDG-only ones, etc.

**Cost/speed:** ~300–900ms per query depending on which engines are enabled.

**License + self-host:** **AGPL-3.0**, 30.8k stars, very active. Official Docker image, runs in <100 MB RAM. Federates Google, Bing, DuckDuckGo, Brave, Qwant, Wikipedia, Reddit, GitHub, StackExchange, arXiv, and ~200 more — engines are plug-in YAML.

**Critical gotcha:** **JSON output is OFF by default.** You must add `json` to `search.formats` in `settings.yml` or you get HTTP 403. This trips up everyone.

**Anti-bot:** SearXNG itself isn't anti-botted — but Google/Bing rate-limit *your* SearXNG instance. With a single user (us) this is fine.

**Verdict: ✅ Adopt as `web_search` upgrade (optional).** Diego is right. Replacing/augmenting our DDGS-only path with a self-hosted SearXNG gives us (a) multi-engine federation with score-merging built in, (b) result provenance per source, (c) graceful failure when one engine rate-limits, (d) Tavily-free operation. AGPL is fine here because we'd run it as a separate Docker container that we talk to over HTTP — we're not linking its code into ours. Keep DDGS as a fallback when SearXNG is unreachable.

---

## 3. Comparison Table

| Tool | License | JS Render | Anti-bot | Self-host | Output | Per-page latency | Stars | Fit |
|---|---|---|---|---|---|---|---|---|
| browser-use | MIT | Playwright | Weak (cloud only) | Yes | DOM index + screenshot | 2–5s/step | 96.2k | Skip |
| Firecrawl | AGPL-3.0 | Playwright + stealth | Strong on cloud, weak on OSS | Yes (heavy) | md/json/html/summary | 2–4s | 126k | Skip |
| Crawl4AI | Apache-2.0 | Playwright | Medium (3-tier + undetected) | Yes (light or Docker) | Markdown + JSON | 2–3s | 67.2k | Adopt (fallback) |
| Playwright MCP | Apache-2.0 | Playwright | None | Yes | a11y snapshot | 1–3s | 33.2k | Adapt pattern |
| extruct | BSD-3 | No | N/A | Yes | JSON-LD/microdata dict | <50ms | 0.97k | Adopt now |
| trafilatura | Apache-2.0 | No | N/A | Yes | text/md/json/xml | <100ms | (current) | Tune config |
| SearXNG | AGPL-3.0 | N/A | N/A | Docker | JSON search results | 0.3–0.9s | 30.8k | Adopt for search |

---

## 4. Concrete Recommendation

### 4.1 Upgrade `fetch_url` to a three-tier extractor

Keep the fast curl_cffi path, add structured extraction, add an opt-in render fallback. Triggered automatically when the fast path returns thin content.

```python
# tools/fetch_url.py
import curl_cffi, trafilatura, extruct
from urllib.parse import urlparse

SPA_HOSTS = {"tripadvisor.com", "yelp.com", "booking.com",
             "expedia.com", "instagram.com", "tiktok.com"}

def fetch_url(url: str, render: bool = False, max_chars: int = 4000) -> dict:
    host = urlparse(url).netloc.lower().lstrip("www.")
    force_render = render or any(h in host for h in SPA_HOSTS)

    html = None
    if not force_render:
        r = curl_cffi.requests.get(url, impersonate="safari17_ios", timeout=15)
        html = r.text if r.status_code == 200 else None

    if force_render or not html or len(html) < 2000:
        html = _render_with_crawl4ai(url)   # see 4.2

    # Tier 1: structured data (cheap, high signal)
    structured = extruct.extract(
        html, base_url=url,
        syntaxes=["json-ld", "microdata", "opengraph"],
        uniform=True,
    )

    # Tier 2: article body
    article = trafilatura.extract(
        html, url=url,
        output_format="json",
        with_metadata=True,
        include_tables=True,
        include_links=True,
        include_formatting=True,
        favor_precision=True,
        deduplicate=True,
    )

    # Tier 3: BS4 visible-text fallback (existing code) if both above are empty
    body_text = _coalesce(article, html)[:max_chars]

    return {
        "url": url,
        "title": _title_from(structured, article),
        "structured": _slim_structured(structured),  # keep ratings/offers/aggregateRating
        "text": body_text,
        "rendered": force_render,
    }
```

Notes for the agent prompt: when `structured` contains `aggregateRating` or `offers`, the surrogate should prefer it over regex over `text`. This is the single biggest content-quality lever.

### 4.2 The render fallback (Crawl4AI subprocess)

Keep Crawl4AI in its own process so cold-start Playwright doesn't sit in our vLLM worker.

```python
# tools/render.py — run as subprocess; import-lazy
async def _render(url):
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
    cfg = CrawlerRunConfig(
        wait_for="css:body",
        scan_full_page=True,            # handles lazy-load / infinite scroll
        page_timeout=20_000,
        magic=True,                     # crawl4ai's stealth defaults
    )
    async with AsyncWebCrawler(headless=True) as c:
        res = await c.arun(url, config=cfg)
        return res.html               # raw HTML; we still run extruct+trafilatura on it
```

Memory budget: pin Crawl4AI to 1 concurrent browser. Cost: ~2s on cache miss, ~0ms when not invoked.

### 4.3 Define `extract_entity` as its own tool

```python
def extract_entity(url_or_html: str, kind: str = "auto") -> dict:
    """Return normalized entity (product/place/review) from a page."""
    html = url_or_html if url_or_html.startswith("<") else fetch_url(url_or_html)["_raw_html"]
    data = extruct.extract(html, syntaxes=["json-ld", "microdata", "opengraph"], uniform=True)

    entities = []
    for item in data.get("json-ld", []) + data.get("microdata", []):
        t = item.get("@type") or item.get("type")
        if kind != "auto" and kind.lower() not in str(t).lower():
            continue
        entities.append({
            "type": t,
            "name": item.get("name"),
            "rating": _dig(item, "aggregateRating", "ratingValue"),
            "review_count": _dig(item, "aggregateRating", "reviewCount"),
            "price": _dig(item, "offers", "price"),
            "currency": _dig(item, "offers", "priceCurrency"),
            "address": item.get("address"),
            "url": item.get("url"),
        })
    return {"entities": entities, "opengraph": data.get("opengraph", [])}
```

This is ~30 lines and answers most "best hotel in Tokyo" verification queries deterministically — no LLM call needed.

### 4.4 Upgrade `web_search` to SearXNG-first, DDGS-fallback

Run SearXNG as a sidecar container. In its `settings.yml`:

```yaml
search:
  formats: [html, json]
```

Then:

```python
SEARXNG = os.getenv("SEARXNG_URL", "http://searxng:8080")
def web_search(query: str, n: int = 10) -> list[dict]:
    try:
        r = httpx.get(f"{SEARXNG}/search",
                      params={"q": query, "format": "json", "categories": "general"},
                      timeout=8)
        if r.status_code == 200:
            return [{"title": x["title"], "url": x["url"],
                     "snippet": x.get("content", ""), "engine": x.get("engine")}
                    for x in r.json()["results"][:n]]
    except Exception:
        pass
    return _ddgs_fallback(query, n)
```

The `engine` field gives the surrogate provenance it can use in `rerank` or `verify_fact`.

### 4.5 What we keep as-is

- `curl_cffi` impersonation — still the fastest tier; Crawl4AI does not replace it.
- BS4 visible-text fallback — keep as Tier 3 inside fetch_url.
- `crawl_dom` UI tab — leave alone for now, but formalize its output as an "accessibility-style outline" (a Playwright-MCP pattern lift), which makes it useful as a tool the LLM can call rather than just a debug view.

### 4.6 What we explicitly don't do

- Don't add browser-use, Firecrawl, or Playwright-MCP as runtime dependencies.
- Don't render every page — keep curl_cffi as the default; render is opt-in by flag or SPA-host list.
- Don't put extruct behind the SPA gate. Run it on every page — it's cheap and tells us instantly whether the page has structured data.

### 4.7 Expected impact

- Structured-data coverage goes from ~0% to ~60% of "best X in Y" target pages (hotels, restaurants, products, books, films) — instant uplift on `verify_fact`.
- TripAdvisor / Yelp / Booking pages stop returning empty bodies; render fallback gets us full content for ~2s extra latency on the pages that need it.
- `web_search` gets multi-engine federation + per-result engine provenance + Tavily-free operation.
- Net new code: ~150 lines (fetch_url tiers + extract_entity + searxng client). One new pip dep (`extruct`). One optional pip dep (`crawl4ai`). One new docker service (`searxng`).

**Sources:**
- [browser-use repository](https://github.com/browser-use/browser-use)
- [Firecrawl repository](https://github.com/mendableai/firecrawl)
- [Firecrawl pricing](https://www.firecrawl.dev/pricing)
- [Crawl4AI repository](https://github.com/unclecode/crawl4ai)
- [Crawl4AI multi-URL crawling docs](https://docs.crawl4ai.com/advanced/multi-url-crawling/)
- [Crawl4AI quickstart](https://docs.crawl4ai.com/core/quickstart/)
- [Playwright MCP repository](https://github.com/microsoft/playwright-mcp)
- [extruct repository](https://github.com/scrapinghub/extruct)
- [Trafilatura Python usage docs](https://trafilatura.readthedocs.io/en/latest/usage-python.html)
- [SearXNG repository](https://github.com/searxng/searxng)
- [SearXNG search settings docs](https://docs.searxng.org/admin/settings/settings_search.html)
- [Spider benchmark: Firecrawl vs Crawl4AI vs Spider](https://spider.cloud/blog/firecrawl-vs-crawl4ai-vs-spider-honest-benchmark)
- [Firecrawl vs Anycrawl vs Crawlee vs Playwright comparison](https://mcp.directory/blog/firecrawl-vs-anycrawl-vs-crawlee-vs-playwright-2026)
