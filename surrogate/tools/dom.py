"""DOM crawler — extract richer structured content from a web page than
`fetch_url`'s "main article" path.

`fetch_url` is good for blog posts (trafilatura's article extraction). It is
bad for LIST PAGES — TripAdvisor "top 10 restaurants", SPA menus, ranked
recommendation pages — because the meaningful structure (ordered items,
ratings, links) sits in nav/ul/li/table elements that trafilatura discards.

`crawl_dom` walks the HTML tree itself and emits a structured text view:
PAGE TITLE, HEADINGS, ORDERED/UNORDERED LISTS (with item count), TABLES,
TOP LINKS, NUMERIC SIGNALS (ratings/prices found near anchor text), and a
cleaned body. The format is a single string suitable for direct LLM evidence
injection.

`compare_doms(url_a, url_b, query)` packs two crawls side-by-side into one
evidence block — the entry point for the "two websites in" presentation flow.
"""
from __future__ import annotations

import re
from typing import Iterable

from bs4 import BeautifulSoup

from surrogate.tools.fetch import fetch_html


# Section budgets (chars). Per-section caps keep one section from starving
# the others when packing two URLs side-by-side into Stage 2's evidence.
MAX_PER_LIST = 1500
MAX_PER_TABLE = 1500
MAX_TOTAL_LINKS = 25
MAX_BODY_CHARS = 2500
MAX_SECTION_HEADINGS = 30


# Ratings appear as "4.8", "4.8/5", "★ 4.8", "(175 reviews)" etc. We don't
# need perfect — we extract anchors near these so the model sees ratings as
# proximate signals to venue/product names.
_RATING_RE = re.compile(r"(?:★\s*)?(\d(?:\.\d+)?)\s*(?:/\s*5|/\s*10|\s*stars?|\s*★)?", re.I)
_REVIEW_COUNT_RE = re.compile(r"\(?\s*(\d{1,6})\s*(?:reviews?|ratings?)\s*\)?", re.I)
_PRICE_RE = re.compile(r"(?:US\$|£|€|\$)\s?\d[\d,.]*", re.I)


def _clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _strip_chrome(soup: BeautifulSoup) -> None:
    """Remove non-content nodes that pollute extraction."""
    for tag in soup(["script", "style", "noscript", "svg", "form", "header",
                     "footer", "nav", "iframe"]):
        tag.decompose()
    # Common ad / cookie banners by class/id heuristic
    for el in soup.find_all(attrs={"role": ["banner", "navigation"]}):
        el.decompose()


def _extract_headings(soup: BeautifulSoup) -> list[str]:
    out = []
    for h in soup.find_all(["h1", "h2", "h3"]):
        t = _clean_ws(h.get_text(" "))
        if t and len(t) <= 200:
            out.append(f"{h.name}: {t}")
        if len(out) >= MAX_SECTION_HEADINGS:
            break
    return out


def _list_to_lines(lst, kind: str) -> list[str]:
    items = []
    for li in lst.find_all("li", recursive=False):
        t = _clean_ws(li.get_text(" "))
        if t:
            items.append(t)
    if not items:
        return []
    head = f"{kind} ({len(items)} items):"
    out, total = [head], len(head)
    bullet = "  -" if kind == "ul" else "  #"
    for i, it in enumerate(items, 1):
        prefix = bullet if kind == "ul" else f"  {i}."
        line = f"{prefix} {it}"
        if total + len(line) > MAX_PER_LIST:
            out.append(f"  ... +{len(items) - i + 1} more items truncated")
            break
        out.append(line)
        total += len(line) + 1
    return out


def _extract_lists(soup: BeautifulSoup) -> list[str]:
    blocks = []
    for lst in soup.find_all(["ul", "ol"]):
        # Skip nested lists (we'll pick them up via their parent's text);
        # also skip nav-like short lists with all-link items.
        if lst.find_parent(["ul", "ol", "nav"]):
            continue
        block = _list_to_lines(lst, lst.name)
        if not block:
            continue
        # Heuristic: a ul whose items are all single short links (≤30 chars) is
        # almost certainly a nav menu — drop it.
        only_short = all(len(_clean_ws(li.get_text(" "))) <= 30
                         for li in lst.find_all("li", recursive=False))
        if only_short and lst.name == "ul":
            continue
        blocks.append("\n".join(block))
    return blocks


def _extract_tables(soup: BeautifulSoup) -> list[str]:
    blocks = []
    for tbl in soup.find_all("table"):
        rows = []
        for tr in tbl.find_all("tr"):
            cells = [_clean_ws(td.get_text(" ")) for td in tr.find_all(["th", "td"])]
            if any(cells):
                rows.append(" | ".join(cells))
        if not rows:
            continue
        cols = max(len(r.split(" | ")) for r in rows)
        head = f"Table ({cols} cols × {len(rows)} rows):"
        block_lines, total = [head], len(head)
        for r in rows:
            line = f"  | {r} |"
            if total + len(line) > MAX_PER_TABLE:
                block_lines.append("  ... rows truncated")
                break
            block_lines.append(line)
            total += len(line) + 1
        blocks.append("\n".join(block_lines))
    return blocks


def _extract_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    out, seen = [], set()
    for a in soup.find_all("a"):
        href = a.get("href") or ""
        text = _clean_ws(a.get_text(" "))
        if not href or not text or len(text) > 120:
            continue
        # Drop pure-fragment anchors and javascript: links
        if href.startswith("#") or href.lower().startswith("javascript:"):
            continue
        # Normalize host-relative
        if href.startswith("/"):
            try:
                from urllib.parse import urljoin
                href = urljoin(base_url, href)
            except Exception:
                pass
        key = (text.lower(), href)
        if key in seen:
            continue
        seen.add(key)
        out.append(f"  {text} -> {href}")
        if len(out) >= MAX_TOTAL_LINKS:
            break
    return out


def _extract_numeric_signals(soup: BeautifulSoup) -> list[str]:
    """Find rating / review-count / price tokens and capture the nearest
    anchor-text or strong-text neighbour."""
    out = []
    text_blocks = []
    for el in soup.find_all(["li", "p", "div", "span", "h2", "h3"]):
        t = _clean_ws(el.get_text(" "))
        if not t:
            continue
        rating_m = _RATING_RE.search(t)
        rev_m = _REVIEW_COUNT_RE.search(t)
        price_m = _PRICE_RE.search(t)
        if not (rating_m or rev_m or price_m):
            continue
        # Find a plausible "name" near it: nearest preceding strong/h/a text.
        name = ""
        for sib in el.find_all_previous(["strong", "b", "h1", "h2", "h3", "a"], limit=3):
            cand = _clean_ws(sib.get_text(" "))
            if 3 <= len(cand) <= 80:
                name = cand
                break
        if not name:
            name = t[:60] + ("…" if len(t) > 60 else "")
        bits = []
        if rating_m:
            bits.append(f"★{rating_m.group(1)}")
        if rev_m:
            bits.append(f"{rev_m.group(1)} reviews")
        if price_m:
            bits.append(f"price {price_m.group(0)}")
        out.append(f"  {' / '.join(bits)}  near: {name}")
        if len(out) >= 40:
            break
    # de-dupe identical entries
    seen, deduped = set(), []
    for line in out:
        if line in seen:
            continue
        seen.add(line); deduped.append(line)
    return deduped


def _extract_body(soup: BeautifulSoup) -> str:
    """A short visible-text body for context (after structured sections)."""
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{2,}", "\n", text)
    if len(text) > MAX_BODY_CHARS:
        text = text[:MAX_BODY_CHARS] + f"\n[... +{len(text) - MAX_BODY_CHARS} chars truncated]"
    return text


def crawl_dom(url: str) -> dict:
    """Fetch url, walk the DOM, return a structured extract:

        {
          'url': ..., 'title': ..., 'ok': bool, 'error': str|None,
          'sections': {
            'headings': [...], 'lists': [...str blocks...],
            'tables': [...str blocks...], 'links': [...lines...],
            'numeric_signals': [...lines...], 'body': '...',
          },
          'as_text': '<single formatted string ready for evidence injection>',
        }
    """
    html = fetch_html(url)
    if html.startswith("(fetch error"):
        return {"url": url, "ok": False, "error": html,
                "title": "", "sections": {}, "as_text": f"URL: {url}\n{html}"}
    soup = BeautifulSoup(html, "lxml")
    _strip_chrome(soup)

    title = ""
    if soup.title and soup.title.string:
        title = _clean_ws(soup.title.string)
    headings = _extract_headings(soup)
    lists = _extract_lists(soup)
    tables = _extract_tables(soup)
    links = _extract_links(soup, base_url=url)
    nums = _extract_numeric_signals(soup)
    body = _extract_body(soup)

    parts: list[str] = [f"URL: {url}", f"PAGE TITLE: {title or '(no <title>)'}"]
    if headings:
        parts.append("")
        parts.append("HEADINGS:")
        parts.extend(f"  {h}" for h in headings)
    if nums:
        parts.append("")
        parts.append("NUMERIC SIGNALS (ratings / review counts / prices found near names):")
        parts.extend(nums)
    if lists:
        parts.append("")
        parts.append(f"LISTS ({len(lists)}):")
        for blk in lists:
            parts.append(blk)
    if tables:
        parts.append("")
        parts.append(f"TABLES ({len(tables)}):")
        for blk in tables:
            parts.append(blk)
    if links:
        parts.append("")
        parts.append(f"TOP LINKS (≤{MAX_TOTAL_LINKS}):")
        parts.extend(links)
    if body:
        parts.append("")
        parts.append("BODY TEXT (cleaned, truncated):")
        parts.append(body)

    return {
        "url": url, "ok": True, "error": None,
        "title": title,
        "sections": {
            "headings": headings, "lists": lists, "tables": tables,
            "links": links, "numeric_signals": nums, "body": body,
        },
        "as_text": "\n".join(parts),
    }


def compare_doms(url_a: str, url_b: str, *, query: str | None = None) -> dict:
    """Crawl two URLs and pack them side-by-side as one evidence block.

    Result['evidence_block'] is the string that callers should inject directly
    into Stage 2's user message (it is shaped to match the existing
    "EVIDENCE GATHERED FROM TOOL CALLS:" framing).
    """
    a = crawl_dom(url_a)
    b = crawl_dom(url_b)
    parts = []
    if query:
        parts.append(f"QUESTION: {query}")
        parts.append("")
    parts.append("EVIDENCE FROM TWO USER-SPECIFIED WEBSITES (DOM crawl):")
    parts.append("")
    parts.append("==== Website A ====")
    parts.append(a["as_text"])
    parts.append("")
    parts.append("==== Website B ====")
    parts.append(b["as_text"])
    return {
        "url_a": url_a, "url_b": url_b,
        "ok_a": a["ok"], "ok_b": b["ok"],
        "a": a, "b": b,
        "evidence_block": "\n".join(parts),
    }
