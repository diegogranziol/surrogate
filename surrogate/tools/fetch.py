from __future__ import annotations

import trafilatura
from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests


# MAX_CHARS caps the text we return TO THE MODEL only — it does NOT strip the
# trace. See CLAUDE.md "Where truncation IS allowed". Raise this if you want
# the model to see more context per page; do NOT add a similar cap to
# logger.py / renderers / printed output.
MAX_CHARS = 4000
TRAFILATURA_MIN_USEFUL = 1500  # below this, trafilatura likely returned only boilerplate

# Order matters: iOS Safari first (defeats Imperva-style WAFs that fingerprint
# Chrome/Playwright TLS; observed working on tripadvisor.co.uk), then Chrome as
# a fallback for sites that prefer Chrome User-Agent semantics.
_IMPERSONATE_ORDER = ("safari17_2_ios", "chrome131")


def _bs4_visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript"]):
        t.extract()
    return soup.get_text(separator="\n", strip=True)


def _http_get(url: str) -> tuple[str | None, str | None, int | None]:
    """Return (html, last_err_repr, last_status). On success last_err is None."""
    last_err = None
    last_status = None
    for impersonate in _IMPERSONATE_ORDER:
        try:
            r = cffi_requests.get(url, impersonate=impersonate, timeout=20)
            last_status = r.status_code
            if 200 <= r.status_code < 300:
                return r.text, None, r.status_code
        except Exception as e:
            last_err = repr(e)
    return None, last_err, last_status


def fetch_html(url: str) -> str:
    """Return raw HTML for `url` (browser-TLS impersonation, multi-fingerprint
    fallback). On failure returns a string starting with '(fetch error'.

    Used by the DOM crawler — it walks the tree itself rather than relying on
    trafilatura's article extract."""
    html, last_err, last_status = _http_get(url)
    if html is None:
        if last_err is not None:
            return f"(fetch error: {last_err})"
        return f"(fetch error: HTTP {last_status} after trying {len(_IMPERSONATE_ORDER)} fingerprints)"
    return html


def fetch_url(url: str, max_chars: int | None = MAX_CHARS) -> str:
    """Fetch a URL and return its main readable text.

    Default behaviour caps the returned text at MAX_CHARS (4k) to protect the
    model's context window — this is what the agent tool path uses. Pass
    `max_chars=None` to get the FULL extracted text (used by the user-RAG
    ingestion path, where we want to chunk-and-index the whole page).

    Uses curl_cffi to impersonate real-browser TLS handshakes so WAFs that
    fingerprint stock-Python/httpx clients (TripAdvisor, Cloudflare-protected
    sites) don't reject the request. Falls back from trafilatura (article-style
    extraction) to BeautifulSoup visible-text when trafilatura returns very
    little — useful for SPA list pages where the content isn't in <article> tags.
    """
    html, last_err, last_status = _http_get(url)
    if html is None:
        if last_err is not None:
            return f"(fetch error: {last_err})"
        return f"(fetch error: HTTP {last_status} after trying {len(_IMPERSONATE_ORDER)} fingerprints)"

    article = (trafilatura.extract(html, include_links=False, include_tables=False) or "").strip()
    if len(article) >= TRAFILATURA_MIN_USEFUL:
        text = article
    else:
        text = _bs4_visible_text(html)

    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[...truncated, full length {len(text)} chars]"
    return f"URL: {url}\n\n{text}"
