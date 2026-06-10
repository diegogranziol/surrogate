"""`verify_fact` tool — deterministic claim/evidence grounding check.

Given a `(claim, evidence_url)`, fetch the page (full text + extracted
structured entities) and check whether the claim's key signals — proper-noun
phrases and numbers — actually appear on the page. Return a structured
verdict (`yes` / `partial` / `no`) with confidence and the densest matching
span.

**No LLM call.** This is deliberate: small-model self-critique is a documented
regression risk (Stanford 2024); a deterministic, external verifier dodges
that. The signal it gives the agent is unambiguous:
  - `supported: yes` (≥70% of claim tokens present)  → safe to cite
  - `supported: partial`                              → some tokens missing
  - `supported: no`                                   → page doesn't back you up

The agent learns to: re-search when `no`, refine claim when `partial`,
and confidently include the citation when `yes`.
"""
from __future__ import annotations

import re

import trafilatura

from surrogate.tools.extract import extract_entity
from surrogate.tools.fetch import fetch_html, fetch_url


# Capitalized words that aren't entity-like signal — drop them as single-word
# matches even when they happen to be capitalized.
_STOP = {
    "the", "a", "an", "of", "in", "on", "and", "or", "for", "with",
    "best", "top", "most", "highest", "good", "great",
    "italian", "french", "japanese", "chinese", "mexican", "indian", "thai",
    "american", "korean", "vietnamese", "spanish", "greek", "turkish",
    "restaurant", "hotel", "place", "spot", "cafe", "bar", "pub", "shop",
    "food", "cuisine", "dish", "meal", "brand", "product", "service",
    "if", "is", "are", "has", "have", "be", "was", "were", "this", "that",
    "these", "those", "there", "here", "i", "you", "we", "they", "it",
}

# Match: a capitalized word, optionally followed by up to 5 more capitalized
# words / typical name connectors (&, of, the, de, la, le, du, and).
_RE_PROPER = re.compile(
    r"\b([A-Z][A-Za-z'&.-]+(?:\s+(?:[A-Z][A-Za-z'&.-]+|of|the|de|la|le|du|&|and))*)\b"
)
_RE_NUMBER = re.compile(r"\$?\b(\d+(?:[.,]\d+)*)\b")


def _signals(claim: str) -> list[str]:
    """Extract checkable tokens: proper-noun-ish phrases + numbers. Dedup
    case-insensitively while preserving order."""
    raw: list[str] = []
    for m in _RE_PROPER.finditer(claim):
        phrase = m.group(1).strip()
        # Drop trailing connector tokens (e.g. "Sette and" -> "Sette")
        tokens = phrase.split()
        while tokens and tokens[-1].lower() in {"of", "the", "de", "la",
                                                  "le", "du", "and", "&"}:
            tokens.pop()
        if not tokens:
            continue
        phrase = " ".join(tokens)
        # Drop single-word stopwords (e.g. "Italian", "The")
        if len(tokens) == 1 and tokens[0].lower() in _STOP:
            continue
        raw.append(phrase)
    for m in _RE_NUMBER.finditer(claim):
        raw.append(m.group(1))

    seen: set[str] = set()
    out: list[str] = []
    for t in raw:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def _full_page_text(url: str) -> str:
    """Get the page's full extracted text (no 4 KB cap) PLUS its structured
    entities — gives the matcher both prose and schema.org fields to search."""
    html = fetch_html(url)
    if html.startswith("(fetch error"):
        return ""
    try:
        body = trafilatura.extract(
            html, url=url, output_format="txt",
            with_metadata=True, include_tables=True,
            favor_precision=True, deduplicate=True,
        ) or ""
    except Exception:
        body = ""
    # Fold in extract_entity's text-formatted structured fields — they often
    # carry the rating / price / address in cleaner form than the article body.
    try:
        struct_text = extract_entity(url)
    except Exception:
        struct_text = ""
    return body + "\n\n" + struct_text


def _densest_span(text: str, terms: list[str], window: int = 240) -> tuple[int, str]:
    """Find the window of size `window` that contains the most distinct
    `terms`. Returns (distinct-term-count-in-span, span-text)."""
    if not text or not terms:
        return 0, ""
    text_lower = text.lower()
    hits: list[tuple[int, str]] = []
    for t in terms:
        idx = text_lower.find(t.lower())
        if idx >= 0:
            hits.append((idx, t))
    if not hits:
        return 0, ""
    # For each hit, count how many distinct terms have a hit within ±window.
    best_count, best_span = 0, ""
    for i, _ in hits:
        seen: set[str] = set()
        for j, t in hits:
            if abs(j - i) <= window:
                seen.add(t.lower())
        if len(seen) > best_count:
            best_count = len(seen)
            start = max(0, i - window // 2)
            end = min(len(text), i + window)
            best_span = text[start:end].strip()
    return best_count, best_span


def verify_fact(claim: str, evidence_url: str) -> str:
    """Return a text observation: does `evidence_url` support `claim`?"""
    if not claim or not evidence_url:
        return "[verify_fact] error: need both 'claim' and 'evidence_url'"

    page = _full_page_text(evidence_url)
    if not page.strip():
        # Fall back to the capped fetch_url if html-fetch failed completely
        page = fetch_url(evidence_url) or ""
    if not page.strip():
        return f"[verify_fact] could not fetch evidence at {evidence_url}"

    terms = _signals(claim)
    if not terms:
        return ("[verify_fact] claim has no checkable signals "
                "(no proper nouns or numbers found). "
                "Rephrase with a specific name, rating, price, or date.")

    page_lower = page.lower()
    matched = [t for t in terms if t.lower() in page_lower]
    missing = [t for t in terms if t.lower() not in page_lower]
    conf = len(matched) / len(terms)

    if conf >= 0.7:
        verdict = "yes"
    elif conf >= 0.4:
        verdict = "partial"
    else:
        verdict = "no"

    _, span = _densest_span(page, terms)
    span_disp = (span[:400] + "…") if len(span) > 400 else span

    return (
        f"Claim:     {claim}\n"
        f"Source:    {evidence_url}\n"
        f"Supported: {verdict}  (confidence {conf:.2f}, matched {len(matched)}/{len(terms)})\n"
        f"Matched:   {matched}\n"
        f"Missing:   {missing}\n"
        f"Span:      {span_disp or '(no co-occurring span found — terms scattered or absent)'}"
    )
