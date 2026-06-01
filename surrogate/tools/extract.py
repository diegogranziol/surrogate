"""`extract_entity` tool — pull structured entity data from a webpage.

Most "best X in Y" target pages (TripAdvisor, Yelp, Booking, Amazon, IMDB,
restaurant/hotel/product pages…) embed schema.org JSON-LD or microdata that
already contains `name`, `aggregateRating`, `offers`, `address`, etc.
Today we throw that signal away and ask the LLM to regex it out of prose.
This tool returns it directly. **No LLM call, ~100ms per page, BSD-3 dep.**

Use case in the agent loop:
    Stage 1 already called `search` → has a list of candidate URLs.
    Instead of `fetch_url`-ing the prose and hoping ratings survive, call
    `extract_entity(url)` and get `{name, rating, review_count, price, address, ...}`
    deterministically.
"""
from __future__ import annotations

import json
from typing import Any

import extruct

from surrogate.tools.fetch import fetch_html


# Schema.org @types we care about for purchase-intent answers.
# Substring match — schema.org has e.g. "Restaurant", "FoodEstablishment",
# "BarOrPub" which are all relevant.
_INTERESTING_TYPES = (
    "Product", "Restaurant", "Hotel", "Lodging", "FoodEstablishment",
    "LocalBusiness", "Place", "TouristAttraction",
    "Book", "Movie", "MusicAlbum", "VideoGame", "SoftwareApplication",
    "Service", "Course", "BarOrPub", "CafeOrCoffeeShop",
)

# How many entities to surface per page (defensive against listing pages
# that embed dozens of identical objects).
MAX_ENTITIES = 12


def _norm_type(t: Any) -> str:
    """Schema.org `@type` can be a string or list of strings. Pick the first."""
    if isinstance(t, list):
        return str(t[0]) if t else ""
    return str(t or "")


def _dig(d: Any, *keys: str) -> Any:
    """Safely traverse nested dicts/lists. Returns None on miss."""
    cur = d
    for k in keys:
        if isinstance(cur, list) and cur:
            cur = cur[0]
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return None
    return cur


def _slim_entity(item: dict) -> dict | None:
    """Reduce a verbose schema.org item to the few fields that matter for a
    purchase recommendation. Returns None if the item isn't an interesting
    type (skips boilerplate like `WebSite`, `Organization`, `BreadcrumbList`)."""
    t = _norm_type(item.get("@type") or item.get("type"))
    if not any(it in t for it in _INTERESTING_TYPES):
        return None
    out = {
        "type": t,
        "name": item.get("name"),
        "url": item.get("url"),
        "rating": _dig(item, "aggregateRating", "ratingValue"),
        "review_count": (
            _dig(item, "aggregateRating", "reviewCount")
            or _dig(item, "aggregateRating", "ratingCount")
        ),
        "price": _dig(item, "offers", "price"),
        "currency": _dig(item, "offers", "priceCurrency"),
        "price_range": item.get("priceRange"),
        "address": item.get("address"),
        "description": item.get("description"),
    }
    return {k: v for k, v in out.items() if v not in (None, "")}


def _walk_json_ld(items: list) -> list[dict]:
    """Flatten @graph wrappers; collect interesting entities."""
    out: list[dict] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        # @graph wraps multiple entities — unwrap it.
        if "@graph" in item:
            for g in item.get("@graph", []) or []:
                if isinstance(g, dict):
                    slim = _slim_entity(g)
                    if slim:
                        out.append(slim)
        slim = _slim_entity(item)
        if slim:
            out.append(slim)
    return out


def extract_entity(url: str) -> str:
    """Fetch URL and return a text summary of structured entities found in
    its JSON-LD / microdata / OpenGraph. No LLM call.

    Returns a multi-line string ready to drop into the agent's tool-response.
    """
    html = fetch_html(url)
    if html.startswith("(fetch error"):
        return f"[extract_entity] {html}"

    try:
        data = extruct.extract(
            html, base_url=url,
            syntaxes=["json-ld", "microdata", "opengraph"],
            uniform=True,
        )
    except Exception as e:
        return f"[extract_entity] parse error: {e!r}"

    entities = []
    entities.extend(_walk_json_ld(data.get("json-ld") or []))
    entities.extend(_walk_json_ld(data.get("microdata") or []))
    # Dedupe by (type, name) — listing pages often repeat the same item.
    seen = set()
    deduped = []
    for e in entities:
        key = (e.get("type"), (e.get("name") or "").strip().lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)

    og = data.get("opengraph") or []

    if not deduped and not og:
        return (
            f"No structured entities found at {url}. "
            "(Page has no JSON-LD, microdata, or OpenGraph markup of the "
            "interesting types — Product/Restaurant/Hotel/Place/Book/etc.)"
        )

    parts = [f"Structured entities extracted from {url}:"]
    for i, e in enumerate(deduped[:MAX_ENTITIES], 1):
        parts.append(f"\n[{i}] @type: {e.get('type')}")
        for k in (
            "name", "rating", "review_count",
            "price", "currency", "price_range",
            "address", "description", "url",
        ):
            v = e.get(k)
            if v is None or v == "":
                continue
            if isinstance(v, (dict, list)):
                v = json.dumps(v, ensure_ascii=False)
            s = str(v).strip()
            if len(s) > 280:
                s = s[:280] + "…"
            parts.append(f"  {k}: {s}")

    if og:
        og_slim = {k: v for k, v in (og[0] if isinstance(og, list) else og).items()
                   if k in ("og:title", "og:description", "og:type", "og:site_name")}
        if og_slim:
            parts.append("")
            parts.append(f"OpenGraph: {json.dumps(og_slim, ensure_ascii=False)[:400]}")

    if len(deduped) > MAX_ENTITIES:
        parts.append(f"\n[…{len(deduped) - MAX_ENTITIES} more entities truncated]")

    return "\n".join(parts)
