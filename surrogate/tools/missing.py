"""`check_missing_fields` tool — the frontier scheduler.

For a candidate the model is about to recommend, this tool reports which
canonical purchase-intent fields are still empty / unverified. The agent
uses the result to decide: run one more search/extract round for the
missing field, or commit via `stop_and_answer`.

Deterministic — pure regex/heuristic. No LLM call.

Canonical fields:
  Required by default:  name, rating, location, source_url
  Recommended-only:     review_count, price, key_features
"""
from __future__ import annotations

import re


DEFAULT_REQUIRED = ["name", "rating", "location", "source_url"]
DEFAULT_OPTIONAL = ["review_count", "price", "key_features"]


# Single-source-of-truth detectors. Each returns True iff the field looks
# present in the candidate description.

def _has_rating(text: str) -> bool:
    # 4.8, 4.8/5, ★4.8, rated 4.5, 5-star, 9.2/10
    return bool(
        re.search(r"(?:[★⭐]\s*)?\b\d(?:\.\d+)?\s*(?:/\s*\d+|stars?|★|⭐|out of)",
                  text, re.I)
        or re.search(r"\brated?\s+\d(?:\.\d+)?\b", text, re.I)
    )


def _has_review_count(text: str) -> bool:
    # 175 reviews, 1,200 ratings, (1.2K reviews)
    return bool(
        re.search(r"\b\d{1,3}(?:[,.]?\d{3})*(?:\s*[Kk])?\s+(?:reviews?|ratings?)", text)
        or re.search(r"\(\s*\d[\d,.]*\s+(?:reviews?|ratings?)\s*\)", text, re.I)
    )


def _has_price(text: str) -> bool:
    # $200, £45, €15, 'around $20', '$$$', priceRange, 'under $500'
    return bool(re.search(
        r"[$£€¥₽]\s*\d+"
        r"|\b\d+\s*(?:USD|EUR|GBP|CAD|AUD|UZS|RUB|JPY|CHF)\b"
        r"|\bunder\s+[$£€¥]?\d+"
        r"|\baround\s+[$£€¥]\d+"
        r"|\$\${1,3}"
        r"|price[ _-]?range",
        text, re.I,
    ))


def _has_location(text: str) -> bool:
    # "in Tashkent", "at 5th Ave", "Address: …", ", New York"
    if re.search(r"\b(?:in|at|on|near)\s+[A-Z][\w'-]+", text):
        return True
    if re.search(r",\s+[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)?\b", text):
        return True
    if re.search(r"\baddress[:\s]+\S", text, re.I):
        return True
    return False


def _has_source_url(text: str) -> bool:
    return bool(re.search(r"https?://", text))


def _has_name(text: str) -> bool:
    # Any capitalized phrase that isn't a generic stopword
    for m in re.finditer(r"\b([A-Z][\w'&.-]+(?:\s+[A-Z][\w'&.-]+){0,3})\b", text):
        words = m.group(1).split()
        if all(w.lower() in {"the", "best", "top", "italian", "french",
                              "japanese", "restaurant", "hotel", "place",
                              "spot", "bar", "cafe", "shop"} for w in words):
            continue
        return True
    return False


def _has_key_features(text: str) -> bool:
    return bool(re.search(
        r"\b(?:cozy|excellent|delicious|elegant|authentic|fresh|popular|"
        r"luxury|spacious|quiet|romantic|family-friendly|outdoor|rooftop|"
        r"wood-fired|terrace|garden|menu|chef|specialty|signature|"
        r"organic|gluten-free|vegan|vegetarian|halal|kosher|"
        r"award-winning|michelin|fine\s+dining)\b",
        text, re.I,
    ))


_DETECTORS = {
    "name":          _has_name,
    "rating":        _has_rating,
    "review_count":  _has_review_count,
    "price":         _has_price,
    "location":      _has_location,
    "source_url":    _has_source_url,
    "key_features":  _has_key_features,
}


def check_missing_fields(
    candidate: str,
    required: list[str] | None = None,
) -> str:
    """Return a text observation listing which canonical fields are still
    missing from the candidate description."""
    if not candidate or not candidate.strip():
        return "[check_missing_fields] error: missing 'candidate' description"
    required = required or DEFAULT_REQUIRED

    missing_required = [f for f in required if not _DETECTORS.get(f, lambda _: True)(candidate)]
    present_required = [f for f in required if f not in missing_required]
    optional_missing = [
        f for f in DEFAULT_OPTIONAL
        if f not in required and not _DETECTORS.get(f, lambda _: True)(candidate)
    ]

    head = candidate[:200] + ("…" if len(candidate) > 200 else "")
    parts = [
        f"Candidate: {head}",
        f"Required present:    {present_required or '(none)'}",
        f"Required MISSING:    {missing_required or '(none)'}",
        f"Recommended missing: {optional_missing or '(none)'}",
    ]
    if missing_required:
        parts.append(
            f"→ Run one more `search`/`extract_entity` for the missing field(s) "
            f"{missing_required}, then re-check before calling `stop_and_answer`."
        )
    else:
        parts.append("→ All required fields present. Safe to include in stop_and_answer.")
    return "\n".join(parts)
