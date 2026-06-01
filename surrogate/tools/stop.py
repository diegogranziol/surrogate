"""`stop_and_answer` — terminating tool with a structured Pydantic payload.

This is the contractual end of the trajectory. Instead of letting the model
emit free-form text in `<answer>...</answer>`, we require a structured
payload:

    {
      "top_picks": [
        {"name": ..., "reasoning": ..., "rating": ..., "source_url": ...},
        ...
      ],
      "summary":   "one-paragraph reasoning over the candidates",
      "citations": [{"title": ..., "url": ...}, ...]
    }

The loop intercepts this tool call. If validation fails, we hand the model
the error message and let it try again. If it succeeds, we render the
payload as user-facing markdown and terminate cleanly.

No LLM call inside the tool — pure Pydantic validate-and-format.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError


class TopPick(BaseModel):
    name: str = Field(..., min_length=1,
                      description="Entity name (restaurant / hotel / product / place / book / etc.)")
    reasoning: str = Field(..., min_length=10,
                           description="One-line reason this is a top pick — concrete, evidence-grounded.")
    rating: Optional[str] = Field(None,
                                  description="Rating as it appears in the source, e.g. '4.8/5' or '★4.5 (175 reviews)'.")
    price: Optional[str] = Field(None,
                                 description="Price or price range as it appears in the source.")
    source_url: Optional[str] = Field(None,
                                      description="URL of the page that supports this pick.")


class Citation(BaseModel):
    title: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)


class StopAndAnswerPayload(BaseModel):
    top_picks: list[TopPick] = Field(..., min_length=3, max_length=10,
                                     description="3–5 ranked recommendations, best first.")
    summary: str = Field(..., min_length=20,
                         description="Short paragraph reasoning over the top picks.")
    citations: list[Citation] = Field(default_factory=list,
                                      description="Sources cited across the answer.")


# JSON-schema spec exposed to the LLM via the system prompt (Hermes block).
STOP_AND_ANSWER_PARAMETERS = {
    "type": "object",
    "properties": {
        "top_picks": {
            "type": "array",
            "minItems": 3, "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "name":       {"type": "string", "description": "Entity name."},
                    "reasoning":  {"type": "string", "description": "One-line evidence-grounded reason."},
                    "rating":     {"type": "string", "description": "e.g. '4.8/5 (175 reviews)'"},
                    "price":      {"type": "string", "description": "e.g. '$50–$80'"},
                    "source_url": {"type": "string", "description": "URL backing this pick."},
                },
                "required": ["name", "reasoning"],
            },
            "description": "Ranked recommendations, best first.",
        },
        "summary":   {"type": "string", "description": "One-paragraph synthesis."},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "url":   {"type": "string"},
                },
                "required": ["title", "url"],
            },
            "description": "All sources referenced.",
        },
    },
    "required": ["top_picks", "summary"],
}


def validate(args: dict) -> tuple[Optional[StopAndAnswerPayload], Optional[str]]:
    """Validate the tool args. Returns (payload, None) on success,
    (None, error_message) on failure."""
    try:
        return StopAndAnswerPayload.model_validate(args), None
    except ValidationError as e:
        # Compact error text so the model can fix without scrolling.
        lines = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err.get("loc", []))
            lines.append(f"  - {loc or '(root)'}: {err.get('msg', err)}")
        return None, "stop_and_answer args invalid:\n" + "\n".join(lines)


def render_markdown(p: StopAndAnswerPayload) -> str:
    """Render the validated payload as user-facing markdown."""
    out: list[str] = []
    for i, pick in enumerate(p.top_picks, 1):
        head = f"{i}. **{pick.name}**"
        bits = []
        if pick.rating:
            bits.append(pick.rating)
        if pick.price:
            bits.append(pick.price)
        if bits:
            head += " — " + " · ".join(bits)
        out.append(head)
        out.append(f"   {pick.reasoning}")
        if pick.source_url:
            out.append(f"   _Source:_ {pick.source_url}")
        out.append("")
    out.append(p.summary)
    if p.citations:
        out.append("")
        out.append("**Citations**")
        for c in p.citations:
            out.append(f"- [{c.title}]({c.url})")
    return "\n".join(out).strip()
