"""Tool registry for the ReAct loop.

Wraps the underlying Python functions (`web_search`, `fetch_url`, and soon
`extract_entity`, `verify_fact`, `rerank`, `check_missing_fields`,
`stop_and_answer`) into `Tool` objects the loop can dispatch.

Each tool is added INCREMENTALLY — as new ones land, register them here and
they become callable from inside the loop's <tool_call> protocol.
"""
from __future__ import annotations

from surrogate.loop import Tool
from surrogate.tools.search import web_search
from surrogate.tools.fetch import fetch_url
from surrogate.tools.extract import extract_entity
from surrogate.tools.verify import verify_fact
from surrogate.tools.missing import check_missing_fields
from surrogate.tools.stop import STOP_AND_ANSWER_PARAMETERS


# ---------------------------------------------------------------------------
# 1. search — batched or single query (Tongyi-style: array signature)
# ---------------------------------------------------------------------------

def _search_call(args: dict) -> str:
    q = args.get("query")
    n = int(args.get("max_results", 5))
    if not q:
        return "[search] error: missing required arg 'query'"
    # Support both array and string for ergonomics during the migration window.
    if isinstance(q, list):
        out = []
        for i, qi in enumerate(q, 1):
            out.append(f"==== Query {i}: {qi!r} ====")
            out.append(web_search(qi, max_results=n))
        return "\n".join(out)
    return web_search(q, max_results=n)


search_tool = Tool(
    name="search",
    description=(
        "Run one or more web searches via DuckDuckGo (or Tavily if a key is "
        "configured). Pass a list of complementary queries in a single call "
        "to fan out coverage; pass a single string for one query. Returns "
        "a numbered list of `title — url — snippet` per query."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}, "minItems": 1},
                ],
                "description": "A single search string OR an array of complementary queries.",
            },
            "max_results": {
                "type": "integer", "default": 5,
                "description": "How many results per query (1–10).",
            },
        },
        "required": ["query"],
    },
    call=_search_call,
)


# ---------------------------------------------------------------------------
# 2. fetch_url — single URL → cleaned article text (still naive; the
#    `goal`-aware extractor + extruct + render fallback come in the next pass)
# ---------------------------------------------------------------------------

def _fetch_url_call(args: dict) -> str:
    url = args.get("url")
    if not url:
        return "[fetch_url] error: missing required arg 'url'"
    # Hook for later: a `goal` arg will trigger an LLM-side extractor that
    # returns `{rational, evidence, summary}` JSON; for now we ignore it.
    return fetch_url(url)


fetch_url_tool = Tool(
    name="fetch_url",
    description=(
        "Fetch a single URL and return its main readable text. Use after "
        "`search` to read the actual content of the most promising result. "
        "Returns up to ~4KB of cleaned article text per call."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url":  {"type": "string", "description": "Absolute http(s) URL to fetch."},
            "goal": {"type": "string", "description":
                     "Optional: the specific information you want from this "
                     "page. (Reserved for the upcoming goal-aware extractor.)"},
        },
        "required": ["url"],
    },
    call=_fetch_url_call,
)


# ---------------------------------------------------------------------------
# 3. extract_entity — pull structured schema.org data from a page (no LLM)
# ---------------------------------------------------------------------------

def _extract_entity_call(args: dict) -> str:
    url = args.get("url")
    if not url:
        return "[extract_entity] error: missing required arg 'url'"
    return extract_entity(url)


extract_entity_tool = Tool(
    name="extract_entity",
    description=(
        "Pull STRUCTURED entity data from a webpage's embedded schema.org "
        "markup (JSON-LD, microdata, OpenGraph). Returns the entity's name, "
        "rating, review_count, price, address, description, etc. as clean "
        "fields — no prose parsing. **Strongly preferred over `fetch_url` "
        "when you need verified facts** about a product, restaurant, hotel, "
        "place, book, movie, etc. Most major review/commerce sites "
        "(TripAdvisor, Yelp, Booking, Amazon, IMDB) ship rich JSON-LD."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Absolute http(s) URL of the page to extract from.",
            },
        },
        "required": ["url"],
    },
    call=_extract_entity_call,
)


# ---------------------------------------------------------------------------
# 4. verify_fact — deterministic claim/evidence grounding check (no LLM)
# ---------------------------------------------------------------------------

def _verify_fact_call(args: dict) -> str:
    claim = args.get("claim")
    url = args.get("evidence_url") or args.get("url")  # accept either name
    if not claim or not url:
        return "[verify_fact] error: need both 'claim' and 'evidence_url'"
    return verify_fact(claim, url)


verify_fact_tool = Tool(
    name="verify_fact",
    description=(
        "Check whether a specific claim is supported by a specific URL. "
        "Fetches the page, looks for the claim's named entities and numbers, "
        "returns supported=yes/partial/no with the matching text span. "
        "**Call this before citing a fact in your final answer** — especially "
        "for specific numbers (ratings, prices, review counts) and named "
        "items (restaurant names, product models). Deterministic, no LLM call."
    ),
    parameters={
        "type": "object",
        "properties": {
            "claim": {
                "type": "string",
                "description": ("The specific factual claim to verify, e.g. "
                                "'Sette Restaurant is rated 4.8 with 175 reviews'."),
            },
            "evidence_url": {
                "type": "string",
                "description": "Absolute URL of the page that should support the claim.",
            },
        },
        "required": ["claim", "evidence_url"],
    },
    call=_verify_fact_call,
)


# ---------------------------------------------------------------------------
# 5. check_missing_fields — the frontier scheduler (no LLM)
# ---------------------------------------------------------------------------

def _check_missing_fields_call(args: dict) -> str:
    cand = args.get("candidate") or args.get("description")
    if not cand:
        return "[check_missing_fields] error: missing required arg 'candidate'"
    required = args.get("required")  # optional override
    return check_missing_fields(cand, required=required)


check_missing_fields_tool = Tool(
    name="check_missing_fields",
    description=(
        "Before recommending a candidate, check which canonical purchase-intent "
        "fields are still empty / unverified in your description: name, rating, "
        "review_count, price, location, source_url, key_features. Returns a "
        "list of missing fields so you can decide whether to run one more "
        "`search`/`extract_entity` round for the missing field, or commit "
        "with `stop_and_answer`. Use this AT LEAST ONCE per top pick "
        "before finalising."
    ),
    parameters={
        "type": "object",
        "properties": {
            "candidate": {
                "type": "string",
                "description": ("Free-text description of the candidate you're about "
                                "to recommend, including everything you know about it "
                                "so far."),
            },
            "required": {
                "type": "array",
                "items": {"type": "string"},
                "description": ("Optional override of which fields are required for "
                                "this question. Defaults to "
                                "['name','rating','location','source_url']."),
            },
        },
        "required": ["candidate"],
    },
    call=_check_missing_fields_call,
)


# ---------------------------------------------------------------------------
# 6. stop_and_answer — terminating tool; the loop intercepts it specially.
#    `call` should not actually be invoked at runtime — the loop short-circuits
#    on this tool name. We still register a no-op `.call` for defensive
#    completeness (and so the schema renders in the system prompt).
# ---------------------------------------------------------------------------

def _stop_and_answer_call(args: dict) -> str:
    # Should be unreachable — the loop handles stop_and_answer before dispatch.
    return ("[stop_and_answer] internal error: this tool should be intercepted "
            "by the loop. If you see this, the dispatcher did not short-circuit.")


stop_and_answer_tool = Tool(
    name="stop_and_answer",
    description=(
        "**Terminating tool.** Call this to finalise your answer with a "
        "STRUCTURED payload. `top_picks` MUST contain 3–5 ranked candidates "
        "(best first), each with `name`, concrete evidence-grounded "
        "`reasoning`, and ideally `rating`, `price`, `source_url`. Also "
        "include a one-paragraph `summary` and a `citations` list. The loop "
        "validates the payload against a schema; if validation fails you get "
        "one more attempt. **Call this AS YOUR FINAL ACTION** — never after this."
    ),
    parameters=STOP_AND_ANSWER_PARAMETERS,
    call=_stop_and_answer_call,
)


# ---------------------------------------------------------------------------
# 7. think — no-op reflection tool (LangChain's `think_tool` pattern).
#    Forces the model to land a structured checkpoint in the trace.
# ---------------------------------------------------------------------------

def _think_call(args: dict) -> str:
    # No-op. The purpose of this tool is the side-effect of the model
    # emitting a structured reflection that lands in the trace.
    return "logged"


think_tool = Tool(
    name="think",
    description=(
        "Force-write a structured reflection checkpoint into the trace. Call "
        "AFTER every 2 `search`/`extract_entity` calls AND BEFORE "
        "`stop_and_answer`. The reflection argument should follow this "
        "shape: (1) what you've learned so far, (2) what's still uncertain "
        "or unverified, (3) what's the next best action. Returns 'logged' — "
        "its job is to make your plan crystallise visibly in the trace, "
        "not to perform any computation."
    ),
    parameters={
        "type": "object",
        "properties": {
            "reflection": {
                "type": "string",
                "description": ("Structured reflection: 'Learned: ... ; Uncertain: ... ; "
                                "Next: ...'  Be concise but specific."),
            },
        },
        "required": ["reflection"],
    },
    call=_think_call,
)


# ---------------------------------------------------------------------------
# default_tools() — what `loop.run()` gets by default. Add tools here as they
# come online: extract_entity ✓ → verify_fact ✓ → check_missing_fields ✓ →
# stop_and_answer ✓ → think (next).
# ---------------------------------------------------------------------------

def default_tools() -> list[Tool]:
    return [
        search_tool,
        fetch_url_tool,
        extract_entity_tool,
        verify_fact_tool,
        check_missing_fields_tool,
        think_tool,
        stop_and_answer_tool,
    ]
