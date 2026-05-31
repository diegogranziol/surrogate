# Workflow Design Synthesis

**Date:** 2026-05-29
**Source:** Four parallel research agents covering (A) open-source deep-research clones, (B) engineering principles + agent frameworks, (C) browser/crawl/extraction tool layer, (D) academic literature + benchmark evidence.
**Use:** Pitch-ready synthesis for the Diego Google-doc and v1 build plan.

---

## 🥇 Golden findings (the executive summary)

### The single most important finding

**Tongyi DeepResearch (Alibaba) is *our project*, minus the open weights and minus our differentiator.** It is the only public system that is: open-weights + ReAct loop + Apache-2.0 + ~10 files of plain Python + state-of-the-art on BrowseComp / HLE / WebWalkerQA / xbench-DeepSearch. Its `inference/react_agent.py` is the template we should clone for our loop — strip its Heavy mode, keep the budget triple (LLM-calls / tokens / wall-clock), keep its `<tool_call>{json}</tool_call>` + `<tool_response>{...}</tool_response>` XML/JSON hybrid surface. We add what they don't have: **visible `<think>` blocks between tool calls** + **purchase-intent specialization** + **the 7-tool spec** + **structured-data extraction (extruct)**.

That's the pitch. Tongyi proves the architecture works at the SOTA. We add the unfilled niche.

### The one correction to our prior plan

**Do not use the ReAct stop-word text protocol with Qwen3-8B.** The Qwen team's own docs explicitly warn against it — their thinking-mode tokens collide with stopword templates. Use **Hermes-format tool calls** (`<tool_call>{json}</tool_call>`) under a Thought/Action/Observation *narrative wrapper*. We keep the visible-thinking story (Thought blocks are still text the model emits between tool calls); we change the tool-call *substrate* from "free-text `Action N: Search[X]`" to a Qwen-tuned JSON envelope. This is a quiet but real change to our previous plan with the ReAct paper.

### The seven things to actually build (ranked by ROI)

1. **`fetch_url(url, goal)` with a three-tier extractor** — curl_cffi → `extruct` for JSON-LD/microdata → trafilatura with `output_format='json'` + `with_metadata=True` + `include_tables=True` → opt-in Crawl4AI render fallback for SPAs. Returns `{title, structured, text, rendered}`. **Highest single content-quality leap available.** ~60% of "best X in Y" pages embed JSON-LD ratings/prices/reviewCounts we currently throw away.
2. **`extract_entity` as its own tool** — ~30 lines of pure Python over extruct. Deterministic. No LLM call. Produces normalized `{type, name, rating, review_count, price, currency, address, url}`. Diego's spec, free.
3. **One upfront *Plan-and-Solve* Thought as the first action in every trajectory** — "User wants X in Y with constraints A,B,C. Plan: (1) search candidates, (2) verify each against constraints, (3) compare." Single highest-ROI prompt tweak in the literature (Plan-and-Solve outperformed Zero-shot CoT on GSM8K/AQuA/SVAMP). Costs ~80 tokens, prevents missing-step errors on multi-constraint queries.
4. **CRAG-style retrieval evaluator with at most one bounded re-search** — after each `search`, lightweight relevance check; if low, refine query and search once more. The single biggest failure mode of ReAct is retrieval misfires; this directly addresses it. Hard cap: one re-search per query.
5. **`think_tool` as a no-op reflection action** — lifted from LangChain Open Deep Research. The model is *instructed* to call `think_tool("...")` after every two web actions. It does nothing computationally; its purpose is to land a structured checkpoint in the trace. This is the cleanest way to make visible-thinking honest and grep-able.
6. **LangChain's hard-limit stopping rules, copied verbatim into the system prompt** — *"Simple queries: 2-3 search calls max. Complex queries: up to 5. Stop when: you can answer comprehensively OR have 3+ relevant examples OR last 2 searches returned similar info."* Free, immediately effective.
7. **SearXNG as `search` backend (with DDGS fallback)** — federated Google/Bing/DuckDuckGo with per-result `engine` provenance. JSON output (`format=json` in `settings.yml`). One Docker container. Frees us from DDGS-only and gives `rerank` a real provenance signal.

### The one big thing we should *not* add now

**No Reflexion-style verbal self-critique loop.** Documented finding: small-model self-critique is a coin flip — it harms 8B performance unless the verifier is external. Our `verify_fact` is the *grounded* alternative (claim → evidence span check), which is empirically what works. Don't bolt on a "now reflect on whether your answer is correct" step.

### The honest reframe for the pitch

**Qwen3-8B is not benchmark-competitive with frontier on GAIA / WebArena / Mind2Web — and that's fine, because our success criterion isn't benchmark.** No 8B sits in top-tier leaderboard slots; the open-source winners are RL-tuned 32Bs (OpAgent on Qwen3-VL-32B, WebDancer-32B). Our pitch should reframe success as **fidelity-on-shared-evidence + visible-thinking transparency + open-only operation**, not generalization scores. The SSR paper (Oct 2025) shows that even GPT-4o/Gemini 2.0 only hit ρ ≈ 0.90 with humans on purchase intent — there's room for an interpretable surrogate to be valuable. **If fidelity hits a ceiling at 8B, the empirically grounded next step is Qwen3-32B quantized, not 8B with more scaffolding.**

### The pitch-ready frame (steal from Anthropic verbatim)

> *"Following Anthropic's three principles in **Building Effective Agents** — simplicity, transparency, agent-computer-interface rigor — we built the simplest engineered workflow that exposes its own reasoning. A single ReAct loop over an augmented Qwen3-8B, seven curated tools with deterministic glue, bounded budgets, and visible `<think>` blocks. This is the workflow side of Anthropic's workflow-vs-agent spectrum, deliberately — purchase intent doesn't require multi-agent autonomy, and the visible reasoning trace IS the product."*

---

## 🔍 What the four agents converged on

Across A (deep-research clones), B (frameworks + engineering principles), C (tool layer), and D (academic + empirical), the convergence is unusually clean. Twelve convergences:

| # | Convergence | Sources |
|---|---|---|
| 1 | **Single ReAct loop beats multi-agent orchestration** for our scale | A (Tongyi single-loop SOTA), B (smolagents tutorial), D (HF 55% GAIA single-loop; HAL Generalist single-agent tops Claude leaderboards) |
| 2 | **Bounded budgets are mandatory and should be in code, not docs** | A (Tongyi triple: LLM-calls / tokens / wall-clock), B (Anthropic explicit; LangGraph `recursion_limit=25`; AutoGen 11 termination types), D (Plan-and-Solve, ReflAct) |
| 3 | **`think`/reflection as a first-class tool** is the cleanest way to expose reasoning | A (LangChain `think_tool` no-op), B (Anthropic transparency principle), D (ReflAct's "thought-grounded-in-goal") |
| 4 | **Hermes-format tool calls > ReAct text stop-word protocol** for Qwen3 | D (Qwen team docs explicit), A (Tongyi uses the XML/JSON hybrid) |
| 5 | **`fetch_url(url, goal)` with extractor sub-call** beats raw text dump | A (Tongyi's `visit(url, goal) → {rational, evidence, summary}`), C (Firecrawl `formats: [markdown, json, summary]`), B (eval-driven tool ergonomics) |
| 6 | **Structured-data extraction (`extruct` + JSON-LD)** is the single biggest cheap win | C (~60% of target pages have JSON-LD; <50ms), A (everyone implicitly does this) |
| 7 | **Few-shot, not zero-shot** for our 8B size | A (Tongyi & all four use zero-shot but they're 30B-A3B / GPT-4 class), D (ReAct paper: 1–6 shots), inferred for 8B |
| 8 | **Plan-and-Solve first Thought** is highest-ROI single tweak on multi-constraint queries | D (explicit), B (Anthropic prompt-chaining pattern) |
| 9 | **CRAG-style grounded retrieval verification** > Reflexion verbal self-critique | D (Stanford 2024: 8B self-critique often hurts; CRAG works) |
| 10 | **No framework adoption — pure Python, ~150 lines** | A (Tongyi 10 files), B (smolagents core ~1k lines for the same reason), C (verdict on each tool: lift patterns, not deps) |
| 11 | **Citation format: inline markdown hyperlinks** `[text](url)` at end of sentences | A (GPT-Researcher), C (LLM-friendly, no numeric bookkeeping for model to mess up) |
| 12 | **Output schema enforced only at the answer boundary**, not inside the loop | A (Tongyi `<answer>`), B (CrewAI `output_pydantic` on tasks), D (vLLM guided decoding at `stop_and_answer`) |

---

## ⚠️ Where the agents conflicted (and what to do)

| Tension | Resolution |
|---|---|
| **ReAct text protocol** (our prior plan, from the ReAct paper) vs **Hermes JSON tool calls** (Agent D, Qwen team docs) | **Hermes wins.** Qwen3 thinking tokens collide with ReAct stopwords. Keep Thought/Action/Observation as the *narrative shape* in the visible trace, but tool calls are emitted as `<tool_call>{json}</tool_call>`. The pitch narrative survives; the substrate is the supported one. |
| **8B sufficiency** (B: visible thinking is the pitch, 8B is fine) vs **8B benchmark non-competitiveness** (D: no 8B in any top-tier slot) | **Reframe success: not benchmarks, but fidelity-on-shared-evidence + transparency.** Have a documented upgrade path: if fidelity hits a wall, move to Qwen3-32B quantized, NOT to more scaffolding. |
| **Voyager-style skill library** (D notes it works) vs **library overkill for our size** (D's own caveat) | **Drop the code-skill library, keep a *trajectory cache*** keyed on `(query_template, geography, constraints)`. Storing successful Thought/Action chains for repeat queries is cheap and produces a "the agent learns from its own runs" pitch line without the executable-skill complexity. |
| **Self-correction on/off** | **On, but grounded only.** CRAG-style relevance check after each search; `verify_fact` checked against evidence spans; no free-form Reflexion outer loop. |
| **Multi-format browser tools** (C: extruct + Crawl4AI fallback; A: HF's 10-tool browser-imitation is overkill) | **C wins.** extruct everywhere (cheap, structured), Crawl4AI fallback only when curl_cffi returns thin. No browser-use, no Playwright-MCP as runtime deps. |

---

## 🏗 The architecture this implies, in concrete form

### The agent loop (~150 lines, single Python file)

```python
# surrogate/loop.py — modeled on Tongyi's inference/react_agent.py
def run(question: str, tools, system_prompt: str,
        max_steps: int = 12, max_tokens: int = 110_000,
        max_seconds: int = 600) -> Trajectory:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    t0 = time.time(); steps = 0
    while True:
        if steps >= max_steps: return _force_stop(messages, "max_steps")
        if _tokens(messages) > max_tokens: return _force_stop(messages, "tokens")
        if time.time() - t0 > max_seconds: return _force_stop(messages, "wall_clock")

        # 1. LLM call — emits free text (visible <think>) + zero or more <tool_call>{json}
        out = vllm_client.chat(messages, model="qwen3-8b",
                               enable_thinking=True, hermes_tool_call=True)
        messages.append({"role": "assistant", "content": out.content})

        # 2. Did the model stop_and_answer? then done.
        if "stop_and_answer" in out.tool_calls:
            return _finalize(messages, out.tool_calls["stop_and_answer"])

        # 3. Otherwise execute each tool call, append observation
        for tc in out.tool_calls:
            obs = tools[tc.name](**tc.args)
            messages.append({"role": "tool",
                             "content": f"<tool_response>{json.dumps(obs)}</tool_response>"})
        steps += 1
```

### The seven tools, with signatures

```python
search(query: str, n: int = 10) -> list[{title, url, snippet, engine}]
    # SearXNG -> DDGS fallback

fetch_url(url: str, goal: str) -> {
    title, structured, summary, evidence_spans, rendered  # tiered extractor
}

extract_entity(url_or_html: str, kind: str = "auto") -> {
    entities: [{type, name, rating, review_count, price, currency, address, url}]
}   # pure extruct, no LLM call

verify_fact(claim: str, evidence_url: str) -> {
    supported: bool, evidence_span: str | None, confidence: float
}

rerank(candidate_ids: list[str], criteria: list[str]) -> list[ranked]
    # pluggable; default: relevance + source diversity + completeness

check_missing_fields(candidate_id: str) -> list[str]
    # which Candidate fields are still empty; the "frontier scheduler"

think(reflection: str) -> "logged"
    # LangChain pattern — no-op for the trace

stop_and_answer(top_picks: list[Pick], reasoning_summary: str,
                citations: list[{title, url}]) -> Final
    # Pydantic schema enforced via vLLM guided decoding
```

### The system prompt skeleton (copying winning patterns verbatim)

```
You are a research assistant that answers purchase-intent questions
("best X in Y for Z") with visible reasoning and cited sources.

## Process
1. FIRST: write a 2-3 step plan in <think>...</think> covering what you'll
   search for and what constraints you'll verify.
2. Then iterate: search → fetch_url → extract_entity → verify_fact → think → repeat.
3. Call check_missing_fields before stop_and_answer.

## Hard limits
- Simple queries: 2-3 search calls maximum.
- Complex queries: up to 5 search calls.
- Stop immediately when ANY of:
  • You can answer comprehensively
  • You have 3+ relevant candidates with verified key facts
  • Last 2 searches returned similar info

## Tool format
- Free reasoning goes in <think>...</think>.
- Actions go in <tool_call>{"name": ..., "arguments": ...}</tool_call>.
- Observations arrive in <tool_response>{...}</tool_response>.
- Finish with: stop_and_answer(...) — never with free text.

## Output contract for stop_and_answer
- top_picks: 3-5 ranked items with one-line reasoning each.
- citations: every claim links to a source URL.
- Use inline markdown: ([title](url)) at end of each sentence.
```

### The dependency story

- **Add:** `extruct`, `searxng` (Docker container), `crawl4ai` (optional, lazy import).
- **Tune:** `trafilatura` config (`output_format='json'`, `with_metadata=True`, `include_tables=True`, `favor_precision=True`, `deduplicate=True`).
- **Keep as-is:** `curl_cffi`, `BeautifulSoup`, `outlines` (for stop_and_answer schema).
- **Skip:** browser-use, Firecrawl, Playwright-MCP, LangGraph, LangChain, smolagents, CrewAI, AutoGen as runtime deps. Lift patterns from their READMEs, not their code.

---

## ✅ Action items, prioritized

| Order | Item | Effort | Why now |
|-------|------|--------|---------|
| 1 | Replace ReAct-text-protocol plan with **Hermes-format tool calls + narrative Thought/Observation wrapper** | half day | One correction; everything downstream depends on it |
| 2 | Write the ~150-line loop modeled on Tongyi's `react_agent.py` | half day | Foundation for everything else |
| 3 | Add **extruct** + tune trafilatura config in `fetch_url` | 2 hours | Single biggest content-quality win |
| 4 | Implement **`extract_entity`** (extruct-based, no LLM call) | 2 hours | Diego's spec, deterministic, free |
| 5 | Add **Plan-and-Solve first Thought** to system prompt | 1 hour | Highest-ROI prompt tweak |
| 6 | Add **`think_tool`** no-op tool | 1 hour | Makes thinking visible-and-structured (LangChain pattern) |
| 7 | Add **CRAG-style relevance check** to the loop (post-search) | half day | Addresses ReAct's documented #1 failure mode |
| 8 | Add **LangChain's hard-limit stopping rules** verbatim to system prompt | 1 hour | Prevents 8B from looping |
| 9 | Implement **`verify_fact`** with deterministic grounding check | 2-3 hours | Diego's spec; grounded > Reflexion |
| 10 | Implement **`check_missing_fields`** + budget on its calls | 2 hours | Diego's spec; the frontier scheduler |
| 11 | Implement **`stop_and_answer`** with Pydantic schema via vLLM guided decoding | 2-3 hours | Diego's spec; structured-output boundary |
| 12 | Spin up **SearXNG** Docker container + JSON enabled | 1 hour | Better search than DDGS; provenance for rerank |
| 13 | Add **Crawl4AI render fallback** behind SPA-host trigger | half day | Closes TripAdvisor gap; opt-in |
| 14 | One curated few-shot trajectory (purchase-intent flavor) in prompt | 2 hours | ReAct paper showed 1-6 shots is where the win comes from |
| 15 | Trajectory cache keyed on `(template, geography, constraints)` | half day | Voyager-spirit "agent learns from own runs" without skill-library overhead |

**Total: ~5–6 focused days to v1.** Items 1–8 alone (~2.5 days) would be a complete pitch.

---

## 🎤 Pitch-ready talking points (steal verbatim)

- *"We followed Anthropic's three principles in 'Building Effective Agents' — simplicity, transparency, ACI rigor — and built the simplest engineered workflow that exposes its own reasoning."*
- *"This is the **workflow** side of Anthropic's workflow-vs-agent spectrum, deliberately. Purchase intent doesn't require multi-agent autonomy; the visible reasoning trace IS the product."*
- *"Smolagents' own tutorial states it bluntly: 'Reduce the number of LLM calls. Logic should be based on deterministic functions rather than agentic decisions.' Our seven-tool spec encodes exactly that."*
- *"The closest open-weights system to ours is Alibaba's Tongyi DeepResearch — it's state-of-the-art on BrowseComp, runs ~10 files of plain Python, and uses the same Hermes-format tool-call substrate we do. We sit in the niche they don't: visible raw `<think>` tokens between actions, purchase-intent specialization, and a structured-extraction tier (`extruct` JSON-LD) before any LLM is asked to interpret a page."*
- *"Our `stop_and_answer` is what Anthropic calls 'a stopping condition such as a maximum number of iterations' — except we promote it to a tool the model MUST call, making termination a contractual part of the protocol, not a runtime check."*
- *"We aren't chasing GAIA scores. No 8B model sits in top tiers of GAIA, WebArena, or Mind2Web — open-source SOTA at that bar is 32B with RL fine-tuning. Our success criterion is fidelity-on-shared-evidence with frontier purchase-intent answers, plus full auditability via the visible trace. We have an upgrade path: if fidelity hits a wall at 8B, the empirically grounded next step is Qwen3-32B quantized, not 8B with more scaffolding."*

---

## 📋 Where the detailed reports live

The four agent reports are in our chat transcript above this synthesis. Quick navigation:

- **Agent A — Deep-research clones.** Compared HF Open Deep Research, LangChain Open Deep Research, GPT-Researcher, Tongyi DeepResearch. Verdict: **clone Tongyi's `inference/react_agent.py`** as the loop template. Source paths and license info per project provided. Comparison table covers loop type, tool count, stop conditions, prompt strategy, thinking exposure, output format, eval, framework weight, license. Key insight: nobody currently occupies the niche of visible-`<think>` + ReAct tool calls + open weights + purchase-intent.

- **Agent B — Engineering principles + frameworks.** Anthropic *Building Effective Agents* deep-dive with verbatim quotes (this is where the pitch-ready talking points come from). Plus LangGraph, smolagents, CrewAI, AutoGen comparison. Verdict: **stay framework-free; lift patterns and quotes**. Key insight: smolagents' "Reduce LLM calls; logic in deterministic functions" tutorial IS our design rationale.

- **Agent C — Tool layer.** Hard-line verdicts on browser-use (skip), Firecrawl (skip), Crawl4AI (adopt as fallback), Playwright-MCP (adapt pattern), extruct (adopt now), trafilatura (tune config), SearXNG (adopt as `search` backend). Includes drop-in code sketches for `fetch_url` three-tier extractor, `extract_entity`, `web_search` over SearXNG. Single highest-ROI cheap win across all four reports: **extruct + JSON-LD**.

- **Agent D — Academic + empirical.** Reflexion / Toolformer / Voyager / Self-RAG / RAFT / Plan-and-Solve / CRAG / WebDancer / ReflAct + benchmark analysis (GAIA / WebArena / Mind2Web / BrowseComp / BFCL). Key methodological correction: **Hermes format, not ReAct stopwords for Qwen3**. Key honest finding: **8B is not benchmark-competitive — reframe success**. Per-paper "this means for us" interpretation provided throughout.

---

## 🧭 What I'd do next, in order

1. **Update the Google doc.** Replace the "ReAct text protocol" pieces with the Hermes-format + narrative-wrapper plan. Add the "Architecture this implies" section above wholesale. Add the upgrade-path note (Qwen3-32B quantized if fidelity insufficient). Send to Diego.
2. **Confirm Diego on the thinking-tag question** — wrap-in-display vs disable-parser. Hermes format implies he gets raw `<think>` because Qwen3 emits the tags natively when thinking mode is on and our parser is the only thing stripping them. Disabling the parser gets us the literally-raw output he asked for.
3. **Build items 1–8 from the action list** (the half-day-each items). That's two-and-a-half working days. After that the system is demonstrably the engineered workflow Diego sketched, with visible reasoning and grounded verification.
4. **Then tackle the remaining 7 items** — they're nice-to-have polish on top of a working pitch demo.
