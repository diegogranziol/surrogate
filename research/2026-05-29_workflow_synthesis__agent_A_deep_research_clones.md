# Agent A — Deep-research clones (full report)

**Date:** 2026-05-29
**Companion file:** `2026-05-29_workflow_synthesis.md` (the integrated golden findings)
**Brief:** Surveyed 4 open-source deep-research projects that solve a problem isomorphic to ours, to extract patterns directly applicable to a Qwen3-8B + ReAct + visible-thinking surrogate with Diego's 7-tool spec.

---

# Research Report: Open-Source Deep-Research Projects

## 1. Executive Bullet

- **Best single project to study deeply: Tongyi DeepResearch (Alibaba-NLP).** It is the only one of the four that is *open weights + open inference code + ReAct text protocol + explicit `<answer>` finish tag + open benchmarks (BrowseComp etc.)*. Its `inference/react_agent.py` loop is ~the spec we want to clone, but their stop logic, multi-modal token budget, and "extractor" sub-prompt are worth lifting directly.
- **Best ideas to lift:** (a) LangChain's explicit `think_tool` as a *first-class reflection action* — perfect fit for our visible-thinking pitch; (b) Tongyi's `visit(url, goal)` two-arg signature with rational/evidence/summary JSON return — much richer than a raw `fetch_url(url)`; (c) LangChain's "Hard Limits: 2-3 simple / 5 complex / stop when 3+ relevant examples" stopping rules — cheap to copy into our system prompt; (d) HF's `PageUp/PageDown/Ctrl-F` browser tool family — a counterpoint we should consciously *reject* (too many tools for an 8B model).
- **Pattern to avoid:** LangChain's plan→supervisor→researcher→compress→report 4-graph + LangGraph machinery is overkill for 8B + ReAct. GPT-Researcher's fixed planner/executor pipeline also throws away the agentic loop that justifies surfacing thinking tokens.
- **Frontier convergence:** All four agree on the same primitives — *search, visit/fetch, think/reflect, finish/answer*. Our 7-tool spec (`search, fetch_url, extract_entity, verify_fact, rerank, check_missing_fields, stop_and_answer`) is broader on the "structured editing" side; nobody else exposes `extract_entity`, `verify_fact`, `check_missing_fields` as separate verbs.
- **Thinking-token exposure:** Only Tongyi treats reasoning as model-native (it's a trained agent model). The other three rely on prompt-level "reflect" steps and/or hide reasoning behind the LangGraph/LangChain stream. Our project is therefore an unfilled niche: visible raw `<think>` + ReAct tool calls + open weights.

---

## 2. Per-Project Deep Dive

### 2.1 Hugging Face Open Deep Research (`smolagents/examples/open_deep_research`)

**Repo path:** `huggingface/smolagents/tree/main/examples/open_deep_research`
**Entry point:** `run.py::main()` which calls `create_agent()` and then `agent.run(question)`.
**Last meaningful commit (to that subdir):** 2025-12-17 (`Fix run_gaia.py token_counts when managed agent is called more than once`, PR #1878).
**License:** Apache-2.0 (smolagents).

**Tool set** (registered in `scripts/text_web_browser.py`, all subclasses of `smolagents.Tool`):
- `web_search(query: str, filter_year?: str)` — Google via Serper/SerpApi.
- `visit_page(url: str)` — fetch + render; also YouTube transcripts.
- `download_file(url: str)` — for xlsx/pptx/wav/mp3/m4a/png.
- `find_archived_url(url: str, date: str YYYYMMDD)` — Wayback Machine.
- `page_up()`, `page_down()` — viewport scrolling.
- `find_on_page_ctrl_f(search_string: str)`, `find_next()` — in-page search.
- Plus a `TextInspectorTool` for re-reading the manager's stored docs, and a `visualizer` for image QA.

That's effectively 10 verbs, modeling a literal browser. This is a deliberate *imitation of human web-browsing*, not the minimal ReAct toolset.

**Loop structure:** Hierarchical multi-agent.
- `Web Browser Agent` = `ToolCallingAgent` with the 8 web tools, `max_steps=20`.
- `Manager Agent` = `CodeAgent` (executes Python in a sandbox), `max_steps=12`, holding the browser as a managed agent + `TextInspectorTool` + `visualizer`.
- Loop logic lives inside `smolagents` itself (not in this example). The `CodeAgent` writes `Thought / Code / Observation` blocks; the inner `ToolCallingAgent` does JSON tool calls.

**Prompt strategy:** Zero-shot, but with smolagents' built-in scratchpad-style scaffolding. The manager prompt is extended in `run.py` with explicit instructions for PDFs/videos and for asking the browser clarification questions. No few-shot examples in this directory.

**Thinking exposure:** smolagents prints the model's `Thought:` and `Code:` blocks to stdout by default, but they are *not* part of a `<think>` channel — they are the agent's own scratchpad. Reasoning is visible only because the framework streams it, not because the prompt enforces it.

**Output structure:** Free-form prose final answer returned from `agent.run()`. GAIA grader (`scripts/gaia_scorer.py`) is the only structured judge.

**Eval:** **GAIA validation: 55% pass@1** (vs OpenAI Deep Research 67%). Has a dedicated `run_gaia.py`.

**Framework weight:** Heavy. Adopting their patterns means adopting `smolagents` (LiteLLM model layer + CodeAgent + ToolCallingAgent + managed agents). Not a fit for our 8-line ReAct loop. Their *individual tool implementations* (especially `text_web_browser.py` and `mdconvert.py`) are however lift-able as standalone files.

---

### 2.2 LangChain Open Deep Research (`langchain-ai/open_deep_research`)

**Entry points:**
- Main graph: `src/open_deep_research/deep_researcher.py` — assembles `deep_researcher`, `supervisor_subgraph`, `researcher_subgraph`.
- Legacy implementations: `src/legacy/graph.py` (plan-and-execute), `src/legacy/multi_agent.py` (supervisor-researcher).

**Last commit:** 2026-05-26 (`Bump the uv group across 1 directory with 5 updates`). Very alive.
**License:** MIT.

**Tool set** (`src/open_deep_research/utils.py` + supervisor tools):
- `tavily_search(queries: List[str], max_results=5, topic="general"|"news"|"finance")` — async, dedupes by URL, runs an LLM-based per-page summarization with 60s timeout.
- `think_tool(reflection: str) -> str` — "Strategic reflection tool for research planning." Just returns a confirmation; the *point* is to force the model to verbalize its plan as a tool call so it lands in the trace.
- `ConductResearch(research_topic: str)` — supervisor-only delegation to a researcher subgraph.
- `ResearchComplete` — Pydantic marker class signaling the supervisor/researcher to exit.
- Optional MCP tools, Anthropic native web search, OpenAI native web search, or "none".

**Loop structure:** LangGraph state machine, three nested graphs.
- Top graph: `clarify_with_user → write_research_brief → research_supervisor → final_report_generation → END`.
- Supervisor sub-graph: `supervisor ↔ supervisor_tools`, exits on `ResearchComplete`, no-tool-calls, or `research_iterations > max_researcher_iterations` (default **6**) or concurrent-units cap (default **5**).
- Researcher sub-graph: `researcher ↔ researcher_tools → compress_research → END`. Exits on `ResearchComplete`, no-tool-calls, or `tool_call_iterations ≥ max_react_tool_calls` (default **10**).
- Compression and final report each retry up to 3× with progressive 10% character truncation on token-limit errors.

**Prompt strategy:** Zero-shot, JSON-schema-constrained at boundaries (Pydantic `ClarifyWithUser`, `ResearchQuestion`, `Summary`). Inside the loops, ordinary tool-calling LLM. Eight named prompt templates in `prompts.py`: `clarify_with_user_instructions`, `transform_messages_into_research_topic_prompt`, `lead_researcher_prompt`, `research_system_prompt`, `compress_research_system_prompt`, `compress_research_simple_human_message`, `final_report_generation_prompt`, `summarize_webpage_prompt`.

The researcher system prompt is the most ReAct-flavored thing in the codebase:
> "1) Read the question carefully — What specific information does the user need? 2) Start with broader searches… 3) After each search, pause and assess… 4) Execute narrower searches… 5) Stop when you can answer confidently."

with hard limits: "Simple queries: 2-3 search calls max. Complex queries: up to 5. Stop immediately when you can answer comprehensively OR have 3+ relevant examples OR last 2 searches returned similar info."

**Thinking exposure:** No raw model reasoning. The `think_tool` calls *are* the visible reasoning surface — those plus tool args/results are what LangSmith displays. Effectively a "fake think channel" via a no-op tool.

**Output structure:** Final report is markdown, with sequential numeric citations enforced by `final_report_generation_prompt`. The `Summary` Pydantic model is `(summary: str, key_excerpts: str)`.

**Eval:** **Deep Research Bench** (100 PhD-level tasks, 22 domains, LLM-as-judge RACE score). Reported: GPT-5 0.4943, Claude Sonnet 4 0.4401, defaults (GPT-4.1) 0.4309. Eval runner: `tests/run_evaluate.py`.

**Framework weight:** Very heavy. LangGraph + LangChain + `init_chat_model` + Pydantic schemas + LangSmith for traces. Adopting any non-trivial chunk pulls in the whole framework. Patterns are lift-able conceptually but the *code* is not.

---

### 2.3 GPT-Researcher (`assafelovic/gpt-researcher`)

**Entry point:** `gpt_researcher/agent.py::GPTResearcher.conduct_research()`.
**Last commit:** 2026-05-28 (`Merge pull request #1781 … anthropic-real-usage-cost-tracking`). Very active.
**License:** Apache-2.0.

**Tool set:** Not really "tools" in the LLM-tool-call sense — GPT-Researcher is a *Python orchestration framework*, not a ReAct agent. The LLM is called as a function. The "tools" are:
- Retrievers (Tavily / Google / DuckDuckGo / Bing / SearXNG / Serper / SerpApi / Exa / Arxiv / PubMed / etc.) selected by config.
- A scraper module (`gpt_researcher/scraper/`) for JS-enabled scraping.
- Document loaders (PDF/TXT/CSV/Excel/MD/PPT/Word).
- Optional MCP tools — these *are* exposed as LLM-callable tools via `generate_mcp_tool_selection_prompt`.
- Image generation via Google Gemini.
- Vector store for embeddings.

**Loop structure:** Fixed planner-executor-publisher pipeline, not an iterative agent. In `agent.py::conduct_research()`:
1. `choose_agent()` — picks agent persona + role prompt.
2. `research_conductor.conduct_research()` (in `skills/researcher.py`) — branches on source type (URLs / web / local / hybrid / Azure / LangChain docs / vector store).
3. Inside the web branch: `generate_sub_queries()` (LLM call) → `asyncio.gather()` over sub-queries → each runs retriever + scraper → optional `curate_sources` LLM filter.
4. Optional image generation.
5. `write_report()` — single LLM call with all gathered context, produces the markdown.

There is a separate `_handle_deep_research()` path using `DeepResearchSkill` with recursive subtopic exploration (configurable breadth/depth). That one *is* iterative but still hand-coded recursion, not an LLM-driven loop.

**Prompt strategy:** Zero-shot, ~19 prompt template functions in `gpt_researcher/prompts.py`: `generate_search_queries_prompt`, `generate_report_prompt`, `curate_sources`, `generate_resource_report_prompt`, `generate_outline_report_prompt`, `generate_deep_research_prompt`, `generate_summary_prompt`, `generate_subtopics_prompt`, `generate_subtopic_report_prompt`, `generate_report_introduction`, `generate_report_conclusion`, `auto_agent_instructions`, plus MCP variants. Subqueries are returned as JSON lists parsed with `json_repair.loads()`.

**Thinking exposure:** None at the model level. The pipeline emits progress logs (sub-query list, sources hit, costs). Reasoning is not surfaced because there isn't really any *agentic* reasoning — the LLM is called as a stateless writer/extractor at each pipeline step.

**Output structure:** Markdown report, target ≥2000 words, with **inline markdown-hyperlink citations**:
> `([in-text citation](url))` at end of sentence/paragraph, plus full APA-style references list at the end.

Export to PDF/Word/Markdown.

**Eval:** No formal benchmark numbers in the repo. The README claims bias-reduction via "scraping multiple sites per research and selecting most frequent information," but no quantitative eval against GAIA/BrowseComp/DRB.

**Framework weight:** Medium. Pure Python + asyncio + a swappable LLM provider layer. Not LangGraph. Adopting their *prompt library* (especially `generate_report_prompt`'s citation format) is cheap; adopting the architecture means giving up the agentic loop.

---

### 2.4 Tongyi DeepResearch (`Alibaba-NLP/DeepResearch`)

**Entry point:** `inference/react_agent.py::_run()`. Launch script: `inference/run_react_infer.sh`. Multi-task runner: `inference/run_multi_react.py`.
**Last commit:** 2026-02-27 (`fix bug`). Less weekly churn than the LangChain repos but actively maintained.
**License:** Apache-2.0.
**Model:** Tongyi-DeepResearch-30B-A3B — MoE, 30.5B total / 3.3B active. Open weights on Hugging Face.

**Tool set** (each in its own file under `inference/`, registered via `@register_tool`):
- `search(query: List[str])` (`tool_search.py`) — batched Google search via Serper, top-10 each. Detects Chinese characters and switches `gl`/`hl` accordingly. 5-retry.
- `visit(url: str | List[str], goal: str)` (`tool_visit.py`) — fetches page (Jina.ai), then calls an LLM-side summarizer with the `EXTRACTOR_PROMPT`. Returns labeled `rational / evidence / summary` text.
- `google_scholar(query: List[str])` (`tool_scholar.py`) — Serper `/scholar` endpoint, returns titles/snippets/dates/citation counts/PDF URLs.
- `PythonInterpreter(code: str inside <code>…</code>)` (`tool_python.py`) — SandboxFusion remote sandbox, 50s timeout, 8-retry, multi-endpoint load balance.
- `parse_file(files: List[str])` (`tool_file.py`) — PDF/DOCX/PPTX/TXT/CSV/XLSX/DOC/ZIP/MP4/MP3 via Dashscope; video routed to a separate `VideoAgent`.

Five tools total. Notice the *batched array signature* on `search` and `google_scholar` — different from our spec.

**Loop structure:** Single-agent ReAct loop, hand-written in `react_agent.py::_run()`:
- Init: system prompt + current date appended.
- While `num_llm_calls_available > 0`:
  - `call_server(messages)` → assistant response.
  - Parse: if `<answer>…</answer>` present → set `termination='answer'`, break. Else if `<tool_call>{json}</tool_call>` present → execute, append `<tool_response>{result}</tool_response>` to messages.
  - Special-case Python: code lives in `<code>…</code>` *inside* the `<tool_call>` block.
- Other termination paths:
  - `num_llm_calls_available <= 0` → `'exceed available llm calls'`.
  - `token_count > 110*1024` → force final-answer generation.
  - Wall-clock > 150 minutes → `'No answer found after 2h30mins'`.

Tool-calling format is **XML-tagged JSON inside text**: `<tool_call>{"name":..., "arguments":...}</tool_call>` → `<tool_response>...</tool_response>`. This is the same surface we want, modulo our `<think>` prefix.

**Prompt strategy:** Zero-shot. `inference/prompt.py` holds two strings:
- `SYSTEM_PROMPT` opens with: *"You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic."* Lists the 5 tools (OpenAPI-style schema). Requires final answer in `<answer></answer>`. No `<think>` convention in the system prompt — reasoning is whatever the model naturally emits between `<tool_call>` blocks.
- `EXTRACTOR_PROMPT` (used inside `visit`): produces JSON `{rational, evidence, summary}` from raw page + goal.

No few-shot examples in `prompt.py`.

**Thinking exposure:** Native. The model was *trained* (CPT + SFT + RL via custom GRPO with token-level policy gradients, leave-one-out advantage, negative-sample filtering) to emit free-text reasoning between tool calls. So everything the model says outside `<tool_call>` and `<answer>` *is* the visible thinking. There is no `<think>` tag because there's no need to delimit it. **This is the closest analogue to what we want from Qwen3-8B**, except Qwen3 already emits `<think>` natively and we want to keep that.

**Output structure:** Free-form text inside `<answer>…</answer>`. The framework does not enforce citations; citation behavior is whatever the trained policy learned. The `IterResearch / Heavy mode` variant additionally "reconstructs a streamlined workspace using only the most essential outputs from the previous round" and integrates findings into an evolving report between rounds.

**Eval:** **State-of-the-art open** on agentic search benchmarks. Reported numbers: HLE 32.9, BrowseComp 43.4, BrowseComp-ZH 46.7, WebWalkerQA 72.2, xbench-DeepSearch 75. Beats OpenAI-o3 and DeepSeek-V3.1 on these tasks per the technical report (arXiv 2510.24701).

**Framework weight:** Minimal. Pure Python in `inference/`, ~10 files. Tool registry via a simple `@register_tool` decorator. No LangChain, no LangGraph, no smolagents. **This is the architectural template most compatible with our 8-line ReAct loop.**

---

## 3. Comparison Table

| Dimension | HF Open Deep Research | LangChain Open Deep Research | GPT-Researcher | Tongyi DeepResearch |
|---|---|---|---|---|
| **Loop type** | Hierarchical multi-agent (CodeAgent + ToolCallingAgent) | LangGraph nested state machines (3 graphs) | Fixed planner→executor→publisher pipeline | Single-agent ReAct loop, hand-written |
| **Loop location** | `smolagents` library internals; configured in `run.py::create_agent` | `src/open_deep_research/deep_researcher.py` | `gpt_researcher/agent.py::conduct_research` + `skills/researcher.py` | `inference/react_agent.py::_run` |
| **Tool count** | 8 web + 2 manager = 10 | 4 (tavily_search, think_tool, ConductResearch, ResearchComplete) + MCP | N/A (Python orchestration; MCP tools optional) | 5 (search, visit, google_scholar, PythonInterpreter, parse_file) |
| **Tool-call format** | smolagents JSON tool calls (browser) + Python code blocks (manager) | LangChain tool-calling JSON | None at the LLM layer; LLM is called as a function | XML-tagged JSON: `<tool_call>{...}</tool_call>` / `<tool_response>...</tool_response>` |
| **Stop condition** | `max_steps=12` (mgr) / `20` (browser) | `ResearchComplete` tool OR iteration caps (6/10) OR no-tool-calls | End of pipeline (no loop) | `<answer>` tag OR LLM-call budget OR 110k token OR 150-min wall clock |
| **Prompt strategy** | Zero-shot, smolagents scaffolding | Zero-shot + Pydantic schemas at boundaries; 8 named templates | Zero-shot, 19 template fns | Zero-shot, single SYSTEM_PROMPT + EXTRACTOR_PROMPT |
| **Few-shot?** | No (in this directory) | No | No | No |
| **Thinking exposure** | smolagents prints `Thought:` blocks (scaffolding-driven) | `think_tool` no-op tool calls = visible reflections | Progress logs only; no model reasoning | Native free-text reasoning between `<tool_call>`s (RL-trained) |
| **Output format** | Free-form prose | Markdown report w/ sequential numeric citations | Markdown ≥2000 words w/ inline `([cite](url))` + APA refs | Free-form text in `<answer>...</answer>`; no enforced citations |
| **Eval** | GAIA validation: 55% pass@1 | Deep Research Bench (RACE): defaults 0.4309 | None reported | HLE 32.9, BrowseComp 43.4, BC-ZH 46.7, WebWalkerQA 72.2, xbench-DS 75 |
| **Framework weight** | Heavy (smolagents) | Very heavy (LangGraph + LangChain + LangSmith) | Medium (pure Python + asyncio + pluggable providers) | **Light** (pure Python, ~10 files in `inference/`) |
| **Open-only?** | Needs OpenAI o1 default; can swap via LiteLLM | Defaults to OpenAI/Anthropic/Tavily; supports OpenRouter/Ollama | Default is OpenAI; supports OpenAI-compatible APIs | **Yes, fully open**: 30.5B-A3B weights on HF + open inference code |
| **License** | Apache-2.0 | MIT | Apache-2.0 | Apache-2.0 |
| **Last commit** | 2025-12-17 (in subdir) | 2026-05-26 | 2026-05-28 | 2026-02-27 |
| **Alive?** | Yes | Yes (most active) | Yes (most active) | Yes |

---

## 4. Insights: What to Lift, Adapt, Avoid (given our constraints)

### What to lift

**(a) Tongyi's loop skeleton, verbatim, as the template for `surrogate/loop.py`.** Their `_run()` is roughly 100 lines of Python and does everything we need: per-iteration LLM call, parse `<answer>` for finish, parse `<tool_call>` for action, append `<tool_response>` to the chat history, check three budgets (LLM calls / tokens / wall clock). Our 7-tool spec drops cleanly into their `@register_tool` pattern. Adapt the tool-call surface from `<tool_call>{json}</tool_call>` to whatever Qwen3-8B emits most cleanly when constrained — our `<think>` blocks will live *between* `<tool_call>`s exactly like Tongyi's free-text reasoning lives there. The XML/JSON hybrid (XML tag wrapper, JSON arguments inside) is a known-stable surface across Qwen-family models and matches how Qwen3 was instruction-tuned.

**(b) LangChain's `think_tool` pattern, adapted to our `check_missing_fields`.** LangChain made the very useful move of *promoting reflection to a tool*. In their prompt, the model is told: "use `think_tool` before calling ConductResearch" and "after each search tool call, use `think_tool` to analyze the results." Even though `think_tool` is a no-op that just echoes the reflection string, it accomplishes two things: forces the model to verbalize a checkpoint, and lands the checkpoint in the trace where users (and graders) can see it. Our spec already has `check_missing_fields` which is morally the same thing — a "stop and self-audit" tool. Adopt LangChain's explicit prompt language: *"Call `check_missing_fields` after every 2 `search` or `fetch_url` actions; if it returns ≥3 missing fields, do not call `stop_and_answer`."* This is also a clean way to keep our visible-thinking story honest: the `<think>` block contains stream-of-consciousness, but the `check_missing_fields` tool call gives a *structured* checkpoint that pairs with it.

**(c) Tongyi's `visit(url, goal)` two-argument signature with rational/evidence/summary JSON return, as our `fetch_url` implementation.** A naive `fetch_url(url) -> markdown` dumps an entire page into context, which kills 8B models. Tongyi's `visit` takes a `goal` string and runs an LLM-side extractor (their `EXTRACTOR_PROMPT`) that returns `{rational, evidence, summary}` in JSON. Same idea is in LangChain's `summarize_webpage_prompt` and GPT-Researcher's `curate_sources`. For our open-only deployment, the extractor can be the *same* Qwen3-8B model called as a sub-routine — no second model needed. The cost is one extra LLM call per fetch, the benefit is bounded context per `fetch_url` Observation. Adapt: change our `fetch_url(url) -> str` signature to `fetch_url(url: str, goal: str) -> {rational, evidence, summary}`. This also gives `verify_fact` something concrete to ingest — the evidence field with span-level quotes from the page.

**(d) LangChain's hard-limit stopping rules, copied straight into our system prompt.** Their `research_system_prompt` includes: *"Simple queries: 2-3 search tool calls maximum. Complex queries: up to 5. Stop Immediately When: You can answer comprehensively OR have 3+ relevant examples OR last 2 searches returned similar information."* This is gold for an 8B model that will otherwise loop forever. Lift verbatim, swap "search tool calls" for "search + fetch_url combined."

**(e) GPT-Researcher's citation format for our final report:** `([in-text citation](url))` markdown hyperlinks at end of sentences/paragraphs, plus a full references list. It's the most LLM-friendly citation format (no numeric bookkeeping for the model to mess up) and matches what end users actually want to see for purchase-intent answers (you want a clickable link next to "best supplements to buy"). Drop the APA-formatted bibliography; for purchase-intent, "Source: [domain.com](https://…)" is enough.

### What to avoid

**(a) LangChain's 4-graph LangGraph architecture.** clarify → brief → supervisor → researcher → compress → report is great when your output is a 5000-word PhD-level report and you have a $20 token budget. For "best Italian restaurant in Tashkent" with Qwen3-8B served by vLLM, it's massive overkill and obscures the visible-thinking story (where do `<think>` tokens go when there are 4 distinct agents each running its own LLM?). Stay single-agent, stay flat, stay legible.

**(b) HF's 10-tool browser-imitation tool set.** `page_up`, `page_down`, `find_on_page_ctrl_f`, `find_next`, `find_archived_url`, `download_file`, `visit_page` — this is a model trying to imitate a human at a keyboard. An 8B model doesn't have the planning depth to use 10 tools well, and the additional tools muddy our spec. Our 7-tool spec already errs on the side of more verbs than Tongyi's 5; do not add browser-emulation tools. If a page is too long, that's `fetch_url(url, goal)`'s problem, not the LLM's.

**(c) GPT-Researcher's "the LLM is called as a function" architecture.** Yes, it produces nice reports. No, it does not produce *agentic reasoning*. Our entire pitch is that the surrogate's thinking is visible. A fixed planner→executor→publisher pipeline has no thinking to show; the only model output is the final markdown. Adopting their pipeline structure deletes our differentiator. We can steal their *prompt strings* (especially the citation format) but not their control flow.

**(d) Heavy/IterResearch's "rewrite workspace between rounds."** Tongyi's Heavy mode discards prior context and rebuilds a streamlined workspace each round, which is a clever test-time scaling trick *for a trained-for-this 30.5B model*. Qwen3-8B has not been RL-tuned for this and will lose track. Stick to the vanilla ReAct loop with full message history (plus a context-budget check that summarizes old observations if needed).

### What to consciously bet on (open-only, visible-thinking, 7-tool ReAct)

The four projects collectively prove the *frontier convergence* on `search / fetch / reflect / finish` as the minimal toolset. Our 7-tool spec is a legitimate extension: `extract_entity`, `verify_fact`, `check_missing_fields`, `rerank` are not exotic — they correspond to operations all four projects perform *implicitly* (Tongyi's extractor does extract_entity, LangChain's `tavily_search` summarizer does rerank, GPT-Researcher's `curate_sources` does verify_fact, etc.). By promoting these to first-class tools we (a) make the reasoning trace richer and more grep-able, (b) give the 8B model smaller, more checkable steps, and (c) match the structured-editing aesthetic that the ReAct paper (Yao 2022) showed wins when retrieval is solid.

The one space *no* current open project occupies is **visible raw `<think>` tokens + ReAct tool calls + open weights + purchase-intent-flavored final answers**. Tongyi is closest but exposes reasoning as untagged free text. LangChain fakes thinking via `think_tool` no-ops. HF surfaces thinking via smolagents scaffolding. GPT-Researcher has no thinking at all. Qwen3-8B + a 100-line ReAct loop + our 7 tools + a system prompt that says *"emit your reasoning inside `<think>…</think>` between tool calls; tool calls go in `<tool_call>{json}</tool_call>`; finish with `stop_and_answer`"* — this is a clean, defensible niche.

### Concrete recommendations for the surrogate

1. Model `surrogate/loop.py` on `inference/react_agent.py` from Tongyi. Strip Heavy mode, keep the budget triple (LLM-calls / token / wall-clock).
2. Use `<think>…</think>` for free reasoning (Qwen3 native), `<tool_call>{json}</tool_call>` for actions, `<tool_response>{...}</tool_response>` for observations, and `stop_and_answer` (a tool that just sets a flag) instead of `<answer>` tags — keeps everything in one schema.
3. Implement `fetch_url(url, goal)` with an extractor sub-call using the same Qwen3-8B; return `{rational, evidence, summary}` JSON. Borrow Tongyi's `EXTRACTOR_PROMPT` verbatim with attribution.
4. Adopt LangChain's hard-limit stopping rules in the system prompt verbatim.
5. Use GPT-Researcher's inline `([cite](url))` markdown citation format in the final `stop_and_answer` payload.
6. Skip framework dependencies entirely. Pure Python + the OpenAI-compatible client pointing at vLLM. Single file, ~150 lines for the loop, one file per tool. Matches our 8-line-ReAct-loop pitch.
7. For eval, target BrowseComp + a small purchase-intent eval set; this gives us a published Tongyi comparison point for the technical narrative.

### Source paths referenced

- `/huggingface/smolagents/examples/open_deep_research/run.py`
- `/huggingface/smolagents/examples/open_deep_research/scripts/text_web_browser.py`
- `/huggingface/smolagents/examples/open_deep_research/scripts/gaia_scorer.py`
- `/langchain-ai/open_deep_research/src/open_deep_research/deep_researcher.py`
- `/langchain-ai/open_deep_research/src/open_deep_research/prompts.py`
- `/langchain-ai/open_deep_research/src/open_deep_research/state.py`
- `/langchain-ai/open_deep_research/src/open_deep_research/utils.py`
- `/langchain-ai/open_deep_research/src/open_deep_research/configuration.py`
- `/assafelovic/gpt-researcher/gpt_researcher/agent.py`
- `/assafelovic/gpt-researcher/gpt_researcher/skills/researcher.py`
- `/assafelovic/gpt-researcher/gpt_researcher/actions/query_processing.py`
- `/assafelovic/gpt-researcher/gpt_researcher/prompts.py`
- `/Alibaba-NLP/DeepResearch/inference/react_agent.py`
- `/Alibaba-NLP/DeepResearch/inference/prompt.py`
- `/Alibaba-NLP/DeepResearch/inference/tool_search.py`
- `/Alibaba-NLP/DeepResearch/inference/tool_visit.py`
- `/Alibaba-NLP/DeepResearch/inference/tool_scholar.py`
- `/Alibaba-NLP/DeepResearch/inference/tool_python.py`
- `/Alibaba-NLP/DeepResearch/inference/tool_file.py`
- Tech report: arXiv 2510.24701 (Tongyi DeepResearch Technical Report)
