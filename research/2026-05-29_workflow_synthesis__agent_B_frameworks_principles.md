# Agent B — Frameworks + engineering principles (full report)

**Date:** 2026-05-29
**Companion file:** `2026-05-29_workflow_synthesis.md` (the integrated golden findings)
**Brief:** Surveyed Anthropic's *Building Effective Agents* and four major agent frameworks (LangGraph, smolagents, CrewAI, AutoGen) to extract design principles defensible in a pitch — quote-level fidelity where possible.

---

# Surrogate Workflow Design — Field Survey of Engineering Principles

## 1. Executive bullet — principles the field has converged on

Ranked by direct usefulness to our Qwen3-8B + ReAct surrogate pitch:

1. **Start with the simplest thing that works; only add agency when it demonstrably helps.** Anthropic states this as a direct rule ("we recommend finding the simplest solution possible, and only increasing complexity when needed"); smolagents repeats it almost verbatim ("The best agentic systems are the simplest: simplify the workflow as much as you can"); CrewAI institutionalises it by splitting their API into Flows (deterministic) and Crews (autonomous).
2. **Distinguish workflow (predefined code paths) from agent (LLM dynamically directs its own process).** This is now the canonical Anthropic distinction and is mirrored as Flows/Crews (CrewAI), graphs/agents (LangGraph), Core/AgentChat (AutoGen).
3. **Stopping conditions are not optional — they are part of the design.** Anthropic explicitly calls out "a maximum number of iterations"; LangGraph ships a default `recursion_limit=25` and a typed `GraphRecursionError`; AutoGen ships 11 named termination conditions composable with `&`/`|`; smolagents has `max_steps` + a `final_answer` sentinel.
4. **Transparency of planning steps is a first-class principle, not a nice-to-have.** Anthropic's principle #2 ("Prioritize transparency by explicitly showing the agent's planning steps") is exactly what our visible `<think>…</think>` channel delivers.
5. **The tool surface itself is a contract worth engineering** ("agent-computer interface", ACI). Anthropic, smolagents (tutorial), and CrewAI all converge on this — clear arg formats, error messages designed for the model, examples in the docstring.
6. **Group tools; reduce LLM calls; prefer deterministic glue.** smolagents states it bluntly: "Reduce the number of LLM calls as much as you can… Whenever possible, logic should be based on deterministic functions rather than agentic decisions."
7. **Eval-driven iteration on tools and prompts beats premature framework adoption.** Anthropic's "Writing tools for agents" centers this: "Building an evaluation allows you to systematically measure the performance of your tools."

## 2. Anthropic — "Building Effective Agents" (deep dive)

This Dec-2024 post is the single most-cited piece of design guidance in the agent-engineering literature and is the spine of our pitch.

**The core architectural distinction (verbatim):**

> "Workflows are systems where LLMs and tools are orchestrated through predefined code paths."
> "Agents, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks."

Both are called "agentic systems"; the *distinction* is whether the control flow is fixed in code or chosen by the model at runtime. They define an agent operationally as: "They are typically just LLMs using tools based on environmental feedback in a loop." The basic building block is "an LLM enhanced with augmentations such as retrieval, tools, and memory" — Anthropic calls this the *augmented LLM*.

**The simplicity rule (verbatim, this is the headline quote for our pitch):**

> "When building applications with LLMs, we recommend finding the simplest solution possible, and only increasing complexity when needed. This might mean not building agentic systems at all. Agentic systems often trade latency and cost for better task performance, and you should consider when this tradeoff makes sense."

And: "For many applications, however, optimizing single LLM calls with retrieval and in-context examples is usually enough." Plus the closer: "Success in the LLM space isn't about building the most sophisticated system. It's about building the *right* system for your needs."

**The five workflow patterns** (in increasing autonomy):

| Pattern | Definition | When it fits |
|---|---|---|
| **Prompt chaining** | Sequential LLM calls; each step processes the previous output, with optional programmatic gates between steps. | Tasks cleanly decomposable into fixed subtasks where accuracy beats latency. |
| **Routing** | Classifier directs input to a specialized downstream prompt/tool. | Distinct input categories handled better by specialized prompts than one generalist. |
| **Parallelization** | Run subtasks concurrently (*sectioning*) or repeat the same task and aggregate (*voting*). | Speed gains or multi-perspective high-confidence outputs. |
| **Orchestrator-workers** | A central LLM dynamically decomposes tasks and delegates to worker LLMs. | Complex tasks where subtasks can't be hardcoded up front. |
| **Evaluator-optimizer** | One LLM produces; another critiques in a feedback loop. | Clear eval criteria exist and iteration demonstrably improves output. |

**Agent vs workflow decision rule (verbatim):**

> "Agents can be used for open-ended problems where it's difficult or impossible to predict the required number of steps, and where you can't hardcode a fixed path."

Counterpoint: "The autonomous nature of agents means higher costs, and the potential for compounding errors." Hence the recommendation to "extensively test in sandboxed environments" with "appropriate guardrails."

**Stopping conditions (verbatim):**

> "it's also common to include stopping conditions (such as a maximum number of iterations) to maintain control."

**Tool design as ACI (verbatim):**

> "One rule of thumb is to think about how much effort goes into human-computer interfaces (HCI), and plan to invest just as much effort in creating good *agent*-computer interfaces (ACI)."
> "A good tool definition often includes example usage, edge cases, input format requirements, and clear boundaries from other tools."

**Three closing principles (verbatim):**

> "1. Maintain **simplicity** in your agent's design. 2. Prioritize **transparency** by explicitly showing the agent's planning steps. 3. Carefully craft your agent-computer interface (ACI) through thorough tool **documentation and testing**."

These three principles are exactly the three things our pitch should claim — they are also why an open Qwen3-8B with visible `<think>` and a fixed 7-tool spec is *not* a regression to "less serious" tooling; it is the recommended starting point.

## 3. Framework comparison

**LangGraph** (~33k stars; production users include Klarna, Replit, LinkedIn, Uber, Elastic, BlackRock, JPMorgan; 1.x release line). The core abstraction is a **StateGraph**: nodes mutate shared typed state, edges (including conditional edges) route the next node. Persistence is a first-class concern — every state transition can be checkpointed, enabling pause/resume, time-travel debugging, and human-in-the-loop interrupts. Tool schemas come from the LangChain interop layer (Python decorators producing JSON Schema). Crucially, LangGraph treats stopping as a *graph property*: there is a default `recursion_limit=25`, and exceeding it raises `GraphRecursionError("Recursion limit of 25 reached without hitting a stop condition")`. That single design decision is the most defensible workflow primitive in the ecosystem — they bake "must hit a terminal node before N steps" into the graph contract itself. Observability is shipped via LangSmith ("visualization tools that trace execution paths, capture state transitions, and provide detailed runtime metrics"). LangGraph philosophically sits closer to *workflow*: it is "a low-level orchestration framework for building, managing, and deploying long-running, stateful agents" — the autonomy is opt-in via node logic, not baked in.

**smolagents** (~28k stars; v1.26 in May 2026; HuggingFace official). Two agent types — `CodeAgent` (writes Python code blobs as actions; their distinctive bet) and `ToolCallingAgent` (JSON/text tool calls). Whole agent core is "~1,000 lines of code" by design. Tools are Python `@tool`-decorated functions that auto-generate JSON Schema; tools can be pulled from MCP servers, LangChain, or HF Hub. The ReAct loop runs until `final_answer(...)` is called (a sentinel tool that exits the loop) or `max_steps` is exceeded; optional `planning_interval=N` injects a no-tool planning step every N actions. **Most importantly for us, smolagents' own "Building Good Agents" tutorial spells out the dominant principle**: *"The best agentic systems are the simplest: simplify the workflow as much as you can… The main guideline is: Reduce the number of LLM calls as much as you can… Whenever possible, logic should be based on deterministic functions rather than agentic decisions."* That paragraph is virtually a written endorsement of our ReAct-on-Qwen approach. Observability is in-memory step logs; no first-class tracing framework but easily Phoenix/Langfuse-instrumentable.

**CrewAI** (~52k stars; >100k certified developers; very active). Core abstractions are **Agents** (role/goal/backstory triples), **Tasks** (descriptions + expected_output), and **Crews** (teams). Most defensible piece of design for our pitch: CrewAI explicitly ships **two parallel APIs** — *Crews* for autonomous, role-based collaboration; *Flows* for "deterministic, event-driven workflow orchestration using Python decorators" with "fine-grained state management" and "predictable execution paths that satisfy enterprise requirements for auditability and reliability." Their own positioning: *"In production environments, not every use case requires full agency; some might require a higher degree of control and different levels of agency, and CrewAI addresses this with Flows, which act as the deterministic backbone of an agentic system."* That is essentially Anthropic's workflow-vs-agent distinction productised. Structured outputs are first-class via `output_pydantic` and `output_json` parameters on tasks. Observability is paywalled in CrewAI AMP; open-source has basic logging only.

**Microsoft AutoGen** (~58k stars; **currently in maintenance mode**; being replaced by "Microsoft Agent Framework" for new projects). Layered architecture: Core (event-driven message passing), AgentChat (high-level conversation API), Extensions (LLM/tool plugins). The strongest design idea worth lifting is **termination conditions as a typed, composable surface**: 11 built-ins including `MaxMessageTermination`, `TextMentionTermination("APPROVE")`, `TokenUsageTermination`, `TimeoutTermination`, `HandoffTermination`, `ExternalTermination`, `FunctionCallTermination`, and they compose with `&` and `|` operators (`max_msg_termination | text_termination`). This is the most rigorous treatment of *stopping* of any framework surveyed. Tools are callable Python functions; MCP supported. Structured outputs are not prominently documented. Maintenance-mode status is itself a finding: it argues against adopting the full framework but vindicates *its termination-condition model* as a reusable primitive.

## 4. Synthesis — what we lift, and how we defend it

Our situation: open Qwen3-8B served by vLLM, no inference APIs (eval-only access to closed models), visible thinking via `<think>…</think>`, a fixed 7-tool surface (`search, fetch_url, extract_entity, verify_fact, rerank, check_missing_fields, stop_and_answer`), ReAct text protocol. Given that, here is the rationale we can attach to each design choice — each backed by something a respected source has already said.

**(a) We're building a *workflow* with one constrained agent loop, not a free-roaming agent.** Defense: this is exactly Anthropic's recommendation. We are not building "systems where LLMs dynamically direct their own processes" with unbounded autonomy — we have a fixed tool spec, a fixed loop shape (ReAct: Thought → Action → Observation → ...), and explicit stop tokens. In Anthropic's taxonomy, this is closest to a **prompt-chaining workflow with a constrained tool-use loop**, lying between *augmented LLM* and *agent*. Quote we lean on: *"we recommend finding the simplest solution possible, and only increasing complexity when needed."*

**(b) Visible `<think>` blocks are not gimmick — they are Anthropic's principle #2.** Defense: *"Prioritize transparency by explicitly showing the agent's planning steps."* The surrogate's job is to *mimic frontier purchase-intent answers with their reasoning* — visible thinking tokens are the demo. Every framework surveyed treats tracing/observability as essential (LangSmith, Console UI, paid AMP); we get it for free by streaming the model's own thought channel.

**(c) The 7-tool fixed spec is exactly the "agent-computer interface" Anthropic argues for.** Defense: Anthropic's ACI quote — *"plan to invest just as much effort in creating good agent-computer interfaces (ACI)"* and *"a good tool definition often includes example usage, edge cases, input format requirements, and clear boundaries from other tools."* A small, curated tool spec is *easier* to engineer with this rigor than an open-ended toolbox. smolagents echoes this: *"Whenever possible, group 2 tools in one"* — our tool list already shows this curation (e.g., `verify_fact` is a deterministic fact-check primitive instead of `search` + `compare` + `decide`).

**(d) ReAct text protocol is the right substrate for an 8B local model, and avoids API lock-in.** Defense: ReAct works with any text LLM; JSON-function-calling fine-tunes vary by model. AutoGen's MCP and CrewAI's tool objects assume strong native tool-use training. Anthropic itself: *"They are typically just LLMs using tools based on environmental feedback in a loop"* — that is ReAct, full stop. Diego's allergy to lock-in becomes a strength: our format is provider-neutral.

**(e) Bounded budgets via `max_steps` + a `stop_and_answer` sentinel is industry standard.** Defense: this composes Anthropic's *"include stopping conditions (such as a maximum number of iterations)"*, LangGraph's `recursion_limit=25` default, AutoGen's `MaxMessageTermination`, smolagents' `max_steps` + `final_answer` pattern. We are doing the most-conservative version of what every framework already does. The fact that `stop_and_answer` is listed as one of the 7 tools — not implicit — is *more* explicit than any framework surveyed: stopping is a *tool the model chooses to call*, the model is contractually required to call it, and it carries the answer payload.

**(f) Eval-driven iteration on tool descriptions, not framework adoption.** Defense: Anthropic's "Writing tools for agents" companion post — *"Building an evaluation allows you to systematically measure the performance of your tools."* Our pitch should commit to a per-tool eval set (e.g., 30 examples per tool) rather than to LangChain/CrewAI/AutoGen. Frameworks add abstraction tax; for a 7-tool surrogate, the framework would weigh more than the agent loop itself (smolagents fits its whole core in ~1k lines for the same reason).

**(g) Structured outputs at the final boundary (the answer schema), not inside the loop.** Defense: CrewAI's `output_pydantic` pattern, vLLM's guided decoding / outlines support. Inside the loop we keep ReAct text (cheap, debuggable); at `stop_and_answer` we enforce a Pydantic / JSON-Schema purchase-intent payload via constrained decoding. This is the cleanest split — text reasoning visible, output validated.

**(h) What we explicitly do *not* do.** We do not build a multi-agent crew, do not use a graph framework, do not depend on MCP servers, do not adopt LangSmith or paid AMP. Each of these is justified by Anthropic's *"This might mean not building agentic systems at all"* — i.e., we are deliberately at the workflow end of the spectrum because the task (mimic frontier purchase-intent answers, with one reasoning trace per query) does not require multi-agent or persistent state. AutoGen's maintenance-mode status is itself a cautionary tale about framework adoption risk that argues for our minimal-dependency approach.

**Pitch-ready talking points** (each is something we can cite with attribution):

- "As Anthropic recommends in *Building Effective Agents*, we found the simplest solution possible: a single constrained ReAct loop over an augmented Qwen3-8B."
- "Following Anthropic's three principles — simplicity, transparency, ACI rigor — our visible `<think>` channel *is* the transparency, and our seven-tool spec *is* the ACI."
- "Our `stop_and_answer` tool is the explicit 'stopping condition (such as a maximum number of iterations)' Anthropic prescribes — except we promote it to a tool the model must call, making termination a contractual part of the protocol."
- "The closest precedent in the literature for a workflow-with-constrained-loop is CrewAI's *Flows* model and LangGraph's bounded-recursion `StateGraph` — both of which justify treating control flow as code and the LLM call as a typed node, not a free-roaming agent."
- "Smolagents' own design tutorial puts our principle bluntly: *'Reduce the number of LLM calls as much as you can. Whenever possible, logic should be based on deterministic functions rather than agentic decisions.'* That is what our seven-tool spec encodes."

## Sources

- [Anthropic — Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
- [Anthropic — Writing Tools for AI Agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [LangChain docs — GRAPH_RECURSION_LIMIT](https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT)
- [smolagents GitHub](https://github.com/huggingface/smolagents)
- [smolagents — Building Good Agents tutorial](https://huggingface.co/docs/smolagents/tutorials/building_good_agents)
- [CrewAI GitHub](https://github.com/joaomdmoura/crewAI)
- [Microsoft AutoGen GitHub](https://github.com/microsoft/autogen)
- [AutoGen — Termination conditions](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/termination.html)
