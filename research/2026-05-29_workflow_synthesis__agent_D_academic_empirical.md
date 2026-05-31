# Agent D — Academic + empirical evidence (full report)

**Date:** 2026-05-29
**Companion file:** `2026-05-29_workflow_synthesis.md` (the integrated golden findings)
**Brief:** Surveyed the academic literature (Reflexion, Toolformer, Voyager, Self-RAG, RAFT, Plan-and-Solve, CRAG, WebDancer, ReflAct) and benchmark leaderboards (GAIA, WebArena, Mind2Web, BrowseComp, BFCL) to ground our workflow choices in empirical evidence rather than plausible-sounding intuition.

---

# Surrogate Workflow Design: Empirically Grounded Choices for a Qwen3-8B Purchase-Intent ReAct Agent

## 1. Executive Bullet (ranked by impact for our specific use case)

1. **Lean on Qwen3-8B's native Hermes-style tool-calling, not ReAct text protocol with stopwords.** The Qwen team explicitly warns against ReAct stopword-based templates for Qwen3 because thinking-mode tokens can collide with stop sequences; Hermes format (`<tool_call>{...}</tool_call>`) is the supported path and gives F1 ~0.93 on tool selection.
2. **Add a *retrieval evaluator + corrective re-search* loop (CRAG/Self-RAG style) — not a generic Reflexion verbal critique.** Small-model self-critique is documented to *harm* small-model performance unless the verifier is strong or external. For purchase intent, the evaluator can be a deterministic relevance check on the search snippet plus the existing `verify_fact` tool — cheap, grounded, and the highest-leverage single addition.
3. **Skip multi-agent planner+workers; use a *single ReAct loop with a lightweight upfront plan step* (Plan-and-Solve flavor).** Open Deep Research (HF) hit 55% GAIA with a *single* code-agent loop; on agent benchmarks where 8B-class models compete (BFCL, GAIA), simple single-loop with strong tool-calling beats orchestration bloat. Planning a few sub-queries before searching reduces missing-step errors at near-zero cost.
4. **Don't fine-tune for tool calls yet (Toolformer is the wrong tier of investment).** Qwen3-8B already ships agent-grade tool calling. The empirical gap closes more by *better tools and verification* than by SFT on tool traces. WebDancer-style RL on top of SFT is the next step *after* the surrogate is shipped, not before.
5. **8B is competitive *if* the workflow is engineered, but the win condition is process visibility + fidelity to frontier purchase-intent answers, not benchmark scores.** No 8B model sits in top tiers of GAIA/Mind2Web/WebArena — winners are frontier (Claude 4.x, GPT-5) or RL-fine-tuned 32B+ (OpAgent on Qwen3-VL-32B). Our pitch must reframe: *we're not chasing GAIA, we're chasing visible-thinking surrogate fidelity on "best X in Y."*
6. **Pre-empt three failure modes**: (a) hallucinated tool names/args, (b) observation flooding from large fetch results, (c) verifier-stall loops where the model re-verifies what it already verified. Concrete remedies: deterministic tool-name routing, per-tool circuit breakers, and a hard cap on `verify_fact` invocations per claim.
7. **Voyager-style "skill library" pays off only when tasks repeat with structure.** Purchase intent has high template repetition (every query is "best X in Y, for Z constraints") — caching a small library of *query templates* and *successful fact-verification chains* is worth doing, but a full executable-skill library is overengineering at our scale.

---

## 2. Per-Paper Summaries

### Reflexion (Shinn et al. 2023, NeurIPS)

**Core technique.** Verbal reinforcement learning: after a failed trial, the agent writes a free-text self-reflection that becomes context for the next trial. No weight updates — the "policy" lives in episodic memory. Three components: Actor (the LLM agent), Evaluator (computes reward signal from env), Self-Reflection generator (verbal critique).

**What it adds over ReAct.** ReAct is single-shot; Reflexion adds multi-trial learning. The memory buffer holds prior trajectories + reflections, letting the agent course-correct without fine-tuning.

**Empirical headline.** 91% pass@1 on HumanEval (vs GPT-4's 80% baseline at the time); substantial AlfWorld and HotpotQA gains. Wins when there's a *clear, fast feedback signal* (compile/run, ground-truth match). Loses when the env signal is ambiguous or when the model's self-critique is itself wrong.

**Implementation gotcha.** Trials are expensive — each failed attempt + reflection + retry can be 3-5x the tokens of a one-shot ReAct call. The memory buffer also grows; it's a sliding-window in practice. Small models are documented to do *worse* with self-critique unless the verifier is external (Stanford / 2024 work).

**For us.** Don't bolt on full multi-trial Reflexion. Instead, take the *idea* — a single end-of-trajectory consistency check that fires `check_missing_fields` and at most one re-run if a required field is empty. Keep critique grounded in the field schema, not free-form. *Verbal critique by an 8B model on itself is a known regression risk.*

### Toolformer (Schick et al. 2023)

**Core technique.** Self-supervised fine-tuning: the base LM proposes tool-call positions in plain text, executes them, keeps the calls whose output reduces perplexity on the next tokens, and fine-tunes on the filtered corpus. Tools: calculator, QA, two search engines, translator, calendar.

**What it adds over prior work.** First demonstration of an LM self-teaching API use from a handful of demos per API — no human annotation.

**Empirical headline.** GPT-J 6.7B + Toolformer matches or beats GPT-3 175B on LAMA, math, and QA *zero-shot*. Single tool per query.

**Implementation gotcha.** No chained tool calls (each call is independent), no interactive correction on errors, and the self-supervised filter requires running every candidate call — expensive at corpus scale. Also: the tool set is fixed at training time; adding a new tool means re-running the pipeline.

**For us.** *Don't replicate this.* Qwen3-8B already ships with tool-calling instruction tuning that handles chained, interactive calls (Toolformer can't). Toolformer is now historical context — its DNA lives in every modern tool-use SFT mix. The actionable takeaway is just: *a handful of high-quality demonstrations per tool is enough for a competent base model to use it well in-context*; we don't need huge prompt examples per tool.

### Voyager (Wang et al. 2023)

**Core technique.** GPT-4 powers a Minecraft agent with three components: (1) automatic curriculum generating progressively harder tasks, (2) ever-growing skill library of executable code (Mineflayer JS), embedding-retrieved by description, (3) iterative prompting incorporating env feedback + execution errors + self-verification critic.

**What it adds.** First open-ended *lifelong* LLM agent. Skills are compositional (complex skills are functions calling simpler ones), interpretable, and persistent across episodes.

**Empirical headline.** 3.3x more unique items, 2.3x distance, 15.3x faster tech-tree milestones vs prior SOTA. Generalizes to new worlds where alternatives fail.

**Implementation gotcha.** GPT-4 dependency is heavy; the iterative loop is multi-call. Skill retrieval works because Minecraft has clean modular goals; less obvious whether it works for tasks without natural decomposition. Average 160 iterations to discover 63 items — not cheap.

**For us.** Don't build a code-execution skill library. *But* the "embedding-indexed library of successful trajectories" is directly applicable: a cache keyed on `(product_category, geography, constraint_set)` storing the search/verification trajectory that worked. Cheap, retrievable, doesn't require a code interpreter, and accelerates repeat queries. This is what gives us *pitch-friendly* properties: "the agent learns from its own runs."

### Self-RAG (Asai et al. 2023)

**Core technique.** Fine-tunes a 7B/13B model to emit special "reflection tokens" during generation: `[Retrieve]` decides whether to retrieve at all, `[ISREL]` rates passage relevance, `[ISSUP]` rates whether claim is supported, `[ISUSE]` rates overall utility. Tokens are integrated into the standard decoding loop.

**What it adds over vanilla RAG.** Adaptive — retrieve on demand, not every turn. Critic is part of the generator, no separate verifier model. Multi-aspect critique (relevance + support + utility) instead of binary accept/reject.

**Empirical headline.** 7B Self-RAG beats ChatGPT and retrieval-augmented Llama2-chat on open-domain QA, fact verification, and citation accuracy. Higher cited-precision on long-form generation.

**Implementation gotcha.** Requires fine-tuning the generator to emit reflection tokens — non-trivial training pipeline. The critic model labels training data for SFT, adding complexity.

**For us.** We don't need to fine-tune Qwen3-8B with reflection tokens. *But* the four-axis critique pattern (Should I retrieve? Is this snippet relevant? Does it support my claim? Is the overall answer useful?) is the right shape for our `verify_fact` and `check_missing_fields` tools. Implement it as explicit tool calls instead of token emissions — same semantics, no fine-tuning required, and *more visible* in our thinking trace (which is the pitch).

### RAFT (Zhang et al. 2024, Berkeley)

**Core technique.** SFT recipe: train on (question, gold doc + distractor docs, chain-of-thought answer that ignores distractors). Teaches the model to (a) read the doc set, (b) cite only the relevant ones, (c) reason explicitly.

**What it adds over vanilla RAG / SFT.** Hardens against imperfect retrieval — the trained model learns to ignore noisy passages instead of being misled by them. CoT-style answers improve faithfulness.

**Empirical headline.** Up to +35.3% F1 on HotpotQA, +6.58% Recall@3 on SQuAD; competitive with GPT-3.5 baselines using small fine-tuned base.

**Implementation gotcha.** Requires curated training data: (q, gold, distractors, CoT). For purchase intent specifically, defining "gold" is hard (no single right answer).

**For us.** Almost certainly post-MVP. *But* if we ever bottleneck on fidelity, the RAFT recipe — train the model on distractor-injected retrieval traces from a frontier teacher — is the right path. For now, the takeaway is *include distractors in our few-shot prompts*: show the model what unhelpful search results look like and what to do with them. This is free at the prompt level.

### Plan-and-Solve (Wang et al. 2023, ACL)

**Core technique.** Two-phase zero-shot prompt: "First devise a plan to solve the problem, then carry out the plan step by step." PS+ variant adds explicit instructions to "pay attention to calculation" and "extract relevant variables."

**What it adds over Zero-shot CoT.** Reduces missing-step errors (the model forgets to do something) by forcing decomposition before execution.

**Empirical headline.** GPT-3 175B + PS prompting beats Zero-shot-CoT on GSM8K, AQuA, SVAMP, ARC by large margins; comparable to 8-shot CoT.

**Implementation gotcha.** Adds tokens. On *easy* tasks the planning step is bloat. Plan quality varies with model strength — weak models produce shallow plans that don't help.

**For us.** Pre-pend a "draft a 2-3 step plan" Thought to the trajectory. Don't bolt on a separate planner agent. The plan IS the first Thought in our ReAct loop. This costs ~50 tokens and demonstrably reduces missing-step errors. *Single highest ROI design tweak.*

### Recent (2024-2025) Deep Research Agents

**CRAG (Yan et al. 2024).** Adds a lightweight retrieval evaluator — if retrieved docs are scored low-confidence, the system rewrites the query and re-searches (web). Compatible with any RAG pipeline. For us: this is the model for our retrieval loop — `search` → evaluate snippet → re-search if low confidence → `fetch_url` only when confident. Cheap and pitch-friendly.

**WebDancer (Wu et al. 2025).** Pipeline: data construction → trajectory sampling → SFT cold start → DAPO RL. Outperforms vanilla ReAct on GAIA / WebWalkerQA across Qwen-2.5-7B, 32B, QwQ-32B. Shows: *RL-on-top-of-SFT is what makes small models competitive on info-seeking benchmarks*. For us: noted as the upgrade path post-MVP.

**HF Open Deep Research.** 55% GAIA with a *single* code-agent loop (CodeAgent + text browser + text inspector borrowed from Magentic-One). Beats more elaborate multi-agent setups in 24-hour engineering time. For us: validates the "single-loop ReAct with strong tools" architecture.

**ReflAct (2025).** Argues the core ReAct failure is *thoughts not grounded in goal state*. Instead of adding reflection on top, redesigns the backbone to evaluate "does my current trajectory align with the goal?" at each step. For us: adopt the spirit — every Thought should mention what fact the agent is trying to establish toward the final purchase-intent answer.

---

## 3. Benchmark Analysis (what the leaderboards actually reveal)

**Top systems on GAIA (frontier-skewed).** HAL Generalist Agent runs the top 6 slots on Claude Sonnet 4.5 / Opus 4.1 — *single-agent* designs, no exotic orchestration. HF Open Deep Research (multi-component) and OPS-Agentic-Search (92% on the public split, likely overfit) round out the top. *No 8B model anywhere near the top.*

**Top systems on WebArena.** OpAgent (Qwen3-VL-32B + RL) at 71.6% — the *only* open-source model in top tier, and it uses a Planner-Grounder-Reflector-Summarizer pipeline *plus* RL fine-tuning. Without RL, Qwen3-VL doesn't break top tier. IBM CUGA at 61.7% (single-agent). Older entries: Jace.AI 57.1%, ScribeAgent 53%, ORCHESTRA 52.1%. *Pattern: planning + RL wins; pure ReAct prompting plateaus.*

**Top systems on Mind2Web 2 / Online-Mind2Web.** OpenAI Deep Research at 50-70% of human performance. *Online-Mind2Web 2025 finding is brutal: most commercial agents underperform the SeeAct academic baseline from early 2024.* Translation: benchmark scores have not improved much; the field is mostly recycling architecture. For us: the gap on Mind2Web 2 is in long-horizon real-time browsing — not our target use case.

**Top systems on BrowseComp.** Deep Research at 51.5%; enabling browsing on GPT-4o moved it from 0.6% to only 1.9%. *Architecture (planner + reasoning + persistence) matters more than tool access alone.* This is the strongest single piece of evidence that *a workflow with verification and persistence beats more tools*.

**Top systems on BFCL v4 (function calling).** GLM-4.5 at 70.9%, Claude Opus 4.1 at 70.4%. Qwen3-8B specifically: F1 0.933 on tool selection (~84s latency) — close to GPT-4's 0.974 (~5s). *We are not constrained by 8B's tool-calling competence; we're constrained by reasoning depth on multi-hop queries.*

**Common patterns of winners.**
- *Frontier models with light scaffolding* (HAL on Claude) > *open-source 8B with heavy scaffolding*. This is the inconvenient truth.
- *RL-fine-tuned 32B models* (OpAgent, WebDancer) is the only open-source recipe that hits frontier-adjacent scores. SFT alone is not enough.
- *Single agents win at the top* on GAIA. Multi-agent orchestration is a tax for most tasks; it pays off only in cleanly decomposable, parallel-search settings.
- *Code agents > JSON agents* for multi-step tool composition (HF's finding: 1 code step vs 20 JSON steps).
- *Verification/reflection* helps when paired with strong models or external verifiers; it *hurts* small models that critique themselves.

**Failure modes the benchmarks expose that we'd hit on purchase intent.**
- **Hallucinated tool names / args.** 155 hallucination events per 200 tasks in a real ReAct deployment (Towards Data Science 2025). Our `verify_fact` and `extract_entity` are abstract enough that the model could call them with malformed args.
- **Observation flooding.** Search/fetch returns dump huge HTML/JSON into context. Without compaction, the trajectory drowns.
- **Verifier loops.** The agent calls `verify_fact` on the same claim multiple times because it forgot it already did. (Observed in the same TDS analysis.)
- **Goal drift.** ReflAct's central finding: thoughts diverge from goal. On a "best espresso machine under $500 in Germany" query, the agent can chase tangents (history of espresso machines, etc).
- **Retrieval misfires.** ReAct's original paper documented this — when search returns wrong info, the agent commits to a wrong answer. This is the *single biggest risk* for purchase intent because the wrong product recommendation is the wrong product recommendation.
- **8B reasoning depth on multi-constraint queries.** "Best X in Y for Z under $W with feature Q" has 4-5 simultaneous constraints. 8B models suffer "logic drift" on long multi-constraint chains (cited in MCPVerse / Llama 8B literature).

---

## 4. Synthesis: Cumulative Evidence for Our Specific Situation

We are building a Qwen3-8B-vLLM ReAct surrogate for "best X in Y"-style purchase-intent answers, with visible thinking as the pitch. Four design questions, answered by the cumulative literature:

### (a) Should we add a verification / self-correction pass?

**Yes, but as targeted CRAG/Self-RAG-style grounded checks, not free-form Reflexion verbal critique.** The Stanford 2024 finding ("small models need strong verifiers to self-correct") and the more general 2025 finding that "meta-cognitive interventions often harm small-model performance" both rule out giving Qwen3-8B a generic "now reflect on whether your answer is correct" prompt. What works: structured, grounded checks against the *retrieved evidence*, not against the model's own opinion. Our `verify_fact` tool is already this. We should add:

- A relevance-evaluator step on each search result (CRAG): is this snippet about the product/category the user asked about? If not, re-search with a refined query.
- A field-coverage check (`check_missing_fields`) before `stop_and_answer`: did we establish every required field (price, geography, key features)? If not, one more search loop.
- *No more than one* re-attempt per check, with a hard cap. This eliminates the verifier-stall failure mode without bloating cost.

**What we should not do:** add a Reflexion-style "trial N+1" outer loop. Costs 2-3x tokens and 8B verbal critique is a coin-flip on whether it improves vs degrades the answer.

### (b) Does planning before acting help at our scale, or is it bloat?

**A lightweight upfront plan helps; a separate planner agent is bloat.** Plan-and-Solve's empirical message — explicit plan reduces missing-step errors on multi-hop queries — applies cleanly to purchase intent. Multi-constraint queries ("best espresso machine under $500 in Germany with milk frothing") are exactly where missing-step errors show up: the model verifies price, forgets geography, never checks the feature.

**Implementation.** First Thought in the ReAct trajectory is: "The user wants [product class] in [geography] with [constraints C1, C2, ...]. I will: (1) search for top candidates, (2) verify each against constraints, (3) compare." Cost: ~50-80 tokens. Benefit: directly maps to thinking-trace visibility (the pitch).

**What we should not do:** spin up a separate planner agent that emits a DAG. Both the benchmark evidence (HAL Generalist beats multi-agent on GAIA, HF Open Deep Research with single loop beats orchestration) and the multi-agent benchmarking paper (hierarchical only beats single-loop at 1.4x cost on a Pareto frontier where 89% of accuracy can be recovered with hybrid simple setups) say multi-agent is overhead for our size.

### (c) Should we fine-tune for tool use (Toolformer), or lean on Qwen3-8B's built-in?

**Lean on Qwen3-8B's built-in, but use Hermes format, not ReAct stopwords.** Qwen3's tool calling F1 of 0.933 on a fair benchmark is enough. Toolformer's recipe is now baked into every modern instruction-tuned model — re-running it on Qwen3-8B will not buy a meaningful gain over a few well-chosen in-context demonstrations.

**Important architectural choice:** the Qwen team's own docs explicitly recommend Hermes-style `<tool_call>{...}</tool_call>` and warn against ReAct-style stopword templates because thinking-mode tokens can trip the stopword. Our "ReAct-style text protocol" pitch needs adjustment: keep the Thought/Action/Observation *narrative structure for visibility*, but emit actions as Hermes tool-calls under the hood. The thinking trace stays human-readable; the tool dispatch becomes reliable.

**Upgrade path (post-MVP).** WebDancer's recipe — SFT cold start on agentic trajectories, then DAPO RL — is what makes 7B/8B models competitive on benchmarks. If we ever need a fidelity step-change against a frontier teacher, that's the next investment. *Not now.*

### (d) Is 8B competitive on purchase-intent fidelity, or do we need 32B+?

**8B is *not* benchmark-competitive against frontier, but our success criterion is not benchmark score.** No 8B sits in any top-tier leaderboard slot. Even RL-tuned WebDancer-7B underperforms its 32B sibling. OpAgent's 71.6% WebArena win is on Qwen3-VL-*32B*. The honest read: open-source SOTA on agent benchmarks is 32B+ with RL.

But our pitch is not "we beat GPT-5 on GAIA." It's "open-source surrogate with visible thinking, mimicking frontier purchase-intent answers." That reframes the success criterion to *correlation/fidelity with frontier answers on a closed query distribution* (purchase intent in specific categories), not generalization to arbitrary agentic tasks. The closest data point is the SSR purchase-intent paper (Oct 2025): GPT-4o and Gemini 2.0 Flash hit ρ ~ 0.90 with humans using free-text + semantic-similarity rating. We can target that *correlation* with Qwen3-8B if:

- We constrain the query distribution to purchase-intent shapes (not open-domain).
- We engineer the workflow tight (CRAG-style retrieval verification, plan-first, hard-capped loops).
- We show the *thinking process* — even an imperfect 8B answer with a transparent, well-engineered trace is more valuable as a research/audit tool than a black-box frontier answer.

If fidelity is too low at 8B (likely on heavy multi-constraint queries), the pragmatic step is **Qwen3-32B** quantized — not 8B with heavier scaffolding. The literature is clear: at the 8B tier, scaffolding has diminishing returns; the next jump is base-model capability.

### Failure modes to pre-emptively guard against

1. **Hallucinated tool name/arg.** Mitigation: deterministic tool dispatch (model emits intent, our wrapper validates and rejects malformed calls before execution); strict JSON schema on Hermes tool calls.
2. **Observation flooding.** Mitigation: every `fetch_url` and `search` result is summarized/truncated by our wrapper to <500 tokens before re-injection; raw payload optionally accessible via a `read_more` reference.
3. **Verifier stall.** Mitigation: hard cap on `verify_fact` (max 3 per fact, max 8 total); each call must include the claim being verified, and our wrapper rejects duplicates.
4. **Goal drift.** Mitigation: the upfront plan (Plan-and-Solve style) is *replayed in the system prompt* on every turn; the model can't forget the constraint list because it's in front of it.
5. **Retrieval misfires (ReAct's documented Achilles heel).** Mitigation: CRAG-style relevance evaluator (lightweight LLM call or even rule-based keyword overlap) on each search result; if low, re-query with a refined search instead of fetching.
6. **8B logic drift on long chains.** Mitigation: cap trajectory at ~12 steps; if not converged by then, force `stop_and_answer` with an explicit "best partial answer + caveats" template.
7. **Wrong-product confidence.** Mitigation: every recommendation must cite at least two verified facts; `stop_and_answer` validates this in the wrapper.

### Bottom line

The literature converges on: *strong base model > orchestration; targeted grounded verification > free-form reflection; plan-then-act on multi-constraint queries; ride the model's native tool format*. For our pitch — visible-thinking purchase-intent surrogate on Qwen3-8B — the highest-leverage decisions are (1) Hermes tool calls under a Thought/Action/Observation narrative wrapper, (2) a one-line Plan-and-Solve first Thought, (3) a CRAG-style relevance check + one bounded re-search, (4) hard-capped `verify_fact` + `check_missing_fields` before `stop_and_answer`, and (5) honest framing that 8B trades benchmark score for transparency, with a clear upgrade path to Qwen3-32B if fidelity demands it.

---

## Sources

- [Reflexion (Shinn et al. 2023)](https://arxiv.org/abs/2303.11366) / [PDF](https://arxiv.org/pdf/2303.11366)
- [Toolformer (Schick et al. 2023)](https://arxiv.org/abs/2302.04761) / [PDF](https://arxiv.org/pdf/2302.04761)
- [Voyager (Wang et al. 2023)](https://voyager.minedojo.org/) / [arXiv](https://arxiv.org/abs/2305.16291)
- [Self-RAG (Asai et al. 2023)](https://arxiv.org/html/2310.11511) / [GitHub](https://github.com/akariasai/self-rag)
- [RAFT (Berkeley 2024)](https://www.superannotate.com/blog/raft-retrieval-augmented-fine-tuning)
- [Plan-and-Solve (Wang et al. 2023, ACL)](https://aclanthology.org/2023.acl-long.147/)
- [CRAG (Yan et al. 2024)](https://arxiv.org/abs/2401.15884)
- [WebDancer (Wu et al. 2025)](https://arxiv.org/html/2505.22648.pdf)
- [Deep Research Agents Survey (2025)](https://arxiv.org/html/2506.18096v2)
- [Open Deep Research (Hugging Face)](https://huggingface.co/blog/open-deep-research)
- [HAL GAIA Leaderboard](https://hal.cs.princeton.edu/gaia)
- [Awesome Agents — GAIA/WebArena/BFCL/Tau2 Leaderboard](https://awesomeagents.ai/leaderboards/agentic-ai-benchmarks-leaderboard/)
- [Online-Mind2Web (2025)](https://github.com/OSU-NLP-Group/Online-Mind2Web)
- [BrowseComp (OpenAI 2025)](https://arxiv.org/html/2504.12516v1)
- [Qwen3 Function Calling docs](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
- [Docker Local LLM Tool Calling eval](https://www.docker.com/blog/local-llm-tool-calling-a-practical-evaluation/)
- [ReflAct (2025)](https://arxiv.org/html/2505.15182v2)
- [Small Language Models Need Strong Verifiers (2024)](https://arxiv.org/pdf/2404.17140)
- [Your ReAct Agent Is Wasting 90% of Its Retries (TDS 2025)](https://towardsdatascience.com/your-react-agent-is-wasting-90-of-its-retries-heres-how-to-stop-it/)
- [LLMs Reproduce Human Purchase Intent via SSR (2025)](https://arxiv.org/html/2510.08338v1)
