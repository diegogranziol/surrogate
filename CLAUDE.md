# CLAUDE.md — project rules

## Prime directive: always output the full stack trace

**This project exists to expose and learn from model thinking.** It is not a polished API.
Every byte the model produced — every `<think>` block, every tool call, every tool result,
every system prompt, every reasoning field — is the **product**. Curation is the
anti-product.

### Hard rules

- **NEVER strip `<think>...</think>` blocks** from any output, snapshot, log, or
  printed view. The reasoning IS the artefact, often more interesting than the
  final answer.
- **NEVER replace raw output with a summary, table, or "answer-only" view** in the
  user-facing reply. A one-line note *after* the full dump is fine; a curated view
  *instead of* the dump is not.
- **NEVER write `re.sub(r'<think>.*?</think>', '', …)`** or any equivalent that
  removes thinking, tool calls, tool results, system prompts, or reasoning fields.
- **NEVER add `[:N]` truncations to LLM content / reasoning / tool_result fields**
  when emitting or saving. (Stdout previews like a tqdm-style 80-char status line
  are tolerable IF the full content is still written to the trace files.)
- When a run completes, **paste the entire raw stdout in the chat reply**, in code
  fences. If the harness truncates it, save it to a file and link the path; do not
  silently drop bytes.

### Where truncation IS allowed (and why)

| File | What's truncated | Why it's OK |
|---|---|---|
| `surrogate/tools/fetch.py` | URL-fetched text capped at `MAX_CHARS` before being **returned to the model** | This protects the model's context window. The full HTML/text is fetched; only what the model sees is capped. If you want more model context, raise `MAX_CHARS`. |
| `surrogate/logger.py::_slug()` | Question text → 40-char filesystem-safe slug for directory names | This is a filename, not content. The full question text is preserved in `session_start.user_question`. |

If you add a new truncation, document it in the table above with a justification.
Otherwise, do not add one.

### Where output is dumped verbatim (do not change)

- `surrogate/logger.py::TraceLogger.event()` writes every event to
  `trace.jsonl` and `transcript.md`, untouched.
- `aggregate_traces.py::render_stage1`, `render_stage2` dump every field of every
  event verbatim into the master `all-answers.md` (system prompt, full TOOL_SPECS
  JSON, raw `function.arguments` strings, full tool results, full
  `reasoning_content`, full `content` including any inline `<think>` blocks).
- `run_batch.py::_section_for()` uses those renderers; it does not curate.

### What "the model" emits

- **Qwen2.5-7B** (stage 1, tool-calling): emits `content` + `tool_calls`. No
  separate reasoning field on this model. Tool calls' `function.arguments` is a
  raw JSON string the model generated — log it verbatim.
- **Nemotron-Super-49B-FP8** (stage 2, reasoning): emits `<think>...</think>`
  **inline in `content`**, not in a separate `reasoning` field. vLLM's
  `--reasoning-parser deepseek_r1` errored on Nemotron's tokenizer, so we run
  *without* a reasoning parser. The renderer dumps `content` verbatim — keep it
  that way; do NOT split the think block away into a sub-section that the
  rendered MD then hides.

### When in doubt

Dump everything. Add a one-line note after. The user has called this out
repeatedly ("FULL FUCKING STACK TRACE", "WE ARE NOT TRYING TO BE AN API WE ARE
TRYING TO EXPOSE THE THINKING AND LEARN FROM IT"). Treat it as load-bearing.
