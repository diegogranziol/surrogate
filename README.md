# surrogate

A two-stage agentic pipeline built to **expose model thinking**, not hide it.
Stage 1 (a small tool-calling model) gathers web evidence. Stage 2 (a larger
reasoning model) reads only the raw tool outputs and produces an answer with
its `<think>` chain of thought visible alongside.

Everything the model produced is logged verbatim: system prompts, the full
JSON tool spec, every tool call (with the raw `function.arguments` string the
model emitted), every tool result, every `<think>...</think>` block, every
sample. See [`CLAUDE.md`](CLAUDE.md) for the project's prime directive on
this — curation is the anti-product.

## What's in here

```
surrogate/
├── agent.py         tool-calling loop (stage 1)
├── two_stage.py     stage 1 + stage 2 orchestrator (stage 2 runs 5 samples:
│                    1× greedy + 4× T=1.0 by default)
├── logger.py        per-session TraceLogger → logs/<ts>-<slug>/{trace.jsonl, transcript.md, meta.json}
├── llm.py           OpenAI-compatible client factory (talks to vLLM)
├── swap.py          remote vLLM model-swap helper (host-aware: local vs SSH)
├── replay.py        load/render past sessions
└── tools/
    ├── search.py    web_search() — ddgs (UK region) / Tavily if key set
    └── fetch.py     fetch_url() — curl_cffi (iOS Safari TLS impersonation)
                      + trafilatura, BeautifulSoup fallback for SPA pages
aggregate_traces.py  produce a master all-answers.md from a directory of bundles
run_batch.py         iterate questions through the two-stage pipeline, write
                     runs/batch-<ts>/all-answers.md with full traces
run_two_stage.py     CLI: python run_two_stage.py "your question"
run_chat.py          single-turn REPL
merge_batches.py     stitch two batch master MDs into one
data/                question lists for batch runs
notebooks/           demo notebook
```

## Quick start

Assumes you have two vLLM OpenAI-compatible endpoints serving:
- Stage 1: `qwen2.5-7b` at `http://127.0.0.1:8000/v1` (with
  `--enable-auto-tool-choice --tool-call-parser hermes`)
- Stage 2: a reasoning-capable model (e.g. `nemotron-super-49b-fp8`) at
  `http://127.0.0.1:8001/v1`

Set up:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then edit STAGE{1,2}_BASE_URL / STAGE{1,2}_MODEL
```

Run:

```bash
python run_two_stage.py "Which restaurant is the best in Oxford, UK for steak?"
# → writes a per-question bundle under logs/two-stage-<ts>/

python run_batch.py data/questions.txt
# → writes runs/batch-<ts>/all-answers.md (one master MD with every full trace)
```

## Reading the output

Each question produces a bundle:

```
logs/two-stage-<ts>/<ts>-<slug>-stage1/
  ├── trace.jsonl     one event per line — session_start, llm_request,
  │                   llm_response, tool_call, tool_result, final_answer
  └── transcript.md   the same events rendered as markdown
logs/two-stage-<ts>/<ts>-<slug>-stage2/   (same shape, with N samples)
logs/two-stage-<ts>/stage2-input.md       verbatim user message stage 2 saw
```

Each `llm_request` event contains the **entire** `messages` array sent to the
model at that step plus the full `tools` array. Each `llm_response` event
contains the raw `content`, every `tool_call` with its unparsed
`function.arguments` string, the `reasoning` field if the model produced one,
and token usage. Tool events log the resolved args and the full text returned
to the model.

`run_batch.py` aggregates many bundles into a single master `all-answers.md`
with a TOC plus each question's full trace in order.

## Models used in this repo's traces

- **Stage 1**: `Qwen/Qwen2.5-7B-Instruct` — native hermes tool calling, no
  separate reasoning field (CoT, if any, is inline in `content`).
- **Stage 2**: `nvidia/Llama-3_3-Nemotron-Super-49B-v1-FP8` — emits
  `<think>...</think>` inline in `content`. vLLM's `--reasoning-parser
  deepseek_r1` errored on its tokenizer, so we run without a parser and
  preserve the tags verbatim. (Qwen3-8B and Qwen3-32B also work as stage-2
  models and put their reasoning in a separate `reasoning` field via
  `--reasoning-parser qwen3`.)

## License

MIT. See [`LICENSE`](LICENSE).
