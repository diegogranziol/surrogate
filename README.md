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
├── two_stage.py     stage 1 + stage 2 orchestrator. 5 stage-2 samples
│                    (1 greedy + 4 T=1.0). Supports use_rag=True and a
│                    sibling run_with_dom_pair(question, url_a, url_b).
├── reference.py     z.ai/GLM reference client (Anthropic-compatible API)
│                    used by the backtest as the "frontier" comparator.
├── backtest.py      surrogate vs reference scoring (top-3 soft-match)
├── rag.py           user-RAG: ingest URLs / text, sentence-transformers
│                    embeddings, SQLite store under userlinks/, cosine
│                    retrieval, evidence-block builder for stage 2.
├── logger.py        per-session TraceLogger → logs/<ts>-<slug>/{trace.jsonl, transcript.md, meta.json}
├── llm.py           OpenAI-compatible client factory (talks to vLLM)
├── swap.py          legacy single-endpoint model-swap helper (bypassed
│                    when SURROGATE_SKIP_SWAP=1, which is the default for us)
├── replay.py        load/render past sessions
└── tools/
    ├── search.py    web_search() — ddgs (UK region) / Tavily if key set
    ├── fetch.py     fetch_url() / fetch_html() — curl_cffi browser-TLS
                      impersonation, trafilatura + BeautifulSoup fallback
    └── dom.py       crawl_dom(url) / compare_doms(a, b) — structured DOM
                      extract (headings/lists/tables/numeric signals/links),
                      richer than fetch_url for ranked-list pages

run_two_stage.py     CLI: surrogate over a single question
run_backtest.py      CLI: surrogate vs GLM on a question file, append to
                     backtests/store.jsonl + write a verbatim run MD
rescore_starter8.py  re-score existing store entries with the current metric
ingest_links.py      CLI: ingest URLs/text into the user-RAG store
streamlit_app.py     UI: Ingest / Documents / Ask / Compare two URLs

scripts/
├── keep_tunnel.sh   self-healing SSH tunnel (Mac:8000 → box:8000), env-var
│                    parameterized (TUNNEL_HOST/USER/PORT/KEY)
└── dom_demo.py      run + dump a DOM-pair demo into reasoning_dumps/

aggregate_traces.py  produce a master all-answers.md from a directory of bundles
run_batch.py         iterate questions through the two-stage pipeline, write
                     runs/batch-<ts>/all-answers.md with full traces
run_chat.py          single-turn REPL
merge_batches.py     stitch two batch master MDs into one

data/                question lists (starter8.txt + the original 101)
reasoning_dumps/     curated, tracked verbatim artifacts (try01, try02, ...)
backtests/           gitignored: store.jsonl + run/rescore MDs
userlinks/           gitignored: the user-RAG SQLite store + embeddings
notebooks/           demo notebook
```

## Quick start

The setup we use: **one** vLLM endpoint serving **one** model (Qwen3-8B) for
both Stage 1 and Stage 2. `SURROGATE_SKIP_SWAP=1` short-circuits the legacy
model-swap path entirely.

### 1. The GPU box — launch vLLM

Any NVIDIA box with CUDA-capable drivers (3090 / 4090 / H100 / B200…). On the
box (ninja must be on PATH for B200's flashinfer JIT — see Ops notes below):

```bash
PATH=$HOME/.local/bin:$PATH setsid nohup python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B --served-model-name qwen3-8b \
  --host 127.0.0.1 --port 8000 --max-model-len 16384 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  </dev/null > ~/vllm.log 2>&1 &
```

### 2. The Mac — env + tunnel + deps

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then edit ZAI_API_KEY etc.

# Self-healing tunnel (env-var parameterized; defaults point at the current box):
nohup ./scripts/keep_tunnel.sh > /tmp/keep_tunnel.log 2>&1 &
curl -sf http://localhost:8000/v1/models   # should list qwen3-8b
```

Override the tunnel target via env vars (no script edit needed):

```bash
TUNNEL_HOST=... TUNNEL_USER=... TUNNEL_PORT=... TUNNEL_KEY=... \
  ./scripts/keep_tunnel.sh
```

### 3. Run something

```bash
# Single question through the surrogate:
python run_two_stage.py "best italian food in Tashkent"

# Batch over a question file:
python run_batch.py data/questions.txt

# Surrogate vs GLM reference, top-3 soft-match scoring:
python run_backtest.py data/starter8.txt
python rescore_starter8.py 8     # re-score existing store entries, no GPU

# User-RAG: ingest your own links / pasted text, then ask:
python ingest_links.py https://wanderlog.com/list/.../tashkent
python ingest_links.py --text "I went to Affresco — truffle pasta was the best."
python -c "from surrogate.two_stage import run_two_stage; \
           print(run_two_stage('best italian in Tashkent', use_rag=True).stage2.answer)"

# DOM crawler taking two URLs:
python scripts/dom_demo.py "best italian in Tashkent" URL_A URL_B 03

# Streamlit UI (Ingest / Documents / Ask / Compare two URLs):
streamlit run streamlit_app.py
# 'Retrieve only' mode in the Ask tab works WITHOUT the GPU.
# Surrogate runs and Compare-two-URLs need the tunnel up.
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

## Backtest (surrogate vs GLM reference)

The metric is **top-3 soft-match overlap** with a GLM-judged "same product /
adjacent version / variant tier OK; different generation NO" rule (lives only
in `surrogate.backtest.soft_match_top3` — swap the rule by editing one
function). Thinking is captured verbatim but NOT used in the score
(production assistant APIs don't expose theirs — a one-sided comparison).

For each question the harness does three model calls — surrogate two-stage,
GLM with the SAME evidence the surrogate gathered (the fair fidelity test),
and bare GLM (documents the memory-vs-web gap) — then top-3-extracts each
free-text answer and runs the judge. Everything is appended to
`backtests/store.jsonl` (date-stamped) and rendered verbatim into
`backtests/run-<ts>.md`.

A starter-8 backtest currently scores **8/8 questions with ≥1 top-3 match on
shared evidence** (16/28 total items overlap); bare GLM drops to 2/28 —
confirming that disagreement when sources differ is an evidence artifact,
not a model-fidelity signal.

## User-RAG (bring your own links)

`surrogate/rag.py` is a small local-only retrieval store: SQLite under
`userlinks/` (gitignored), sentence-transformers (`all-MiniLM-L6-v2`)
embeddings, brute-force cosine retrieval. `ingest_url` fetches the full page
text (uses `fetch_url(max_chars=None)`), chunks it (paragraph-aware char
windows), embeds, stores. At query time the top-k chunks are appended to
Stage 2's evidence pack via `run_two_stage(use_rag=True)` — so the surrogate
reasons over BOTH the live-web evidence Stage 1 gathered AND the user's
provided sources, with explicit "User-source N" framing in the prompt.

`try04_…_rag_….md` shows this end-to-end: a hand-written "I went to Affresco,
truffle pasta was the best" note correctly outranked the aggregated review
sites in Stage 2's final pick and was cited verbatim in the answer.

## DOM-pair flow (two URLs in)

`surrogate.tools.dom.crawl_dom(url)` walks the HTML tree itself (vs
`fetch_url` which uses trafilatura's "main article" path) and emits headings,
ordered/unordered lists, tables, numeric signals (ratings / review counts /
prices found near anchor text), and top links. **Much richer for ranked-list
pages** (e.g. on a Wanderlog "50 best restaurants in Tashkent" page the
ranked names show up as 30 `<h2>` headings — invisible to trafilatura).

`compare_doms(a, b)` packs two URLs side-by-side, and
`two_stage.run_with_dom_pair(question, url_a, url_b)` bypasses Stage 1 and
runs Stage 2 directly over that packed evidence. The Streamlit
**Compare two URLs** tab is the user-facing entry point. The presentation
demo lives in `scripts/dom_demo.py`.

## Switching the GPU box (no script edits)

`box_config.json` (gitignored) holds host / user / port / key / local_port +
a "last_used" timestamp. The Streamlit **Settings** tab is the friendly
interface: pick a preset (Mithril / vast.ai / Custom), edit any field,
click *Save & Restart tunnel*. The bash keeper is restarted with the new
env vars; status badges and a *Test endpoint* button confirm health.
Same control is available programmatically:

```python
from surrogate.box import save_settings, restart_tunnel
save_settings({"host": "NEW_IP", "user": "ubuntu", "port": 22,
               "key": "/path/to/key.pem", "local_port": 8000})
restart_tunnel()
```

## Ops notes (real things that bit us)

- **Mithril/cloud cgroup cleanup kills detached processes when SSH ends.**
  `tmux new-session -d` doesn't survive. Use `setsid nohup ... </dev/null
  >log 2>&1 &` so the process escapes the SSH session cgroup. (The vLLM
  launch line in Quick Start already does this.)
- **B200 + vLLM 0.18 needs `ninja` on PATH** because the flashinfer TRT-LLM
  attention backend JIT-compiles a kernel on first run. `ninja` may exist on
  the box but not on the default PATH — prepend `$HOME/.local/bin` (or
  whatever venv-equivalent) before launching vLLM. Without it, vLLM dies
  silently right after the "Using TRTLLM prefill attention" log line.
- **Plain `ssh -fN -L` tunnels drop on idle / network blips and kill long
  batch jobs.** Always use `scripts/keep_tunnel.sh` (auto-respawns with
  `ServerAliveInterval=15`).
- **Ubuntu 24+ PEP 668 blocks `pip install`** into system Python without
  `--break-system-packages` (Mithril ships with vLLM/torch already in user
  site-packages, so usually fine — only matters when adding missing tooling
  like ninja or build-essential).
- **vast.ai overlay disk was 32 GB total** which is uncomfortably tight for
  vLLM (8 GB) + Qwen3-8B (16 GB) + HF download overhead. Mithril's 230 GB is
  much more comfortable.

## Models used in this repo's traces

- **Surrogate (both stages, 2026-05+)**: `Qwen/Qwen3-8B` — hermes tool
  calling, `--reasoning-parser qwen3` extracts thinking into a separate
  `reasoning_content` field (clean, no `<think>` tags left in `content`).
  Fits any 24 GB+ GPU.
- **Reference**: GLM-4.6 (or glm-5.1) via z.ai's Anthropic-compatible API;
  thinking is returned as Anthropic-style `thinking` content blocks when
  `thinking={"type":"enabled", ...}` is accepted (it is, on the GLM Coding
  Plan).
- **Original repo defaults (legacy, kept for reference)**:
  `Qwen/Qwen2.5-7B-Instruct` (Stage 1) +
  `nvidia/Llama-3_3-Nemotron-Super-49B-v1-FP8` (Stage 2). Qwen3-32B also
  works as a stage-2 model and puts its reasoning in a separate `reasoning`
  field via `--reasoning-parser qwen3`.

## License

MIT. See [`LICENSE`](LICENSE).
