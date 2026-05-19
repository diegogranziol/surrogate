# reasoning_dumps/

Curated, human-readable dumps of full verbatim model reasoning + answers,
pulled from `logs/two-stage-<ts>/` bundles for easy review/sharing.

These are convenience copies. The authoritative verbatim record is always the
bundle's `trace.jsonl` / `transcript.md` (never edited, per CLAUDE.md).

## Naming convention

```
tryNN_<ISO-datetime>_<surrogate-model>_<question-slug>.md
```

- `tryNN`   — attempt number, zero-padded (try01, try02, …). A "try" = one
              notable run we deliberately keep (first success, a tuned variant,
              a backtest comparison, etc.).
- ISO datetime — `YYYY-MM-DDTHH-MM-SS` (matches the source bundle timestamp).
- surrogate-model — e.g. `qwen3-8b`. For reference models use the model id
              (e.g. `glm-4.6`) so surrogate vs reference dumps are distinguishable.
- question-slug — short kebab-case of the question.

Each file contains: metadata header (question, datetime, attempt, model,
source bundle, Stage 1 tool calls), then every Stage 2 sample's full
`reasoning_content` and `content`, verbatim, no truncation.

## Index

- `try01_2026-05-19T17-50-15_qwen3-8b_italian-restaurant-tashkent.md`
  — first clean 5/5 Phase-0 run. Q: "which restaurant is the best for
  italian food in Tashkent". Surrogate qwen3-8b. Stage 1 used web_search +
  fetch_url (Wanderlog).
- `try02_2026-05-19T22-42-53_glm-4.6_italian-restaurant-tashkent.md`
  — first z.ai REFERENCE call (Phase 1 smoke). glm-4.6, no evidence (answered
  from internal knowledge, web_search=0). NOT yet comparable to try01 — Phase 2
  will re-run the reference WITH the surrogate's gathered evidence.
