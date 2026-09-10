# FutureSim real-simulator integration

Runs the **actual FutureSim simulator** (their orchestration code, unmodified
agent scaffolds) inside the BeliefLens date-gated sandbox, so the `futuresim`
harness in E1-real is the real system, not a prompt shape.

## Provenance

- Upstream: https://github.com/OpenForecaster/futuresim
- Pinned commit: `908322f3791afd43bcc540daad7524de3e4d054c`
- Local clone lives at `vendor/futuresim/` (not committed; reproduce with the
  steps below).

## What we add (three small pieces, everything else untouched)

| File | Where it goes in the clone | What it does |
|---|---|---|
| `openai_direct.py` | `inference/openai_direct.py` | `OpenAIDirectInference(OpenRouterInference)`: OpenAI's `/v1/chat/completions` is wire-compatible with OpenRouter's, so we subclass and strip OpenRouter-only payload fields (`usage`, `provider`), drop `temperature`/`top_k` (rejected by gpt-5 reasoning endpoints), and map `reasoning: {effort}` → `reasoning_effort`. |
| `bm25_sqlite.py` | `agents/search_tools/bm25_sqlite.py` | Implements their `BaseSearchTool` over the BeliefLens index `data/news.db` (SQLite FTS5/BM25, 339,801 articles, 2025-12-01→2026-03-31). Date gating (`a.date <= max_date`) is enforced in SQL in addition to their SearchHandler's own ceiling. Replaces their 48 GB LanceDB hybrid index. Note: `futuresim_agents` is a path alias for `agents/` — the file must live under `agents/search_tools/`. |
| `run_forecast_sim.patch` | `scripts/run_forecast_sim.py` | Adds `openai` to `--provider` choices (factory + single-agent mode) and `bm25` to `--search_backend` choices with a construction branch (index path from `BL_INDEX` env, default `../../data/news.db`). |

## Reproduce

```bash
git clone https://github.com/OpenForecaster/futuresim vendor/futuresim
cd vendor/futuresim
git checkout 908322f3791afd43bcc540daad7524de3e4d054c
git apply ../../futuresim_integration/run_forecast_sim.patch
cp ../../futuresim_integration/openai_direct.py inference/openai_direct.py
cp ../../futuresim_integration/bm25_sqlite.py agents/search_tools/bm25_sqlite.py
```

## Questions file

`fsim_data/questions.jsonl` carries the two E1 questions (hockey_usa /
hockey_can) in their custom-dataset schema (`environment/datasets/custom.py`
field synonyms: qid, title, background, resolution_criteria, answer_type,
resolution_date, ground_truth_answer via `answer`, options).

Their CLI semantics: `--start_date/--end_date` define the **resolution
window** (must bracket `resolution_date` = 2026-02-22 for a question to be
active); the simulated day loop starts `--lookback_days` before it.

## Validated structurally (no LLM calls)

```bash
python scripts/run_forecast_sim.py --sim_name bl_structcheck --no_inference \
  --matching exact --search_backend bm25 \
  --dataset custom --dataset_path ../../fsim_data/questions.jsonl \
  --start_date 2026-02-22 --end_date 2026-02-22 --lookback_days 1
```

Output confirms: custom dataset loads, both questions initialize prediction
histories, BM25 tool ready over 339,801 articles, exact matching selected
(no matcher model), day loop enters. `articles_base` is left empty — corpus
staging is then a no-op and the agent's *only* information channel is the
search tool, i.e. our date-gated index.

## Live smoke (needs OPENAI_API_KEY)

```bash
python scripts/run_forecast_sim.py --sim_name bl_smoke \
  --provider openai --openrouter_model gpt-5-mini --scaffold basic \
  --matching exact --search_backend bm25 \
  --dataset custom --dataset_path ../../fsim_data/questions.jsonl \
  --start_date 2026-02-22 --end_date 2026-02-22 --lookback_days 1
```

(`--openrouter_model` is their model-name arg for all API providers.)
