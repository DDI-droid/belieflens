# E1 v2 data package

Everything Experiment 1 (v2) produced, in the form downstream work needs it.
Nothing here has to be re-run: the traces preserve each system's internal state
at every date, not just its output probability.

Run: 2026-09-11, 2h22m, 19,385 model calls, `gpt-5-mini`, 12 workers.

---

## Files

| file | rows | contents |
|---|---|---|
| `e1_runs.jsonl` | 120 | sequential chains: 4 harnesses x 6 questions x 5 rollouts |
| `e1_independent.jsonl` | 720 | parallel singles: 4 x 6 x 6 dates x 5 rollouts |
| `metrics_v2.json` | — | every metric in the report, per harness |
| `usage.json` | — | call and token ledger |
| `smoke_*.json` | 4 | one-date smoke per harness, kept as a shape reference |

Regenerate metrics with `python scripts/analyze_v2.py`, figures with
`python report_v2/figures_v2.py`.

## Record shapes

**`e1_runs.jsonl`** — one JSON object per sequential chain:

```
{"mode": "seq", "harness": ..., "question_id": ..., "sample": 0..4,
 "turns": [{"date": "YYYY-MM-DD", "forecast": float|null,
            "trace": {...}, "secs": float, "error": ""}, ...]}
```

Six turns per chain, in date order: `turns[0..4]` are the trajectory
(D1..D5) and **`turns[5]` is the outcome probe (D6 = resolution + 2 days)**.
Anything computing belief dynamics must drop `turns[5]`; by then the corpus
contains the answer.

**`e1_independent.jsonl`** — one object per parallel single:

```
{"mode": "par", "harness": ..., "question_id": ..., "date": ...,
 "rollout": 0..4, "forecast": float|null, "trace": {...},
 "secs": float, "error": ""}
```

## What is in `trace`, per harness

These differ by design: each system's trace holds the state that system
actually maintains.

**`analytica_full`**
- `tree`: `{node_id: {id, statement, parent, p, report, b0, betas, last_grounded}}`
  — the full proposition tree at that date. `statement` is the sub-proposition,
  `report` the grounder's written justification, `p` its soft truth value,
  `b0`/`betas` the synthesizer's intercept and weights for that node's children,
  `last_grounded` the date the leaf last searched.
- `edits`: `{added: [...], removed: [...]}` from that date's analyzer edit pass.
- `searches`: `[{leaf, query, from, n, titles}]`.

The forecast is `clamp(b0 + sum(beta_j * p_j))` computed in code, so the
composition rule is **known ground truth** — useful for validating any method
that claims to recover a system's reasoning.

**`blf_full`**
- `trial_ps`: the K=5 independent trial probabilities for that date.
- `aggregate`: the shrunken logit-mean of those, which is the forecast.
- `alpha_note`: the aggregator's alpha definition.
- `belief`: `{p, confidence, evidence_for, evidence_against, open_questions}`
  — the carried belief state (the median trial's).
- `searches`: `[{trial, step, query, n, titles}]`, attributed per trial.

**`futuresim_full`**
- `actions`: `[{step, tool, code?, result?}]` — the action sequence, including
  the Python the agent wrote for `query_df` and what came back.
- `memory`: the memory store after the session.
- `memory_ops`: what the end-of-session memory phase did.
- `submitted_today`, `declined_to_update`, `ended_by_agent`: control-flow flags.
- `searches`: `[{query, from, n, titles}]`.

**`react`**
- `text`: the reasoning prose (**capped at 1500 characters** — see caveats).
- `searches`: `[{query, from_date, n, titles}]`.
- `forced_stop`: whether the answer was forced by exhausting the search budget.

## Coverage and gaps

- 713 of 720 sequential turns carry a trace; 718 of 720 parallel singles.
- 9 forecasts of 1,440 (0.6%) are `null` with `error` set — rate-limit 429s
  that outlived five retries. Scattered; no cell lost more than one rollout.
- 1,780 retries were absorbed successfully and are not visible as errors.

## Caveats for downstream use

1. **`react` reasoning text is truncated at 1500 characters.** Anything that
   needs its full chain of thought will need react re-run with a higher cap
   (roughly $5 at this scale). The other three harnesses store structured
   state rather than prose, so they are unaffected.
2. **`react` token counts are missing** from `usage.json`: it calls through
   `HarnessRunner`'s own client, so its calls are counted but its tokens are
   not. The true token total is roughly 10-15% above the recorded figure.
3. **D6 is not a forecast in the same sense as D1-D5.** The answer is public
   by then. Treat it as a separate diagnostic.
4. **`futuresim_full` was forced to submit every date.** Its source system may
   end a session without submitting; `declined_to_update` is therefore always
   false here and carries no information in this run.

## Questions

Defined in `belieflens/questions_v2.py`, which is the authority. Three matched
groups, each binarising one free-form question from the FutureSim benchmark
split into its two finalists; three resolve YES and three NO.

| id | truth | resolves |
|---|---|---|
| `hockey_usa` | YES | 2026-02-22 |
| `hockey_can` | NO | 2026-02-22 |
| `ausopen_alcaraz` | YES | 2026-01-31 |
| `ausopen_djokovic` | NO | 2026-01-31 |
| `superbowl_sea` | YES | 2026-02-07 |
| `superbowl_ne` | NO | 2026-02-07 |

Dates per question are offsets from resolution R: D1..D5 at R-39, -25, -14,
-7, -2 and D6 at R+2.
