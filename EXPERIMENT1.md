# Experiment 1 — four harnesses, sequential replay, program recovery

## What it does

**E1.** Four harnesses forecast the same question on K dates in chronological
order. Each day a harness sees its own previous forecasts, and — where its shape
has one — its own memory or belief state carried forward. All four read the
same date-gated corpus through the same search tool. Nothing later than the
current simulation date is retrievable, enforced in SQL.

**E2.** For every forecast produced, a strong model reads that day's trajectory
and writes down the quantities the reasoning used, as `name = value`. Those
tables are reconciled into one canonical schema. Then a single **literal-free
program** is synthesised against all of that harness's forecasts; the pass gate
is MAE <= 0.02 AND >=80% of instances within tolerance AND zero failed
executions, with an out-of-sample holdout (fit on sample 0, test on sample 1)
as the honest check against memorisation.

The point of the literal ban: there is nowhere for a number to hide, so fitting
one program across all instances forces the split

```
program  = the reasoning structure   (must hold still across dates)
bindings = the belief state           (free to move with evidence)
```

A low fit error means the structure held and everything that moved, moved
through the variables — which is what it takes for the variables to *be* the
belief. A high fit error means the structure itself changed as evidence arrived,
which is a different and equally reportable finding.

## The four harnesses

One model, one search tool, one corpus. Only the reasoning shape varies —
installing four upstream repos would compare four repos instead.

| | shape |
|---|---|
| `react` | linear think / search / think loop. No structure beyond the shared forecast history all four carry — the history-anchored floor. |
| `futuresim` | ReAct plus the engineering the FutureSim paper describes: procedural forecasting guidelines, base-rate-first ordering, an explicit disconfirming-evidence step, and a forced memory-update phase whose block is the only thing carried to the next day. |
| `analytica` | three labelled stages — decompose into 3–5 independent sub-propositions, ground each by search, then synthesise with weights written out. |
| `bayesian` | an explicit linguistic belief state: prior, new evidence, likelihood ratio per item, posterior, then an updated belief block. |

FutureSim's own harness is essentially *ReAct plus engineering*, so `react` vs
`futuresim` is an engineering-quality axis while `analytica` and `bayesian` are
genuine structure axes. That is deliberate — it separates the two.

## The question pair

Both from OpenForesight `aljazeera2026Q1`, binarised, sharing one evidence
stream and resolving in **opposite directions**:

- `hockey_usa` — will the USA win men's ice hockey gold at Milano Cortina? **true**
- `hockey_can` — will Canada? **false**

Dates: 2026-01-14, 01-28, 02-08, 02-15, 02-20. The tournament runs mid-February,
so evidence concentrates sharply across the last three.

**Correction (adversarial review, 2026-08-27):** the original plan scored
"separation" — net(USA) up, net(CAN) down. That is wrong for how the event
actually unfolded: both teams won their semifinals on 02-20 (the last simulated
date) and the final resolved 02-22, after the window, so a correct forecaster
raises BOTH beliefs toward ~0.5. The corrected metrics are **semi-rise** (both
beliefs rise across the semifinal date), **sum@last** (P(USA)+P(CAN) approaches
1 by 02-20, never exceeding it), and complement-coherence error. The pair is a
control for coherence under shared evidence, not for direction.

## Data

Real FutureSim corpus, real date gating — 339,801 articles, 2025-12-01 to
2026-03-31, 412 MB of parquet from `shash42/forecast-news`.

The one deviation from the paper: we do **not** download their 48 GB prebuilt
embedding index, and build a local SQLite FTS5 (BM25) index instead. Retrieval
is weaker than their hybrid search. It is also identical for all four harnesses,
so it is a constant of the experiment rather than a confound in it — the
comparison is between reasoning structures, not retrieval systems.

## Model choice

Harnesses run on **`gpt-5-mini`** (released 2025-08-07). The questions resolve
January–March 2026, so any model from `gpt-5.4` onward would be *recalling*
rather than forecasting — and a model that already knows the answer will not
move its belief, which would destroy the dynamics half of the experiment.
The extractor is a later model, where contamination is irrelevant.

## Guarding against a meaningless fit

The literal ban stops the program hiding numbers in its body. It does not stop
step 1 declaring `final_answer = 0.62` and step 2 writing `return final_answer`
— a perfect fit that measures nothing. Four defences, all mechanical:

1. Step 1 is instructed to record only *inputs*, never the output.
2. Any extracted variable equal to that turn's forecast is dropped in code.
3. A program whose body is a bare `return <name>` is rejected outright.
4. `degeneracy()` reports the strongest single-variable correlation with the
   target and the program's operation count; near-passthrough programs are
   rejected and re-requested.

A perfect fit with high degeneracy is a failed extraction, not a result.

## Running it

```bash
export OPENAI_API_KEY=...
python scripts/fetch_corpus.py            # ~400 MB, once
python scripts/build_index.py             # ~2 min, once

python scripts/run_exp1.py --stage e1 --model gpt-5-mini --effort medium --samples 2
python scripts/run_exp1.py --stage e2 --extractor gpt-5.2
python scripts/analyze_exp1.py results/exp1
```

## What comes out

`results/exp1/e1_runs.jsonl` — every turn, its searches, its reasoning, its forecast.
`results/exp1/e2_programs.json` — per harness×question: the schema, the program,
fit residuals, degeneracy, and the variable trajectory across dates.

The measurements:

| | reads |
|---|---|
| separation | does the harness track evidence at all |
| total abs movement | is the belief plastic or stuck |
| fit MAE | is the reasoning program-like |
| schema size | how many quantities the shape needs |
| variable range across dates | which parts of the belief moved, and which held fixed |
| ops / max abs corr | is the fit real or degenerate |
