# Handoff — read this first

You're picking up a working research codebase mid-stride. One command gets you
running:

```bash
python bootstrap.py     # deps → data (~400 MB) → index (~1.5 GB) → verification
```

Then set your **own** `OPENAI_API_KEY` (needs `gpt-5-mini` and `gpt-5.2`).
The key used for the runs so far is compromised (it leaked into logs) and is
being rotated — never commit a key; `.env` is gitignored.

## What this project is

Measuring **belief dynamics in LLM forecasters**: does a forecaster's stated
belief move with evidence, does its reasoning have a stable recoverable
structure, and where does rigidity live? Three papers anchor it:

- **Analytica** (arXiv:2604.23072) — proposition-tree forecasting, linear synthesis
- **FutureSim** (arXiv:2605.15188) — day-by-day news replay past the model cutoff
- **Agent-BRACE** (arXiv:2605.11436) — belief state as claims + ordinal certainty

Extracted texts of all three are in `papers/`. Two design documents (rendered
HTML, self-contained) are in the repo root:

- `belieflens_plan.html` — the research program: belief representation
  `B=(C,π,f)`, the distance ladder, twelve axioms whose violations are the
  measurements, the EXEC-vs-MODEL interpreter question.
- `exp1_ledger.html` — **the complete build ledger for Experiment 1**: every
  harness prompt verbatim, data provenance, an independent adversarial review
  (2 fatal, 7 serious findings) with every correction, and first results.
  Open it in a browser; it is the single most useful document here.

## What has been done (state as of 2026-08-27 evening)

**Experiment 1 is complete end-to-end.** Four harness *shapes* (react,
futuresim, analytica, bayesian) — same model (`gpt-5-mini`, pre-2026 cutoff),
same date-gated corpus (339,801 real FutureSim articles, BM25, gate enforced in
SQL), same sequential 5-date replay on a paired question control (USA/Canada
hockey gold, Milano Cortina 2026).

- **E1** (`results/exp1/e1_runs.jsonl`): 16 runs, 80/80 turns with forecasts.
- **E2** (`results/exp1/e2_programs.json`): per harness×question, a
  **literal-free program** recovered from the trajectories (variables = the
  belief, program = the structure), with in-sample fit, out-of-sample holdout
  (train sample 0 → test sample 1), degeneracy report, and full audit trail.
- An **adversarial review** of the whole setup ran mid-experiment; all fixable
  findings were patched and re-verified before E2. Read `exp1_ledger.html` §9.
  Notable: the originally planned "separation" metric was *wrong* for the
  actual event path (both teams reached the final inside the window) and was
  replaced with coherence metrics — the ledger shows the correction openly.

**Headline pilot findings** (1 question pair, 2 samples — directions, not laws):

1. **Rigidity inversion**: the most program-like harness (futuresim: fit MAE
   0.0175, OOS 0.0140 @ 80% in-tol) was the most rigid — on 02-20, with the
   USA already in the final, it still held P(USA gold) ≈ 0.17–0.22. The best
   evidence-tracker (analytica: semi-rise 1.00, P(USA)+P(CAN)=0.99) had the
   least stable structure (MAE 0.11–0.30). Structural stability and
   evidence-tracking anticorrelated.
2. **bayesian's reasoning is genuinely recoverable**: a log-odds additive
   update + shrink-to-prior program, trained on sample 0, predicts sample 1 at
   MAE 0.016 — verified NOT to use the transformed-copy variable (see ledger).
3. **react resisted compression**: no valid program in 4 repair attempts.

## Repo map

```
belieflens/
  evidence.py    corpus index + THE search tool (date gate in SQL; thread-safe)
  harnesses.py   the four harness prompts + sequential protocol + strict parser
  progdsl.py     literal-free program space: AST validator + sandboxed executor
  extract.py     E2 steps 1–3 (variables → schema → program) + degeneracy defences
  dsl.py, prompts.py, arms.py, metrics.py, llm.py   earlier belief-DSL work
                 (BeliefSpec elicitation arms + mock backend; see EXPERIMENTS.md)
scripts/
  run_exp1.py        E1/E2 orchestration        analyze_exp1.py  metrics tables
  fetch_corpus.py    HF corpus download         build_index.py   BM25 build
  run_experiment.py  offline mock pipeline      analyze.py       its analyzer
results/exp1/        the collected data + config JSONs (keep; it is the record)
results/exp1_buggy/  quarantined first run (thread-safety bug; do not use)
EXPERIMENT1.md       experiment writeup   ·   EXPERIMENTS.md  earlier protocol
e1.log, e2.log       run logs quoted in the ledger
```

## Queued next steps, in priority order

1. **Extend the near-target drop to transforms** (`extract.py`,
   `step1_variables`): drop odds/logit/percent forms of the forecast too —
   the `posterior_odds` variable reconstructed the answer to 3 decimals this
   run (program didn't use it, but close the channel).
2. **More questions, longer date grids.** n=1 pair is a pilot. The question
   loader is `QUESTIONS` in `scripts/run_exp1.py`; candidates with ≥35-day
   spans are in the OpenForesight `aljazeera2026Q1` split (bootstrap downloads
   it). Keep pairs/controls where possible.
3. **Forced-stop covariate**: now logged per turn (`forced_stop`) but absent
   from the collected E1 — next run gets it free. Consider budgeting by search
   *calls* instead of rounds (review F8: analytica gets truncated more).
4. **Fifth harness**: a coding-agent shape whose reasoning is already code —
   the natural upper bound for program-recovery fit.
5. **The ICML sequential-Bayesian PDF**: the `bayesian` harness is a
   reconstruction from the paper's abstract; align it when the PDF arrives.
6. **NeurIPS 2026 workshop date is 29 Aug AoE** (suggested date; each workshop
   sets its own). The ledger + results are most of a 4-page workshop paper:
   candidate venues in `EXPERIMENTS.md` §7.

## Things to know before trusting numbers

- `results/exp1_buggy/` is quarantined — 32% of turns lost to a since-fixed
  SQLite threading bug. Never mix it with `results/exp1/`.
- No E2 group passed the strict fit gate (MAE ≤ 0.02 ∧ ≥80% in-tol ∧ 0
  failures); all fits are descriptive. The OOS holdout is the honest gate.
- Retrieval is local BM25, weaker than FutureSim's 48 GB hybrid index —
  identical across harnesses (comparisons stand), absolute quality understated.
- The full risk ledger with statuses is `exp1_ledger.html` §8.
