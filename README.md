# BeliefLens

Does a structured forecasting scaffold *read out* a language model's belief, or
*manufacture* a new one — and once it exists, does the model stay true to it
when the evidence turns against it?

Built on three papers:

| | |
|---|---|
| **Analytica** — [arXiv:2604.23072](https://arxiv.org/abs/2604.23072) | Decomposes a query into a tree of soft propositions, grounds the leaves with tool-equipped agents, and composes back up with a linear rule. +15.8% accuracy, lowest variance. Code: [chengjunyan1/analytica](https://github.com/chengjunyan1/analytica) (Apache-2.0) |
| **FutureSim** — [arXiv:2605.15188](https://arxiv.org/abs/2605.15188) | Replays dated news day-by-day past the knowledge cutoff; agents must update forecasts as evidence arrives. Documents severe anchoring. Code: [OpenForecaster/futuresim](https://github.com/OpenForecaster/futuresim) |
| **Agent-BRACE** — [arXiv:2605.11436](https://arxiv.org/abs/2605.11436) | Belief = a set of atomic NL claims, each with an ordinal certainty label from the Words-of-Estimative-Probability scale. Code: [joykirat18/Agent-BRACE](https://github.com/joykirat18/Agent-BRACE) |

## Install

```bash
pip install -r requirements.txt      # only `anthropic`/`openai` are needed for real runs
```

## Run

Everything works offline first. The `mock` provider is a deterministic
pseudo-model that emits well-formed BeliefSpecs and a *deliberately* 70%
follow-through on its own stated updates, so every metric can be validated
against a known answer before spending anything:

```bash
python scripts/run_experiment.py --stage all --provider mock --k 3
python scripts/analyze.py results/mock_mock-1
```

Expected on the mock: `local` mode returns `rigidity = 0.000` and
`beta_comp = 0.000` (the frozen-coefficient identity holds exactly), and
`follow_through = 0.700` in every cell (the metric recovers the planted
shortfall). If those three numbers move, the instrumentation is broken.

Then point it at a real model:

```bash
export ANTHROPIC_API_KEY=...
python scripts/run_experiment.py --stage all --provider anthropic --model claude-opus-5 \
       --questions data/futuresim.jsonl --limit 60 --k 5
python scripts/analyze.py results/anthropic_claude-opus-5
```

Other backends: `--provider openai --model gpt-...`, or
`--provider openrouter --model qwen/...` for open-weight families.
All calls are cached under `cache/` keyed by prompt, so re-analysis is free
and reruns are exact.

Real questions:

```bash
python scripts/fetch_futuresim_questions.py --out data/futuresim.jsonl
```

## What it measures

**Three arms**, all evidence-free, so no search, no tools, no ground truth needed:

- **A `direct`** — forecast the question. No structure. The reference belief.
- **B `composed:<rule>`** — build a BeliefSpec, then a *decoder* computes the root.
  The model never states an overall answer. Analytica minus grounding.
- **C `conditioned`** — build the same BeliefSpec, then forecast in free form
  conditioned on it. The structure is context; no arithmetic is applied.

`C − A` is the **elicitation effect** (what writing the belief down does).
`B − C` is the **aggregation effect** (what the arithmetic does).
Together they decompose Analytica's reported gain into its two halves.

**Four decoders** run over the same structures — `linear` (Analytica as
published), `linear_simplex` (Analytica Appendix B's coherence constraint,
which the paper derives but does not enforce), `noisy_or`, and `wep_only`
(ordinal labels only, numbers discarded). The gap between `linear` and
`linear_simplex` is the price of being a probability model.

**Perturbation.** Each leaf's BeliefSpec carries `sensitivity` slots — the
model states, *before* seeing anything, what observation would move that claim
and to what value. We hand those cues back as dated news items and measure:

| metric | question |
|---|---|
| `follow_through` | did it move the node to where it promised? (1.0 = kept its word) |
| `rigidity` | `1 − actual/implied` root movement at frozen coefficients. 0 = faithful, >0 = the conclusion resisted, <0 = over-reaction |
| `beta_comp` | coefficient drift. Under the local protocol it must be 0; anything else is the model rewriting the weights to protect the conclusion |
| `locality` | share of total belief movement that landed on the targeted node |
| asymmetry | does confirming evidence move the belief more than disconfirming evidence of the model's own stated equal strength? |
| placebo | an unrelated news item. The noise floor every other number must clear |

Three update protocols: `local` (only `p`/`wep` may change), `free` (anything
may change), `direct` (no structure at all — the unstructured control).

## Layout

```
belieflens/dsl.py       BeliefSpec: parse, validate, decode, coherence checks
belieflens/prompts.py   grammar + the three arms + update protocols
belieflens/llm.py       anthropic / openai / openrouter / mock, with caching
belieflens/arms.py      arm runners and the perturbation probe
belieflens/metrics.py   faithfulness, coherence, locality, plasticity
scripts/run_experiment.py
scripts/analyze.py
scripts/fetch_futuresim_questions.py
```

See `EXPERIMENTS.md` for the protocol, the confounds each control kills, and
what each possible outcome would mean.
