# Protocol

## 0. What "belief" means here

We do not claim access to anything internal. Everything below is defined on
observable behaviour, and the paper should say so in the first paragraph.

- **Reference belief** `p_M(q)` — the distribution of probabilities the model
  states for `q` when asked directly, under repeated sampling, with no evidence
  and no structure. Arm A. We report its mean *and* spread; the spread is the
  quantity Analytica's `Var` column is about.
- **Belief representation** `R` — a BeliefSpec the model authored for `q`.
- **Decoder** `D(R) → [0,1]` — one of `linear`, `linear_simplex`, `noisy_or`,
  `wep_only`. Deterministic, ours, not the model's.

A representation is **faithful** if `D(R)` agrees with `p_M(q)` in
distribution — not just in mean. Faithfulness is *not* accuracy: it needs no
ground truth, which is why the core of this study can run in a day.

The honest framing of the whole paper is: *we measure the consistency of a
model's stated beliefs across elicitation formats and across time.* Anything
stronger is unsupported.

## 1. Hypotheses

**H1 (elicitation).** Writing the belief down as a structure changes the point
estimate: `|C − A| > 0` beyond the sampling floor.

**H2 (aggregation).** The arithmetic changes it further, and does most of the
variance reduction: `sd(B) < sd(C) ≈ sd(A)`. Analytica §4.2 predicts exactly
this — variance falls as `Σ βᵢ′² Var(lᵢ)` with `βᵢ′² → 0` as leaves multiply.
If instead `sd(C) < sd(A)` too, the stabilisation is coming from the *act of
decomposing*, not from averaging, and Analytica's derivation is describing
the wrong mechanism.

**H3 (coherence is optional).** Analytica's Appendix B proves the linear rule
equals a Bayes net only when `βⱼ ∈ [0,1]` and `β₀ + Σβⱼ ≤ 1`, and states the
constraint "was not strictly enforced in our existing experiments." So we
measure the violation rate directly, and the root-level gap between `linear`
and `linear_simplex`. If violations are common and enforcing coherence costs
accuracy, then the gain came from an aggregation that is *not* a probability
model — which undercuts reading the tree as a belief at all.

**H4 (rigidity).** Fed back the model's own stated evidence cues, the root
moves by less than the frozen-coefficient identity implies, and under the free
protocol part of the shortfall shows up as coefficient drift rather than as an
honest change of mind. FutureSim §4.3 already reports "self-conditioning" —
agents treating their own earlier rationales as hard truths — and §5.1 shows
agents that cannot escape a bad initial anchor even when told their Brier score
is negative. H4 says that failure is *localisable* inside the representation.

**H5 (asymmetry).** Confirming evidence moves the belief more than
disconfirming evidence that the model itself rated equally strong.

## 2. Design

```
                        no evidence                  evidence injected
                 ┌──────────────────────┐   ┌────────────────────────────────┐
   A direct      │ p ~ p_M(q)           │   │ direct update (control)        │
   B composed    │ D(R), 4 decoders     │   │ local update  (β frozen)       │
   C conditioned │ free-form given R    │   │ free update   (β may move)     │
                 └──────────────────────┘   └────────────────────────────────┘
```

Arms B and C share the *same* elicited `R`, so the comparison is paired: the
only difference is what happens to the structure afterwards. That pairing is
what makes the elicitation/aggregation split clean.

Injected evidence comes from the model's own `sensitivity` slots, written
before it saw anything. This is the load-bearing design choice. It means we
never have to argue about whether the evidence "should" have moved the belief:
the model already told us it should, and by how much.

## 3. Controls, and the confound each one kills

| control | kills |
|---|---|
| **placebo** — unrelated dated news of matched length | "the belief moves because *something* arrived". Every effect must clear this floor. |
| **local protocol** — coefficients frozen by instruction | separates a genuine change of mind from a rewrite of the aggregation. Also a hard correctness check: rigidity must be exactly 0 here by arithmetic, so a non-zero value means the model violated the protocol. |
| **direct update** — same evidence, no structure | tells you whether rigidity is a property of the *representation* or of the model. If the unstructured control is equally rigid, the DSL is innocent. |
| **`k` samples per cell** | β and p are both LLM-generated, so β-drift has a noise floor. Any rigidity claim must exceed the spread of β across independent elicitations of the *same* question. Run this first; if the floor is large the H4 result is not available at this sample size. |
| **`stated` decoder** | the model is told not to state an overall answer, but it fills the root's `p` anyway. `composed:stated` vs `composed:linear` reveals whether it is quietly answering top-down and back-filling children to match — which would make the whole tree post-hoc rationalisation. **This is the cheapest way the central claim can die, so run it first.** |

## 4. Reading the outcomes

| result | reading |
|---|---|
| `\|B − A\|` small, `sd(B) < sd(A)` | the scaffold is a faithful *stabiliser*: same belief, less noise. The strongest possible support for Analytica's story. |
| `\|B − A\|` large, direction systematic | the scaffold **constructs** a belief. Then Analytica's accuracy gain is not "grounding a prior" — it is a different prior, and the paper should say which. |
| `composed:stated ≈ composed:linear`, children incoherent | top-down rationalisation. The tree is an explanation, not a computation. |
| `wep_only ≈ linear` | the numbers are decoration on the ordinal labels; Agent-BRACE's 7-point scale is doing the work and the reals are noise. |
| `rigidity > 0` under local, `≈ 0` for the direct control | rigidity is a cost of the representation — a genuinely new and slightly uncomfortable result for structured-reasoning scaffolds. |
| `rigidity ≈ 0` everywhere | the representation is inert with respect to updating. Fine, and worth reporting: it means Analytica-style trees are safe to update in place, which is a useful engineering claim. |

Every one of these is publishable at a workshop. There is no outcome of this
design that yields nothing, which is the main reason to run it under time
pressure.

## 5. Threats to validity, stated up front

1. **Contamination.** FutureSim's questions resolve Jan–Mar 2026 and are past
   the cutoff of the models it tested — not necessarily past the cutoff of
   whatever you run. For *faithfulness* this is harmless (a contaminated prior
   is still the model's belief). For any accuracy claim it is fatal. Report
   each model's cutoff in the table and do not mix the two claims.
2. **Sampling noise on Anthropic models.** `temperature` is rejected (400) on
   the current reasoning models, so the `k` draws are ordinary sampling
   variation, not temperature-controlled. Say so. `sample_nonce` exists as an
   escape hatch but perturbs the prompt, so it is a confound — leave it off by
   default and disclose it if used.
3. **Prompt sensitivity.** "The" belief is really a family of conditionals
   indexed by prompt. Two paraphrases of the arm-A prompt, reported as a
   robustness row, cost almost nothing and pre-empt the obvious reviewer
   objection.
4. **Instruction-following, not rigidity.** A model that ignores the local
   protocol may simply be bad at constrained editing. The `direct` control and
   the format-compliance rate together separate these; report the compliance
   rate.
5. **The mock is not evidence.** It exists to validate instrumentation. Never
   put a mock number in the paper.

## 6. Three-day plan

The NeurIPS 2026 workshop submission date is **29 Aug 2026 AoE**.

**Day 1.** `--stage arms`, 3 models × 60 FutureSim questions × k=5. Produces
Tables 1–4. Run the `stated`-decoder check first — it is the cheapest way to
find out the premise is wrong. ~2.7k calls, cached.

**Day 2.** `--stage perturb`, same specs, 3 protocols × 3 conditions. Table 5,
the asymmetry row, and the β-drift noise floor. This is the headline figure:
a waterfall of `Δroot = Σ βᵢ′ Δlᵢ + Δβ` terms, which is exactly computable
under the linear rule and is the thing no unstructured forecaster can show.

**Day 3.** Write. Four pages: the elicitation/aggregation decomposition, the
coherence-violation table, the rigidity result, and the asymmetry.

**Explicitly out of scope for this deadline:** the full 88-day FutureSim
replay with search and memory. It needs the 7.36M-article corpus, a LanceDB
index, and long-horizon runs. Do it for the camera-ready. Day-0 elicitation
plus injected evidence gets the belief-dynamics claim without the
infrastructure, and the replay then upgrades it from synthetic cues to real
arriving news.

## 7. Venues

No NeurIPS 2026 workshop is dedicated to forecasting. Best fits, in order:

1. **Foundation Models for Temporal Systems: From Forecasting to World Modeling** (Sydney) — the forecasting home.
2. **Interpreting Agent Behavior** (Sydney) — arguably the better fit: the paper is about whether an externalised representation faithfully reflects the policy that produced it.
3. **Test-Time Continual Learning Agents** (Atlanta) / **Foundations of LLM Post-Training in Changing Environments** (Paris) — the updating half.
4. **I Can't Believe It's Not Better** (Sydney) — if the headline lands as a negative result.

Submitting the same work to two of these is normal for workshops and doubles
the acceptance odds; check each workshop's own CFP, since the 29 Aug date is
NeurIPS's *suggested* date and individual workshops set their own.
