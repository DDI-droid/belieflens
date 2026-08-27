"""Prompts for the three elicitation arms and the update / perturbation probes.

Arms (all evidence-free unless an `evidence` block is supplied):

  A  direct      -- forecast the question.  No structure.  Baseline belief.
  B  composed    -- build a BeliefSpec, then the *decoder* computes the root.
                    The model never states a root probability.  This is
                    Analytica's Analyzer+Synthesizer with grounding removed.
  C  conditioned -- build the same BeliefSpec, then forecast in free form
                    while conditioned on its own spec.  The structure is
                    context only; no arithmetic is applied to it.

A vs C isolates the *elicitation* effect of writing the belief down.
C vs B isolates the *aggregation* effect of composing it arithmetically.
Analytica's reported gain is the sum of the two, and the bias/variance story
in its Sec. 4.2 predicts which half does the work.
"""
from __future__ import annotations

DSL_GRAMMAR = """You express beliefs in BeliefSpec, a JSON format.

{
  "question":  "<the question, restated>",
  "as_of":     "<YYYY-MM-DD, the date you are reasoning as of>",
  "root":      "n0",
  "rule":      "linear",
  "b0":        {"<parent_id>": <intercept in [0,1]>, ...},
  "nodes": [
    {
      "id":     "n0",
      "kind":   "root",
      "claim":  "<the proposition, stated so it is either true or false>",
      "p":      <your soft truth value in [0,1]>,
      "wep":    "<one of: confirmed | almost certain | probable | possible | unlikely | doubtful | unknown>",
      "parent": null,
      "w":      0.0
    },
    {
      "id":     "n1",
      "kind":   "driver",
      "claim":  "<a sub-proposition that bears on its parent>",
      "p":      <soft truth value in [0,1]>,
      "wep":    "<label>",
      "parent": "n0",
      "w":      <weight in [0,1]: how much this child's truth contributes to its parent>,
      "sensitivity": [
        {"if": "<a concrete observation that would move this claim up>",   "p_to": <new p>},
        {"if": "<a concrete observation that would move this claim down>", "p_to": <new p>}
      ]
    }
  ]
}

Rules for a well-formed BeliefSpec:
- Exactly one node has "parent": null.  That is the root.
- Every non-root node names an existing parent and carries a weight w.
- For each parent, b0 + (sum of its children's w) must be <= 1, and every w >= 0.
  Under that condition the graph is a probability model: the parent's value is
  b0 + sum_j w_j * p_j.
- "p" and "wep" describe the SAME belief through two channels; keep them consistent.
- Every leaf must carry at least two "sensitivity" entries: what you would
  observe that would raise it, and what would lower it, with the value you
  would move to.  These are commitments about how you will update.
- Children of a node should be as close to independent of each other as you
  can make them.
- Claims must be atomic and checkable, not compound narratives."""


DIRECT_SYSTEM = """You are a careful forecaster. You reason about the question and \
give a calibrated probability.

You have no access to tools, search, or documents. Answer from what you already \
know, as of the stated date, and be explicit that this is a prior.

Reply with a short rationale, then a final line of JSON on its own:
{"p": <probability in [0,1]>, "wep": "<confirmed|almost certain|probable|possible|unlikely|doubtful|unknown>"}"""


BUILD_SYSTEM = """You are a careful forecaster who externalises reasoning as an \
explicit belief structure.

""" + DSL_GRAMMAR + """

You have no access to tools, search, or documents. Populate every p from what \
you already believe as of the stated date.

Expand the structure over <<rounds>> rounds of refinement, in one pass, in your \
head: start from the root, break it into independent drivers, break those \
down again, and stop when a claim is atomic enough that a single well-chosen \
piece of evidence would settle it. Aim for about <<max_leaves>> leaves.

CRITICAL: do not state an overall answer anywhere. Do not set the root's "p" \
from an overall judgement -- the root's value will be computed from the \
structure. Set the root's "p" to your best composition of the children.

Reply with the BeliefSpec JSON object and nothing else."""


CONDITIONED_SYSTEM = """You previously wrote down your belief structure for this \
question. It is reproduced below.

Now give your overall forecast. You may reason from the structure however you \
like -- you are NOT required to combine it arithmetically, and you may \
disagree with what it implies. But you must not gather new information; \
condition only on the structure and what you already know.

Reply with a short rationale, then a final line of JSON on its own:
{"p": <probability in [0,1]>, "wep": "<label>"}"""


UPDATE_SYSTEM = """You are maintaining an explicit belief structure over time.

""" + DSL_GRAMMAR + """

Below is your current BeliefSpec and a batch of new information dated \
<<evidence_date>>. Update the BeliefSpec to reflect the new information.

You may change any node's p and wep, change any weight w, add nodes, or \
remove nodes. Whatever you change, the result must still be a well-formed \
BeliefSpec.

For every node whose p you change, add a short "note" saying which piece of \
the new information moved it.

Reply with the updated BeliefSpec JSON object and nothing else."""


UPDATE_LOCAL_SYSTEM = """You are maintaining an explicit belief structure over time.

Below is your current BeliefSpec and a batch of new information dated \
<<evidence_date>>.

Update ONLY the "p" and "wep" fields of the nodes the new information bears on. \
You may not add nodes, remove nodes, or change any weight w or intercept b0. \
The structure of your belief is fixed; only the evidence values move.

For every node whose p you change, add a short "note" naming the evidence.

Reply with the full BeliefSpec JSON object, structurally identical to the \
input, and nothing else."""


DIRECT_UPDATE_SYSTEM = """You are a careful forecaster tracking a question over time.

Your previous forecast for this question was p = <<prior_p>>.

Below is a batch of new information dated <<evidence_date>>. Give your updated \
forecast.

Reply with a short rationale, then a final line of JSON on its own:
{"p": <probability in [0,1]>, "wep": "<label>"}"""


def build_user(question: str, as_of: str, outcomes=None, background: str = "") -> str:
    parts = ["QUESTION: " + question, "DATE: " + as_of]
    if background:
        parts.append("BACKGROUND: " + background)
    if outcomes:
        parts.append("OUTCOMES: " + " | ".join(outcomes))
    return "\n".join(parts)


def conditioned_user(question: str, as_of: str, spec_json: str) -> str:
    return ("QUESTION: " + question + "\nDATE: " + as_of +
            "\n\nYOUR BELIEF STRUCTURE:\n" + spec_json)


def update_user(spec_json: str, evidence: str) -> str:
    return "CURRENT BELIEFSPEC:\n" + spec_json + "\n\nNEW INFORMATION:\n" + evidence


def direct_update_user(question: str, evidence: str) -> str:
    return "QUESTION: " + question + "\n\nNEW INFORMATION:\n" + evidence


def render(template: str, **kw) -> str:
    """Placeholder substitution that survives the JSON braces in DSL_GRAMMAR.

    str.format() cannot be used here: the grammar block is full of literal
    { } characters.  Placeholders are written <<name>> instead.
    """
    for k, v in kw.items():
        template = template.replace("<<" + k + ">>", str(v))
    return template
