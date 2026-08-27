"""The elicitation arms and the evidence-perturbation probe."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import prompts
from .dsl import BeliefSpec, DSLError, extract_probability, clip01
from .llm import LLM


@dataclass
class Draw:
    """One sample from one arm on one question."""
    arm: str
    question_id: str
    sample_id: int
    p: float | None = None
    spec: BeliefSpec | None = None
    raw: str = ""
    error: str = ""

    def ok(self) -> bool:
        return self.error == "" and self.p is not None


@dataclass
class Question:
    id: str
    text: str
    as_of: str
    background: str = ""
    outcomes: list = field(default_factory=list)
    resolution: str | None = None      # ground truth, if known
    resolution_date: str | None = None

    @staticmethod
    def from_dict(d: dict) -> "Question":
        return Question(
            id=str(d.get("id") or d.get("question_id") or abs(hash(d.get("question", ""))) % 10**8),
            text=str(d.get("question") or d.get("text") or ""),
            as_of=str(d.get("as_of") or d.get("open_date") or d.get("date") or ""),
            background=str(d.get("background", "")),
            outcomes=list(d.get("outcomes", []) or []),
            resolution=d.get("resolution") or d.get("answer"),
            resolution_date=d.get("resolution_date"),
        )


# ---------------------------------------------------------------- arms

def arm_direct(llm: LLM, q: Question, k: int = 1) -> list[Draw]:
    """A: forecast with no structure and no evidence."""
    user = prompts.build_user(q.text, q.as_of, q.outcomes, q.background)
    out = []
    for i in range(k):
        r = llm.chat(prompts.DIRECT_SYSTEM, user, sample_id=i)
        d = Draw(arm="direct", question_id=q.id, sample_id=i, raw=r.text)
        try:
            d.p = extract_probability(r.text)
        except DSLError as e:
            d.error = str(e)
        out.append(d)
    return out


def arm_build(llm: LLM, q: Question, k: int = 1, rounds: int = 3,
              max_leaves: int = 8) -> list[Draw]:
    """Shared front half of arms B and C: elicit the BeliefSpec."""
    system = prompts.render(prompts.BUILD_SYSTEM, rounds=rounds, max_leaves=max_leaves)
    user = prompts.build_user(q.text, q.as_of, q.outcomes, q.background)
    out = []
    for i in range(k):
        r = llm.chat(system, user, sample_id=i)
        d = Draw(arm="spec", question_id=q.id, sample_id=i, raw=r.text)
        try:
            d.spec = BeliefSpec.from_text(r.text)
            d.p = d.spec.decode("linear")
        except (DSLError, KeyError, TypeError, ValueError) as e:
            d.error = "%s: %s" % (type(e).__name__, e)
        out.append(d)
    return out


def arm_composed(specs: list[Draw], rule: str = "linear") -> list[Draw]:
    """B: the decoder computes the root from the structure."""
    out = []
    for s in specs:
        d = Draw(arm="composed:" + rule, question_id=s.question_id,
                 sample_id=s.sample_id, spec=s.spec, raw=s.raw, error=s.error)
        if s.spec is not None:
            try:
                d.p = s.spec.decode(rule)
            except DSLError as e:
                d.error = str(e)
        out.append(d)
    return out


def arm_conditioned(llm: LLM, q: Question, specs: list[Draw]) -> list[Draw]:
    """C: free-form forecast conditioned on the model's own structure."""
    out = []
    for s in specs:
        d = Draw(arm="conditioned", question_id=q.id, sample_id=s.sample_id, spec=s.spec)
        if s.spec is None:
            d.error = s.error or "no spec"
            out.append(d)
            continue
        user = prompts.conditioned_user(q.text, q.as_of, s.spec.to_json())
        r = llm.chat(prompts.CONDITIONED_SYSTEM, user, sample_id=s.sample_id)
        d.raw = r.text
        try:
            d.p = extract_probability(r.text)
        except DSLError as e:
            d.error = str(e)
        out.append(d)
    return out


# ---------------------------------------------------------- perturbation

@dataclass
class Perturbation:
    """One injected evidence item aimed at one node."""
    target: str          # node id
    condition: str       # support | contradict | placebo
    text: str
    stated_p_to: float | None   # what the model said it would move that node to
    date: str = ""


def make_perturbations(spec: BeliefSpec, date: str,
                       placebo_text: str | None = None) -> list[Perturbation]:
    """Turn each leaf's own `sensitivity` slots into injected news items.

    Using the model's *own* stated cues is what makes the probe fair: the
    evidence is, by the model's own account, exactly the thing that should
    move that node, and by exactly that much.  Any shortfall in the update is
    the model contradicting itself, not us mis-specifying the evidence.
    """
    out: list[Perturbation] = []
    for leaf in spec.leaves():
        if leaf.id == spec.root:
            continue
        ups = [c for c in leaf.sensitivity if c.p_to > leaf.p]
        downs = [c for c in leaf.sensitivity if c.p_to < leaf.p]
        if ups:
            out.append(Perturbation(leaf.id, "support",
                                    _as_news(ups[0].cue, date), ups[0].p_to, date))
        if downs:
            out.append(Perturbation(leaf.id, "contradict",
                                    _as_news(downs[0].cue, date), downs[0].p_to, date))
    if placebo_text:
        out.append(Perturbation(spec.root, "placebo", _as_news(placebo_text, date), None, date))
    return out


def _as_news(cue: str, date: str) -> str:
    cue = cue.strip().rstrip(".")
    return "[%s] Reuters reports: %s." % (date, cue[0].upper() + cue[1:] if cue else cue)


@dataclass
class UpdateResult:
    question_id: str
    sample_id: int
    mode: str                  # local | free | direct
    condition: str
    target: str
    prior: BeliefSpec | None
    post: BeliefSpec | None
    prior_root: float | None
    post_root: float | None
    stated_p_to: float | None
    raw: str = ""
    error: str = ""


def apply_perturbation(llm: LLM, q: Question, spec: BeliefSpec, pert: Perturbation,
                       sample_id: int, mode: str = "local",
                       rule: str = "linear") -> UpdateResult:
    if mode == "local":
        system = prompts.render(prompts.UPDATE_LOCAL_SYSTEM, evidence_date=pert.date or q.as_of)
    elif mode == "free":
        system = prompts.render(prompts.UPDATE_SYSTEM, evidence_date=pert.date or q.as_of)
    else:
        raise ValueError("mode must be local|free")
    user = prompts.update_user(spec.to_json(), pert.text)
    r = llm.chat(system, user, sample_id=sample_id)
    res = UpdateResult(q.id, sample_id, mode, pert.condition, pert.target,
                       spec, None, spec.decode(rule), None, pert.stated_p_to, r.text)
    try:
        res.post = BeliefSpec.from_text(r.text)
        res.post_root = res.post.decode(rule)
    except (DSLError, KeyError, TypeError, ValueError) as e:
        res.error = "%s: %s" % (type(e).__name__, e)
    return res


def apply_perturbation_direct(llm: LLM, q: Question, prior_p: float,
                              pert: Perturbation, sample_id: int) -> UpdateResult:
    """Unstructured control: same evidence, no belief representation."""
    system = prompts.render(prompts.DIRECT_UPDATE_SYSTEM,
                            prior_p=round(prior_p, 3), evidence_date=pert.date or q.as_of)
    user = prompts.direct_update_user(q.text, pert.text)
    r = llm.chat(system, user, sample_id=sample_id)
    res = UpdateResult(q.id, sample_id, "direct", pert.condition, pert.target,
                       None, None, prior_p, None, pert.stated_p_to, r.text)
    try:
        res.post_root = extract_probability(r.text)
    except DSLError as e:
        res.error = str(e)
    return res


def serialise(obj) -> dict:
    """JSONL-friendly rendering of a Draw or an UpdateResult."""
    if isinstance(obj, Draw):
        return {"kind": "draw", "arm": obj.arm, "question_id": obj.question_id,
                "sample_id": obj.sample_id, "p": obj.p, "error": obj.error,
                "spec": obj.spec.to_dict() if obj.spec else None}
    if isinstance(obj, UpdateResult):
        return {"kind": "update", "question_id": obj.question_id,
                "sample_id": obj.sample_id, "mode": obj.mode,
                "condition": obj.condition, "target": obj.target,
                "prior_root": obj.prior_root, "post_root": obj.post_root,
                "stated_p_to": obj.stated_p_to, "error": obj.error,
                "prior": obj.prior.to_dict() if obj.prior else None,
                "post": obj.post.to_dict() if obj.post else None}
    raise TypeError(type(obj))


def write_jsonl(path, rows) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(serialise(r)) + "\n")
