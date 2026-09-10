"""Real-orchestration harnesses, trimmed to live inside the date-gated sandbox.

These are NOT prompt shapes. The orchestration that defines each source system
runs in code here; only the evidence layer is swapped for the sandbox's
date-gated corpus search (which is what keeps the experiment valid -- the
systems' own live search would read the future).

ANALYTICA (arXiv:2604.23072, Algorithms 1-2, Sec. 4.1)
  - Analyze: an Analyzer LLM expands a proposition TREE (code-side data
    structure) until the leaf limit; separate calls, not one context.
  - Ground: one search-equipped Grounder call PER LEAF, run in parallel.
  - Synthesize: per internal node a Synthesizer LLM emits linear weights;
    the composition p = clip(b0 + sum b_j p_j) is COMPUTED BY THIS CODE.
  - Continual extension (approved design): the tree is the carried state.
    Each new day an Analyzer edit pass may add/remove nodes; existing leaves
    DELTA-ground (they see their previous value and search only the window
    since their last visit); code resynthesizes bottom-up, reusing stored
    weights for unchanged parents (the paper's resynthesis semantics) and
    re-eliciting weights only where children changed.

BLF (arXiv:2604.18576, Algorithm 1)
  - Iterative loop, at each step ONE generation returns (action, belief);
    belief is JSON {p, confidence, evidence_for, evidence_against,
    open_questions}; actions: search / submit; Tmax=10; submit clamps p to
    [0.05, 0.95]. The paper's LLM leak-filter is unnecessary here: the date
    gate is enforced in SQL, which is strictly stronger.
  - K trials per forecast (approved: K=3), aggregated by the paper's shrunken
    logit-space mean  p = sigmoid(alpha * mean_k logit(p_k)); the paper leaves
    alpha's dependence on between-trial variance unspecified, so we instantiate
    alpha = 1/(1 + var_k(logit(p_k))) and document it. Platt calibration is
    omitted (it requires a training set of resolved questions) and disclosed.
  - Continual: each rollout is a chain; all K trials of day t start from the
    chain's carried belief; the day's forecast is the aggregate; the carried
    belief is the median trial's.

Cost discipline: every LLM call logs token usage; a hard global call budget
aborts the run rather than overspending.
"""
from __future__ import annotations

import json
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .evidence import NewsIndex


class BudgetExceeded(RuntimeError):
    pass


class LLM:
    """Shared call wrapper: usage logging + hard call budget."""

    def __init__(self, client, model: str, budget: int = 4000):
        self.client = client
        self.model = model
        self.budget = budget
        self.calls = 0
        self.tok_in = 0
        self.tok_out = 0
        self._lock = threading.Lock()

    def __call__(self, system: str, user: str, effort: str = "medium") -> str:
        with self._lock:
            if self.calls >= self.budget:
                raise BudgetExceeded("call budget %d exhausted" % self.budget)
            self.calls += 1
        r = self.client.chat.completions.create(
            model=self.model, reasoning_effort=effort,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        u = getattr(r, "usage", None)
        if u:
            with self._lock:
                self.tok_in += u.prompt_tokens or 0
                self.tok_out += u.completion_tokens or 0
        return r.choices[0].message.content or ""

    def usage(self) -> dict:
        return {"calls": self.calls, "tokens_in": self.tok_in, "tokens_out": self.tok_out}


def _json_block(text: str) -> dict:
    m = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text or "", re.S)
    cands = m or re.findall(r"(\{.*\})", text or "", re.S)
    for c in cands:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON object in model output")


def _clip01(x) -> float:
    try:
        return min(1.0, max(0.0, float(x)))
    except (TypeError, ValueError):
        return 0.5


# ===================================================================== ANALYTICA

@dataclass
class Node:
    id: str
    statement: str
    parent: str | None = None
    p: float | None = None
    report: str = ""
    b0: float = 0.0
    betas: dict = field(default_factory=dict)   # child_id -> beta
    last_grounded: str | None = None            # date of last grounder visit

    def to_dict(self) -> dict:
        return {"id": self.id, "statement": self.statement, "parent": self.parent,
                "p": self.p, "report": self.report[:400], "b0": self.b0,
                "betas": self.betas, "last_grounded": self.last_grounded}


ANALYZER_SYS = """You are the Analyzer in the Analytica forecasting system.
You expand a proposition tree. Each proposition must be atomic, checkable, and
child propositions of one parent should be as independent of each other as
possible. Leaves should be simple enough that a search-equipped analyst can
score them from news evidence.

Respond with JSON only:
{"add": [{"parent": "<existing node id>", "statement": "<new proposition>"}, ...],
 "done": true/false}
Add 2-4 children per expansion; set done=true when the leaves are atomic enough
or the leaf budget is reached."""

ANALYZER_EDIT_SYS = """You are the Analyzer in the Analytica forecasting system,
revisiting an existing proposition tree on a new day. New information may have
made some propositions obsolete or new drivers relevant. Most days need no
change.

Respond with JSON only:
{"add": [{"parent": "<node id>", "statement": "..."}, ...],
 "remove": ["<leaf node id>", ...],
 "done": true}
Only add or remove if genuinely warranted; otherwise return empty lists."""

GROUNDER_SYS = """You are a Grounder agent in the Analytica forecasting system.
Your job: assess ONE atomic proposition against news evidence and output a soft
truth value.

Today is {date}. You may request searches of a dated news archive (articles up
to today only). First reply with up to {max_q} search queries as JSON:
{{"queries": ["...", "..."]}}
After receiving results you will be asked to conclude."""

GROUNDER_CONCL_SYS = """You are a Grounder agent in the Analytica forecasting
system. You have already gathered the evidence shown below for ONE atomic
proposition. Today is {date}; the evidence contains only articles up to today.

Assess the proposition's soft truth value from the evidence (and general
pre-{date} knowledge). Do NOT request searches. Output JSON only:
{{"p_true": <0..1>, "report": "<3-6 sentence evidence summary with dates>"}}"""

GROUNDER_DELTA_NOTE = """You previously assessed this proposition on {prev_date}
as p_true={prev_p} with report: {prev_report}
Search results below cover ONLY the period since then. If nothing material
changed, keep your value close and say so."""

SYNTH_SYS = """You are the Synthesizer in the Analytica forecasting system.
Given a parent proposition and its scored children, output linear-combination
weights expressing how the children's truth values compose into the parent's:
p_parent = b0 + sum_j beta_j * p_child_j.
Respond with JSON only:
{"b0": <float>, "betas": {"<child_id>": <float>, ...},
 "report": "<2-4 sentence synthesis rationale>"}
Prefer b0 >= 0, betas >= 0, and b0 + sum(betas) <= 1 so the result is a
probability."""


class AnalyticaFull:
    """Real Analytica orchestration over the date-gated corpus."""

    def __init__(self, llm: LLM, index: NewsIndex, max_leaves: int = 8,
                 expand_rounds: int = 3, queries_per_leaf: int = 2,
                 results_per_query: int = 5, workers: int = 6):
        self.llm = llm
        self.ix = index
        self.max_leaves = max_leaves
        self.expand_rounds = expand_rounds
        self.qpl = queries_per_leaf
        self.rpq = results_per_query
        self.workers = workers

    # -------- tree utilities (code, not model)
    @staticmethod
    def leaves(tree: dict) -> list:
        parents = {n.parent for n in tree.values() if n.parent}
        return [n for n in tree.values() if n.id not in parents]

    @staticmethod
    def _tree_json(tree: dict) -> str:
        return json.dumps([n.to_dict() for n in tree.values()], indent=1)

    def _compose(self, tree: dict, root_id: str) -> float:
        """The linear rule, computed HERE -- structure is code-enforced."""
        def val(nid: str) -> float:
            n = tree[nid]
            kids = [c for c in tree.values() if c.parent == nid]
            if not kids:
                return _clip01(n.p if n.p is not None else 0.5)
            s = n.b0 + sum(n.betas.get(c.id, 0.0) * val(c.id) for c in kids)
            n.p = _clip01(s)
            return n.p
        return val(root_id)

    # -------- stages
    def _analyze(self, tree: dict, root_id: str, day1: bool, date: str) -> dict:
        sys_p = ANALYZER_SYS if day1 else ANALYZER_EDIT_SYS
        rounds = self.expand_rounds if day1 else 1
        nid = [max((int(k[1:]) for k in tree if k[1:].isdigit()), default=0)]
        edits = {"added": [], "removed": []}
        for _ in range(rounds):
            if len(self.leaves(tree)) >= self.max_leaves and day1:
                break
            out = self.llm(sys_p, "DATE: %s\nLEAF BUDGET: %d\nCURRENT TREE:\n%s"
                           % (date, self.max_leaves, self._tree_json(tree)))
            try:
                d = _json_block(out)
            except ValueError:
                break
            for rm in (d.get("remove") or []):
                if rm in tree and rm != root_id and not any(
                        c.parent == rm for c in tree.values()):
                    del tree[rm]
                    for n in tree.values():
                        n.betas.pop(rm, None)
                    edits["removed"].append(rm)
            for a in (d.get("add") or []):
                if len(self.leaves(tree)) >= self.max_leaves:
                    break
                par = a.get("parent")
                if par in tree:
                    nid[0] += 1
                    new_id = "n%d" % nid[0]
                    tree[new_id] = Node(id=new_id, statement=str(a.get("statement", "")),
                                        parent=par)
                    edits["added"].append(new_id)
            if d.get("done"):
                break
        return edits

    def _ground_leaf(self, leaf: Node, date: str, searches_log: list) -> None:
        delta = leaf.last_grounded is not None
        sys_p = GROUNDER_SYS.format(date=date, max_q=self.qpl)
        user = "PROPOSITION: " + leaf.statement
        if delta:
            user += "\n\n" + GROUNDER_DELTA_NOTE.format(
                prev_date=leaf.last_grounded, prev_p=round(leaf.p or 0.5, 2),
                prev_report=(leaf.report or "")[:300])
        out = self.llm(sys_p, user, effort="low")
        try:
            queries = [str(q) for q in _json_block(out).get("queries", [])][:self.qpl]
        except ValueError:
            queries = [leaf.statement[:80]]
        from_date = None
        if delta:
            from_date = leaf.last_grounded
        ev = []
        for q in queries:
            arts = self.ix.search(q, to_date=date, from_date=from_date, k=self.rpq)
            searches_log.append({"leaf": leaf.id, "query": q, "from": from_date,
                                 "n": len(arts), "titles": [a.title for a in arts]})
            ev.append("QUERY: %s\n%s" % (q, "\n\n".join(a.render(500) for a in arts)
                                          or "(no results)"))
        concl_sys = GROUNDER_CONCL_SYS.format(date=date)
        concl_user = user + "\n\nEVIDENCE:\n" + "\n\n".join(ev)
        got = None
        for _attempt in range(2):
            out2 = self.llm(concl_sys, concl_user, effort="low")
            try:
                d = _json_block(out2)
                if "p_true" in d:
                    got = d
                    break
            except ValueError:
                pass
            concl_user += "\n\nYour previous reply was invalid. Output ONLY the JSON object with p_true and report."
        if got is not None:
            leaf.p = _clip01(got.get("p_true"))
            leaf.report = str(got.get("report", ""))[:1200]
        else:
            leaf.p = leaf.p if leaf.p is not None else 0.5
            leaf.report = "[GROUND-PARSE-FAILED] " + (leaf.report or "")
        leaf.last_grounded = date

    def _synthesize_weights(self, tree: dict, node: Node) -> None:
        kids = [c for c in tree.values() if c.parent == node.id]
        payload = {"parent": node.statement,
                   "children": [{"id": c.id, "statement": c.statement,
                                 "p": round(c.p or 0.5, 3),
                                 "report": (c.report or "")[:250]} for c in kids]}
        out = self.llm(SYNTH_SYS, json.dumps(payload, indent=1))
        try:
            d = _json_block(out)
            node.b0 = float(d.get("b0", 0.0))
            node.betas = {c.id: float((d.get("betas") or {}).get(c.id, 0.0))
                          for c in kids}
            node.report = str(d.get("report", ""))[:800]
        except (ValueError, TypeError):
            node.b0 = 0.0
            node.betas = {c.id: 1.0 / max(1, len(kids)) for c in kids}
            node.report = "[SYNTH-PARSE-FAILED] uniform fallback weights"

    # -------- one forecasting day
    def run_day(self, question_text: str, date: str, state: dict | None,
                history: list) -> tuple:
        searches: list = []
        if state is None:
            root = Node(id="n0", statement=question_text)
            tree = {"n0": root}
            edits = self._analyze(tree, "n0", day1=True, date=date)
        else:
            tree = {k: Node(**v) for k, v in state.items()}
            ctx = tree["n0"]
            if history:
                ctx.statement = question_text + " [Previous forecasts: %s]" % (
                    ", ".join(history))
            edits = self._analyze(tree, "n0", day1=False, date=date)

        to_ground = [lf for lf in self.leaves(tree) if lf.id != "n0"]
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            list(ex.map(lambda lf: self._ground_leaf(lf, date, searches), to_ground))

        # weights: elicit where missing or where the child set changed (paper's
        # resynthesis reuses weights elsewhere -- composition is recomputed in code)
        for n in tree.values():
            kids = [c.id for c in tree.values() if c.parent == n.id]
            if kids and set(kids) != set(n.betas.keys()):
                self._synthesize_weights(tree, n)
        forecast = self._compose(tree, "n0")

        new_state = {k: n.to_dict() for k, n in tree.items()}
        trace = {"tree": new_state, "edits": edits, "searches": searches}
        return forecast, new_state, trace


# ========================================================================= BLF

BLF_SYS = """You are a forecasting agent. Question: {question}
Today is {date}; you know nothing after this date. Resolution: {resolution}.

At every step output JSON only, with BOTH an action and your updated belief:
{{"action": {{"type": "search", "query": "..."}} OR {{"type": "submit", "p": <0..1>}},
 "belief": {{"p": <0..1>, "confidence": "low|medium|high",
            "evidence_for": ["..."], "evidence_against": ["..."],
            "open_questions": ["..."]}}}}

Search returns dated news articles (up to today only). Submit when further
search would not change your belief. You have at most {tmax} steps."""


class BLFFull:
    """Real BLF loop (Algorithm 1) over the date-gated corpus."""

    def __init__(self, llm: LLM, index: NewsIndex, tmax: int = 10,
                 k_trials: int = 3, results_per_query: int = 5):
        self.llm = llm
        self.ix = index
        self.tmax = tmax
        self.k = k_trials
        self.rpq = results_per_query

    def _trial(self, question, date, resolution, belief0, history, log: list) -> tuple:
        sys_p = BLF_SYS.format(question=question, date=date,
                               resolution=resolution, tmax=self.tmax)
        msgs = []
        if history:
            msgs.append("YOUR PREVIOUS FORECASTS:\n" + "\n".join(history))
        if belief0:
            msgs.append("YOUR BELIEF STATE FROM YESTERDAY:\n" +
                        json.dumps(belief0, indent=1))
        msgs.append("Step 1. Output your action and belief.")
        belief = belief0 or {"p": 0.5}
        for step in range(self.tmax):
            out = self.llm(sys_p, "\n\n".join(msgs), effort="low")
            try:
                d = _json_block(out)
                belief = d.get("belief") or belief
                action = d.get("action") or {}
            except ValueError:
                action = {}
            if action.get("type") == "submit":
                p = _clip01(action.get("p", belief.get("p", 0.5)))
                return min(0.95, max(0.05, p)), belief          # paper's clamp
            if action.get("type") == "search":
                q = str(action.get("query", ""))[:120]
                arts = self.ix.search(q, to_date=date, k=self.rpq)
                log.append({"query": q, "n": len(arts),
                            "titles": [a.title for a in arts]})
                msgs.append("SEARCH RESULTS for %r:\n%s" % (
                    q, "\n\n".join(a.render(450) for a in arts) or "(none)"))
                msgs.append("Step %d. Output your action and belief." % (step + 2))
            else:
                msgs.append("Invalid action. Output valid JSON with an action "
                            "and belief. Step %d." % (step + 2))
        p = _clip01(belief.get("p", 0.5))
        return min(0.95, max(0.05, p)), belief                   # forced submit

    @staticmethod
    def aggregate(ps: list) -> float:
        """Shrunken logit-space mean; alpha = 1/(1+var(logits)) (our
        instantiation of the paper's variance-dependent alpha)."""
        logits = [math.log(p / (1 - p)) for p in ps]
        m = sum(logits) / len(logits)
        var = sum((l - m) ** 2 for l in logits) / len(logits)
        alpha = 1.0 / (1.0 + var)
        z = alpha * m
        return 1.0 / (1.0 + math.exp(-z))

    def run_day(self, question_text: str, date: str, state: dict | None,
                history: list, resolution: str = "") -> tuple:
        log: list = []
        trials = []
        with ThreadPoolExecutor(max_workers=self.k) as ex:
            futs = [ex.submit(self._trial, question_text, date, resolution,
                              state, history, log) for _ in range(self.k)]
            for f in futs:
                trials.append(f.result())
        ps = [t[0] for t in trials]
        forecast = self.aggregate(ps)
        med_belief = sorted(trials, key=lambda t: t[0])[len(trials) // 2][1]
        trace = {"trial_ps": ps, "aggregate": forecast, "alpha_note":
                 "alpha=1/(1+var(logits))", "belief": med_belief, "searches": log}
        return forecast, med_belief, trace
