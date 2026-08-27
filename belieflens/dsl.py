"""BeliefSpec: a DSL for externalised LLM belief states.

Design notes
------------
BeliefSpec is deliberately a *superset* of the two representations we are
comparing against:

  * Analytica (arXiv:2604.23072) represents a belief as a tree of soft
    propositions with p_true in [0,1] and a *linear* synthesis rule
    p_parent = b0 + sum_j beta_j * p_child_j.  Analytica's Appendix B shows
    this is equivalent to a linear Bayesian network ONLY when
    beta_j in [0,1] and b0 + sum_j beta_j <= 1 -- a constraint the paper
    explicitly did not enforce.  We keep both the unconstrained rule
    (`linear`) and the coherent projection (`linear_simplex`) so the cost of
    coherence is measurable rather than assumed.

  * Agent-BRACE (arXiv:2605.11436) represents a belief as a set of atomic
    NL claims each carrying an ordinal Words-of-Estimative-Probability
    label.  We attach a `wep` field to every node, which gives a second,
    independent readout of the same belief and therefore a free internal
    consistency check (verbal vs numeric).

The one thing neither paper records, and the thing belief *dynamics* needs,
is how a node is expected to move under evidence.  BeliefSpec adds a
`sensitivity` slot: verbalised conditionals ("if X is observed, p -> 0.9").
Collected *before* the evidence arrives, this turns belief updating into a
falsifiable prediction: we can compare the update the model said it would
make with the update it actually makes.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any

# Agent-BRACE's ordinal scale, with the nominal probabilities used in that
# paper's calibration analysis (Sec. 4, Fig. 4).
WEP_SCALE: dict[str, float] = {
    "confirmed": 1.0,
    "almost certain": 0.9,
    "probable": 0.75,
    "possible": 0.5,
    "unlikely": 0.25,
    "doubtful": 0.1,
    "unknown": 0.5,   # genuinely uninformative -> centre of the interval
}
WEP_ORDER = ["confirmed", "almost certain", "probable", "possible",
             "unlikely", "doubtful", "unknown"]


class DSLError(ValueError):
    pass


@dataclass
class Conditional:
    """A verbalised likelihood slot: 'if `cue` is observed, p becomes `p_to`'."""
    cue: str
    p_to: float

    @staticmethod
    def parse(d: dict) -> "Conditional":
        return Conditional(cue=str(d.get("if") or d.get("cue", "")),
                           p_to=clip01(float(d["p_to"])))


@dataclass
class Node:
    id: str
    claim: str
    p: float                      # soft truth value (Analytica p_true)
    wep: str = "possible"         # ordinal label (Agent-BRACE)
    parent: str | None = None
    w: float = 0.0                # edge coefficient beta_j into `parent`
    kind: str = "driver"          # root | driver | leaf
    sensitivity: list[Conditional] = field(default_factory=list)
    note: str = ""

    @property
    def is_root(self) -> bool:
        return self.parent is None


@dataclass
class BeliefSpec:
    question: str
    as_of: str
    nodes: list[Node]
    root: str
    b0: dict[str, float] = field(default_factory=dict)   # intercept per parent
    rule: str = "linear"
    outcomes: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # ---------- structure ----------
    def by_id(self) -> dict[str, Node]:
        return {n.id: n for n in self.nodes}

    def children(self, nid: str) -> list[Node]:
        return [n for n in self.nodes if n.parent == nid]

    def leaves(self) -> list[Node]:
        parents = {n.parent for n in self.nodes if n.parent}
        return [n for n in self.nodes if n.id not in parents]

    def depth(self) -> int:
        idx = self.by_id()

        def d(nid: str, seen: frozenset) -> int:
            n = idx[nid]
            if n.parent is None or n.parent in seen:
                return 0
            return 1 + d(n.parent, seen | {nid})

        return max((d(n.id, frozenset()) for n in self.nodes), default=0)

    def path_weight(self, nid: str) -> float:
        """Analytica's beta-path: product of edge betas from root to `nid`."""
        idx, w, cur, guard = self.by_id(), 1.0, nid, 0
        while idx[cur].parent is not None and guard < 64:
            w *= idx[cur].w
            cur = idx[cur].parent
            guard += 1
        return w

    # ---------- decoding: BeliefSpec -> a single probability ----------
    def decode(self, rule: str | None = None) -> float:
        rule = rule or self.rule
        if rule == "stated":
            return clip01(self.by_id()[self.root].p)
        if rule == "wep_only":
            return self._decode_recursive(self.root, self._combine_wep)
        if rule == "linear":
            return self._decode_recursive(self.root, self._combine_linear)
        if rule == "linear_simplex":
            return self._decode_recursive(self.root, self._combine_simplex)
        if rule == "noisy_or":
            return self._decode_recursive(self.root, self._combine_noisy_or)
        raise DSLError("unknown decode rule " + repr(rule))

    def _decode_recursive(self, nid, combine, seen=frozenset()) -> float:
        if nid in seen:
            raise DSLError("cycle at " + nid)
        kids = self.children(nid)
        if not kids:
            return clip01(self.by_id()[nid].p)
        vals = {k.id: self._decode_recursive(k.id, combine, seen | {nid}) for k in kids}
        return combine(nid, kids, vals)

    def _combine_linear(self, nid, kids, vals) -> float:
        return clip01(self.b0.get(nid, 0.0) + sum(k.w * vals[k.id] for k in kids))

    def _combine_simplex(self, nid, kids, vals) -> float:
        """Analytica App. B's coherence constraint, enforced by projection.

        beta_j >= 0 and b0 + sum beta_j <= 1.  Renormalise when violated so the
        graph really is a Bayes net rather than an arbitrary affine map.
        """
        b0 = max(0.0, self.b0.get(nid, 0.0))
        ws = [max(0.0, k.w) for k in kids]
        total = b0 + sum(ws)
        if total > 1.0 and total > 0:
            b0 = b0 / total
            ws = [w / total for w in ws]
        return clip01(b0 + sum(w * vals[k.id] for w, k in zip(ws, kids)))

    def _combine_noisy_or(self, nid, kids, vals) -> float:
        q = 1.0 - clip01(self.b0.get(nid, 0.0))
        for k in kids:
            q *= (1.0 - clip01(abs(k.w)) * vals[k.id])
        return clip01(1.0 - q)

    def _combine_wep(self, nid, kids, vals) -> float:
        """Ignore the numbers entirely; combine the ordinal labels only.

        Tests whether the numeric channel carries information beyond the
        verbal one, or is decoration on top of it.
        """
        idx = self.by_id()
        num = sum(max(0.0, k.w) * WEP_SCALE.get(idx[k.id].wep, 0.5) for k in kids)
        den = sum(max(0.0, k.w) for k in kids) or 1.0
        return clip01(num / den)

    # ---------- serialisation ----------
    def to_dict(self) -> dict:
        d = asdict(self)
        out_nodes = []
        for n in self.nodes:
            nd = {k: v for k, v in asdict(n).items() if k != "sensitivity"}
            nd["sensitivity"] = [{"if": c.cue, "p_to": c.p_to} for c in n.sensitivity]
            out_nodes.append(nd)
        d["nodes"] = out_nodes
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_dict(d: dict) -> "BeliefSpec":
        nodes = []
        for raw in d.get("nodes", []):
            par = raw.get("parent")
            if par in (None, "", "null", "None"):
                par = None
            else:
                par = str(par)
            nodes.append(Node(
                id=str(raw["id"]),
                claim=str(raw.get("claim", "")),
                p=clip01(float(raw.get("p", 0.5))),
                wep=norm_wep(raw.get("wep", "possible")),
                parent=par,
                w=float(raw.get("w", 0.0)),
                kind=str(raw.get("kind", "driver")),
                sensitivity=[Conditional.parse(c) for c in raw.get("sensitivity", [])
                             if isinstance(c, dict) and "p_to" in c],
                note=str(raw.get("note", "")),
            ))
        if not nodes:
            raise DSLError("no nodes")
        root = d.get("root") or next((n.id for n in nodes if n.parent is None), nodes[0].id)
        b0 = {str(k): float(v) for k, v in (d.get("b0") or {}).items()}
        return BeliefSpec(
            question=str(d.get("question", "")),
            as_of=str(d.get("as_of", "")),
            nodes=nodes, root=str(root), b0=b0,
            rule=str(d.get("rule", "linear")),
            outcomes=[str(o) for o in d.get("outcomes", [])],
            meta=d.get("meta", {}) or {},
        )

    @staticmethod
    def from_text(text: str) -> "BeliefSpec":
        return BeliefSpec.from_dict(extract_json(text))


# ---------- validation ----------

def validate(spec: BeliefSpec) -> list[str]:
    """Structural + coherence problems.  Returned, not raised: violation
    *rates* are one of the things we are measuring."""
    problems: list[str] = []
    idx = spec.by_id()
    if len(idx) != len(spec.nodes):
        problems.append("duplicate node ids")
    if spec.root not in idx:
        problems.append("root id missing")
    roots = [n for n in spec.nodes if n.parent is None]
    if len(roots) != 1:
        problems.append("expected exactly 1 root, found %d" % len(roots))
    for n in spec.nodes:
        if n.parent and n.parent not in idx:
            problems.append("%s: dangling parent %s" % (n.id, n.parent))
        if not (0.0 <= n.p <= 1.0):
            problems.append("%s: p out of range" % n.id)
        if n.wep not in WEP_SCALE:
            problems.append("%s: unknown WEP label %r" % (n.id, n.wep))
    for n in spec.nodes:
        cur, guard = n.id, 0
        while cur in idx and idx[cur].parent is not None and guard < 128:
            cur = idx[cur].parent
            guard += 1
        if guard >= 128:
            problems.append("%s: cycle" % n.id)
            break
    # Analytica App. B coherence condition
    for n in spec.nodes:
        kids = spec.children(n.id)
        if not kids:
            continue
        b0 = spec.b0.get(n.id, 0.0)
        if any(k.w < 0 for k in kids):
            problems.append("%s: negative beta (breaks Bayes-net equivalence)" % n.id)
        if b0 < 0:
            problems.append("%s: negative intercept" % n.id)
        tot = b0 + sum(k.w for k in kids)
        if tot > 1.0 + 1e-9:
            problems.append("%s: b0+sum(beta)=%.3f > 1 (not a probability)" % (n.id, tot))
    return problems


def coherence_violation_rate(spec: BeliefSpec) -> float:
    """Fraction of internal nodes whose synthesis rule is not a probability."""
    internal = [n for n in spec.nodes if spec.children(n.id)]
    if not internal:
        return 0.0
    bad = 0
    for n in internal:
        kids = spec.children(n.id)
        b0 = spec.b0.get(n.id, 0.0)
        if b0 < 0 or any(k.w < 0 for k in kids) or b0 + sum(k.w for k in kids) > 1.0 + 1e-9:
            bad += 1
    return bad / len(internal)


def wep_numeric_disagreement(spec: BeliefSpec) -> float:
    """Mean |p - nominal(wep)| across nodes: do the two channels agree?"""
    if not spec.nodes:
        return float("nan")
    return sum(abs(n.p - WEP_SCALE.get(n.wep, 0.5)) for n in spec.nodes) / len(spec.nodes)


def wep_rank_violations(spec: BeliefSpec) -> float:
    """Fraction of node pairs where numeric and ordinal orderings disagree."""
    ns = [n for n in spec.nodes if n.wep != "unknown"]
    bad = tot = 0
    for i in range(len(ns)):
        for j in range(i + 1, len(ns)):
            a, b = ns[i], ns[j]
            va, vb = WEP_SCALE[a.wep], WEP_SCALE[b.wep]
            if abs(va - vb) < 1e-9 or abs(a.p - b.p) < 1e-9:
                continue
            tot += 1
            if (a.p > b.p) != (va > vb):
                bad += 1
    return bad / tot if tot else float("nan")


# ---------- helpers ----------

def clip01(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.5
    if x != x or x in (float("inf"), float("-inf")):
        return 0.5
    return min(1.0, max(0.0, x))


def norm_wep(s) -> str:
    s = str(s).strip().lower().replace("_", " ")
    if s in WEP_SCALE:
        return s
    aliases = {"certain": "confirmed", "very likely": "almost certain",
               "likely": "probable", "maybe": "possible",
               "very unlikely": "doubtful", "unsure": "unknown", "": "possible"}
    return aliases.get(s, s)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> dict:
    """Pull the first well-formed JSON object out of a model response."""
    for cand in _FENCE.findall(text or ""):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
    s = text or ""
    start = s.find("{")
    while start != -1:
        depth, instr, esc = 0, False, False
        for i in range(start, len(s)):
            c = s[i]
            if instr:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    instr = False
                continue
            if c == '"':
                instr = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = s.find("{", start + 1)
    raise DSLError("no JSON object found in response")


def extract_probability(text: str) -> float:
    """For the unstructured arm: read a bare probability out of prose."""
    try:
        d = extract_json(text)
        for k in ("p", "probability", "p_true", "answer"):
            if k in d:
                return clip01(d[k])
    except (DSLError, TypeError, ValueError):
        pass
    t = text or ""
    m = re.findall(r"(?:probability|p)\s*[:=]\s*([01]?\.\d+|[01](?:\.0+)?)", t, re.I)
    if m:
        return clip01(m[-1])
    m = re.findall(r"([0-9]{1,3}(?:\.[0-9]+)?)\s*%", t)
    if m:
        return clip01(float(m[-1]) / 100.0)
    m = re.findall(r"\b(0\.\d+)\b", t)
    if m:
        return clip01(m[-1])
    raise DSLError("no probability found in response")
