"""Metrics for the four properties a belief representation can have.

Faithfulness  does the decoded structure agree with the belief the same model
              reports directly?
Coherence     is the structure a probability model at all, and do its numeric
              and verbal channels agree?
Locality      when evidence targets one node, does the change stay there?
Plasticity    does the belief move the way the model itself said it would --
              and does it move differently for confirming vs disconfirming
              evidence?
"""
from __future__ import annotations

import math
from statistics import mean, pstdev

from .dsl import (BeliefSpec, WEP_SCALE, coherence_violation_rate,
                  wep_numeric_disagreement, wep_rank_violations)


# ------------------------------------------------------------ small stats

def _f(xs):
    return [x for x in xs if x is not None and isinstance(x, float) and x == x]


def summarise(xs) -> dict:
    xs = _f(xs)
    if not xs:
        return {"n": 0, "mean": None, "sd": None, "median": None}
    s = sorted(xs)
    mid = len(s) // 2
    med = s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
    return {"n": len(xs), "mean": mean(xs),
            "sd": pstdev(xs) if len(xs) > 1 else 0.0, "median": med}


def spearman(xs, ys) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys)
             if x is not None and y is not None and x == x and y == y]
    if len(pairs) < 3:
        return None
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


# ----------------------------------------------------------- faithfulness

def faithfulness(by_arm: dict) -> dict:
    """`by_arm` maps arm -> {question_id: [p, p, ...]} across samples.

    Reports, per arm pair, the mean absolute gap in the per-question mean, the
    rank agreement across questions, and each arm's within-question spread
    (which is exactly Analytica's `Var` column, measured without grounding).
    """
    qmean = {a: {q: mean(_f(ps)) for q, ps in d.items() if _f(ps)}
             for a, d in by_arm.items()}
    qsd = {a: {q: pstdev(_f(ps)) for q, ps in d.items() if len(_f(ps)) > 1}
           for a, d in by_arm.items()}

    out = {"per_arm": {}, "pairs": {}}
    for a in by_arm:
        out["per_arm"][a] = {
            "mean_p": summarise(list(qmean.get(a, {}).values())),
            "within_question_sd": summarise(list(qsd.get(a, {}).values())),
        }
    arms = list(by_arm)
    for i in range(len(arms)):
        for j in range(len(arms)):
            if i == j:
                continue
            a, b = arms[i], arms[j]
            shared = sorted(set(qmean.get(a, {})) & set(qmean.get(b, {})))
            if not shared:
                continue
            xs = [qmean[a][q] for q in shared]
            ys = [qmean[b][q] for q in shared]
            out["pairs"]["%s->%s" % (a, b)] = {
                "n_questions": len(shared),
                "mean_abs_gap": mean(abs(x - y) for x, y in zip(xs, ys)),
                "mean_signed_gap": mean(y - x for x, y in zip(xs, ys)),
                "spearman": spearman(xs, ys),
                "frac_within_0.05": mean(1.0 if abs(x - y) <= 0.05 else 0.0
                                         for x, y in zip(xs, ys)),
            }
    return out


def decomposition(by_arm: dict, direct="direct", conditioned="conditioned",
                  composed="composed:linear") -> dict:
    """Split the total structure effect into elicitation and aggregation.

    total = composed - direct
          = (conditioned - direct)  [writing the belief down]
          + (composed - conditioned) [composing it arithmetically]
    """
    qm = {a: {q: mean(_f(ps)) for q, ps in d.items() if _f(ps)}
          for a, d in by_arm.items()}
    shared = sorted(set(qm.get(direct, {})) & set(qm.get(conditioned, {}))
                    & set(qm.get(composed, {})))
    if not shared:
        return {"n": 0}
    elic = [qm[conditioned][q] - qm[direct][q] for q in shared]
    aggr = [qm[composed][q] - qm[conditioned][q] for q in shared]
    tot = [qm[composed][q] - qm[direct][q] for q in shared]
    return {"n": len(shared),
            "elicitation_effect": summarise([abs(x) for x in elic]),
            "elicitation_signed": summarise(elic),
            "aggregation_effect": summarise([abs(x) for x in aggr]),
            "aggregation_signed": summarise(aggr),
            "total_effect": summarise([abs(x) for x in tot]),
            "aggregation_share": (mean(abs(x) for x in aggr) /
                                  (mean(abs(x) for x in aggr) + mean(abs(x) for x in elic))
                                  if (mean(abs(x) for x in aggr) + mean(abs(x) for x in elic)) else None)}


# -------------------------------------------------------------- coherence

def coherence(specs) -> dict:
    specs = [s for s in specs if s is not None]
    if not specs:
        return {"n": 0}
    gaps = []
    for s in specs:
        try:
            gaps.append(abs(s.decode("linear") - s.decode("linear_simplex")))
        except Exception:  # noqa: BLE001
            pass
    return {
        "n": len(specs),
        "simplex_violation_rate": summarise([coherence_violation_rate(s) for s in specs]),
        "frac_specs_with_any_violation": mean(
            1.0 if coherence_violation_rate(s) > 0 else 0.0 for s in specs),
        "linear_vs_simplex_gap": summarise(gaps),
        "wep_numeric_disagreement": summarise([wep_numeric_disagreement(s) for s in specs]),
        "wep_rank_violation_rate": summarise([wep_rank_violations(s) for s in specs]),
        "n_nodes": summarise([float(len(s.nodes)) for s in specs]),
        "n_leaves": summarise([float(len(s.leaves())) for s in specs]),
        "depth": summarise([float(s.depth()) for s in specs]),
    }


# ------------------------------------------------- locality and plasticity

def _node_deltas(prior: BeliefSpec, post: BeliefSpec) -> dict:
    a, b = prior.by_id(), post.by_id()
    return {nid: b[nid].p - a[nid].p for nid in a if nid in b}


def locality(prior: BeliefSpec, post: BeliefSpec, target: str) -> float | None:
    d = _node_deltas(prior, post)
    tot = sum(abs(v) for k, v in d.items() if k != prior.root)
    if tot < 1e-9:
        return None
    return abs(d.get(target, 0.0)) / tot


def structural_churn(prior: BeliefSpec, post: BeliefSpec) -> dict:
    a, b = set(prior.by_id()), set(post.by_id())
    inter, union = len(a & b), len(a | b)
    pa, pb = prior.by_id(), post.by_id()
    wdrift = [abs(pb[n].w - pa[n].w) for n in (a & b) if pa[n].parent is not None]
    b0drift = [abs(post.b0.get(k, 0.0) - v) for k, v in prior.b0.items()]
    return {"node_jaccard": inter / union if union else 1.0,
            "nodes_added": len(b - a), "nodes_removed": len(a - b),
            "weight_l1_drift": sum(wdrift), "weight_max_drift": max(wdrift) if wdrift else 0.0,
            "b0_l1_drift": sum(b0drift)}


def rigidity(prior: BeliefSpec, post: BeliefSpec, target: str,
             rule: str = "linear") -> dict:
    """How much of the root movement the evidence *implied* actually happened.

    At frozen coefficients the root must move by beta_path(target) * dp_target.
    R = 1 - actual/implied.   R ~ 0 faithful; R > 0 the conclusion resisted the
    evidence; R < 0 it over-reacted.  `beta_compensation` is the coefficient
    movement that absorbed the difference -- under the local-update protocol it
    must be 0, so any non-zero value is protocol violation, which is itself a
    finding.
    """
    d = _node_deltas(prior, post)
    dp = d.get(target)
    if dp is None:
        return {"error": "target missing from post"}
    bpath = prior.path_weight(target)
    implied = bpath * dp
    actual = post.decode(rule) - prior.decode(rule)
    ch = structural_churn(prior, post)
    r = None
    if abs(implied) > 1e-4:
        r = 1.0 - actual / implied
    return {"dp_target": dp, "beta_path": bpath, "implied_droot": implied,
            "actual_droot": actual, "rigidity": r,
            "beta_compensation": ch["weight_l1_drift"] + ch["b0_l1_drift"],
            "structural_churn": ch}


def stated_vs_enacted(prior: BeliefSpec, post: BeliefSpec, target: str,
                      stated_p_to: float | None) -> dict:
    """Did the model move the node to where it said it would?

    `stated_p_to` came from the node's own `sensitivity` slot, written before
    the evidence was shown.  This is the cleanest self-consistency test in the
    suite: the model authored both the promise and the update.
    """
    if stated_p_to is None:
        return {"n": 0}
    a, b = prior.by_id(), post.by_id()
    if target not in a or target not in b:
        return {"error": "target missing"}
    promised = stated_p_to - a[target].p
    enacted = b[target].p - a[target].p
    ratio = enacted / promised if abs(promised) > 1e-4 else None
    return {"promised_delta": promised, "enacted_delta": enacted,
            "shortfall": promised - enacted, "follow_through": ratio,
            "sign_agreement": (promised == 0 and enacted == 0) or
                              (promised * enacted > 0)}


def aggregate_updates(results) -> dict:
    """Roll up a list of `UpdateResult`s into the paper's headline table."""
    buckets: dict = {}
    for r in results:
        if r.error or r.post is None or r.prior is None:
            continue
        key = (r.mode, r.condition)
        b = buckets.setdefault(key, {"rigidity": [], "locality": [],
                                     "follow_through": [], "beta_comp": [],
                                     "droot": [], "abs_droot": [],
                                     "jaccard": [], "sign_agree": []})
        rg = rigidity(r.prior, r.post, r.target)
        if "error" in rg:
            continue
        b["rigidity"].append(rg["rigidity"])
        b["beta_comp"].append(rg["beta_compensation"])
        b["droot"].append(rg["actual_droot"])
        b["abs_droot"].append(abs(rg["actual_droot"]))
        b["jaccard"].append(rg["structural_churn"]["node_jaccard"])
        loc = locality(r.prior, r.post, r.target)
        if loc is not None:
            b["locality"].append(loc)
        sve = stated_vs_enacted(r.prior, r.post, r.target, r.stated_p_to)
        if sve.get("follow_through") is not None:
            b["follow_through"].append(sve["follow_through"])
        if "sign_agreement" in sve:
            b["sign_agree"].append(1.0 if sve["sign_agreement"] else 0.0)

    out = {}
    for (mode, cond), b in buckets.items():
        out["%s/%s" % (mode, cond)] = {k: summarise(v) for k, v in b.items()}

    # Asymmetry: does confirming evidence move the belief more than
    # disconfirming evidence of the model's own stated equal strength?
    for mode in {m for m, _ in buckets}:
        sup = buckets.get((mode, "support"), {}).get("abs_droot", [])
        con = buckets.get((mode, "contradict"), {}).get("abs_droot", [])
        pla = buckets.get((mode, "placebo"), {}).get("abs_droot", [])
        if sup and con:
            out.setdefault("asymmetry", {})[mode] = {
                "mean_abs_droot_support": mean(sup),
                "mean_abs_droot_contradict": mean(con),
                "support_minus_contradict": mean(sup) - mean(con),
                "placebo_noise_floor": mean(pla) if pla else None,
            }
    return out
