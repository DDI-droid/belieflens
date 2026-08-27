"""Turn results/<run>/{draws,updates}.jsonl into the paper's tables.

  python scripts/analyze.py results/mock_mock-1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from belieflens import metrics as M                     # noqa: E402
from belieflens.arms import UpdateResult                # noqa: E402
from belieflens.dsl import BeliefSpec                   # noqa: E402


def load(path: Path):
    rows = []
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def fmt(s) -> str:
    if not isinstance(s, dict) or s.get("mean") is None:
        return "     --"
    return "%6.3f +-%.3f (n=%d)" % (s["mean"], s.get("sd") or 0.0, s["n"])


def main() -> None:
    run = Path(sys.argv[1] if len(sys.argv) > 1 else "results/mock_mock-1")
    draws = load(run / "draws.jsonl")
    updates = load(run / "updates.jsonl")

    # ---------------- Table 1: faithfulness ----------------
    by_arm: dict = {}
    specs = []
    for r in draws:
        if r["kind"] != "draw":
            continue
        if r["arm"] == "spec" and r.get("spec"):
            specs.append(BeliefSpec.from_dict(r["spec"]))
        if r.get("p") is None:
            continue
        by_arm.setdefault(r["arm"], {}).setdefault(r["question_id"], []).append(r["p"])

    if by_arm:
        f = M.faithfulness(by_arm)
        print("\n=== Table 1. Arms: level and within-question spread ===")
        print("%-24s %-24s %s" % ("arm", "mean p", "within-question sd"))
        for arm, v in sorted(f["per_arm"].items()):
            print("%-24s %-24s %s" % (arm, fmt(v["mean_p"]), fmt(v["within_question_sd"])))

        print("\n=== Table 2. Faithfulness: does the structure agree with the direct belief? ===")
        print("%-38s %8s %8s %8s %8s" % ("pair", "|gap|", "signed", "rho", "<=.05"))
        for pair, v in sorted(f["pairs"].items()):
            if not pair.startswith("direct->"):
                continue
            rho = "%8.3f" % v["spearman"] if v["spearman"] is not None else "      --"
            print("%-38s %8.3f %8.3f %s %8.2f"
                  % (pair, v["mean_abs_gap"], v["mean_signed_gap"], rho, v["frac_within_0.05"]))

        dec = M.decomposition(by_arm)
        if dec.get("n"):
            print("\n=== Table 3. Where does the structure effect come from? ===")
            print("  n questions            %d" % dec["n"])
            print("  elicitation |C - A|    %s" % fmt(dec["elicitation_effect"]))
            print("  aggregation |B - C|    %s" % fmt(dec["aggregation_effect"]))
            print("  total       |B - A|    %s" % fmt(dec["total_effect"]))
            if dec["aggregation_share"] is not None:
                print("  aggregation share      %.1f%%" % (100 * dec["aggregation_share"]))

    # ---------------- Table 4: coherence ----------------
    if specs:
        c = M.coherence(specs)
        print("\n=== Table 4. Coherence of the elicited structures (n=%d) ===" % c["n"])
        print("  nodes / leaves / depth     %s / %s / %s"
              % (fmt(c["n_nodes"]), fmt(c["n_leaves"]), fmt(c["depth"])))
        print("  simplex violation rate     %s" % fmt(c["simplex_violation_rate"]))
        print("  specs w/ any violation     %.2f" % c["frac_specs_with_any_violation"])
        print("  |linear - simplex| at root %s" % fmt(c["linear_vs_simplex_gap"]))
        print("  |p - nominal(WEP)|         %s" % fmt(c["wep_numeric_disagreement"]))
        print("  WEP/numeric rank conflicts %s" % fmt(c["wep_rank_violation_rate"]))

    # ---------------- Table 5: update dynamics ----------------
    if updates:
        objs = []
        for r in updates:
            if r["kind"] != "update":
                continue
            u = UpdateResult(
                question_id=r["question_id"], sample_id=r["sample_id"], mode=r["mode"],
                condition=r["condition"], target=r["target"],
                prior=BeliefSpec.from_dict(r["prior"]) if r.get("prior") else None,
                post=BeliefSpec.from_dict(r["post"]) if r.get("post") else None,
                prior_root=r.get("prior_root"), post_root=r.get("post_root"),
                stated_p_to=r.get("stated_p_to"), error=r.get("error", ""))
            objs.append(u)
        agg = M.aggregate_updates(objs)
        print("\n=== Table 5. Belief dynamics under injected evidence ===")
        cols = ["rigidity", "follow_through", "locality", "beta_comp", "abs_droot"]
        print("%-22s %s" % ("mode/condition", " ".join("%-22s" % c for c in cols)))
        for key in sorted(k for k in agg if k != "asymmetry"):
            row = agg[key]
            print("%-22s %s" % (key, " ".join("%-22s" % fmt(row.get(c)) for c in cols)))
        if "asymmetry" in agg:
            print("\n  --- update asymmetry (support vs contradict, same stated strength) ---")
            for mode, v in agg["asymmetry"].items():
                floor = ("%.3f" % v["placebo_noise_floor"]) if v["placebo_noise_floor"] is not None else "--"
                print("  %-8s support %.3f | contradict %.3f | diff %+.3f | placebo floor %s"
                      % (mode, v["mean_abs_droot_support"], v["mean_abs_droot_contradict"],
                         v["support_minus_contradict"], floor))

    # unstructured control moves
    direct_moves = [abs((r.get("post_root") or 0) - (r.get("prior_root") or 0))
                    for r in updates if r.get("mode") == "direct" and not r.get("error")
                    and r.get("post_root") is not None and r.get("prior_root") is not None]
    if direct_moves:
        print("\n  unstructured control: mean |dp| = %.3f (n=%d)"
              % (sum(direct_moves) / len(direct_moves), len(direct_moves)))
    print()


if __name__ == "__main__":
    main()
