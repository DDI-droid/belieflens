"""Turn E1 trajectories and E2 programs into the measurements.

E1 asks: did the belief move, and did it move the right way? The two hockey
questions share one evidence stream and resolve opposite ways, so a harness
tracking evidence must separate them. A harness that moves both the same way is
responding to something other than the news.

E2 asks: was the reasoning program-like? If one literal-free program reproduces
every forecast a harness made, then its structure was stable and everything
that moved, moved through the variables -- which is what it means for the
variables to be the belief.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load(p: Path):
    if not p.exists():
        return None
    if p.suffix == ".jsonl":
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return json.loads(p.read_text(encoding="utf-8"))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, w=6, p=3):
    return " " * w if x is None else ("%*.*f" % (w, p, x))


def e1(runs) -> None:
    print("\n" + "=" * 78)
    print("E1 — sequential forecasts under a shared, date-gated evidence stream")
    print("=" * 78)

    by = defaultdict(list)
    dates = []
    for r in runs:
        by[(r["harness"], r["question_id"])].append(r)
        for t in r["turns"]:
            if t["date"] not in dates:
                dates.append(t["date"])
    dates.sort()

    print("\n%-11s %-11s %-3s %s" % ("harness", "question", "s", "  ".join(d[5:] for d in dates)))
    print("-" * 78)
    traj = {}
    for (h, q), rs in sorted(by.items()):
        for r in sorted(rs, key=lambda x: x["sample"]):
            row = {t["date"]: t["forecast"] for t in r["turns"]}
            traj[(h, q, r["sample"])] = row
            cells = "  ".join(fmt(row.get(d), 5, 2) for d in dates)
            print("%-11s %-11s %-3d %s" % (h, q, r["sample"], cells))

    # CORRECTED METRICS (adversarial review, 2026-08-27).  The original plan
    # scored "separation": net(USA) up, net(CAN) down.  That sign convention is
    # WRONG for how the event actually unfolded: BOTH teams won their
    # semifinals on 02-20 (verified in the corpus) and the final resolved on
    # 02-22, AFTER the last simulation date.  Inside the window the evidence
    # never discriminates the two finalists -- a correct forecaster raises
    # BOTH beliefs toward ~0.5 as other teams are eliminated.  The right
    # checks are therefore:
    #   semi-rise   both beliefs rise across the semifinal date (02-15->02-20)
    #   sum@last    P(USA)+P(CAN) approaches 1 by 02-20 and never exceeds ~1
    #   |sum-1|     complement coherence error on the eve of the final
    print("\n--- movement, and coherence of the paired beliefs ---")
    print("  (event path: BOTH finalists reached the gold-medal game inside the")
    print("   window; the final resolved after it. So both beliefs should RISE")
    print("   by 02-20 and P(USA)+P(CAN) should approach 1 without exceeding it.)")
    first, last = dates[0], dates[-1]
    prev_last = dates[-2] if len(dates) >= 2 else None
    print("\n%-11s %9s %9s %9s %10s %9s" % ("harness", "|Δ|total", "net USA",
                                            "net CAN", "semi-rise", "sum@last"))
    print("-" * 78)
    for h in sorted({k[0] for k in traj}):
        moves = []
        nets = {"hockey_usa": [], "hockey_can": []}
        semis, sums_last = [], []
        for (hh, qq, s), row in traj.items():
            if hh != h:
                continue
            seq = [row.get(d) for d in dates if row.get(d) is not None]
            if len(seq) >= 2:
                moves.append(sum(abs(seq[i + 1] - seq[i]) for i in range(len(seq) - 1)))
            # nets only over runs holding BOTH endpoint dates, so every net
            # spans the same window (review: mixed windows are not comparable)
            if qq in nets and row.get(first) is not None and row.get(last) is not None:
                nets[qq].append(row[last] - row[first])
        for s in sorted({k[2] for k in traj if k[0] == h}):
            ru = traj.get((h, "hockey_usa", s), {})
            rc = traj.get((h, "hockey_can", s), {})
            if ru.get(last) is not None and rc.get(last) is not None:
                sums_last.append(ru[last] + rc[last])
            if prev_last and None not in (ru.get(prev_last), ru.get(last),
                                          rc.get(prev_last), rc.get(last)):
                semis.append(1.0 if (ru[last] >= ru[prev_last]
                                     and rc[last] >= rc[prev_last]) else 0.0)
        print("%-11s %s %s %s %s %s"
              % (h, fmt(mean(moves), 9), fmt(mean(nets["hockey_usa"]), 9),
                 fmt(mean(nets["hockey_can"]), 9), fmt(mean(semis), 10, 2),
                 fmt(mean(sums_last), 9, 2)))
    print("\n  semi-rise: fraction of samples where BOTH beliefs rose across the")
    print("  semifinal date -- the evidence-tracking signal for this event path.")
    print("  sum@last near 1 with both nets positive = coherent tracking;")
    print("  sum well below 1 = at least one belief ignored the eliminations.")

    print("\n--- search behaviour ---")
    print("%-11s %10s %10s" % ("harness", "searches", "unique q"))
    print("-" * 78)
    for h in sorted({k[0] for k in by}):
        n, uq = [], set()
        for (hh, q), rs in by.items():
            if hh != h:
                continue
            for r in rs:
                for t in r["turns"]:
                    n.append(len(t["tool_calls"]))
                    for c in t["tool_calls"]:
                        uq.add(c["query"].lower()[:60])
        print("%-11s %s %10d" % (h, fmt(mean(n), 10, 2), len(uq)))


def e2(rep) -> None:
    print("\n" + "=" * 78)
    print("E2 — one literal-free program per harness, fitted to its own forecasts")
    print("=" * 78)
    print("\n%-22s %4s %5s %7s %7s %5s %6s %s"
          % ("harness / question", "inst", "vars", "MAE", "in-tol", "ops", "|corr|", "verdict"))
    print("-" * 78)
    for k, v in sorted(rep.items()):
        f, d = v.get("fit") or {}, v.get("degeneracy") or {}
        verdict = ("degenerate" if d.get("suspect") else
                   "fits" if v.get("ok") else "partial")
        print("%-22s %4d %5d %s %6s%% %5d %s  %s"
              % (k, v.get("n_instances", 0), len(v.get("schema", [])),
                 fmt(f.get("mae"), 7, 4),
                 "  --" if f.get("within_tol") is None else "%4.0f" % (100 * f["within_tol"]),
                 d.get("n_ops", 0), fmt(d.get("max_abs_corr"), 6, 2), verdict))

    print("\n  MAE is the headline: a low MAE means one fixed program reproduced every")
    print("  forecast, so the reasoning structure held still and only the variables moved.")
    print("  A high MAE means the structure itself changed as evidence arrived.")

    print("\n--- what moved, in the discovered basis ---")
    for k, v in sorted(rep.items()):
        insts = v.get("instances") or []
        if len(insts) < 2:
            continue
        schema = v.get("schema") or []
        by_s = defaultdict(list)
        for i in insts:
            by_s[i["sample"]].append(i)
        print("\n  %s" % k)
        moved = []
        for name in schema:
            spans = []
            for s, group in by_s.items():
                g = sorted(group, key=lambda x: x["date"])
                vals = [x["bindings"].get(name) for x in g]
                vals = [x for x in vals if x is not None]
                if len(vals) >= 2:
                    spans.append(max(vals) - min(vals))
            if spans:
                moved.append((mean(spans), name))
        moved.sort(reverse=True)
        for rng, name in moved[:6]:
            print("      %-42s range %.3f" % (name, rng))
        still = [n for r, n in moved if r < 1e-9]
        if still:
            print("      held fixed across every date: %s" % ", ".join(still[:6]))

    print("\n--- recovered programs ---")
    for k, v in sorted(rep.items()):
        if not v.get("program"):
            continue
        print("\n  # %s   (MAE %s)" % (k, fmt((v.get("fit") or {}).get("mae"), 6, 4).strip()))
        for line in v["program"].splitlines():
            print("    " + line)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/exp1")
    runs = load(out / "e1_runs.jsonl")
    rep = load(out / "e2_programs.json")
    if runs:
        e1(runs)
    if rep:
        e2(rep)
    if not runs and not rep:
        print("nothing in %s -- run scripts/run_exp1.py first" % out)
    print()


if __name__ == "__main__":
    main()
