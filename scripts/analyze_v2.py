"""Metrics for the v2 run: 6 questions in 3 matched groups, 4 real harnesses.

Two things differ from the v1 analysis and both matter.

1. COHERENCE IS PER GROUP.  Each group is a two-sided contest binarised into
   both finalists, so P(a) + P(b) must be <= 1 always, and -> 1 once the field
   has narrowed to those two.  v1 could only do this for hockey; here every
   group gets it.

2. THE OUTCOME PROBE IS NOT PART OF THE TRAJECTORY.  D6 = R+2, by which point
   the corpus contains the result.  Mixing it into a belief-dynamics slope
   would corrupt the slope, so every dynamics metric runs on D1..D5 and the
   probe is reported on its own: does the harness move to the truth when the
   answer is in the news?

Writes results/exp1_v2/metrics_v2.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from belieflens.questions_v2 import QUESTIONS, GROUPS, TRAJECTORY, PROBE_DATE  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/exp1_v2")
TRUTH = {q.id: float(q.truth) for q in QUESTIONS}
QIDS = [q.id for q in QUESTIONS]


def jlines(p):
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def load():
    seq = defaultdict(dict)      # (h,q,sample) -> {date: p}
    seq_search = defaultdict(list)
    for r in jlines(OUT / "e1_runs.jsonl"):
        for t in r["turns"]:
            if t.get("forecast") is not None:
                seq[(r["harness"], r["question_id"], r["sample"])][t["date"]] = t["forecast"]
            tr = t.get("trace") or {}
            seq_search[r["harness"]].append(len(tr.get("searches") or []))
    par = defaultdict(list)      # (h,q,date) -> [p...]
    par_search = defaultdict(list)
    for r in jlines(OUT / "e1_independent.jsonl"):
        if r.get("forecast") is not None:
            par[(r["harness"], r["question_id"], r["date"])].append(r["forecast"])
        tr = r.get("trace") or {}
        par_search[r["harness"]].append(len(tr.get("searches") or []))
    return seq, par, seq_search, par_search


def slope(pts):
    xs = [a for a, _ in pts]
    ys = [b for _, b in pts]
    if len(xs) < 3 or pstdev(xs) < 1e-9:
        return None
    mx, my = mean(xs), mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None


def main():
    seq, par, seq_search, par_search = load()
    harnesses = sorted({h for h, _, _ in seq} | {h for h, _, _ in par})
    if not harnesses:
        raise SystemExit("no data in %s yet" % OUT)
    samples = sorted({s for _, _, s in seq})

    M = {"harnesses": harnesses, "questions": QIDS, "per": {}}
    for h in harnesses:
        e = {"gap": [], "lag0": [], "lag1": [], "gain_pts": [], "pit": [],
             "stab": {}, "brier_seq": {}, "brier_par": {}, "tot_move": [],
             "coh_seq": {}, "coh_par": {}, "probe": {},
             "search_seq": mean(seq_search[h]) if seq_search[h] else 0.0,
             "search_par": mean(par_search[h]) if par_search[h] else 0.0}

        # ---- per-question dynamics, D1..D5 only
        for q in QIDS:
            dates = TRAJECTORY[q]
            pmean = {d: (mean(par[(h, q, d)]) if par.get((h, q, d)) else None) for d in dates}
            for s in samples:
                tr = seq.get((h, q, s))
                if not tr:
                    continue
                path = [tr[d] for d in dates if d in tr]
                if len(path) > 1:
                    e["tot_move"].append(sum(abs(path[i + 1] - path[i])
                                             for i in range(len(path) - 1)))
                for i, d in enumerate(dates):
                    if d not in tr:
                        continue
                    draws = par.get((h, q, d)) or []
                    if draws:
                        below = sum(1 for x in draws if x < tr[d])
                        ties = sum(1 for x in draws if x == tr[d])
                        e["pit"].append(100.0 * (below + 0.5 * ties) / len(draws))
                    if i >= 1 and pmean[d] is not None:
                        e["gap"].append(tr[d] - pmean[d])
                        e["lag0"].append(abs(tr[d] - pmean[d]))
                        if pmean[dates[i - 1]] is not None:
                            e["lag1"].append(abs(tr[d] - pmean[dates[i - 1]]))
                            if dates[i - 1] in tr:
                                e["gain_pts"].append(
                                    (pmean[d] - pmean[dates[i - 1]], tr[d] - tr[dates[i - 1]]))

        # ---- stabilisation and Brier, indexed by trajectory step (dates differ per question)
        for step in range(5):
            sds_s, sds_p, bs, bp = [], [], [], []
            for q in QIDS:
                d = TRAJECTORY[q][step]
                vals = [seq[(h, q, s)][d] for s in samples
                        if (h, q, s) in seq and d in seq[(h, q, s)]]
                if len(vals) > 1:
                    sds_s.append(pstdev(vals))
                draws = par.get((h, q, d)) or []
                if len(draws) > 1:
                    sds_p.append(pstdev(draws))
                bs += [(v - TRUTH[q]) ** 2 for v in vals]
                bp += [(v - TRUTH[q]) ** 2 for v in draws]
            key = "D%d" % (step + 1)
            if sds_s and sds_p and mean(sds_p) > 1e-9:
                e["stab"][key] = mean(sds_s) / mean(sds_p)
            if bs:
                e["brier_seq"][key] = mean(bs)
            if bp:
                e["brier_par"][key] = mean(bp)

        # ---- coherence per group: the pair must sum to <=1, and ->1 late
        for gname, (qa, qb) in GROUPS.items():
            cs, cp = {}, {}
            for step in range(5):
                da, db = TRAJECTORY[qa][step], TRAJECTORY[qb][step]
                pair = []
                for s in samples:
                    a = seq.get((h, qa, s), {}).get(da)
                    b = seq.get((h, qb, s), {}).get(db)
                    if a is not None and b is not None:
                        pair.append(a + b)
                if pair:
                    cs["D%d" % (step + 1)] = mean(pair)
                pa, pb = par.get((h, qa, da)) or [], par.get((h, qb, db)) or []
                if pa and pb:
                    cp["D%d" % (step + 1)] = mean(pa) + mean(pb)
            e["coh_seq"][gname] = cs
            e["coh_par"][gname] = cp

        # ---- the outcome probe (D6): the answer is in the corpus by now
        for q in QIDS:
            d = PROBE_DATE[q]
            sv = [seq[(h, q, s)][d] for s in samples
                  if (h, q, s) in seq and d in seq[(h, q, s)]]
            pv = par.get((h, q, d)) or []
            last = TRAJECTORY[q][-1]
            prev = [seq[(h, q, s)][last] for s in samples
                    if (h, q, s) in seq and last in seq[(h, q, s)]]
            e["probe"][q] = {
                "truth": TRUTH[q],
                "seq_mean": mean(sv) if sv else None,
                "par_mean": mean(pv) if pv else None,
                "seq_prev_mean": mean(prev) if prev else None,
                "move_to_truth": (mean(sv) - mean(prev)) * (1 if TRUTH[q] else -1)
                                 if sv and prev else None,
                "abs_err_seq": abs(mean(sv) - TRUTH[q]) if sv else None,
            }
        M["per"][h] = e

    (OUT / "metrics_v2.json").write_text(json.dumps(M, indent=1), encoding="utf-8")

    # ---------------------------------------------------------------- report
    print("E1 v2  --  %d harnesses, %d questions, %d rollouts\n"
          % (len(harnesses), len(QIDS), len(samples)))
    print("%-16s %8s %7s %7s %7s %6s %8s %10s"
          % ("harness", "|D|tot", "gap", "|gap|", "gain", "PIT", "lag adv", "srch s/p"))
    for h in harnesses:
        e = M["per"][h]
        g = slope(e["gain_pts"])
        print("%-16s %8.3f %+7.3f %7.3f %7s %6.0f %+8.3f %5.1f/%.1f"
              % (h, mean(e["tot_move"]) if e["tot_move"] else float("nan"),
                 mean(e["gap"]) if e["gap"] else float("nan"),
                 mean(e["lag0"]) if e["lag0"] else float("nan"),
                 "--" if g is None else "%.2f" % g,
                 sorted(e["pit"])[len(e["pit"]) // 2] if e["pit"] else float("nan"),
                 (mean(e["lag0"]) - mean(e["lag1"])) if e["lag1"] else float("nan"),
                 e["search_seq"], e["search_par"]))

    print("\ncoherence at D5, per group (target ~1, and <=1 always):")
    print("%-16s %s" % ("harness", "  ".join("%-10s" % g for g in GROUPS)))
    for h in harnesses:
        row = ["%-10s" % ("%.2f" % M["per"][h]["coh_seq"][g].get("D5", float("nan")))
               for g in GROUPS]
        print("%-16s %s" % (h, "  ".join(row)))

    print("\nOUTCOME PROBE (D6 = R+2, the answer is in the corpus):")
    print("%-16s %9s %9s %9s" % ("harness", "prev->D6", "|err| D6", "n moved right"))
    for h in harnesses:
        pr = M["per"][h]["probe"]
        mv = [v["move_to_truth"] for v in pr.values() if v["move_to_truth"] is not None]
        er = [v["abs_err_seq"] for v in pr.values() if v["abs_err_seq"] is not None]
        print("%-16s %+9.3f %9.3f %6d/%d"
              % (h, mean(mv) if mv else float("nan"),
                 mean(er) if er else float("nan"),
                 sum(1 for x in mv if x > 0.01), len(mv)))
    print("\nwrote", OUT / "metrics_v2.json")


if __name__ == "__main__":
    main()
