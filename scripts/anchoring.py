"""Sequential vs independent: the anchoring measurement.

Two E1 conditions on identical evidence:
  sequential   the harness carries its own forecast history (and memory) --
               k full-trajectory rollouts               (e1_runs.jsonl)
  independent  fresh eyes per date, no history, no carry -- k2 rollouts per
               date, giving a per-date DISTRIBUTION      (e1_independent.jsonl)

What falls out:
  sanity      day-1 sequential IS an independent draw (no history yet), so it
              must sit inside the day-1 independent spread. If it doesn't,
              the comparison is broken and nothing below is trustworthy.
  anchor gap  seq_t minus the independent mean at t, dates 2..K. The
              independent runs say what the evidence alone supports today;
              the gap is what carrying your own past does to you.
  lag         is seq_t closer to the independent belief of TODAY or of the
              PREVIOUS date? A harness that lags is updating one date behind
              the evidence.
  crispness   sd of the independent draws per date -- how determinate the
              evidence-conditioned belief even is, per harness.

    python scripts/anchoring.py results/exp1
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def load(out: Path):
    seq = defaultdict(dict)     # (h,q,sample) -> {date: p}
    for line in (out / "e1_runs.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        for t in r["turns"]:
            if t["forecast"] is not None:
                seq[(r["harness"], r["question_id"], r["sample"])][t["date"]] = t["forecast"]
    ind = defaultdict(list)     # (h,q,date) -> [p,...]
    for line in (out / "e1_independent.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["forecast"] is not None:
            ind[(r["harness"], r["question_id"], r["date"])].append(r["forecast"])
    return seq, ind


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/exp1")
    seq, ind = load(out)
    hqs = sorted({(h, q) for h, q, _ in ind})
    report = {}

    print("=" * 96)
    print("SEQUENTIAL vs INDEPENDENT  (independent shown as mean +- sd [min..max], n draws)")
    print("=" * 96)
    for h, q in hqs:
        dates = sorted({d for (hh, qq, d) in ind if (hh, qq) == (h, q)})
        print("\n  %s / %s" % (h, q))
        print("    %-6s %-26s %s" % ("date", "independent (fresh eyes)", "sequential s0 / s1"))
        for d in dates:
            draws = ind[(h, q, d)]
            s0 = seq.get((h, q, 0), {}).get(d)
            s1 = seq.get((h, q, 1), {}).get(d)
            print("    %-6s %5.2f +-%.2f [%.2f..%.2f] n=%d    %s / %s"
                  % (d[5:], mean(draws), pstdev(draws) if len(draws) > 1 else 0.0,
                     min(draws), max(draws), len(draws),
                     "--" if s0 is None else "%.2f" % s0,
                     "--" if s1 is None else "%.2f" % s1))

    print("\n" + "=" * 96)
    print("METRICS per harness (pooled over questions and samples)")
    print("=" * 96)
    print("%-11s %9s %9s %9s %10s %9s %11s"
          % ("harness", "sane@d1", "gap", "|gap|", "lag0|lag1", "lags?", "crispness"))
    print("-" * 96)
    for h in sorted({h for h, _ in hqs}):
        sane_in = sane_tot = 0
        gaps, lag0, lag1, sds = [], [], [], []
        for hh, q in hqs:
            if hh != h:
                continue
            dates = sorted({d for (h2, q2, d) in ind if (h2, q2) == (h, q)})
            means = {d: mean(ind[(h, q, d)]) for d in dates}
            sds += [pstdev(ind[(h, q, d)]) for d in dates if len(ind[(h, q, d)]) > 1]
            for smp in (0, 1):
                tr = seq.get((h, q, smp), {})
                if dates and dates[0] in tr:
                    lo, hi = min(ind[(h, q, dates[0])]), max(ind[(h, q, dates[0])])
                    pad = 0.05
                    sane_tot += 1
                    sane_in += int(lo - pad <= tr[dates[0]] <= hi + pad)
                for i, d in enumerate(dates):
                    if d not in tr:
                        continue
                    if i >= 1:
                        gaps.append(tr[d] - means[d])
                        lag0.append(abs(tr[d] - means[d]))
                        lag1.append(abs(tr[d] - means[dates[i - 1]]))
        report[h] = {"sanity": "%d/%d" % (sane_in, sane_tot),
                     "gap_signed": mean(gaps) if gaps else None,
                     "gap_abs": mean(lag0) if lag0 else None,
                     "lag0": mean(lag0) if lag0 else None,
                     "lag1": mean(lag1) if lag1 else None,
                     "crispness_sd": mean(sds) if sds else None}
        r = report[h]
        print("%-11s %9s %+9.3f %9.3f %5.3f|%.3f %9s %11.3f"
              % (h, r["sanity"], r["gap_signed"], r["gap_abs"],
                 r["lag0"], r["lag1"],
                 "YES" if r["lag1"] < r["lag0"] else "no",
                 r["crispness_sd"]))
    print("""
  sane@d1    day-1 sequential inside the day-1 independent spread (+-0.05 pad)
  gap        signed seq - independent mean, dates 2..K: what history does to you
  lags?      YES = the sequential belief tracks YESTERDAY'S evidence-conditioned
             belief better than today's -- it updates one date behind
  crispness  mean per-date sd of the fresh-eyes draws (belief determinacy)""")

    (out / "anchoring.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote %s" % (out / "anchoring.json"))


if __name__ == "__main__":
    main()
