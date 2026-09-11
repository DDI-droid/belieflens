"""Progress monitor for a run_real_e1 job.

  python scripts/watch_run.py                 # one snapshot
  python scripts/watch_run.py --follow         # refresh until the run ends

Reads the runner's own log rather than instrumenting it, so it can be attached
to an already-running job.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

LOG = Path("e1_v2.log")
OUT = Path("results/exp1_v2")
N_SEQ, N_PAR = 120, 720
BUDGET = 32000
EST_TOTAL_CALLS = 25500   # 120 seq chains + 720 par singles, measured mix


def bar(done, total, width=34):
    if total <= 0:
        return " " * width
    f = min(1.0, done / total)
    n = int(f * width)
    return "#" * n + "-" * (width - n)


def snapshot():
    txt = LOG.read_text(encoding="utf-8", errors="replace") if LOG.exists() else ""
    lines = txt.splitlines()
    seq_done = sum(1 for l in lines if re.match(r"\s*\[\s*\d+/\s*\d+\]", l))
    par = [l for l in lines if "latest" in l]
    par_done = 0
    if par:
        m = re.search(r"\[\s*(\d+)/\s*(\d+)\]", par[-1])
        if m:
            par_done = int(m.group(1))
    in_par = "PAR:" in txt
    if in_par:
        seq_done = N_SEQ
    calls = 0
    for l in reversed(lines):
        m = re.search(r"calls=(\d+)", l)
        if m:
            calls = int(m.group(1))
            break
    elapsed = 0.0
    for l in reversed(lines):
        m = re.search(r"\((\d+)s,", l)
        if m:
            elapsed = float(m.group(1))
            break
    units_done = seq_done + par_done
    units_tot = N_SEQ + N_PAR
    rate = calls / elapsed if elapsed > 0 else 0.0
    # cost model: measured 1978 in / 452 out tokens per call, $0.25 / $2.00 per 1M
    cost = calls * (1978 * 0.25 + 452 * 2.00) / 1e6
    # ETA on CALLS, not on units: a 6-date sequential chain and a 1-date
    # parallel single are not the same amount of work.
    eta = ""
    if calls and rate > 0:
        remaining = max(0.0, EST_TOTAL_CALLS - calls)
        eta = "~%.0f min left" % (remaining / rate / 60)
    return dict(seq=seq_done, par=par_done, units=units_done, tot=units_tot,
                calls=calls, elapsed=elapsed, rate=rate, cost=cost, eta=eta,
                done=("EXIT" in txt or units_done >= units_tot))


def render(s):
    return (
        "\rSEQ [%s] %3d/%d   PAR [%s] %3d/%d   %s calls (%.0f%% of cap)  "
        "$%.2f  %.0f/min  %s" %
        (bar(s["seq"], N_SEQ, 22), s["seq"], N_SEQ,
         bar(s["par"], N_PAR, 22), s["par"], N_PAR,
         "{:,}".format(s["calls"]), 100.0 * s["calls"] / BUDGET,
         s["cost"], s["rate"] * 60, s["eta"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--every", type=int, default=60)
    a = ap.parse_args()
    while True:
        s = snapshot()
        sys.stdout.write(render(s))
        sys.stdout.flush()
        if not a.follow or s["done"]:
            print()
            return
        time.sleep(a.every)


if __name__ == "__main__":
    main()
