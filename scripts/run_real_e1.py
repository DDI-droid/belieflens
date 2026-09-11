"""E1 on the REAL-orchestration harnesses (analytica-full, blf-full, futuresim-full).

  python scripts/run_real_e1.py --smoke              # one day, one rollout each
  python scripts/run_real_e1.py --stage seq          # 5 rollouts x 5 dates
  python scripts/run_real_e1.py --stage par          # 5 dates x 5 fresh rollouts
  python scripts/run_real_e1.py --stage all

Per-call token usage is recorded; a hard global call budget aborts before
overspending. Output mirrors the shape of results/exp1 so the existing
anchoring/chart/report tooling reads it: results/exp1_real/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from belieflens.evidence import NewsIndex                              # noqa: E402
from belieflens.real_harnesses import (AnalyticaFull, BLFFull,          # noqa: E402
                                       FutureSimFull, ReactReal, LLM,
                                       BudgetExceeded)
from belieflens.harnesses import HarnessRunner                          # noqa: E402
from belieflens.questions_v2 import QUESTIONS, GROUPS                   # noqa: E402
from scripts.run_exp1 import client                                     # noqa: E402

OUT = Path("results/exp1_v2")
HARNESSES = ("analytica_full", "blf_full", "futuresim_full", "react")


def build(llm, ix, name, model="gpt-5-mini"):
    if name == "analytica_full":
        return AnalyticaFull(llm, ix)
    if name == "futuresim_full":
        return FutureSimFull(llm, ix)
    if name == "react":
        # ReAct IS its source system; ReactReal only slices it per date.
        return ReactReal(HarnessRunner(llm.client, model, ix, max_tool_rounds=6,
                                       max_results=6, reasoning_effort="low"), llm)
    return BLFFull(llm, ix)



def _day(hobj, h, q, d, state, history, dates):
    """Per-harness run_day signature. futuresim-full additionally needs the
    question id (its `df` is keyed by qid) and the update cadence, because its
    architecture lets the agent decide whether today is worth updating."""
    if h == "blf_full":
        return hobj.run_day(q.text, d, state, history, resolution=q.resolution_date)
    if h == "react":
        return hobj.run_day(q.text, d, state, history,
                            background=getattr(q, "background", ""),
                            resolution_date=q.resolution_date)
    if h == "futuresim_full":
        later = [x for x in dates if x > d]
        cadence = ("Current date: %s. %s Resolution date: %s."
                   % (d,
                      ("Next scheduled update: %s." % later[0]) if later
                      else "No later updates are scheduled.",
                      q.resolution_date))
        return hobj.run_day(q.text, d, state, history, qid=q.id,
                            resolution=getattr(q, "resolution", "") or q.text,
                            resolution_date=q.resolution_date, cadence=cadence)
    return hobj.run_day(q.text, d, state, history)


def run_chain(h, hobj, q, dates, rollout):
    """Sequential chain: state carried, history carried."""
    state, history, turns = None, [], []
    for d in dates:
        t0 = time.time()
        try:
            fc, state, trace = _day(hobj, h, q, d, state, history, dates)
            turns.append({"date": d, "forecast": fc, "trace": trace,
                          "secs": round(time.time() - t0, 1), "error": ""})
            history.append("  %s: %.3f" % (d, fc))
        except BudgetExceeded:
            raise
        except Exception as e:                                # noqa: BLE001
            turns.append({"date": d, "forecast": None, "trace": None,
                          "secs": round(time.time() - t0, 1),
                          "error": "%s: %s" % (type(e).__name__, str(e)[:200])})
    return {"mode": "seq", "harness": h, "question_id": q.id,
            "sample": rollout, "turns": turns}


def run_single(h, hobj, q, date, rollout):
    """Parallel condition: fresh, no state, no history."""
    t0 = time.time()
    try:
        fc, _, trace = _day(hobj, h, q, date, None, [], [date])
        err = ""
    except BudgetExceeded:
        raise
    except Exception as e:                                    # noqa: BLE001
        fc, trace, err = None, None, "%s: %s" % (type(e).__name__, str(e)[:200])
    return {"mode": "par", "harness": h, "question_id": q.id, "date": date,
            "rollout": rollout, "forecast": fc, "trace": trace,
            "secs": round(time.time() - t0, 1), "error": err}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["seq", "par", "all"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--budget", type=int, default=32000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--k2", type=int, default=5)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ix = NewsIndex()
    llm = LLM(client(), args.model, budget=args.budget)
    qs = list(QUESTIONS)
    t0 = time.time()

    if args.smoke:
        print("SMOKE: one day (%s), one rollout, both harnesses, question=%s"
              % (qs[0].dates[0], qs[0].id))
        for h in HARNESSES:
            hobj = build(llm, ix, h)
            r = run_single(h, hobj, qs[0], qs[0].dates[0], 0)
            fn = OUT / ("smoke_%s.json" % h)
            fn.write_text(json.dumps(r, indent=1), encoding="utf-8")
            print("  %-15s forecast=%s  %.0fs  usage=%s -> %s"
                  % (h, r["forecast"], r["secs"], llm.usage(), fn))
        return

    if args.stage in ("seq", "all"):
        jobs = [(h, q, s) for h in HARNESSES for q in qs for s in range(args.k)]
        print("SEQ: %d chains x %d dates" % (len(jobs), len(qs[0].dates)))
        with open(OUT / "e1_runs.jsonl", "w", encoding="utf-8") as fh, \
                ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_chain, h, build(llm, ix, h), q, q.dates, s):
                    (h, q.id, s) for h, q, s in jobs}
            for i, f in enumerate(as_completed(futs), 1):
                h, qid, s = futs[f]
                try:
                    r = f.result()
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    fc = ["--" if t["forecast"] is None else "%.2f" % t["forecast"]
                          for t in r["turns"]]
                    print("  [%2d/%2d] %-15s %-11s s%d  %s  (%.0fs, calls=%d)"
                          % (i, len(jobs), h, qid, s, " ".join(fc),
                             time.time() - t0, llm.calls), flush=True)
                except BudgetExceeded as e:
                    print("BUDGET ABORT:", e)
                    break

    if args.stage in ("par", "all"):
        jobs = [(h, q, d, r) for h in HARNESSES for q in qs
                for d in q.dates for r in range(args.k2)]
        print("PAR: %d fresh single-day runs" % len(jobs))
        with open(OUT / "e1_independent.jsonl", "w", encoding="utf-8") as fh, \
                ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_single, h, build(llm, ix, h), q, d, r):
                    (h, q.id, d, r) for h, q, d, r in jobs}
            for i, f in enumerate(as_completed(futs), 1):
                h, qid, d, rr = futs[f]
                try:
                    r = f.result()
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    if i % 10 == 0 or i == len(jobs):
                        print("  [%3d/%3d] latest %-15s %-11s %s r%d -> %s "
                              "(%.0fs, calls=%d)"
                              % (i, len(jobs), h, qid, d[5:], rr,
                                 "--" if r["forecast"] is None else "%.2f" % r["forecast"],
                                 time.time() - t0, llm.calls), flush=True)
                except BudgetExceeded as e:
                    print("BUDGET ABORT:", e)
                    break

    (OUT / "usage.json").write_text(json.dumps(
        dict(llm.usage(), model=args.model, elapsed_s=round(time.time() - t0)),
        indent=1), encoding="utf-8")
    print("usage:", llm.usage())


if __name__ == "__main__":
    main()
