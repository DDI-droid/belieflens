"""Experiment 1: four harnesses, sequential replay, then program recovery.

E1  each harness forecasts the same question on K dates in order, carrying its
    own prior forecasts (and, where the harness has one, its memory / belief
    state) forward. Identical date-gated corpus for all four.

E2  for every forecast produced, recover the variables its reasoning used, then
    synthesise ONE literal-free program per (harness, question) that reproduces
    every forecast from those variables alone.

  python scripts/run_exp1.py --stage e1 --samples 2
  python scripts/run_exp1.py --stage e2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from belieflens import extract as X                                  # noqa: E402
from belieflens.evidence import NewsIndex                            # noqa: E402
from belieflens.harnesses import (HARNESSES, HarnessRunner, Run,     # noqa: E402
                                  Turn, trajectory_text)


@dataclass
class Q:
    id: str
    text: str
    background: str = ""
    resolution_date: str = ""
    truth: int = 1          # 1 = the event happened, 0 = it did not
    dates: list = field(default_factory=list)


# Binarised from OpenForesight aljazeera2026Q1.  The two hockey questions share
# one evidence stream and resolve in opposite directions, which makes belief
# movement interpretable: under the same news, one belief should climb and the
# other should fall. A harness that moves them the same way is not tracking
# evidence.
QUESTIONS = [
    Q(id="hockey_usa",
      text=("Will the United States win the men's ice hockey gold medal at the "
            "Milano Cortina 2026 Winter Olympics?"),
      background=("The Milano Cortina 2026 Winter Olympics men's ice hockey "
                  "tournament concludes with the gold medal game on 22 February 2026. "
                  "NHL players are participating."),
      resolution_date="2026-02-22", truth=1,
      dates=["2026-01-14", "2026-01-28", "2026-02-08", "2026-02-15", "2026-02-20"]),
    Q(id="hockey_can",
      text=("Will Canada win the men's ice hockey gold medal at the Milano "
            "Cortina 2026 Winter Olympics?"),
      background=("The Milano Cortina 2026 Winter Olympics men's ice hockey "
                  "tournament concludes with the gold medal game on 22 February 2026. "
                  "NHL players are participating."),
      resolution_date="2026-02-22", truth=0,
      dates=["2026-01-14", "2026-01-28", "2026-02-08", "2026-02-15", "2026-02-20"]),
    # Scale-up set (experiment 8): four more OpenForesight aljazeera2026Q1
    # questions with >=5-week spans inside the corpus window, binarised on the
    # dataset's recorded answers. All four resolve YES -- disclosed: the hockey
    # pair above remains the only directional control; these power the
    # program-recovery statistics, not directional claims.
    Q(id="iqair_loni",
      text=("Will IQAir's annual world air quality report covering 2025 name "
            "Loni as the world's most polluted city?"),
      background=("IQAir publishes an annual World Air Quality Report ranking "
                  "cities by PM2.5 concentration; the report covering calendar "
                  "2025 is expected by late March 2026."),
      resolution_date="2026-03-23", truth=1,
      dates=["2026-01-10", "2026-01-31", "2026-02-20", "2026-03-08", "2026-03-20"]),
    Q(id="carrick_newcastle",
      text=("Will Newcastle United be the first club to defeat Manchester "
            "United under interim boss Michael Carrick?"),
      background=("Manchester United are playing under interim boss Michael "
                  "Carrick. The question resolves on the first competitive "
                  "defeat of his tenure, expected by mid-March 2026."),
      resolution_date="2026-03-03", truth=1,
      dates=["2026-01-27", "2026-02-05", "2026-02-14", "2026-02-23", "2026-03-01"]),
    Q(id="nepal_shah",
      text="Will Balendra Shah be sworn in as Nepal's prime minister on 27 March 2026?",
      background=("Nepal's political process is expected to produce a "
                  "prime-ministerial swearing-in in late March 2026."),
      resolution_date="2026-03-27", truth=1,
      dates=["2026-02-01", "2026-02-20", "2026-03-05", "2026-03-15", "2026-03-25"]),
    Q(id="sa_captain_maharaj",
      text=("Will Keshav Maharaj be named captain of South Africa's squad for "
            "the five-match men's T20 tour of New Zealand in March 2026?"),
      background=("Cricket South Africa is due to announce its squad for a "
                  "five-match T20 tour of New Zealand in March 2026."),
      resolution_date="2026-02-19", truth=1,
      dates=["2026-01-15", "2026-01-24", "2026-02-02", "2026-02-10", "2026-02-17"]),
]


def client():
    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("export OPENAI_API_KEY first")
    return OpenAI(timeout=240.0, max_retries=3)


def _dump_config(out: Path, name: str, args, extra: dict | None = None) -> None:
    """The reproducibility record the outputs were missing: every flag, model
    id, tolerance and timestamp, written next to the data it produced."""
    d = dict(vars(args))
    d["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if extra:
        d.update(extra)
    (out / name).write_text(json.dumps(d, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ E1

def stage_e1(args) -> None:
    ix = NewsIndex()
    print("corpus span %s .. %s (%d articles)" % ix.span())
    runner = HarnessRunner(client(), args.model, ix,
                           max_tool_rounds=args.tool_rounds,
                           reasoning_effort=args.effort or None)
    qs = [q for q in QUESTIONS if not args.questions or q.id in args.questions.split(",")]
    jobs = [(h, q, s) for q in qs for h in HARNESSES for s in range(args.samples)]
    print("%d runs = %d questions x %d harnesses x %d samples, %d dates each"
          % (len(jobs), len(qs), len(HARNESSES), args.samples, len(qs[0].dates)))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    _dump_config(out, "config_e1.json", args, {"corpus_span": list(ix.span()),
                                               "n_jobs": len(jobs)})
    n_done, t0 = 0, time.time()

    # Stream each run to disk as it completes (a crash at 15/16 must not lose
    # 90 minutes of paid calls), into a .tmp that is renamed only when the
    # stage is complete -- so no downstream reader can see a partial file.
    tmp = out / "e1_runs.jsonl.tmp"
    with open(tmp, "w", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(runner.run, h, q, q.dates, s): (h, q.id, s) for h, q, s in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            h, qid, s = futs[f]
            try:
                r = f.result()
                fc = ["--" if x is None else "%.2f" % x for x in r.forecasts()]
                print("  [%2d/%2d] %-10s %-11s s%d  %s  (%.0fs)"
                      % (i, len(jobs), h, qid, s, " ".join(fc), time.time() - t0), flush=True)
                fh.write(json.dumps(r.to_dict()) + "\n")
                fh.flush()
                n_done += 1
            except Exception as e:                              # noqa: BLE001
                print("  [%2d/%2d] %-10s %-11s s%d  FAILED %s: %s"
                      % (i, len(jobs), h, qid, s, type(e).__name__, str(e)[:160]), flush=True)

    p = out / "e1_runs.jsonl"
    if p.exists():
        p.unlink()
    tmp.rename(p)
    if n_done < len(jobs):
        print("WARNING: only %d/%d runs succeeded -- the jsonl is PARTIAL" % (n_done, len(jobs)))
    print("\nwrote %s (%d runs, %.0fs total)" % (p, n_done, time.time() - t0))


# ------------------------------------------------------------------ E2

def load_runs(path: Path) -> list:
    runs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            r = Run(harness=d["harness"], question_id=d["question_id"], sample=d["sample"])
            r.turns = [Turn(date=t["date"], forecast=t["forecast"], text=t["text"],
                            tool_calls=t.get("tool_calls", []), carry=t.get("carry", ""),
                            error=t.get("error", ""),
                            forced_stop=t.get("forced_stop", False)) for t in d["turns"]]
            runs.append(r)
    return runs


def stage_e2(args) -> None:
    cl = client()
    out = Path(args.out)
    runs = load_runs(out / "e1_runs.jsonl")
    print("loaded %d runs" % len(runs))
    _dump_config(out, "config_e2.json", args)

    groups: dict = {}
    for r in runs:
        groups.setdefault((r.harness, r.question_id), []).append(r)

    report: dict = {}
    report_path = out / "e2_programs.json"
    for (h, qid), rs in sorted(groups.items()):
        print("\n=== %s / %s ===" % (h, qid))
        # one group's failure must not discard every other group's paid work
        try:
            entry = _extract_group(cl, args, h, qid, rs)
        except Exception as e:                          # noqa: BLE001
            print("  GROUP FAILED: %s: %s" % (type(e).__name__, str(e)[:300]))
            entry = {"error": "%s: %s" % (type(e).__name__, str(e)[:300])}
        report["%s/%s" % (h, qid)] = entry
        # incremental write: a crash in a later group loses nothing
        report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print("\nwrote %s (%d groups)" % (report_path, len(report)))


def _extract_group(cl, args, h: str, qid: str, rs: list) -> dict:
    from belieflens.progdsl import fit_error

    rs = sorted(rs, key=lambda r: r.sample)   # deterministic excerpt choice

    # previous forecast per (sample, date): the legitimate anchor variable,
    # exempt from the near-target drop in step 1
    prev_map: dict = {}
    for r in rs:
        prev = None
        for t in sorted(r.turns, key=lambda t: t.date):
            prev_map[(r.sample, t.date)] = prev
            if t.forecast is not None:
                prev = t.forecast

    tasks = [(r, t) for r in rs for t in r.turns if t.forecast is not None and not t.error]
    if len(tasks) < 3:
        print("  too few usable forecasts (%d) -- skipping" % len(tasks))
        return {"skipped": "only %d usable forecasts" % len(tasks)}

    def one(rt):
        r, t = rt
        searches = "\n".join("  %s -> %s" % (c["query"], "; ".join(c["titles"][:3]))
                             for c in t.tool_calls)
        b = X.step1_variables(cl, args.extractor, t.text, searches, t.forecast,
                              prev_target=prev_map.get((r.sample, t.date)))
        return X.Instance(harness=h, question_id=qid, sample=r.sample, date=t.date,
                          target=t.forecast, bindings=b, raw=t.text)

    instances: list = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for inst in ex.map(one, tasks):
            instances.append(inst)
    instances.sort(key=lambda i: (i.sample, i.date))
    sizes = [len(i.bindings) for i in instances]
    print("  step 1: %d instances, %.1f variables each (min %d, max %d)"
          % (len(instances), sum(sizes) / len(sizes), min(sizes), max(sizes)))

    # step 2 -- one shared coordinate system; unreconciled instances are
    # excluded loudly, never zero-filled
    rec = X.step2_reconcile(cl, args.extractor, instances)
    if rec["excluded"]:
        print("  step 2: EXCLUDED %d instances (keys not reconciled): %s"
              % (len(rec["excluded"]), ", ".join(rec["excluded"])))
        instances = [i for i in instances if i.key() not in set(rec["excluded"])]
    print("  step 2: schema of %d -- %s" % (len(rec["schema"]), ", ".join(rec["schema"][:8])))
    if len(instances) < 3:
        return {"skipped": "only %d instances after reconciliation" % len(instances),
                "excluded": rec["excluded"]}

    # step 3 -- one program over every instance (descriptive fit)
    excerpt = trajectory_text(rs[0])[:6000]
    syn = X.step3_synthesise(cl, args.extractor, instances, rec["schema"],
                             tol=args.tol, max_attempts=args.attempts, excerpt=excerpt)
    f, d = syn.fit, syn.degeneracy
    if f:
        print("  step 3: attempts=%d  MAE=%.4f  within_tol=%.0f%%  ops=%d  "
              "branches=%d  max|corr|=%.2f (%s)%s"
              % (syn.attempts, f["mae"], 100 * f["within_tol"], d["n_ops"],
                 d.get("n_branches", 0), d["max_abs_corr"], d["max_corr_var"],
                 "  SUSPECT" if d["suspect"] else ""))
    else:
        print("  step 3: no valid program after %d attempts" % syn.attempts)

    # out-of-sample honesty check (reviewer: in-sample fit alone cannot
    # distinguish structure from memorisation): synthesise on the first
    # sample's instances only, evaluate untouched on the other sample's.
    oos = None
    samples = sorted({i.sample for i in instances})
    if len(samples) >= 2:
        train = [i for i in instances if i.sample == samples[0]]
        test = [i for i in instances if i.sample != samples[0]]
        if len(train) >= 3 and test:
            syn_o = X.step3_synthesise(cl, args.extractor, train, rec["schema"],
                                       tol=args.tol,
                                       max_attempts=max(2, args.attempts - 1),
                                       excerpt=excerpt)
            if syn_o.program:
                tf = fit_error(syn_o.program,
                               [(i.bindings, i.target) for i in test], tol=args.tol)
                oos = {"train_sample": samples[0], "train_n": len(train),
                       "test_n": len(test),
                       "train_mae": (syn_o.fit or {}).get("mae"),
                       "test_mae": tf["mae"], "test_within_tol": tf["within_tol"],
                       "test_failed": tf["n_failed"], "program": syn_o.program,
                       "degeneracy": syn_o.degeneracy}
                print("  OOS: train s%s (n=%d) -> test (n=%d)  test MAE=%s  within_tol=%s"
                      % (samples[0], len(train), len(test),
                         ("%.4f" % tf["mae"]) if tf["mae"] == tf["mae"] else "nan",
                         "%.0f%%" % (100 * tf["within_tol"])))

    return {
        "n_instances": len(instances),
        "schema": rec["schema"], "schema_notes": rec.get("notes", ""),
        "excluded": rec["excluded"],
        "program": syn.program,
        "fit": {k: v for k, v in (syn.fit or {}).items() if k != "residuals"},
        "residuals": (syn.fit or {}).get("residuals", []),
        "degeneracy": syn.degeneracy, "ok": syn.ok, "attempts": syn.attempts,
        "history": syn.history, "oos": oos,
        "instances": [{"key": i.key(), "date": i.date, "sample": i.sample,
                       "target": i.target, "bindings": i.bindings,
                       "raw_bindings": i.raw_bindings} for i in instances],
        "trajectory": X.variable_trajectory(instances),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="e1", choices=["e1", "e2", "all"])
    ap.add_argument("--model", default="gpt-5-mini",
                    help="harness model -- MUST have a pre-2026 knowledge cutoff; "
                         "the default matches the documented runs")
    ap.add_argument("--extractor", default="gpt-5.2",
                    help="step 1-3 model; contamination is irrelevant here")
    ap.add_argument("--effort", default="", help="reasoning_effort, if the model takes it")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--questions", default="")
    ap.add_argument("--tool-rounds", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--out", default="results/exp1")
    args = ap.parse_args()

    if args.stage in ("e1", "all"):
        stage_e1(args)
    if args.stage in ("e2", "all"):
        stage_e2(args)


if __name__ == "__main__":
    main()
