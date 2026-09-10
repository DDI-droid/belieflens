"""Top up the sequential condition to k=5 rollouts (samples 2-4), hockey pair.
Appends Run rows (same schema as e1_runs.jsonl) to e1_runs_extra.jsonl."""
import json, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from belieflens.evidence import NewsIndex
from belieflens.harnesses import HARNESSES, HarnessRunner
from scripts.run_exp1 import QUESTIONS, client

qs = [q for q in QUESTIONS if q.id in ("hockey_usa", "hockey_can")]
runner = HarnessRunner(client(), "gpt-5-mini", NewsIndex(),
                       max_tool_rounds=5, reasoning_effort="medium")
jobs = [(h, q, s) for q in qs for h in HARNESSES for s in (2, 3, 4)]
print("%d top-up sequential runs (samples 2-4)" % len(jobs), flush=True)
out = Path("results/exp1/e1_runs_extra.jsonl")
t0 = time.time()
with open(out, "a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(runner.run, h, q, q.dates, s): (h, q.id, s) for h, q, s in jobs}
    for i, f in enumerate(as_completed(futs), 1):
        h, qid, s = futs[f]
        try:
            r = f.result()
            fh.write(json.dumps(r.to_dict()) + "\n"); fh.flush()
            fc = ["--" if x is None else "%.2f" % x for x in r.forecasts()]
            print("  [%2d/%2d] %-10s %-11s s%d  %s  (%.0fs)"
                  % (i, len(jobs), h, qid, s, " ".join(fc), time.time()-t0), flush=True)
        except Exception as e:
            print("  [%2d/%2d] FAILED %s %s s%d: %s" % (i, len(jobs), h, qid, s, str(e)[:120]), flush=True)
print("done", flush=True)
