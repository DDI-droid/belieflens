"""One-command setup for a fresh clone.

    python bootstrap.py            # everything: deps, data (~400 MB), index, checks
    python bootstrap.py --no-data  # deps + checks only (offline dev against the mock)

What it does, in order:
  1. installs Python dependencies (requirements.txt)
  2. downloads the FutureSim news corpus slice, 2025-12..2026-03 (~412 MB parquet)
     and the OpenForesight question sets from Hugging Face -- public, no token
  3. builds the local BM25 search index (data/news.db, ~1.5 GB, ~2 min)
  4. runs the verification battery: date-gate check, program-space validator
     exploits (must all REJECT), forecast-parser cases, offline mock pipeline

After it passes, the only thing you need that this script cannot give you is
an OpenAI API key:

    export OPENAI_API_KEY=sk-...          # your own -- see HANDOFF.md
    python scripts/run_exp1.py --stage e1
    python scripts/run_exp1.py --stage e2
    python scripts/analyze_exp1.py results/exp1

Idempotent: re-running skips anything already present.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def step(msg: str) -> None:
    print("\n=== %s ===" % msg, flush=True)


def run(cmd: list, **kw) -> None:
    print("  $ " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT, **kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-data", action="store_true",
                    help="skip corpus download + index build (offline dev)")
    args = ap.parse_args()

    if sys.version_info < (3, 10):
        sys.exit("Python >= 3.10 required (found %s)" % sys.version.split()[0])

    step("1/4 dependencies")
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])

    if not args.no_data:
        step("2/4 corpus + questions (~412 MB, skipped if present)")
        if os.path.isdir(os.path.join(ROOT, "data", "corpus", "2026")):
            print("  data/corpus already present -- skipping download")
        else:
            run([sys.executable, "scripts/fetch_corpus.py",
                 "--start", "2025-12", "--end", "2026-03"])

        step("3/4 search index (~1.5 GB, ~2 min, skipped if present)")
        if os.path.exists(os.path.join(ROOT, "data", "news.db")):
            print("  data/news.db already present -- skipping build")
        else:
            run([sys.executable, "scripts/build_index.py"])
    else:
        step("2-3/4 SKIPPED (--no-data)")

    step("4/4 verification battery")
    code = r'''
import sys; sys.path.insert(0, %r)
failures = []

# --- program-space validator: the reviewer's exploit battery must all REJECT
from belieflens.progdsl import check, run
exploits = {
    "const-smuggle":   "def forecast():\n    x = TENTH*(TWO+TWO+TWO)\n    return x*base",
    "bool-smuggle":    "def forecast():\n    t = True\n    return base/(t+t)",
    "compare-smuggle": "def forecast():\n    one = (EPS < ONE) + (EPS < ONE)\n    return base*one",
    "helper-fn":       "def forecast():\n    def h(y):\n        return y\n    return h(base)",
}
for name, src in exploits.items():
    if check(src, declared={"base"}).ok:
        failures.append("validator let through: " + name)
legit = "def forecast():\n    p = base + w*(sig - base)\n    return clamp(p, ZERO, ONE)"
r = check(legit, declared={"base", "w", "sig"})
if not r.ok:
    failures.append("validator rejects a legitimate program: " + r.report())
else:
    v = run(legit, {"base": 0.2, "w": 0.5, "sig": 0.6})
    if abs(v - 0.4) > 1e-9:
        failures.append("executor wrong: %%r" %% v)

# --- forecast parser: strict cases
from belieflens.harnesses import parse_forecast
for text, want in [("FORECAST: 85%%", 0.85), ("FORECAST: 0,45", 0.45),
                   ("FORECAST: 0.62", 0.62), ("FORECAST: 85", None),
                   ("1. first\n2. second", None)]:
    if parse_forecast(text) != want:
        failures.append("parser: %%r -> %%r (want %%r)" %% (text, parse_forecast(text), want))

# --- date gate (only when the index exists)
import os
if os.path.exists(os.path.join(%r, "data", "news.db")):
    from belieflens.evidence import NewsIndex
    ix = NewsIndex(os.path.join(%r, "data", "news.db"))
    lo, hi, n = ix.span()
    print("  corpus: %%d articles, %%s .. %%s" %% (n, lo, hi))
    arts = ix.search("olympic hockey gold medal", to_date="2026-01-20", k=8)
    bad = [a.date for a in arts if a.date > "2026-01-20"]
    if bad:
        failures.append("DATE GATE BREACH: " + ", ".join(bad))
    if n < 300000:
        failures.append("corpus looks incomplete: %%d articles" %% n)
else:
    print("  (no index -- date-gate check skipped)")

if failures:
    print("\nVERIFICATION FAILED:")
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("  all checks passed")
''' % (ROOT, ROOT, ROOT)
    run([sys.executable, "-c", code])

    print("""
Setup complete.  Next:

  1. Read HANDOFF.md  -- state of the project, results so far, queued next steps
  2. export OPENAI_API_KEY=sk-...   (your own key; needs gpt-5-mini + gpt-5.2)
  3. python scripts/run_exp1.py --stage e1     # replay (or reuse results/exp1)
     python scripts/run_exp1.py --stage e2     # program recovery
     python scripts/analyze_exp1.py results/exp1

Offline (no key, no data): the mock pipeline exercises every metric
  python scripts/run_experiment.py --stage all --provider mock --k 3
  python scripts/analyze.py results/mock_mock-1
""")


if __name__ == "__main__":
    main()
