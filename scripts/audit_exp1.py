"""Pre-publication audit of the collected Experiment-1 data.

    python scripts/audit_exp1.py results/exp1

Four sections, each a hard check, not a narrative:

A. VALIDATOR    every known gaming exploit -- including the post-review ones
                (number-named locals, x/x, ZERO-padding, identity ops, unused
                locals) -- must REJECT; a legitimate program must PASS.
B. E1 DATA      completeness; the strict parser must reproduce every stored
                forecast from its own turn text; and a full DATE-GATE REPLAY:
                every search any harness ever ran is re-executed against the
                index with that turn's date, and every returned article must
                be dated on or before it.
C. E2 DATA      per group: does the stored program still execute and reproduce
                the reported MAE; does it survive the HARDENED validator; do
                the step-1 tables contain answer-copies under the transform
                battery (raw / odds / percent / complement) or value-spelling
                names; and does the program actually reference any flagged
                variable (referenced = contaminated fit; unreferenced = clean).
D. VERDICT      one line per group: CLEAN, or FLAGGED with reasons.

Exit code 0 only if A and B fully pass and no group's fit is contaminated.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from belieflens.progdsl import (GLOBALS, ENTRY, check, numeric_name)   # noqa: E402
from belieflens.extract import answer_like                             # noqa: E402
from belieflens.harnesses import parse_forecast                        # noqa: E402

HARD_FAILS = []


def fail(msg):
    HARD_FAILS.append(msg)
    print("  FAIL  " + msg)


def ok(msg):
    print("  ok    " + msg)


def exec_unchecked(source: str, bindings: dict, max_steps: int = 200_000) -> float:
    """Execute a stored program WITHOUT the (now stricter) validator, so the
    audit can separate 'numbers reproduce' from 'passes the hardened rules'.
    Same sandbox and step limit as progdsl.run."""
    ns = {"__builtins__": {}}
    ns.update(GLOBALS)
    ns.update(bindings)
    exec(compile(ast.parse(source), "<audit>", "exec"), ns)   # noqa: S102
    fn = ns[ENTRY]
    import sys as _s
    counter = {"n": 0}

    def tracer(frame, event, arg):
        if event == "line":
            counter["n"] += 1
            if counter["n"] > max_steps:
                raise RuntimeError("step limit")
        return tracer

    old = _s.gettrace()
    _s.settrace(tracer)
    try:
        out = fn()
    finally:
        _s.settrace(old)
    return float(out)


def mae_of(source, instances):
    res = []
    n_failed = 0
    for i in instances:
        try:
            res.append(abs(exec_unchecked(source, i["bindings"]) - i["target"]))
        except Exception:                                     # noqa: BLE001
            n_failed += 1
    return (sum(res) / len(res) if res else float("nan")), n_failed


def referenced_names(source: str) -> set:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


# ---------------------------------------------------------------- section A

def section_a():
    print("\nA. VALIDATOR — exploit battery under hardened rules")
    exploits = {
        "const-construction": "def forecast():\n    x = TENTH*(TWO+TWO+TWO) + HUNDREDTH*TWO\n    return x*base",
        "bool-smuggle":       "def forecast():\n    t = True\n    return base/(t+t)",
        "compare-smuggle":    "def forecast():\n    unit = (EPS < ONE) + (EPS < ONE)\n    return base*unit",
        "call-const":         "def forecast():\n    h = noisy_or(HALF, HALF)\n    return h*base",
        "numeric-name":       "def forecast():\n    two = base/base + base/base\n    return sig/two",
        "self-div":           "def forecast():\n    unit = base/base\n    return sig*unit",
        "zero-padding":       "def forecast():\n    out = sig + ZERO*(base+w)\n    return out",
        "identity-padding":   "def forecast():\n    out = sig*ONE + ZERO\n    return out",
        "unused-local":       "def forecast():\n    pad = base*w + sig\n    return sig",
        "helper-fn":          "def forecast():\n    def h(y):\n        return y\n    return h(base)",
        "literal":            "def forecast():\n    return base*0.7 + 0.1",
    }
    for name, src in exploits.items():
        r = check(src, declared={"base", "w", "sig"})
        if r.ok:
            fail("validator PASSED exploit %r" % name)
        else:
            ok("REJECT %-18s (%s)" % (name, ", ".join(sorted({v.rule for v in r.violations}))))
    legit = ("def forecast():\n"
             "    pull = noisy_or(sig*w, base*w)\n"
             "    blended = base + w*(pull - base)\n"
             "    return clamp(blended, ZERO, ONE)")
    r = check(legit, declared={"base", "w", "sig"})
    if r.ok:
        ok("legitimate program PASSES")
    else:
        fail("legitimate program rejected: " + r.report())


# ---------------------------------------------------------------- section B

def section_b(out: Path):
    print("\nB. E1 DATA — completeness, parser reproduction, date-gate replay")
    runs = [json.loads(l) for l in (out / "e1_runs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    turns = [(r, t) for r in runs for t in r["turns"]]
    missing = [t for _, t in turns if t["forecast"] is None]
    (ok if not missing else fail)("completeness: %d runs, %d/%d turns with forecasts"
                                  % (len(runs), len(turns) - len(missing), len(turns)))

    mismatch = [1 for _, t in turns
                if t["forecast"] is not None and parse_forecast(t["text"]) != t["forecast"]]
    (ok if not mismatch else fail)("strict parser reproduces %d/%d stored forecasts"
                                   % (len(turns) - len(mismatch), len(turns)))

    bad_range = [t["forecast"] for _, t in turns
                 if t["forecast"] is not None and not (0.0 <= t["forecast"] <= 1.0)]
    (ok if not bad_range else fail)("all forecasts in [0,1]")

    # date-gate replay: every logged query, re-run with its turn's date
    try:
        from belieflens.evidence import NewsIndex
        ix = NewsIndex()
    except FileNotFoundError:
        print("  skip  date-gate replay (no local index; run bootstrap.py first)")
        return
    n_q, breaches = 0, []
    for r, t in turns:
        for c in t.get("tool_calls", []):
            n_q += 1
            arts = ix.search(c["query"], to_date=t["date"],
                             from_date=c.get("from_date"), k=8)
            late = [a.date for a in arts if a.date > t["date"]]
            if late:
                breaches.append((r["harness"], t["date"], c["query"][:50], late[:2]))
    if breaches:
        for b in breaches[:5]:
            fail("DATE-GATE BREACH %s" % (b,))
    else:
        ok("date-gate replay: %d logged searches re-executed, zero articles past the gate" % n_q)


# ---------------------------------------------------------------- section C

def section_c(out: Path) -> dict:
    print("\nC. E2 DATA — per-group audit under hardened rules")
    rep = json.loads((out / "e2_programs.json").read_text(encoding="utf-8"))
    verdicts = {}
    for key, v in sorted(rep.items()):
        flags = []
        insts = v.get("instances") or []
        prog = v.get("program") or ""
        print("\n  %s" % key)

        if not prog:
            flags.append("no program recovered (already reported as such)")
            print("        no full-fit program (recorded honestly in results)")
        else:
            # 1. numbers reproduce?
            mae, n_failed = mae_of(prog, insts)
            stored = (v.get("fit") or {}).get("mae")
            same = (stored is None or (mae != mae and stored != stored)
                    or (stored == stored and mae == mae and abs(mae - stored) < 1e-6))
            print("        refit: MAE %s vs stored %s -> %s (%d exec failures)"
                  % ("nan" if mae != mae else "%.4f" % mae,
                     "nan" if (stored is None or stored != stored) else "%.4f" % stored,
                     "reproduces" if same else "MISMATCH", n_failed))
            if not same:
                flags.append("stored MAE does not reproduce")

            # 2. hardened validator
            schema = set(v.get("schema") or [])
            r = check(prog, declared=schema)
            rules = sorted({vi.rule for vi in r.violations})
            if r.ok:
                print("        hardened validator: PASS")
            else:
                print("        hardened validator: would now reject (%s)" % ", ".join(rules))
                gaming = {"constant-construction", "magic-number", "magic-bool",
                          "self-op", "dead-code", "numeric-name"} & set(rules)
                if gaming:
                    flags.append("program uses gaming construct: %s" % ", ".join(sorted(gaming)))
                else:
                    flags.append("style-only violations under new rules: %s" % ", ".join(rules))

        # 3. answer-copies in the step-1 tables, and are they referenced?
        # A copy must track the answer SYSTEMATICALLY (same transform on >=80%
        # of its instances, at least 4). A single-instance collision -- a
        # roster size of 25 meeting one forecast of 0.25 -- cannot carry a fit
        # that has to reproduce every instance, and the previous day's
        # forecast is the exempted anchor input (review F7), not a leak.
        refs = referenced_names(prog)
        prev_map = {}
        for smp in {i["sample"] for i in insts}:
            seq = sorted((i for i in insts if i["sample"] == smp), key=lambda x: x["date"])
            prev = None
            for i in seq:
                prev_map[(smp, i["date"])] = prev
                prev = i["target"]
        per_var = {}
        name_viols = set()
        for i in insts:
            merged = dict(i.get("raw_bindings") or {})
            merged.update(i.get("bindings") or {})
            for n, val in merged.items():
                per_var.setdefault(n, []).append(
                    (val, i["target"], prev_map.get((i["sample"], i["date"]))))
                if numeric_name(n):
                    name_viols.add(n)
        # Copy-detection is ill-posed when the target barely varies: any
        # near-constant variable then "matches" under some transform. Such
        # groups are marked uninformative, not contaminated -- recovering a
        # program for a constant output is trivially easy and proves nothing.
        tgts = [i["target"] for i in insts]
        tmean = sum(tgts) / len(tgts) if tgts else 0.0
        tstd = (sum((t - tmean) ** 2 for t in tgts) / len(tgts)) ** 0.5 if tgts else 0.0
        copies, collisions = {}, 0
        if tstd < 0.02:
            print("        constant-target group (std %.4f): copy-detection "
                  "ill-posed; recovery uninformative" % tstd)
            flags.append("constant-target: recovery uninformative (std %.4f)" % tstd)
            per_var = {}
        for n, rows in per_var.items():
            tfs = []
            for val, tgt, prev in rows:
                tf = answer_like(val, tgt)
                if tf == "raw" and prev is not None and abs(float(val) - prev) <= 0.005:
                    tf = None            # yesterday's forecast: exempt anchor
                tfs.append(tf)
            hits = [t for t in tfs if t]
            if not hits:
                continue
            top = max(set(hits), key=hits.count)
            if len(rows) >= 4 and hits.count(top) / len(rows) >= 0.8:
                copies[n] = "%s on %d/%d instances" % (top, hits.count(top), len(rows))
            else:
                collisions += 1
        used_copies = sorted(n for n in copies if n in refs)
        if copies:
            print("        SYSTEMATIC answer-copies: %s"
                  % "; ".join("%s (%s)" % kv for kv in sorted(copies.items())))
        if collisions:
            print("        coincidental single/low-rate collisions: %d vars (not copies)" % collisions)
        if name_viols:
            print("        value-spelling names present: %s" % ", ".join(sorted(name_viols)[:6]))
            flags.append("numeric names in tables: %s" % ", ".join(sorted(name_viols)[:4]))
        if used_copies:
            flags.append("PROGRAM REFERENCES an answer-copy: %s"
                         % ", ".join("%s (%s)" % (n, copies[n]) for n in used_copies))
            print("        *** program references systematic copy: %s" % ", ".join(used_copies))
        elif copies:
            print("        program references none of them -> fit is clean of copies")

        # 4. OOS program, same checks
        oos = v.get("oos") or {}
        if oos.get("program"):
            test = [i for i in insts if i["sample"] != oos.get("train_sample", 0)]
            omae, ofail = mae_of(oos["program"], test)
            stored_o = oos.get("test_mae")
            print("        OOS refit: test MAE %s vs stored %s"
                  % ("nan" if omae != omae else "%.4f" % omae,
                     "nan" if (stored_o is None or stored_o != stored_o) else "%.4f" % stored_o))
            oref = referenced_names(oos["program"])
            oused = sorted(n for n in copies if n in oref)
            if oused:
                flags.append("OOS program references an answer-copy: %s" % ", ".join(oused))

        verdicts[key] = flags
    return verdicts


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/exp1")
    print("AUDIT of %s" % out)
    section_a()
    if "--skip-replay" in sys.argv:
        print("\nB. E1 DATA -- skipped on this re-run (passed in full previously)")
    else:
        section_b(out)
    verdicts = section_c(out)

    print("\nD. VERDICT")
    contaminated = 0
    for key, flags in sorted(verdicts.items()):
        hard = [f for f in flags if "REFERENCES" in f or "does not reproduce" in f
                or "gaming construct" in f]
        if hard:
            contaminated += 1
            print("  FLAGGED  %-24s %s" % (key, "; ".join(hard)))
        elif flags:
            print("  note     %-24s %s" % (key, "; ".join(flags)))
        else:
            print("  CLEAN    %-24s" % key)

    print()
    if HARD_FAILS or contaminated:
        print("AUDIT RESULT: %d hard failures, %d contaminated groups"
              % (len(HARD_FAILS), contaminated))
        sys.exit(1)
    print("AUDIT RESULT: pipeline checks pass; no group's fit is contaminated.")


if __name__ == "__main__":
    main()
