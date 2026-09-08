"""Belief-dynamics analyses over the recovered programs (items 1-4).

  python scripts/beliefdyn.py results/exp1                # 1, 2a, 3, 4 (free)
  python scripts/beliefdyn.py results/exp1 --temporal     # + 2b (extractor calls)

1   variable trajectories -- the belief, plotted in its discovered coordinates:
    which moved with evidence, which held fixed, which drifted.
2a  residual-over-time -- signed program-minus-actual per date: did the one
    fitted structure hold early and break late?
2b  temporal holdout -- synthesise a program on the first three dates only
    (both samples), run it untouched on the last two: did the EARLY theory
    predict the LATE forecasts?  This is accumulated theory-drift, measured
    against the harness's own recovered rule.
3   sensitivity -- numeric d(forecast)/d(variable) around each instance's
    bindings: which belief coordinate does each harness's conclusion hinge on.
4   program distance -- full-fit program vs the OOS (sample-0-trained) program,
    Monte-Carlo over the observed binding ranges: did two halves of the data
    yield the same theory?  (Cross-harness distance is NOT computed: schemas
    differ, so a function-space distance between them is undefined -- saying
    so beats faking it.)

Known systematic answer-copies (audit) are excluded from any NEW synthesis.
Everything is written to results/<out>/beliefdyn.json as well as printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_exp1 import exec_unchecked, referenced_names      # noqa: E402
from belieflens.extract import answer_like                           # noqa: E402

KNOWN_COPIES = {"posterior_odds_usa_gold"}   # audit: odds-form, 10/10 instances


def load(out: Path) -> dict:
    return json.loads((out / "e2_programs.json").read_text(encoding="utf-8"))


def runp(prog: str, bindings: dict):
    try:
        return exec_unchecked(prog, bindings)
    except Exception:                                    # noqa: BLE001
        return None


def systematic_copies(insts: list) -> set:
    """Same criterion as the audit: tracks the answer on >=80% of instances."""
    prev_map = {}
    for smp in {i["sample"] for i in insts}:
        seq = sorted((i for i in insts if i["sample"] == smp), key=lambda x: x["date"])
        prev = None
        for i in seq:
            prev_map[(smp, i["date"])] = prev
            prev = i["target"]
    per = {}
    for i in insts:
        for n, v in (i.get("bindings") or {}).items():
            per.setdefault(n, []).append((v, i["target"], prev_map.get((i["sample"], i["date"]))))
    out = set()
    for n, rows in per.items():
        hits = 0
        for v, t, prev in rows:
            tf = answer_like(v, t)
            if tf == "raw" and prev is not None and abs(float(v) - prev) <= 0.005:
                tf = None
            hits += bool(tf)
        if len(rows) >= 4 and hits / len(rows) >= 0.8:
            out.add(n)
    return out | KNOWN_COPIES


# ---------------------------------------------------------------- item 1

def trajectories(key: str, v: dict, rep_out: dict) -> None:
    insts = v.get("instances") or []
    schema = v.get("schema") or []
    if not insts:
        return
    dates = sorted({i["date"] for i in insts})
    samples = sorted({i["sample"] for i in insts})
    print("\n  %s" % key)
    hdr = "    %-46s " % "variable" + "  ".join(d[5:] for d in dates)
    print(hdr + "   (per date: mean over samples)")
    rows = {}
    for n in schema:
        vals_by_date = []
        for d in dates:
            vs = [i["bindings"].get(n) for i in insts if i["date"] == d and n in i["bindings"]]
            vs = [x for x in vs if x is not None]
            vals_by_date.append(sum(vs) / len(vs) if vs else None)
        rows[n] = vals_by_date
        rng = (max(x for x in vals_by_date if x is not None)
               - min(x for x in vals_by_date if x is not None)) if any(
                   x is not None for x in vals_by_date) else 0.0
        cells = "  ".join("  --" if x is None else "%5.2f" % x for x in vals_by_date)
        tag = ("  <- FIXED" if rng < 1e-9 else
               ("  <- moved %.2f" % rng if rng >= 0.15 else ""))
        print("    %-46s %s%s" % (n[:46], cells, tag))
    tg = []
    for d in dates:
        ts = [i["target"] for i in insts if i["date"] == d]
        tg.append(sum(ts) / len(ts))
    print("    %-46s %s   <- the forecast itself"
          % ("(actual forecast)", "  ".join("%5.2f" % x for x in tg)))
    rep_out.setdefault("trajectories", {})[key] = {
        "dates": dates, "samples": samples, "variables": rows, "forecast": tg}


# ---------------------------------------------------------------- item 2a

def residuals_over_time(key: str, v: dict, rep_out: dict) -> None:
    prog, insts = v.get("program") or "", v.get("instances") or []
    if not prog or not insts:
        return
    dates = sorted({i["date"] for i in insts})
    by_date = []
    for d in dates:
        rs = []
        for i in insts:
            if i["date"] != d:
                continue
            got = runp(prog, i["bindings"])
            if got is not None:
                rs.append(got - i["target"])
        by_date.append(sum(rs) / len(rs) if rs else None)
    cells = "  ".join("   --" if x is None else "%+5.2f" % x for x in by_date)
    drift = (None if (by_date[0] is None or by_date[-1] is None)
             else abs(by_date[-1]) - abs(by_date[0]))
    print("  %-24s %s%s" % (key, cells,
          "" if drift is None else "   |resid| drift first->last %+0.3f" % drift))
    rep_out.setdefault("residuals_over_time", {})[key] = {"dates": dates, "signed": by_date}


# ---------------------------------------------------------------- item 3

def sensitivity(key: str, v: dict, rep_out: dict) -> list:
    prog, insts = v.get("program") or "", v.get("instances") or []
    schema = v.get("schema") or []
    if not prog or not insts:
        return []
    sens = {}
    for n in schema:
        deltas = []
        for i in insts:
            b = i["bindings"]
            if n not in b:
                continue
            base = runp(prog, b)
            if base is None:
                continue
            v0 = float(b[n])
            step = max(0.02, abs(v0) * 0.10)
            hi = runp(prog, {**b, n: v0 + step})
            lo = runp(prog, {**b, n: v0 - step})
            if hi is not None and lo is not None:
                deltas.append(abs(hi - lo) / 2.0)   # forecast move per one-step nudge
        if deltas:
            sens[n] = sum(deltas) / len(deltas)
    ranked = sorted(sens.items(), key=lambda kv: -kv[1])
    print("\n  %s" % key)
    for n, s in ranked[:5]:
        print("    %-46s dF per nudge = %.3f" % (n[:46], s))
    dead = [n for n, s in sens.items() if s < 1e-6]
    if dead:
        print("    inert in the program: %s" % ", ".join(sorted(dead)[:5]))
    rep_out.setdefault("sensitivity", {})[key] = dict(ranked)
    return ranked


# ---------------------------------------------------------------- item 4

def program_distance(key: str, v: dict, rep_out: dict) -> None:
    import random
    prog = v.get("program") or ""
    oos = (v.get("oos") or {}).get("program") or ""
    insts = v.get("instances") or []
    schema = v.get("schema") or []
    if not prog or not oos or not insts:
        return
    lo = {n: min(i["bindings"].get(n, 0.0) for i in insts) for n in schema}
    hi = {n: max(i["bindings"].get(n, 0.0) for i in insts) for n in schema}
    rng = random.Random(7)
    diffs, fails = [], 0
    for _ in range(400):
        b = {n: (lo[n] if lo[n] == hi[n] else rng.uniform(lo[n], hi[n])) for n in schema}
        a, c = runp(prog, b), runp(oos, b)
        if a is None or c is None:
            fails += 1
            continue
        diffs.append(abs(a - c))
    obs = [abs((runp(prog, i["bindings"]) or 0) - (runp(oos, i["bindings"]) or 0))
           for i in insts]
    if diffs:
        mc = sum(diffs) / len(diffs)
        print("  %-24s d(full, oos-trained) = %.4f over binding box (%d ok/%d), "
              "%.4f on observed bindings"
              % (key, mc, len(diffs), 400, sum(obs) / len(obs)))
        rep_out.setdefault("program_distance", {})[key] = {
            "mc_mean_abs": mc, "mc_ok": len(diffs), "observed_mean_abs": sum(obs) / len(obs)}


# ---------------------------------------------------------------- item 2b

def temporal_holdout(out: Path, rep: dict, rep_out: dict, workers: int = 4) -> None:
    import os
    from openai import OpenAI
    from belieflens import extract as X
    if not os.environ.get("OPENAI_API_KEY"):
        print("  (no OPENAI_API_KEY -- skipping temporal holdout)")
        return
    cl = OpenAI(timeout=240.0, max_retries=2)
    model = "gpt-5.2"
    print("\n2b. TEMPORAL HOLDOUT -- early theory, late days (synthesised fresh, "
          "hardened rules, copies excluded)")
    for key, v in sorted(rep.items()):
        insts_raw = v.get("instances") or []
        schema = [n for n in (v.get("schema") or [])
                  if n not in systematic_copies(insts_raw)]
        if len(insts_raw) < 8 or not schema:
            continue
        dates = sorted({i["date"] for i in insts_raw})
        train_dates, test_dates = set(dates[:3]), set(dates[3:])
        mk = lambda i: X.Instance(harness=key.split("/")[0], question_id=key.split("/")[1],
                                  sample=i["sample"], date=i["date"], target=i["target"],
                                  bindings={n: i["bindings"].get(n, 0.0) for n in schema})
        train = [mk(i) for i in insts_raw if i["date"] in train_dates]
        test = [mk(i) for i in insts_raw if i["date"] in test_dates]
        if len(train) < 4 or not test:
            continue
        try:
            syn = X.step3_synthesise(cl, model, train, schema, tol=0.02,
                                     max_attempts=3)
        except Exception as e:                            # noqa: BLE001
            print("  %-24s synthesis failed: %s" % (key, str(e)[:120]))
            continue
        if not syn.program:
            print("  %-24s no early-theory program found" % key)
            continue
        from belieflens.progdsl import fit_error
        tr = fit_error(syn.program, [(i.bindings, i.target) for i in train], tol=0.02)
        te = fit_error(syn.program, [(i.bindings, i.target) for i in test], tol=0.02)
        print("  %-24s early-fit MAE %s -> late-days MAE %s  (drift %+0.3f)"
              % (key,
                 "nan" if tr["mae"] != tr["mae"] else "%.4f" % tr["mae"],
                 "nan" if te["mae"] != te["mae"] else "%.4f" % te["mae"],
                 (te["mae"] - tr["mae"]) if (tr["mae"] == tr["mae"] and te["mae"] == te["mae"]) else float("nan")))
        rep_out.setdefault("temporal_holdout", {})[key] = {
            "train_dates": sorted(train_dates), "test_dates": sorted(test_dates),
            "train_mae": tr["mae"], "test_mae": te["mae"],
            "test_within_tol": te["within_tol"], "program": syn.program}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="results/exp1")
    ap.add_argument("--temporal", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    rep = load(out)
    rep_out: dict = {}

    print("=" * 78)
    print("1. VARIABLE TRAJECTORIES -- the belief in its discovered coordinates")
    print("=" * 78)
    for key, v in sorted(rep.items()):
        if v.get("program"):
            trajectories(key, v, rep_out)

    print("\n" + "=" * 78)
    print("2a. RESIDUAL OVER TIME -- signed program-minus-actual per date")
    print("=" * 78)
    for key, v in sorted(rep.items()):
        residuals_over_time(key, v, rep_out)

    print("\n" + "=" * 78)
    print("3. SENSITIVITY -- which coordinate the conclusion hinges on")
    print("=" * 78)
    for key, v in sorted(rep.items()):
        sensitivity(key, v, rep_out)

    print("\n" + "=" * 78)
    print("4. PROGRAM DISTANCE -- full-fit vs sample-0-trained theory")
    print("   (cross-harness distance undefined: schemas differ; not computed)")
    print("=" * 78)
    for key, v in sorted(rep.items()):
        program_distance(key, v, rep_out)

    if args.temporal:
        temporal_holdout(out, rep, rep_out)

    p = out / "beliefdyn.json"
    p.write_text(json.dumps(rep_out, indent=1), encoding="utf-8")
    print("\nwrote %s" % p)


if __name__ == "__main__":
    main()
