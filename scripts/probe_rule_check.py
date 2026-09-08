"""Probe addendum: multi-coordinate rule-compliance.

The probe's `implied` column stipulates a change to ONE variable, but real
evidence legitimately moves several coordinates at once, so ratio > 1 there is
ambiguous: over-reaction, or just a wider (still rule-following) update.

This closes the gap. For every probe turn, the extractor reads the probe-day
reasoning and produces updated values for the group's OWN schema (defaulting to
the last replay day's value where the reasoning left one untouched). Then

    p_rule = program(updated bindings)     -- what its structure implies given
                                              everything it said it saw
    gap    = p_actual - p_rule             -- live structure-compliance

A small |gap| means the harness still computes with its recovered structure
even under injected evidence; a large one means the probe knocked it off its
own rule.  Placebos included: their gap is the compliance noise floor.

    python scripts/probe_rule_check.py --out results/exp1
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_exp1 import exec_unchecked                        # noqa: E402
from scripts.run_exp1 import client                                  # noqa: E402

SYSTEM = """You are given: a schema of named quantities from a forecaster's
reasoning structure, yesterday's value for each, today's news, and the full
reasoning the forecaster wrote today.

Output today's value for EVERY schema name, as JSON: {"name": <number>, ...}.

- If today's reasoning gives or implies a new value for a quantity, use it.
- If today's reasoning does not touch a quantity, repeat yesterday's value.
- Do NOT include the final forecast itself anywhere. Values only, all names."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/exp1")
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)

    rep = json.loads((out / "e2_programs.json").read_text(encoding="utf-8"))
    probes = [json.loads(l) for l in
              (out / "probe_texts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    cl = client()

    def one(j):
        if j.get("p_probe") is None:
            return None
        v = rep[j["key"]]
        schema = v.get("schema") or []
        prog = v.get("program") or ""
        last = next(i for i in v["instances"]
                    if i["sample"] == j["sample"] and i["date"] == "2026-02-20")
        user = ("SCHEMA WITH YESTERDAY'S VALUES:\n"
                + json.dumps({n: last["bindings"].get(n, 0.0) for n in schema}, indent=1)
                + "\n\nTODAY'S NEWS:\n- " + j.get("evidence", "")
                + "\n\nTODAY'S REASONING BY THE FORECASTER:\n" + j.get("text", "")[:8000]
                + "\n\nOutput today's value for every schema name.")
        r = cl.chat.completions.create(
            model=args.model, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": user}])
        try:
            got = json.loads(r.choices[0].message.content or "{}")
            b = {n: float(got.get(n, last["bindings"].get(n, 0.0)) or 0.0) for n in schema}
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        try:
            p_rule = exec_unchecked(prog, b)
        except Exception:                                    # noqa: BLE001
            return None
        return dict(key=j["key"], sample=j["sample"], condition=j["condition"],
                    p_last=j["p_last"], p_probe=j["p_probe"], p_rule=p_rule,
                    gap=j["p_probe"] - p_rule,
                    rule_delta=p_rule - j["p_last"],
                    actual_delta=j["p_probe"] - j["p_last"],
                    bindings=b)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = [r for r in ex.map(one, probes) if r]

    print("%-22s %s %-9s %8s %8s %8s %8s" % ("group", "s", "cond",
          "Δrule", "Δactual", "p_rule", "gap"))
    print("-" * 78)
    for r in sorted(rows, key=lambda x: (x["key"], x["sample"], x["condition"])):
        print("%-22s %d %-9s %+8.3f %+8.3f %8.3f %+8.3f"
              % (r["key"], r["sample"], r["condition"], r["rule_delta"],
                 r["actual_delta"], r["p_rule"], r["gap"]))
    by_cond = {}
    for r in rows:
        by_cond.setdefault(r["condition"], []).append(abs(r["gap"]))
    print()
    for c, gs in sorted(by_cond.items()):
        print("  mean |gap| %-9s = %.3f  (n=%d)" % (c, sum(gs) / len(gs), len(gs)))

    (out / "probe_rule_check.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\nwrote %s" % (out / "probe_rule_check.json"))


if __name__ == "__main__":
    main()
