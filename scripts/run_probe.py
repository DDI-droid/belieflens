"""Experiment 7 -- the rigidity probe.

For each probed group we hold the harness's own recovered program as the
referee: perturb ONE variable in the last day's bindings by a stipulated,
evidence-implied amount, and the program yields the exact forecast change its
own reasoning structure demands.  Then the HARNESS itself gets the same
evidence -- continued from its real end-of-run state (its five forecasts and
its carry block) on a probe day, 2026-02-21 -- and we measure what it actually
does.

    rigidity = 1 - (actual change) / (implied change)

  ~0   it honours its own rule          >0  the conclusion resists its rule
  <0   it over-reacts                   n/a implied ~ 0 (rule says "immovable")

Controls: a PLACEBO item (irrelevant news, implied change = 0) per cell gives
the noise floor; search is disabled on the probe day so the injected item is
the only new information (stated to the harness as a tool outage) -- otherwise
real 02-21 corpus news would confound the targeted evidence.  Evidence items
name roles, never real people.

    python scripts/run_probe.py --out results/exp1
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from belieflens.harnesses import SYSTEMS, parse_forecast              # noqa: E402
from scripts.audit_exp1 import exec_unchecked                         # noqa: E402
from scripts.run_exp1 import QUESTIONS, client                        # noqa: E402

PROBE_DATE = "2026-02-21"

EVIDENCE = {
    "E_CAN_BAD": ("Canada's starting goaltender was injured during the team's "
                  "morning practice and has been ruled out of tomorrow's "
                  "gold-medal game against the United States; the backup "
                  "goaltender will start the final."),
    "E_USA_BAD": ("The United States' top-line centre and leading tournament "
                  "scorer has been ruled out of tomorrow's gold-medal game "
                  "against Canada after an injury in practice; a fourth-line "
                  "forward has been promoted into the lineup."),
    "E_PLACEBO": ("The IOC confirmed a record cumulative attendance across the "
                  "Milano Cortina curling venues, and organisers praised the "
                  "smooth running of ticketing operations throughout the Games."),
}

# group -> directional probes: (condition, evidence, target variable, change)
# "toward"/"against" are relative to the group's own question resolving YES.
# ("delta", x) shifts the variable; ("set", x) pins it. p_* variables clip to [0,1].
PROBES = {
    "futuresim/hockey_can": [
        ("against", "E_CAN_BAD", "adjustment_goaltending_net", ("delta", -0.10)),
        ("toward",  "E_USA_BAD", "adjustment_usa_rival_strength", ("delta", +0.06)),
    ],
    "bayesian/hockey_usa": [
        ("toward",  "E_CAN_BAD", "lr_opponent_canada_strength", ("set", 1.6)),
        ("against", "E_USA_BAD", "lr_injury_and_replacements", ("set", 0.55)),
    ],
    "analytica/hockey_usa": [
        ("toward",  "E_CAN_BAD", "p_usa_win_final_given_reach", ("delta", +0.20)),
        ("against", "E_USA_BAD", "p_usa_win_final_given_reach", ("delta", -0.20)),
    ],
    "react/hockey_can": [
        ("against", "E_CAN_BAD", "adj_injuries_availability", ("delta", -0.08)),
        ("toward",  "E_USA_BAD", "adj_rivals_strength", ("delta", +0.06)),
    ],
}


def perturb(bindings: dict, var: str, how) -> dict:
    b = dict(bindings)
    cur = float(b.get(var, 0.0))
    kind, x = how
    new = x if kind == "set" else cur + x
    if var.startswith("p_"):
        new = min(1.0, max(0.0, new))
    b[var] = new
    return b


def probe_turn(cl, model: str, harness: str, qid: str, history: list,
               carry: str, evidence: str) -> tuple:
    q = next(x for x in QUESTIONS if x.id == qid)
    system = SYSTEMS[harness].format(date=PROBE_DATE)
    parts = ["QUESTION: " + q.text]
    if q.background:
        parts.append("BACKGROUND: " + q.background)
    parts.append("RESOLVES: " + q.resolution_date)
    parts.append("YOUR PREVIOUS FORECASTS:\n" + "\n".join(history))
    if carry:
        tag = "MEMORY" if harness == "futuresim" else "BELIEF STATE"
        parts.append("YOUR %s FROM YESTERDAY:\n%s" % (tag, carry))
    parts.append("NOTE: the news search tool is unavailable today (outage). "
                 "The only new information available is today's verified news "
                 "digest below.")
    parts.append("TODAY'S VERIFIED NEWS DIGEST (%s):\n- %s" % (PROBE_DATE, evidence))
    parts.append("Today is %s. Produce your forecast for today." % PROBE_DATE)
    r = cl.chat.completions.create(
        model=model, reasoning_effort="medium",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": "\n\n".join(parts)}])
    text = r.choices[0].message.content or ""
    return parse_forecast(text), text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/exp1")
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    out = Path(args.out)

    rep = json.loads((out / "e2_programs.json").read_text(encoding="utf-8"))
    runs = [json.loads(l) for l in (out / "e1_runs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    cl = client()

    jobs = []
    for key, plist in PROBES.items():
        h, qid = key.split("/")
        v = rep[key]
        prog = v.get("program") or ""
        for smp in (0, 1):
            run = next((r for r in runs if r["harness"] == h
                        and r["question_id"] == qid and r["sample"] == smp), None)
            if run is None or not prog:
                continue
            turns = sorted(run["turns"], key=lambda t: t["date"])
            history = ["  %s: %.3f" % (t["date"], t["forecast"])
                       for t in turns if t["forecast"] is not None]
            carry = next((t["carry"] for t in reversed(turns) if t.get("carry")), "")
            last = next(i for i in v["instances"]
                        if i["sample"] == smp and i["date"] == "2026-02-20")
            b0 = last["bindings"]
            base = exec_unchecked(prog, b0)
            for cond, ev, var, how in plist:
                implied = exec_unchecked(prog, perturb(b0, var, how)) - base
                jobs.append(dict(key=key, harness=h, qid=qid, sample=smp,
                                 condition=cond, evidence=ev, var=var,
                                 p_last=last["target"], implied=implied))
            jobs.append(dict(key=key, harness=h, qid=qid, sample=smp,
                             condition="placebo", evidence="E_PLACEBO", var=None,
                             p_last=last["target"], implied=0.0))

    print("%d probe turns (search disabled, single injected item each)" % len(jobs))

    def run_one(j):
        h, qid, smp = j["harness"], j["qid"], j["sample"]
        run = next(r for r in runs if r["harness"] == h
                   and r["question_id"] == qid and r["sample"] == smp)
        turns = sorted(run["turns"], key=lambda t: t["date"])
        history = ["  %s: %.3f" % (t["date"], t["forecast"])
                   for t in turns if t["forecast"] is not None]
        carry = next((t["carry"] for t in reversed(turns) if t.get("carry")), "")
        for _ in range(2):
            p, text = probe_turn(cl, args.model, h, qid, history, carry,
                                 EVIDENCE[j["evidence"]])
            if p is not None:
                j["p_probe"], j["text"] = p, text
                return j
        j["p_probe"], j["text"] = None, text
        return j

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(run_one, jobs))

    print("\n%-22s %s %-9s %-30s %8s %8s %8s %9s"
          % ("group", "s", "cond", "target variable", "implied", "actual", "ratio", "rigidity"))
    print("-" * 108)
    for j in sorted(results, key=lambda x: (x["key"], x["sample"], x["condition"])):
        if j.get("p_probe") is None:
            print("%-22s %d %-9s %-30s   NO PARSE" % (j["key"], j["sample"],
                                                      j["condition"], j["var"] or "-"))
            continue
        act = j["p_probe"] - j["p_last"]
        imp = j["implied"]
        if abs(imp) >= 0.01:
            ratio = act / imp
            rig = 1.0 - ratio
            print("%-22s %d %-9s %-30s %+8.3f %+8.3f %8.2f %9.2f"
                  % (j["key"], j["sample"], j["condition"], (j["var"] or "-")[:30],
                     imp, act, ratio, rig))
        else:
            note = ("placebo floor" if j["condition"] == "placebo"
                    else "rule says immovable (implied~0)")
            print("%-22s %d %-9s %-30s %+8.3f %+8.3f      --   %s"
                  % (j["key"], j["sample"], j["condition"], (j["var"] or "-")[:30],
                     imp, act, note))

    p = out / "probe_results.json"
    p.write_text(json.dumps([{k: v for k, v in j.items() if k != "text"}
                             for j in results], indent=1), encoding="utf-8")
    (out / "probe_texts.jsonl").write_text(
        "\n".join(json.dumps(j) for j in results), encoding="utf-8")
    print("\nwrote %s" % p)


if __name__ == "__main__":
    main()
