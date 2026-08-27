"""Recover a program from a harness's own reasoning trace.

Step 1  read one turn's trajectory, write down every quantity the reasoning
        actually used, as `name = value`.  Per instance.
Step 2  reconcile those tables into one canonical schema -- the coordinate
        system, discovered rather than imposed.
Step 3  synthesise ONE literal-free program over that schema that reproduces
        the harness's forecast at every instance, repairing against validator
        violations and fit residuals until it does or the budget runs out.

THE DEGENERACY PROBLEM.  The no-literal rule stops the program hiding numbers
in its body, but nothing stops step 1 from declaring `final_answer = 0.62` and
step 2 writing `return final_answer`.  That fits perfectly and means nothing.
Three defences, all enforced here rather than requested politely:

  * step 1 is forbidden to record the output, and any extracted variable whose
    value equals the turn's forecast is dropped mechanically;
  * `degeneracy()` reports the best |correlation| between any single variable
    and the target across instances, and flags near-passthrough programs;
  * a program whose body is a bare `return <name>` is rejected outright.

A perfect fit with high degeneracy is a failed extraction, not a result.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .progdsl import SPEC_FOR_PROMPT, check, fit_error, run as run_program

# ------------------------------------------------------------------ step 1

STEP1_SYSTEM = """You are reverse-engineering a forecaster's reasoning.

Below is one day of a forecasting agent's trajectory: the searches it ran and
the reasoning it wrote. It ended with a probability.

Your job is to write down EVERY quantity that reasoning used, as a flat list of
name = value declarations. These are the inputs its reasoning operated on:
base rates, sub-probabilities it assigned, weights it gave things, adjustments
it applied, thresholds it compared against, counts it cited.

RULES
- One per line, exactly:   snake_case_name = <number>
- Numbers only. No expressions, no units, no strings, no comments on the line.
- Names must be descriptive of the QUANTITY, not of the date or the run.
  Good:  base_rate_no_cut, weight_inflation_signal, adjustment_for_hawkish_tone
  Bad:   x1, value_on_jan_15, v2
- If the reasoning implied a number without writing it ("this roughly halves
  the odds"), record it with the value it implies (0.5) and a name that says so.
- NEVER record the final forecast, the answer, or the posterior it reported.
  You are recording the INPUTS to the reasoning, never its OUTPUT.
- Aim for 5 to 15 variables. Prefer the quantities that did real work.

Output nothing but the declaration lines."""


@dataclass
class Instance:
    """One forecast the harness produced, plus what its reasoning ran on."""
    harness: str
    question_id: str
    sample: int
    date: str
    target: float
    bindings: dict = field(default_factory=dict)
    raw_bindings: dict = field(default_factory=dict)  # step-1 output, pre-reconcile audit trail
    raw: str = ""

    def key(self) -> str:
        return "%s|%s|s%d|%s" % (self.harness, self.question_id, self.sample, self.date)


_DECL = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$")


def parse_declarations(text: str) -> dict:
    out = {}
    for line in (text or "").splitlines():
        m = _DECL.match(line)
        if m:
            try:
                out[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return out


def step1_variables(client, model: str, turn_text: str, searches: str,
                    target: float, prev_target: float | None = None) -> dict:
    user = ("SEARCHES RUN:\n" + (searches or "(none)") +
            "\n\nREASONING WRITTEN:\n" + turn_text +
            "\n\nIt reported a final probability. Record the inputs to that "
            "reasoning, never the probability itself.")
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": STEP1_SYSTEM},
                  {"role": "user", "content": user}])
    text = r.choices[0].message.content or ""
    d = parse_declarations(text)
    # Mechanical defence against the answer wearing a hat, revised per review:
    #  * a TOLERANCE BAND (not exact equality), so `posterior_before_rounding
    #    = 0.205` cannot slip past a forecast of 0.21;
    #  * an EXEMPTION for values equal to the PREVIOUS day's forecast -- for an
    #    anchored harness, yesterday's number is a legitimate (and highly
    #    predictive) input, and dropping it would bias fit quality against
    #    exactly the harnesses whose anchoring we study;
    #  * every drop is logged, so the audit trail survives.
    kept, dropped = {}, []
    for name, v in d.items():
        near_output = abs(v - target) <= 0.005
        is_yesterday = prev_target is not None and abs(v - prev_target) <= 1e-9
        if near_output and not is_yesterday:
            dropped.append("%s=%.4g" % (name, v))
        else:
            kept[name] = v
    if dropped:
        print("    step1 dropped near-target vars: %s" % ", ".join(dropped), flush=True)
    return kept


# ------------------------------------------------------------------ step 2

STEP2_SYSTEM = """You are given variable tables extracted from several days of
one forecasting agent's reasoning. The names differ between days because they
were extracted independently, but many refer to the same underlying quantity.

Produce a single canonical schema: the union of quantities, deduplicated and
consistently named, plus, for every instance, the value each canonical name
takes there.

Output JSON only, exactly this shape:

{
  "schema": ["canonical_name_1", "canonical_name_2", ...],
  "bindings": { "<instance key>": {"canonical_name_1": <number>, ...}, ... },
  "notes": "one line on what you merged"
}

RULES
- Every instance must bind EVERY name in the schema. If a quantity was absent
  from an instance's reasoning, infer the value that reasoning implicitly
  treated it as (usually a neutral value), and still bind it.
- Merge aggressively: base_rate_cut and prior_probability_of_cut are one name.
- Keep the schema between 5 and 14 names.
- Names must describe quantities, never dates or instances."""


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if v == v else default
    except (TypeError, ValueError):
        return default


def step2_reconcile(client, model: str, instances: list,
                    max_attempts: int = 2) -> dict:
    """Reviewer fixes baked in: instance keys the LLM fails to copy verbatim
    used to be silently zero-filled (making a fit failure indistinguishable
    from 'the structure changed'), and the pre-reconcile bindings were
    overwritten, destroying the audit trail.  Now: missing keys are retried
    once, still-missing instances are EXCLUDED loudly rather than zero-filled,
    step-1 bindings are preserved on the instance, and non-numeric values
    coerce with a warning instead of crashing the whole stage."""
    for i in instances:
        i.raw_bindings = dict(i.bindings)

    tables = {i.key(): i.raw_bindings for i in instances}
    messages = [{"role": "system", "content": STEP2_SYSTEM},
                {"role": "user", "content":
                 "VARIABLE TABLES BY INSTANCE:\n" +
                 json.dumps(tables, indent=1)[:60000] +
                 "\n\nProduce the canonical schema and complete bindings."}]

    schema: list = []
    binds: dict = {}
    for attempt in range(max_attempts):
        r = client.chat.completions.create(
            model=model, response_format={"type": "json_object"},
            messages=messages)
        text = r.choices[0].message.content or "{}"
        try:
            d = json.loads(text)
        except json.JSONDecodeError:
            d = {}
        schema = [str(s) for s in d.get("schema", [])]
        binds = d.get("bindings", {}) or {}
        missing = [i.key() for i in instances if i.key() not in binds]
        if not missing or attempt == max_attempts - 1:
            break
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content":
                         "Your bindings are missing these instance keys, copied "
                         "verbatim below. Output the full JSON again with EVERY "
                         "key present:\n" + "\n".join(missing)})

    excluded = [i.key() for i in instances if i.key() not in binds]
    coerced = 0
    for i in instances:
        if i.key() in binds:
            b = binds[i.key()] or {}
            vals = {}
            for n in schema:
                raw = b.get(n, 0.0)
                v = _safe_float(raw)
                if not isinstance(raw, (int, float)):
                    coerced += 1
                vals[n] = v
            i.bindings = vals
    if coerced:
        print("    step2 coerced %d non-numeric values to floats" % coerced, flush=True)
    return {"schema": schema, "notes": d.get("notes", ""), "excluded": excluded}


# ------------------------------------------------------------------ step 3

STEP3_SYSTEM = """You are writing the program that this forecaster's reasoning
actually was.

You get a canonical variable schema, and for several instances the values those
variables took together with the probability the forecaster produced. Write ONE
program that reproduces every instance's output from its variables.

""" + SPEC_FOR_PROMPT + """

TWO THINGS MATTER EQUALLY.

FIDELITY. The program must mirror how the forecaster actually reasoned -- the
same intermediate quantities, combined in the same order, by the same kind of
operation. If it set a base rate and then applied adjustments, your program
sets a base rate and applies adjustments. Do not write a curve fit that happens
to land on the numbers.

NON-TRIVIALITY. A program that returns one variable unchanged, or that leans
almost entirely on a single variable, will be rejected even if it fits
perfectly. The variables are inputs; the program must do the combining work.

Output only the program, in a ```python block. No commentary."""


def _extract_code(text: str) -> str:
    m = re.findall(r"```(?:python)?\s*(.*?)```", text or "", re.S)
    return (m[0] if m else (text or "")).strip()


def degeneracy(instances: list, source: str) -> dict:
    """How much of the fit is one variable doing on its own?"""
    import ast as _ast
    names = sorted({n for i in instances for n in i.bindings})
    ys = [i.target for i in instances]
    n = len(ys)
    best, best_name = 0.0, None
    if n >= 3:
        my = sum(ys) / n
        for nm in names:
            xs = [i.bindings.get(nm, 0.0) for i in instances]
            mx = sum(xs) / n
            num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
            dx = sum((a - mx) ** 2 for a in xs) ** .5
            dy = sum((b - my) ** 2 for b in ys) ** .5
            if dx > 1e-12 and dy > 1e-12:
                c = abs(num / (dx * dy))
                if c > best:
                    best, best_name = c, nm

    passthrough = False
    try:
        tree = _ast.parse(source)
        fn = next(x for x in tree.body if isinstance(x, _ast.FunctionDef))
        stmts = [s for s in fn.body if not (isinstance(s, _ast.Expr)
                                            and isinstance(s.value, _ast.Constant))]
        if len(stmts) == 1 and isinstance(stmts[0], _ast.Return) \
                and isinstance(stmts[0].value, _ast.Name):
            passthrough = True
        ops = sum(1 for x in _ast.walk(fn)
                  if isinstance(x, (_ast.BinOp, _ast.Compare, _ast.Call, _ast.IfExp)))
        branches = sum(1 for x in _ast.walk(fn)
                       if isinstance(x, (_ast.If, _ast.IfExp)))
    except (SyntaxError, StopIteration):
        ops = 0
        branches = 0
    # A single variable tracking the target almost perfectly is only damning
    # when there are enough instances for that to be surprising AND the program
    # is doing little work of its own.
    near_passthrough = (n >= 6 and best > 0.98 and ops < 5)
    # a branch per instance is a lookup table, not a reasoning structure
    lookup_smell = (n >= 4 and branches >= n - 1)
    return {"max_abs_corr": best, "max_corr_var": best_name,
            "passthrough": passthrough, "n_ops": ops, "n_branches": branches,
            "near_passthrough": near_passthrough, "lookup_smell": lookup_smell,
            "suspect": passthrough or ops < 3 or near_passthrough or lookup_smell}


@dataclass
class Synthesis:
    program: str = ""
    fit: dict = field(default_factory=dict)
    degeneracy: dict = field(default_factory=dict)
    attempts: int = 0
    history: list = field(default_factory=list)
    ok: bool = False


def step3_synthesise(client, model: str, instances: list, schema: list,
                     tol: float = 0.02, max_attempts: int = 4,
                     excerpt: str = "") -> Synthesis:
    rows = [{"instance": i.key(), "variables": i.bindings,
             "forecaster_output": i.target} for i in instances]
    base_user = ("CANONICAL SCHEMA:\n" + json.dumps(schema, indent=1) +
                 "\n\nINSTANCES:\n" + json.dumps(rows, indent=1)[:50000])
    if excerpt:
        base_user += ("\n\nEXCERPT OF THE ORIGINAL REASONING (mirror its shape):\n"
                      + excerpt[:6000])

    syn = Synthesis()
    messages = [{"role": "system", "content": STEP3_SYSTEM},
                {"role": "user", "content": base_user}]
    pairs = [(i.bindings, i.target) for i in instances]

    for attempt in range(1, max_attempts + 1):
        syn.attempts = attempt
        r = client.chat.completions.create(model=model, messages=messages)
        text = r.choices[0].message.content or ""
        src = _extract_code(text)
        messages.append({"role": "assistant", "content": text})

        res = check(src, declared=set(schema))
        if not res.ok:
            syn.history.append({"attempt": attempt, "stage": "validator",
                                "problems": res.report()[:2000]})
            messages.append({"role": "user", "content":
                             "The program was REJECTED by the checker:\n\n" +
                             res.report() +
                             "\n\nFix every violation and output the program again."})
            continue

        fit = fit_error(src, pairs, tol=tol)
        deg = degeneracy(instances, src)
        syn.program, syn.fit, syn.degeneracy = src, fit, deg
        syn.history.append({"attempt": attempt, "stage": "fit",
                            "mae": fit["mae"], "within_tol": fit["within_tol"],
                            "degeneracy": deg})

        if deg["suspect"]:
            messages.append({"role": "user", "content":
                             "Rejected as trivial: the program does almost no combining "
                             "(%d operations%s). Rewrite it so it reproduces the "
                             "forecaster's actual reasoning steps."
                             % (deg["n_ops"],
                                ", and it just returns a variable unchanged"
                                if deg["passthrough"] else "")})
            continue

        # MAE alone is too weak a gate: a program can average under tolerance
        # while most individual instances miss. Require both.
        if (fit["mae"] == fit["mae"] and fit["mae"] <= tol
                and fit["within_tol"] >= 0.8 and fit["n_failed"] == 0):
            syn.ok = True
            return syn

        worst = sorted(
            [(r_, instances[j].key(), instances[j].target)
             for j, r_ in enumerate(fit["residuals"]) if r_ == r_],
            reverse=True)[:6]
        detail = "\n".join("  %s: your program is off by %.3f (target %.3f)"
                           % (k, r_, t) for r_, k, t in worst)
        errs = "\n".join("  %s -> %s" % (instances[j].key(), e)
                         for j, e in fit["errors"][:5])
        messages.append({"role": "user", "content":
                         ("Fit is not good enough. MAE %.4f, %.0f%% of instances within "
                          "tolerance.\n\nWorst instances:\n%s\n%s\n\nRevise the program. "
                          "Keep it faithful to the reasoning; do not add variables."
                          % (fit["mae"], 100 * fit["within_tol"], detail,
                             ("\nInstances that failed to run:\n" + errs) if errs else ""))})
    return syn


def variable_trajectory(instances: list) -> dict:
    """Bindings over dates -- the belief trajectory in the discovered basis."""
    out: dict = {}
    for i in sorted(instances, key=lambda x: (x.sample, x.date)):
        out.setdefault("s%d" % i.sample, {})[i.date] = dict(i.bindings, _target=i.target)
    return out
