"""The program space a recovered reasoning trace is allowed to occupy.

The rule that makes the whole experiment mean something: **the program body may
not contain a single literal.**  Not 2, not 1, not 0.5, not a string.  Every
value enters through a name that was declared in step 1 by reading the
harness's own trajectory.  That forces the split we care about --

    the program  = the reasoning structure   (shared across all instances)
    the bindings = the belief state           (one per instance)

-- because there is nowhere else for a number to hide.  A program that fits
every instance with only its bindings changing has isolated what the harness
holds fixed from what it believes.

Four name classes are visible inside a program, and nothing else:

    globals    a fixed, predeclared namespace (operators and named constants)
    variables  the step-1 extraction for this instance
    locals     intermediate names the program assigns itself
    keywords   Python control flow

Functions are banned.  The program *is* one function, `forecast()`, taking no
arguments; names resolve from the injected namespace.  It may call itself, so
recursion is available, and `while` is available, so the space is Turing
complete -- which is deliberate.  We are not trying to make fitting hard, we
are trying to make the *literals* impossible.
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field

ENTRY = "forecast"

# ---------------------------------------------------------------- globals
# Everything a program could otherwise want a literal for.  Named, so the body
# stays literal-free; fixed, so programs remain comparable across harnesses.

def _clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def _logistic(x):
    if x < -700.0:
        return 0.0
    if x > 700.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def _noisy_or(*ps):
    q = 1.0
    for p in ps:
        q *= (1.0 - p)
    return 1.0 - q


def _odds(p):
    p = _clamp(p, 1e-9, 1.0 - 1e-9)
    return p / (1.0 - p)


def _prob(o):
    return o / (1.0 + o) if o >= 0 else 0.0


GLOBALS: dict = {
    # named constants -- the body never needs a numeral
    "ZERO": 0.0, "ONE": 1.0, "TWO": 2.0, "HALF": 0.5,
    "TENTH": 0.1, "HUNDREDTH": 0.01, "EPS": 1e-9,
    "E": math.e, "PI": math.pi,
    # operators
    "min": min, "max": max, "abs": abs, "round": round,
    "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
    "clamp": _clamp, "logistic": _logistic, "noisy_or": _noisy_or,
    "odds": _odds, "prob": _prob,
}

_ALLOWED_STMT = (
    ast.FunctionDef, ast.Assign, ast.AugAssign, ast.AnnAssign,
    ast.If, ast.While, ast.Return, ast.Pass, ast.Break, ast.Continue, ast.Expr,
)
_ALLOWED_EXPR = (
    ast.Name, ast.Load, ast.Store, ast.Constant, ast.BinOp, ast.UnaryOp,
    ast.BoolOp, ast.Compare, ast.Call, ast.IfExp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)

_BANNED_NODES = (
    ast.Import, ast.ImportFrom, ast.Lambda, ast.ClassDef, ast.AsyncFunctionDef,
    ast.Attribute, ast.Subscript, ast.ListComp, ast.SetComp, ast.DictComp,
    ast.GeneratorExp, ast.Yield, ast.YieldFrom, ast.Await, ast.Global,
    ast.Nonlocal, ast.Try, ast.With, ast.Assert, ast.Delete, ast.Raise,
    ast.For, ast.List, ast.Dict, ast.Set, ast.Tuple, ast.Starred,
    ast.NamedExpr, ast.JoinedStr, ast.Slice,
)


class ProgramError(ValueError):
    pass


# Names whose values are known at validation time.  An expression whose every
# leaf is one of these is a literal wearing a costume: TENTH*(TWO+TWO+TWO) is
# 0.6 as surely as writing 0.6, and the point of the no-literal rule is that
# constants must enter through step-1 declarations.  (Reviewer finding.)
_CONST_NAMES = frozenset({"ZERO", "ONE", "TWO", "HALF", "TENTH", "HUNDREDTH",
                          "EPS", "E", "PI"})


# ---- name rules: a value may not hide in a NAME any more than in a literal.
# `two = 2`, `half = 0.5`, `const_0_62 = 0.62` are literals with extra steps.
# A name must say what the quantity IS in the world (roster_size, base_rate_cut),
# never what its value is.
_NUM_WORDS = frozenset("""zero one two three four five six seven eight nine ten
eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen
twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million
half halves quarter quarters third thirds tenth tenths hundredth hundredths
point dot decimal minus neg negative num number const constant value val
literal fixed magic""".split())


def numeric_name(name: str) -> bool:
    """True if the name carries no semantic content beyond a value spelling."""
    toks = [t for t in str(name).lower().split("_") if t]
    if not toks:
        return True
    return all(t.isdigit() or t in _NUM_WORDS for t in toks)


def encodes_value(name: str, value: float) -> bool:
    """True if the name's digits spell the bound value (p_62 = 0.62)."""
    digits_in_name = "".join(c for c in str(name) if c.isdigit())
    if len(digits_in_name) < 2:
        return False
    try:
        sig = ("%g" % abs(float(value))).replace(".", "").lstrip("0")
    except (TypeError, ValueError):
        return False
    return len(sig) >= 2 and sig in digits_in_name


def _name_is(node, ident: str) -> bool:
    return isinstance(node, ast.Name) and node.id == ident


def _const_only(node) -> bool:
    """True if the expression's only Name leaves are named constants."""
    has_const_name = False
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            if sub.id in _CONST_NAMES:
                has_const_name = True
            elif sub.id in GLOBALS:      # a callable like clamp -- keep looking
                continue
            else:
                return False             # a real variable participates
    return has_const_name


@dataclass
class Violation:
    line: int
    rule: str
    detail: str

    def __str__(self) -> str:
        return "line %d: %s -- %s" % (self.line, self.rule, self.detail)


@dataclass
class CheckResult:
    ok: bool
    violations: list = field(default_factory=list)
    names_used: set = field(default_factory=set)

    def report(self) -> str:
        return "\n".join(str(v) for v in self.violations)


def check(source: str, declared: set | None = None) -> CheckResult:
    """Static validation.  `declared` is the step-1 variable schema; when given,
    any free name outside globals+declared is a violation."""
    vs: list = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return CheckResult(False, [Violation(e.lineno or 0, "syntax", str(e.msg))])

    # exactly one top-level def, named ENTRY, zero args
    body = [n for n in tree.body if not isinstance(n, ast.Expr)
            or not isinstance(getattr(n, "value", None), ast.Constant)]
    defs = [n for n in body if isinstance(n, ast.FunctionDef)]
    if len(body) != 1 or len(defs) != 1:
        vs.append(Violation(1, "shape",
                            "module must contain exactly one def and nothing else "
                            "(found %d top-level statements)" % len(body)))
    if defs:
        fn = defs[0]
        if fn.name != ENTRY:
            vs.append(Violation(fn.lineno, "shape", "entry point must be named %r" % ENTRY))
        a = fn.args
        if a.args or a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg:
            vs.append(Violation(fn.lineno, "shape",
                                "%s() takes no arguments; names resolve from the "
                                "injected namespace" % ENTRY))
    else:
        return CheckResult(False, vs)

    fn = defs[0]
    assigned: set = set()
    used: set = set()

    # a leading docstring is documentation, not data -- exempt it
    docstring_node = None
    if (fn.body and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)
            and isinstance(fn.body[0].value.value, str)):
        docstring_node = fn.body[0].value

    for node in ast.walk(fn):
        # constant-construction: arithmetic whose every leaf is a named
        # constant is a smuggled literal (reviewer finding)
        if (isinstance(node, (ast.BinOp, ast.Compare, ast.BoolOp, ast.UnaryOp))
                and _const_only(node)):
            vs.append(Violation(getattr(node, "lineno", 0), "constant-construction",
                                "expression built only from named constants -- "
                                "declare the value as a step-1 variable instead"))
        if (isinstance(node, ast.Call) and node.args
                and all(_const_only(a) for a in node.args)):
            vs.append(Violation(node.lineno, "constant-construction",
                                "call whose every argument is a named constant"))

        # dead code and op-count inflation (review F2 residual, now closed):
        # multiplying by ZERO, identity operations, and self-ops that
        # manufacture 0 or 1 from any variable (x-x, x/x) do no work and exist
        # only to pad the operation count or launder a constant.
        if isinstance(node, ast.BinOp):
            L, R, op = node.left, node.right, node.op
            if isinstance(op, ast.Mult) and (_name_is(L, "ZERO") or _name_is(R, "ZERO")):
                vs.append(Violation(node.lineno, "dead-code", "multiplication by ZERO"))
            elif isinstance(op, (ast.Div, ast.FloorDiv, ast.Mod)) and _name_is(L, "ZERO"):
                vs.append(Violation(node.lineno, "dead-code", "ZERO divided by anything"))
            elif ((isinstance(op, ast.Mult) and (_name_is(L, "ONE") or _name_is(R, "ONE")))
                  or (isinstance(op, ast.Add) and (_name_is(L, "ZERO") or _name_is(R, "ZERO")))
                  or (isinstance(op, ast.Sub) and _name_is(R, "ZERO"))
                  or (isinstance(op, (ast.Div, ast.Pow)) and _name_is(R, "ONE"))):
                vs.append(Violation(node.lineno, "identity-op",
                                    "operation with no effect -- inflates the op count"))
            elif (isinstance(op, (ast.Sub, ast.Div)) and not _name_is(L, "ZERO")
                  and ast.dump(L) == ast.dump(R)):
                vs.append(Violation(node.lineno, "self-op",
                                    "x-x / x/x manufactures a constant from a variable"))

        # number-named locals: `two = ...` is a literal wearing a name
        if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                and numeric_name(node.id)):
            vs.append(Violation(node.lineno, "numeric-name",
                                "%r spells a value, not a quantity -- name what it IS"
                                % node.id))

        if isinstance(node, _BANNED_NODES):
            vs.append(Violation(getattr(node, "lineno", 0), "banned-construct",
                                type(node).__name__ + " is not in the program space"))
            continue

        # the only FunctionDef allowed is the entry point itself
        if isinstance(node, ast.FunctionDef) and node is not fn:
            vs.append(Violation(node.lineno, "no-functions",
                                "helper function %r -- the program is a single function" % node.name))

        # THE rule: no literals in the body
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                vs.append(Violation(node.lineno, "magic-bool",
                                    "%r -- True/False are 1/0 in disguise; use "
                                    "comparisons of declared names" % (node.value,)))
            elif isinstance(node.value, (int, float, complex)):
                vs.append(Violation(node.lineno, "magic-number",
                                    "literal %r -- declare it as a variable in step 1"
                                    % (node.value,)))
            elif isinstance(node.value, (str, bytes)) and node is not docstring_node:
                vs.append(Violation(node.lineno, "magic-string",
                                    "literal %r -- declare it as a variable" % (node.value,)))

        if isinstance(node, ast.Call):
            f = node.func
            if not isinstance(f, ast.Name):
                vs.append(Violation(node.lineno, "call", "only bare-name calls are allowed"))
            elif f.id not in GLOBALS and f.id != ENTRY:
                vs.append(Violation(node.lineno, "call",
                                    "%r is not a global; callable globals are: %s"
                                    % (f.id, ", ".join(sorted(k for k, v in GLOBALS.items()
                                                              if callable(v))))))
            if node.keywords:
                vs.append(Violation(node.lineno, "call", "keyword arguments are not allowed"))

        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                assigned.add(node.id)
            else:
                used.add(node.id)

    free = {n for n in used if n not in assigned and n not in GLOBALS and n != ENTRY}
    if declared is not None:
        for n in sorted(free - set(declared)):
            vs.append(Violation(fn.lineno, "undeclared",
                                "%r is used but was never declared in step 1" % n))

    if not any(isinstance(n, ast.Return) for n in ast.walk(fn)):
        vs.append(Violation(fn.lineno, "shape", "no return statement"))

    # a local that is assigned and never read is padding, not reasoning
    for n in sorted(assigned - used):
        vs.append(Violation(fn.lineno, "unused-local",
                            "%r is assigned but never used -- dead padding" % n))

    return CheckResult(not vs, vs, free)


# ------------------------------------------------------------- execution

class StepLimit(RuntimeError):
    pass


def run(source: str, bindings: dict, max_steps: int = 200_000) -> float:
    """Execute a validated program against one instance's bindings.

    `while` and recursion are allowed, so a step limit is mandatory; a program
    that will not terminate is a failed fit, not a hung experiment.
    """
    import sys

    res = check(source, declared=set(bindings))
    if not res.ok:
        raise ProgramError(res.report())

    ns: dict = {"__builtins__": {}}
    ns.update(GLOBALS)
    ns.update(bindings)

    exec(compile(ast.parse(source), "<program>", "exec"), ns)   # noqa: S102 - validated above
    fn = ns[ENTRY]

    counter = {"n": 0}

    def tracer(frame, event, arg):
        if event == "line":
            counter["n"] += 1
            if counter["n"] > max_steps:
                raise StepLimit("exceeded %d steps" % max_steps)
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        out = fn()
    finally:
        sys.settrace(old)

    if isinstance(out, bool) or not isinstance(out, (int, float)):
        raise ProgramError("forecast() must return a number, got %r" % type(out).__name__)
    return float(out)


def fit_error(source: str, instances: list, tol: float = 0.02) -> dict:
    """Residuals of one program across every instance.

    `instances` is a list of (bindings, target).  This is the number the whole
    experiment turns on: if a single literal-free program reproduces every
    forecast the harness made, the reasoning was program-like and the bindings
    are the belief.
    """
    residuals, errors = [], []
    for i, (bindings, target) in enumerate(instances):
        try:
            got = run(source, bindings)
            residuals.append(abs(got - target))
        except (ProgramError, StepLimit, ZeroDivisionError, ValueError,
                OverflowError, TypeError, KeyError, RecursionError,
                NameError) as e:  # NameError incl. UnboundLocalError: a program
                                  # that binds a name on only one branch
            residuals.append(float("nan"))
            errors.append((i, "%s: %s" % (type(e).__name__, e)))
    good = [r for r in residuals if r == r]
    return {
        "n": len(instances),
        "n_ran": len(good),
        "n_failed": len(errors),
        "mae": sum(good) / len(good) if good else float("nan"),
        "max": max(good) if good else float("nan"),
        "within_tol": sum(1 for r in good if r <= tol) / len(instances) if instances else 0.0,
        "residuals": residuals,
        "errors": errors,
    }


SPEC_FOR_PROMPT = """PROGRAM SPACE — these rules are checked mechanically and a
violation is rejected outright.

1. The whole program is ONE function:            def forecast():
   It takes no arguments. Names resolve from an injected namespace.
2. NO LITERALS ANYWHERE IN THE BODY. Not 2, not 1, not 0.5, not "".
   Every value must arrive as a declared variable name. This is absolute.
3. No other function definitions, no lambdas, no classes, no imports.
   forecast() may call itself, so recursion is available.
4. Allowed statements: assignment, augmented assignment, if/elif/else, while,
   break, continue, return.
   Banned: for, try, with, comprehensions, generators, attribute access (a.b),
   subscripting (a[b]), tuples, lists, dicts, sets, f-strings.
5. Calls may only target these globals, by bare name:
     min max abs round exp log sqrt clamp logistic noisy_or odds prob
   Named constants also available (use these instead of numerals):
     ZERO ONE TWO HALF TENTH HUNDREDTH EPS E PI
6. Return a single number: the forecast probability.
7. NO VALUE MAY HIDE IN A NAME. Locals like two, half, const_val are rejected;
   every name must say what the quantity IS (blended_prior, evidence_pull).
8. NO DEAD CODE. Multiplying by ZERO, adding ZERO, dividing by ONE, x-x, x/x,
   and assigned-but-unused locals are all rejected mechanically.
9. NO EXPRESSION MAY BE BUILT ONLY FROM NAMED CONSTANTS. TENTH*(TWO+TWO+TWO)
   is a smuggled literal and is rejected. A value either arrives as a declared
   variable or is genuinely computed from declared variables.

Intermediate local variables are encouraged — name the quantities your
reasoning actually used."""
