"""Run the BeliefLens experiments.

  python scripts/run_experiment.py --stage all --provider mock --questions data/questions_sample.jsonl

Stages
  arms     A/B/C elicitation on every question, k samples each.
  perturb  inject the model's own stated evidence cues back at it and measure
           how the structure moves (local + free + unstructured control).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from belieflens import arms as A                       # noqa: E402
from belieflens.arms import Question                    # noqa: E402
from belieflens.llm import LLM                          # noqa: E402

PLACEBO = ("the International Astronomical Union confirmed the naming of three "
           "minor planets discovered by amateur observers in 2024")


def load_questions(path: str, limit: int | None) -> list[Question]:
    qs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                qs.append(Question.from_dict(json.loads(line)))
    return qs[:limit] if limit else qs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["arms", "perturb", "all"])
    ap.add_argument("--provider", default="mock")
    ap.add_argument("--model", default="mock-1")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--questions", default="data/questions_sample.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=3, help="samples per question per arm")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-leaves", type=int, default=8)
    ap.add_argument("--modes", default="local,free,direct")
    ap.add_argument("--max-perturb-per-spec", type=int, default=2)
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    tag = "%s_%s" % (args.provider, args.model.replace("/", "-"))
    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)

    llm = LLM(provider=args.provider, model=args.model, effort=args.effort,
              use_cache=not args.no_cache)
    questions = load_questions(args.questions, args.limit)
    print("%d questions | %s/%s | k=%d" % (len(questions), args.provider, args.model, args.k))

    draws, spec_draws_by_q = [], {}

    if args.stage in ("arms", "all"):
        for qi, q in enumerate(questions, 1):
            print("  [%d/%d] arms: %s" % (qi, len(questions), q.text[:70]))
            d = A.arm_direct(llm, q, k=args.k)
            specs = A.arm_build(llm, q, k=args.k, rounds=args.rounds,
                                max_leaves=args.max_leaves)
            spec_draws_by_q[q.id] = specs
            draws += d
            draws += specs
            for rule in ("linear", "linear_simplex", "wep_only", "noisy_or", "stated"):
                draws += A.arm_composed(specs, rule=rule)
            draws += A.arm_conditioned(llm, q, specs)
        A.write_jsonl(out / "draws.jsonl", draws)
        n_bad = sum(1 for d in draws if d.error)
        print("  wrote %s (%d draws, %d parse failures)" % (out / "draws.jsonl", len(draws), n_bad))

    if args.stage in ("perturb", "all"):
        if not spec_draws_by_q:
            spec_draws_by_q = _reload_specs(out / "draws.jsonl", questions)
        modes = [m for m in args.modes.split(",") if m]
        results = []
        for qi, q in enumerate(questions, 1):
            for sd in spec_draws_by_q.get(q.id, []):
                if sd.spec is None:
                    continue
                perts = A.make_perturbations(sd.spec, date=q.as_of, placebo_text=PLACEBO)
                # keep one support + one contradict + the placebo per spec
                chosen, seen = [], set()
                for p in perts:
                    if p.condition in seen and p.condition != "placebo":
                        continue
                    seen.add(p.condition)
                    chosen.append(p)
                    if len([c for c in chosen if c.condition != "placebo"]) >= \
                            args.max_perturb_per_spec and "placebo" in seen:
                        break
                print("  [%d/%d] perturb s%d: %d probes" % (qi, len(questions), sd.sample_id, len(chosen)))
                for p in chosen:
                    for mode in modes:
                        if mode == "direct":
                            results.append(A.apply_perturbation_direct(
                                llm, q, sd.spec.decode("linear"), p, sd.sample_id))
                        else:
                            results.append(A.apply_perturbation(
                                llm, q, sd.spec, p, sd.sample_id, mode=mode))
        A.write_jsonl(out / "updates.jsonl", results)
        print("  wrote %s (%d updates, %d failures)"
              % (out / "updates.jsonl", len(results), sum(1 for r in results if r.error)))


def _reload_specs(path: Path, questions):
    from belieflens.dsl import BeliefSpec
    by_q: dict = {}
    if not path.exists():
        raise SystemExit("no draws at %s -- run --stage arms first" % path)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("kind") == "draw" and row.get("arm") == "spec" and row.get("spec"):
                d = A.Draw(arm="spec", question_id=row["question_id"],
                           sample_id=row["sample_id"], p=row.get("p"))
                d.spec = BeliefSpec.from_dict(row["spec"])
                by_q.setdefault(row["question_id"], []).append(d)
    return by_q


if __name__ == "__main__":
    main()
