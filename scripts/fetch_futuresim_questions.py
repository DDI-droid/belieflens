"""Pull FutureSim / OpenForesight questions into our JSONL format.

FutureSim (arXiv:2605.15188, github.com/OpenForecaster/futuresim) loads its
questions from a Hugging Face dataset; the default split used in the paper is
`aljazeera2026Q1` -- 330 short-answer questions resolving 1 Jan - 28 Mar 2026,
with the simulation starting 2025-12-24.

  python scripts/fetch_futuresim_questions.py --split aljazeera2026Q1 --out data/futuresim.jsonl

If the dataset id has moved, pass --repo explicitly.  Nothing else in this
codebase depends on HF being reachable -- the arms and perturbation stages
only need the JSONL.
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="OpenForecaster/OpenForesight")
    ap.add_argument("--split", default="aljazeera2026Q1")
    ap.add_argument("--out", default="data/futuresim.jsonl")
    ap.add_argument("--sim-start", default="2025-12-24",
                    help="as_of date for the day-0 elicitation")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from datasets import load_dataset          # pip install datasets
    ds = load_dataset(args.repo, split=args.split)

    n = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for i, row in enumerate(ds):
            if args.limit and n >= args.limit:
                break
            q = (row.get("question") or row.get("title") or "").strip()
            if not q:
                continue
            rec = {
                "id": str(row.get("id") or row.get("question_id") or i),
                "question": q,
                "as_of": args.sim_start,
                "background": (row.get("background") or row.get("context") or "").strip(),
                "outcomes": row.get("outcomes") or [],
                "resolution": row.get("answer") or row.get("resolution"),
                "resolution_date": str(row.get("resolution_date") or ""),
                "source": "futuresim/" + args.split,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print("wrote %d questions to %s" % (n, args.out))
    print("column names in the source split: %s" % (list(ds.features),))


if __name__ == "__main__":
    main()
