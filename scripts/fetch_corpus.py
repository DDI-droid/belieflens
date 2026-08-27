"""Download the FutureSim article corpus for the simulation window, plus the questions.

We take the real corpus (shash42/forecast-news) but NOT the 48 GB prebuilt
embedding index (shash42/forecast-news-embeddings).  A local keyword index is
built instead by scripts/build_index.py.  Retrieval quality is therefore lower
than the paper's hybrid search -- but it is held identical across all four
harnesses, so it cannot bias the comparison, which is the only thing this
experiment turns on.

Parquet, not jsonl: same content, ~2.7x smaller.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

CORPUS_REPO = "shash42/forecast-news"
QUESTIONS_REPO = "nikhilchandak/OpenForesight"


def month_range(start: str, end: str):
    sy, sm = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]), int(end[5:7])
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield "%04d/%02d" % (y, m)
        m += 1
        if m == 13:
            y, m = y + 1, 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-12", help="first month, YYYY-MM")
    ap.add_argument("--end", default="2026-03", help="last month, YYYY-MM")
    ap.add_argument("--out", default="data/corpus")
    ap.add_argument("--questions-out", default="data/questions")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    months = list(month_range(args.start, args.end))
    patterns = ["%s/*/articles_b*.parquet" % m for m in months]
    print("corpus: %d months %s -> %s" % (len(months), months[0] + ".." + months[-1], args.out))

    snapshot_download(
        repo_id=CORPUS_REPO, repo_type="dataset",
        local_dir=args.out, allow_patterns=patterns,
        max_workers=args.workers,
    )

    print("questions -> %s" % args.questions_out)
    snapshot_download(
        repo_id=QUESTIONS_REPO, repo_type="dataset",
        local_dir=args.questions_out, allow_patterns=["data/*.parquet", "README.md"],
        max_workers=4,
    )

    n = sum(1 for _ in Path(args.out).rglob("*.parquet"))
    mb = sum(p.stat().st_size for p in Path(args.out).rglob("*.parquet")) / 1e6
    print("done: %d parquet files, %.0f MB" % (n, mb))


if __name__ == "__main__":
    main()
