#!/usr/bin/env python3
"""Pixel-level recall / weighted precision / F1 on GSA, with one threshold tuned per category (HyperFree / ESIA
protocol; false positives weighted 0.5; event macro within category, then category macro).

    python scripts/gsa/evaluate.py --results-root results/gsa --label-root data/GSA/original
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e2forecast.anomaly import evaluate_score


def main(args: argparse.Namespace) -> None:
    output_dir = args.output_dir or args.results_root / "threshold_eval"
    summaries = [evaluate_score(args.results_root, args.label_root, score, output_dir / score, args.background_weight,
                                args.bins, args.exact) for score in args.scores]
    table = [{k: s[k] for k in ("score", "macro_recall", "macro_weighted_precision", "macro_weighted_f1")}
             for s in summaries]
    (output_dir / "summary.json").write_text(json.dumps(table, indent=2) + "\n")
    print(json.dumps(table, indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", type=Path, required=True, help="output directory of scripts/gsa/infer.py")
    p.add_argument("--label-root", type=Path, required=True, help="10 m GSA root with <category>/<event>/*label*.tif")
    p.add_argument("--output-dir", type=Path, help="default: <results-root>/threshold_eval")
    p.add_argument("--scores", nargs="+", default=["e2forecast"], choices=("e2forecast", "mahalanobis", "cosine"))
    p.add_argument("--background-weight", type=float, default=0.5)
    p.add_argument("--bins", type=int, default=65_536, help="histogram threshold search resolution")
    p.add_argument("--exact", action="store_true", help="search every observed score instead of histogram bins")
    return p


if __name__ == "__main__":
    main(parser().parse_args())
