#!/usr/bin/env python3
"""Compare a candidate voice benchmark run against a baseline."""

from __future__ import annotations

import argparse
import json

from benchmark.compare import compare_metrics, load_metrics


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--max-p95-regression-pct", type=float, default=10.0)
    args = parser.parse_args()

    result = compare_metrics(
        load_metrics(args.baseline),
        load_metrics(args.candidate),
        max_p95_regression_pct=args.max_p95_regression_pct,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
