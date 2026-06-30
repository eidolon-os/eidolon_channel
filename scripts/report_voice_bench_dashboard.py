#!/usr/bin/env python3
"""Render a visual dashboard for a complete voice benchmark run."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.dashboard import DashboardRunner, write_dashboard


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-run", required=True)
    parser.add_argument("--direct-run", required=True)
    parser.add_argument("--livekit-room-run")
    parser.add_argument("--output", required=True)
    parser.add_argument("--component-regression-pct", type=float, default=30.0)
    parser.add_argument("--direct-regression-pct", type=float, default=30.0)
    args = parser.parse_args()

    full_run = Path(args.full_run)
    direct_run = Path(args.direct_run)
    write_dashboard(
        runners=[
            DashboardRunner(
                name="policy",
                candidate=full_run / "policy",
                baseline=Path("benchmark/baselines/current/policy"),
                max_p95_regression_pct=10.0,
            ),
            DashboardRunner(
                name="headless",
                candidate=full_run / "headless",
                baseline=Path("benchmark/baselines/current/headless"),
                max_p95_regression_pct=10.0,
            ),
            DashboardRunner(
                name="component",
                candidate=full_run / "component",
                baseline=Path("benchmark/baselines/current/component"),
                max_p95_regression_pct=args.component_regression_pct,
            ),
            DashboardRunner(
                name="headless_direct",
                candidate=direct_run,
                baseline=Path("benchmark/baselines/current/headless_direct"),
                max_p95_regression_pct=args.direct_regression_pct,
            ),
        ]
        + (
            [
                DashboardRunner(
                    name="livekit_room",
                    candidate=Path(args.livekit_room_run),
                    baseline=None,
                )
            ]
            if args.livekit_room_run
            else []
        ),
        output_path=Path(args.output),
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
