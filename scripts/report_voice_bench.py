#!/usr/bin/env python3
"""Regenerate report files for a benchmark run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.report import render_html, render_markdown


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    (run_dir / "report.md").write_text(render_markdown(payload), encoding="utf-8")
    (run_dir / "report.html").write_text(render_html(payload), encoding="utf-8")
    print(run_dir / "report.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
