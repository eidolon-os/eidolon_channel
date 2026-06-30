"""Baseline-vs-candidate comparison helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_metrics(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if p.is_dir():
        p = p / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8"))


def compare_metrics(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    max_p95_regression_pct: float = 10.0,
) -> dict[str, Any]:
    failures: list[str] = []
    rows: list[dict[str, Any]] = []

    b_summary = baseline["summary"]
    c_summary = candidate["summary"]
    if c_summary["failed"] > b_summary["failed"]:
        failures.append(
            f"failed cases increased: {b_summary['failed']} -> {c_summary['failed']}"
        )

    for key, b_values in b_summary["metrics"].items():
        c_values = c_summary["metrics"].get(key)
        if not c_values:
            continue
        b_p95 = b_values.get("p95")
        c_p95 = c_values.get("p95")
        delta_pct = None
        if isinstance(b_p95, (int, float)) and b_p95 > 0 and isinstance(c_p95, (int, float)):
            delta_pct = (c_p95 - b_p95) / b_p95 * 100
            if delta_pct > max_p95_regression_pct:
                failures.append(
                    f"{key}.p95 regressed by {delta_pct:.1f}%: {b_p95:.1f} -> {c_p95:.1f}"
                )
        rows.append(
            {
                "metric": key,
                "baseline_p95": b_p95,
                "candidate_p95": c_p95,
                "delta_pct": delta_pct,
            }
        )

    return {
        "passed": not failures,
        "failures": failures,
        "rows": rows,
    }
