"""Benchmark metric aggregation and report rendering."""

from __future__ import annotations

import html
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .schema import CaseResult, RunResult


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = (len(ordered) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _summary_stats(values: list[float]) -> dict[str, Any]:
    """Summary statistics for a metric distribution.

    ``stdev`` is the sample standard deviation, exposed as the jitter signal so
    callers can tell a stable metric from a noisy one. It is ``None`` for fewer
    than two samples because jitter is undefined there.
    """

    return {
        "count": len(values),
        "avg": statistics.fmean(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else None,
        "min": min(values) if values else None,
        "stdev": statistics.stdev(values) if len(values) >= 2 else None,
    }


def _metric_distribution(cases: list[CaseResult]) -> dict[str, Any]:
    metric_keys = sorted(
        {
            key
            for case in cases
            for key, value in case.metrics.items()
            if isinstance(value, (int, float)) and value is not None
        }
    )
    metrics: dict[str, Any] = {}
    for key in metric_keys:
        values = [
            float(case.metrics[key])
            for case in cases
            if isinstance(case.metrics.get(key), (int, float))
        ]
        metrics[key] = _summary_stats(values)
    return metrics


def aggregate(cases: list[CaseResult]) -> dict[str, Any]:
    total = len(cases)
    passed = sum(1 for case in cases if case.passed)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else 0.0,
        "metrics": _metric_distribution(cases),
    }


def aggregate_runs(runs: list[RunResult]) -> dict[str, Any]:
    """Aggregate one or more repeats of the same suite.

    The flat ``metrics`` block pools every (case x repeat) sample so the
    existing SLO and regression-gate readers keep working but now see real
    distributions instead of a single value per metric. ``per_case`` adds the
    statistically honest view: each case's metric summarized over its own N
    repeats, with jitter and per-case pass rate so flaky cases are visible.
    """

    all_cases = [case for run in runs for case in run.cases]
    by_case: dict[str, list[CaseResult]] = {}
    for case in all_cases:
        by_case.setdefault(case.case_id, []).append(case)

    passed_cases = sum(
        1 for cases in by_case.values() if cases and all(c.passed for c in cases)
    )
    flaky_cases = sum(
        1
        for cases in by_case.values()
        if any(c.passed for c in cases) and not all(c.passed for c in cases)
    )
    total_cases = len(by_case)

    per_case: dict[str, Any] = {}
    for case_id, cases in sorted(by_case.items()):
        case_passed = sum(1 for c in cases if c.passed)
        per_case[case_id] = {
            "runs": len(cases),
            "passed": case_passed,
            "pass_rate": case_passed / len(cases) if cases else 0.0,
            "errors": sorted({err for c in cases for err in c.errors}),
            "metrics": _metric_distribution(cases),
        }

    return {
        "repeats": len(runs),
        "total": total_cases,
        "passed": passed_cases,
        "failed": total_cases - passed_cases,
        "flaky": flaky_cases,
        "pass_rate": passed_cases / total_cases if total_cases else 0.0,
        "metrics": _metric_distribution(all_cases),
        "per_case": per_case,
    }


def write_metrics(run: RunResult, output_dir: Path) -> dict[str, Any]:
    payload = {
        "run": {
            "run_id": run.run_id,
            "git_sha": run.git_sha,
            "runner": run.runner,
            "profile": run.profile,
            "provider_config": run.provider_config,
        },
        "summary": aggregate(run.cases),
        "cases": [asdict(case) for case in run.cases],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    run = payload["run"]
    summary = payload["summary"]
    repeats = summary.get("repeats")
    lines = [
        f"# Voice Benchmark Report: {run['run_id']}",
        "",
        f"- runner: `{run['runner']}`",
        f"- profile: `{run['profile']}`",
        f"- git_sha: `{run['git_sha']}`",
        f"- pass_rate: `{summary['passed']}/{summary['total']}`",
    ]
    if repeats:
        lines.append(f"- repeats: `{repeats}`")
        if summary.get("flaky"):
            lines.append(f"- flaky cases: `{summary['flaky']}`")
    lines.extend(
        [
            "",
            "## Metric Summary",
            "",
            "| metric | p50 | p95 | max | stdev | count |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, values in summary["metrics"].items():
        lines.append(
            "| {key} | {p50} | {p95} | {maxv} | {stdev} | {count} |".format(
                key=key,
                p50=_fmt(values["p50"]),
                p95=_fmt(values["p95"]),
                maxv=_fmt(values["max"]),
                stdev=_fmt(values.get("stdev")),
                count=values["count"],
            )
        )
    per_case = summary.get("per_case")
    if per_case:
        lines.extend(
            [
                "",
                "## Per-Case Stability",
                "",
                "| case | pass | runs | pass_rate |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for case_id, info in per_case.items():
            lines.append(
                "| {case} | {passed} | {runs} | {rate:.0%} |".format(
                    case=case_id,
                    passed=info["passed"],
                    runs=info["runs"],
                    rate=info["pass_rate"],
                )
            )
    lines.extend(["", "## Cases", ""])
    for case in payload["cases"]:
        status = "PASS" if case["passed"] else "FAIL"
        lines.append(f"### {case['case_id']} - {status}")
        if case["errors"]:
            lines.append("")
            lines.extend(f"- {err}" for err in case["errors"])
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(case["metrics"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def render_html(payload: dict[str, Any]) -> str:
    markdown = render_markdown(payload)
    body = html.escape(markdown)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Voice Benchmark {html.escape(payload['run']['run_id'])}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 32px; }}
    pre {{ white-space: pre-wrap; background: #f7f7f8; padding: 16px; border-radius: 8px; }}
  </style>
</head>
<body>
  <pre>{body}</pre>
</body>
</html>
"""


def write_reports(run: RunResult, output_dir: Path) -> dict[str, Any]:
    payload = write_metrics(run, output_dir)
    (output_dir / "report.md").write_text(render_markdown(payload), encoding="utf-8")
    (output_dir / "report.html").write_text(render_html(payload), encoding="utf-8")
    return payload


def write_repeated_reports(runs: list[RunResult], output_dir: Path) -> dict[str, Any]:
    """Write a merged metrics/report payload for one or more suite repeats.

    Every (case x repeat) result is preserved under ``cases`` so the per-case
    drilldown stays honest; ``summary`` carries both the pooled distribution and
    the per-case stability view from :func:`aggregate_runs`.
    """

    if not runs:
        raise ValueError("write_repeated_reports requires at least one run")
    head = runs[0]
    payload = {
        "run": {
            "run_id": head.run_id,
            "git_sha": head.git_sha,
            "runner": head.runner,
            "profile": head.profile,
            "provider_config": head.provider_config,
        },
        "summary": aggregate_runs(runs),
        "cases": [asdict(case) for run in runs for case in run.cases],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(render_markdown(payload), encoding="utf-8")
    (output_dir / "report.html").write_text(render_html(payload), encoding="utf-8")
    return payload


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)
