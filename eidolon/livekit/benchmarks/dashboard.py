"""Visual dashboard renderer for a full voice benchmark run."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .compare import compare_metrics, load_metrics
from .slo import evaluate_slo_gates
from .timeline import load_timeline_records, summarize_timeline_records


@dataclass(frozen=True)
class DashboardRunner:
    name: str
    candidate: Path
    baseline: Path | None = None
    max_p95_regression_pct: float = 10.0


def write_dashboard(
    *,
    runners: list[DashboardRunner],
    output_path: Path,
) -> dict[str, Any]:
    payload = _build_payload(runners)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_dashboard_html(payload), encoding="utf-8")
    (output_path.parent / "dashboard.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def render_dashboard_html(payload: dict[str, Any]) -> str:
    runner_cards = "\n".join(_runner_card(runner) for runner in payload["runners"])
    metric_sections = "\n".join(
        _metric_section(runner) for runner in payload["runners"]
    )
    slo_sections = "\n".join(_slo_section(runner) for runner in payload["runners"])
    timeline_sections = "\n".join(
        _timeline_section(runner) for runner in payload["runners"]
    )
    findings = "\n".join(_finding_item(item) for item in payload["findings"])
    if not findings:
        findings = "<li class='ok'>No failed cases or regression gates.</li>"

    title = html.escape(payload["title"])
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f8fafc;
      --panel: #ffffff;
      --text: #111827;
      --muted: #64748b;
      --line: #e2e8f0;
      --ok: #047857;
      --warn: #b45309;
      --bad: #b91c1c;
      --blue: #2563eb;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 28px 0 12px;
      font-size: 18px;
    }}
    .subtitle {{
      color: var(--muted);
      margin-bottom: 22px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 12px;
    }}
    .card, .section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .card h3 {{
      margin: 0 0 10px;
      font-size: 16px;
    }}
    .big {{
      font-size: 28px;
      font-weight: 700;
      margin: 6px 0;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    .ok {{ color: var(--ok); }}
    .warn {{ color: var(--warn); }}
    .bad {{ color: var(--bad); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{ text-align: left; white-space: normal; }}
    .bar-track {{
      height: 7px;
      background: #e5e7eb;
      border-radius: 99px;
      overflow: hidden;
    }}
    .bar {{
      height: 100%;
      background: var(--blue);
    }}
    .stage {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      background: #eef2ff;
      color: #3730a3;
      font-size: 12px;
      font-weight: 600;
    }}
    .compact td, .compact th {{
      padding: 6px 8px;
      font-size: 12px;
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
    }}
    li {{
      margin: 6px 0;
      line-height: 1.45;
    }}
    a {{ color: var(--blue); text-decoration: none; }}
  </style>
</head>
<body>
<main>
  <h1>{title}</h1>
  <div class="subtitle">{html.escape(payload["subtitle"])}</div>
  <div class="grid">{runner_cards}</div>
  <h2>Findings</h2>
  <div class="section"><ul>{findings}</ul></div>
  {slo_sections}
  {metric_sections}
  {timeline_sections}
</main>
</body>
</html>
"""


def _build_payload(runners: list[DashboardRunner]) -> dict[str, Any]:
    runner_payloads: list[dict[str, Any]] = []
    findings: list[dict[str, str]] = []
    run_ids: list[str] = []

    for runner in runners:
        metrics = load_metrics(runner.candidate)
        run = metrics["run"]
        summary = metrics["summary"]
        run_ids.append(str(run["run_id"]))
        comparison = None
        if runner.baseline is not None:
            comparison = compare_metrics(
                load_metrics(runner.baseline),
                metrics,
                max_p95_regression_pct=runner.max_p95_regression_pct,
            )
        if summary["failed"]:
            findings.append(
                {
                    "severity": "bad",
                    "text": f"{runner.name}: {summary['failed']} failed case(s).",
                }
            )
        findings.extend(_flaky_findings(runner.name, summary))
        if comparison is not None and not comparison["passed"]:
            findings.extend(
                {
                    "severity": "warn",
                    "text": f"{runner.name}: {failure}",
                }
                for failure in comparison["failures"]
            )
        timeline = summarize_timeline_records(load_timeline_records(runner.candidate))
        findings.extend(_timeline_coverage_findings(runner.name, metrics, timeline))
        findings.extend(_real_call_findings(runner.name, metrics))
        runner_payloads.append(
            {
                "name": runner.name,
                "path": str(runner.candidate),
                "run": run,
                "summary": summary,
                "comparison": comparison,
                "timeline": timeline,
            }
        )
        slo_results = evaluate_slo_gates(runner_payloads[-1])
        runner_payloads[-1]["slo_gates"] = slo_results
        findings.extend(
            {
                "severity": "warn",
                "text": _slo_failure_text(result),
            }
            for result in slo_results
            if not result["passed"]
        )

    return {
        "title": "Eidolon Voice Benchmark Dashboard",
        "subtitle": "Run IDs: " + ", ".join(dict.fromkeys(run_ids)),
        "runners": runner_payloads,
        "findings": findings,
    }


def _real_call_findings(
    runner_name: str,
    metrics: dict[str, Any],
) -> list[dict[str, str]]:
    """Surface whether each case provably hit real providers.

    Only runners that ran the real-call verifier set ``real_call_verified`` on
    their cases; for others this is silent.
    """

    cases = [
        case
        for case in metrics.get("cases", [])
        if "real_call_verified" in (case.get("metrics") or {})
    ]
    if not cases:
        return []
    findings: list[dict[str, str]] = []
    verified = 0
    for case in cases:
        case_metrics = case.get("metrics") or {}
        if case_metrics.get("real_call_verified"):
            verified += 1
            continue
        detail = case_metrics.get("real_call_warnings") or "; ".join(
            err for err in (case.get("errors") or []) if err.startswith("real-call:")
        )
        findings.append(
            {
                "severity": "bad",
                "text": (
                    f"{runner_name}: real call NOT verified for "
                    f"{case.get('case_id')}: {detail or 'no evidence'}"
                ),
            }
        )
    severity = "ok" if verified == len(cases) else "warn"
    findings.insert(
        0,
        {
            "severity": severity,
            "text": f"{runner_name}: real calls verified for {verified}/{len(cases)} cases.",
        },
    )
    return findings


def _flaky_findings(
    runner_name: str,
    summary: dict[str, Any],
) -> list[dict[str, str]]:
    """Surface cases that passed in some repeats but not all.

    A single-repeat run cannot be flaky, so this is silent unless ``--repeat``
    produced more than one sample per case.
    """

    if int(summary.get("repeats") or 0) <= 1:
        return []
    findings: list[dict[str, str]] = []
    for case_id, info in (summary.get("per_case") or {}).items():
        pass_rate = info.get("pass_rate")
        if isinstance(pass_rate, (int, float)) and 0.0 < pass_rate < 1.0:
            findings.append(
                {
                    "severity": "warn",
                    "text": (
                        f"{runner_name}: flaky case {case_id} passed "
                        f"{info.get('passed')}/{info.get('runs')} repeats."
                    ),
                }
            )
    return findings


def _timeline_coverage_findings(
    runner_name: str,
    metrics: dict[str, Any],
    timeline: dict[str, Any],
) -> list[dict[str, str]]:
    if runner_name != "livekit_room":
        return []
    case_ids = {
        str(case.get("case_id") or "")
        for case in metrics.get("cases", [])
        if case.get("case_id")
    }
    if not case_ids:
        return []
    covered = set((timeline.get("cases") or {}).keys())
    missing = sorted(case_ids - covered)
    if not missing:
        return []
    return [
        {
            "severity": "warn",
            "text": (
                f"{runner_name}: timeline coverage missing "
                f"{len(missing)}/{len(case_ids)} case(s): {', '.join(missing)}"
            ),
        }
    ]


def _runner_card(runner: dict[str, Any]) -> str:
    summary = runner["summary"]
    comparison = runner.get("comparison")
    slo_gates = runner.get("slo_gates") or []
    slo_failed = sum(1 for gate in slo_gates if not gate.get("passed"))
    slo_label = (
        f"SLO gates pass ({len(slo_gates)})"
        if slo_failed == 0
        else f"SLO warning ({slo_failed}/{len(slo_gates)})"
    )
    slo_class = "ok" if slo_failed == 0 else "warn"
    compare_label = "no baseline"
    compare_class = "meta"
    if comparison is not None:
        compare_label = "regression gate pass" if comparison["passed"] else "regression gate warning"
        compare_class = "ok" if comparison["passed"] else "warn"
    status_class = "ok" if summary["failed"] == 0 else "bad"
    repeats = int(summary.get("repeats") or 0)
    flaky = int(summary.get("flaky") or 0)
    repeat_line = ""
    if repeats > 1:
        flaky_label = f", {flaky} flaky" if flaky else ""
        repeat_line = (
            f'<div class="meta">{repeats} repeats per case{flaky_label}</div>'
        )
    return f"""<div class="card">
  <h3>{html.escape(runner["name"])}</h3>
  <div class="big {status_class}">{summary["passed"]}/{summary["total"]}</div>
  <div class="meta">cases passed</div>
  {repeat_line}
  <div class="{compare_class}">{html.escape(compare_label)}</div>
  <div class="{slo_class}">{html.escape(slo_label)}</div>
  {_provider_config_line(runner["run"].get("provider_config"))}
  <div class="meta">profile: {html.escape(str(runner["run"]["profile"]))}</div>
  <div class="meta">path: {html.escape(runner["path"])}</div>
</div>"""


def _provider_config_line(provider_config: Any) -> str:
    if not isinstance(provider_config, dict) or not provider_config:
        return ""
    parts = ", ".join(f"{role}={name}" for role, name in provider_config.items())
    return f'<div class="meta">providers: {html.escape(parts)}</div>'


def _metric_section(runner: dict[str, Any]) -> str:
    metrics = runner["summary"]["metrics"]
    rows = "\n".join(_metric_row(key, value) for key, value in metrics.items())
    return f"""<h2>{html.escape(runner["name"])} Metrics</h2>
<div class="section">
  <table>
    <thead>
      <tr><th>Metric</th><th>P50</th><th>P95</th><th>Max</th><th>Count</th><th>P95 Bar</th></tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""


def _slo_section(runner: dict[str, Any]) -> str:
    gates = runner.get("slo_gates") or []
    if not gates:
        return ""
    rows = "\n".join(_slo_row(gate) for gate in gates)
    return f"""<h2>{html.escape(runner["name"])} SLO Gates</h2>
<div class="section">
  <table>
    <thead>
      <tr><th>Gate</th><th>Tier</th><th>Source</th><th>Metric</th><th>Value</th><th>Max</th><th>N</th><th>Status</th></tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""


def _slo_row(gate: dict[str, Any]) -> str:
    if gate.get("advisory"):
        status, status_class = "ADVISORY", "warn"
    elif gate.get("passed"):
        status, status_class = "PASS", "ok"
    else:
        status, status_class = "FAIL", "bad"
    metric = f"{gate.get('metric')}.{gate.get('statistic')}"
    return f"""<tr>
  <td>{html.escape(str(gate.get("name", "")))}</td>
  <td>{html.escape(str(gate.get("tier", "")))}</td>
  <td>{html.escape(str(gate.get("source", "")))}</td>
  <td>{html.escape(metric)}</td>
  <td>{_fmt(gate.get("value"))}</td>
  <td>{_fmt(gate.get("max_value"))}</td>
  <td>{html.escape(str(gate.get("count", "")))}</td>
  <td class="{status_class}">{status}</td>
</tr>"""


def _metric_row(key: str, values: dict[str, Any]) -> str:
    p95 = values.get("p95")
    max_value = values.get("max")
    return f"""<tr>
  <td>{html.escape(key)}</td>
  <td>{_fmt(values.get("p50"))}</td>
  <td>{_fmt(p95)}</td>
  <td>{_fmt(max_value)}</td>
  <td>{html.escape(str(values.get("count", "")))}</td>
  <td>{_bar(p95, max_value)}</td>
</tr>"""


def _timeline_section(runner: dict[str, Any]) -> str:
    timeline = runner.get("timeline") or {}
    count = int(timeline.get("count") or 0)
    if count <= 0:
        return f"""<h2>{html.escape(runner["name"])} Turn Timeline</h2>
<div class="section">
  <div class="meta">No timeline JSONL found for this runner. Set observability.timeline_debug_path for the agent worker to collect per-turn timeline data.</div>
</div>"""

    latency_rows = "\n".join(
        _metric_row(key, value)
        for key, value in (timeline.get("latencies") or {}).items()
    )
    provider_segment_rows = "\n".join(
        _provider_segment_row(value)
        for value in (timeline.get("provider_segments") or {}).values()
    )
    provider_case_rows = "\n".join(
        _provider_case_row(record)
        for record in (timeline.get("record_summaries") or [])
    )
    decision_rows = "\n".join(
        _count_row(key, value)
        for key, value in (timeline.get("decision_reasons") or {}).items()
    )
    flush_rows = "\n".join(
        _count_row(key, value)
        for key, value in (timeline.get("flush_reasons") or {}).items()
    )
    case_rows = "\n".join(
        _count_row(key, value)
        for key, value in (timeline.get("cases") or {}).items()
    )
    if not decision_rows:
        decision_rows = "<tr><td colspan='2' class='meta'>No turn-policy decisions recorded.</td></tr>"
    if not flush_rows:
        flush_rows = "<tr><td colspan='2' class='meta'>No timeline flush reasons recorded.</td></tr>"
    if not case_rows:
        case_rows = "<tr><td colspan='2' class='meta'>No case IDs recorded in timeline attrs.</td></tr>"
    if not provider_segment_rows:
        provider_segment_rows = "<tr><td colspan='7' class='meta'>No provider segments recorded.</td></tr>"
    if not provider_case_rows:
        provider_case_rows = "<tr><td colspan='11' class='meta'>No per-turn provider segments recorded.</td></tr>"

    return f"""<h2>{html.escape(runner["name"])} Turn Timeline</h2>
<div class="section">
  <div class="meta">{count} turn timeline record(s)</div>
  <h3>Case Coverage</h3>
  <table>
    <thead><tr><th>Case</th><th>Timeline Records</th></tr></thead>
    <tbody>{case_rows}</tbody>
  </table>
  <h3>Provider Latency Breakdown</h3>
  <table>
    <thead>
      <tr><th>Stage</th><th>Segment</th><th>P50</th><th>P95</th><th>Max</th><th>Count</th><th>P95 Bar</th></tr>
    </thead>
    <tbody>{provider_segment_rows}</tbody>
  </table>
  <h3>Per-Turn Provider Segments</h3>
  <table class="compact">
    <thead>
      <tr>
        <th>Case</th><th>Action</th><th>Intent</th><th>STT Audio</th>
        <th>STT Provider</th><th>STT LiveKit</th><th>STT Final</th>
        <th>Turn Commit</th><th>Brain Delta</th><th>TTS First Audio</th><th>Interrupt</th>
      </tr>
    </thead>
    <tbody>{provider_case_rows}</tbody>
  </table>
  <h3>Node Latency</h3>
  <table>
    <thead>
      <tr><th>Timeline Metric</th><th>P50</th><th>P95</th><th>Max</th><th>Count</th><th>P95 Bar</th></tr>
    </thead>
    <tbody>{latency_rows}</tbody>
  </table>
  <h3>Decision Reasons</h3>
  <table>
    <thead><tr><th>Reason</th><th>Count</th></tr></thead>
    <tbody>{decision_rows}</tbody>
  </table>
  <h3>Flush Reasons</h3>
  <table>
    <thead><tr><th>Reason</th><th>Count</th></tr></thead>
    <tbody>{flush_rows}</tbody>
  </table>
</div>"""


def _provider_segment_row(values: dict[str, Any]) -> str:
    metric_values = {
        "p50": values.get("p50"),
        "p95": values.get("p95"),
        "max": values.get("max"),
        "count": values.get("count"),
    }
    stage = html.escape(str(values.get("stage", "")))
    label = html.escape(str(values.get("label", values.get("name", ""))))
    return f"""<tr>
  <td><span class="stage">{stage}</span></td>
  <td>{label}</td>
  <td>{_fmt(metric_values.get("p50"))}</td>
  <td>{_fmt(metric_values.get("p95"))}</td>
  <td>{_fmt(metric_values.get("max"))}</td>
  <td>{html.escape(str(metric_values.get("count", "")))}</td>
  <td>{_bar(metric_values.get("p95"), metric_values.get("max"))}</td>
</tr>"""


def _provider_case_row(record: dict[str, Any]) -> str:
    segments = {
        str(segment.get("name")): segment
        for segment in record.get("segments", [])
        if isinstance(segment, dict)
    }

    def segment(name: str) -> str:
        return _fmt(segments.get(name, {}).get("duration_ms"))

    return f"""<tr>
  <td>{html.escape(str(record.get("case_id") or record.get("turn_id") or ""))}</td>
  <td>{html.escape(str(record.get("interrupt_action") or ""))}</td>
  <td>{html.escape(str(record.get("intent") or ""))}</td>
  <td>{segment("stt_first_audio_sent")}</td>
  <td>{segment("stt_provider_first_partial")}</td>
  <td>{segment("stt_livekit_first_interim")}</td>
  <td>{segment("stt_final")}</td>
  <td>{segment("turn_commit")}</td>
  <td>{segment("brain_first_delta")}</td>
  <td>{segment("tts_first_audio")}</td>
  <td>{segment("interrupt_resolution")}</td>
</tr>"""


def _slo_failure_text(result: dict[str, Any]) -> str:
    value = result.get("value")
    if value is None:
        value_label = "missing"
    elif isinstance(value, float):
        value_label = f"{value:.1f}"
    else:
        value_label = str(value)
    tier = result.get("tier", "")
    tier_label = f" [{tier}]" if tier else ""
    return (
        f"{result['runner']}: SLO {result['name']}{tier_label} failed "
        f"({result['metric']}.{result['statistic']}={value_label} "
        f"> {result['max_value']:.1f})"
    )


def _count_row(key: str, value: Any) -> str:
    return f"""<tr>
  <td>{html.escape(str(key))}</td>
  <td>{html.escape(str(value))}</td>
</tr>"""


def _finding_item(item: dict[str, str]) -> str:
    severity = html.escape(item["severity"])
    text = html.escape(item["text"])
    return f"<li class='{severity}'>{text}</li>"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}"
    return html.escape(str(value))


def _bar(value: Any, max_value: Any) -> str:
    width = 0
    if (
        isinstance(value, (int, float))
        and isinstance(max_value, (int, float))
        and max_value > 0
    ):
        width = max(2, min(100, round(value / max_value * 100)))
    return f'<div class="bar-track"><div class="bar" style="width: {width}%"></div></div>'
