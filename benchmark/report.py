"""Benchmark metric aggregation and report rendering."""

from __future__ import annotations

import html
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .schema import CaseResult, RunResult


INTERRUPT_FOCUS_METRICS: tuple[tuple[str, str], ...] = (
    (
        "timeline_interrupt_speech_to_started_ms",
        "VAD 起声 -> duck/手动打断开始",
    ),
    (
        "timeline_interrupt_speech_to_first_transcript_ms",
        "VAD 起声 -> 首次转写",
    ),
    (
        "timeline_stt_speech_to_actionable_transcript_ms",
        "VAD 起声 -> 首个可行动转写",
    ),
    (
        "timeline_stt_first_transcript_to_actionable_transcript_ms",
        "首次转写 -> 首个可行动转写",
    ),
    (
        "timeline_interrupt_first_transcript_to_intent_admitted_ms",
        "首次转写 -> 直接意图通过",
    ),
    (
        "timeline_interrupt_first_transcript_to_resolved_ms",
        "首次转写 -> cancel/rollback 完成",
    ),
    (
        "timeline_interrupt_first_transcript_to_cancel_resolved_ms",
        "首次转写 -> cancel 完成",
    ),
    (
        "timeline_decision_hold_recheck_ms",
        "策略 HOLD -> 下次 recheck 预算",
    ),
    (
        "timeline_interrupt_intent_admitted_to_resolved_ms",
        "直接意图通过 -> cancel 完成",
    ),
    (
        "timeline_interrupt_intent_admitted_to_cancel_resolved_ms",
        "直接意图通过 -> cancel 完成(精确)",
    ),
    (
        "timeline_interrupt_started_to_resolved_ms",
        "duck/手动打断开始 -> 完成",
    ),
    (
        "timeline_interrupt_speech_to_resolved_ms",
        "VAD 起声 -> 完成",
    ),
    (
        "timeline_interrupt_speech_to_cancel_resolved_ms",
        "VAD 起声 -> cancel 完成",
    ),
    (
        "timeline_interrupt_speech_to_rollback_resolved_ms",
        "VAD 起声 -> rollback 完成",
    ),
)

CASE_FOCUS_METRICS: tuple[tuple[str, str], ...] = (
    ("elapsed_ms", "用例总耗时"),
    ("interrupt_decision_ms", "策略决策耗时"),
    ("timeline_vad_start_to_interrupt_resolved", "VAD 起声 -> 打断完成"),
    (
        "timeline_vad_start_to_interrupt_cancel_resolved",
        "VAD 起声 -> cancel 完成",
    ),
    (
        "timeline_vad_start_to_interrupt_rollback_resolved",
        "VAD 起声 -> rollback 完成",
    ),
    ("timeline_interrupt_speech_to_started_ms", "VAD 起声 -> duck/手动打断开始"),
    ("timeline_interrupt_speech_to_first_transcript_ms", "VAD 起声 -> 首次转写"),
    ("timeline_stt_speech_to_actionable_transcript_ms", "VAD 起声 -> 首个可行动转写"),
    (
        "timeline_stt_first_transcript_to_actionable_transcript_ms",
        "首次转写 -> 首个可行动转写",
    ),
    (
        "timeline_interrupt_first_transcript_to_intent_admitted_ms",
        "首次转写 -> 直接意图通过",
    ),
    (
        "timeline_interrupt_first_transcript_to_resolved_ms",
        "首次转写 -> cancel/rollback 完成",
    ),
    (
        "timeline_interrupt_first_transcript_to_cancel_resolved_ms",
        "首次转写 -> cancel 完成",
    ),
    (
        "timeline_decision_hold_recheck_ms",
        "策略 HOLD -> 下次 recheck 预算",
    ),
    (
        "timeline_interrupt_intent_admitted_to_resolved_ms",
        "直接意图通过 -> cancel 完成",
    ),
    (
        "timeline_interrupt_intent_admitted_to_cancel_resolved_ms",
        "直接意图通过 -> cancel 完成(精确)",
    ),
    ("timeline_transcript_admission_event_count", "transcript admission 事件数"),
    ("timeline_transcript_admission_last_reason", "transcript admission 最后原因"),
    (
        "timeline_transcript_admission_last_preview",
        "transcript admission 最后文本",
    ),
    (
        "timeline_transcript_admission_rejected_chain",
        "transcript admission 拒绝链",
    ),
    ("timeline_semantic_gate_event_count", "semantic gate 事件数"),
    ("timeline_semantic_gate_last_reason", "semantic gate 最后原因"),
    ("timeline_semantic_gate_last_preview", "semantic gate 最后文本"),
    ("timeline_semantic_gate_blocked_chain", "semantic gate 阻断链"),
    ("timeline_interrupt_speech_to_resolved_ms", "VAD 起声 -> 完成"),
    ("timeline_interrupt_speech_to_cancel_resolved_ms", "VAD 起声 -> cancel 完成"),
    (
        "timeline_interrupt_speech_to_rollback_resolved_ms",
        "VAD 起声 -> rollback 完成",
    ),
    ("timeline_record_count", "timeline 记录数"),
    ("real_call_verified", "真实调用校验"),
    ("user_done_to_agent_audio_after_user_done_ms", "用户音频结束 -> 下一段 agent 音频"),
    ("publish_to_agent_audio_first_ms", "发布音频 -> 首段 agent 音频"),
    ("timeline_commit_to_tts_first_audio_ms", "commit -> TTS 首音频"),
)


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
        "functional_outcome_failed": _false_metric_count(
            cases,
            "functional_outcome_passed",
        ),
        "experience_slo_failed": _false_metric_count(
            cases,
            "experience_slo_passed",
        ),
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
        "functional_outcome_failed": _false_metric_count(
            all_cases,
            "functional_outcome_passed",
        ),
        "experience_slo_failed": _false_metric_count(
            all_cases,
            "experience_slo_passed",
        ),
        "metrics": _metric_distribution(all_cases),
        "per_case": per_case,
    }


def _false_metric_count(cases: list[CaseResult], key: str) -> int:
    return sum(1 for case in cases if case.metrics.get(key) is False)


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
        f"# 语音基准测试报告：{run['run_id']}",
        "",
        f"- 运行器：`{run['runner']}`",
        f"- 配置画像：`{run['profile']}`",
        f"- Git 提交：`{run['git_sha']}`",
        f"- 通过率：`{summary['passed']}/{summary['total']}`",
    ]
    if repeats:
        lines.append(f"- 重复次数：`{repeats}`")
        if summary.get("flaky"):
            lines.append(f"- 不稳定用例数：`{summary['flaky']}`")
    if summary.get("functional_outcome_failed") is not None:
        lines.append(
            f"- 功能 outcome 失败样本：`{summary['functional_outcome_failed']}`"
        )
    if summary.get("experience_slo_failed") is not None:
        lines.append(f"- 体验 SLO 失败样本：`{summary['experience_slo_failed']}`")
    lines.extend(
        [
            "",
            "## 指标汇总",
            "",
            "| 指标 | p50 | p95 | 最大值 | 标准差 | 样本数 |",
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
    interrupt_lines = _interrupt_focus_lines(summary["metrics"])
    if interrupt_lines:
        lines.extend(
            [
                "",
                "## 打断延迟拆解",
                "",
                "| 分段 | 指标 | p50 | p95 | 最大值 | 样本数 |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
                *interrupt_lines,
            ]
        )
    per_case = summary.get("per_case")
    if per_case:
        lines.extend(
            [
                "",
                "## 用例稳定性",
                "",
                "| 用例 | 通过次数 | 运行次数 | 通过率 |",
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
    lines.extend(["", "## 用例详情", ""])
    for case in payload["cases"]:
        status = "通过" if case["passed"] else "失败"
        lines.append(f"### {case['case_id']} - {status}")
        if case["errors"]:
            lines.append("")
            lines.extend(f"- 错误：{err}" for err in case["errors"])
        lines.extend(_case_detail_lines(case))
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
  <title>语音基准测试 {html.escape(payload['run']['run_id'])}</title>
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


def _interrupt_focus_lines(metrics: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key, label in INTERRUPT_FOCUS_METRICS:
        values = metrics.get(key)
        if not isinstance(values, dict) or not values.get("count"):
            continue
        lines.append(
            "| {label} | `{key}` | {p50} | {p95} | {maxv} | {count} |".format(
                label=label,
                key=key,
                p50=_fmt(values.get("p50")),
                p95=_fmt(values.get("p95")),
                maxv=_fmt(values.get("max")),
                count=values.get("count"),
            )
        )
    return lines


def _case_detail_lines(case: dict[str, Any]) -> list[str]:
    metrics = _mapping(case.get("metrics"))
    lines: list[str] = []
    overview = _case_overview_rows(case, metrics)
    if overview:
        lines.extend(["", "#### 判定摘要", "", "| 项目 | 值 |", "| --- | --- |"])
        lines.extend(f"| {key} | {value} |" for key, value in overview)
    focus = _case_focus_metric_rows(metrics)
    if focus:
        lines.extend(["", "#### 关键指标", "", "| 指标 | 值 |", "| --- | ---: |"])
        lines.extend(f"| {label} | {value} |" for label, value in focus)
    decision_lines = _case_decision_lines(case)
    if decision_lines:
        lines.extend(
            [
                "",
                "#### 决策路径",
                "",
                "| interim | attention | decision | intent | reason |",
                "| --- | --- | --- | --- | --- |",
                *decision_lines,
            ]
        )
    diagnosis = _case_diagnosis(case, metrics)
    if diagnosis:
        lines.extend(["", "#### 诊断", "", diagnosis])
    return lines


def _case_overview_rows(
    case: dict[str, Any],
    metrics: dict[str, Any],
) -> list[tuple[str, str]]:
    rows = [("结果", "通过" if case.get("passed") else "失败")]
    for key, label in (
        ("expected_action", "期望动作"),
        ("actual_action", "实际动作"),
        ("timeline_actions", "timeline 动作"),
        ("timeline_decision_actions", "timeline 决策动作"),
        ("expected_intent", "期望意图"),
        ("actual_intent", "实际意图"),
        ("timeline_intents", "timeline 意图"),
        ("timeline_decision_intents", "timeline 决策意图"),
        ("forbid_actions", "禁止动作"),
        ("timeline_interrupted_context_count", "被打断上下文捕获次数"),
        ("timeline_interrupted_context_source", "被打断上下文来源"),
        ("timeline_interrupted_context_played_seconds", "用户已听时长"),
        ("timeline_interrupted_context_preview", "被打断内容预览"),
    ):
        value = metrics.get(key)
        if value not in (None, ""):
            rows.append((label, _fmt_metric_value(value)))
    return rows


def _case_focus_metric_rows(metrics: dict[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for key, label in CASE_FOCUS_METRICS:
        value = metrics.get(key)
        if value is None:
            continue
        rows.append((f"{label} (`{key}`)", _fmt_metric_value(value)))
    return rows


def _case_decision_lines(case: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    decisions = case.get("decisions")
    if not isinstance(decisions, list):
        return rows
    for item in decisions[:8]:
        if not isinstance(item, dict):
            continue
        interim = _escape_cell(str(item.get("interim_text") or ""))
        attention = _mapping(item.get("attention_admission"))
        attention_text = _escape_cell(
            _compact_parts(
                attention.get("action"),
                attention.get("reason"),
            )
        )
        decision = _mapping(item.get("decision"))
        decision_text = _escape_cell(str(decision.get("action") or "-"))
        intent = _escape_cell(str(decision.get("intent") or "-"))
        reason = _escape_cell(str(decision.get("reason") or "-"))
        rows.append(
            f"| {interim or '-'} | {attention_text or '-'} | "
            f"{decision_text} | {intent} | {reason} |"
        )
    if len(decisions) > 8:
        rows.append(f"| ... | ... | ... | ... | 还有 {len(decisions) - 8} 条 |")
    return rows


def _case_diagnosis(case: dict[str, Any], metrics: dict[str, Any]) -> str:
    if case.get("errors"):
        return "该用例失败，优先查看上方错误和原始 metrics。"
    action = str(metrics.get("timeline_actions") or metrics.get("actual_action") or "")
    intent = str(metrics.get("timeline_intents") or metrics.get("actual_intent") or "")
    if "cancel" in action:
        total = _fmt_metric_value(
            _first_metric(
                metrics,
                (
                    "timeline_vad_start_to_interrupt_cancel_resolved",
                    "timeline_interrupt_speech_to_cancel_resolved_ms",
                    "timeline_vad_start_to_interrupt_resolved",
                    "timeline_interrupt_speech_to_resolved_ms",
                    "interrupt_decision_ms",
                ),
            )
        )
        first_transcript = metrics.get("timeline_interrupt_speech_to_first_transcript_ms")
        actionable_transcript = metrics.get(
            "timeline_stt_speech_to_actionable_transcript_ms"
        )
        first_to_actionable = metrics.get(
            "timeline_stt_first_transcript_to_actionable_transcript_ms"
        )
        hold_recheck = metrics.get("timeline_decision_hold_recheck_ms")
        after_transcript = _first_metric(
            metrics,
            (
                "timeline_interrupt_first_transcript_to_cancel_resolved_ms",
                "timeline_interrupt_first_transcript_to_resolved_ms",
            ),
        )
        if (
            first_transcript is not None
            and actionable_transcript is not None
            and after_transcript is not None
        ):
            extra = ""
            if first_to_actionable is not None:
                extra = (
                    "，首次转写到可行动转写约 "
                    f"{_fmt_metric_value(first_to_actionable)}"
                )
            if hold_recheck is not None:
                extra += (
                    "，策略 HOLD recheck 预算约 "
                    f"{_fmt_metric_value(hold_recheck)}"
                )
            return (
                f"该用例完成 `{intent or 'unknown'}` 打断，总耗时约 {total}；"
                f"其中首次转写约 {_fmt_metric_value(first_transcript)}，"
                f"首个可行动转写约 {_fmt_metric_value(actionable_transcript)}"
                f"{extra}，转写后到完成约 {_fmt_metric_value(after_transcript)}。"
            )
        if first_transcript is not None and after_transcript is not None:
            return (
                f"该用例完成 `{intent or 'unknown'}` 打断，总耗时约 {total}；"
                f"其中首次转写约 {_fmt_metric_value(first_transcript)}，"
                f"转写后到完成约 {_fmt_metric_value(after_transcript)}。"
            )
        return f"该用例完成 `{intent or 'unknown'}` 打断，总耗时约 {total}。"
    if "rollback" in action:
        total = _fmt_metric_value(
            _first_metric(
                metrics,
                (
                    "timeline_vad_start_to_interrupt_rollback_resolved",
                    "timeline_interrupt_speech_to_rollback_resolved_ms",
                    "timeline_vad_start_to_interrupt_resolved",
                    "timeline_interrupt_speech_to_resolved_ms",
                ),
            )
        )
        return f"该用例被识别为短反馈/噪声路径，执行 rollback，完成耗时约 {total}。"
    if metrics.get("real_call_verified") is True:
        return "该用例通过真实调用校验，且未触发取消路径。"
    return ""


def _first_metric(metrics: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return value
    return None


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _compact_parts(*parts: Any) -> str:
    return " / ".join(str(part) for part in parts if part not in (None, ""))


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _fmt_metric_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)
