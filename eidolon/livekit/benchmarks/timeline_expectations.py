"""Validate benchmark expectations against channel turn timeline records."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .schema import BenchmarkSuite, RunResult
from .timeline import load_timeline_records


def apply_timeline_expectations(
    run: RunResult,
    suites: list[BenchmarkSuite],
    timeline_path: Path,
) -> None:
    """Annotate livekit_room case results with timeline-derived failures."""

    records = load_timeline_records(timeline_path)
    by_case = _records_by_case(records)
    expectations = {
        case.case_id: case.expectations
        for suite in suites
        for case in suite.cases
    }

    for result in run.cases:
        case_records = _records_for_result(
            result_case_id=result.case_id,
            result_room_name=result.metrics.get("room_name"),
            by_case=by_case,
        )
        result.metrics["timeline_record_count"] = len(case_records)
        result.metrics["timeline_actions"] = ",".join(_actions(case_records))
        result.metrics["timeline_intents"] = ",".join(_intents(case_records))

        expected = expectations.get(result.case_id)
        if expected is None:
            continue

        errors = _expectation_errors(result.case_id, expected, case_records)
        result.errors.extend(errors)
        result.passed = result.passed and not errors


def _expectation_errors(case_id: str, expected: Any, records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    actions = _actions(records)
    intents = _intents(records)
    allowed_attention = _allowed_attention_actions(expected, records)

    forbidden = set(expected.forbid_actions)
    forbidden_seen = sorted(forbidden.intersection(actions))
    if forbidden_seen:
        errors.append(f"timeline saw forbidden actions={forbidden_seen}")

    if expected.action in ("", "any"):
        pass
    elif expected.action == "none":
        unexpected = [action for action in actions if action in {"cancel", "rollback"}]
        if unexpected:
            errors.append(
                f"timeline expected no interrupt action, got {sorted(set(unexpected))}"
            )
    elif (
        expected.action
        and expected.action not in actions
        and not allowed_attention
    ):
        errors.append(
            f"timeline expected action={expected.action}, got {actions or ['<none>']}"
        )

    if expected.intent not in ("", "uncertain") and not allowed_attention:
        if expected.intent not in intents:
            errors.append(
                f"timeline expected intent={expected.intent}, got {intents or ['<none>']}"
            )

    if expected.topic_switch_hint and not _any_decision_flag(records, "topic_switch_hint"):
        errors.append("timeline expected topic_switch_hint=True")
    if expected.correction_hint and not _any_decision_flag(records, "correction_hint"):
        errors.append("timeline expected correction_hint=True")

    if expected.max_interrupt_decision_ms is not None:
        durations = _interrupt_decision_durations_ms(records)
        if durations:
            slow = [
                round(duration, 1)
                for duration in durations
                if duration > expected.max_interrupt_decision_ms
            ]
            if slow:
                errors.append(
                    "timeline interrupt decision exceeded "
                    f"{expected.max_interrupt_decision_ms}ms: {slow}"
                )
        elif expected.action not in ("", "any", "none"):
            errors.append("timeline missing interrupt decision duration")

    if expected.max_interrupt_resolution_after_started_ms is not None:
        durations = _interrupt_resolution_after_started_ms(records)
        if durations:
            slow = [
                round(duration, 1)
                for duration in durations
                if duration > expected.max_interrupt_resolution_after_started_ms
            ]
            if slow:
                errors.append(
                    "timeline interrupt resolution-after-start exceeded "
                    f"{expected.max_interrupt_resolution_after_started_ms}ms: {slow}"
                )
        elif expected.action not in ("", "any", "none"):
            errors.append("timeline missing interrupt started-to-resolved duration")

    if not records:
        errors.append(f"timeline missing for case={case_id}")
    return errors


def _records_for_result(
    *,
    result_case_id: str,
    result_room_name: Any,
    by_case: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    records = by_case.get(result_case_id, [])
    if not isinstance(result_room_name, str) or not result_room_name:
        return records
    exact = [
        record
        for record in records
        if _room_name(record) == result_room_name
    ]
    return exact if exact else records


def _allowed_attention_actions(
    expected: Any,
    records: list[dict[str, Any]],
) -> list[str]:
    allowed = set(getattr(expected, "allow_attention_actions", ()) or ())
    if not allowed:
        return []
    seen = sorted(allowed.intersection(_attention_actions(records)))
    if not seen:
        return []
    interrupt_actions = set(_actions(records)).intersection({"cancel", "rollback"})
    return [] if interrupt_actions else seen


def _records_by_case(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        case_id = _case_id(record)
        if case_id:
            grouped[case_id].append(record)
    return dict(grouped)


def _case_id(record: dict[str, Any]) -> str:
    room_name = _room_name(record)
    if not room_name:
        return ""
    prefix = "voice-bench-"
    if not room_name.startswith(prefix):
        return ""
    rest = room_name[len(prefix) :]
    case_id, _, suffix = rest.rpartition("-")
    return case_id if case_id and len(suffix) == 8 else ""


def _room_name(record: dict[str, Any]) -> str:
    attrs = record.get("attrs") if isinstance(record.get("attrs"), dict) else {}
    room_name = attrs.get("room_name")
    if not isinstance(room_name, str):
        return ""
    return room_name


def _actions(records: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for record in records:
        attrs = record.get("attrs") if isinstance(record.get("attrs"), dict) else {}
        action = attrs.get("interrupt_action")
        if isinstance(action, str) and action:
            values.append(action)
    return values


def _intents(records: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for record in records:
        attrs = record.get("attrs") if isinstance(record.get("attrs"), dict) else {}
        decision = attrs.get("decision") if isinstance(attrs.get("decision"), dict) else {}
        intent = decision.get("intent")
        if isinstance(intent, str) and intent:
            values.append(intent)
    return values


def _attention_actions(records: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for record in records:
        attrs = record.get("attrs") if isinstance(record.get("attrs"), dict) else {}
        admissions = attrs.get("attention_admission_events")
        if isinstance(admissions, list):
            for item in admissions:
                if not isinstance(item, dict):
                    continue
                action = item.get("action")
                if isinstance(action, str) and action:
                    values.append(action)
        admission = attrs.get("attention_admission")
        if isinstance(admission, dict):
            action = admission.get("action")
            if isinstance(action, str) and action:
                values.append(action)
    return values


def _any_decision_flag(records: list[dict[str, Any]], key: str) -> bool:
    for record in records:
        attrs = record.get("attrs") if isinstance(record.get("attrs"), dict) else {}
        decision = attrs.get("decision") if isinstance(attrs.get("decision"), dict) else {}
        if bool(decision.get(key)):
            return True
    return False


def _interrupt_decision_durations_ms(records: list[dict[str, Any]]) -> list[float]:
    durations: list[float] = []
    for record in records:
        duration = _number(
            (
                record.get("durations_ms")
                if isinstance(record.get("durations_ms"), dict)
                else {}
            ).get("vad_start_to_interrupt_resolved")
        )
        if duration is None:
            timestamps = (
                record.get("timestamps")
                if isinstance(record.get("timestamps"), dict)
                else {}
            )
            start = _number(timestamps.get("speech_started_at"))
            end = _number(timestamps.get("interrupt_resolved_at"))
            if end is None:
                end = _number(timestamps.get("turn_committed_at"))
            if start is not None and end is not None:
                duration = max(0.0, (end - start) * 1000.0)
        if duration is not None:
            durations.append(duration)
    return durations


def _interrupt_resolution_after_started_ms(records: list[dict[str, Any]]) -> list[float]:
    durations: list[float] = []
    for record in records:
        timestamps = (
            record.get("timestamps")
            if isinstance(record.get("timestamps"), dict)
            else {}
        )
        start = _number(timestamps.get("interrupt_intent_admitted_at"))
        if start is None:
            start = _number(timestamps.get("interrupt_started_at"))
        end = _number(timestamps.get("interrupt_resolved_at"))
        if start is not None and end is not None:
            durations.append(max(0.0, (end - start) * 1000.0))
    return durations


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
