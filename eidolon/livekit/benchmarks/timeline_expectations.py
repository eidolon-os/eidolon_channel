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
        case_records = by_case.get(result.case_id, [])
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
    elif expected.action and expected.action not in actions:
        errors.append(
            f"timeline expected action={expected.action}, got {actions or ['<none>']}"
        )

    if expected.intent not in ("", "uncertain"):
        if expected.intent not in intents:
            errors.append(
                f"timeline expected intent={expected.intent}, got {intents or ['<none>']}"
            )

    if expected.topic_switch_hint and not _any_decision_flag(records, "topic_switch_hint"):
        errors.append("timeline expected topic_switch_hint=True")
    if expected.correction_hint and not _any_decision_flag(records, "correction_hint"):
        errors.append("timeline expected correction_hint=True")

    if not records:
        errors.append(f"timeline missing for case={case_id}")
    return errors


def _records_by_case(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        case_id = _case_id(record)
        if case_id:
            grouped[case_id].append(record)
    return dict(grouped)


def _case_id(record: dict[str, Any]) -> str:
    attrs = record.get("attrs") if isinstance(record.get("attrs"), dict) else {}
    room_name = attrs.get("room_name")
    if not isinstance(room_name, str):
        return ""
    prefix = "voice-bench-"
    if not room_name.startswith(prefix):
        return ""
    rest = room_name[len(prefix) :]
    case_id, _, suffix = rest.rpartition("-")
    return case_id if case_id and len(suffix) == 8 else ""


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


def _any_decision_flag(records: list[dict[str, Any]], key: str) -> bool:
    for record in records:
        attrs = record.get("attrs") if isinstance(record.get("attrs"), dict) else {}
        decision = attrs.get("decision") if isinstance(attrs.get("decision"), dict) else {}
        if bool(decision.get(key)):
            return True
    return False
