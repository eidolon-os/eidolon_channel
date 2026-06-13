"""Validate benchmark expectations against channel turn timeline records."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .schema import ROOM_NAME_PREFIX, BenchmarkSuite, RunResult
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
        result.metrics["timeline_decision_actions"] = ",".join(
            _decision_actions(case_records)
        )
        result.metrics["timeline_decision_intents"] = ",".join(
            _decision_intents(case_records)
        )
        result.metrics.update(_latency_metrics(case_records))
        result.metrics.update(_interrupted_context_metrics(case_records))

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
    decision_actions = _decision_actions(records)
    decision_intents = _decision_intents(records)
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

    decision_action = str(getattr(expected, "decision_action", "") or "")
    if decision_action not in ("", "any"):
        if decision_action not in decision_actions:
            errors.append(
                "timeline expected decision_action="
                f"{decision_action}, got {decision_actions or ['<none>']}"
            )

    decision_intent = str(getattr(expected, "decision_intent", "") or "")
    if decision_intent not in ("", "uncertain"):
        if decision_intent not in decision_intents:
            errors.append(
                "timeline expected decision_intent="
                f"{decision_intent}, got {decision_intents or ['<none>']}"
            )

    voiceprint_expectation = str(getattr(expected, "voiceprint", "any") or "any")
    if voiceprint_expectation not in ("", "any"):
        allowed_values = _voiceprint_allowed_values(records)
        if voiceprint_expectation == "allowed":
            if True not in allowed_values:
                errors.append(
                    "timeline expected voiceprint allowed, got "
                    f"{allowed_values or ['<missing>']}"
                )
        elif voiceprint_expectation == "blocked":
            if False not in allowed_values:
                errors.append(
                    "timeline expected voiceprint blocked, got "
                    f"{allowed_values or ['<missing>']}"
                )
        else:
            errors.append(f"unknown voiceprint expectation={voiceprint_expectation!r}")

    brain_expectation = str(getattr(expected, "brain", "any") or "any")
    if brain_expectation not in ("", "any"):
        has_brain = _has_brain_request(records)
        if brain_expectation == "required" and not has_brain:
            errors.append("timeline expected brain_request_sent_at")
        elif brain_expectation == "forbidden" and has_brain:
            errors.append("timeline expected no brain_request_sent_at")
        elif brain_expectation not in ("required", "forbidden"):
            errors.append(f"unknown brain expectation={brain_expectation!r}")

    min_brain_requests = getattr(expected, "min_brain_requests", None)
    max_brain_requests = getattr(expected, "max_brain_requests", None)
    if min_brain_requests is not None or max_brain_requests is not None:
        brain_requests = _brain_request_count(records)
        if min_brain_requests is not None and brain_requests < min_brain_requests:
            errors.append(
                "timeline expected >="
                f"{min_brain_requests} brain requests, got {brain_requests}"
            )
        if max_brain_requests is not None and brain_requests > max_brain_requests:
            errors.append(
                "timeline expected <="
                f"{max_brain_requests} brain requests, got {brain_requests}"
            )

    max_speech_stop_to_commit_ms = getattr(
        expected, "max_speech_stop_to_commit_ms", None
    )
    if max_speech_stop_to_commit_ms is not None:
        durations = _speech_stop_to_commit_durations_ms(records)
        if durations:
            slow = [
                round(duration, 1)
                for duration in durations
                if duration > max_speech_stop_to_commit_ms
            ]
            if slow:
                errors.append(
                    "timeline speech-stop-to-commit exceeded "
                    f"{max_speech_stop_to_commit_ms}ms: {slow}"
                )
        else:
            errors.append("timeline missing speech-stop-to-commit duration")

    rejected_turn_brain = str(getattr(expected, "rejected_turn_brain", "any") or "any")
    if rejected_turn_brain not in ("", "any"):
        rejected_records = _rejected_turn_records(records)
        rejected_has_brain = _has_brain_request(rejected_records)
        if rejected_turn_brain == "forbidden" and rejected_has_brain:
            errors.append("timeline expected rejected turns to have no brain_request_sent_at")
        elif rejected_turn_brain == "required" and not rejected_has_brain:
            errors.append("timeline expected rejected turns to have brain_request_sent_at")
        elif rejected_turn_brain not in ("required", "forbidden"):
            errors.append(f"unknown rejected_turn_brain expectation={rejected_turn_brain!r}")

    for needle in getattr(expected, "canonical_contains", ()) or ():
        if not _canonical_contains(records, str(needle)):
            errors.append(f"timeline canonical text missing {needle!r}")

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
    prefix = f"{ROOM_NAME_PREFIX}-"
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
        if not _is_resolved_interrupt(record):
            continue
        attrs = record.get("attrs") if isinstance(record.get("attrs"), dict) else {}
        action = attrs.get("interrupt_action")
        if isinstance(action, str) and action:
            values.append(action)
    return values


def _intents(records: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for record in records:
        if not _is_resolved_interrupt(record):
            continue
        attrs = record.get("attrs") if isinstance(record.get("attrs"), dict) else {}
        decision = attrs.get("decision") if isinstance(attrs.get("decision"), dict) else {}
        intent = decision.get("intent")
        if isinstance(intent, str) and intent:
            values.append(intent)
    return values


def _decision_actions(records: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for record in records:
        attrs = record.get("attrs") if isinstance(record.get("attrs"), dict) else {}
        action = attrs.get("interrupt_action")
        if isinstance(action, str) and action:
            values.append(action)
    return values


def _decision_intents(records: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for record in records:
        attrs = record.get("attrs") if isinstance(record.get("attrs"), dict) else {}
        decision = attrs.get("decision") if isinstance(attrs.get("decision"), dict) else {}
        intent = decision.get("intent")
        if isinstance(intent, str) and intent:
            values.append(intent)
    return values


def _voiceprint_allowed_values(records: list[dict[str, Any]]) -> list[bool]:
    values: list[bool] = []
    for record in records:
        attrs = record.get("attrs") if isinstance(record.get("attrs"), dict) else {}
        gate = attrs.get("voiceprint_commit_gate")
        if isinstance(gate, dict) and isinstance(gate.get("allowed"), bool):
            values.append(bool(gate["allowed"]))
            continue
        voiceprint = attrs.get("voiceprint")
        if isinstance(voiceprint, dict) and isinstance(
            voiceprint.get("commit_allowed"), bool
        ):
            values.append(bool(voiceprint["commit_allowed"]))
    return values


def _has_brain_request(records: list[dict[str, Any]]) -> bool:
    return _brain_request_count(records) > 0


def _brain_request_count(records: list[dict[str, Any]]) -> int:
    count = 0
    for record in records:
        timestamps = _mapping(record.get("timestamps"))
        if isinstance(timestamps.get("brain_request_sent_at"), (int, float)):
            count += 1
    return count


def _rejected_turn_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rejected: list[dict[str, Any]] = []
    for record in records:
        attrs = record.get("attrs") if isinstance(record.get("attrs"), dict) else {}
        coordinator = attrs.get("user_turn_coordinator")
        if isinstance(coordinator, dict) and coordinator.get("state") == "rejected":
            rejected.append(record)
            continue
        flush_reason = attrs.get("timeline_flush_reason")
        if isinstance(flush_reason, str) and flush_reason.startswith(
            "non_semantic_completed_turn:"
        ):
            rejected.append(record)
    return rejected


def _canonical_contains(records: list[dict[str, Any]], needle: str) -> bool:
    if not needle:
        return True
    for record in records:
        attrs = record.get("attrs") if isinstance(record.get("attrs"), dict) else {}
        candidates: list[str] = []
        for key in (
            "canonical_user_text",
            "framework_completed_turn",
            "user_turn_coordinator",
            "stt_stream",
        ):
            value = attrs.get(key)
            if not isinstance(value, dict):
                continue
            for text_key in (
                "text_preview",
                "selected_text_preview",
                "committed_transcript_preview",
                "last_text_preview",
            ):
                text = value.get(text_key)
                if isinstance(text, str) and text:
                    candidates.append(text)
        if any(needle in text for text in candidates):
            return True
    return False


def _is_resolved_interrupt(record: dict[str, Any]) -> bool:
    timestamps = record.get("timestamps")
    if isinstance(timestamps, dict) and isinstance(
        timestamps.get("interrupt_resolved_at"),
        (int, float),
    ):
        return True
    durations = record.get("durations_ms")
    if isinstance(durations, dict) and isinstance(
        durations.get("vad_start_to_interrupt_resolved"),
        (int, float),
    ):
        return True
    attrs = record.get("attrs")
    if not isinstance(attrs, dict):
        return False
    provider_latency = attrs.get("provider_latency_ms")
    return isinstance(provider_latency, dict) and isinstance(
        provider_latency.get("interrupt_speech_to_resolved_ms"),
        (int, float),
    )


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


def _speech_stop_to_commit_durations_ms(records: list[dict[str, Any]]) -> list[float]:
    """End-of-turn wait per committed turn: user stopped speaking -> commit."""

    durations: list[float] = []
    for record in records:
        provider_latency = _mapping(_mapping(record.get("attrs")).get("provider_latency_ms"))
        duration = _number(provider_latency.get("speech_stop_to_commit_ms"))
        if duration is None:
            duration = _number(_mapping(record.get("durations_ms")).get("speech_stop_to_commit"))
        if duration is None:
            timestamps = _mapping(record.get("timestamps"))
            stop = _number(timestamps.get("speech_stopped_at"))
            commit = _number(timestamps.get("turn_committed_at"))
            if stop is not None and commit is not None:
                duration = max(0.0, (commit - stop) * 1000.0)
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


def _latency_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    """Expose timeline latency segments as per-case benchmark metrics.

    A case should usually have one timeline record. When retries or residual
    speech produce more, keep the maximum sample for each segment so the report
    surfaces the slowest user-visible path instead of averaging it away.
    """

    samples: dict[str, list[float]] = {}
    for record in records:
        durations = _mapping(record.get("durations_ms"))
        attrs = _mapping(record.get("attrs"))
        provider_latency = _mapping(attrs.get("provider_latency_ms"))
        for key, value in provider_latency.items():
            number = _number(value)
            if number is not None:
                samples.setdefault(f"timeline_{key}", []).append(number)
        decision = _mapping(attrs.get("decision"))
        hold_recheck_ms = _number(decision.get("hold_recheck_ms"))
        if hold_recheck_ms is not None:
            samples.setdefault("timeline_decision_hold_recheck_ms", []).append(
                hold_recheck_ms
            )
        total_interrupt = _number(durations.get("vad_start_to_interrupt_resolved"))
        if total_interrupt is not None:
            samples.setdefault(
                "timeline_vad_start_to_interrupt_resolved",
                [],
            ).append(total_interrupt)
    return {
        key: max(values)
        for key, values in sorted(samples.items())
        if values
    }


def _interrupted_context_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Expose interrupted-context capture so reports can diagnose repetition."""

    contexts: list[dict[str, Any]] = []
    for record in records:
        attrs = _mapping(record.get("attrs"))
        context = _mapping(attrs.get("interrupted_context"))
        if context:
            contexts.append(context)
    if not contexts:
        return {}

    latest = contexts[-1]
    metrics: dict[str, Any] = {
        "timeline_interrupted_context_count": len(contexts),
    }
    source = latest.get("source")
    if isinstance(source, str) and source:
        metrics["timeline_interrupted_context_source"] = source
    played_seconds = _number(latest.get("played_seconds"))
    if played_seconds is not None:
        metrics["timeline_interrupted_context_played_seconds"] = played_seconds
    preview = latest.get("text_preview")
    if isinstance(preview, str) and preview:
        metrics["timeline_interrupted_context_preview"] = preview
    return metrics


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
