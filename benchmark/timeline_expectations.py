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
        result.metrics.update(_decision_event_metrics(case_records))
        result.metrics.update(_latency_metrics(case_records))
        result.metrics.update(_interrupted_context_metrics(case_records))
        result.metrics.update(_transcript_admission_metrics(case_records))
        result.metrics.update(_attention_admission_metrics(case_records))
        result.metrics.update(_semantic_gate_metrics(case_records))
        result.metrics.update(_framework_completed_gate_metrics(case_records))
        result.metrics.update(_interruption_owner_metrics(case_records))

        expected = expectations.get(result.case_id)
        if expected is None:
            continue

        prior_errors = list(result.errors)
        errors = _expectation_errors(result.case_id, expected, case_records)
        functional_errors, experience_errors = _split_expectation_errors(errors)
        all_functional_errors = [*prior_errors, *functional_errors]
        result.metrics["functional_outcome_passed"] = not all_functional_errors
        result.metrics["experience_slo_passed"] = not experience_errors
        result.metrics["functional_outcome_errors"] = "\n".join(all_functional_errors)
        result.metrics["experience_slo_errors"] = "\n".join(experience_errors)
        result.errors.extend(errors)
        result.passed = result.passed and not errors


def _split_expectation_errors(errors: list[str]) -> tuple[list[str], list[str]]:
    functional: list[str] = []
    experience: list[str] = []
    for error in errors:
        if _is_experience_slo_error(error):
            experience.append(error)
        else:
            functional.append(error)
    return functional, experience


def _is_experience_slo_error(error: str) -> bool:
    if " exceeded " in error or "too slow" in error:
        return True
    return error in {
        "timeline missing speech-stop-to-commit duration",
        "timeline missing interrupt decision duration",
        "timeline missing interrupt started-to-resolved duration",
        "timeline missing speech-start-to-suspend duration",
        "timeline missing speech-start-to-cancel duration",
        "timeline missing speech-start-to-resume duration",
    }


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
    if decision_action == "none":
        if decision_actions:
            errors.append(
                "timeline expected no decision_action, got "
                f"{decision_actions or ['<none>']}"
            )
    elif decision_action not in ("", "any"):
        if decision_action not in decision_actions:
            errors.append(
                "timeline expected decision_action="
                f"{decision_action}, got {decision_actions or ['<none>']}"
            )

    decision_intent = str(getattr(expected, "decision_intent", "") or "")
    if decision_action == "none":
        pass
    elif decision_intent not in ("", "uncertain"):
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
        durations = _interrupt_decision_durations_ms(records, expected.action)
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
        durations = _interrupt_resolution_after_started_ms(records, expected.action)
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

    max_speech_start_to_suspend_ms = getattr(
        expected,
        "max_speech_start_to_suspend_ms",
        None,
    )
    if max_speech_start_to_suspend_ms is not None:
        durations = _speech_start_to_suspend_durations_ms(records)
        if durations:
            slow = [
                round(duration, 1)
                for duration in durations
                if duration > max_speech_start_to_suspend_ms
            ]
            if slow:
                errors.append(
                    "timeline speech-start-to-suspend exceeded "
                    f"{max_speech_start_to_suspend_ms}ms: {slow}"
                )
        else:
            errors.append("timeline missing speech-start-to-suspend duration")

    max_speech_start_to_cancel_ms = getattr(
        expected,
        "max_speech_start_to_cancel_ms",
        None,
    )
    if max_speech_start_to_cancel_ms is not None:
        durations = _speech_start_to_cancel_durations_ms(records)
        if durations:
            slow = [
                round(duration, 1)
                for duration in durations
                if duration > max_speech_start_to_cancel_ms
            ]
            if slow:
                errors.append(
                    "timeline speech-start-to-cancel exceeded "
                    f"{max_speech_start_to_cancel_ms}ms: {slow}"
                )
        else:
            errors.append("timeline missing speech-start-to-cancel duration")

    max_speech_start_to_resume_ms = getattr(
        expected,
        "max_speech_start_to_resume_ms",
        None,
    )
    if max_speech_start_to_resume_ms is not None:
        durations = _speech_start_to_resume_durations_ms(records)
        if durations:
            slow = [
                round(duration, 1)
                for duration in durations
                if duration > max_speech_start_to_resume_ms
            ]
            if slow:
                errors.append(
                    "timeline speech-start-to-resume exceeded "
                    f"{max_speech_start_to_resume_ms}ms: {slow}"
                )
        else:
            errors.append("timeline missing speech-start-to-resume duration")

    playback_stop_sent = getattr(expected, "playback_stop_sent", None)
    if playback_stop_sent is not None:
        saw_stop = _client_control_sent(records, "playback.stop")
        if playback_stop_sent and not saw_stop:
            errors.append("timeline expected playback.stop client control")
        elif not playback_stop_sent and saw_stop:
            errors.append("timeline expected no playback.stop client control")

    ptt_terminal_action = str(getattr(expected, "ptt_terminal_action", "") or "")
    if ptt_terminal_action:
        actions = _ptt_terminal_actions(records)
        if ptt_terminal_action not in actions:
            errors.append(
                "timeline expected PTT terminal action="
                f"{ptt_terminal_action}, got {actions or ['<none>']}"
            )

    ptt_terminal_reason = str(getattr(expected, "ptt_terminal_reason", "") or "")
    if ptt_terminal_reason:
        reasons = _ptt_terminal_reasons(records)
        if ptt_terminal_reason not in reasons:
            errors.append(
                "timeline expected PTT terminal reason="
                f"{ptt_terminal_reason}, got {reasons or ['<none>']}"
            )

    if bool(getattr(expected, "no_full_assistant_context_commit", False)):
        if _has_cancel(records) and not _has_interrupted_context(records):
            errors.append(
                "timeline expected interrupted_context for truncated assistant reply"
            )

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
    if isinstance(timestamps, dict):
        for key in (
            "interrupt_resolved_at",
            "interrupt_cancel_resolved_at",
            "interrupt_rollback_resolved_at",
        ):
            if isinstance(timestamps.get(key), (int, float)):
                return True
    durations = record.get("durations_ms")
    if isinstance(durations, dict):
        for key in (
            "vad_start_to_interrupt_resolved",
            "vad_start_to_interrupt_cancel_resolved",
            "vad_start_to_interrupt_rollback_resolved",
        ):
            if isinstance(durations.get(key), (int, float)):
                return True
    attrs = record.get("attrs")
    if not isinstance(attrs, dict):
        return False
    provider_latency = attrs.get("provider_latency_ms")
    if not isinstance(provider_latency, dict):
        return False
    for key in (
        "interrupt_speech_to_resolved_ms",
        "interrupt_speech_to_cancel_resolved_ms",
        "interrupt_speech_to_rollback_resolved_ms",
    ):
        if isinstance(provider_latency.get(key), (int, float)):
            return True
    return False


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


def _interrupt_decision_durations_ms(
    records: list[dict[str, Any]],
    expected_action: str = "",
) -> list[float]:
    durations: list[float] = []
    for record in records:
        duration = _speech_start_to_action_resolved_ms(record, expected_action)
        if duration is None:
            timestamps = (
                record.get("timestamps")
                if isinstance(record.get("timestamps"), dict)
                else {}
            )
            start = _number(timestamps.get("speech_started_at"))
            end = _first_number(
                timestamps.get(key)
                for key in _resolved_timestamp_keys(expected_action)
            )
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


def _interrupt_resolution_after_started_ms(
    records: list[dict[str, Any]],
    expected_action: str = "",
) -> list[float]:
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
        end = _first_number(
            timestamps.get(key)
            for key in _resolved_timestamp_keys(expected_action)
        )
        if start is not None and end is not None:
            durations.append(max(0.0, (end - start) * 1000.0))
    return durations


def _speech_start_to_suspend_durations_ms(records: list[dict[str, Any]]) -> list[float]:
    durations: list[float] = []
    for record in records:
        duration = _timestamp_delta_ms(
            record,
            "speech_started_at",
            "interrupt_started_at",
        )
        if duration is None:
            duration = _duck_started_duration_ms(record)
        if duration is not None:
            durations.append(duration)
    return durations


def _speech_start_to_cancel_durations_ms(records: list[dict[str, Any]]) -> list[float]:
    durations: list[float] = []
    for record in records:
        if not _record_is_cancel(record):
            continue
        duration = _speech_start_to_action_resolved_ms(record, "cancel")
        if duration is not None:
            durations.append(duration)
    return durations


def _speech_start_to_resume_durations_ms(records: list[dict[str, Any]]) -> list[float]:
    durations: list[float] = []
    for record in records:
        if not _record_is_resume(record):
            continue
        duration = _speech_start_to_action_resolved_ms(record, "rollback")
        if duration is not None:
            durations.append(duration)
    return durations


def _speech_start_to_action_resolved_ms(
    record: dict[str, Any],
    action: str = "",
) -> float | None:
    provider_latency = _mapping(_mapping(record.get("attrs")).get("provider_latency_ms"))
    for provider_key in _resolved_provider_latency_keys(action):
        duration = _number(provider_latency.get(provider_key))
        if duration is not None:
            return max(0.0, duration)
    durations = _mapping(record.get("durations_ms"))
    for duration_key in _resolved_duration_keys(action):
        duration = _number(durations.get(duration_key))
        if duration is not None:
            return max(0.0, duration)
    for timestamp_key in _resolved_timestamp_keys(action):
        duration = _timestamp_delta_ms(record, "speech_started_at", timestamp_key)
        if duration is not None:
            return duration
    return None


def _resolved_provider_latency_keys(action: str) -> tuple[str, ...]:
    if action == "cancel":
        return (
            "interrupt_speech_to_cancel_resolved_ms",
            "interrupt_speech_to_resolved_ms",
        )
    if action in {"rollback", "resume"}:
        return (
            "interrupt_speech_to_rollback_resolved_ms",
            "interrupt_speech_to_resolved_ms",
        )
    return ("interrupt_speech_to_resolved_ms",)


def _resolved_duration_keys(action: str) -> tuple[str, ...]:
    if action == "cancel":
        return (
            "vad_start_to_interrupt_cancel_resolved",
            "vad_start_to_interrupt_resolved",
        )
    if action in {"rollback", "resume"}:
        return (
            "vad_start_to_interrupt_rollback_resolved",
            "vad_start_to_interrupt_resolved",
        )
    return ("vad_start_to_interrupt_resolved",)


def _resolved_timestamp_keys(action: str) -> tuple[str, ...]:
    if action == "cancel":
        return ("interrupt_cancel_resolved_at", "interrupt_resolved_at")
    if action in {"rollback", "resume"}:
        return ("interrupt_rollback_resolved_at", "interrupt_resolved_at")
    return ("interrupt_resolved_at",)


def _timestamp_delta_ms(
    record: dict[str, Any],
    start_key: str,
    end_key: str,
) -> float | None:
    timestamps = _mapping(record.get("timestamps"))
    start = _number(timestamps.get(start_key))
    end = _number(timestamps.get(end_key))
    if start is None or end is None:
        return None
    return max(0.0, (end - start) * 1000.0)


def _duck_started_duration_ms(record: dict[str, Any]) -> float | None:
    for event in _duck_events(record):
        if event.get("event") != "duck_started":
            continue
        duration = _number(event.get("vad_to_duck_ms"))
        if duration is not None:
            return max(0.0, duration)
    return None


def _client_control_sent(records: list[dict[str, Any]], op: str) -> bool:
    for record in records:
        attrs = _mapping(record.get("attrs"))
        events = attrs.get("client_control_events")
        if not isinstance(events, list):
            continue
        for event in events:
            if isinstance(event, dict) and event.get("op") == op:
                return True
    return False


def _ptt_terminal_actions(records: list[dict[str, Any]]) -> list[str]:
    return [
        action
        for action, _reason in (_ptt_terminal(record) for record in records)
        if action
    ]


def _ptt_terminal_reasons(records: list[dict[str, Any]]) -> list[str]:
    return [
        reason
        for _action, reason in (_ptt_terminal(record) for record in records)
        if reason
    ]


def _ptt_terminal(record: dict[str, Any]) -> tuple[str, str]:
    attrs = _mapping(record.get("attrs"))
    segment_terminal = _mapping(attrs.get("ptt_segment_terminal"))
    if segment_terminal:
        return (
            str(segment_terminal.get("action") or ""),
            str(segment_terminal.get("reason") or ""),
        )
    owner = _mapping(attrs.get("ptt_turn_owner"))
    if bool(owner.get("terminal")):
        return str(owner.get("action") or ""), str(owner.get("reason") or "")
    rejected = _mapping(attrs.get("ptt_turn_rejected"))
    if rejected:
        return "reject", str(rejected.get("reason") or "")
    return "", ""


def _has_interrupted_context(records: list[dict[str, Any]]) -> bool:
    for record in records:
        context = _mapping(_mapping(record.get("attrs")).get("interrupted_context"))
        if context:
            return True
    return False


def _has_cancel(records: list[dict[str, Any]]) -> bool:
    return any(_record_is_cancel(record) for record in records)


def _record_is_cancel(record: dict[str, Any]) -> bool:
    attrs = _mapping(record.get("attrs"))
    if attrs.get("interrupt_action") == "cancel":
        return True
    if isinstance(attrs.get("cancel_reason"), str) and attrs.get("cancel_reason"):
        return True
    return any(event.get("event") == "duck_cancelled" for event in _duck_events(record))


def _record_is_resume(record: dict[str, Any]) -> bool:
    attrs = _mapping(record.get("attrs"))
    if attrs.get("interrupt_action") == "rollback":
        return True
    if isinstance(attrs.get("rollback_reason"), str) and attrs.get("rollback_reason"):
        return True
    return any(event.get("event") == "duck_unducked" for event in _duck_events(record))


def _duck_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    attrs = _mapping(record.get("attrs"))
    events = attrs.get("duck_events")
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


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
        cancel_interrupt = _number(
            durations.get("vad_start_to_interrupt_cancel_resolved")
        )
        if cancel_interrupt is not None:
            samples.setdefault(
                "timeline_vad_start_to_interrupt_cancel_resolved",
                [],
            ).append(cancel_interrupt)
        rollback_interrupt = _number(
            durations.get("vad_start_to_interrupt_rollback_resolved")
        )
        if rollback_interrupt is not None:
            samples.setdefault(
                "timeline_vad_start_to_interrupt_rollback_resolved",
                [],
            ).append(rollback_interrupt)
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


def _decision_event_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Expose decision history before later fallback/framework decisions overwrite it."""

    events: list[dict[str, Any]] = []
    for record in records:
        attrs = _mapping(record.get("attrs"))
        raw_events = attrs.get("decision_events")
        if not isinstance(raw_events, list):
            decision = _mapping(attrs.get("decision"))
            if decision:
                events.append(decision)
            continue
        events.extend(event for event in raw_events if isinstance(event, dict))
    if not events:
        return {}

    latest = events[-1]
    metrics: dict[str, Any] = {
        "timeline_decision_event_count": len(events),
    }
    for source_key, metric_key in (
        ("action", "timeline_decision_event_last_action"),
        ("reason", "timeline_decision_event_last_reason"),
        ("intent", "timeline_decision_event_last_intent"),
        ("source", "timeline_decision_event_last_source"),
        ("resolved_reason", "timeline_decision_event_last_resolved_reason"),
        ("transcript_preview", "timeline_decision_event_last_preview"),
    ):
        value = latest.get(source_key)
        if isinstance(value, str) and value:
            metrics[metric_key] = value

    chain = _decision_event_chain(events[-8:])
    if chain:
        metrics["timeline_decision_event_chain"] = chain
    hold_chain = _decision_event_chain(
        [event for event in events if event.get("action") == "hold"][-8:]
    )
    if hold_chain:
        metrics["timeline_decision_event_hold_chain"] = hold_chain
    terminal_chain = _decision_event_chain(
        [
            event
            for event in events
            if event.get("action") in {"cancel", "rollback"}
        ][-8:]
    )
    if terminal_chain:
        metrics["timeline_decision_event_terminal_chain"] = terminal_chain
    return metrics


def _semantic_gate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Expose transcript hot-path gate events for early-interim diagnosis."""

    events: list[dict[str, Any]] = []
    for record in records:
        attrs = _mapping(record.get("attrs"))
        raw_events = attrs.get("semantic_interrupt_gate_events")
        if not isinstance(raw_events, list):
            continue
        events.extend(event for event in raw_events if isinstance(event, dict))
    if not events:
        return {}

    latest = events[-1]
    metrics: dict[str, Any] = {
        "timeline_semantic_gate_event_count": len(events),
    }
    for source_key, metric_key in (
        ("stage", "timeline_semantic_gate_last_stage"),
        ("action", "timeline_semantic_gate_last_action"),
        ("reason", "timeline_semantic_gate_last_reason"),
        ("transcript_preview", "timeline_semantic_gate_last_preview"),
    ):
        value = latest.get(source_key)
        if isinstance(value, str) and value:
            metrics[metric_key] = value

    chain = _semantic_gate_chain(events[-8:])
    if chain:
        metrics["timeline_semantic_gate_chain"] = chain
    blocked = _semantic_gate_chain(
        [
            event
            for event in events
            if event.get("action") in {"inactive", "forward_and_stop"}
        ][-8:]
    )
    if blocked:
        metrics["timeline_semantic_gate_blocked_chain"] = blocked
    return metrics


def _attention_admission_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Expose attention admission reasons that block or allow hot-path EOT."""

    events: list[dict[str, Any]] = []
    for record in records:
        attrs = _mapping(record.get("attrs"))
        raw_events = attrs.get("attention_admission_events")
        if not isinstance(raw_events, list):
            continue
        events.extend(event for event in raw_events if isinstance(event, dict))
    if not events:
        return {}

    latest = events[-1]
    metrics: dict[str, Any] = {
        "timeline_attention_admission_event_count": len(events),
    }
    for source_key, metric_key in (
        ("action", "timeline_attention_admission_last_action"),
        ("reason", "timeline_attention_admission_last_reason"),
        ("transcript_preview", "timeline_attention_admission_last_preview"),
        ("tier", "timeline_attention_admission_last_tier"),
        ("tier_reason", "timeline_attention_admission_last_tier_reason"),
    ):
        value = latest.get(source_key)
        if isinstance(value, str) and value:
            metrics[metric_key] = value

    chain = _attention_admission_chain(events[-8:])
    if chain:
        metrics["timeline_attention_admission_chain"] = chain
    blocked = _attention_admission_chain(
        [
            event
            for event in events
            if event.get("action") in {"observe", "ignore"}
        ][-8:]
    )
    if blocked:
        metrics["timeline_attention_admission_blocked_chain"] = blocked
    return metrics


def _framework_completed_gate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Expose LiveKit completed-turn gate events for fallback-path diagnosis."""

    events: list[dict[str, Any]] = []
    for record in records:
        attrs = _mapping(record.get("attrs"))
        raw_events = attrs.get("framework_completed_gate_events")
        if not isinstance(raw_events, list):
            continue
        events.extend(event for event in raw_events if isinstance(event, dict))
    if not events:
        return {}

    latest = events[-1]
    metrics: dict[str, Any] = {
        "timeline_framework_completed_gate_event_count": len(events),
    }
    for source_key, metric_key in (
        ("stage", "timeline_framework_completed_gate_last_stage"),
        ("action", "timeline_framework_completed_gate_last_action"),
        ("reason", "timeline_framework_completed_gate_last_reason"),
        ("transcript_preview", "timeline_framework_completed_gate_last_preview"),
    ):
        value = latest.get(source_key)
        if isinstance(value, str) and value:
            metrics[metric_key] = value

    chain = _gate_event_chain(events[-8:])
    if chain:
        metrics["timeline_framework_completed_gate_chain"] = chain
    skipped = _gate_event_chain(
        [
            event
            for event in events
            if event.get("action") in {"ignore", "skip"}
        ][-8:]
    )
    if skipped:
        metrics["timeline_framework_completed_gate_skip_chain"] = skipped
    return metrics


def _interruption_owner_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Expose interruption owner events for backchannel/resume diagnosis."""

    events: list[dict[str, Any]] = []
    for record in records:
        attrs = _mapping(record.get("attrs"))
        raw_events = attrs.get("interruption_orchestrator_events")
        if not isinstance(raw_events, list):
            continue
        events.extend(event for event in raw_events if isinstance(event, dict))
    if not events:
        return {}

    latest = events[-1]
    metrics: dict[str, Any] = {
        "timeline_interruption_owner_event_count": len(events),
    }
    for source_key, metric_key in (
        ("event", "timeline_interruption_owner_last_event"),
        ("state", "timeline_interruption_owner_last_state"),
        ("action", "timeline_interruption_owner_last_action"),
        ("reason", "timeline_interruption_owner_last_reason"),
        ("transcript_preview", "timeline_interruption_owner_last_preview"),
        ("text_preview", "timeline_interruption_owner_last_preview"),
    ):
        value = latest.get(source_key)
        if isinstance(value, str) and value:
            metrics[metric_key] = value

    elapsed = _number(latest.get("elapsed_ms"))
    if elapsed is not None:
        metrics["timeline_interruption_owner_last_elapsed_ms"] = elapsed
    since_last = _number(latest.get("since_last_event_ms"))
    if since_last is not None:
        metrics["timeline_interruption_owner_last_since_previous_ms"] = since_last

    for event_name, metric_key in (
        (
            "short_false_interruption_fast_resume",
            "timeline_interruption_owner_fast_resume_elapsed_ms",
        ),
        ("post_speech_evidence_wait", "timeline_interruption_owner_wait_elapsed_ms"),
        ("candidate_resolved", "timeline_interruption_owner_resolved_elapsed_ms"),
    ):
        value = _latest_event_number(events, event_name, "elapsed_ms")
        if value is not None:
            metrics[metric_key] = value

    chain = _interruption_owner_chain(events[-8:])
    if chain:
        metrics["timeline_interruption_owner_chain"] = chain
    wait_chain = _interruption_owner_chain(
        [
            event
            for event in events
            if event.get("event")
            in {
                "turn_policy_decision",
                "post_speech_evidence_wait",
                "short_false_interruption_fast_resume",
            }
        ][-8:]
    )
    if wait_chain:
        metrics["timeline_interruption_owner_wait_chain"] = wait_chain
    return metrics


def _transcript_admission_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Expose transcript admission events before semantic hot-path routing."""

    events: list[dict[str, Any]] = []
    for record in records:
        attrs = _mapping(record.get("attrs"))
        raw_events = attrs.get("transcript_admission_events")
        if not isinstance(raw_events, list):
            continue
        events.extend(event for event in raw_events if isinstance(event, dict))
    if not events:
        return {}

    latest = events[-1]
    metrics: dict[str, Any] = {
        "timeline_transcript_admission_event_count": len(events),
    }
    for source_key, metric_key in (
        ("reason", "timeline_transcript_admission_last_reason"),
        ("transcript_preview", "timeline_transcript_admission_last_preview"),
    ):
        value = latest.get(source_key)
        if isinstance(value, str) and value:
            metrics[metric_key] = value
    accepted = latest.get("accepted")
    if isinstance(accepted, bool):
        metrics["timeline_transcript_admission_last_accepted"] = accepted

    rejected = [
        event
        for event in events
        if event.get("accepted") is False
    ][-8:]
    rejected_chain = _transcript_admission_chain(rejected)
    if rejected_chain:
        metrics["timeline_transcript_admission_rejected_chain"] = rejected_chain
    return metrics


def _transcript_admission_chain(events: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for event in events:
        reason = str(event.get("reason") or "?")
        preview = str(event.get("transcript_preview") or "")[:40]
        parts.append(f"{reason}:{preview}")
    return " ; ".join(parts)


def _decision_event_chain(events: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for event in events:
        action = str(event.get("action") or "?")
        reason = str(event.get("reason") or "?")
        intent = str(event.get("intent") or "")
        source = str(event.get("source") or "")
        preview = str(event.get("transcript_preview") or "")[:40]
        parts.append(f"{action}:{reason}:{intent}:{source}:{preview}")
    return " ; ".join(parts)


def _attention_admission_chain(events: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for event in events:
        action = str(event.get("action") or "?")
        reason = str(event.get("reason") or "?")
        preview = str(event.get("transcript_preview") or "")[:40]
        parts.append(f"{action}:{reason}:{preview}")
    return " ; ".join(parts)


def _semantic_gate_chain(events: list[dict[str, Any]]) -> str:
    return _gate_event_chain(events)


def _latest_event_number(
    events: list[dict[str, Any]],
    event_name: str,
    field: str,
) -> float | None:
    for event in reversed(events):
        if event.get("event") != event_name:
            continue
        value = _number(event.get(field))
        if value is not None:
            return value
    return None


def _interruption_owner_chain(events: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for event in events:
        name = str(event.get("event") or "?")
        elapsed = _number(event.get("elapsed_ms"))
        if elapsed is not None:
            name = f"{name}@{elapsed:.0f}ms"
        action = str(
            event.get("action")
            or event.get("last_policy_action")
            or event.get("turn_policy_action")
            or "?"
        )
        reason = str(event.get("reason") or event.get("last_policy_reason") or "?")
        preview = str(
            event.get("transcript_preview")
            or event.get("text_preview")
            or ""
        )[:40]
        parts.append(f"{name}:{action}:{reason}:{preview}")
    return " ; ".join(parts)


def _gate_event_chain(events: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for event in events:
        stage = str(event.get("stage") or "?")
        action = str(event.get("action") or "?")
        reason = str(event.get("reason") or "?")
        preview = str(event.get("transcript_preview") or "")[:40]
        parts.append(f"{stage}:{action}:{reason}:{preview}")
    return " ; ".join(parts)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _first_number(values: Any) -> float | None:
    for value in values:
        number = _number(value)
        if number is not None:
            return number
    return None
