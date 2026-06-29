"""Post-run HIL checks for full-duplex barge-in timeline evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .timeline import load_timeline_records


@dataclass
class HilBargeInReport:
    passed: bool
    records: int
    room_names: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "records": self.records,
            "room_names": self.room_names,
            "findings": self.findings,
            "evidence": self.evidence,
        }


def analyze_hil_barge_in(
    timeline_path: Path,
    *,
    room_contains: str = "",
    latest: int = 0,
    require_cancel: bool = False,
    require_resume: bool = False,
    max_speech_start_to_suspend_ms: float = 120.0,
    max_speech_start_to_cancel_ms: float | None = 500.0,
    max_speech_start_to_resume_ms: float | None = 900.0,
) -> HilBargeInReport:
    """Check real-device timeline records for channel-owned barge-in evidence.

    This is intentionally independent of benchmark case ids. A real ESP32 HIL
    room will not necessarily use the ``voice-bench-*`` naming convention, but
    it should still expose the same worker timeline evidence.
    """

    records = _filter_records(
        load_timeline_records(timeline_path),
        room_contains=room_contains,
        latest=latest,
    )
    findings: list[str] = []
    evidence = _collect_evidence(records)

    if not records:
        findings.append("no matching timeline records")
    if not evidence["soft_duck_admitted"]:
        findings.append("missing playback_speech_start_soft_duck admission")
    if not evidence["duck_started"]:
        findings.append("missing duck_started / interrupt_started_at")
    else:
        suspend_ms = evidence.get("speech_start_to_suspend_ms")
        if suspend_ms is not None and suspend_ms > max_speech_start_to_suspend_ms:
            findings.append(
                "speech-start-to-suspend exceeded "
                f"{max_speech_start_to_suspend_ms}ms: {round(suspend_ms, 1)}"
            )

    if require_cancel or evidence["cancelled"]:
        if not evidence["cancelled"]:
            findings.append("missing cancel evidence")
        else:
            cancel_ms = evidence.get("speech_start_to_cancel_ms")
            if (
                cancel_ms is not None
                and max_speech_start_to_cancel_ms is not None
                and cancel_ms > max_speech_start_to_cancel_ms
            ):
                findings.append(
                    "speech-start-to-cancel exceeded "
                    f"{max_speech_start_to_cancel_ms}ms: {round(cancel_ms, 1)}"
                )
        if not evidence["playback_stop_sent"]:
            findings.append("missing playback.stop client control")
        if not evidence["interrupted_context"]:
            findings.append("missing interrupted_context for cancelled reply")

    if require_resume or evidence["resumed"]:
        if not evidence["resumed"]:
            findings.append("missing resume / duck_unducked evidence")
        else:
            resume_ms = evidence.get("speech_start_to_resume_ms")
            if (
                resume_ms is not None
                and max_speech_start_to_resume_ms is not None
                and resume_ms > max_speech_start_to_resume_ms
            ):
                findings.append(
                    "speech-start-to-resume exceeded "
                    f"{max_speech_start_to_resume_ms}ms: {round(resume_ms, 1)}"
                )

    if evidence["observe_without_duck"]:
        findings.append(
            "saw observe-only playback speech; this reproduces the old blocked path"
        )

    return HilBargeInReport(
        passed=not findings,
        records=len(records),
        room_names=sorted({name for name in (_room_name(r) for r in records) if name}),
        findings=findings,
        evidence=evidence,
    )


def _filter_records(
    records: list[dict[str, Any]],
    *,
    room_contains: str,
    latest: int,
) -> list[dict[str, Any]]:
    if room_contains:
        records = [
            record
            for record in records
            if room_contains in _room_name(record)
        ]
    if latest > 0:
        records = records[-latest:]
    return records


def _collect_evidence(records: list[dict[str, Any]]) -> dict[str, Any]:
    soft_duck_admitted = False
    observe_without_duck = False
    duck_started = False
    cancelled = False
    resumed = False
    playback_stop_sent = False
    interrupted_context = False
    suspend_samples: list[float] = []
    cancel_samples: list[float] = []
    resume_samples: list[float] = []

    for record in records:
        attrs = _mapping(record.get("attrs"))
        for admission in _attention_admissions(attrs):
            if admission.get("reason") == "playback_speech_start_soft_duck":
                soft_duck_admitted = True
            if admission.get("reason") == "client_playback_active_without_direct_signal":
                observe_without_duck = True

        if _has_duck_started(record):
            duck_started = True
        if _is_cancel(record):
            cancelled = True
        if _is_resume(record):
            resumed = True
        if _client_control_sent(attrs, "playback.stop"):
            playback_stop_sent = True
        if _mapping(attrs.get("interrupted_context")):
            interrupted_context = True

        suspend_ms = _speech_start_to_suspend_ms(record)
        if suspend_ms is not None:
            suspend_samples.append(suspend_ms)
        if _is_cancel(record):
            cancel_ms = _speech_start_to_resolved_ms(record)
            if cancel_ms is not None:
                cancel_samples.append(cancel_ms)
        if _is_resume(record):
            resume_ms = _speech_start_to_resolved_ms(record)
            if resume_ms is not None:
                resume_samples.append(resume_ms)

    return {
        "soft_duck_admitted": soft_duck_admitted,
        "observe_without_duck": observe_without_duck,
        "duck_started": duck_started,
        "cancelled": cancelled,
        "resumed": resumed,
        "playback_stop_sent": playback_stop_sent,
        "interrupted_context": interrupted_context,
        "speech_start_to_suspend_ms": max(suspend_samples) if suspend_samples else None,
        "speech_start_to_cancel_ms": max(cancel_samples) if cancel_samples else None,
        "speech_start_to_resume_ms": max(resume_samples) if resume_samples else None,
    }


def _attention_admissions(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    admission = attrs.get("attention_admission")
    if isinstance(admission, dict):
        values.append(admission)
    events = attrs.get("attention_admission_events")
    if isinstance(events, list):
        values.extend(item for item in events if isinstance(item, dict))
    return values


def _has_duck_started(record: dict[str, Any]) -> bool:
    if _timestamp(record, "interrupt_started_at") is not None:
        return True
    return any(event.get("event") == "duck_started" for event in _duck_events(record))


def _is_cancel(record: dict[str, Any]) -> bool:
    attrs = _mapping(record.get("attrs"))
    return (
        attrs.get("interrupt_action") == "cancel"
        or bool(attrs.get("cancel_reason"))
        or any(event.get("event") == "duck_cancelled" for event in _duck_events(record))
    )


def _is_resume(record: dict[str, Any]) -> bool:
    attrs = _mapping(record.get("attrs"))
    return (
        attrs.get("interrupt_action") == "rollback"
        or bool(attrs.get("rollback_reason"))
        or any(event.get("event") == "duck_unducked" for event in _duck_events(record))
    )


def _client_control_sent(attrs: dict[str, Any], op: str) -> bool:
    events = attrs.get("client_control_events")
    if not isinstance(events, list):
        return False
    return any(isinstance(event, dict) and event.get("op") == op for event in events)


def _speech_start_to_suspend_ms(record: dict[str, Any]) -> float | None:
    duration = _timestamp_delta_ms(record, "speech_started_at", "interrupt_started_at")
    if duration is not None:
        return duration
    for event in _duck_events(record):
        if event.get("event") == "duck_started":
            duration = _number(event.get("vad_to_duck_ms"))
            if duration is not None:
                return max(0.0, duration)
    return None


def _speech_start_to_resolved_ms(record: dict[str, Any]) -> float | None:
    duration = _number(_mapping(record.get("durations_ms")).get("vad_start_to_interrupt_resolved"))
    if duration is not None:
        return max(0.0, duration)
    return _timestamp_delta_ms(record, "speech_started_at", "interrupt_resolved_at")


def _timestamp_delta_ms(
    record: dict[str, Any],
    start_key: str,
    end_key: str,
) -> float | None:
    start = _timestamp(record, start_key)
    end = _timestamp(record, end_key)
    if start is None or end is None:
        return None
    return max(0.0, (end - start) * 1000.0)


def _timestamp(record: dict[str, Any], key: str) -> float | None:
    return _number(_mapping(record.get("timestamps")).get(key))


def _duck_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    events = _mapping(record.get("attrs")).get("duck_events")
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def _room_name(record: dict[str, Any]) -> str:
    value = _mapping(record.get("attrs")).get("room_name")
    return value if isinstance(value, str) else ""


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
