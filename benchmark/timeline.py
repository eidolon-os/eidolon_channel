"""Timeline debug ingestion for voice benchmark dashboards."""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eidolon.livekit.agent.observability import PROVIDER_LATENCY_SEGMENTS


@dataclass(frozen=True)
class TimelineCapture:
    """Byte-range capture of a long-lived worker timeline JSONL file."""

    source: Path | None
    start_size: int = 0

    @classmethod
    def start(cls, source_path: str) -> "TimelineCapture":
        if not source_path:
            return cls(source=None)
        source = Path(source_path).expanduser()
        size = source.stat().st_size if source.exists() else 0
        return cls(source=source, start_size=size)

    def write_new_lines(self, output_path: Path) -> int:
        if self.source is None or not self.source.exists():
            return 0
        current_size = self.source.stat().st_size
        start = self.start_size if current_size >= self.start_size else 0
        with self.source.open("rb") as f:
            f.seek(start)
            data = f.read()
        if not data:
            return 0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)
        return sum(1 for line in data.splitlines() if line.strip())


def load_timeline_records(path: Path) -> list[dict[str, Any]]:
    """Load per-turn timeline records from a benchmark output directory."""

    records: list[dict[str, Any]] = []
    for file_path in _timeline_files(path):
        with file_path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid timeline JSONL at {file_path}:{line_no}"
                    ) from exc
                if isinstance(record, dict):
                    record.setdefault("_source_path", str(file_path))
                    records.append(record)
    return records


def summarize_timeline_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize timeline records into dashboard-friendly metrics."""

    latency_samples: dict[str, list[float]] = {}
    segment_samples: dict[str, dict[str, Any]] = {}
    decisions: dict[str, int] = {}
    flush_reasons: dict[str, int] = {}
    cases: dict[str, int] = {}
    record_summaries: list[dict[str, Any]] = []
    preemptive = {"triggered": 0, "reused": 0, "discarded": 0}

    for record in records:
        attrs = _mapping(record.get("attrs"))
        case_id = _case_id_from_room_name(attrs.get("room_name"))
        if case_id:
            cases[case_id] = cases.get(case_id, 0) + 1

        pre = _preemptive_outcome(record)
        if pre is not None:
            preemptive["triggered"] += 1
            preemptive["discarded" if pre == "discarded" else "reused"] += 1
        durations = _mapping(record.get("durations_ms"))
        provider_latency = _mapping(attrs.get("provider_latency_ms"))
        for key, value in {**durations, **provider_latency}.items():
            if isinstance(value, (int, float)):
                latency_samples.setdefault(key, []).append(float(value))

        segments = _provider_segments(record)
        for segment in segments:
            value = segment.get("duration_ms")
            if not isinstance(value, (int, float)):
                continue
            name = str(segment.get("name") or "")
            if not name:
                continue
            bucket = segment_samples.setdefault(
                name,
                {
                    "name": name,
                    "label": str(segment.get("label") or name),
                    "stage": str(segment.get("stage") or ""),
                    "start": str(segment.get("start") or ""),
                    "end": str(segment.get("end") or ""),
                    "values": [],
                },
            )
            bucket["values"].append(float(value))

        decision_reason = attrs.get("decision_reason")
        if isinstance(decision_reason, str) and decision_reason:
            decisions[decision_reason] = decisions.get(decision_reason, 0) + 1
        decision = _mapping(attrs.get("decision"))
        hold_recheck_ms = decision.get("hold_recheck_ms")
        if isinstance(hold_recheck_ms, (int, float)) and not isinstance(
            hold_recheck_ms,
            bool,
        ):
            latency_samples.setdefault("decision_hold_recheck_ms", []).append(
                float(hold_recheck_ms)
            )

        flush_reason = attrs.get("timeline_flush_reason")
        if isinstance(flush_reason, str) and flush_reason:
            flush_reasons[flush_reason] = flush_reasons.get(flush_reason, 0) + 1

        record_summaries.append(
            {
                "turn_id": record.get("turn_id", ""),
                "case_id": case_id,
                "room_name": attrs.get("room_name", ""),
                "decision_reason": attrs.get("decision_reason", ""),
                "interrupt_action": attrs.get("interrupt_action", ""),
                "intent": _intent(attrs),
                "segments": segments,
                "preemptive": _preemptive_outcome(record) or "",
            }
        )

    triggered = preemptive["triggered"]
    preemptive["reuse_rate"] = (
        preemptive["reused"] / triggered if triggered else None
    )
    preemptive["waste_rate"] = (
        preemptive["discarded"] / triggered if triggered else None
    )

    return {
        "count": len(records),
        "preemptive": preemptive,
        "latencies": {
            key: _aggregate_values(values)
            for key, values in sorted(latency_samples.items())
        },
        "provider_segments": _summarize_segments(segment_samples),
        "decision_reasons": dict(sorted(decisions.items())),
        "flush_reasons": dict(sorted(flush_reasons.items())),
        "cases": dict(sorted(cases.items())),
        "record_summaries": record_summaries,
        "records": records,
    }


def _timeline_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if _is_timeline_file(path) else []
    if not path.exists():
        return []
    return sorted(
        file_path for file_path in path.rglob("*.jsonl") if _is_timeline_file(file_path)
    )


def _is_timeline_file(path: Path) -> bool:
    name = path.name.lower()
    return "timeline" in name or name in {"turns.jsonl", "turn_timeline.jsonl"}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _provider_segments(record: dict[str, Any]) -> list[dict[str, Any]]:
    attrs = _mapping(record.get("attrs"))
    raw_segments = attrs.get("provider_segments")
    if isinstance(raw_segments, list):
        return [
            segment
            for segment in raw_segments
            if isinstance(segment, dict)
        ]
    timestamps = _mapping(record.get("timestamps"))
    return [
        {
            "name": segment.name,
            "label": segment.label,
            "stage": segment.stage,
            "start": segment.start,
            "end": segment.end,
            "duration_ms": _duration_ms(timestamps, segment.start, segment.end),
            "available": (
                isinstance(timestamps.get(segment.start), (int, float))
                and isinstance(timestamps.get(segment.end), (int, float))
            ),
        }
        for segment in PROVIDER_LATENCY_SEGMENTS
    ]


def _duration_ms(timestamps: dict[str, Any], start: str, end: str) -> float | None:
    start_value = timestamps.get(start)
    end_value = timestamps.get(end)
    if not isinstance(start_value, (int, float)) or not isinstance(
        end_value, (int, float)
    ):
        return None
    return (float(end_value) - float(start_value)) * 1000


def _preemptive_outcome(record: dict[str, Any]) -> str | None:
    """Derive the preemptive-generation outcome from timeline marks.

    A turn is "preemptive" when the brain request started BEFORE the turn was
    committed (the framework fired generation on a stable interim). If that
    speculative turn was later cancelled (``brain_cancelled_at`` present) it was
    discarded; otherwise it was reused. Returns None when the brain started at
    or after commit (no preemption — the normal post-commit path).
    """

    ts = _mapping(record.get("timestamps"))
    started = ts.get("brain_request_started_at")
    committed = ts.get("turn_committed_at")
    if not isinstance(started, (int, float)) or not isinstance(committed, (int, float)):
        return None
    if started >= committed:
        return None
    return "discarded" if "brain_cancelled_at" in ts else "reused"


def _intent(attrs: dict[str, Any]) -> str:
    decision = _mapping(attrs.get("decision"))
    intent = decision.get("intent")
    return str(intent) if intent else ""


def _summarize_segments(
    segment_samples: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    summarized: dict[str, dict[str, Any]] = {}
    for name, bucket in sorted(segment_samples.items()):
        values = bucket.pop("values")
        summarized[name] = {
            **bucket,
            **_aggregate_values(values),
        }
    return summarized


def _case_id_from_room_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    match = re.match(r"^voice-bench-(?P<case>.+)-[0-9a-f]{8}$", value)
    return match.group("case") if match else ""


def _aggregate_values(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "avg": statistics.fmean(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else None,
    }


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
