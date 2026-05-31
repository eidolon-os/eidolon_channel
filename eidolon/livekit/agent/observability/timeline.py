"""Per-turn timeline for OpenAI-Realtime-style experience metrics."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TIMELINE_FIELDS = (
    "speech_started_at",
    "speech_stopped_at",
    "stt_stream_started_at",
    "stt_ws_connected_at",
    "stt_stream_first_audio_sent_at",
    "stt_first_audio_sent_at",
    "stt_flush_sent_at",
    "stt_provider_first_partial_at",
    "stt_provider_final_at",
    "transcript_interim_first_at",
    "transcript_final_at",
    "turn_committed_at",
    "llm_started_at",
    "llm_first_delta_at",
    "brain_request_started_at",
    "brain_request_sent_at",
    "brain_first_delta_at",
    "brain_done_at",
    "tts_request_started_at",
    "tts_provider_first_audio_at",
    "tts_first_audio_at",
    "agent_audio_playback_done_at",
    "interrupt_started_at",
    "interrupt_resolved_at",
    "idle_timeout_triggered_at",
)


@dataclass(frozen=True)
class ProviderLatencySegment:
    """A stable, dashboard-friendly latency segment between two timeline marks."""

    name: str
    label: str
    stage: str
    start: str
    end: str


PROVIDER_LATENCY_SEGMENTS: tuple[ProviderLatencySegment, ...] = (
    ProviderLatencySegment(
        name="stt_first_audio_sent",
        label="STT: speech start -> first turn audio sent",
        stage="stt",
        start="speech_started_at",
        end="stt_first_audio_sent_at",
    ),
    ProviderLatencySegment(
        name="stt_provider_first_partial",
        label="STT: speech start -> provider partial",
        stage="stt",
        start="speech_started_at",
        end="stt_provider_first_partial_at",
    ),
    ProviderLatencySegment(
        name="stt_livekit_first_interim",
        label="STT: provider partial -> LiveKit interim",
        stage="stt",
        start="stt_provider_first_partial_at",
        end="transcript_interim_first_at",
    ),
    ProviderLatencySegment(
        name="stt_final",
        label="STT: speech start -> LiveKit final",
        stage="stt",
        start="speech_started_at",
        end="transcript_final_at",
    ),
    ProviderLatencySegment(
        name="stt_provider_final_to_livekit_final",
        label="STT: provider final -> LiveKit final",
        stage="stt",
        start="stt_provider_final_at",
        end="transcript_final_at",
    ),
    ProviderLatencySegment(
        name="turn_commit",
        label="Turn: speech stop -> commit",
        stage="turn_policy",
        start="speech_stopped_at",
        end="turn_committed_at",
    ),
    ProviderLatencySegment(
        name="brain_start",
        label="Brain: commit -> RPC start",
        stage="brain",
        start="turn_committed_at",
        end="brain_request_started_at",
    ),
    ProviderLatencySegment(
        name="brain_request_write",
        label="Brain: RPC start -> request sent",
        stage="brain",
        start="brain_request_started_at",
        end="brain_request_sent_at",
    ),
    ProviderLatencySegment(
        name="brain_first_delta",
        label="Brain: request sent -> first delta",
        stage="brain",
        start="brain_request_sent_at",
        end="brain_first_delta_at",
    ),
    ProviderLatencySegment(
        name="brain_stream_duration",
        label="Brain: request sent -> done",
        stage="brain",
        start="brain_request_sent_at",
        end="brain_done_at",
    ),
    ProviderLatencySegment(
        name="stt_final_after_commit",
        label="Endpoint: commit -> STT provider final",
        stage="turn_policy",
        start="turn_committed_at",
        end="stt_provider_final_at",
    ),
    ProviderLatencySegment(
        name="tts_first_audio",
        label="TTS: brain first delta -> first audio",
        stage="tts",
        start="brain_first_delta_at",
        end="tts_first_audio_at",
    ),
    ProviderLatencySegment(
        name="tts_request_to_first_audio",
        label="TTS: request started -> provider first audio (TTFB)",
        stage="tts",
        start="tts_request_started_at",
        end="tts_provider_first_audio_at",
    ),
    ProviderLatencySegment(
        name="tts_provider_to_agent_audio",
        label="TTS: provider first audio -> agent audio",
        stage="tts",
        start="tts_provider_first_audio_at",
        end="tts_first_audio_at",
    ),
    ProviderLatencySegment(
        name="commit_to_first_audio",
        label="E2E: commit -> first audio",
        stage="experience",
        start="turn_committed_at",
        end="tts_first_audio_at",
    ),
    ProviderLatencySegment(
        name="playback_duration",
        label="Playback: first audio -> done",
        stage="playback",
        start="tts_first_audio_at",
        end="agent_audio_playback_done_at",
    ),
    ProviderLatencySegment(
        name="interrupt_resolution",
        label="Interrupt: VAD start -> resolved",
        stage="interrupt",
        start="speech_started_at",
        end="interrupt_resolved_at",
    ),
)


@dataclass
class TurnTimeline:
    turn_id: str
    timestamps: dict[str, float] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.attrs.setdefault("stt_final_changed_after_preemptive", None)

    def mark(self, name: str) -> None:
        if name not in TIMELINE_FIELDS:
            raise ValueError(f"unknown timeline mark {name!r}")
        self.timestamps.setdefault(name, time.monotonic())

    def mark_at(self, name: str, timestamp: float) -> None:
        if name not in TIMELINE_FIELDS:
            raise ValueError(f"unknown timeline mark {name!r}")
        self.timestamps.setdefault(name, timestamp)

    def mark_after(self, name: str, anchor: str, offset_sec: float) -> None:
        if anchor not in self.timestamps:
            return
        self.mark_at(name, self.timestamps[anchor] + offset_sec)

    def set_attr(self, key: str, value: Any) -> None:
        self.attrs[key] = value

    def record_decision(
        self,
        *,
        action: str,
        reason: str,
        rollback_drop_buffered: bool,
        intent: str | None = None,
        intent_source: str = "",
        intent_confidence: float = 0.0,
        topic_switch_hint: bool = False,
        correction_hint: bool = False,
        source: str = "turn_policy",
        resolved_reason: str | None = None,
        eot_score: float | None = None,
        transcript_preview: str = "",
        vad_active: bool | None = None,
    ) -> None:
        """Attach a normalized interruption decision to this turn.

        This is observability-only: the decision has already been made by the
        turn-policy layer. Keeping the shape stable makes JSONL, dashboards,
        and future OpenTelemetry export use the same vocabulary.
        """

        payload = {
            "action": action,
            "reason": reason,
            "rollback_drop_buffered": rollback_drop_buffered,
            "intent": intent,
            "intent_source": intent_source,
            "intent_confidence": intent_confidence,
            "topic_switch_hint": topic_switch_hint,
            "correction_hint": correction_hint,
            "source": source,
            "resolved_reason": resolved_reason,
            "eot_score": eot_score,
            "transcript_preview": transcript_preview,
            "vad_active": vad_active,
        }
        self.attrs["decision"] = payload
        self.attrs["decision_reason"] = reason
        self.attrs["interrupt_action"] = action
        self.attrs["rollback_drop_buffered"] = rollback_drop_buffered

    def duration_ms(self, start: str, end: str) -> float | None:
        if start not in self.timestamps or end not in self.timestamps:
            return None
        return (self.timestamps[end] - self.timestamps[start]) * 1000

    def provider_latency_ms(self) -> dict[str, float | None]:
        """Return provider/experience latencies derivable from known marks."""

        return {
            "stt_interim_first_ms": self.duration_ms(
                "speech_started_at", "transcript_interim_first_at"
            ),
            "stt_final_ms": self.duration_ms(
                "speech_started_at", "transcript_final_at"
            ),
            "stt_first_audio_sent_ms": self.duration_ms(
                "speech_started_at", "stt_first_audio_sent_at"
            ),
            "stt_first_audio_to_provider_partial_ms": self.duration_ms(
                "stt_first_audio_sent_at", "stt_provider_first_partial_at"
            ),
            "stt_speech_to_provider_partial_ms": self.duration_ms(
                "speech_started_at", "stt_provider_first_partial_at"
            ),
            "stt_provider_partial_to_livekit_interim_ms": self.duration_ms(
                "stt_provider_first_partial_at", "transcript_interim_first_at"
            ),
            "stt_provider_final_to_livekit_final_ms": self.duration_ms(
                "stt_provider_final_at", "transcript_final_at"
            ),
            "speech_stop_to_commit_ms": self.duration_ms(
                "speech_stopped_at", "turn_committed_at"
            ),
            "commit_to_llm_started_ms": self.duration_ms(
                "turn_committed_at", "llm_started_at"
            ),
            "commit_to_llm_first_delta_ms": self.duration_ms(
                "turn_committed_at", "llm_first_delta_at"
            ),
            "llm_started_to_first_delta_ms": self.duration_ms(
                "llm_started_at", "llm_first_delta_at"
            ),
            "commit_to_brain_request_started_ms": self.duration_ms(
                "turn_committed_at", "brain_request_started_at"
            ),
            "brain_request_write_ms": self.duration_ms(
                "brain_request_started_at", "brain_request_sent_at"
            ),
            "brain_request_to_first_delta_ms": self.duration_ms(
                "brain_request_sent_at", "brain_first_delta_at"
            ),
            "brain_stream_duration_ms": self.duration_ms(
                "brain_request_sent_at", "brain_done_at"
            ),
            "commit_to_tts_first_audio_ms": self.duration_ms(
                "turn_committed_at", "tts_first_audio_at"
            ),
            "llm_started_to_tts_first_audio_ms": self.duration_ms(
                "llm_started_at", "tts_first_audio_at"
            ),
            "llm_first_delta_to_tts_first_audio_ms": self.duration_ms(
                "llm_first_delta_at", "tts_first_audio_at"
            ),
            "brain_first_delta_to_tts_first_audio_ms": self.duration_ms(
                "brain_first_delta_at", "tts_first_audio_at"
            ),
            "tts_request_to_provider_first_audio_ms": self.duration_ms(
                "tts_request_started_at", "tts_provider_first_audio_at"
            ),
            "tts_provider_first_audio_to_agent_audio_ms": self.duration_ms(
                "tts_provider_first_audio_at", "tts_first_audio_at"
            ),
            "stt_final_after_commit_ms": self.duration_ms(
                "turn_committed_at", "stt_provider_final_at"
            ),
            "tts_playback_duration_ms": self.duration_ms(
                "tts_first_audio_at", "agent_audio_playback_done_at"
            ),
        }

    def provider_segments(self) -> list[dict[str, Any]]:
        """Return normalized provider/experience segments for dashboards."""

        segments: list[dict[str, Any]] = []
        for segment in PROVIDER_LATENCY_SEGMENTS:
            duration = self.duration_ms(segment.start, segment.end)
            segments.append(
                {
                    "name": segment.name,
                    "label": segment.label,
                    "stage": segment.stage,
                    "start": segment.start,
                    "end": segment.end,
                    "duration_ms": duration,
                    "available": duration is not None,
                }
            )
        return segments

    def snapshot(self) -> dict[str, Any]:
        provider_latency = self.provider_latency_ms()
        provider_segments = self.provider_segments()
        attrs = dict(self.attrs)
        attrs.setdefault("provider_latency_ms", provider_latency)
        attrs.setdefault("provider_segments", provider_segments)
        return {
            "turn_id": self.turn_id,
            "timestamps": dict(self.timestamps),
            "attrs": attrs,
            "durations_ms": {
                "vad_start_to_interrupt_resolved": self.duration_ms(
                    "speech_started_at", "interrupt_resolved_at"
                ),
                "speech_stop_to_commit": self.duration_ms(
                    "speech_stopped_at", "turn_committed_at"
                ),
                "commit_to_llm_started": self.duration_ms(
                    "turn_committed_at", "llm_started_at"
                ),
                "commit_to_llm_first_delta": self.duration_ms(
                    "turn_committed_at", "llm_first_delta_at"
                ),
                "commit_to_brain_request_started": self.duration_ms(
                    "turn_committed_at", "brain_request_started_at"
                ),
                "brain_request_to_first_delta": self.duration_ms(
                    "brain_request_sent_at", "brain_first_delta_at"
                ),
                "commit_to_tts_first_audio": self.duration_ms(
                    "turn_committed_at", "tts_first_audio_at"
                ),
                "llm_first_delta_to_tts_first_audio": self.duration_ms(
                    "llm_first_delta_at", "tts_first_audio_at"
                ),
                "brain_first_delta_to_tts_first_audio": self.duration_ms(
                    "brain_first_delta_at", "tts_first_audio_at"
                ),
            },
        }

    def append_debug_jsonl(self, path: str) -> None:
        if not path:
            return
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.snapshot(), ensure_ascii=False) + "\n")
