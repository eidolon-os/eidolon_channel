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
    "transcript_interim_first_at",
    "transcript_final_at",
    "turn_committed_at",
    "llm_first_delta_at",
    "tts_first_audio_at",
    "agent_audio_playback_done_at",
    "interrupt_started_at",
    "interrupt_resolved_at",
    "idle_timeout_triggered_at",
)


@dataclass
class TurnTimeline:
    turn_id: str
    timestamps: dict[str, float] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)

    def mark(self, name: str) -> None:
        if name not in TIMELINE_FIELDS:
            raise ValueError(f"unknown timeline mark {name!r}")
        self.timestamps.setdefault(name, time.monotonic())

    def set_attr(self, key: str, value: Any) -> None:
        self.attrs[key] = value

    def duration_ms(self, start: str, end: str) -> float | None:
        if start not in self.timestamps or end not in self.timestamps:
            return None
        return (self.timestamps[end] - self.timestamps[start]) * 1000

    def snapshot(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "timestamps": dict(self.timestamps),
            "attrs": dict(self.attrs),
            "durations_ms": {
                "vad_start_to_interrupt_resolved": self.duration_ms(
                    "speech_started_at", "interrupt_resolved_at"
                ),
                "speech_stop_to_commit": self.duration_ms(
                    "speech_stopped_at", "turn_committed_at"
                ),
                "commit_to_llm_first_delta": self.duration_ms(
                    "turn_committed_at", "llm_first_delta_at"
                ),
                "llm_first_delta_to_tts_first_audio": self.duration_ms(
                    "llm_first_delta_at", "tts_first_audio_at"
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

