"""Experience SLO constants and helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperienceSlo:
    vad_start_to_duck_p95_ms: int = 80
    interrupt_decision_p95_ms: int = 500
    user_speech_stop_to_commit_p95_ms: int = 900
    eidolon_agent_commit_to_first_delta_p95_ms: int = 1200
    llm_first_delta_to_tts_first_audio_p95_ms: int = 500
    false_interrupt_rate_max: float = 0.05
    backchannel_cancel_rate_max: float = 0.02


DEFAULT_SLO = ExperienceSlo()
