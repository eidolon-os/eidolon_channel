"""Observe-only client audio-state signals from LiveKit data channels."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal


CLIENT_AUDIO_STATE_TOPIC = "eidolon.audio_state"
INPUT_MODE_AUTO = "auto"
INPUT_MODE_PTT = "ptt"
INPUT_MODE_MANUAL = "manual"
INPUT_MODE_UNKNOWN = "unknown"
PLAYBACK_STATE_IDLE = "idle"
PLAYBACK_STATE_AGENT_SPEAKING = "agent_speaking"
PLAYBACK_STATE_UNKNOWN = "unknown"

InputMode = Literal["auto", "ptt", "manual", "unknown"]
PlaybackState = Literal["idle", "agent_speaking", "unknown"]


@dataclass(frozen=True)
class ClientAudioState:
    participant_identity: str
    input_mode: InputMode = INPUT_MODE_UNKNOWN
    ptt: bool = False
    manual_interrupt: bool = False
    playback_state: PlaybackState = PLAYBACK_STATE_UNKNOWN
    mic_muted: bool = False
    rms: float | None = None
    snr_hint: float | None = None
    client_ts_ms: int | None = None
    received_at: float = 0.0

    def is_fresh(self, *, now: float | None = None, max_age_sec: float = 2.0) -> bool:
        ref = time.monotonic() if now is None else now
        return ref - self.received_at <= max_age_sec

    def as_timeline_attr(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["age_ms"] = round((time.monotonic() - self.received_at) * 1000)
        return payload


def parse_client_audio_state(
    data: bytes,
    *,
    participant_identity: str,
    received_at: float | None = None,
) -> ClientAudioState:
    """Parse and sanitize the lightweight Web/ESP32 audio-state packet."""

    try:
        raw = json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise ValueError("client.audio_state payload must be UTF-8 JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("client.audio_state payload must be a JSON object")
    if raw.get("type") not in (None, CLIENT_AUDIO_STATE_TOPIC):
        raise ValueError("client.audio_state has unexpected type")

    return ClientAudioState(
        participant_identity=participant_identity,
        input_mode=_input_mode(raw.get("input_mode")),
        ptt=bool(raw.get("ptt", False)),
        manual_interrupt=bool(raw.get("manual_interrupt", False)),
        playback_state=_playback_state(raw.get("playback_state")),
        mic_muted=bool(raw.get("mic_muted", False)),
        rms=_optional_float(raw.get("rms")),
        snr_hint=_optional_float(raw.get("snr_hint")),
        client_ts_ms=_optional_int(raw.get("client_ts_ms")),
        received_at=time.monotonic() if received_at is None else received_at,
    )


def _input_mode(value: Any) -> InputMode:
    if value in (INPUT_MODE_AUTO, INPUT_MODE_PTT, INPUT_MODE_MANUAL):
        return value
    return INPUT_MODE_UNKNOWN


def _playback_state(value: Any) -> PlaybackState:
    if value in (PLAYBACK_STATE_IDLE, PLAYBACK_STATE_AGENT_SPEAKING):
        return value
    return PLAYBACK_STATE_UNKNOWN


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    return max(0.0, min(1.0, result))


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
