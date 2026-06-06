"""Observe-only client audio-state signals from LiveKit data channels."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal


CLIENT_AUDIO_STATE_TOPIC = "client.audio_state"

InputMode = Literal["auto", "ptt", "manual", "unknown"]
PlaybackState = Literal["idle", "agent_speaking", "unknown"]


@dataclass(frozen=True)
class ClientAudioState:
    participant_identity: str
    input_mode: InputMode = "unknown"
    ptt: bool = False
    manual_interrupt: bool = False
    playback_state: PlaybackState = "unknown"
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
    if value in ("auto", "ptt", "manual"):
        return value
    return "unknown"


def _playback_state(value: Any) -> PlaybackState:
    if value in ("idle", "agent_speaking"):
        return value
    return "unknown"


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
