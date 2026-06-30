"""Observe-only client audio-state signals from LiveKit data channels."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

# Wire-contract vocabulary is defined once in eidolon_sdk.biz.contracts; this module
# imports only what it uses to parse the packet. Other modules import these
# constants directly from eidolon_sdk.biz.contracts, not from here.
from eidolon_sdk.biz.contracts import (
    CLIENT_AUDIO_STATE_KNOWN_KEYS,
    CLIENT_AUDIO_STATE_TYPE,
    INPUT_MODE_UNKNOWN,
    PLAYBACK_STATE_UNKNOWN,
    VALID_INPUT_MODES,
    VALID_PLAYBACK_STATES,
    WIRE_SCHEMA_VERSION,
    InputMode,
    PlaybackState,
)

logger = logging.getLogger("agent.client_audio_state")

# Fail-loud strictness. Inbound contract drift (an unknown key like ``ppt``, an
# unrecognized enum value, an unsupported ``schema_v``) is surfaced instead of
# silently degrading to ``unknown``. In strict mode it raises (use in dev/CI so a
# typo fails a test); otherwise it logs a ``[contract]`` warning and parses
# best-effort (production: never drop a live device over a cosmetic drift). Off by
# default; opt in with ``EIDOLON_CONTRACT_STRICT=1`` (or pass ``strict=True``).
_STRICT_DEFAULT = os.getenv("EIDOLON_CONTRACT_STRICT", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _contract_violation(message: str, *, strict: bool) -> None:
    """Loudly surface a contract drift: raise in strict mode, else warn."""
    if strict:
        raise ValueError(f"client.audio_state contract violation: {message}")
    logger.warning("[contract] client.audio_state %s", message)


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
    strict: bool | None = None,
) -> ClientAudioState:
    """Parse and validate the lightweight Web/ESP32 audio-state packet.

    Malformed framing (non-JSON, non-object, wrong ``type``) is always rejected.
    Field-level contract drift — an unknown key (e.g. ``ppt``), an unrecognized
    enum value, or an unsupported ``schema_v`` — is surfaced via
    :func:`_contract_violation` (raise in strict mode, warn otherwise) and then
    parsed best-effort. ``strict`` defaults to ``_STRICT_DEFAULT`` (env).
    """
    is_strict = _STRICT_DEFAULT if strict is None else strict

    try:
        raw = json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise ValueError("client.audio_state payload must be UTF-8 JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("client.audio_state payload must be a JSON object")
    if raw.get("type") not in (None, CLIENT_AUDIO_STATE_TYPE):
        raise ValueError("client.audio_state has unexpected type")

    schema_v = raw.get("schema_v")
    if schema_v is not None and schema_v != WIRE_SCHEMA_VERSION:
        _contract_violation(
            f"unsupported schema_v={schema_v!r} (expected {WIRE_SCHEMA_VERSION})",
            strict=is_strict,
        )

    unknown = set(raw) - CLIENT_AUDIO_STATE_KNOWN_KEYS
    if unknown:
        _contract_violation(
            f"unknown field(s) {sorted(unknown)} (typo, or add to the contract)",
            strict=is_strict,
        )

    return ClientAudioState(
        participant_identity=participant_identity,
        input_mode=_input_mode(raw.get("input_mode"), strict=is_strict),
        ptt=bool(raw.get("ptt", False)),
        manual_interrupt=bool(raw.get("manual_interrupt", False)),
        playback_state=_playback_state(raw.get("playback_state"), strict=is_strict),
        mic_muted=bool(raw.get("mic_muted", False)),
        rms=_optional_float(raw.get("rms")),
        snr_hint=_optional_float(raw.get("snr_hint")),
        client_ts_ms=_optional_int(raw.get("client_ts_ms")),
        received_at=time.monotonic() if received_at is None else received_at,
    )


def _input_mode(value: Any, *, strict: bool) -> InputMode:
    if value is None:
        return INPUT_MODE_UNKNOWN
    if value in VALID_INPUT_MODES:
        return value
    _contract_violation(f"unrecognized input_mode={value!r}", strict=strict)
    return INPUT_MODE_UNKNOWN


def _playback_state(value: Any, *, strict: bool) -> PlaybackState:
    if value is None:
        return PLAYBACK_STATE_UNKNOWN
    if value in VALID_PLAYBACK_STATES:
        return value
    _contract_violation(f"unrecognized playback_state={value!r}", strict=strict)
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
