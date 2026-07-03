"""Session-local client control helpers.

Channel uses ``eidolon.control`` for best-effort commands that only make sense
inside the current voice room, such as playback stop and PTT turn lifecycle
status.  Keep the wire envelope and timeline event rules in one place so the
streaming and half-duplex pipelines cannot drift.
"""

from __future__ import annotations

from typing import Any

from eidolon_sdk.biz.contracts import CONTROL_OP_PTT_TURN_STATUS
from eidolon_sdk.biz.control import CONTROL_PROTOCOL_VERSION, unix_ms

CHANNEL_CONTROL_SOURCE_ID = "eidolon_channel"
CHANNEL_CONTROL_SOURCE_TYPE = "channel"
SESSION_LOCAL_CONTROL_TTL_MS = 5_000
CLIENT_CONTROL_EVENT_LIMIT = 12

PTT_OUTCOME_RECORDING = "recording"
PTT_OUTCOME_FINALIZING = "finalizing"
PTT_OUTCOME_COMMITTED = "committed"
PTT_TRANSCRIPT_PREVIEW_CHARS = 80

PTT_NO_TURN_TERMINAL_REASONS = frozenset(
    {
        "empty_hold",
        "speech_without_transcript",
        "tap_to_stop",
    }
)


def ptt_rejected_outcome(reason: str) -> str:
    return f"rejected:{reason or 'ptt_rejected'}"


def build_ptt_turn_status_payload(
    outcome: str,
    reason: str,
    *,
    transcript: str = "",
) -> dict[str, object]:
    return {
        "outcome": outcome,
        "reason": reason,
        "transcript_preview": transcript[:PTT_TRANSCRIPT_PREVIEW_CHARS],
    }


def build_client_control_event(
    *,
    op: str,
    reason: str,
    turn_id: str = "",
) -> dict[str, str]:
    return {
        "op": op,
        "reason": reason,
        "turn_id": turn_id,
    }


def should_drop_pending_client_control_event(
    *,
    op: str,
    reason: str,
    turn_id: str = "",
) -> bool:
    return (
        not turn_id
        and op == CONTROL_OP_PTT_TURN_STATUS
        and reason in PTT_NO_TURN_TERMINAL_REASONS
    )


def append_client_control_event(
    events: list[dict[str, str]],
    event: dict[str, str],
) -> list[dict[str, str]]:
    return [*events, event][-CLIENT_CONTROL_EVENT_LIMIT:]


def build_session_client_control_envelope(
    *,
    op: str,
    reason: str,
    payload: dict[str, object] | None = None,
    turn_id: str = "",
    now_ms: int | None = None,
) -> dict[str, Any]:
    control_payload: dict[str, object] = {"reason": reason, "turn_id": turn_id}
    if payload:
        control_payload.update(payload)
    timestamp_ms = unix_ms() if now_ms is None else int(now_ms)
    return {
        "v": CONTROL_PROTOCOL_VERSION,
        "kind": "cmd",
        "id": f"{op}:{timestamp_ms}",
        "op": op,
        "src": {"type": CHANNEL_CONTROL_SOURCE_TYPE, "id": CHANNEL_CONTROL_SOURCE_ID},
        "payload": control_payload,
        "ts": timestamp_ms,
        "ttl_ms": SESSION_LOCAL_CONTROL_TTL_MS,
    }
