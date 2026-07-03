"""Session-local client control helpers.

Channel uses ``eidolon.control`` for best-effort commands that only make sense
inside the current voice room.  Keep the shared wire envelope and generic
timeline event rules in one place so the full-duplex and half-duplex pipelines
cannot drift.
"""

from __future__ import annotations

from typing import Any

from eidolon_sdk.biz.control import CONTROL_PROTOCOL_VERSION, unix_ms

CHANNEL_CONTROL_SOURCE_ID = "eidolon_channel"
CHANNEL_CONTROL_SOURCE_TYPE = "channel"
SESSION_LOCAL_CONTROL_TTL_MS = 5_000
CLIENT_CONTROL_EVENT_LIMIT = 12

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
