"""Idle watchdog wiring for the full-duplex runtime."""

from __future__ import annotations

import asyncio
from typing import Any

from eidolon_sdk.biz.contracts import SESSION_END_IDLE_NORMAL

from eidolon.livekit.common.config import TurnPolicyConfig

from ..session.idle import IdleWatchdog


def build_full_duplex_idle_watchdog(pipeline: Any) -> IdleWatchdog:
    return IdleWatchdog(
        timeout_sec=pipeline._idle_timeout_sec,
        get_session=lambda: getattr(pipeline, "_session", None),
        get_room=lambda: getattr(pipeline, "_room", None),
        get_timeline=lambda: getattr(pipeline, "_timeline", None),
        session_closed_event=pipeline._session_closed_event,
        on_idle_disconnect=pipeline._on_idle_disconnect,
        on_session_end=pipeline._on_session_end,
        disconnect_grace_sec=pipeline._idle_disconnect_grace_sec,
        idle_end_reason=pipeline._idle_end_reason,
    )


def ensure_full_duplex_idle_watchdog(pipeline: Any) -> None:
    if not hasattr(pipeline, "_session_closed_event"):
        pipeline._session_closed_event = asyncio.Event()
    if not hasattr(pipeline, "_idle_timeout_sec"):
        pipeline._idle_timeout_sec = 0.0
    if not hasattr(pipeline, "_on_idle_disconnect"):
        pipeline._on_idle_disconnect = None
    if not hasattr(pipeline, "_on_session_end"):
        pipeline._on_session_end = None
    if not hasattr(pipeline, "_idle_disconnect_grace_sec"):
        turn_policy = getattr(pipeline, "_turn_policy", TurnPolicyConfig())
        pipeline._idle_disconnect_grace_sec = (
            turn_policy.idle.disconnect_grace_ms / 1000.0
        )
    if not hasattr(pipeline, "_idle_end_reason"):
        pipeline._idle_end_reason = SESSION_END_IDLE_NORMAL
    if not hasattr(pipeline, "_idle_watchdog_controller"):
        pipeline._idle_watchdog_controller = build_full_duplex_idle_watchdog(pipeline)
    pipeline._idle_watchdog_controller.timeout_sec = pipeline._idle_timeout_sec
    pipeline._idle_watchdog_controller.disconnect_grace_sec = (
        pipeline._idle_disconnect_grace_sec
    )
