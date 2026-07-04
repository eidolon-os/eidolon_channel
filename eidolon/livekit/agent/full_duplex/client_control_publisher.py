"""Full-duplex client-control publishing and timeline recording."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING

from eidolon_sdk.biz.contracts import (
    COMPANION_UI_STATE_TOPIC,
    CONTROL_TOPIC,
    WIRE_SCHEMA_VERSION,
)

from ..observability import TurnTimeline
from ..session.client_control import (
    append_client_control_event,
    build_client_control_event,
    build_session_client_control_envelope,
)

if TYPE_CHECKING:
    from .pipeline import StreamingPipeline

logger = logging.getLogger("agent")


class FullDuplexClientControlPublisher:
    """Publish session-local client control packets and keep timeline evidence."""

    def __init__(self, pipeline: StreamingPipeline) -> None:
        self._pipeline = pipeline

    def record_event(
        self,
        *,
        timeline: TurnTimeline | None,
        op: str,
        reason: str,
        turn_id: str,
    ) -> None:
        pipeline = self._pipeline
        event = build_client_control_event(op=op, reason=reason, turn_id=turn_id)
        if timeline is None:
            pending = list(getattr(pipeline, "_pending_client_control_events", []) or [])
            pipeline._pending_client_control_events = append_client_control_event(
                pending,
                event,
            )
            return
        events = list(timeline.attrs.get("client_control_events") or ())
        timeline.set_attr(
            "client_control_events",
            append_client_control_event(events, event),
        )

    def apply_pending(self, timeline: TurnTimeline | None = None) -> None:
        pipeline = self._pipeline
        pending = list(getattr(pipeline, "_pending_client_control_events", []) or [])
        if not pending:
            return
        timeline = timeline or getattr(pipeline, "_timeline", None)
        if timeline is None:
            return
        turn_id = getattr(timeline, "turn_id", "")
        events = list(timeline.attrs.get("client_control_events") or ())
        for event in pending:
            attached = dict(event)
            if not attached.get("turn_id"):
                attached["turn_id"] = turn_id
            events = append_client_control_event(events, attached)
        timeline.set_attr("client_control_events", events)
        pipeline._pending_client_control_events = []

    def publish_companion_ui_state(self, state: str, reason: str) -> None:
        """Best-effort state bridge for thin clients such as ESP32 displays."""

        pipeline = self._pipeline
        room = getattr(pipeline, "_room", None)
        local = getattr(room, "local_participant", None) if room else None
        if local is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        payload = {
            "schema_v": WIRE_SCHEMA_VERSION,
            "type": COMPANION_UI_STATE_TOPIC,
            "state": state,
            "reason": reason,
            "ts_ms": int(time.time() * 1000),
        }

        async def _send() -> None:
            await local.publish_data(
                json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                reliable=True,
                topic=COMPANION_UI_STATE_TOPIC,
            )

        task = loop.create_task(_send())

        def _log_failure(done: asyncio.Task[None]) -> None:
            try:
                done.result()
            except Exception:
                logger.debug(
                    "[StreamingPipeline] failed to publish companion UI state",
                    exc_info=True,
                )

        task.add_done_callback(_log_failure)

    def publish_client_control(
        self,
        op: str,
        *,
        reason: str,
        payload: dict[str, object] | None = None,
    ) -> None:
        """Best-effort session-local command for thin clients."""

        pipeline = self._pipeline
        room = getattr(pipeline, "_room", None)
        local = getattr(room, "local_participant", None) if room else None
        if local is None:
            logger.warning(
                "[StreamingPipeline] skipped client control op=%s reason=%s "
                "turn_id=%s because local participant is unavailable",
                op,
                reason,
                getattr(getattr(pipeline, "_timeline", None), "turn_id", ""),
            )
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning(
                "[StreamingPipeline] skipped client control op=%s reason=%s "
                "because no running event loop is available",
                op,
                reason,
            )
            return

        timeline = getattr(pipeline, "_timeline", None)
        turn_id = getattr(timeline, "turn_id", "") if timeline is not None else ""
        envelope = build_session_client_control_envelope(
            op=op,
            reason=reason,
            payload=payload,
            turn_id=turn_id,
        )
        control_payload = envelope["payload"]
        self.record_event(timeline=timeline, op=op, reason=reason, turn_id=turn_id)

        outcome = control_payload.get("outcome")
        logger.info(
            "[StreamingPipeline] queued client control op=%s reason=%s outcome=%s "
            "turn_id=%s topic=%s",
            op,
            reason,
            outcome,
            turn_id,
            CONTROL_TOPIC,
        )

        async def _send() -> None:
            await local.publish_data(
                json.dumps(envelope, separators=(",", ":")).encode("utf-8"),
                reliable=True,
                topic=CONTROL_TOPIC,
            )

        task = loop.create_task(_send())

        def _log_failure(done: asyncio.Task[None]) -> None:
            try:
                done.result()
            except Exception:
                logger.warning(
                    "[StreamingPipeline] failed to publish client control op=%s "
                    "reason=%s outcome=%s turn_id=%s topic=%s",
                    op,
                    reason,
                    outcome,
                    turn_id,
                    CONTROL_TOPIC,
                    exc_info=True,
                )
                return
            logger.info(
                "[StreamingPipeline] published client control op=%s reason=%s "
                "outcome=%s turn_id=%s topic=%s",
                op,
                reason,
                outcome,
                turn_id,
                CONTROL_TOPIC,
            )

        task.add_done_callback(_log_failure)
