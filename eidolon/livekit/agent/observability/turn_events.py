"""Non-blocking projection of Channel turn facts into ``eidolon_data.events``.

The voice hot path owns decisions; this sink only observes them.  Producers use
``put_nowait`` and never await SQLite/NATS/network I/O.  A bounded background
writer persists a small semantic vocabulary that Mission Control can replay.
Raw transcript/audio never leaves the local per-turn timeline.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eidolon_data import DataStore
from eidolon_data import load_settings as load_data_settings
from eidolon_sdk.biz.contracts import (
    SESSION_FLOW_ID_FIELD,
    normalize_session_flow_id,
)

from ..runtime.resolver import _participant_identity_and_metadata
from .timeline import TurnTimeline

logger = logging.getLogger("agent.observability.turn_events")

_QUEUE_MAX = 256
_MILESTONE_MARKS = {
    "speech_started": "speech_started_at",
    "speech_stopped": "speech_stopped_at",
    "transcript": "transcript_actionable_first_at",
    "framework_completed": "framework_completed_turn_at",
    "turn_committed": "turn_committed_at",
    "brain_first_delta": "brain_first_delta_at",
    "first_audio": "tts_first_audio_at",
    "playback_done": "agent_audio_playback_done_at",
}


@dataclass(frozen=True)
class ChannelEventContext:
    owner_id: str
    companion_id: str
    device_id: str | None
    room_name: str
    session_flow_id: str | None


@dataclass(frozen=True)
class _PendingEvent:
    event_type: str
    subject_type: str
    subject_id: str
    trace_id: str | None
    severity: str | None
    outcome: str | None
    reason: str | None
    payload: dict[str, Any]
    event_id: str | None = None


class ChannelTurnEventSink:
    """Best-effort Channel event writer with bounded, non-blocking producers."""

    def __init__(self, *, queue_max: int = _QUEUE_MAX) -> None:
        self._queue: asyncio.Queue[_PendingEvent | None] = asyncio.Queue(maxsize=queue_max)
        self._store: DataStore | None = None
        self._context: ChannelEventContext | None = None
        self._writer: asyncio.Task[None] | None = None
        self._phase_seq: dict[str, int] = {}
        self._milestone_seq: dict[str, int] = {}
        self._terminal_turns: set[str] = set()
        self._dropped = 0

    @property
    def enabled(self) -> bool:
        return self._context is not None and self._writer is not None

    @property
    def dropped_count(self) -> int:
        return self._dropped

    async def start(self, room: Any) -> None:
        """Resolve the room identity once and start the dedicated DB writer."""

        if self._writer is not None:
            return
        store: DataStore | None = None
        try:
            settings = load_data_settings()
            sqlite_path = Path(settings.sqlite_path).expanduser()
            if not sqlite_path.exists():
                logger.warning("Channel turn events disabled: data DB missing at %s", sqlite_path)
                return
            store = DataStore.open(settings)
            context = await _resolve_event_context(store, room)
        except Exception as exc:  # noqa: BLE001 - observability must not break voice
            if store is not None:
                await store.close()
            logger.warning("Channel turn events disabled: %s", exc)
            return
        self._store = store
        self._context = context
        self._writer = asyncio.create_task(self._run_writer(), name="channel-turn-events")
        self._enqueue(
            _PendingEvent(
                event_type="channel.session.started",
                subject_type="device" if context.device_id else "companion",
                subject_id=context.device_id or context.companion_id,
                trace_id=context.session_flow_id,
                severity=None,
                outcome=None,
                reason="session_started",
                payload=self._base_payload(),
            )
        )

    async def close(self, *, reason: str = "session_ended") -> None:
        writer = self._writer
        context = self._context
        if writer is None or context is None:
            return
        session_failed = reason == "session_error"
        self._enqueue(
            _PendingEvent(
                event_type="channel.session.failed" if session_failed else "channel.session.ended",
                subject_type="device" if context.device_id else "companion",
                subject_id=context.device_id or context.companion_id,
                trace_id=context.session_flow_id,
                severity="error" if session_failed else None,
                outcome="failure" if session_failed else None,
                reason=reason,
                payload={**self._base_payload(), "dropped_event_count": self._dropped},
            )
        )
        try:
            await asyncio.wait_for(self._queue.join(), timeout=2.0)
        except asyncio.TimeoutError:
            logger.warning("Channel turn event drain timed out pending=%s", self._queue.qsize())
        try:
            await asyncio.wait_for(self._queue.put(None), timeout=1.0)
            await asyncio.wait_for(writer, timeout=1.0)
        except asyncio.TimeoutError:
            writer.cancel()
            await asyncio.gather(writer, return_exceptions=True)
        if self._store is not None:
            await self._store.close()
        self._writer = None
        self._store = None
        self._context = None

    def phase_changed(
        self,
        *,
        timeline: TurnTimeline | None,
        previous_phase: str,
        phase: str,
        event: str,
        reason: str,
        side_effect: str,
        occurred_at: float,
        details: dict[str, object] | None = None,
    ) -> None:
        if timeline is None or not self.enabled:
            return
        turn_id = timeline.turn_id
        seq = self._phase_seq.get(turn_id, 0) + 1
        self._phase_seq[turn_id] = seq
        payload = {
            **self._base_payload(),
            "channel_turn_id": turn_id,
            "phase": phase,
            "previous_phase": previous_phase,
            "transition_seq": seq,
            "transition_event": event,
            "side_effect": side_effect,
            "elapsed_ms": _elapsed_ms(timeline, occurred_at),
        }
        if details:
            payload["details"] = _safe_details(details)
        self._enqueue(
            _PendingEvent(
                event_type="channel.turn.phase_changed",
                subject_type="turn",
                subject_id=turn_id,
                trace_id=turn_id,
                severity=None,
                outcome="deferred" if phase in {"user_turn_pending", "evidence_arbitration"} else None,
                reason=reason,
                payload=payload,
                event_id=f"evt_ch_phase_{turn_id}_{seq}",
            )
        )

    def milestone(self, timeline: TurnTimeline | None, milestone: str, *, reason: str = "") -> None:
        if timeline is None or not self.enabled:
            return
        turn_id = timeline.turn_id
        seq = self._milestone_seq.get(turn_id, 0) + 1
        self._milestone_seq[turn_id] = seq
        now = time.monotonic()
        brain = dict(timeline.attrs.get("brain_rpc") or {})
        payload = {
            **self._base_payload(),
            "channel_turn_id": turn_id,
            "milestone": milestone,
            "milestone_seq": seq,
            "elapsed_ms": _elapsed_ms(timeline, now),
            "brain_turn_id": str(brain.get("turn_id") or "") or None,
            "conversation_id": str(brain.get("conversation_id") or "") or None,
        }
        self._enqueue(
            _PendingEvent(
                event_type="channel.turn.milestone",
                subject_type="turn",
                subject_id=turn_id,
                trace_id=turn_id,
                severity=(
                    "error"
                    if milestone in {"brain_error", "llm_error", "tts_error", "session_error"}
                    else None
                ),
                outcome=(
                    "failure"
                    if milestone in {"brain_error", "llm_error", "tts_error", "session_error"}
                    else None
                ),
                reason=reason or milestone,
                payload=payload,
                event_id=f"evt_ch_mark_{turn_id}_{seq}",
            )
        )

    def terminal(self, timeline: TurnTimeline | None, reason: str) -> None:
        if timeline is None or not self.enabled or timeline.turn_id in self._terminal_turns:
            return
        self._terminal_turns.add(timeline.turn_id)
        snapshot = timeline.snapshot()
        attrs = snapshot.get("attrs") or {}
        phase = str(((attrs.get("full_duplex_state") or {}).get("phase")) or "")
        status, event_type, severity, outcome = _terminal_classification(timeline, phase, reason)
        brain = dict(attrs.get("brain_rpc") or {})
        missing = [name for name, mark in _MILESTONE_MARKS.items() if mark not in timeline.timestamps]
        payload = {
            **self._base_payload(),
            "channel_turn_id": timeline.turn_id,
            "brain_turn_id": str(brain.get("turn_id") or "") or None,
            "conversation_id": str(brain.get("conversation_id") or "") or None,
            "phase": phase,
            "status": status,
            "terminal_reason": reason,
            "durations_ms": snapshot.get("durations_ms") or {},
            "missing_milestones": missing,
            "dropped_event_count": self._dropped,
            "unexpected_transition_count": int(
                attrs.get("full_duplex_unexpected_transition_count") or 0
            ),
        }
        self._enqueue(
            _PendingEvent(
                event_type=event_type,
                subject_type="turn",
                subject_id=timeline.turn_id,
                trace_id=timeline.turn_id,
                severity=severity,
                outcome=outcome,
                reason=reason,
                payload=payload,
                event_id=f"evt_ch_terminal_{timeline.turn_id}",
            )
        )

    def _base_payload(self) -> dict[str, Any]:
        context = self._context
        if context is None:
            return {}
        return {
            "room_name": context.room_name,
            "device_id": context.device_id,
        }

    def _enqueue(self, event: _PendingEvent) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self._dropped += 1
            logger.warning("Dropped Channel turn event type=%s dropped=%s", event.event_type, self._dropped)

    async def _run_writer(self) -> None:
        while True:
            pending = await self._queue.get()
            try:
                if pending is None:
                    return
                await self._write(pending)
            except Exception:  # noqa: BLE001 - persistence cannot affect the session
                logger.exception("Failed to persist Channel event type=%s", getattr(pending, "event_type", ""))
            finally:
                self._queue.task_done()

    async def _write(self, pending: _PendingEvent) -> None:
        store = self._store
        context = self._context
        if store is None or context is None:
            return
        await store.events.record_event(
            event_type=pending.event_type,
            event_id=pending.event_id,
            owner_id=context.owner_id,
            companion_id=context.companion_id,
            subject_type=pending.subject_type,
            subject_id=pending.subject_id,
            actor_type="service",
            actor_id="eidolon-channel",
            trace_id=pending.trace_id,
            severity=pending.severity,
            outcome=pending.outcome,
            reason=pending.reason,
            data_classification="safe",
            schema_version=1,
            payload_json=pending.payload,
        )


async def _resolve_event_context(store: DataStore, room: Any) -> ChannelEventContext:
    participant = _participant_identity_and_metadata(room)
    if participant is None:
        raise RuntimeError("runtime participant missing")
    identity, metadata = participant
    kind = str(metadata.get("kind") or "").strip().lower()
    device_id: str | None = None
    if kind == "device":
        device_id = str(metadata.get("device_id") or identity).strip()
        device = await store.devices.get_device(device_id)
        if device is None or not device.owner_id or not device.bound_companion_id:
            raise RuntimeError(f"device {device_id!r} is not claimed and bound")
        owner_id = device.owner_id
        companion_id = device.bound_companion_id
    elif kind in {"owner", "user"}:
        owner_id = str(
            metadata.get("owner_id") or metadata.get("user_id") or identity
        ).strip()
        companions = [
            row
            for row in await store.companions.list_for_owner(owner_id)
            if row.status == "active"
            and row.default_memory_realm_id
            and row.current_genome_id
        ]
        if not companions:
            raise RuntimeError(
                f"owner {owner_id!r} has no active companion with memory/genome"
            )
        companion_id = companions[0].companion_id
    else:
        raise RuntimeError(f"unsupported runtime participant kind {kind!r}")
    return ChannelEventContext(
        owner_id=owner_id,
        companion_id=companion_id,
        device_id=device_id,
        room_name=str(getattr(room, "name", "") or ""),
        session_flow_id=normalize_session_flow_id(
            str(metadata.get(SESSION_FLOW_ID_FIELD) or "")
        ),
    )


def _elapsed_ms(timeline: TurnTimeline, at: float) -> float:
    start = timeline.timestamps.get("speech_started_at")
    if not isinstance(start, (int, float)):
        return 0.0
    return round(max(0.0, (at - start) * 1000), 1)


def _safe_details(details: dict[str, object]) -> dict[str, object]:
    """Keep only bounded scalar decision evidence; never transcript text."""

    allowed = {"source", "action", "vad_active", "eot_score", "state", "verdict"}
    return {
        key: value
        for key, value in details.items()
        if key in allowed and isinstance(value, (str, int, float, bool, type(None)))
    }


def _terminal_classification(
    timeline: TurnTimeline,
    phase: str,
    reason: str,
) -> tuple[str, str, str | None, str | None]:
    if phase == "user_turn_rejected":
        return "rejected", "channel.turn.rejected", "warn", "denied"
    if reason.startswith("interrupted_"):
        return "interrupted", "channel.turn.completed", None, None
    if "llm_error_at" in timeline.timestamps or "brain_error_at" in timeline.timestamps:
        return "failed", "channel.turn.failed", "error", "failure"
    lowered = reason.lower()
    if any(token in lowered for token in ("error", "failed", "timeout")):
        return "failed", "channel.turn.failed", "error", "failure"
    return "completed", "channel.turn.completed", None, None


__all__ = ["ChannelEventContext", "ChannelTurnEventSink"]
