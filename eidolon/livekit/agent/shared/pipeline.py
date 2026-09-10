"""Base pipeline — shared state and logic for all duplex implementations.

============================================================================
ARCHITECTURE DECISION (ADR — why Stage wrappers exist)
============================================================================

The ``providers/`` subpackage wraps each plugin (STT, TTS, VAD, LLM) in
a Stage class with ``warmup()`` / ``shutdown()`` methods. This may look
like boilerplate, but the framework doesn't manage warmup itself:

  * Framework's ``AgentSession.start(agent, room, ...)`` accepts raw
    plugin instances and connects them to the room I/O. It does NOT
    call ``plugin.prewarm()`` or any equivalent.
  * Our SenseTime STT/TTS plugins maintain persistent WebSocket
    connections that MUST be opened (with ``task_start`` handshake)
    before the first user audio frame arrives — otherwise first-token
    latency is unacceptable.
  * Our FireRed VAD has an ONNX session that must be loaded once.

Stage wrappers give us a uniform place to call ``warmup()`` BEFORE
``session.start()`` and ``shutdown()`` AFTER ``session.aclose()``.

The wrappers are thin (mostly delegation + lifecycle hooks). If
livekit-agents adds an official prewarm hook, the wrappers should be
collapsed to a single ``LifecycleStage`` helper.
============================================================================
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from eidolon.livekit.common.config import ObservabilityConfig

from ..factory import SharedStageFactory
from ..observability.session_trace import SessionTraceSettings, SessionTraceWriter
from .types import PipelineCallbacks, PipelineState

logger = logging.getLogger("agent")


class BasePipeline(ABC):
    """Abstract base for all pipeline implementations.

    Subclasses must implement :meth:`run`, which sets up the room event
    handlers and blocks until the room disconnects. The shared shutdown
    sequence is provided by :meth:`shutdown`.

    Shared state:
        - ``_factory``: the :class:`SharedStageFactory` providing stt/llm/tts/vad
        - ``_callbacks``: :class:`PipelineCallbacks` for external event notification
        - ``_state``: current :class:`PipelineState`
        - ``_room``: the connected :class:`Room`, or None
        - ``_started``: True once :meth:`run` has been called
    """

    def __init__(
        self,
        factory: SharedStageFactory,
        *,
        callbacks: PipelineCallbacks | None = None,
    ) -> None:
        self._factory = factory
        self._callbacks = callbacks or PipelineCallbacks()
        self._state: PipelineState = PipelineState.IDLE
        self._room: "Room | None" = None
        self._started: bool = False
        self._session_trace: SessionTraceWriter | None = None
        self._pending_session_marks: list[tuple[str, float, dict[str, Any]]] = []

    @property
    def factory(self) -> SharedStageFactory:
        return self._factory

    # -------------------------------------------------------------------------
    # Session-scoped trace (room join -> leave)
    #
    # It lives here, not in the full-duplex pipeline, because a session is not a
    # turn-taking concept: PTT enters and leaves a room the same way and has no
    # ``ChannelTurnEventSink`` at all. Both duplex implementations inherit the
    # open/close, so "how long did joining take" is answerable in either mode.
    #
    # Turn-scoped facts stay where they are. In particular the interruption
    # clock is not this clock: after a barge-in, playback resumes one
    # ``speech_merge_grace_ms`` after VAD stop while the candidate keeps its own
    # evidence deadline for seconds longer. Session marks must not be read as
    # either of those.
    # -------------------------------------------------------------------------

    def session_mark(self, name: str, **fields: Any) -> None:
        """Record one session milestone, buffering it until the writer exists.

        Callable from the first line of ``run()``: the marks that happen before
        the Owner and Companion are known (and therefore before the file can be
        named) are held with their own timestamps and replayed at open.
        """

        writer = getattr(self, "_session_trace", None)
        if writer is not None:
            writer.session_mark(name, **fields)
            return
        # ``getattr`` rather than the attribute: focused tests build a pipeline
        # through ``object.__new__`` and never run this constructor, and a
        # session mark must not be the thing that makes one of them fail.
        pending = getattr(self, "_pending_session_marks", None)
        if pending is None:
            pending = self._pending_session_marks = []
        if len(pending) < 32:
            pending.append((name, time.monotonic(), dict(fields)))

    def open_session_trace(
        self,
        room: "Room | None",
        *,
        observability: ObservabilityConfig | None = None,
        owner_id: str = "",
        companion_id: str = "",
        interaction_mode: str = "",
    ) -> None:
        """Open this session's trace file. Safe to call once identity is known."""

        if getattr(self, "_session_trace", None) is not None:
            return
        config = observability or getattr(self, "_observability", None) or ObservabilityConfig()
        writer = SessionTraceWriter.open(
            settings=SessionTraceSettings(
                root=config.session_trace_path,
                max_queue=config.session_trace_max_queue,
                max_file_bytes=config.session_trace_max_file_bytes,
                retention_days=config.session_trace_retention_days,
            ),
            session_id=getattr(self._factory, "runtime_session_id", "") or "",
            owner_id=owner_id,
            companion_id=companion_id,
            room_name=str(getattr(room, "name", "") or ""),
            interaction_mode=interaction_mode,
        )
        self._session_trace = writer
        pending = getattr(self, "_pending_session_marks", None) or []
        self._pending_session_marks = []
        for name, at, fields in pending:
            writer.session_mark(name, at=at, **fields)

    def close_session_trace(self, reason: str) -> None:
        writer = getattr(self, "_session_trace", None)
        self._session_trace = None
        if writer is not None:
            writer.close(reason=reason)

    @property
    def state(self) -> PipelineState:
        return self._state

    @abstractmethod
    async def run(self, room: "Room") -> None:
        """Start the pipeline. Blocks until the room disconnects.

        Subclasses must:
        1. Store ``room`` in ``self._room``
        2. Set ``self._started = True``
        3. Register room event handlers
        4. Run the main loop (e.g. ``while room.isconnected``)
        5. Call ``await self.shutdown()`` before returning
        """
        ...

    async def shutdown(self) -> None:
        """Shared graceful shutdown — resets state and marks pipeline as stopped."""
        logger.info("[BasePipeline] shutting down")
        self.close_session_trace(getattr(self, "_session_trace_close_reason", "session_ended"))
        self._state = PipelineState.IDLE
        self._started = False

    # -------------------------------------------------------------------------
    # Generic stage lifecycle (warmup / shutdown)
    #
    # Stages with persistent network connections (e.g. SenseTime STT/TTS)
    # expose ``warmup()`` / ``shutdown()`` methods. Per-stream stages (e.g.
    # Bailian STT) don't — the ``hasattr`` checks make this a no-op for them.
    # -------------------------------------------------------------------------

    def _lifecycle_stages(self) -> list[Any]:
        """Stages that may have warmup/shutdown hooks.

        Subclasses can override to add new stages without touching the
        warmup/shutdown plumbing. Stages without warmup/shutdown methods
        are safely no-op'd via ``hasattr`` checks in ``_warmup_stages`` /
        ``_shutdown_stages``.
        """
        stages: list[Any] = [self._factory.stt, self._factory.tts]
        # VAD is wrapped in VadStage (mirrors SttStage/TtsStage). Its
        # warmup/shutdown forward to the underlying VAD plugin only if the
        # plugin defines them; for plugins that preload at load() time
        # (e.g. FireredPvadVAD), this is effectively a no-op.
        if self._factory.vad is not None:
            stages.append(self._factory.vad)
        voiceprint = getattr(self._factory, "voiceprint_provider", None)
        if voiceprint is not None:
            stages.append(voiceprint)
        return stages

    async def _warmup_stages(self) -> None:
        """Call ``warmup()`` on each lifecycle stage that supports it.

        Failures in any single stage are logged but do not abort the pipeline:
        a TTS warmup miss is recoverable (first stream just pays the connect
        cost), and we want STT to still warm up even if TTS failed.
        """
        for stage in self._lifecycle_stages():
            if not hasattr(stage, "warmup"):
                warm_up = getattr(stage, "warm_up", None)
                if warm_up is None:
                    continue
                try:
                    await warm_up()
                except Exception:
                    logger.exception(
                        "[BasePipeline] %s warm_up failed",
                        type(stage).__name__,
                    )
                continue
            try:
                await stage.warmup()
            except Exception:
                logger.exception(
                    "[BasePipeline] %s warmup failed",
                    type(stage).__name__,
                )

    async def _shutdown_stages(self) -> None:
        """Call ``shutdown()`` on each lifecycle stage that supports it.

        Continues on error so one stage's misbehaviour can't strand others.
        """
        for stage in self._lifecycle_stages():
            if not hasattr(stage, "shutdown"):
                continue
            try:
                await stage.shutdown()
            except Exception:
                logger.exception(
                    "[BasePipeline] %s shutdown failed",
                    type(stage).__name__,
                )

    # -------------------------------------------------------------------------
    # Shared session event handlers
    # -------------------------------------------------------------------------

    def _on_agent_state_changed(self, event: Any) -> None:
        """Handle agent state change events from AgentSession.

        Shared by all pipelines that use AgentSession (e.g. StreamingPipeline).
        """
        try:
            old = event.old_state
            new = event.new_state
            logger.info("[BasePipeline] agent_state: %s -> %s", old, new)
            if new == "speaking":
                self._state = PipelineState.SPEAKING
                self._callbacks.on_agent_started_speaking()
            elif new == "thinking":
                self._state = PipelineState.GENERATING
            elif new in ("idle", "listening"):
                # G1 fix (2026-05-16): the framework's quiet state is reported
                # as "listening", not "idle" — so self._state was permanently
                # stuck at SPEAKING after the first turn, defeating F3.2's
                # output-flow duck guard. Accept both names for forward-compat
                # in case the framework ever sends "idle".
                self._state = PipelineState.IDLE
                self._callbacks.on_agent_ended_speaking()
                self._callbacks.on_agent_response_done()
        except Exception:
            logger.exception("[BasePipeline] error in _on_agent_state_changed")

    def _on_user_transcribed(self, event: Any) -> None:
        """Handle user transcription events from AgentSession.

        Shared by all pipelines that use AgentSession.
        """
        try:
            if event.is_final and event.transcript:
                logger.debug("[EOT-DBG] FINAL transcript received: %r", event.transcript)
                self._callbacks.on_user_message(event.transcript)
            elif event.is_final and not event.transcript:
                logger.debug("[EOT-DBG] FINAL transcript received but EMPTY")
            elif event.transcript:
                logger.debug("[EOT-DBG] INTERIM transcript: %r", event.transcript)
        except Exception:
            logger.exception("[BasePipeline] error in _on_user_transcribed")

    def _on_session_error(self, event: Any) -> None:
        """Handle error events from AgentSession.

        Shared by all pipelines that use AgentSession.
        """
        try:
            err = getattr(event, "error", event)
            logger.error("[BasePipeline] session error: %s", err)
            self._callbacks.on_error(err)
        except Exception:
            logger.exception("[BasePipeline] error in _on_session_error")


if TYPE_CHECKING:
    from livekit.rtc import Room
