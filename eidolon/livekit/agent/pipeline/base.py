"""Base pipeline — shared state and logic for all pipeline implementations.

============================================================================
ARCHITECTURE DECISION (ADR — why Stage wrappers exist)
============================================================================

The ``pipeline/`` subpackage wraps each plugin (STT, TTS, VAD, LLM) in
a Stage class with ``warmup()`` / ``shutdown()`` methods. This may look
like boilerplate, but framework doesn't manage warmup itself:

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
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ..factory import SharedStageFactory
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

    @property
    def factory(self) -> SharedStageFactory:
        return self._factory

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
                # state-guard in _duck_and_arm_timeout. Accept both names for
                # forward-compat in case the framework ever sends "idle".
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
