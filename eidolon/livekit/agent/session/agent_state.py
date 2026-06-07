"""Agent state transition side effects for streaming sessions."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.output import FillerManager, OutputDuckingController

logger = logging.getLogger("agent.session.agent_state")


class AgentStateEffectHandler:
    """Apply session-side effects after BasePipeline mirrors agent state."""

    def __init__(
        self,
        *,
        get_timeline: Callable[[], TurnTimeline | None],
        mark_activity: Callable[[], None],
        cancel_soft_interrupt: Callable[[], None],
        soft_interrupt_active: Callable[[], bool],
        ducking: OutputDuckingController,
        get_filler: Callable[[], FillerManager | None],
        flush_timeline_debug: Callable[[str, bool], None],
    ) -> None:
        self._get_timeline = get_timeline
        self._mark_activity = mark_activity
        self._cancel_soft_interrupt = cancel_soft_interrupt
        self._soft_interrupt_active = soft_interrupt_active
        self._ducking = ducking
        self._get_filler = get_filler
        self._flush_timeline_debug = flush_timeline_debug

    def handle(self, event: Any) -> None:
        old_state = event.old_state
        new_state = event.new_state

        if new_state in ("thinking", "speaking"):
            self._mark_activity()
            filler = self._get_filler()
            if filler is not None:
                filler.cancel()

        self._mark_timeline(old_state, new_state)

        if new_state in ("thinking", "speaking") and self._soft_interrupt_active():
            logger.info(
                "[AgentStateEffectHandler] agent started %s; "
                "cancelling pending soft interrupt",
                new_state,
            )
            self._cancel_soft_interrupt()

        if new_state == "speaking":
            self._ducking.on_agent_started_speaking()

        if new_state == "thinking" and self._ducking.reset_if_cancelled():
            logger.info(
                "[AgentStateEffectHandler] OutputController CANCELLED->NORMAL "
                "(new turn starting; clearing prior-interrupt state)"
            )

    def _mark_timeline(self, old_state: str, new_state: str) -> None:
        timeline = self._get_timeline()
        if timeline is None:
            return
        if new_state == "thinking":
            timeline.mark("llm_started_at")
        elif new_state == "speaking":
            timeline.mark("tts_first_audio_at")
        elif old_state == "speaking" and new_state in ("idle", "listening"):
            timeline.mark("agent_audio_playback_done_at")
            self._flush_timeline_debug("agent_audio_playback_done", True)
