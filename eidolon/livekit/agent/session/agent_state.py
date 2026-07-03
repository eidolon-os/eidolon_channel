"""Agent state transition side effects for streaming sessions."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.output import FillerManager, OutputDuckingController
from .agent_output_coordinator import AgentOutputCoordinator

logger = logging.getLogger("agent.session.agent_state")


@dataclass(frozen=True)
class AgentStateTransition:
    old_state: str = ""
    new_state: str = ""

    @classmethod
    def from_event(cls, event: Any) -> AgentStateTransition:
        if isinstance(event, cls):
            return event
        return cls(
            old_state=getattr(event, "old_state", "") or "",
            new_state=getattr(event, "new_state", "") or "",
        )

    @property
    def starts_output_activity(self) -> bool:
        return self.new_state in ("thinking", "speaking")

    @property
    def starts_playback(self) -> bool:
        return self.new_state == "speaking"

    @property
    def starts_generation(self) -> bool:
        return self.new_state == "thinking"

    @property
    def completes_playback(self) -> bool:
        return self.old_state == "speaking" and self.new_state in ("idle", "listening")


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
        should_flush_on_playback_done: Callable[[], bool] | None = None,
        agent_output: AgentOutputCoordinator | None = None,
    ) -> None:
        self._get_timeline = get_timeline
        self._mark_activity = mark_activity
        self._cancel_soft_interrupt = cancel_soft_interrupt
        self._soft_interrupt_active = soft_interrupt_active
        self._ducking = ducking
        self._get_filler = get_filler
        self._flush_timeline_debug = flush_timeline_debug
        self._should_flush_on_playback_done = (
            should_flush_on_playback_done or (lambda: True)
        )
        self._agent_output = agent_output or AgentOutputCoordinator()

    def handle(self, event: Any) -> None:
        transition = AgentStateTransition.from_event(event)

        if transition.starts_output_activity:
            self._mark_activity()
            filler = self._get_filler()
            if filler is not None:
                filler.cancel()

        self._mark_timeline(transition)

        if transition.starts_output_activity and self._soft_interrupt_active():
            logger.info(
                "[AgentStateEffectHandler] agent started %s; "
                "cancelling pending soft interrupt",
                transition.new_state,
            )
            self._cancel_soft_interrupt()

        if transition.starts_playback:
            self._ducking.on_agent_started_speaking()

        if transition.starts_generation and self._ducking.reset_if_cancelled():
            logger.info(
                "[AgentStateEffectHandler] OutputController CANCELLED->NORMAL "
                "(new turn starting; clearing prior-interrupt state)"
            )

    def _mark_timeline(self, transition: AgentStateTransition) -> None:
        timeline = self._get_timeline()
        if timeline is None:
            return
        if transition.starts_generation:
            timeline.mark("llm_started_at")
        elif transition.starts_playback:
            timeline.mark("tts_first_audio_at")
        elif transition.completes_playback:
            timeline.mark("agent_audio_playback_done_at")
            if self._should_flush_on_playback_done():
                self._flush_timeline_debug("agent_audio_playback_done", True)
        self._agent_output.record_agent_state(
            timeline,
            old_state=transition.old_state,
            new_state=transition.new_state,
        )
