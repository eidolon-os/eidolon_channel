"""Full-duplex interrupted context ledger adapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from eidolon.livekit.agent.context import InterruptedContextManager
from eidolon.livekit.agent.context.interrupted import InterruptedContextConsumption
from eidolon.livekit.agent.observability import TurnTimeline


class FullDuplexContextLedger:
    """Bridge full-duplex runtime state into interrupted-context handling.

    The underlying :class:`InterruptedContextManager` owns capture/injection
    policy. This adapter owns the full-duplex wiring: AgentSession, TTS factory,
    ducking playback offset, config, and timeline observability.
    """

    def __init__(
        self,
        *,
        get_session: Callable[[], Any | None],
        get_factory: Callable[[], Any | None],
        get_duck_mixer: Callable[[], Any | None],
        get_config: Callable[[], Any],
        get_timeline: Callable[[], TurnTimeline | None],
        get_assistant_text: Callable[[], str] | None = None,
        manager: InterruptedContextManager | None = None,
    ) -> None:
        self._get_session = get_session
        self._get_factory = get_factory
        self._get_duck_mixer = get_duck_mixer
        self._get_config = get_config
        self._get_timeline = get_timeline
        self._get_assistant_text = get_assistant_text or (lambda: "")
        self._manager = manager or InterruptedContextManager()

    @property
    def last_context(self) -> dict[str, Any] | None:
        return self._manager.last_context

    def snapshot(self) -> None:
        """Capture interrupted assistant context and annotate the timeline."""

        captured = self._manager.snapshot(
            session=self._get_session(),
            factory=self._get_factory(),
            duck_mixer=self._get_duck_mixer(),
            config=self._get_config(),
            assistant_text=self._get_assistant_text(),
        )
        timeline = self._get_timeline()
        if timeline is not None and captured is not None:
            timeline.set_attr(
                "interrupted_context",
                {
                    "source": captured.get("source"),
                    "played_seconds": captured.get("played_seconds"),
                    "text_preview": str(captured.get("text") or "")[:120],
                },
            )

    def consume_for_turn(
        self,
        *,
        turn_context: Any,
        timeline: TurnTimeline | None,
    ) -> InterruptedContextConsumption:
        """Consume pending context at the accepted framework turn boundary."""

        result = self._manager.consume_into(
            turn_context=turn_context,
            config=self._get_config(),
        )
        if timeline is not None and result.outcome != "none":
            timeline.set_attr(
                "interrupted_context_consumption",
                {
                    "outcome": result.outcome,
                    "source": result.source,
                    "age_ms": result.age_ms,
                },
            )
        return result
