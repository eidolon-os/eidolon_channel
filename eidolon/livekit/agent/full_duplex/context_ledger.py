"""Full-duplex interrupted context ledger adapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from eidolon.livekit.agent.context import InterruptedContextManager
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
        manager: InterruptedContextManager | None = None,
    ) -> None:
        self._get_session = get_session
        self._get_factory = get_factory
        self._get_duck_mixer = get_duck_mixer
        self._get_config = get_config
        self._get_timeline = get_timeline
        self._manager = manager or InterruptedContextManager()

    @property
    def last_context(self) -> dict[str, Any] | None:
        return self._manager.last_context

    def snapshot(self) -> None:
        """Capture interrupted assistant context and annotate the timeline."""

        self._manager.snapshot(
            session=self._get_session(),
            factory=self._get_factory(),
            duck_mixer=self._get_duck_mixer(),
            config=self._get_config(),
        )
        context = self._manager.last_context
        timeline = self._get_timeline()
        if timeline is not None and context is not None:
            timeline.set_attr(
                "interrupted_context",
                {
                    "source": context.get("source"),
                    "played_seconds": context.get("played_seconds"),
                    "text_preview": str(context.get("text") or "")[:120],
                },
            )

    def inject(self) -> None:
        """Inject captured interrupted context before committing the next turn."""

        self._manager.inject(
            session=self._get_session(),
            config=self._get_config(),
        )
