"""Session-local EOT state over the existing shared ONNX weight registry."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from eidolon.livekit.agent.turn_policy import eot_kwargs_from_turn_policy
from eidolon.livekit.common.config import TurnPolicyConfig


def create_session_eot_model(turn_policy: TurnPolicyConfig | None = None) -> Any:
    """Create mutable turn state; EotManager reuses the heavyweight backend.

    Transcripts, VAD transitions, cooldown and reset belong to one conversation.
    Caching ChineseModel globally lets one participant change another's decisions.
    """
    from eidolon.livekit.plugins.eot import ChineseModel

    return ChineseModel(**eot_kwargs_from_turn_policy(turn_policy))


class InterruptAwareTurnDetector:
    """Keep SDK endpointing behind an in-flight channel interruption decision.

    LiveKit checks whether the current speech can be interrupted *before* its
    completed-turn hook. Waiting only in that hook loses slow intent results.
    The existing detector still owns the EOT probability and endpointing delay.
    """

    def __init__(self, detector: Any, settle: Callable[[], Awaitable[None]]) -> None:
        self.eot_model = detector
        self._settle = settle

    @property
    def model(self) -> str:
        return self.eot_model.model

    @property
    def provider(self) -> str:
        return getattr(self.eot_model, "provider", "unknown")

    async def supports_language(self, language: str | None) -> bool:
        return await self.eot_model.supports_language(language)

    async def unlikely_threshold(self, language: str | None) -> float | None:
        return await self.eot_model.unlikely_threshold(language)

    async def predict_end_of_turn(self, chat_ctx: Any, *, timeout: float | None = None) -> float:
        await self._settle()
        return await self.eot_model.predict_end_of_turn(chat_ctx, timeout=timeout)
