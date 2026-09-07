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
    """Align SDK endpointing with a Channel-owned interruption candidate.

    LiveKit checks whether the current speech can be interrupted *before* its
    completed-turn hook. A Channel cancellation must finish its SpeechHandle
    before SDK reply scheduling, including when no model intent is configured.
    Waiting only in that hook loses the reply or slow intent results.
    The existing detector still owns the EOT probability and endpointing delay.
    """

    def __init__(
        self, detector: Any, settle: Callable[[], Awaitable[None]],
        *, resolve_transcript: Callable[[str], str] | None = None,
    ) -> None:
        self.eot_model = detector
        self._settle = settle
        self._resolve_transcript = resolve_transcript

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
        if self._resolve_transcript is not None:
            # Snapshot before awaiting: a later candidate must not rewrite an
            # earlier SDK query. Copy both container and message; ChatContext's
            # public copy() deliberately shares its message objects.
            for index in range(len(chat_ctx.items) - 1, -1, -1):
                message = chat_ctx.items[index]
                if message.type != "message" or message.role != "user":
                    continue
                text = message.text_content or ""
                resolved = self._resolve_transcript(text)
                if resolved != text:
                    chat_ctx = chat_ctx.copy()
                    chat_ctx.items[index] = message.model_copy(update={"content": [resolved]})
                break
        await self._settle()
        return await self.eot_model.predict_end_of_turn(chat_ctx, timeout=timeout)
