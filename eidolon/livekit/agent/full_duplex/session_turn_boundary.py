"""Session boundary effects for full-duplex user-turn completion."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..observability import TurnTimeline

if TYPE_CHECKING:
    from .pipeline import StreamingPipeline

logger = logging.getLogger("agent")


class FullDuplexSessionTurnBoundary:
    """Apply user-turn side effects at the AgentSession / LLM boundary."""

    def __init__(self, pipeline: StreamingPipeline) -> None:
        self._pipeline = pipeline

    def publish_canonical_user_text(
        self,
        new_message: Any,
        transcript: str,
        *,
        source: str,
        timeline: TurnTimeline | None,
    ) -> None:
        stripped = transcript.strip()
        if not stripped:
            return
        try:
            # LiveKit explicitly documents ``on_user_turn_completed`` as the
            # public boundary where user code may edit ``new_message`` before
            # it reaches the LLM and framework ChatContext.  Publishing the
            # canonical turn here keeps every LLM adapter and future EOT
            # history aligned without a provider-specific override channel.
            new_message.content = [stripped]
            if timeline is not None:
                timeline.set_attr(
                    "canonical_user_text",
                    {
                        "source": source,
                        "text_preview": stripped[:120],
                        "text_length": len(stripped),
                    },
                )
        except Exception:
            logger.debug(
                "[StreamingPipeline] failed to edit framework user message",
                exc_info=True,
            )

    def consume_interrupted_context(
        self,
        turn_ctx: Any,
        *,
        timeline: TurnTimeline | None,
    ) -> None:
        """Apply one-shot interrupted output context to this generation only."""

        self._pipeline._ensure_context_ledger().consume_for_turn(
            turn_context=turn_ctx,
            timeline=timeline,
        )

    def notify_context_error_once(self, reason: str) -> None:
        pipeline = self._pipeline
        if getattr(pipeline, "_context_error_notified", False):
            return
        pipeline._context_error_notified = True
        logger.error(
            "[StreamingPipeline] conversation blocked: session context unresolved "
            "(reason=%s). The user/device likely references a missing agent "
            "binding; turns are dropped until it is rebound in admin.",
            reason,
        )
        session = getattr(pipeline, "_session", None)
        say = getattr(session, "say", None) if session is not None else None
        if not callable(say):
            return
        try:
            say(
                "抱歉，我暂时无法连接到你的助手，请检查账号绑定或联系管理员。",
                allow_interruptions=True,
            )
        except Exception:
            logger.exception("[StreamingPipeline] context-error fallback announcement failed")
            return
        try:
            pipeline._mark_activity()
        except Exception:
            logger.debug(
                "[StreamingPipeline] mark_activity after context-error say failed",
                exc_info=True,
            )
