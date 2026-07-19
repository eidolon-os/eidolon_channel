"""Session boundary effects for full-duplex user-turn completion."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..observability import TurnTimeline

if TYPE_CHECKING:
    from .pipeline import StreamingPipeline

logger = logging.getLogger("agent")

_SILENT_OUTPUT_FALLBACK_TEXT = "刚才卡了一下，请再说一遍好吗？"


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

    def notify_silent_output_failure_once(
        self,
        *,
        timeline: TurnTimeline | None,
        error_type: str,
    ) -> bool:
        """Speak one local fallback for a terminal LLM failure with no answer.

        The fallback is deliberately outside the LLM and conversation context:
        it communicates product state without becoming assistant memory. It is
        interruptible, and is suppressed if the user has already started the
        next utterance or if any brain answer delta was observed.
        """

        if timeline is None or error_type != "llm_error":
            return False
        if "brain_first_answer_delta_at" in timeline.timestamps:
            return False

        fallback = dict(timeline.attrs.get("silent_failure_fallback") or {})
        if fallback.get("attempted"):
            return False

        session = getattr(self._pipeline, "_session", None)
        if session is None:
            timeline.set_attr(
                "silent_failure_fallback",
                {"attempted": True, "spoken": False, "reason": "session_unavailable"},
            )
            return False
        if str(getattr(session, "user_state", "") or "") == "speaking":
            timeline.set_attr(
                "silent_failure_fallback",
                {"attempted": True, "spoken": False, "reason": "user_speaking"},
            )
            return False

        say = getattr(session, "say", None)
        if not callable(say):
            timeline.set_attr(
                "silent_failure_fallback",
                {"attempted": True, "spoken": False, "reason": "say_unavailable"},
            )
            return False

        # Claim before calling into LiveKit so duplicate terminal events cannot
        # enqueue the same fallback twice even if say() raises.
        timeline.set_attr(
            "silent_failure_fallback",
            {"attempted": True, "spoken": False, "reason": "llm_error_without_delta"},
        )
        try:
            say(
                _SILENT_OUTPUT_FALLBACK_TEXT,
                allow_interruptions=True,
                add_to_chat_ctx=False,
            )
        except Exception:
            logger.exception("[StreamingPipeline] silent-output fallback announcement failed")
            return False

        timeline.set_attr(
            "silent_failure_fallback",
            {"attempted": True, "spoken": True, "reason": "llm_error_without_delta"},
        )
        try:
            self._pipeline._mark_activity()
        except Exception:
            logger.debug(
                "[StreamingPipeline] mark_activity after silent-output fallback failed",
                exc_info=True,
            )
        return True
