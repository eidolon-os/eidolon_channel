"""Session boundary effects for full-duplex user-turn completion."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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
        transcript: str,
        *,
        source: str,
        timeline: TurnTimeline | None,
    ) -> None:
        pipeline = self._pipeline
        stripped = transcript.strip()
        if not stripped:
            return
        try:
            llm_plugin = getattr(getattr(pipeline._factory, "llm", None), "llm", None)
            setter = getattr(llm_plugin, "set_next_user_text", None)
            if setter is None:
                return
            setter(stripped, source=source)
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
                "[StreamingPipeline] failed to publish canonical user text",
                exc_info=True,
            )

    def clear_session_user_turn(self, reason: str) -> None:
        pipeline = self._pipeline
        self._clear_pending_canonical_user_text(reason)
        session = getattr(pipeline, "_session", None)
        if session is None:
            return
        clear_user_turn = getattr(session, "clear_user_turn", None)
        if clear_user_turn is None:
            return
        try:
            clear_user_turn()
            logger.info("[StreamingPipeline] cleared user turn reason=%s", reason)
        except Exception:
            logger.exception(
                "[StreamingPipeline] failed to clear user turn reason=%s",
                reason,
            )
        if "context_error" in (reason or ""):
            self._notify_context_error_once(reason)

    def clear_residual_audio_user_turn(self, reason: str) -> None:
        pipeline = self._pipeline
        session = getattr(pipeline, "_session", None)
        if session is None:
            return
        clear_user_turn = getattr(session, "clear_user_turn", None)
        if clear_user_turn is None:
            return
        try:
            clear_user_turn()
            logger.info(
                "[StreamingPipeline] cleared residual audio user turn reason=%s",
                reason,
            )
        except Exception:
            logger.exception(
                "[StreamingPipeline] failed to clear residual audio user turn reason=%s",
                reason,
            )

    def _clear_pending_canonical_user_text(self, reason: str) -> None:
        pipeline = self._pipeline
        try:
            factory = getattr(pipeline, "_factory", None)
            llm_plugin = getattr(getattr(factory, "llm", None), "llm", None)
            clearer = getattr(llm_plugin, "clear_next_user_text", None)
            if clearer is not None:
                clearer(reason=reason)
                return
            setter = getattr(llm_plugin, "set_next_user_text", None)
            if setter is not None:
                setter("", source=f"clear:{reason}")
        except Exception:
            logger.debug(
                "[StreamingPipeline] failed to clear canonical user text",
                exc_info=True,
            )

    def _notify_context_error_once(self, reason: str) -> None:
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
