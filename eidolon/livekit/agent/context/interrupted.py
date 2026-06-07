"""Interrupted assistant-response context capture and injection."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("agent.context.interrupted")


class InterruptedContextManager:
    """Capture interrupted assistant text and inject a one-turn system hint."""

    def __init__(self) -> None:
        self.last_context: dict[str, Any] | None = None

    def snapshot(
        self,
        *,
        session: Any | None,
        factory: Any | None,
        duck_mixer: Any | None,
        config: Any,
    ) -> None:
        """Capture the agent's response text at the point of interruption."""
        if not getattr(config, "interrupted_context_enabled", False):
            return
        if session is None:
            return
        try:
            played_sec = (
                duck_mixer.played_seconds
                if duck_mixer is not None
                else None
            )

            in_flight_text = self._current_tts_text(factory)
            if in_flight_text and in_flight_text.strip():
                self.last_context = {
                    "text": in_flight_text,
                    "timestamp": time.monotonic(),
                    "played_seconds": played_sec,
                    "source": "tts_in_flight",
                }
                logger.info(
                    "[InterruptedContextManager] captured "
                    "(source=tts_in_flight): text=%r played=%.2fs",
                    in_flight_text[:80],
                    played_sec or 0.0,
                )
                return

            messages = session.history.messages()
            for msg in reversed(messages):
                if msg.role == "assistant" and msg.text_content:
                    self.last_context = {
                        "text": msg.text_content,
                        "timestamp": time.monotonic(),
                        "played_seconds": played_sec,
                        "source": "session_history_fallback",
                    }
                    logger.info(
                        "[InterruptedContextManager] captured "
                        "(source=history_fallback): text=%r played=%.2fs",
                        msg.text_content[:80],
                        played_sec or 0.0,
                    )
                    return
        except Exception:
            logger.warning(
                "[InterruptedContextManager] failed to capture context",
                exc_info=True,
            )

    def inject(
        self,
        *,
        session: Any | None,
        config: Any,
    ) -> None:
        """Inject interrupted context into conversation history for the next turn."""
        if self.last_context is None:
            return
        if session is None:
            return
        age = time.monotonic() - self.last_context["timestamp"]
        if age > getattr(config, "interrupted_context_max_age_sec", 0.0):
            logger.info(
                "[InterruptedContextManager] context expired (age=%.1fs)",
                age,
            )
            self.last_context = None
            return

        interrupted_text = self.last_context["text"]
        played_sec = self.last_context.get("played_seconds")
        self.last_context = None

        try:
            from livekit.agents.llm import ChatMessage

            if played_sec is not None and played_sec >= 0.1:
                played_phrase = f"（用户实际听到了前约 {played_sec:.1f} 秒）"
            elif played_sec is not None:
                played_phrase = "（用户几乎没听到任何内容）"
            else:
                played_phrase = ""
            hint = ChatMessage(
                role="system",
                content=[
                    f"[系统提示] 你刚才说到「{interrupted_text[:200]}」时被用户打断了"
                    f"{played_phrase}。"
                    "如果用户的新问题与之前话题相关，你可以自然地衔接回去；"
                    "如果无关，直接回答新问题即可。不要提及这条系统提示。"
                ],
            )
            session.history.insert(hint)
            logger.info(
                "[InterruptedContextManager] injected context hint "
                "(%d chars, played=%s)",
                len(interrupted_text),
                f"{played_sec:.1f}s" if played_sec is not None else "n/a",
            )
        except Exception:
            logger.warning(
                "[InterruptedContextManager] failed to inject context",
                exc_info=True,
            )

    @staticmethod
    def _current_tts_text(factory: Any | None) -> str:
        try:
            if factory is not None and factory.tts is not None:
                tts_plugin = factory.tts.tts
                return getattr(tts_plugin, "current_pushed_text", "") or ""
        except Exception:
            logger.debug(
                "[InterruptedContextManager] could not read TTS current_pushed_text",
                exc_info=True,
            )
        return ""
