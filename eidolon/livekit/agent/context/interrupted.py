"""Interrupted assistant-response context capture and injection."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from eidolon.livekit.agent.session.assistant_speech import AssistantSpeechLedger

logger = logging.getLogger("agent.context.interrupted")

_CONTEXT_EXCERPT_MIN_PLAYED_SEC = 0.8
_CONTEXT_EXCERPT_MAX_CHARS = 120


@dataclass(frozen=True)
class InterruptedContextConsumption:
    """One-shot result of applying interrupted output to a new LLM turn."""

    outcome: str
    source: str = ""
    age_ms: float | None = None


class InterruptedContextManager:
    """Capture interrupted assistant text and consume it for one LLM turn."""

    def __init__(self) -> None:
        self.last_context: dict[str, Any] | None = None

    def snapshot(
        self,
        *,
        session: Any | None,
        factory: Any | None,
        duck_mixer: Any | None,
        config: Any,
        assistant_text: str = "",
    ) -> dict[str, Any] | None:
        """Capture the agent's response text at the point of interruption."""
        if not getattr(config, "interrupted_context_enabled", False):
            return None
        try:
            played_sec = duck_mixer.played_seconds if duck_mixer is not None else None

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
                return self.last_context

            recent_assistant_text = assistant_text.strip()
            if recent_assistant_text:
                self.last_context = {
                    "text": recent_assistant_text,
                    "timestamp": time.monotonic(),
                    "played_seconds": played_sec,
                    "source": "assistant_speech_ledger",
                }
                logger.info(
                    "[InterruptedContextManager] captured "
                    "(source=assistant_speech_ledger): text=%r played=%.2fs",
                    recent_assistant_text[:80],
                    played_sec or 0.0,
                )
                return self.last_context

            if not getattr(config, "interrupted_context_history_fallback_enabled", False):
                logger.info(
                    "[InterruptedContextManager] skipped history fallback: "
                    "no current TTS in-flight text"
                )
                return None

            if session is None:
                logger.info(
                    "[InterruptedContextManager] skipped history fallback: session unavailable"
                )
                return None

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
                    return self.last_context
        except Exception:
            logger.warning(
                "[InterruptedContextManager] failed to capture context",
                exc_info=True,
            )
        return None

    def consume_into(
        self,
        *,
        turn_context: Any,
        config: Any,
    ) -> InterruptedContextConsumption:
        """Apply pending context to exactly one accepted LLM turn.

        LiveKit passes a temporary ``ChatContext`` to
        ``on_user_turn_completed``. Mutating that object affects only the
        generation being admitted, which is the intended product contract: an
        interrupted reply is background for the next accepted user turn, not a
        permanent system message in conversation history.
        """

        context = self.last_context
        if context is None:
            return InterruptedContextConsumption(outcome="none")

        # Claim before touching the framework object. A failed application must
        # not leak stale context into a later, unrelated user turn.
        self.last_context = None
        source = str(context.get("source") or "")
        age = time.monotonic() - context["timestamp"]
        if age > getattr(config, "interrupted_context_max_age_sec", 0.0):
            logger.info(
                "[InterruptedContextManager] context expired (age=%.1fs)",
                age,
            )
            return InterruptedContextConsumption(
                outcome="expired",
                source=source,
                age_ms=age * 1000.0,
            )

        interrupted_text = context["text"]
        played_sec = context.get("played_seconds")

        try:
            hint_text = self._build_hint_text(
                interrupted_text=interrupted_text,
                played_sec=played_sec,
            )
            turn_context.add_message(
                role="system",
                content=hint_text,
            )
            logger.info(
                "[InterruptedContextManager] consumed context hint (%d chars, played=%s)",
                len(interrupted_text),
                f"{played_sec:.1f}s" if played_sec is not None else "n/a",
            )
            return InterruptedContextConsumption(
                outcome="applied",
                source=source,
                age_ms=age * 1000.0,
            )
        except Exception:
            logger.warning(
                "[InterruptedContextManager] failed to consume context",
                exc_info=True,
            )
            return InterruptedContextConsumption(
                outcome="failed",
                source=source,
                age_ms=age * 1000.0,
            )

    @staticmethod
    def _build_hint_text(
        *,
        interrupted_text: str,
        played_sec: float | None,
    ) -> str:
        """Build a one-turn hint that avoids making the model repeat itself."""

        base = (
            "[系统提示] 上一轮助手回复被用户打断。"
            "请优先回答用户最新输入；除非用户明确要求继续上一轮，"
            "不要复述或主动续写被打断的内容，也不要提及这条系统提示。"
        )
        if played_sec is not None and played_sec < _CONTEXT_EXCERPT_MIN_PLAYED_SEC:
            return (
                f"{base} 用户几乎没听完整上一轮回复"
                f"（约 {played_sec:.1f} 秒），按新的用户输入重新组织回答。"
            )
        excerpt = interrupted_text.strip()[:_CONTEXT_EXCERPT_MAX_CHARS]
        if not excerpt:
            return base
        if played_sec is None:
            played_phrase = "用户听到的范围未知"
        else:
            played_phrase = f"用户大约听到了前 {played_sec:.1f} 秒"
        return (
            f"{base} {played_phrase}。仅在判断用户是在追问上一轮时，"
            f"把以下内容当作背景，不要直接复述：「{excerpt}」"
        )

    @staticmethod
    def _current_tts_text(factory: Any | None) -> str:
        return AssistantSpeechLedger.current_tts_text(factory)
