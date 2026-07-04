"""Track assistant speech text that is relevant to echo/context guards."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("agent.session.assistant_speech")


@dataclass(frozen=True)
class AssistantSpeechSnapshot:
    text: str
    source: str
    timestamp: float


class AssistantSpeechLedger:
    """Small text ledger for the assistant output currently heard by the user.

    Provider ``current_pushed_text`` remains the primary source for streaming
    replies. The ledger fills the gap for Channel-issued fixed speech such as
    the welcome line, where LiveKit may start playback before provider text is
    readable from the active synth stream.
    """

    def __init__(self, *, clock: Any | None = None) -> None:
        self._clock = clock or time.monotonic
        self._latest: AssistantSpeechSnapshot | None = None

    @property
    def latest(self) -> AssistantSpeechSnapshot | None:
        return self._latest

    def record(self, text: str, *, source: str) -> None:
        stripped = text.strip()
        if not stripped:
            return
        self._latest = AssistantSpeechSnapshot(
            text=stripped,
            source=source,
            timestamp=float(self._clock()),
        )

    def current_or_recent_text(
        self,
        *,
        factory: Any | None = None,
        max_age_ms: int = 0,
    ) -> str:
        current = self.current_tts_text(factory)
        if current.strip():
            self.record(current, source="tts_in_flight")
            return current
        latest = self._latest
        if latest is None or max_age_ms <= 0:
            return ""
        age_ms = (float(self._clock()) - latest.timestamp) * 1000.0
        if age_ms > max_age_ms:
            logger.debug(
                "[AssistantSpeechLedger] skipped stale assistant text "
                "source=%s age_ms=%.0f max_age_ms=%d",
                latest.source,
                age_ms,
                max_age_ms,
            )
            return ""
        return latest.text

    @staticmethod
    def current_tts_text(factory: Any | None) -> str:
        try:
            if factory is not None and getattr(factory, "tts", None) is not None:
                tts_plugin = factory.tts.tts
                return getattr(tts_plugin, "current_pushed_text", "") or ""
        except Exception:
            logger.debug(
                "[AssistantSpeechLedger] could not read TTS current_pushed_text",
                exc_info=True,
            )
        return ""
