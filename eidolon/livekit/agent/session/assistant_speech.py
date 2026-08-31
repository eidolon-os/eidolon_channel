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
        self._fixed_speech_pending = False
        self._fixed_speech_active = False

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

    def queue_fixed_speech(self, text: str, *, source: str) -> None:
        """Record Channel-issued speech that will start on the next playback.

        ``session.say()`` can queue text several seconds before the matching
        audio reaches the room.  A wall-clock TTL measured from queue time can
        therefore expire while the user is still hearing the sentence.  The
        ledger models that lifecycle explicitly without depending on provider
        task IDs or LiveKit private speech handles.
        """

        self.record(text, source=source)
        if self._latest is not None:
            self._fixed_speech_pending = True
            self._fixed_speech_active = False

    def on_playback_started(self) -> None:
        """Activate the fixed speech queued for this playback, if any."""

        if not self._fixed_speech_pending:
            return
        self._fixed_speech_pending = False
        self._fixed_speech_active = True

    def on_playback_finished(self) -> None:
        """Close active fixed speech and start its residual-echo tail window."""

        if not self._fixed_speech_active:
            return
        self._fixed_speech_active = False
        latest = self._latest
        if latest is not None:
            self._latest = AssistantSpeechSnapshot(
                text=latest.text,
                source=latest.source,
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
        if self._fixed_speech_active and latest is not None:
            return latest.text
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
