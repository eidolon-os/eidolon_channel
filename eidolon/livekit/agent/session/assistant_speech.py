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

    The provider-neutral source is LiveKit's public ``Agent.tts_node`` text
    stream. Provider ``current_pushed_text`` is an optional compatibility
    source when present. The ledger also covers Channel-issued fixed speech
    such as the welcome line.
    """

    def __init__(self, *, clock: Any | None = None) -> None:
        self._clock = clock or time.monotonic
        self._latest: AssistantSpeechSnapshot | None = None
        self._fixed_speech_pending = False
        self._fixed_speech_active = False
        self._playback_active = False
        self._streamed_speech_pending = False
        self._streamed_speech_active = False
        self._stream_id = 0
        self._stream_text = ""

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

    def begin_streamed_speech(self) -> int:
        """Open one public ``tts_node`` text stream and return its identity."""

        self._stream_id += 1
        self._stream_text = ""
        self._streamed_speech_pending = False
        self._streamed_speech_active = False
        return self._stream_id

    def append_streamed_speech(self, stream_id: int, text: str) -> bool:
        """Append a text delta from the active provider-neutral TTS stream."""

        if stream_id != self._stream_id or not isinstance(text, str) or not text:
            return False
        self._stream_text += text
        self.record(self._stream_text, source="livekit_tts_node")
        if self._playback_active:
            self._streamed_speech_active = True
        else:
            self._streamed_speech_pending = True
        return True

    def abort_streamed_speech(self, stream_id: int) -> None:
        """Discard an unplayed text stream when synthesis fails."""

        if stream_id != self._stream_id:
            return
        if not self._streamed_speech_active:
            self._streamed_speech_pending = False
            self._stream_text = ""

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

        self._playback_active = True
        if self._streamed_speech_pending:
            self._streamed_speech_pending = False
            self._streamed_speech_active = True
        if not self._fixed_speech_pending:
            return
        self._fixed_speech_pending = False
        self._fixed_speech_active = True

    def on_playback_finished(self) -> None:
        """Close active speech and start its residual-echo tail window."""

        latest = self._latest
        belongs_to_playback = self._fixed_speech_active or self._streamed_speech_active
        self._playback_active = False
        self._fixed_speech_active = False
        self._streamed_speech_active = False
        if latest is not None and belongs_to_playback:
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
        latest = self._latest
        if latest is not None and (
            self._fixed_speech_active or self._streamed_speech_active
        ):
            return latest.text
        current = self.current_tts_text(factory)
        if current.strip():
            self.record(current, source="tts_in_flight")
            return current
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
