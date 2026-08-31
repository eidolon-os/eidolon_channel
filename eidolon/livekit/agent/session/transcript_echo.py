"""Content-based transcript echo suppression.

Open-mic full-duplex sessions can leak the agent's own TTS back into STT even
after hardware AEC.  Energy thresholds are not a reliable separator on the
current devices, so this helper compares the transcript content against the
agent's in-flight TTS text.  The caller still owns when this gate applies.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .assistant_speech import AssistantSpeechLedger

logger = logging.getLogger("agent.session.transcript_echo")


class TranscriptEchoGate:
    """Detect transcripts that are content echoes of current agent speech."""

    def __init__(
        self,
        *,
        factory: Any | None = None,
        get_agent_text: Callable[[], str] | None = None,
        min_normalized_chars: int = 3,
    ) -> None:
        self._factory = factory
        self._get_agent_text = get_agent_text
        self._min_normalized_chars = max(1, int(min_normalized_chars))

    def is_echo(self, transcript: str) -> bool:
        normalized_transcript = normalize_for_echo(transcript)
        if len(normalized_transcript) < self._min_normalized_chars:
            return False
        normalized_agent_text = normalize_for_echo(self.agent_text())
        if not normalized_agent_text:
            return False
        return normalized_transcript in normalized_agent_text

    def agent_text(self) -> str:
        if self._get_agent_text is not None:
            return self._get_agent_text() or ""
        return AssistantSpeechLedger.current_tts_text(self._factory)


def normalize_for_echo(text: str) -> str:
    """Normalize ASR/TTS text for echo matching.

    Keep CJK and alphanumerics, drop punctuation and whitespace, and lowercase
    latin text so ASR punctuation noise does not affect the containment check.
    """

    return "".join(char for char in text if char.isalnum()).lower()
