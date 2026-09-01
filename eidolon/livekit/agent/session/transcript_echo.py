"""Content-based transcript echo suppression.

Open-mic full-duplex sessions can leak the agent's own TTS back into STT even
after hardware AEC.  Energy thresholds are not a reliable separator on the
current devices, so this helper compares the transcript content against the
agent's in-flight TTS text.  The caller still owns when this gate applies.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from enum import Enum
from typing import Any

from .assistant_speech import AssistantSpeechLedger

logger = logging.getLogger("agent.session.transcript_echo")

_LEADING_BOUNDARY_DUPLICATE = re.compile(
    r"^\s*([\w])(?:\s|[,，、:：;；])+\1",
    flags=re.UNICODE,
)


class EchoEvidence(str, Enum):
    """Provider-neutral relationship between STT text and assistant speech."""

    NONE = "none"
    POSSIBLE = "possible"
    CONFIRMED = "confirmed"


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
        return self.classify(transcript) is EchoEvidence.CONFIRMED

    def classify(self, transcript: str) -> EchoEvidence:
        """Classify exact echo evidence without treating a short prefix as final.

        A two-character streaming fragment can be part of the assistant's
        current speech but is too ambiguous to reject as echo. It is still
        useful HOLD evidence: later STT revisions can confirm echo or diverge
        into a real user interruption.
        """

        normalized_agent_text = normalize_for_echo(self.agent_text())
        if not normalized_agent_text:
            return EchoEvidence.NONE
        candidates = echo_candidates(transcript)
        if any(
            len(candidate) >= self._min_normalized_chars
            and candidate in normalized_agent_text
            for candidate in candidates
        ):
            return EchoEvidence.CONFIRMED
        if any(
            2 <= len(candidate) < self._min_normalized_chars
            and candidate in normalized_agent_text
            for candidate in candidates
        ):
            return EchoEvidence.POSSIBLE
        return EchoEvidence.NONE

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


def echo_candidates(text: str) -> tuple[str, ...]:
    """Return exact-content views for common streaming-boundary duplication.

    Streaming ASR can repeat the first character across a punctuation boundary
    (for example ``我，我是你的AI助手``). Keep the ordinary normalized text and,
    only for that explicit boundary shape, add the de-duplicated view. This is
    still exact containment against assistant-owned speech, not fuzzy matching.
    """

    normalized = normalize_for_echo(text)
    deduplicated = normalize_for_echo(_LEADING_BOUNDARY_DUPLICATE.sub(r"\1", text, count=1))
    if deduplicated and deduplicated != normalized:
        return normalized, deduplicated
    return (normalized,)
