"""Interrupt-intent classification for real-time turn control."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from eidolon.livekit.common.config.defaults import (
    DEFAULT_CORRECTION_LEXICON,
    DEFAULT_HARD_STOP_PREFIX_LEXICON,
    DEFAULT_HARD_STOP_LEXICON,
    DEFAULT_TOPIC_SWITCH_LEXICON,
)
from eidolon.livekit.plugins.eot.impl.constants import (
    BACKCHANNEL_WORDS,
    NOISE_LIKE_TRANSCRIPTIONS,
)

from .constants import (
    ASR_EXACT_CANONICALIZATIONS,
    ASR_PREFIX_CANONICALIZATIONS,
    INTERRUPT_TEXT_TRAILING_CHARS,
    REPEATED_NOISE_CHARS,
)


class InterruptIntent(str, Enum):
    HARD_STOP = "hard_stop"
    TOPIC_SWITCH = "topic_switch"
    CORRECTION = "correction"
    BACKCHANNEL = "backchannel"
    NOISE = "noise"
    NORMAL_INTERRUPT = "normal_interrupt"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class InterruptIntentResult:
    intent: InterruptIntent
    confidence: float
    source: str
    reason: str = ""


class InterruptIntentClassifier:
    """Interface for hot-path interrupt intent classification."""

    def classify(
        self,
        text: str,
        *,
        vad_active: bool,
        agent_speaking: bool,
        eot_score: float,
    ) -> InterruptIntentResult:
        raise NotImplementedError


def normalize_interrupt_text(text: str) -> str:
    return text.strip().lower().rstrip(INTERRUPT_TEXT_TRAILING_CHARS)


def canonicalize_interrupt_text(text: str) -> str:
    """Normalize common hot-path STT confusions without changing raw logs."""

    stripped = normalize_interrupt_text(text)
    if stripped in ASR_EXACT_CANONICALIZATIONS:
        return ASR_EXACT_CANONICALIZATIONS[stripped]
    for source_prefix, target_prefix in ASR_PREFIX_CANONICALIZATIONS:
        if stripped.startswith(source_prefix):
            return target_prefix + stripped[len(source_prefix) :]
    return stripped


def hard_stop_prefix_intent(
    text: str,
    *,
    min_chars: int = 2,
    min_cjk_chars: int = 2,
) -> InterruptIntent | None:
    """Classify high-precision Tier0 hard-stop prefixes."""

    stripped = canonicalize_interrupt_text(text)
    if len(stripped) < min_chars or _count_cjk(stripped) < min_cjk_chars:
        return None
    prefixes = tuple(
        normalize_interrupt_text(item)
        for item in DEFAULT_HARD_STOP_PREFIX_LEXICON
    )
    return InterruptIntent.HARD_STOP if stripped in prefixes else None


def hard_stop_intent(text: str) -> InterruptIntent | None:
    """Classify only Tier0 hard-stop text.

    Non-Tier0 interruption semantics intentionally do not live in lexical
    hot-path rules. They should come from transcript evidence, EOT/turn
    detection, or a future validated lightweight model.
    """

    stripped = canonicalize_interrupt_text(text)
    if not stripped:
        return None
    hard_stops = tuple(
        normalize_interrupt_text(item)
        for item in DEFAULT_HARD_STOP_LEXICON
    )
    if any(candidate and candidate in stripped for candidate in hard_stops):
        return InterruptIntent.HARD_STOP
    return hard_stop_prefix_intent(stripped)


def _count_cjk(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


class LexiconInterruptClassifier(InterruptIntentClassifier):
    """Low-latency classifier for Tier0 plus non-lexical noise shape.

    Topic switch, correction, and backchannel word lists are deliberately not
    used here. Outside Tier0, hot-path intent should be decided by EOT/semantic
    evidence or a separately validated model.
    """

    def __init__(self, *, fast_intents: bool = False) -> None:
        self._fast_intents = fast_intents
        self._hard_stop = tuple(
            normalize_interrupt_text(x) for x in DEFAULT_HARD_STOP_LEXICON
        )
        self._topic_switch = tuple(
            normalize_interrupt_text(x) for x in DEFAULT_TOPIC_SWITCH_LEXICON
        )
        self._correction = tuple(
            normalize_interrupt_text(x) for x in DEFAULT_CORRECTION_LEXICON
        )

    def classify(
        self,
        text: str,
        *,
        vad_active: bool,
        agent_speaking: bool,
        eot_score: float,
    ) -> InterruptIntentResult:
        stripped = canonicalize_interrupt_text(text)
        if not stripped:
            return InterruptIntentResult(
                InterruptIntent.NOISE, 1.0, "lexicon", "empty_transcript"
            )

        if self._contains_any(stripped, self._hard_stop):
            return InterruptIntentResult(
                InterruptIntent.HARD_STOP, 1.0, "lexicon", "hard_stop"
            )
        if self._fast_intents and self._contains_any(stripped, self._topic_switch):
            return InterruptIntentResult(
                InterruptIntent.TOPIC_SWITCH, 0.95, "lexicon", "topic_switch"
            )
        if self._fast_intents and self._contains_any(stripped, self._correction):
            return InterruptIntentResult(
                InterruptIntent.CORRECTION, 0.85, "lexicon", "correction"
            )
        if self._fast_intents and stripped in BACKCHANNEL_WORDS:
            return InterruptIntentResult(
                InterruptIntent.BACKCHANNEL, 0.95, "lexicon", "backchannel"
            )
        if self._fast_intents and stripped in NOISE_LIKE_TRANSCRIPTIONS:
            return InterruptIntentResult(
                InterruptIntent.NOISE, 0.90, "lexicon", "noise_like"
            )
        if (
            2 <= len(stripped) <= 6
            and len(set(stripped)) == 1
            and stripped[0] in REPEATED_NOISE_CHARS
        ):
            return InterruptIntentResult(
                InterruptIntent.NOISE, 0.85, "lexicon", "repeated_noise_char"
            )
        return InterruptIntentResult(
            InterruptIntent.UNCERTAIN, 0.0, "lexicon", "no_lexical_match"
        )

    @staticmethod
    def _contains_any(text: str, candidates: tuple[str, ...]) -> bool:
        return any(c and c in text for c in candidates)

    @staticmethod
    def _starts_with_any(text: str, candidates: tuple[str, ...]) -> bool:
        return any(c and text.startswith(c) for c in candidates)


class NoopModelInterruptClassifier(InterruptIntentClassifier):
    """Placeholder model classifier that intentionally makes no decision."""

    def classify(
        self,
        text: str,
        *,
        vad_active: bool,
        agent_speaking: bool,
        eot_score: float,
    ) -> InterruptIntentResult:
        return InterruptIntentResult(
            InterruptIntent.UNCERTAIN, 0.0, "model_noop", "model_not_configured"
        )


class OnnxInterruptClassifier(InterruptIntentClassifier):
    """Future ONNX INT8 classifier slot.

    TODO: Train/distill a Chinese hard-interrupt classifier with labels:
    hard_stop, topic_switch, correction, backchannel, noise, normal_interrupt,
    uncertain. Target CPU P95 <= 30 ms before enabling in production.
    """

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        raise NotImplementedError(
            "OnnxInterruptClassifier is a reserved integration point; "
            "train and validate the model in a separate task first."
        )
