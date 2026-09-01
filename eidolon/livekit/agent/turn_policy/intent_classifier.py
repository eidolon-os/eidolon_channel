"""Replaceable model-intent interface for real-time turn observability.

Production turn control does not import or execute fixed phrase classifiers.
An optional learned classifier may annotate a decision, but EOT/VAD/finality
evidence remains the authority for irreversible output effects.
"""

from __future__ import annotations

from eidolon_sdk.biz.dialogue_control import (
    InterruptIntent,
    InterruptIntentResult,
)

__all__ = [
    "InterruptIntent",
    "InterruptIntentClassifier",
    "InterruptIntentResult",
    "NoopModelInterruptClassifier",
    "OnnxInterruptClassifier",
    "normalize_transcript_text",
]


def normalize_transcript_text(text: str) -> str:
    """Normalize transport whitespace without interpreting transcript words."""

    return " ".join((text or "").strip().lower().split())


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
