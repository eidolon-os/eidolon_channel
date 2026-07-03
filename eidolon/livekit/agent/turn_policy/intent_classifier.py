"""Interrupt-intent classification for real-time turn control.

The taxonomy, lexicons, and deterministic classifiers now live in
``eidolon_sdk.biz.dialogue_control`` — shared with eidolon_agent's reflex
layer so both sides of the Chat stream agree on what counts as a stop.
This module re-exports them under their historical names and keeps the
channel-only classifier slots (noop/ONNX).
"""

from __future__ import annotations

from eidolon_sdk.biz.dialogue_control import (
    InterruptIntent,
    InterruptIntentResult,
    LexiconInterruptClassifier,
    canonicalize_interrupt_text,
    hard_stop_intent,
    hard_stop_prefix_intent,
    normalize_interrupt_text,
)

__all__ = [
    "InterruptIntent",
    "InterruptIntentClassifier",
    "InterruptIntentResult",
    "LexiconInterruptClassifier",
    "NoopModelInterruptClassifier",
    "OnnxInterruptClassifier",
    "canonicalize_interrupt_text",
    "hard_stop_intent",
    "hard_stop_prefix_intent",
    "normalize_interrupt_text",
]


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
