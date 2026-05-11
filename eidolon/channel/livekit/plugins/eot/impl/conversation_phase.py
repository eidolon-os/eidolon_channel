# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Conversation phase tracker (Round 7 G5).

A coarse-grained tagger that classifies the current conversation moment
into one of four phases. Each phase carries different expectations about
turn rhythm:

- ``GREETING``: opening 1–2 turns ("你好", "请问"). Users expect quick
  responses; pauses should be short.
- ``GATHERING``: normal Q&A in the middle of a conversation. Default rhythm.
- ``THINKING``: user just emitted a hesitation marker ("我想想", "嗯…").
  Allow longer pauses; don't cut prematurely.
- ``CLOSING``: closing words or summaries ("再见", "就这样"). Quick exit
  is fine; longer pauses suggest the conversation has ended.

The detector is **stateless** — each call returns a phase based on the
provided text and history. The pipeline records phase transitions in the
EOT state via ``EidolonEOTModel.update_phase``.

Phase changes the dynamic silence threshold (in TurnEndPolicy):

| phase     | multiplier |
|-----------|-----------:|
| GREETING  |       0.85 |
| GATHERING |       1.00 |
| THINKING  |       1.50 |
| CLOSING   |       0.85 |

Multipliers are applied AFTER the priority-based threshold is chosen
(command / tail / punctuation / question / EOT-tier), so the relative
ordering of priorities is preserved.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

from .constants import (
    GREETING_WORDS,
    STRONG_ENDING_WORDS,
    SUMMARY_CLOSING_WORDS,
    THINKING_WORDS,
)

if TYPE_CHECKING:
    pass


class ConversationPhase(Enum):
    """Coarse-grained conversation phase."""

    GREETING = auto()
    """Opening 1–2 turns; user expects quick response."""

    GATHERING = auto()
    """Normal middle-of-conversation Q&A. Default phase."""

    THINKING = auto()
    """User just emitted a hesitation marker; allow longer pause."""

    CLOSING = auto()
    """User used a closing/farewell phrase; conversation winding down."""


# Multipliers applied to the dynamic silence threshold.
# Values are conservative: don't make tighter than 0.85 or looser than 1.5
# without empirical justification.
PHASE_THRESHOLD_MULTIPLIERS: dict[ConversationPhase, float] = {
    ConversationPhase.GREETING: 0.85,
    ConversationPhase.GATHERING: 1.00,
    ConversationPhase.THINKING: 1.50,
    ConversationPhase.CLOSING: 0.85,
}


class ConversationPhaseDetector:
    """Stateless detector that classifies a moment into a phase.

    Detection is order-sensitive (first match wins) to avoid ambiguity:
      1. CLOSING — closing/farewell text (highest priority; ends the chat)
      2. THINKING — hesitation marker present (overrides early-turn detection)
      3. GREETING — first 2 turns AND text contains a greeting word
      4. GATHERING — fallback default
    """

    # Configuration: how many turns count as "early conversation"
    # (and thus eligible for GREETING tagging when the text is also a greeting).
    GREETING_MAX_TURNS: int = 1

    def detect(
        self,
        text: str,
        history_length: int,
    ) -> ConversationPhase:
        """Classify the moment.

        Args:
            text: Latest ASR text (interim or final).
            history_length: Number of completed user turns so far in this
                session (0 = this is the user's first turn).

        Returns:
            The detected ConversationPhase.
        """
        if not text:
            return ConversationPhase.GATHERING
        stripped = text.strip().rstrip("。.!?！？,，")
        if not stripped:
            return ConversationPhase.GATHERING

        # 1. CLOSING — strongest priority (overrides everything else).
        # Match against full text since closing words may appear at end:
        # "好的，就这样吧" → CLOSING
        if any(w in stripped for w in SUMMARY_CLOSING_WORDS):
            return ConversationPhase.CLOSING
        # Tighter check for shorter terminal markers:
        if stripped in STRONG_ENDING_WORDS - SUMMARY_CLOSING_WORDS:
            # Single-word strong endings ("再见", "拜拜") only count as
            # CLOSING when they're the WHOLE utterance.
            return ConversationPhase.CLOSING

        # 2. THINKING — hesitation marker anywhere in text.
        if any(w in stripped for w in THINKING_WORDS):
            return ConversationPhase.THINKING

        # 3. GREETING — first turn(s), text is a greeting.
        if history_length <= self.GREETING_MAX_TURNS:
            if any(g in stripped for g in GREETING_WORDS):
                return ConversationPhase.GREETING

        return ConversationPhase.GATHERING

    def get_threshold_multiplier(self, phase: ConversationPhase) -> float:
        """Return the multiplier to apply to dynamic silence thresholds."""
        return PHASE_THRESHOLD_MULTIPLIERS.get(phase, 1.0)
