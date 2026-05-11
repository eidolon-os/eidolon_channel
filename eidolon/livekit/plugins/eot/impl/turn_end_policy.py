# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TurnEndPolicy: Computes dynamic silence threshold (seconds) for turn-end decisions.
Copied from eidolon/pipeline/src/manager/turn_detection/turn_end_policy.py
"""

from __future__ import annotations

import re
from typing import Optional

from .constants import (
    COMMAND_WORDS,
    CONTINUATION_INTENT_PATTERNS,
    FILLER_WORDS,
    INCOMPLETE_CONJUNCTIONS,
    INTERRUPT_INTENT_WORDS,
    PUNCTUATION_RADIUS,
    QUESTION_BOUND_FORMS_END,
    QUESTION_END_PARTICLES,
    STRONG_INTERRUPT_INTENT_WORDS,
    TERMINAL_PUNCTUATION,
    THINKING_WORDS,
    WAIT_PUNCTUATION,
    WEAK_INTERRUPT_INTENT_WORDS,
)
from .conversation_phase import (
    PHASE_THRESHOLD_MULTIPLIERS,
    ConversationPhase,
)


class TurnEndPolicy:
    def __init__(
        self,
        t_min: float = 0.4,
        t_max: float = 2.2,
        t_urgent: float = 0.18,
        t_normal: float = 0.8,
        t_fast: float = 0.25,
        t_mid: float = 1.0,
        t_deep: float = 2.0,
        t_tail_hang: float = 2.5,
        enable_semantic_tail_hang: bool = True,
        is_final_reduction: float = 0.2,
    ):
        self.T_MIN = t_min
        self.T_MAX = t_max
        self.T_URGENT = t_urgent
        self.T_NORMAL = t_normal
        self.enable_semantic_tail_hang = enable_semantic_tail_hang
        self.T_FAST = t_fast
        self.T_MID = t_mid
        self.T_DEEP = t_deep
        self.T_TAIL_HANG = t_tail_hang
        self._is_final_reduction = is_final_reduction

    def is_interrupt_intent(self, text: str) -> bool:
        """Whether the text contains any interrupt intent word (strong or weak)."""
        t = (text or "").strip()
        if not t:
            return False
        return any(w in t for w in INTERRUPT_INTENT_WORDS)

    def is_strong_interrupt_intent(self, text: str) -> bool:
        """Whether the text contains a strong interrupt word (stop, shut up, wrong, etc.)."""
        t = (text or "").strip()
        if not t:
            return False
        return any(w in t for w in STRONG_INTERRUPT_INTENT_WORDS)

    def is_weak_interrupt_intent(self, text: str) -> bool:
        """Whether the text contains a weak interrupt word (OK, um, that, then, etc.)."""
        t = (text or "").strip()
        if not t:
            return False
        return any(w in t for w in WEAK_INTERRUPT_INTENT_WORDS)

    def is_continuation_intent(self, text: str) -> bool:
        """Whether the text is a continuation/replacement request that should NOT interrupt.

        Phrases like "再换一个", "继续讲一个", "不好笑换一个" signal the user wants
        the agent to keep working (switch topic or continue) rather than stop. Even if they
        contain strong-interrupt substrings (e.g. "不对" in "不对，你要换一个新的"),
        their overall intent is to continue, not halt.
        """
        t = (text or "").strip()
        if not t:
            return False
        return any(pattern in t for pattern in CONTINUATION_INTENT_PATTERNS)

    def is_valid_speech(self, text: str) -> bool:
        """Whether the text is worth sending to the LLM (filters pure filler)."""
        clean_text = text.strip().replace("。", "").replace("，", "")
        if not clean_text:
            return False

        if clean_text in FILLER_WORDS:
            return False

        if len(clean_text) <= 4:
            if all(ch in FILLER_WORDS for ch in clean_text):
                return False

        return True

    def _ends_with_tail(self, text: str) -> bool:
        """Whether the text ends with a conjunction/thinking word that needs a longer silence to break."""
        if not self.enable_semantic_tail_hang:
            return False
        text = (text or "").strip()
        text = re.sub(r"[，。！？!?、;；…\.\s]+$", "", text)
        return (
            any(text.endswith(w) for w in THINKING_WORDS)
            or any(text.endswith(w) for w in INCOMPLETE_CONJUNCTIONS)
        )

    def is_question_ending(self, text: str) -> bool:
        """Round 7 G3: Detect Chinese question ending without relying on "?".

        Streaming ASR often delivers raw text without trailing punctuation,
        so a sentence like "你好吗" looks declarative even though it's
        clearly a question. This helper catches the implicit question by
        looking at sentence-ending grammar markers.

        Returns True when the text (after stripping trailing punctuation
        — both ASCII and full-width) ends with:
          - a sentence-final question particle (吗 / 呢 / 么 / 嘛), OR
          - a bound interrogative form (是不是 / 对不对 / etc.)

        These are equivalent in turn-end semantics to ending with "?",
        so callers can use the same urgent threshold.
        """
        if not text:
            return False
        # Strip trailing whitespace + punctuation (both ASCII and full-width).
        clean = re.sub(r"[，。！？!?、;；…\.,;\s]+$", "", text.strip())
        if not clean:
            return False
        if clean.endswith(QUESTION_END_PARTICLES):
            return True
        if clean.endswith(QUESTION_BOUND_FORMS_END):
            return True
        return False

    def get_dynamic_threshold(
        self,
        text: str,
        p_complete: float,
        is_final: bool,
        phase: "Optional[ConversationPhase]" = None,
    ) -> float:
        """
        Stepped dynamic silence threshold (seconds).
        Priority order:
          1. exact-match COMMAND_WORDS (短命令)              → T_URGENT
          2. tail-hang (thinking / incomplete conjunction at end) → T_TAIL_HANG
          3. terminal punctuation in last few chars           → T_URGENT/T_MIN
          4. **question ending (G3, no punctuation needed)**  → T_URGENT/T_MIN
          5. WAIT_PUNCTUATION at end                          → T_MAX
          6. EOT tier (fast/mid/deep) based on p_complete

        Round 7 G5: a phase-aware multiplier is applied AFTER the priority
        match to nudge the threshold based on conversational context
        (e.g. THINKING phase loosens, GREETING / CLOSING tightens). The
        relative ordering of priorities is preserved; only the absolute
        threshold is scaled.

        Args:
            text: Latest ASR text.
            p_complete: Current EOT completeness probability ∈ [0, 1].
            is_final: Whether ASR has marked this as the final transcript.
            phase: Optional ConversationPhase. When None or GATHERING,
                no multiplier is applied (factor=1.0).

        Returns:
            Silence threshold in seconds.
        """
        text = text.strip()

        # Priority-matched threshold + whether to clamp at T_MAX.
        # tail_hang explicitly EXCEEDS T_MAX (it's semantically a "give the
        # user extra thinking time" override, not bound by EOT ceilings),
        # so we tag the path and skip clamping for it. All other paths
        # follow the original behavior: EOT-tier clamps at T_MAX.
        clamp_at_max = True

        if any(cmd == text for cmd in COMMAND_WORDS):
            base = self.T_URGENT
        elif self._ends_with_tail(text):
            base = self.T_TAIL_HANG
            clamp_at_max = False  # tail_hang intentionally allowed to exceed T_MAX
        else:
            text_tail = (
                text[-PUNCTUATION_RADIUS:]
                if len(text) >= PUNCTUATION_RADIUS
                else text
            )
            if any(p in text_tail for p in TERMINAL_PUNCTUATION):
                base = self.T_URGENT if is_final else self.T_MIN
            elif self.is_question_ending(text):
                # Round 7 G3: implicit question (no terminal punctuation).
                base = self.T_URGENT if is_final else self.T_MIN
            elif any(text.endswith(w) for w in WAIT_PUNCTUATION):
                base = self.T_MAX
            else:
                if p_complete >= 0.8:
                    base = self.T_FAST
                elif p_complete >= 0.4:
                    base = self.T_MID
                else:
                    base = self.T_DEEP
                if is_final:
                    base = max(self.T_URGENT, base - self._is_final_reduction)

        # Round 7 G5: phase multiplier (None and GATHERING → factor 1.0).
        # Applied after priority match so all paths can be scaled by phase.
        if phase is not None and phase != ConversationPhase.GATHERING:
            multiplier = PHASE_THRESHOLD_MULTIPLIERS.get(phase, 1.0)
            base = base * multiplier
            # Floor at T_URGENT — phase multipliers (e.g. GREETING=0.85)
            # should never push the threshold below the urgency minimum.
            # tail_hang is exempt because it explicitly exceeds T_MAX going
            # the other way; if a multiplier shrinks tail_hang below T_URGENT
            # something is misconfigured, but we still floor for safety.
            base = max(base, self.T_URGENT)

        # Clamp T_MAX only when the priority path is bound by it (not tail_hang).
        if clamp_at_max:
            base = min(base, self.T_MAX)
        return base
