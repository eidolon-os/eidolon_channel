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
Lightweight context-enhanced EOT detector.

Adds contextual adjustments to the base EOT score: dialogue history, user profile,
follow-up words, strong endings, greetings, short complete phrases, noun patterns,
hesitation patterns, punctuation, etc.
Copied from eidolon/pipeline/src/manager/turn_detection/context_enhanced_eot.py
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..log import logger
from .eot_manager import EotManager
from .constants import (
    CONTINUATION_INTENT_PATTERNS,
    FILLER_WORDS,
    FOLLOWUP_INDICATORS,
    GREETING_WORDS,
    NOUN_PATTERNS,
    PUNCTUATION_RADIUS,
    SHORT_COMPLETE_PHRASES,
    STRONG_ENDING_WORDS,
    TERMINAL_PUNCTUATION,
)


@dataclass
class TurnContext:
    """Single turn dialogue context."""

    text: str
    timestamp: float
    is_complete: bool
    utterance_end_score: float
    turn_type: str = "user"


@dataclass
class UserProfile:
    """User speaking habit profile."""

    avg_response_length: float = 8.0
    filler_word_ratio: float = 0.0
    avg_pause_between_turns: float = 0.5
    speaking_pace: str = "normal"

    _recent_lengths: deque = field(default_factory=lambda: deque(maxlen=10))
    _recent_pauses: deque = field(default_factory=lambda: deque(maxlen=10))
    _filler_count: int = field(default=0)
    _total_words: int = field(default=0)

    def update(self, text: str, pause_duration: float) -> None:
        """Update user profile."""
        self._recent_lengths.append(len(text))
        self._recent_pauses.append(pause_duration)

        if self._recent_lengths:
            self.avg_response_length = sum(self._recent_lengths) / len(
                self._recent_lengths
            )

        if self._recent_pauses:
            self.avg_pause_between_turns = sum(self._recent_pauses) / len(
                self._recent_pauses
            )

        pace = self.avg_response_length / max(self.avg_pause_between_turns, 0.1)
        if pace > 15:
            self.speaking_pace = "fast"
        elif pace < 8:
            self.speaking_pace = "slow"
        else:
            self.speaking_pace = "normal"


class ContextEnhancedEot:
    """
    Lightweight context-enhanced EOT detector.

    Adds on top of base EOT:
    1. Dialogue history (recent 3 turns)
    2. User behavior profile
    3. Lightweight rule adjustments
    """

    tag = "ContextEnhancedEot"

    def __init__(
        self,
        base_eot: Optional[EotManager] = None,
        max_history: int = 3,
        enable_user_profile: bool = True,
        enable_temporary_compensations: bool = True,
        cooldown_period: float = 0.8,
        similarity_threshold: float = 0.85,
        max_profiles: int = 1000,
    ):
        """
        Args:
            base_eot: Base EOT manager (EotManager), auto-creates if None.
            max_history: Number of dialogue history turns to retain.
            enable_user_profile: Enable user profile adjustments.
            enable_temporary_compensations: Enable temporary compensation rules.
            cooldown_period: Minimum seconds between cuts (from EidolonEOTConfig).
            similarity_threshold: Jaccard similarity threshold for duplicate detection.
            max_profiles: LRU cap on cached UserProfiles (Round 8 P2.L8).
                When the cache exceeds this size on ``start_session``, the
                least-recently-touched profile is evicted. Profiles are
                also explicitly removed on ``end_session`` (preferred).
                The cap protects long-running daemon processes from
                unbounded memory growth across many rooms.
        """
        from collections import OrderedDict

        self._base_eot = base_eot or EotManager()
        self._max_history = max_history
        self._enable_profile = enable_user_profile
        self._enable_temp_comp = enable_temporary_compensations
        self._temp_comp_checked = False
        self._cooldown_period = cooldown_period
        self._similarity_threshold = similarity_threshold
        self._max_profiles = max_profiles

        # Dialogue history. The maxlen is the configured max_history (default 3
        # via EidolonEOTConfig.utterance_end_max_history). Earlier code re-bound
        # this deque with a hard-coded 10 below — that bug silently ignored the
        # config value. Removed in Round 7 (G0b).
        self._dialogue_history: deque = deque(maxlen=max_history)

        # User profiles (per session). Round 8 P2.L8: changed from plain
        # dict to OrderedDict so we can cap the size with LRU eviction.
        # Was unbounded; long-running daemons accumulated profiles
        # forever across rooms. Now: explicit ``end_session`` removes
        # the profile when a session closes, AND start_session enforces
        # the cap as a safety net for paths that miss end_session.
        self._user_profiles: "OrderedDict[str, UserProfile]" = OrderedDict()

        # Current session state
        self._current_session_id: Optional[str] = None
        self._last_turn_time: float = 0.0
        self._current_turn_start: float = 0.0

        # State reset fields. ``_last_interrupt_time`` and ``_last_text`` are
        # written by :meth:`record_interrupt` (NOT directly by external callers
        # — see G0a in Round 7 which replaced the private-field writes in
        # ``EidolonEOTModel.should_interrupt`` with this public method).
        self._last_interrupt_time: Optional[float] = None
        self._last_text: str = ""
        self._current_eot_score: float = 0.0

    @property
    def _temp_comp_enabled(self) -> bool:
        """Lazily check whether temp compensations should be active.

        Deferred to first use to avoid 3x ONNX inference during __init__.
        """
        if not self._temp_comp_checked:
            self._temp_comp_checked = True
            if self._enable_temp_comp and self._check_eot_strength():
                self._enable_temp_comp = False
                logger.info(f"[{self.tag}] Strong EOT model detected, disabling temp compensations")
        return self._enable_temp_comp

    def _check_eot_strength(self) -> bool:
        """Detect if base EOT model is strong enough (can remove temp compensations)."""
        try:
            test_cases = [
                ("北京天气", 0.7),
                ("谢谢", 0.7),
                ("你好", 0.7),
            ]
            passed = 0
            for text, threshold in test_cases:
                score = self._base_eot.p_complete_score(text)
                if score >= threshold:
                    passed += 1
            return passed >= 3
        except Exception:
            return False

    def start_session(self, session_id: str) -> None:
        """Start a new session.

        Round 8 P2.L8: enforces ``max_profiles`` LRU cap. If the cache is
        full and we're adding a new session, the least-recently-used
        entry is evicted. Existing entries are moved to "most recent"
        so they don't get evicted on the next start.
        """
        self._current_session_id = session_id
        self._dialogue_history.clear()
        self._last_turn_time = time.time()
        self._current_turn_start = time.time()

        if session_id in self._user_profiles:
            # Mark as most recently used.
            self._user_profiles.move_to_end(session_id)
        else:
            # Enforce LRU cap before adding.
            while len(self._user_profiles) >= self._max_profiles:
                evicted_id, _ = self._user_profiles.popitem(last=False)
                logger.info(
                    "[%s] LRU evicted UserProfile for session=%s "
                    "(cache full at %d)",
                    self.tag, evicted_id, self._max_profiles,
                )
            self._user_profiles[session_id] = UserProfile()

        logger.info(f"[{self.tag}] Session started: {session_id}")

    def end_session(self, session_id: Optional[str] = None) -> None:
        """Round 8 P2.L8: explicit cleanup hook for ending sessions.

        Removes the session's UserProfile from the cache and clears
        ``_current_session_id`` if it matches. Should be called when a
        room/session closes (StreamingPipeline session_close event).

        Calling with ``session_id=None`` clears the currently-active
        session. Calling with a specific session_id is idempotent
        (no-op if profile already evicted by LRU).
        """
        target = session_id or self._current_session_id
        if target is None:
            return
        removed = self._user_profiles.pop(target, None)
        if removed is not None:
            logger.info(
                "[%s] end_session: removed UserProfile for %s "
                "(cache size now %d)",
                self.tag, target, len(self._user_profiles),
            )
        if self._current_session_id == target:
            self._current_session_id = None
            self._dialogue_history.clear()

    def update_asr(self, text: str, is_final: bool = False) -> None:
        """Update ASR text and record user input."""
        if not self._current_session_id:
            return

        current_time = time.time()
        turn_duration = current_time - self._current_turn_start
        pause_duration = current_time - self._last_turn_time if self._last_turn_time else 0

        if self._enable_profile and self._current_session_id:
            profile = self._user_profiles[self._current_session_id]
            profile.update(text, pause_duration)

        self._detect_hesitation_pattern(text)

    def _detect_hesitation_pattern(self, text: str) -> bool:
        """Detect if user is hesitating."""
        if len(self._dialogue_history) < 2:
            return False

        recent_texts = [turn.text for turn in list(self._dialogue_history)[-2:]]
        recent_texts.append(text)

        short_utterances = sum(1 for t in recent_texts if len(t) <= 4)
        filler_count = sum(1 for t in recent_texts if self._is_mostly_filler(t))

        if short_utterances >= 2 and filler_count >= 2:
            logger.debug(f"[{self.tag}] Hesitation pattern detected: {recent_texts}")
            return True

        return False

    def _is_mostly_filler(self, text: str) -> bool:
        """Check if text is mostly filler words."""
        words = self._extract_words(text)
        if not words:
            return False
        filler_count = sum(1 for w in words if w in FILLER_WORDS)
        return filler_count / len(words) > 0.6

    def _extract_words(self, text: str) -> List[str]:
        """Simple word extraction."""
        words = []
        i = 0
        while i < len(text):
            if i + 2 <= len(text):
                two_char = text[i : i + 2]
                if two_char in FILLER_WORDS or two_char in FOLLOWUP_INDICATORS:
                    words.append(two_char)
                    i += 2
                    continue
            words.append(text[i])
            i += 1
        return words

    def _is_follow_up(self, text: str) -> bool:
        """Check if text is a follow-up question."""
        text = text.strip()

        for indicator in FOLLOWUP_INDICATORS:
            if text.startswith(indicator) or text.endswith(indicator):
                return True

        pronouns = ["它", "他", "她", "这个", "那个", "这样", "那样"]
        if any(p in text for p in pronouns):
            if len(text) <= 8:
                return True

        return False

    def _has_strong_ending(self, text: str) -> bool:
        """Check for strong ending signal."""
        text = text.strip()

        for ending in STRONG_ENDING_WORDS:
            if text.endswith(ending):
                return True

        text_tail = (
            text[-PUNCTUATION_RADIUS:] if len(text) >= PUNCTUATION_RADIUS else text
        )
        if any(p in text_tail for p in TERMINAL_PUNCTUATION):
            return True

        return False

    def _get_context_adjustment(
        self, text: str, base_score: float, profile: Optional[UserProfile]
    ) -> float:
        """Compute context adjustment value."""
        adjustment = 0.0

        text_stripped = text.strip()
        if not text_stripped or len(text_stripped) == 0:
            return -0.5

        if self._temp_comp_enabled:
            # Greetings in first turn
            if len(self._dialogue_history) == 0 and text_stripped in GREETING_WORDS:
                adjustment += 0.3
                logger.debug(f"[{self.tag}] [TEMP] First-turn greeting, +0.3")

            # Short complete phrases
            if text_stripped in SHORT_COMPLETE_PHRASES:
                if len(self._dialogue_history) <= 1:
                    adjustment += 0.35
                    logger.debug(f"[{self.tag}] [TEMP] Short complete phrase, +0.35")

            # Noun pattern matching
            for pattern in NOUN_PATTERNS:
                if re.match(pattern, text_stripped):
                    if len(self._dialogue_history) <= 1:
                        adjustment += 0.40
                        logger.debug(f"[{self.tag}] [TEMP] Noun pattern match, +0.40")
                    break

        # Follow-up question
        if self._is_follow_up(text):
            adjustment += 0.15
            logger.debug(f"[{self.tag}] Follow-up question, +0.15")

        # Strong ending signal
        if self._has_strong_ending(text):
            adjustment += 0.18

        # Hesitation pattern
        if self._detect_hesitation_pattern(text):
            adjustment -= 0.15
            logger.debug(f"[{self.tag}] Hesitation pattern, -0.15")

        # Pure filler
        if self._is_mostly_filler(text_stripped):
            adjustment -= 0.25
            logger.debug(f"[{self.tag}] Pure filler, -0.25")

        # User profile
        if profile:
            if profile.speaking_pace == "fast":
                adjustment += 0.05
            if profile.filler_word_ratio > 0.3:
                adjustment -= 0.05

        # First turn leniency
        if len(self._dialogue_history) == 0:
            if len(text_stripped) <= 4:
                adjustment += 0.1

        # Length vs average
        if profile and profile.avg_response_length > 0:
            current_len = len(text_stripped)
            if current_len < profile.avg_response_length * 0.5:
                if len(self._dialogue_history) > 1:
                    pass
                else:
                    adjustment -= 0.05
            elif current_len > profile.avg_response_length * 1.5:
                adjustment += 0.05

        return max(-0.3, min(0.35, adjustment))

    def predict(
        self, text: str, is_final: bool = False, threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Predict EOT probability.

        Returns:
            {
                'is_complete': bool,
                'probability': float,
                'base_score': float,
                'adjustment': float,
                'confidence': str,
                'reason': str,
            }
        """
        base_score = self._base_eot.p_complete_score(text)

        profile = None
        if self._enable_profile and self._current_session_id:
            profile = self._user_profiles.get(self._current_session_id)

        adjustment = self._get_context_adjustment(text, base_score, profile)

        final_score = base_score + adjustment
        final_score = max(0.0, min(1.0, final_score))

        confidence = self._calculate_confidence(final_score, adjustment)
        reason = self._generate_reason(text, base_score, adjustment, profile)

        return {
            "is_complete": final_score >= threshold,
            "probability": final_score,
            "base_score": base_score,
            "adjustment": adjustment,
            "confidence": confidence,
            "reason": reason,
        }

    def p_complete_score(self, text: str) -> float:
        """Compatibility interface: return EOT probability score (0-1)."""
        result = self.predict(text, is_final=False)
        return result["probability"]

    def semantic_completeness_score(self, text: str) -> float:
        """Shared scorer used by BOTH ``predict_end_of_turn`` (framework's
        endpointing question) and ``compute_score`` (interrupt question).

        Encodes the gates that are **semantically-true regardless of which
        path asks** — i.e. "is this even a valid turn?":

        * Continuation intent (``"再换一首"``) → 0.0
        * Too-short / pure-filler ("嗯", "啊") → 0.0
        * Otherwise: raw ONNX score + context adjustment (greeting, follow-up,
          hesitation, profile, etc.)

        Path-B-specific guards (cooldown after recent interrupt, similarity
        to last-cut text) are **NOT** here — they belong in ``compute_score``
        because they protect against double-interrupting, not against
        committing a turn.

        Production bug 2026-05-07: framework's ``predict_end_of_turn`` was
        previously calling raw ``p_complete_score`` directly, so backchannels
        like "嗯" got raw=0.733 and crossed framework's unlikely_threshold —
        framework committed the turn and fired LLM, even though
        ``compute_score`` (correctly) returned 0.0 from the too-short gate.
        Two paths, two answers, one user yelling at the agent.
        """
        stripped = (text or "").strip()
        if not stripped:
            return 0.0
        if any(p in stripped for p in CONTINUATION_INTENT_PATTERNS):
            logger.debug(
                "[%s] semantic_completeness_score: continuation intent %r → 0.0",
                self.tag, text[:30],
            )
            return 0.0
        is_valid, reason = self._is_valid_speech(text)
        if not is_valid:
            logger.debug(
                "[%s] semantic_completeness_score: invalid speech %r — %s → 0.0",
                self.tag, text, reason,
            )
            return 0.0
        # Raw model + context adjustment (same as p_complete_score path).
        return self.p_complete_score(text)

    def _calculate_confidence(self, score: float, adjustment: float) -> str:
        """Calculate confidence level."""
        if abs(score - 0.5) < 0.15 and abs(adjustment) > 0.1:
            return "low"
        elif abs(score - 0.5) < 0.25:
            return "medium"
        else:
            return "high"

    def _generate_reason(
        self,
        text: str,
        base_score: float,
        adjustment: float,
        profile: Optional[UserProfile],
    ) -> str:
        """Generate decision reason."""
        reasons = []

        if self._is_follow_up(text):
            reasons.append("follow_up")
        if self._has_strong_ending(text):
            reasons.append("strong_ending")
        if self._is_mostly_filler(text):
            reasons.append("mostly_filler")
        if profile:
            reasons.append(f"pace:{profile.speaking_pace}")

        if not reasons:
            reasons.append("base_eot_context")

        return ";".join(reasons)

    def record_turn(
        self, text: str, is_complete: bool, eot_score: float
    ) -> None:
        """Record a turn of dialogue."""
        turn = TurnContext(
            text=text,
            timestamp=time.time(),
            is_complete=is_complete,
            utterance_end_score=eot_score,
            turn_type="user",
        )
        self._dialogue_history.append(turn)
        self._last_turn_time = time.time()
        self._current_turn_start = time.time()

    def record_interrupt(self, text: str) -> None:
        """Record that an interrupt has just been issued.

        Updates the cooldown / similarity bookkeeping that
        :meth:`compute_score` consults on subsequent calls. This is the
        public replacement for direct private-field writes
        (``_last_interrupt_time`` / ``_last_text``) — see G0a in Round 7.
        """
        self._last_interrupt_time = time.time()
        self._last_text = text

    def reset_turn(self) -> None:
        """Reset per-turn state while preserving dialogue history and user profiles."""
        self._last_interrupt_time = None
        self._last_text = ""
        self._current_eot_score = 0.0
        self._current_turn_start = time.time()
        logger.debug(f"[{self.tag}] Turn reset complete (history preserved: {len(self._dialogue_history)} turns)")

    def reset_session(self) -> None:
        """Full session reset — clears dialogue history, profiles, and all state."""
        self._dialogue_history.clear()
        self._last_turn_time = 0.0
        self._current_turn_start = time.time()

        self._last_interrupt_time = None
        self._last_text = ""
        self._current_eot_score = 0.0

        logger.debug(f"[{self.tag}] Session reset complete")

    def get_current_state(self) -> Dict[str, Any]:
        """Get current state (for testing and debugging)."""
        return {
            "text": self._last_text,
            "eot_score": self._current_eot_score,
            "last_interrupt_time": self._last_interrupt_time,
            "history_length": len(self._dialogue_history),
        }

    def _check_cooldown(self) -> Tuple[bool, Optional[str]]:
        """
        Check if in cooldown period.

        Returns:
            (is_in_cooldown, reason)
        """
        if self._last_interrupt_time is None:
            return False, None

        elapsed = time.time() - self._last_interrupt_time
        if elapsed < self._cooldown_period:
            return True, f"cooldown({elapsed:.1f}s < {self._cooldown_period}s)"

        return False, None

    def _check_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using longest-common-prefix ratio.

        Prefix-based similarity is more appropriate for streaming ASR text
        which evolves by appending characters.

        Returns:
            similarity: 0.0-1.0, higher means more similar
        """
        if not text1 or not text2:
            return 0.0

        from .utils import longest_common_prefix_len

        prefix_len = longest_common_prefix_len(text1, text2)
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 0.0
        return prefix_len / max_len

    def _is_valid_speech(self, text: str) -> Tuple[bool, str]:
        """
        Check if text is valid speech input.

        Returns:
            (is_valid, reason)
        """
        text = text.strip()

        if not text:
            return False, "empty"

        if text in FILLER_WORDS:
            return False, "filler_word"

        if len(text) <= 2:
            if text not in GREETING_WORDS and text not in SHORT_COMPLETE_PHRASES:
                return False, f"too_short({len(text)} chars)"

        return True, "valid"

    def compute_score(
        self,
        text: str,
        is_final: bool = False,
    ) -> float:
        """
        Compute the context-enhanced EOT score (0-1).

        This is a pure scorer — it does NOT make any cut decision and does NOT
        update internal state (cooldown, last_text). The caller (base.py) is
        responsible for both the threshold decision and any side effects.

        Steps:
        1. Short/invalid speech filtering
        2. Cooldown period check
        3. Similarity check (after cooldown)
        4. EOT score + context enhancement

        Args:
            text: Current input text.
            is_final: Whether ASR has returned final result.

        Returns:
            Context-enhanced EOT score in [0.0, 1.0].
        """
        # Step 1+2 (continuation-intent + too-short/filler) live in the
        # shared ``semantic_completeness_score`` so both this path and
        # framework's ``predict_end_of_turn`` agree on what counts as a
        # valid turn end. Returns 0.0 when text is not a valid completion.
        semantic_score = self.semantic_completeness_score(text)
        if semantic_score == 0.0:
            return 0.0

        # Step 3: Cooldown check (interrupt-specific — protects against
        # double-cutting, has no analogue in endpointing).
        in_cooldown, cooldown_reason = self._check_cooldown()
        if in_cooldown:
            logger.debug(f"[{self.tag}] compute_score: in cooldown - {cooldown_reason}")
            return 0.0

        # Step 4: Similarity check after cooldown (interrupt-specific —
        # blocks repeat cuts on near-identical text within window).
        if self._last_text and not in_cooldown:
            similarity = self._check_similarity(text, self._last_text)
            if similarity >= self._similarity_threshold:
                logger.debug(
                    f"[{self.tag}] compute_score: blocked by similarity {similarity:.2f} >= {self._similarity_threshold}"
                )
                return 0.0

        # Decompose for log compatibility (downstream tests grep "base=...adj=...").
        eot_score = self._base_eot.p_complete_score(text)
        profile = None
        if self._enable_profile and self._current_session_id:
            profile = self._user_profiles.get(self._current_session_id)
        adjustment = self._get_context_adjustment(text, eot_score, profile)

        logger.debug(
            f"[{self.tag}] compute_score: text={text[:30]!r} "
            f"base={eot_score:.3f} adj={adjustment:.3f} final={semantic_score:.3f}"
        )
        return semantic_score

    # Alias for backward compat — old call sites still reference should_cut.
    def should_cut(self, text: str, is_final: bool = False) -> float:
        """
        Alias for compute_score(). Returns the context-enhanced EOT score (0-1).
        Threshold comparison is no longer done here; it is handled by the caller.
        """
        return self.compute_score(text, is_final=is_final)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        stats = {
            "history_length": len(self._dialogue_history),
            "user_profiles_count": len(self._user_profiles),
        }

        if self._current_session_id:
            profile = self._user_profiles.get(self._current_session_id)
            if profile:
                stats["current_user"] = {
                    "avg_length": profile.avg_response_length,
                    "pace": profile.speaking_pace,
                    "filler_ratio": profile.filler_word_ratio,
                }

        return stats
