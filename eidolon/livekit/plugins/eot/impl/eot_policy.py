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
Turn Detection policy chain.
Copied from eidon/pipeline/src/processor/turn_detector/policies.py
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from .constants import (
    ASR_TRAILING_PUNCTUATION,
    BACKCHANNEL_COMPOUND_CHARS,
    BACKCHANNEL_WORDS,
    NOISE_LIKE_TRANSCRIPTIONS,
    REPEATED_NOISE_CHARS,
)
from .state import TurnDetectionStateManager
from .turn_end_policy import TurnEndPolicy
from .utils import compute_text_hash, is_similar_text


@dataclass
class CutDecision:
    """Cut decision result."""

    should_cut: bool
    reason: str
    silence_duration: float = 0.0


class CutPolicy(ABC):
    """Base class for cut policies."""

    @abstractmethod
    def check(
        self,
        state: TurnDetectionStateManager,
        turn_end_policy: TurnEndPolicy,
    ) -> Optional[CutDecision]:
        """
        Check if should cut.

        Returns:
            CutDecision if cut needed, None otherwise (let chain continue).
        """
        pass


# -----------------------------------------------------------------------------
# Guard Policies — block cuts without making a decision themselves
# -----------------------------------------------------------------------------


class InterruptCooldownPolicy(CutPolicy):
    """Interrupt cooldown check: blocks cuts if cooldown has not elapsed yet."""

    def check(
        self,
        state: TurnDetectionStateManager,
        turn_end_policy: TurnEndPolicy,
    ) -> Optional[CutDecision]:
        if state.check_interrupt_cooldown():
            return CutDecision(
                should_cut=False,
                reason="Interrupt cooldown",
                silence_duration=0.0,
            )
        return None


class MinIntervalPolicy(CutPolicy):
    """Minimum interval check: blocks cuts if too little time has passed since last cut."""

    def check(
        self,
        state: TurnDetectionStateManager,
        turn_end_policy: TurnEndPolicy,
    ) -> Optional[CutDecision]:
        if not state.check_min_interval():
            return CutDecision(
                should_cut=False,
                reason=f"Min interval {state.min_cut_interval}s not met",
                silence_duration=0.0,
            )
        return None


class BackchannelSuppressionPolicy(CutPolicy):
    """Block cuts when ASR text is a backchannel acknowledgement.

    Backchannels are short utterances ("嗯嗯", "好的", "OK") that a listener
    emits while the speaker continues — they're acknowledgements, NOT
    turn-takes. Without this guard, a user nodding along with "嗯嗯" while
    the agent is talking would falsely trigger an interrupt.

    Used only in the **semantic interruption** chain (where the agent is
    speaking and the user starts talking). In the normal turn-end chain,
    a user uttering "嗯嗯" is genuinely the entire turn and should be
    treated as completion.

    Added in Round 7 G1.
    """

    def check(
        self,
        state: TurnDetectionStateManager,
        turn_end_policy: TurnEndPolicy,
    ) -> Optional[CutDecision]:
        text = state.current_text
        if not text:
            return None
        # Strip end punctuation that ASR sometimes adds.
        stripped = text.strip().rstrip(ASR_TRAILING_PUNCTUATION)
        if not stripped:
            return None
        if stripped.lower() in BACKCHANNEL_WORDS:
            return CutDecision(
                should_cut=False,
                reason=f"Backchannel detected: {stripped!r}",
                silence_duration=0.0,
            )
        # Compound backchannel like "嗯嗯好" / "嗯好的" — rare but seen in practice.
        # Only consider very short texts (≤ 4 chars) to avoid false positives.
        if 2 <= len(stripped) <= 4 and all(
            stripped[i] in BACKCHANNEL_COMPOUND_CHARS for i in range(len(stripped))
        ):
            return CutDecision(
                should_cut=False,
                reason=f"Compound backchannel: {stripped!r}",
                silence_duration=0.0,
            )
        return None


class NoiseLikeTranscriptPolicy(CutPolicy):
    """Block cuts when ASR transcript looks like a vocalization (cough / sigh).

    Coughs, throat-clearing, and sighs frequently get transcribed by ASR
    as single-character or repeated-character utterances ("啊", "啊啊啊",
    "咳", "嗯哼"). These are not intentional speech and shouldn't drive
    turn-end / interrupt decisions.

    Complementary to ``MinSpeakingDurationPolicy`` (which catches short
    audio segments) and ``VADStabilityPolicy`` (which catches rapid VAD
    flapping). This one targets the case where ASR DID emit a transcript
    but the content itself is non-lexical noise.

    Added in Round 7 G2a.
    """

    def check(
        self,
        state: TurnDetectionStateManager,
        turn_end_policy: TurnEndPolicy,
    ) -> Optional[CutDecision]:
        text = state.current_text
        if not text:
            return None
        stripped = text.strip().rstrip(ASR_TRAILING_PUNCTUATION)
        if not stripped:
            return None
        if stripped in NOISE_LIKE_TRANSCRIPTIONS:
            return CutDecision(
                should_cut=False,
                reason=f"Noise-like transcript: {stripped!r}",
                silence_duration=0.0,
            )
        # Detect repeated single-character sequences ("啊啊啊啊", "咳咳咳"),
        # which are typical of coughs / sighs / extended noises.
        if 2 <= len(stripped) <= 6 and len(set(stripped)) == 1:
            char = stripped[0]
            if char in REPEATED_NOISE_CHARS:
                return CutDecision(
                    should_cut=False,
                    reason=f"Repeated noise char: {stripped!r}",
                    silence_duration=0.0,
                )
        return None


class DuplicateTextPolicy(CutPolicy):
    """Duplicate text check: blocks cuts if current text is too similar to last cut text.

    Uses Jaccard similarity when last_cut_text is available (preferred), falls back to
    exact hash match for backward compatibility.
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    def check(
        self,
        state: TurnDetectionStateManager,
        turn_end_policy: TurnEndPolicy,
    ) -> Optional[CutDecision]:
        current_text = state.current_text.strip()
        if not current_text:
            return None

        last_text = state._sentence.last_cut_text
        if last_text:
            # Use Jaccard similarity for richer duplicate detection.
            if is_similar_text(current_text, last_text, self.similarity_threshold):
                return CutDecision(
                    should_cut=False,
                    reason="Duplicate text (similarity)",
                    silence_duration=0.0,
                )
        else:
            # Fallback to exact hash match when no last_cut_text is stored.
            last_hash = state._sentence.last_cut_text_hash
            if last_hash and compute_text_hash(current_text) == last_hash:
                return CutDecision(
                    should_cut=False,
                    reason="Duplicate text (hash)",
                    silence_duration=0.0,
                )
        return None


class MinSpeakingDurationPolicy(CutPolicy):
    """Minimum speaking duration: blocks cuts if the user's speech segment is too short.

    Filters out coughs, sneezes, ambient noise, and other non-intentional vocalizations
    that are shorter than min_speech_duration_sec.

    G18c (2026-05-18) — the original Round 7 G2b "average VAD probability"
    gate was removed. It required ≥1s of frame accumulation to stabilise,
    which made it the single largest contributor to the multi-second
    interrupt-delay problem fixed by G18a's first-signal triggers. The
    InterruptDecider's first-INTERIM + backchannel filter handles echo /
    noise classification with no latency penalty.

    The ``min_avg_vad_confidence`` and ``confidence_window_sec`` constructor
    arguments remain for backward compatibility but are NO LONGER USED
    by ``check()``. They will be removed in a follow-up release once
    callers (and tests) have been migrated.
    """

    def __init__(
        self,
        min_speech_duration_sec: float = 0.15,
        min_avg_vad_confidence: float = 0.0,  # G18c: unused, kept for compat
        confidence_window_sec: float = 2.0,  # G18c: unused, kept for compat
    ):
        self.min_speech_duration_sec = min_speech_duration_sec
        # G18c (2026-05-18): these fields are kept as no-op attributes for
        # the deprecation transition. Reading them is fine; their values
        # have no effect on cut decisions.
        self.min_avg_vad_confidence = min_avg_vad_confidence
        self.confidence_window_sec = confidence_window_sec

    def check(
        self,
        state: TurnDetectionStateManager,
        turn_end_policy: TurnEndPolicy,
    ) -> Optional[CutDecision]:
        if not state.vad_active:
            return None

        speech_duration = state.get_speech_duration()
        if speech_duration < self.min_speech_duration_sec:
            return CutDecision(
                should_cut=False,
                reason=f"Speech too short {speech_duration:.3f}s < {self.min_speech_duration_sec}s",
                silence_duration=0.0,
            )
        # G18c: no more VAD-confidence-ramp gate. Returning None hands the
        # decision back to the chain (subsequent policies, EOT semantic
        # score, then InterruptDecider).
        return None


class VADStabilityPolicy(CutPolicy):
    """VAD stability check: blocks cuts if VAD has been toggling too rapidly.

    Filters out coughs, ambient noise, and other non-speech VAD events that cause
    rapid active/inactive transitions within a short time window.
    """

    def __init__(
        self,
        flip_window_sec: float = 1.0,
        flip_count_threshold: int = 3,
    ):
        self.flip_window_sec = flip_window_sec
        self.flip_count_threshold = flip_count_threshold

    def check(
        self,
        state: TurnDetectionStateManager,
        turn_end_policy: TurnEndPolicy,
    ) -> Optional[CutDecision]:
        # Only relevant when VAD is active (user is speaking).
        if not state.vad_active:
            return None

        transitions = state._vad.transition_times
        if not transitions:
            return None

        now = time.time()
        recent = [t for t in transitions if now - t <= self.flip_window_sec]
        if len(recent) >= self.flip_count_threshold:
            return CutDecision(
                should_cut=False,
                reason=f"VAD unstable ({len(recent)} toggles in {self.flip_window_sec}s)",
                silence_duration=0.0,
            )
        return None


class ASRStabilityPolicy(CutPolicy):
    """ASR stability check: requires ASR text to be confirmed stable before allowing cut.

    Prevents cutting on partial transcripts that haven't been confirmed as the final
    result yet. Short texts stabilize faster than long texts.
    """

    def __init__(
        self,
        stability_short_sec: float = 0.10,
        stability_long_sec: float = 0.40,
        long_char_threshold: int = 10,
    ):
        self.stability_short_sec = stability_short_sec
        self.stability_long_sec = stability_long_sec
        self.long_char_threshold = long_char_threshold

    def check(
        self,
        state: TurnDetectionStateManager,
        turn_end_policy: TurnEndPolicy,
    ) -> Optional[CutDecision]:
        # Only relevant for confirmed ASR final.
        if not state._asr.is_final:
            return None

        text = state.current_text
        stability_window = (
            self.stability_long_sec
            if len(text) >= self.long_char_threshold
            else self.stability_short_sec
        )

        stable_since = state._asr.stable_since
        if stable_since is None:
            return None

        stable_duration = time.time() - stable_since
        if stable_duration < stability_window:
            return CutDecision(
                should_cut=False,
                reason=f"ASR not stable yet {stable_duration:.2f}s < {stability_window}s",
                silence_duration=state.get_silence_duration(),
            )
        return None


# -----------------------------------------------------------------------------
# Intent Override Policies — intent-based cut decisions
# -----------------------------------------------------------------------------


class InterruptIntentPolicy(CutPolicy):
    """Intent-based override: maps user speech intent to cut decisions.

    Priority:
    1. Strong interrupt intent  → should_cut=True  (e.g. "停", "闭嘴", "不对")
    2. Continuation intent      → should_cut=False (e.g. "再换一个", "继续说")

    All other cases return None, letting the chain continue.
    """

    def check(
        self,
        state: TurnDetectionStateManager,
        turn_end_policy: TurnEndPolicy,
    ) -> Optional[CutDecision]:
        text = state.current_text
        if not text:
            return None

        # Priority 1: Strong interrupt → always cut immediately.
        if turn_end_policy.is_strong_interrupt_intent(text):
            return CutDecision(
                should_cut=True,
                reason="strong interrupt intent",
            )

        # Priority 2: Continuation intent → block the cut.
        if turn_end_policy.is_continuation_intent(text):
            return CutDecision(
                should_cut=False,
                reason="continuation intent",
            )

        return None


# -----------------------------------------------------------------------------
# Cut Policies — make positive cut decisions
# -----------------------------------------------------------------------------


class MaxDurationPolicy(CutPolicy):
    """Max duration safety net: force-cut if sentence exceeds max duration while VAD active.

    This is a last-resort safety net: it only triggers when the user appears to be
    still speaking (vad_active=True) but the sentence has exceeded the maximum duration
    without an EOT signal. If VAD is already silent, the user has finished speaking
    and we should let the normal soft-interrupt flow handle it.
    """

    def check(
        self,
        state: TurnDetectionStateManager,
        turn_end_policy: TurnEndPolicy,
    ) -> Optional[CutDecision]:
        # Only trigger if VAD is still detecting speech.
        if not state.vad_active:
            return None
        if state.check_max_duration():
            return CutDecision(
                should_cut=True,
                reason=f"Max duration {state.max_sentence_duration}s exceeded",
                silence_duration=state.get_silence_duration(),
            )
        return None


class VADStalePolicy(CutPolicy):
    """VAD stale cut policy: force-cut if silence exceeded vad_stale_timeout."""

    def check(
        self,
        state: TurnDetectionStateManager,
        turn_end_policy: TurnEndPolicy,
    ) -> Optional[CutDecision]:
        if state.check_vad_stale():
            return CutDecision(
                should_cut=True,
                reason=f"VAD stale {state.vad_stale_timeout}s",
                silence_duration=state.get_silence_duration(),
            )
        return None


class ASRFinalCutPolicy(CutPolicy):
    """ASR Final cut policy: cut when ASR confirms final + silence elapsed."""

    def __init__(self, final_cut_min_silence_sec: float = 0.2):
        self.final_cut_min_silence_sec = final_cut_min_silence_sec

    def check(
        self,
        state: TurnDetectionStateManager,
        turn_end_policy: TurnEndPolicy,
    ) -> Optional[CutDecision]:
        if not state._asr.is_final:
            return None

        silence_duration = state.get_silence_duration()
        if silence_duration >= self.final_cut_min_silence_sec:
            return CutDecision(
                should_cut=True,
                reason=f"ASR final + silence {silence_duration:.2f}s",
                silence_duration=silence_duration,
            )
        return None


class EOTScorePolicy(CutPolicy):
    """EOT score-based cut policy (silence-duration variant).

    Used for normal turn-end detection: cut when EOT score is high enough AND
    silence duration has exceeded the dynamic threshold computed by TurnEndPolicy.
    """

    def check(
        self,
        state: TurnDetectionStateManager,
        turn_end_policy: TurnEndPolicy,
    ) -> Optional[CutDecision]:
        if not state.current_text:
            return None

        score = state.eot_score
        silence_duration = state.get_silence_duration()

        threshold = turn_end_policy.get_dynamic_threshold(
            state.current_text,
            score,
            state._asr.is_final,
        )

        if silence_duration >= threshold:
            return CutDecision(
                should_cut=True,
                reason=f"EOT score {score:.2f} + silence {silence_duration:.2f}s >= threshold {threshold:.2f}s",
                silence_duration=silence_duration,
            )
        return None


class EOTScoreSemanticPolicy(CutPolicy):
    """EOT score-based cut policy (semantic interruption variant).

    Used when the agent is speaking and the user starts speaking: cut when the
    EOT score exceeds a threshold. Unlike EOTScorePolicy, this uses the EOT score
    directly rather than silence duration (since the user is still speaking).

    Threshold is adjusted only by acoustic VAD state. Lexical phrase lists are
    not turn-control evidence.
    """

    def __init__(
        self,
        base_threshold: float = 0.7,
        weak_intent_threshold: float = 0.5,
        vad_active_delta: float = 0.1,
    ):
        self.base_threshold = base_threshold
        del weak_intent_threshold
        self.vad_active_delta = vad_active_delta

    def check(
        self,
        state: TurnDetectionStateManager,
        turn_end_policy: TurnEndPolicy,
    ) -> Optional[CutDecision]:
        text = state.current_text
        if not text:
            return None

        score = state.eot_score

        threshold = self.base_threshold

        # VAD signal: if VAD active, user is still speaking → slightly lower threshold.
        if state.vad_active:
            threshold -= self.vad_active_delta
        else:
            threshold += self.vad_active_delta

        if score >= threshold:
            return CutDecision(
                should_cut=True,
                reason=f"EOT score {score:.2f} >= threshold {threshold:.2f}",
                silence_duration=state.get_silence_duration(),
            )
        return None


# -----------------------------------------------------------------------------
# Policy Chain
# -----------------------------------------------------------------------------


class PolicyChain:
    """
    Policy chain: combines multiple policies and executes them in priority order.

    The chain stops at the first policy that returns a CutDecision (first-match).
    Returning None means "no opinion" — the chain continues to the next policy.
    """

    def __init__(self, policies: list[CutPolicy]):
        self._policies = policies

    def check(
        self,
        state: TurnDetectionStateManager,
        turn_end_policy: TurnEndPolicy,
    ) -> Optional[CutDecision]:
        """
        Execute policy chain.

        Returns:
            CutDecision: First matching policy result.
            None: No policy matched.
        """
        for policy in self._policies:
            decision = policy.check(state, turn_end_policy)
            if decision is not None:
                return decision
        return None

    @classmethod
    def for_semantic_interruption(
        cls,
        base_threshold: float = 0.7,
        weak_threshold: float = 0.5,
        vad_active_delta: float = 0.1,
        vad_flip_window_sec: float = 1.0,
        vad_flip_count_threshold: int = 3,
        min_speech_duration_sec: float = 0.15,
        similarity_threshold: float = 0.85,
        min_avg_vad_confidence: float = 0.0,
        confidence_window_sec: float = 2.0,
    ) -> "PolicyChain":
        """
        Semantic interruption chain: agent is speaking, user starts speaking.

        Policy order (priority):
        1. InterruptCooldownPolicy      — prevent too-frequent interruptions
        2. MinSpeakingDurationPolicy    — acoustic short-speech guard
        3. VADStabilityPolicy           — filter rapid VAD toggling
        4. MinIntervalPolicy            — interval debounce (guard only, no decision)
        5. DuplicateTextPolicy          — ASR duplicate text filter
        6. MaxDurationPolicy            — duration safety net
        7. VADStalePolicy               — VAD silence safety net
        8. EOTScoreSemanticPolicy       — provider-neutral EOT score threshold

        Order rationale:
          - No fixed phrase list participates. Semantic confidence, VAD and
            timing evidence decide whether to cut.
        """
        return cls(
            [
                InterruptCooldownPolicy(),
                MinSpeakingDurationPolicy(
                    min_speech_duration_sec,
                    min_avg_vad_confidence=min_avg_vad_confidence,
                    confidence_window_sec=confidence_window_sec,
                ),
                VADStabilityPolicy(vad_flip_window_sec, vad_flip_count_threshold),
                MinIntervalPolicy(),
                DuplicateTextPolicy(similarity_threshold),
                MaxDurationPolicy(),
                VADStalePolicy(),
                EOTScoreSemanticPolicy(base_threshold, weak_threshold, vad_active_delta),
            ]
        )

    @classmethod
    def for_normal_turn_end(
        cls,
        final_cut_min_silence_sec: float = 0.2,
        stability_short_sec: float = 0.10,
        stability_long_sec: float = 0.40,
        stability_long_char_threshold: int = 10,
        vad_flip_window_sec: float = 1.0,
        vad_flip_count_threshold: int = 3,
        min_speech_duration_sec: float = 0.15,
        similarity_threshold: float = 0.85,
        min_avg_vad_confidence: float = 0.0,
        confidence_window_sec: float = 2.0,
    ) -> "PolicyChain":
        """
        Normal turn-end chain: agent is idle, user finishes speaking.

        Policy order (priority):
        1. InterruptCooldownPolicy      — prevent too-frequent interruptions
        2. MinSpeakingDurationPolicy     — acoustic short-speech guard
        3. VADStabilityPolicy            — filter rapid VAD toggling
        4. MinIntervalPolicy             — interval debounce (guard only)
        5. DuplicateTextPolicy           — ASR duplicate text filter
        6. ASRStabilityPolicy            — ASR final stability requirement
        7. MaxDurationPolicy             — duration safety net
        8. VADStalePolicy                — VAD silence safety net
        9. ASRFinalCutPolicy             — ASR final + silence cutoff
        10. EOTScorePolicy               — EOT score + silence threshold
        """
        return cls(
            [
                InterruptCooldownPolicy(),
                MinSpeakingDurationPolicy(
                    min_speech_duration_sec,
                    min_avg_vad_confidence=min_avg_vad_confidence,
                    confidence_window_sec=confidence_window_sec,
                ),
                VADStabilityPolicy(vad_flip_window_sec, vad_flip_count_threshold),
                MinIntervalPolicy(),
                DuplicateTextPolicy(similarity_threshold),
                ASRStabilityPolicy(
                    stability_short_sec, stability_long_sec, stability_long_char_threshold
                ),
                MaxDurationPolicy(),
                VADStalePolicy(),
                ASRFinalCutPolicy(final_cut_min_silence_sec),
                EOTScorePolicy(),
            ]
        )
