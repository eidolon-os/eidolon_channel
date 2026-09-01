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
Turn Detection state management.
Copied from eidolon/pipeline/src/processor/turn_detector/state.py
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VADState:
    """VAD state."""

    active: bool = False
    active_since: Optional[float] = None
    silence_since: Optional[float] = None
    last_active_time: float = 0.0
    last_silence_time: float = 0.0
    transition_times: deque[float] = field(default_factory=lambda: deque(maxlen=20))
    confirmed_active: bool = False
    has_active_in_segment: bool = False
    # Round 7 G6: per-frame VAD inference probability samples for confidence
    # gating. Each entry is (timestamp, probability ∈ [0, 1]). Window size
    # 200 samples × ~32 ms inference window ≈ 6 s of history (more than
    # enough to compute a 2 s rolling average).
    probability_samples: deque[tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=200)
    )


@dataclass
class ASRState:
    """ASR state."""

    current_text: str = ""
    last_text: str = ""
    is_final: bool = False
    last_final_time: float = 0.0
    stability_start_time: Optional[float] = None
    stable_since: Optional[float] = None


@dataclass
class SentenceState:
    """Sentence-level state."""

    start_time: Optional[float] = None
    last_cut_time: float = 0.0
    last_cut_text_hash: Optional[str] = None
    last_cut_text: Optional[str] = None
    turn_key: Optional[tuple[str, int]] = None


@dataclass
class InterruptState:
    """Interrupt-related state."""

    last_interrupt_time: float = 0.0
    cooldown_seconds: float = 0.8


@dataclass
class NoiseState:
    """Noise mode state."""

    enabled_until: float = 0.0
    flip_count: int = 0
    flip_window_start: float = 0.0


class TurnDetectionStateManager:
    """
    Unified state manager for turn detection.
    """

    def __init__(
        self,
        max_sentence_duration: float = 15.0,
        vad_stale_timeout: float = 0.35,
        min_cut_interval: float = 0.5,
    ):
        self._vad = VADState()
        self._asr = ASRState()
        self._sentence = SentenceState()
        self._interrupt = InterruptState()
        self._noise = NoiseState()

        self.max_sentence_duration = max_sentence_duration
        self.vad_stale_timeout = vad_stale_timeout
        self.min_cut_interval = min_cut_interval

        self._current_eot_score: float = 0.0

    @property
    def vad_active(self) -> bool:
        return self._vad.active

    @vad_active.setter
    def vad_active(self, value: bool) -> None:
        self._vad.active = value

    @property
    def current_text(self) -> str:
        return self._asr.current_text

    @current_text.setter
    def current_text(self, value: str) -> None:
        self._asr.current_text = value

    @property
    def eot_score(self) -> float:
        return self._current_eot_score

    def update_vad(
        self,
        is_active: bool,
        speech_duration: float = 0.0,
        silence_duration: float = 0.0,
    ) -> None:
        """Update VAD state."""
        now = time.time()

        if is_active != self._vad.active:
            self._vad.transition_times.append(now)

        self._vad.active = is_active

        if is_active:
            self._vad.has_active_in_segment = True
            if self._vad.active_since is None:
                self._vad.active_since = now
            self._vad.last_active_time = now
            if self._sentence.start_time is None:
                self._sentence.start_time = now
        else:
            self._vad.active_since = None
            self._vad.last_silence_time = now

    def update_vad_probability(self, probability: float) -> None:
        """Round 7 G6: record a per-frame VAD inference probability.

        Called by the orchestrator (StreamingPipeline) with each
        ``INFERENCE_DONE`` event from the FireRed VAD callback. Lets policies
        consult ``recent_avg_vad_confidence`` for confidence-gated decisions
        (e.g. block cuts when VAD is uncertain — likely echo/noise).

        Args:
            probability: Probability in [0, 1] from the VAD's last inference.
        """
        # Clamp to valid range and ignore None just in case.
        if probability is None:
            return
        prob = max(0.0, min(1.0, float(probability)))
        self._vad.probability_samples.append((time.time(), prob))

    def recent_avg_vad_confidence(self, window_sec: float = 2.0) -> float:
        """Round 7 G6: average VAD probability over the last ``window_sec``.

        Returns 0.0 when no samples are present (e.g. VAD callback hasn't
        been wired yet, or no inference has run). Callers comparing against
        a threshold (e.g. ``< 0.6 → likely noise``) should treat 0.0 as
        "unknown" — i.e. don't gate on this signal until samples accumulate.

        Args:
            window_sec: Time window in seconds (default 2.0).

        Returns:
            Average probability ∈ [0, 1], or 0.0 if no recent samples.
        """
        if not self._vad.probability_samples:
            return 0.0
        cutoff = time.time() - window_sec
        recent = [p for (t, p) in self._vad.probability_samples if t >= cutoff]
        if not recent:
            return 0.0
        return sum(recent) / len(recent)

    def update_asr(self, text: str, is_final: bool) -> None:
        """Update ASR state."""
        now = time.time()

        self._asr.last_text = self._asr.current_text
        self._asr.current_text = text
        self._asr.is_final = is_final

        if is_final:
            self._asr.last_final_time = now
            # Text is confirmed stable at this point.
            if self._asr.stable_since is None:
                self._asr.stable_since = now
        else:
            # Text is still streaming — reset stability tracker.
            self._asr.stable_since = None

    def update_eot_score(self, score: float) -> None:
        """Update EOT score."""
        self._current_eot_score = score

    def update_interrupt(self) -> None:
        """Record interrupt."""
        self._interrupt.last_interrupt_time = time.time()

    def check_interrupt_cooldown(self, now: Optional[float] = None) -> bool:
        """Check if in interrupt cooldown."""
        if now is None:
            now = time.time()
        return (
            now - self._interrupt.last_interrupt_time
        ) < self._interrupt.cooldown_seconds

    def check_vad_stale(self, now: Optional[float] = None) -> bool:
        """Check if VAD is stale."""
        if now is None:
            now = time.time()
        if not self._vad.active:
            return False
        return (now - self._vad.last_active_time) > self.vad_stale_timeout

    def check_max_duration(self, now: Optional[float] = None) -> bool:
        """Check if max sentence duration exceeded."""
        if now is None:
            now = time.time()
        if self._sentence.start_time is None:
            return False
        return (now - self._sentence.start_time) > self.max_sentence_duration

    def check_min_interval(self, now: Optional[float] = None) -> bool:
        """Check if min cut interval satisfied."""
        if now is None:
            now = time.time()
        return (now - self._sentence.last_cut_time) >= self.min_cut_interval

    def record_cut(self, text_hash: str, text: Optional[str] = None, now: Optional[float] = None) -> None:
        """Record a cut."""
        if now is None:
            now = time.time()
        self._sentence.last_cut_time = now
        self._sentence.last_cut_text_hash = text_hash
        self._sentence.last_cut_text = text

    def reset_session(self, session_id: str, turn_id: int = 0) -> None:
        """Reset full session state (used at session boundary).

        Wipes VAD state including in-progress speech timers — only safe
        when we're sure no speech segment is currently in flight. Use
        :meth:`reset_turn` between turns to avoid losing VAD state for
        the very next utterance.
        """
        self._vad = VADState()
        self._asr = ASRState()
        self._sentence.start_time = None
        self._sentence.turn_key = (session_id, turn_id)
        self._current_eot_score = 0.0

    def reset_turn(self) -> None:
        """Reset per-turn state; preserve VAD activity tracking.

        Called at every ``user_state: speaking → listening`` transition
        from :class:`StreamingPipeline`. Previously this used
        ``reset_session("", 0)`` which also rebuilt :class:`VADState` —
        in the streaming-STT regime the next utterance can begin within
        a few hundred ms (Bailian interim), causing
        ``MinSpeakingDurationPolicy`` to read ``vad.active_since=None``
        and report ``Speech too short 0.000s`` for every interim cut.

        We now keep :class:`VADState` intact so the caller's
        ``update_vad(False)`` (already issued before reset) and the
        subsequent ``update_vad(True)`` (when the next VAD start fires)
        manage active/active_since correctly. Only ASR / sentence /
        score state are cleared.
        """
        self._asr = ASRState()
        self._sentence.start_time = None
        self._current_eot_score = 0.0

    def get_silence_duration(self, now: Optional[float] = None) -> float:
        """Get current silence duration."""
        if now is None:
            now = time.time()
        if self._vad.active:
            return 0.0
        if self._vad.last_active_time == 0:
            return 0.0
        return now - self._vad.last_active_time

    def get_speech_duration(self, now: Optional[float] = None) -> float:
        """Get current speech duration."""
        if now is None:
            now = time.time()
        if not self._vad.active:
            return 0.0
        if self._vad.active_since is None:
            return 0.0
        return now - self._vad.active_since
