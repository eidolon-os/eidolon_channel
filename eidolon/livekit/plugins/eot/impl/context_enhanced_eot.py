"""Provider-neutral learned EOT scoring with interruption state guards.

Transcript wording is evidence for the learned EOT model, never an input to
hand-maintained phrase rules. This wrapper owns only state not provided by the
model or LiveKit: session history counters, recent-cut cooldown, and streaming
prefix duplicate suppression.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from .eot_manager import EotManager


@dataclass(frozen=True, slots=True)
class TurnContext:
    text: str
    timestamp: float
    is_complete: bool
    utterance_end_score: float


class ContextEnhancedEot:
    """Learned EOT score plus non-linguistic interruption guards."""

    tag = "ContextEnhancedEot"

    def __init__(
        self,
        base_eot: EotManager | None = None,
        max_history: int = 3,
        cooldown_period: float = 0.8,
        similarity_threshold: float = 0.85,
    ) -> None:
        self._base_eot = base_eot or EotManager()
        self._dialogue_history: deque[TurnContext] = deque(maxlen=max_history)
        self._cooldown_period = cooldown_period
        self._similarity_threshold = similarity_threshold
        self._current_session_id: str | None = None
        self._last_interrupt_time: float | None = None
        self._last_text = ""
        self._current_eot_score = 0.0

    def start_session(self, session_id: str) -> None:
        self._current_session_id = session_id
        self.reset_session()

    def end_session(self, session_id: str | None = None) -> None:
        if session_id is not None and session_id != self._current_session_id:
            return
        self.reset_session()
        self._current_session_id = None

    def semantic_completeness_score(self, text: str) -> float:
        stripped = (text or "").strip()
        if not stripped:
            self._current_eot_score = 0.0
            return 0.0
        score = float(self._base_eot.p_complete_score(stripped))
        self._current_eot_score = max(0.0, min(1.0, score))
        return self._current_eot_score

    def p_complete_score(self, text: str) -> float:
        return self.semantic_completeness_score(text)

    def record_turn(self, text: str, is_complete: bool, eot_score: float) -> None:
        self._dialogue_history.append(
            TurnContext(
                text=text,
                timestamp=time.time(),
                is_complete=is_complete,
                utterance_end_score=eot_score,
            )
        )

    def record_interrupt(self, text: str) -> None:
        self._last_interrupt_time = time.time()
        self._last_text = text

    def reset_turn(self) -> None:
        self._last_interrupt_time = None
        self._last_text = ""
        self._current_eot_score = 0.0

    def reset_session(self) -> None:
        self._dialogue_history.clear()
        self.reset_turn()

    def get_current_state(self) -> dict[str, object]:
        return {
            "text": self._last_text,
            "eot_score": self._current_eot_score,
            "last_interrupt_time": self._last_interrupt_time,
            "history_length": len(self._dialogue_history),
        }

    def _in_cooldown(self) -> bool:
        return bool(
            self._last_interrupt_time is not None
            and time.time() - self._last_interrupt_time < self._cooldown_period
        )

    @staticmethod
    def _prefix_similarity(left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        from .utils import longest_common_prefix_len

        return float(longest_common_prefix_len(left, right)) / max(len(left), len(right))

    def compute_score(self, text: str) -> float:
        score = self.semantic_completeness_score(text)
        if score == 0.0 or self._in_cooldown():
            return 0.0
        if (
            self._last_text
            and self._prefix_similarity(text, self._last_text) >= self._similarity_threshold
        ):
            return 0.0
        return score

    def get_stats(self) -> dict[str, object]:
        return {
            "history_length": len(self._dialogue_history),
            "session_active": self._current_session_id is not None,
        }
