# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""G18b (2026-05-18) — interrupt-decision logic consolidated into one module.

============================================================================
WHY
============================================================================

Until G18b the interrupt decision was spread across four methods in
``StreamingPipeline``:

  * ``_run_eot_check`` duck-active branch — first-signal cancel /
    score-based cancel / score-based unduck / mid-band hold (G18a).

  * ``_duck_suspend_timeout_fallback`` — at the 500 ms deadline, branch
    on VAD-still-active to either cancel (trust VAD) or unduck
    (drop_buffered, G17a + G18a).

  * ``_duck_cancel_and_interrupt`` — real-interrupt path (snapshot
    context, cancel mixer, interrupt TTS).

  * ``_duck_unduck_if_suspended`` — false-interrupt path (timer cancel,
    unduck mixer, optional drop_buffered).

The decision policy and the I/O orchestration were mixed. This module
splits them: :class:`InterruptDecider` returns a pure :class:`Decision`
(an enum + reason string) given the current signal, and
``StreamingPipeline`` executes the side effects.

============================================================================
SCOPE
============================================================================

Phase 2 extraction. **No behaviour change**: the policy here is byte-for-
byte equivalent to what G18a put in ``StreamingPipeline`` last commit.
Subsequent Phase 2 work (G18c, G17c) will trim what's not needed.

============================================================================
TESTING
============================================================================

Pure functions = pure unit tests. See
``eidolon/livekit/tests/agent/test_interrupt_decider.py``. Integration
behaviour (timer wiring, mixer side effects) stays covered by the
existing ``test_first_signal_trigger.py`` against ``StreamingPipeline``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


# Backchannel / noise-like transcript sets are owned by the EOT plugin's
# policy module; we re-import here so the decider stays the single
# authority on "is this transcript meaningful enough to cancel".
from eidolon.livekit.plugins.eot.impl.eot_policy import (
    BACKCHANNEL_WORDS,
    NOISE_LIKE_TRANSCRIPTIONS,
)


class Action(Enum):
    """Concrete decision outcome.

    NONE       — keep waiting; signal didn't cross any threshold.
    CANCEL     — confirmed real interrupt; cancel TTS + drop in-flight.
    ROLLBACK   — confirmed false interrupt; resume agent's speech.
    HOLD       — duck-active mid-band score; wait for next INTERIM or timeout.
    """

    NONE = "none"
    CANCEL = "cancel"
    ROLLBACK = "rollback"
    HOLD = "hold"


@dataclass(frozen=True)
class Decision:
    """The decider's verdict on one input signal.

    Attributes:
        action: What the caller should do (Action.NONE = no-op).
        reason: Short human-readable explanation, used for log lines and
            telemetry ("reason=..." field in production logs).
        rollback_drop_buffered: Hint to the caller for the ROLLBACK action.
            True = drop the buffered TTS frames (stale, ≥500 ms old);
            False = drain them with fade-in (fresh, fast soft-unduck).
    """

    action: Action
    reason: str
    rollback_drop_buffered: bool = False


def _strip_punct(text: str) -> str:
    return text.strip().rstrip("。.!？?！,，")


def is_backchannel_text(text: str) -> bool:
    """Return True when ``text`` is a backchannel or noise-like
    vocalization that should NOT trigger cancellation."""
    stripped = _strip_punct(text)
    if not stripped:
        return True
    if stripped.lower() in BACKCHANNEL_WORDS:
        return True
    if stripped in NOISE_LIKE_TRANSCRIPTIONS:
        return True
    return False


class InterruptDecider:
    """Pure decision policy. No side effects, no timers, no async.

    The caller (``StreamingPipeline``) feeds it events and acts on the
    returned :class:`Decision`. Designed so unit tests can hammer the
    state space without needing a full pipeline harness.

    Args:
        min_interim_chars: Minimum (post-punctuation-strip) length of an
            STT INTERIM to be considered a "real interrupt" signal in the
            first-signal fast path. Default 2 — aligned with industry
            (Pipecat ``interrupt_min_words``).
        early_cancel_score_threshold: EOT score at/above which the
            score-based cancel path fires. Phase 1 fallback for slow STT.
        early_resume_score_threshold: EOT score at/below which the
            score-based unduck path fires (false interrupt confirmed by
            low-confidence transcript).
    """

    def __init__(
        self,
        *,
        min_interim_chars: int = 2,
        early_cancel_score_threshold: float = 0.7,
        early_resume_score_threshold: float = 0.2,
    ) -> None:
        self._min_interim_chars = max(1, int(min_interim_chars))
        self._cancel_thr = float(early_cancel_score_threshold)
        self._resume_thr = float(early_resume_score_threshold)

    # ------------------------------------------------------------------
    # Decision entry points
    # ------------------------------------------------------------------

    def on_strong_intent(self) -> Decision:
        """Strong-interrupt-intent classifier already fired (e.g. "停下",
        explicit cancellation phrases). Always cancel."""
        return Decision(action=Action.CANCEL, reason="strong_intent")

    def on_stt_interim(self, text: str, score: float) -> Decision:
        """Score one STT INTERIM during the SUSPENDED window.

        Priority chain:

          1. First-signal: non-backchannel + length ≥ min_chars → CANCEL.
             (G18a fast-path; bypasses score, lands in <500 ms.)

          2. Score crossing high threshold → CANCEL.
             (fallback for slow STT with confident EOT score.)

          3. Score crossing low threshold (and > 0) → ROLLBACK.
             (false-positive confirmed by low-confidence transcript;
             drop_buffered=False because this is a fast soft-unduck.)

          4. Otherwise → HOLD (wait for next interim or deadline).
        """
        stripped = _strip_punct(text)

        # 1. first-signal fast path
        if (
            not is_backchannel_text(text)
            and len(stripped) >= self._min_interim_chars
        ):
            return Decision(
                action=Action.CANCEL,
                reason=f"first_signal_interim len={len(stripped)}",
            )

        # 2. high-score cancel
        if score >= self._cancel_thr:
            return Decision(
                action=Action.CANCEL,
                reason=f"eot_score_high score={score:.2f}>={self._cancel_thr:.2f}",
            )

        # 3. low-score rollback (positive low; 0.0 means "no signal yet")
        if 0.0 < score <= self._resume_thr:
            return Decision(
                action=Action.ROLLBACK,
                reason=f"eot_score_low score={score:.2f}<={self._resume_thr:.2f}",
                rollback_drop_buffered=False,
            )

        # 4. mid-band — wait
        return Decision(
            action=Action.HOLD,
            reason=f"eot_score_mid score={score:.2f}",
        )

    def on_decision_deadline(self, vad_still_active: bool) -> Decision:
        """The 500 ms decision budget has elapsed without a definitive
        signal. Branch on VAD: still speaking → cancel (trust VAD);
        silent → rollback with drop_buffered=True (stale)."""
        if vad_still_active:
            return Decision(
                action=Action.CANCEL,
                reason="deadline_trust_vad",
            )
        return Decision(
            action=Action.ROLLBACK,
            reason="deadline_vad_idle_drop_stale",
            rollback_drop_buffered=True,
        )

    def on_user_silent(self) -> Decision:
        """User went silent before the decision deadline (VAD speaking →
        listening transition within ~300 ms). Always rollback with drain
        (the buffer is genuinely fresh)."""
        return Decision(
            action=Action.ROLLBACK,
            reason="user_silent_fast_rollback",
            rollback_drop_buffered=False,
        )


__all__ = ["Action", "Decision", "InterruptDecider", "is_backchannel_text"]
