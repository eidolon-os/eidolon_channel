"""G18b (2026-05-18): InterruptDecider — pure-function decision policy.

The decider is the single source of truth for "given an interrupt-window
signal, what should we do?". These tests cover the policy directly,
without spinning up a full pipeline harness. Integration coverage (timer
wiring, mixer side effects) lives in test_first_signal_trigger.py.
"""

from __future__ import annotations

import pytest

from eidolon.livekit.agent.interrupt_decider import (
    Action,
    Decision,
    InterruptDecider,
    is_backchannel_text,
)


# ---------------------------------------------------------------------------
# is_backchannel_text helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("嗯", True),
        ("嗯嗯", True),
        ("好的", True),
        ("OK", True),
        ("ok", True),
        ("Yeah", True),
        ("嗯。", True),  # trailing punctuation stripped
        ("", True),  # empty counts as backchannel
        ("   ", True),  # whitespace-only stripped → empty
        ("你好", False),
        ("我不相信你", False),
        ("停一下", False),
    ],
)
def test_is_backchannel_text(text: str, expected: bool) -> None:
    assert is_backchannel_text(text) is expected


# ---------------------------------------------------------------------------
# on_strong_intent
# ---------------------------------------------------------------------------


def test_strong_intent_always_cancels() -> None:
    d = InterruptDecider()
    decision = d.on_strong_intent()
    assert decision.action is Action.CANCEL
    assert "strong_intent" in decision.reason


# ---------------------------------------------------------------------------
# on_stt_interim — first-signal cancel
# ---------------------------------------------------------------------------


def test_first_signal_cancel_on_substantive_interim() -> None:
    """Non-backchannel + len ≥ min_chars → immediate cancel (G18a)."""
    d = InterruptDecider(min_interim_chars=2)
    # Score 0 (no signal yet) — first-signal path is the only thing that
    # could fire a cancel here, so this validates it bypasses the score.
    decision = d.on_stt_interim("我不相信", score=0.0)
    assert decision.action is Action.CANCEL
    assert "first_signal" in decision.reason


def test_first_signal_skips_backchannel() -> None:
    """Backchannel-only first INTERIM falls through to HOLD."""
    d = InterruptDecider(min_interim_chars=2)
    decision = d.on_stt_interim("嗯", score=0.0)
    assert decision.action is Action.HOLD


def test_first_signal_skips_compound_backchannel() -> None:
    """Compound backchannel like '嗯嗯' is in the set → HOLD."""
    d = InterruptDecider(min_interim_chars=2)
    decision = d.on_stt_interim("嗯嗯", score=0.0)
    assert decision.action is Action.HOLD


def test_first_signal_below_min_chars_is_hold() -> None:
    """Single-char non-backchannel ('你') under min_chars → HOLD."""
    d = InterruptDecider(min_interim_chars=2)
    decision = d.on_stt_interim("你", score=0.0)
    assert decision.action is Action.HOLD


def test_first_signal_strips_punctuation_before_length_check() -> None:
    """Backchannel + punctuation ('嗯。') is recognized after strip."""
    d = InterruptDecider(min_interim_chars=2)
    decision = d.on_stt_interim("嗯。", score=0.0)
    assert decision.action is Action.HOLD


def test_first_signal_respects_min_chars_setting() -> None:
    d = InterruptDecider(min_interim_chars=5)
    # 3-char substantive interim should NOT trigger with min=5
    decision = d.on_stt_interim("你好世", score=0.0)
    assert decision.action is Action.HOLD


# ---------------------------------------------------------------------------
# on_stt_interim — score-based fallback
# ---------------------------------------------------------------------------


def test_high_score_cancels_even_on_backchannel_text() -> None:
    """If text is backchannel but EOT score is high (rare), the score
    path still triggers cancel — the strong-EOT-score path is a fallback
    for the slow-STT case where text might be misleading."""
    d = InterruptDecider(early_cancel_score_threshold=0.7)
    decision = d.on_stt_interim("嗯", score=0.85)
    assert decision.action is Action.CANCEL
    assert "eot_score_high" in decision.reason


def test_low_score_rollback_only_when_positive() -> None:
    """Score in (0, resume_thr] → ROLLBACK with drain (drop_buffered=False)."""
    d = InterruptDecider(early_resume_score_threshold=0.2)
    decision = d.on_stt_interim("嗯", score=0.1)
    assert decision.action is Action.ROLLBACK
    assert decision.rollback_drop_buffered is False
    assert "eot_score_low" in decision.reason


def test_zero_score_means_no_signal_yet_holds() -> None:
    """Score exactly 0.0 = 'text too short for ONNX' → HOLD, not rollback."""
    d = InterruptDecider(early_resume_score_threshold=0.2)
    decision = d.on_stt_interim("嗯", score=0.0)
    assert decision.action is Action.HOLD


def test_mid_band_score_holds() -> None:
    """Score between resume_thr and cancel_thr → HOLD."""
    d = InterruptDecider(
        early_cancel_score_threshold=0.7,
        early_resume_score_threshold=0.2,
    )
    decision = d.on_stt_interim("嗯", score=0.5)
    assert decision.action is Action.HOLD
    assert "eot_score_mid" in decision.reason


# ---------------------------------------------------------------------------
# on_decision_deadline — VAD-active branch
# ---------------------------------------------------------------------------


def test_deadline_vad_active_cancels_trust_vad() -> None:
    d = InterruptDecider()
    decision = d.on_decision_deadline(vad_still_active=True)
    assert decision.action is Action.CANCEL
    assert "trust_vad" in decision.reason


def test_deadline_vad_idle_rollback_drop_buffered() -> None:
    """VAD idle at deadline → rollback WITH drop_buffered=True (buffer
    is stale after ≥500 ms suspend)."""
    d = InterruptDecider()
    decision = d.on_decision_deadline(vad_still_active=False)
    assert decision.action is Action.ROLLBACK
    assert decision.rollback_drop_buffered is True


# ---------------------------------------------------------------------------
# on_user_silent — fast rollback path
# ---------------------------------------------------------------------------


def test_user_silent_fast_rollback_drains() -> None:
    """User-silent transition <300 ms → ROLLBACK with drain (buffer fresh)."""
    d = InterruptDecider()
    decision = d.on_user_silent()
    assert decision.action is Action.ROLLBACK
    assert decision.rollback_drop_buffered is False


# ---------------------------------------------------------------------------
# Decision frozen-ness
# ---------------------------------------------------------------------------


def test_decision_is_frozen() -> None:
    """Decision is intentionally immutable so accidental mutation doesn't
    corrupt the policy."""
    d = Decision(action=Action.HOLD, reason="x")
    with pytest.raises((AttributeError, Exception)):
        d.reason = "y"  # type: ignore[misc]
