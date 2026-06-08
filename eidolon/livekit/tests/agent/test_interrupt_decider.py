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


def test_hard_stop_prefix_cancels_on_hot_path() -> None:
    d = InterruptDecider(min_interim_chars=2)

    decision = d.on_stt_interim("别说", score=0.0)

    assert decision.action is Action.CANCEL
    assert decision.intent is not None
    assert decision.intent.value == "hard_stop"
    assert decision.intent_source == "lexicon_prefix"


def test_ambiguous_hard_stop_fragment_does_not_cancel() -> None:
    d = InterruptDecider(min_interim_chars=2)

    decision = d.on_stt_interim("不要", score=0.0)

    assert decision.action is not Action.CANCEL


# ---------------------------------------------------------------------------
# on_stt_interim — semantic-tiered first signal
# ---------------------------------------------------------------------------


def test_substantive_interim_holds_until_semantic_score() -> None:
    """Substantive CJK interim is evidence, not enough by itself to cancel."""
    d = InterruptDecider(min_interim_chars=2)

    decision = d.on_stt_interim("我不相信", score=0.0)

    assert decision.action is Action.HOLD
    assert "semantic_score_wait" in decision.reason


def test_substantive_interim_cancels_with_high_semantic_score() -> None:
    """Once EOT semantic confidence is high, the same interim can cancel."""
    d = InterruptDecider(min_interim_chars=2, early_cancel_score_threshold=0.7)

    decision = d.on_stt_interim("我不相信", score=0.82)

    assert decision.action is Action.CANCEL
    assert "eot_score_high" in decision.reason


def test_dogfood_partial_prefix_holds_until_intent_forms() -> None:
    """Real room: '啊那你' is too early to cancel before the intent appears."""
    d = InterruptDecider(min_interim_chars=2)

    decision = d.on_stt_interim("啊那你", score=0.0)

    assert decision.action is Action.HOLD
    assert "semantic_score_wait" in decision.reason


def test_dogfood_low_score_long_interim_holds_instead_of_rollback() -> None:
    """Real room: low-score long interim should wait, not cancel or unduck."""
    d = InterruptDecider(min_interim_chars=2, early_resume_score_threshold=0.2)

    decision = d.on_stt_interim("嗯我给你弄了", score=0.07)

    assert decision.action is Action.HOLD
    assert "semantic_score_wait" in decision.reason


def test_short_latin_artifact_holds_instead_of_cancel() -> None:
    """Short latin-only ASR artifacts like 'If' must not hard-cancel audio."""
    d = InterruptDecider(min_interim_chars=2)

    decision = d.on_stt_interim("If", score=0.0)

    assert decision.action is Action.HOLD
    assert "short_latin_artifact" in decision.reason


def test_final_short_latin_transcript_still_waits_for_semantics() -> None:
    """Provider finals prove ASR stability, not interrupt intent."""
    d = InterruptDecider(min_interim_chars=2)

    decision = d.on_stt_interim("If", score=0.0, is_final=True)

    assert decision.action is Action.HOLD
    assert "semantic_score_wait" in decision.reason
    assert "final=true" in decision.reason


def test_final_substantive_transcript_still_waits_for_semantics() -> None:
    """Ordinary final text still needs EOT or explicit intent to cancel."""
    d = InterruptDecider(min_interim_chars=2)

    decision = d.on_stt_interim("我想问一下", score=0.0, is_final=True)

    assert decision.action is Action.HOLD
    assert "semantic_score_wait" in decision.reason
    assert "final=true" in decision.reason


def test_first_signal_holds_single_char_backchannel() -> None:
    """Single-char backchannel fragments wait for more speech evidence."""
    d = InterruptDecider(min_interim_chars=2)
    decision = d.on_stt_interim("嗯", score=0.0)
    assert decision.action is Action.HOLD


def test_first_signal_skips_compound_backchannel() -> None:
    """Compound backchannel like '嗯嗯' is in the set → ROLLBACK."""
    d = InterruptDecider(min_interim_chars=2)
    decision = d.on_stt_interim("嗯嗯", score=0.0)
    assert decision.action is Action.ROLLBACK


def test_first_signal_below_min_chars_is_hold() -> None:
    """Single-char non-backchannel ('你') under min_chars → HOLD."""
    d = InterruptDecider(min_interim_chars=2)
    decision = d.on_stt_interim("你", score=0.0)
    assert decision.action is Action.HOLD


def test_semantic_prefix_holds_for_more_interim() -> None:
    d = InterruptDecider(min_interim_chars=2)
    decision = d.on_stt_interim("换个", score=0.0)
    assert decision.action is Action.HOLD
    assert "semantic_prefix" in decision.reason


def test_topic_switch_confused_prefix_holds_for_more_interim() -> None:
    d = InterruptDecider(min_interim_chars=2)
    decision = d.on_stt_interim("半个", score=0.0)
    assert decision.action is Action.HOLD
    assert "semantic_prefix" in decision.reason


def test_long_topic_switch_prefix_becomes_stable_tier1_candidate() -> None:
    d = InterruptDecider(min_interim_chars=2)
    decision = d.on_stt_interim("换个话", score=0.0)
    assert decision.action is Action.CANCEL
    assert decision.intent.value == "topic_switch"
    assert decision.intent_source == "lexicon_prefix"
    assert decision.topic_switch_hint is True


def test_correction_confused_prefix_holds_for_more_interim() -> None:
    d = InterruptDecider(min_interim_chars=2)
    decision = d.on_stt_interim("是我", score=0.0)
    assert decision.action is Action.HOLD
    assert "semantic_prefix" in decision.reason


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


def test_compound_backchannel_rolls_back_even_with_high_score() -> None:
    """Explicit backchannel/noise classification wins over EOT score."""
    d = InterruptDecider(early_cancel_score_threshold=0.7)
    decision = d.on_stt_interim("嗯嗯", score=0.85)
    assert decision.action is Action.ROLLBACK
    assert "intent:" in decision.reason


def test_low_score_rollback_only_when_positive() -> None:
    """Score in (0, resume_thr] → ROLLBACK with drain (drop_buffered=False)."""
    d = InterruptDecider(early_resume_score_threshold=0.2)
    decision = d.on_stt_interim("你", score=0.1)
    assert decision.action is Action.ROLLBACK
    assert decision.rollback_drop_buffered is False
    assert "eot_score_low" in decision.reason


def test_zero_score_means_no_signal_yet_holds() -> None:
    """Single-char non-backchannel with score 0.0 holds."""
    d = InterruptDecider(early_resume_score_threshold=0.2)
    decision = d.on_stt_interim("你", score=0.0)
    assert decision.action is Action.HOLD


def test_mid_band_score_holds() -> None:
    """Score between resume_thr and cancel_thr → HOLD."""
    d = InterruptDecider(
        early_cancel_score_threshold=0.7,
        early_resume_score_threshold=0.2,
    )
    decision = d.on_stt_interim("你", score=0.5)
    assert decision.action is Action.HOLD
    assert "eot_score_mid" in decision.reason


def test_topic_switch_cancels_with_hint() -> None:
    d = InterruptDecider()
    decision = d.on_stt_interim("换个话题吧", score=0.0)
    assert decision.action is Action.CANCEL
    assert decision.topic_switch_hint is True


def test_topic_switch_confused_transcript_cancels_with_hint() -> None:
    d = InterruptDecider()
    decision = d.on_stt_interim("半个话题吧", score=0.0)
    assert decision.action is Action.CANCEL
    assert decision.topic_switch_hint is True


def test_correction_cancels_with_hint() -> None:
    d = InterruptDecider()
    decision = d.on_stt_interim("不是，我的意思是", score=0.0)
    assert decision.action is Action.CANCEL
    assert decision.correction_hint is True


def test_correction_confused_transcript_cancels_with_hint() -> None:
    d = InterruptDecider()
    decision = d.on_stt_interim("是我刚", score=0.0)
    assert decision.action is Action.CANCEL
    assert decision.correction_hint is True


# ---------------------------------------------------------------------------
# on_decision_deadline — timeout branches
# ---------------------------------------------------------------------------


def test_deadline_vad_active_without_transcript_holds() -> None:
    d = InterruptDecider()
    decision = d.on_decision_deadline(vad_still_active=True)
    assert decision.action is Action.HOLD
    assert "wait_for_transcript" in decision.reason


def test_deadline_vad_active_with_transcript_cancels() -> None:
    d = InterruptDecider()
    decision = d.on_decision_deadline(
        vad_still_active=True,
        has_transcript=True,
        transcript="等我查一下",
        eot_score=0.8,
    )
    assert decision.action is Action.CANCEL
    assert "trust_vad" in decision.reason


def test_deadline_vad_active_waits_for_semantic_score() -> None:
    d = InterruptDecider()
    decision = d.on_decision_deadline(
        vad_still_active=True,
        has_transcript=True,
        transcript="啊那你",
        eot_score=0.0,
    )
    assert decision.action is Action.HOLD
    assert "wait_for_semantic_score" in decision.reason


def test_deadline_vad_active_with_noise_fragment_holds() -> None:
    d = InterruptDecider()
    decision = d.on_decision_deadline(
        vad_still_active=True,
        has_transcript=True,
        transcript="啊",
    )
    assert decision.action is Action.HOLD


def test_deadline_vad_active_with_single_char_transcript_holds() -> None:
    d = InterruptDecider()
    decision = d.on_decision_deadline(
        vad_still_active=True,
        has_transcript=True,
        transcript="听",
    )
    assert decision.action is Action.HOLD
    assert "wait_for_more_transcript" in decision.reason


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


def test_user_silent_backchannel_keeps_intent() -> None:
    d = InterruptDecider()
    decision = d.on_user_silent("好")
    assert decision.action is Action.ROLLBACK
    assert decision.intent is not None
    assert decision.intent.value == "backchannel"


# ---------------------------------------------------------------------------
# Decision frozen-ness
# ---------------------------------------------------------------------------


def test_decision_is_frozen() -> None:
    """Decision is intentionally immutable so accidental mutation doesn't
    corrupt the policy."""
    d = Decision(action=Action.HOLD, reason="x")
    with pytest.raises((AttributeError, Exception)):
        d.reason = "y"  # type: ignore[misc]
