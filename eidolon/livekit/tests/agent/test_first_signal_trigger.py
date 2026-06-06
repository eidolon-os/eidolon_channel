"""G18a (2026-05-18): first-signal cancel path + 500ms decision budget.

Unit tests for the two new behaviours in StreamingPipeline:

  1. In the duck-active branch of ``_run_eot_check``, a non-backchannel
     STT INTERIM of ≥ ``interrupt_min_interim_chars`` chars triggers an
     immediate cancel — bypassing the slower EOT score path.

  2. In ``_duck_suspend_timeout_fallback``, when the timeout fires while
     VAD is still active, the resolution depends on transcript evidence:
     without transcript it holds the suspended output, with transcript it
     confirms the interrupt.

Together these keep real interrupts responsive while avoiding transcript-free
timeout cancels that are too aggressive in WebSocket streaming STT flows.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers — build a barely-initialised StreamingPipeline for the unit paths
# we want to exercise. Full pipeline instantiation requires LiveKit Room
# setup; we bypass __init__ and patch in just the surface we touch.
# ---------------------------------------------------------------------------


def _make_pipeline(*, vad_user_state: str = "listening"):
    """Build a stub pipeline with a SUSPENDED duck_mixer."""
    from eidolon.livekit.agent.streaming import StreamingPipeline
    from eidolon.livekit.plugins.eot.config import EidolonEOTConfig

    pipeline = StreamingPipeline.__new__(StreamingPipeline)

    # Configuration carrier; G18a reads these.
    cfg = EidolonEOTConfig()
    eot_model = MagicMock()
    eot_model._config = cfg
    # is_strong_interrupt_intent → return False so we exercise the score path
    eot_model._turn_end_policy.is_strong_interrupt_intent.return_value = False
    # should_interrupt + current_eot_score are read in the score branch; the
    # first-signal path should fire BEFORE these matter.
    eot_model.should_interrupt.return_value = False
    eot_model.current_eot_score = 0.0
    pipeline._get_eot_model = MagicMock(return_value=eot_model)

    # Mock duck mixer
    mixer = MagicMock()
    mixer.state = "SUSPENDED"
    mixer.buffered_frames = 5
    mixer.buffered_sec = 0.05
    pipeline._duck_mixer = mixer

    pipeline._session = MagicMock()
    pipeline._session.user_state = vad_user_state
    pipeline._duck_suspend_start = time.monotonic() - 0.1
    pipeline._duck_timeout_task = None
    pipeline._soft_interrupt_active = False
    pipeline._soft_interrupt_timer = None
    pipeline._last_unduck_time = 0.0
    pipeline._user_speaking_start_time = None

    pipeline._callbacks = MagicMock()
    pipeline._snapshot_interrupted_context = MagicMock()
    pipeline._interrupt_current_turn = MagicMock()

    return pipeline


# ---------------------------------------------------------------------------
# 1. first-signal cancel in _run_eot_check duck-active path
# ---------------------------------------------------------------------------


def test_first_signal_cancel_on_substantive_interim() -> None:
    """Substantive CJK INTERIM with enough evidence → immediate cancel."""
    pipeline = _make_pipeline()

    pipeline._run_eot_check("我不相信你", is_final=False)

    # Pipeline should have entered cancel path:
    pipeline._snapshot_interrupted_context.assert_called_once()
    pipeline._duck_mixer.cancel.assert_called_once()
    pipeline._interrupt_current_turn.assert_called_once()


def test_first_signal_holds_short_latin_artifact() -> None:
    """Short latin-only ASR artifacts should not cancel the agent turn."""
    pipeline = _make_pipeline()

    pipeline._run_eot_check("If", is_final=False)

    pipeline._duck_mixer.cancel.assert_not_called()
    pipeline._interrupt_current_turn.assert_not_called()


def test_first_signal_skips_backchannel() -> None:
    """Backchannel ("嗯") must NOT trigger first-signal cancel — caller
    should wait for the next INTERIM."""
    pipeline = _make_pipeline()

    pipeline._run_eot_check("嗯", is_final=False)

    pipeline._duck_mixer.cancel.assert_not_called()
    pipeline._interrupt_current_turn.assert_not_called()


def test_first_signal_skips_compound_backchannel() -> None:
    """Compound backchannel ("嗯嗯好") also rejected."""
    pipeline = _make_pipeline()

    pipeline._run_eot_check("嗯嗯", is_final=False)

    pipeline._duck_mixer.cancel.assert_not_called()


def test_first_signal_skips_single_char() -> None:
    """Single char INTERIM (likely cough / partial) doesn't trigger cancel
    even if it's not in the backchannel list."""
    pipeline = _make_pipeline()

    pipeline._run_eot_check("你", is_final=False)  # 1 char < min 2

    pipeline._duck_mixer.cancel.assert_not_called()


def test_first_signal_respects_turn_policy_min_chars() -> None:
    """``turn_policy.interrupt.min_interim_chars`` raises the threshold."""
    from dataclasses import replace

    from eidolon.livekit.agent.turn_policy import TurnPolicyRuntime
    from eidolon.livekit.common.config import InterruptPolicyConfig, TurnPolicyConfig

    pipeline = _make_pipeline()
    policy = replace(
        TurnPolicyConfig(),
        interrupt=replace(InterruptPolicyConfig(), min_interim_chars=5),
    )
    pipeline._turn_policy = policy
    pipeline._turn_runtime = TurnPolicyRuntime(policy)

    pipeline._run_eot_check("你好世", is_final=False)  # 3 chars < new 5

    pipeline._duck_mixer.cancel.assert_not_called()


def test_first_signal_strips_punctuation() -> None:
    """Punctuation should be stripped before the length check (Bailian
    sometimes attaches "。"/"," to short INTERIMs)."""
    pipeline = _make_pipeline()

    pipeline._run_eot_check("嗯。", is_final=False)  # stripped → "嗯" backchannel

    pipeline._duck_mixer.cancel.assert_not_called()


def test_fallback_semantic_correction_cancels_after_late_interim() -> None:
    """Late STT text after VAD-end still drives semantic turn control."""
    pipeline = _make_pipeline(vad_user_state="listening")
    pipeline._duck_mixer.state = "NORMAL"

    pipeline._run_eot_check("我刚才说错了", is_final=False)

    pipeline._duck_mixer.cancel.assert_called_once()
    pipeline._interrupt_current_turn.assert_called_once()


# ---------------------------------------------------------------------------
# 2. timeout-fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_with_vad_still_active_without_transcript_holds() -> None:
    """At the decision deadline, VAD alone is not enough evidence to cancel."""
    pipeline = _make_pipeline(vad_user_state="speaking")
    pipeline._cancel_duck_timeout = MagicMock()

    await pipeline._duck_suspend_timeout_fallback(0.01)

    pipeline._duck_mixer.cancel.assert_not_called()
    pipeline._duck_mixer.unduck.assert_not_called()
    pipeline._interrupt_current_turn.assert_not_called()


@pytest.mark.asyncio
async def test_timeout_with_vad_still_active_and_transcript_cancels() -> None:
    """If streaming STT has produced text by the deadline, timeout can confirm."""
    pipeline = _make_pipeline(vad_user_state="speaking")
    pipeline._latest_asr_text = "等一下"
    pipeline._cancel_duck_timeout = MagicMock()

    await pipeline._duck_suspend_timeout_fallback(0.01)

    pipeline._duck_mixer.cancel.assert_called_once()
    pipeline._duck_mixer.unduck.assert_not_called()
    pipeline._interrupt_current_turn.assert_called_once()


@pytest.mark.asyncio
async def test_timeout_with_vad_idle_unducks_drop_buffered() -> None:
    """G18a + G17a: if VAD is no longer active at the deadline (user
    really did go silent), unduck with drop_buffered=True."""
    pipeline = _make_pipeline(vad_user_state="listening")
    pipeline._cancel_duck_timeout = MagicMock()

    await pipeline._duck_suspend_timeout_fallback(0.01)

    pipeline._duck_mixer.unduck.assert_called_once_with(drop_buffered=True)
    pipeline._duck_mixer.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_timeout_noop_if_already_resolved() -> None:
    """If the duck was already resolved (state != SUSPENDED) before the
    timeout fires, the fallback should be a no-op."""
    pipeline = _make_pipeline(vad_user_state="speaking")
    pipeline._duck_mixer.state = "NORMAL"  # already resolved
    pipeline._cancel_duck_timeout = MagicMock()

    await pipeline._duck_suspend_timeout_fallback(0.01)

    pipeline._duck_mixer.cancel.assert_not_called()
    pipeline._duck_mixer.unduck.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Config plumbing — turn_policy knobs land correctly
# ---------------------------------------------------------------------------


def test_turn_policy_decision_ms_maps_to_eot_timeout() -> None:
    """``turn_policy.interrupt.decision_timeout_ms`` feeds the EOT timeout."""
    from dataclasses import replace

    from eidolon.livekit.agent.turn_policy import eot_kwargs_from_turn_policy
    from eidolon.livekit.common.config import InterruptPolicyConfig, TurnPolicyConfig

    policy = replace(
        TurnPolicyConfig(),
        interrupt=replace(InterruptPolicyConfig(), decision_timeout_ms=750),
    )

    assert eot_kwargs_from_turn_policy(policy)[
        "duck_suspend_timeout_sec"
    ] == pytest.approx(0.75)


def test_fast_profile_has_shorter_decision_budget() -> None:
    """The e2e-like profile is more eager than the balanced default."""
    from eidolon.livekit.common.config.profiles import profile_defaults

    balanced = profile_defaults("balanced_semantic")
    fast = profile_defaults("fast_e2e_like")

    assert balanced.interrupt.decision_timeout_ms == 500
    assert fast.interrupt.decision_timeout_ms < balanced.interrupt.decision_timeout_ms


def test_default_decision_budget_is_500ms() -> None:
    """Sanity: plugin default still lands at the 0.5s G18a target."""
    from eidolon.livekit.plugins.eot.config import EidolonEOTConfig

    cfg = EidolonEOTConfig()
    assert cfg.duck_suspend_timeout_sec == pytest.approx(0.5)


def test_default_min_chars_is_2() -> None:
    from eidolon.livekit.plugins.eot.config import EidolonEOTConfig

    cfg = EidolonEOTConfig()
    assert cfg.interrupt_min_interim_chars == 2


def test_default_vad_confidence_gate_disabled() -> None:
    """G18a: default ``min_avg_vad_confidence`` is now 0.0 (gate off)."""
    from eidolon.livekit.plugins.eot.config import EidolonEOTConfig

    cfg = EidolonEOTConfig()
    assert cfg.min_avg_vad_confidence == 0.0
