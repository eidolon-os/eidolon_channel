"""G18a+ (2026-05-18): semantic-tiered first signal + decision budget.

Unit tests for the two new behaviours in StreamingPipeline:

  1. In the duck-active semantic interrupt branch, an
     STT INTERIM of ≥ ``interrupt_min_interim_chars`` chars is enough
     transcript evidence to keep evaluating, but ordinary text still needs
     EOT semantic confidence before canceling.

  2. In the duck suspend-window deadline handler, when the timeout fires while
     VAD is still active, the resolution depends on transcript evidence
     and EOT semantic score: without transcript or semantic confidence it
     holds the suspended output; high-confidence transcript confirms.

Together these keep real interrupts responsive while avoiding transcript-free
timeout cancels and early-interim cuts that are too aggressive in WebSocket
streaming STT flows.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from eidolon.livekit.agent.output.ducking import OutputDuckingController


# ---------------------------------------------------------------------------
# Helpers — build a barely-initialised StreamingPipeline for the unit paths
# we want to exercise. Full pipeline instantiation requires LiveKit Room
# setup; we bypass __init__ and patch in just the surface we touch.
# ---------------------------------------------------------------------------


def _make_pipeline(*, vad_user_state: str = "listening", eot_score: float = 0.0):
    """Build a stub pipeline with a SUSPENDED duck_mixer."""
    from eidolon.livekit.agent.full_duplex import StreamingPipeline
    from eidolon.livekit.plugins.eot.config import EidolonEOTConfig

    pipeline = StreamingPipeline.__new__(StreamingPipeline)

    # Configuration carrier; G18a reads these.
    cfg = EidolonEOTConfig()
    eot_model = MagicMock()
    eot_model._config = cfg
    # should_interrupt + current_eot_score are read in the semantic-tiered path.
    eot_model.should_interrupt.return_value = False
    eot_model.current_eot_score = eot_score
    pipeline._get_eot_model = MagicMock(return_value=eot_model)

    # Mock duck mixer
    mixer = MagicMock()
    mixer.state = "SUSPENDED"
    mixer.buffered_frames = 5
    mixer.buffered_sec = 0.05
    pipeline._ducking = OutputDuckingController()
    pipeline._ducking.mixer = mixer

    pipeline._session = MagicMock()
    pipeline._session.user_state = vad_user_state
    pipeline._ducking.suspend_start = time.monotonic() - 0.1
    pipeline._ducking.timeout_task = None
    pipeline._ducking.last_unduck_time = 0.0
    pipeline._user_speaking_start_time = None

    pipeline._callbacks = MagicMock()
    pipeline._context_ledger = MagicMock()

    return pipeline


def _run_semantic_check(pipeline, text: str, *, is_final: bool = False) -> None:
    pipeline._ensure_runtime_defaults()
    pipeline._semantic_interrupts.run(text, is_final=is_final)


async def _run_duck_deadline(pipeline, timeout_sec: float) -> None:
    pipeline._ensure_runtime_defaults()
    await pipeline._duck_deadline.run(timeout_sec)


# ---------------------------------------------------------------------------
# 1. semantic-tiered first signal in duck-active path
# ---------------------------------------------------------------------------


def test_first_signal_holds_substantive_interim_without_semantic_score() -> None:
    """Substantive CJK INTERIM waits for EOT semantics instead of raw length."""
    pipeline = _make_pipeline()

    _run_semantic_check(pipeline, "我不相信你", is_final=False)

    pipeline._context_ledger.snapshot.assert_not_called()
    pipeline._ducking.mixer.cancel.assert_not_called()


def test_first_signal_cancel_on_high_semantic_score() -> None:
    """High EOT semantic confidence keeps the quick cancel tier."""
    pipeline = _make_pipeline(eot_score=0.82)

    _run_semantic_check(pipeline, "我不相信你", is_final=False)

    pipeline._ducking.mixer.cancel.assert_called_once()


def test_first_signal_holds_short_latin_artifact() -> None:
    """Short latin-only ASR artifacts should not cancel the agent turn."""
    pipeline = _make_pipeline()

    _run_semantic_check(pipeline, "If", is_final=False)

    pipeline._ducking.mixer.cancel.assert_not_called()


def test_first_signal_holds_short_latin_hard_stop_artifact() -> None:
    """A short latin hard-stop lexicon hit is still provisional evidence."""
    pipeline = _make_pipeline()

    _run_semantic_check(pipeline, "stop", is_final=False)

    pipeline._ducking.mixer.cancel.assert_not_called()


def test_first_signal_skips_backchannel() -> None:
    """Backchannel ("嗯") must NOT trigger first-signal cancel — caller
    should wait for the next INTERIM."""
    pipeline = _make_pipeline()

    _run_semantic_check(pipeline, "嗯", is_final=False)

    pipeline._ducking.mixer.cancel.assert_not_called()


def test_first_signal_skips_compound_backchannel() -> None:
    """Compound backchannel ("嗯嗯好") also rejected."""
    pipeline = _make_pipeline()

    _run_semantic_check(pipeline, "嗯嗯", is_final=False)

    pipeline._ducking.mixer.cancel.assert_not_called()


def test_first_signal_skips_single_char() -> None:
    """Single char INTERIM (likely cough / partial) doesn't trigger cancel
    even if it's not in the backchannel list."""
    pipeline = _make_pipeline()

    _run_semantic_check(pipeline, "你", is_final=False)  # 1 char < min 2

    pipeline._ducking.mixer.cancel.assert_not_called()


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

    _run_semantic_check(pipeline, "你好世", is_final=False)  # 3 chars < new 5

    pipeline._ducking.mixer.cancel.assert_not_called()


def test_first_signal_strips_punctuation() -> None:
    """Punctuation should be stripped before the length check (Bailian
    sometimes attaches "。"/"," to short INTERIMs)."""
    pipeline = _make_pipeline()

    _run_semantic_check(pipeline, "嗯。", is_final=False)  # stripped → "嗯"

    pipeline._ducking.mixer.cancel.assert_not_called()


def test_fallback_semantic_correction_waits_without_semantic_score() -> None:
    """Late correction text after VAD-end waits for semantic confidence."""
    pipeline = _make_pipeline(vad_user_state="listening")
    pipeline._ducking.mixer.state = "NORMAL"

    _run_semantic_check(pipeline, "我刚才说错了", is_final=False)

    pipeline._ducking.mixer.cancel.assert_not_called()


# ---------------------------------------------------------------------------
# 2. timeout-fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_with_vad_still_active_without_transcript_holds() -> None:
    """At the decision deadline, VAD alone is not enough evidence to cancel."""
    pipeline = _make_pipeline(vad_user_state="speaking")

    with patch("eidolon.livekit.agent.full_duplex.pipeline.asyncio.create_task") as create_task:
        await _run_duck_deadline(pipeline, 0.01)

    pipeline._ducking.mixer.cancel.assert_not_called()
    pipeline._ducking.mixer.unduck.assert_not_called()
    create_task.assert_called_once()
    create_task.call_args.args[0].close()


@pytest.mark.asyncio
async def test_timeout_with_vad_still_active_and_transcript_cancels() -> None:
    """At deadline, transcript + high semantic score can confirm interrupt."""
    pipeline = _make_pipeline(vad_user_state="speaking", eot_score=0.82)
    pipeline._latest_asr_text = "我不相信你"

    await _run_duck_deadline(pipeline, 0.01)

    pipeline._ducking.mixer.cancel.assert_called_once()
    pipeline._ducking.mixer.unduck.assert_not_called()


@pytest.mark.asyncio
async def test_timeout_with_low_score_transcript_holds() -> None:
    """A partial transcript at the deadline waits for the semantic tier."""
    pipeline = _make_pipeline(vad_user_state="speaking", eot_score=0.0)
    pipeline._latest_asr_text = "啊那你"

    with patch("eidolon.livekit.agent.full_duplex.pipeline.asyncio.create_task") as create_task:
        await _run_duck_deadline(pipeline, 0.01)

    pipeline._ducking.mixer.cancel.assert_not_called()
    pipeline._ducking.mixer.unduck.assert_not_called()
    create_task.assert_called_once()
    create_task.call_args.args[0].close()


@pytest.mark.asyncio
async def test_timeout_hold_rolls_back_after_max_suspend_budget() -> None:
    """A HOLD deadline must not leave the duck mixer suspended forever."""
    pipeline = _make_pipeline(vad_user_state="speaking", eot_score=0.0)
    pipeline._latest_asr_text = "啊那你"
    pipeline._ducking.suspend_start = (
        time.monotonic()
        - pipeline._get_eot_model()._config.duck_buffer_max_sec
        - 0.1
    )

    await _run_duck_deadline(pipeline, 0.01)

    pipeline._ducking.mixer.unduck.assert_called_once_with(drop_buffered=True)
    pipeline._ducking.mixer.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_timeout_with_vad_idle_unducks_drop_buffered() -> None:
    """G18a + G17a: if VAD is no longer active at the deadline (user
    really did go silent), unduck with drop_buffered=True."""
    pipeline = _make_pipeline(vad_user_state="listening")

    await _run_duck_deadline(pipeline, 0.01)

    pipeline._ducking.mixer.unduck.assert_called_once_with(drop_buffered=True)
    pipeline._ducking.mixer.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_timeout_noop_if_already_resolved() -> None:
    """If the duck was already resolved (state != SUSPENDED) before the
    timeout fires, the fallback should be a no-op."""
    pipeline = _make_pipeline(vad_user_state="speaking")
    pipeline._ducking.mixer.state = "NORMAL"  # already resolved

    await _run_duck_deadline(pipeline, 0.01)

    pipeline._ducking.mixer.cancel.assert_not_called()
    pipeline._ducking.mixer.unduck.assert_not_called()


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


def test_turn_policy_filler_maps_to_eot_config() -> None:
    """Latency-mask filler envelope is runtime config, not output constants."""
    from dataclasses import replace

    from eidolon.livekit.agent.turn_policy import eot_kwargs_from_turn_policy
    from eidolon.livekit.common.config import FillerPolicyConfig, TurnPolicyConfig

    policy = replace(
        TurnPolicyConfig(),
        filler=FillerPolicyConfig(
            enabled=True,
            phrases=("嗯...", "收到..."),
            fade_in_ms=45,
            fade_out_ms=95,
            silence_lead_in_ms=180,
        ),
    )

    kwargs = eot_kwargs_from_turn_policy(policy)

    assert kwargs["filler_enabled"] is True
    assert kwargs["filler_phrases"] == ("嗯...", "收到...")
    assert kwargs["filler_fade_in_ms"] == 45
    assert kwargs["filler_fade_out_ms"] == 95
    assert kwargs["filler_silence_lead_in_ms"] == 180


def test_turn_policy_suspended_passthrough_maps_to_eot_config() -> None:
    """Audible-hold passthrough is runtime config and defaults off."""
    from dataclasses import replace

    from eidolon.livekit.agent.turn_policy import eot_kwargs_from_turn_policy
    from eidolon.livekit.common.config import DuckingPolicyConfig, TurnPolicyConfig

    policy = replace(
        TurnPolicyConfig(),
        ducking=DuckingPolicyConfig(
            suspended_passthrough_enabled=True,
            suspended_passthrough_volume=0.2,
        ),
    )

    kwargs = eot_kwargs_from_turn_policy(policy)

    assert kwargs["duck_suspended_passthrough_enabled"] is True
    assert kwargs["duck_suspended_passthrough_volume"] == 0.2


def test_fast_profile_has_shorter_decision_budget() -> None:
    """The e2e-like profile is more eager than the balanced default."""
    from eidolon.livekit.common.config.profiles import profile_defaults

    balanced = profile_defaults("balanced_semantic")
    fast = profile_defaults("fast_e2e_like")

    assert balanced.interrupt.decision_timeout_ms == 450
    assert fast.interrupt.decision_timeout_ms < balanced.interrupt.decision_timeout_ms


def test_default_decision_budget_keeps_margin_under_500ms_sla() -> None:
    """Sanity: plugin default keeps scheduler margin under the 0.5s G18a target."""
    from eidolon.livekit.plugins.eot.config import EidolonEOTConfig

    cfg = EidolonEOTConfig()
    assert cfg.duck_suspend_timeout_sec == pytest.approx(0.45)
    assert cfg.duck_suspended_passthrough_enabled is False


def test_default_min_chars_is_2() -> None:
    from eidolon.livekit.plugins.eot.config import EidolonEOTConfig

    cfg = EidolonEOTConfig()
    assert cfg.interrupt_min_interim_chars == 2
