"""G18a (2026-05-18): first-signal cancel path + 500ms decision budget.

Unit tests for the two new behaviours in StreamingPipeline:

  1. In the duck-active branch of ``_run_eot_check``, a non-backchannel
     STT INTERIM of ≥ ``interrupt_min_interim_chars`` chars triggers an
     immediate cancel — bypassing the slower EOT score path.

  2. In ``_duck_suspend_timeout_fallback``, when the timeout fires while
     VAD is still active, the resolution is CANCEL (trust VAD) rather
     than unduck. The user is clearly still speaking; brief unduck would
     yo-yo the audio.

Together these limit worst-case interrupt latency to ``duck_suspend_timeout_sec``
(now 0.5s default) without compromising real-interrupt correctness.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
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
    """Non-backchannel INTERIM with len ≥ min_chars → immediate cancel."""
    pipeline = _make_pipeline()

    pipeline._run_eot_check("我不相信你", is_final=False)

    # Pipeline should have entered cancel path:
    pipeline._snapshot_interrupted_context.assert_called_once()
    pipeline._duck_mixer.cancel.assert_called_once()
    pipeline._interrupt_current_turn.assert_called_once()


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


def test_first_signal_respects_min_chars_env(monkeypatch) -> None:
    """``EIDOLON_INTERRUPT_MIN_INTERIM_CHARS`` env raises the threshold."""
    monkeypatch.setenv("EIDOLON_INTERRUPT_MIN_INTERIM_CHARS", "5")
    pipeline = _make_pipeline()
    # Override cfg to pick up the new env (the fixture built it before
    # monkeypatch ran). Build a fresh cfg now.
    from eidolon.livekit.plugins.eot.config import EidolonEOTConfig
    pipeline._get_eot_model().return_value._config = EidolonEOTConfig()
    # re-wire just the cfg via the existing mock chain
    eot_model = pipeline._get_eot_model.return_value
    eot_model._config = EidolonEOTConfig()

    pipeline._run_eot_check("你好世", is_final=False)  # 3 chars < new 5

    pipeline._duck_mixer.cancel.assert_not_called()


def test_first_signal_strips_punctuation() -> None:
    """Punctuation should be stripped before the length check (Bailian
    sometimes attaches "。"/"," to short INTERIMs)."""
    pipeline = _make_pipeline()

    pipeline._run_eot_check("嗯。", is_final=False)  # stripped → "嗯" backchannel

    pipeline._duck_mixer.cancel.assert_not_called()


# ---------------------------------------------------------------------------
# 2. timeout-fallback: VAD-still-active → cancel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_with_vad_still_active_cancels() -> None:
    """G18a: at the 0.5s deadline, if user is still speaking VAD-wise,
    trust VAD and cancel (rather than unducking and praying)."""
    pipeline = _make_pipeline(vad_user_state="speaking")
    pipeline._cancel_duck_timeout = MagicMock()

    await pipeline._duck_suspend_timeout_fallback(0.01)

    # Should have called cancel path, not unduck
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
# 3. Config plumbing — env knobs land correctly
# ---------------------------------------------------------------------------


def test_decision_ms_env_overrides_default(monkeypatch) -> None:
    """``EIDOLON_INTERRUPT_DECISION_MS=750`` → 0.75s timeout."""
    monkeypatch.setenv("EIDOLON_INTERRUPT_DECISION_MS", "750")
    from eidolon.livekit.plugins.eot.config import EidolonEOTConfig
    cfg = EidolonEOTConfig()
    assert cfg.duck_suspend_timeout_sec == pytest.approx(0.75)


def test_decision_ms_legacy_env_still_works(monkeypatch) -> None:
    """Legacy ``EIDOLON_DUCK_SUSPEND_TIMEOUT_SEC=0.6`` still honored."""
    monkeypatch.delenv("EIDOLON_INTERRUPT_DECISION_MS", raising=False)
    monkeypatch.setenv("EIDOLON_DUCK_SUSPEND_TIMEOUT_SEC", "0.6")
    from eidolon.livekit.plugins.eot.config import EidolonEOTConfig
    cfg = EidolonEOTConfig()
    assert cfg.duck_suspend_timeout_sec == pytest.approx(0.6)


def test_decision_ms_takes_precedence_over_legacy(monkeypatch) -> None:
    """When both env vars are set, the new one wins."""
    monkeypatch.setenv("EIDOLON_INTERRUPT_DECISION_MS", "400")
    monkeypatch.setenv("EIDOLON_DUCK_SUSPEND_TIMEOUT_SEC", "1.5")
    from eidolon.livekit.plugins.eot.config import EidolonEOTConfig
    cfg = EidolonEOTConfig()
    assert cfg.duck_suspend_timeout_sec == pytest.approx(0.4)


def test_default_decision_budget_is_500ms() -> None:
    """Sanity: with no env set, the default lands at 0.5s — the G18a target."""
    import os
    for key in ("EIDOLON_INTERRUPT_DECISION_MS", "EIDOLON_DUCK_SUSPEND_TIMEOUT_SEC"):
        os.environ.pop(key, None)
    from eidolon.livekit.plugins.eot.config import EidolonEOTConfig
    cfg = EidolonEOTConfig()
    assert cfg.duck_suspend_timeout_sec == pytest.approx(0.5)


def test_default_min_chars_is_2() -> None:
    import os
    os.environ.pop("EIDOLON_INTERRUPT_MIN_INTERIM_CHARS", None)
    from eidolon.livekit.plugins.eot.config import EidolonEOTConfig
    cfg = EidolonEOTConfig()
    assert cfg.interrupt_min_interim_chars == 2


def test_default_vad_confidence_gate_disabled() -> None:
    """G18a: default ``min_avg_vad_confidence`` is now 0.0 (gate off)."""
    import os
    os.environ.pop("EIDOLON_EOT_MIN_VAD_CONFIDENCE", None)
    from eidolon.livekit.plugins.eot.config import EidolonEOTConfig
    cfg = EidolonEOTConfig()
    assert cfg.min_avg_vad_confidence == 0.0
