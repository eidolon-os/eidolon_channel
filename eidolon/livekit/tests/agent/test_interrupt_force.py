"""FullDuplexInterruptionEffects: forced explicit interrupt reaches the framework.

Regression background: an explicit client stop stopped the agent's audio
(mixer cancel) but gave no reply. Root cause: ``cancel_and_interrupt``
cancels the output FIRST (is_cancelled=True), then calls
``interrupt_current_turn(force=True)`` — which short-circuited on
``is_cancelled`` and never called ``session.interrupt(force=True)``. So the
uninterruptible speech handle was never ended, agent_state stayed "speaking",
and the captured user turn couldn't commit. force must bypass that guard.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from eidolon.livekit.agent.full_duplex.interruption_effects import (
    FullDuplexInterruptionEffects,
)


def _effects(
    *,
    is_cancelled: bool,
    allow_interruptions: bool,
) -> tuple[FullDuplexInterruptionEffects, MagicMock]:
    session = MagicMock()
    eot_model = MagicMock()
    effects = FullDuplexInterruptionEffects(
        ducking=SimpleNamespace(is_cancelled=is_cancelled),
        callbacks=MagicMock(),
        get_session=lambda: session,
        allow_interruptions=lambda: allow_interruptions,
        get_eot_model=lambda: eot_model,
        get_timeline=lambda: None,
        get_latest_asr_text=lambda: "",
        get_state_label=lambda: "SPEAKING",
        get_interruption_orchestrator=MagicMock(),
        publish_playback_stop=MagicMock(),
        snapshot_interrupted_context=MagicMock(),
        cancel_residual_commit_suppress_sec=lambda: 0.0,
        semantic_interrupt_run=MagicMock(),
        correction_topic_stability_window_ms=lambda: 120,
        set_interrupt_cancel_suppression=MagicMock(),
        soft_interrupt_timeout_sec=lambda: 0.5,
    )
    return effects, session


def test_forced_interrupt_fires_even_when_output_cancelled() -> None:
    # Explicit client path: cancel_and_interrupt already cancelled output,
    # but the forced interrupt must still end the speech handle.
    effects, session = _effects(is_cancelled=True, allow_interruptions=False)
    effects.interrupt_current_turn(force=True)
    session.interrupt.assert_called_once_with(force=True)


def test_policy_interrupt_skips_when_output_cancelled() -> None:
    # Non-forced (policy / full-duplex) path keeps the redundant-interrupt guard.
    effects, session = _effects(is_cancelled=True, allow_interruptions=True)
    effects.interrupt_current_turn(force=False)
    session.interrupt.assert_not_called()


def test_policy_interrupt_fires_when_not_cancelled() -> None:
    effects, session = _effects(is_cancelled=False, allow_interruptions=True)
    effects.interrupt_current_turn(force=False)
    session.interrupt.assert_called_once_with(force=False)


def test_non_forced_blocked_when_interruptions_disabled() -> None:
    # Non-forced policy path (allow_interruptions=False) is still blocked.
    effects, session = _effects(is_cancelled=False, allow_interruptions=False)
    effects.interrupt_current_turn(force=False)
    session.interrupt.assert_not_called()
