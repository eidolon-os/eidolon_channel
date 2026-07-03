"""_interrupt_current_turn: forced explicit interrupt must reach the framework.

Regression background: an explicit client stop stopped the agent's audio
(mixer cancel) but gave no reply. Root cause: ``_duck_cancel_and_interrupt``
cancels the output FIRST (is_cancelled=True), then calls
``_interrupt_current_turn(force=True)`` — which short-circuited on
``is_cancelled`` and never called ``session.interrupt(force=True)``. So the
uninterruptible speech handle was never ended, agent_state stayed "speaking",
and the captured user turn couldn't commit. force must bypass that guard.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from eidolon.livekit.agent.full_duplex import StreamingPipeline


def _pipe(*, is_cancelled: bool, allow_interruptions: bool) -> StreamingPipeline:
    p = StreamingPipeline.__new__(StreamingPipeline)
    p._ducking = SimpleNamespace(is_cancelled=is_cancelled)
    p._allow_interruptions = allow_interruptions
    p._session = MagicMock()
    p._get_eot_model = MagicMock(return_value=MagicMock())
    return p


def test_forced_interrupt_fires_even_when_output_cancelled() -> None:
    # Explicit client path: _duck_cancel_and_interrupt already cancelled output,
    # but the forced interrupt must still end the speech handle.
    p = _pipe(is_cancelled=True, allow_interruptions=False)
    p._interrupt_current_turn(force=True)
    p._session.interrupt.assert_called_once_with(force=True)


def test_policy_interrupt_skips_when_output_cancelled() -> None:
    # Non-forced (policy / full-duplex) path keeps the redundant-interrupt guard.
    p = _pipe(is_cancelled=True, allow_interruptions=True)
    p._interrupt_current_turn(force=False)
    p._session.interrupt.assert_not_called()


def test_policy_interrupt_fires_when_not_cancelled() -> None:
    p = _pipe(is_cancelled=False, allow_interruptions=True)
    p._interrupt_current_turn(force=False)
    p._session.interrupt.assert_called_once_with(force=False)


def test_non_forced_blocked_when_interruptions_disabled() -> None:
    # Non-forced policy path (allow_interruptions=False) is still blocked.
    p = _pipe(is_cancelled=False, allow_interruptions=False)
    p._interrupt_current_turn(force=False)
    p._session.interrupt.assert_not_called()
