"""Unit tests for G1: `BasePipeline._on_agent_state_changed` state-mirror.

The framework emits `agent_state_changed` events with `new_state` values of
"speaking" / "thinking" / "listening" / occasionally "idle". The pipeline
must keep `self._state` in sync, because downstream code (e.g. F3's
output-flow duck guard) reads it as the source of truth.

Pre-G1 bug: only "idle" branch reset `self._state` to IDLE — but framework
never sends "idle", it sends "listening". So `self._state` got stuck at
SPEAKING after the first agent turn and F3's `if self._state != SPEAKING:
return` never fired.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from eidolon.livekit.agent.pipeline.base import BasePipeline
from eidolon.livekit.agent.pipeline.types import PipelineState


class _StubPipeline(BasePipeline):
    """Minimal concrete subclass to exercise `_on_agent_state_changed`."""

    def __init__(self) -> None:
        # Skip BasePipeline.__init__'s factory wiring — we only care about
        # the state-mirror behaviour, which is independent of factory.
        self._state = PipelineState.IDLE
        self._callbacks = MagicMock()

    async def run(self, room) -> None:  # noqa: D401 — abstract impl required for instantiation
        """Unused; only present to satisfy the ABC."""
        raise NotImplementedError


def _ev(old: str, new: str) -> SimpleNamespace:
    return SimpleNamespace(old_state=old, new_state=new)


def test_speaking_sets_state_speaking() -> None:
    p = _StubPipeline()
    p._on_agent_state_changed(_ev("listening", "speaking"))
    assert p._state == PipelineState.SPEAKING
    p._callbacks.on_agent_started_speaking.assert_called_once()


def test_thinking_sets_state_generating() -> None:
    p = _StubPipeline()
    p._on_agent_state_changed(_ev("listening", "thinking"))
    assert p._state == PipelineState.GENERATING


def test_listening_resets_state_to_idle() -> None:
    """G1 regression: framework sends 'listening' as the quiet state. Before
    the fix, only 'idle' was handled, leaving _state stuck at SPEAKING."""
    p = _StubPipeline()
    # Go SPEAKING first, then back to listening.
    p._on_agent_state_changed(_ev("listening", "speaking"))
    assert p._state == PipelineState.SPEAKING
    p._on_agent_state_changed(_ev("speaking", "listening"))
    assert p._state == PipelineState.IDLE
    p._callbacks.on_agent_ended_speaking.assert_called_once()
    p._callbacks.on_agent_response_done.assert_called_once()


def test_idle_also_resets_state_to_idle() -> None:
    """Forward-compat: if the framework ever sends 'idle' explicitly, we
    still honor it (both names map to PipelineState.IDLE)."""
    p = _StubPipeline()
    p._on_agent_state_changed(_ev("listening", "speaking"))
    p._on_agent_state_changed(_ev("speaking", "idle"))
    assert p._state == PipelineState.IDLE


def test_unknown_new_state_does_not_raise() -> None:
    """Hardened against future framework state names we don't know."""
    p = _StubPipeline()
    p._on_agent_state_changed(_ev("speaking", "some_new_value"))
    # No assertion on _state; the test passes if no exception was raised.
