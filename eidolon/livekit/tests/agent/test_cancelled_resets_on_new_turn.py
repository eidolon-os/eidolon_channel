"""G22-fix (2026-05-18): OutputController CANCELLED → NORMAL on new turn.

Bug found in production round-2 interrupt log:
  - User interrupted agent's reply at T+9s → mixer.cancel() → CANCELLED.
  - User spoke a second utterance → LLM responded → TTS generated 17s of
    audio (544320 bytes).
  - agent_state STAYED in "thinking", never transitioned to "speaking".
  - User heard NOTHING for the second reply.

Root cause: OutputController.cancel() leaves state="CANCELLED" with no
auto-recovery path. Every subsequent ``capture_frame`` hits the
``if self._state == "CANCELLED": return`` guard and drops the frame.
The framework gates ``agent_state → speaking`` on the first audio frame
reaching the inner sink (RoomIO), so the dropped-frames pipeline never
trips that transition.

Fix: in ``StreamingPipeline._on_agent_state_changed``, on transition to
``thinking`` (= new LLM call = new agent turn coming), if the mixer is
still in CANCELLED, call ``mixer.reset()`` to clear it back to NORMAL.

These tests verify both halves: the reset() helper, and the wiring.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from livekit import rtc
from livekit.agents.voice import io as lk_io

from eidolon.livekit.agent.output.controller import OutputController
from eidolon.livekit.agent.output.ducking import OutputDuckingController


SAMPLE_RATE = 16000


class _StubInner(lk_io.AudioOutput):
    def __init__(self) -> None:
        super().__init__(
            label="stub",
            capabilities=lk_io.AudioOutputCapabilities(pause=True),
            sample_rate=SAMPLE_RATE,
        )
        self.captured: int = 0

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        self.captured += 1

    def flush(self) -> None:
        super().flush()

    def clear_buffer(self) -> None:
        pass


def _frame(value: int = 1000, samples: int = 800) -> rtc.AudioFrame:
    arr = np.full(samples, value, dtype=np.int16)
    return rtc.AudioFrame(
        data=arr.tobytes(),
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=samples,
    )


# ---------------------------------------------------------------------------
# OutputController.reset() — already existed; verify it clears CANCELLED.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reset_recovers_from_cancelled() -> None:
    """After cancel(), reset() must return state to NORMAL so subsequent
    frames pass through again."""
    inner = _StubInner()
    mixer = OutputController(inner, sample_rate=SAMPLE_RATE)

    mixer.cancel()
    assert mixer.state == "CANCELLED"

    # Frame dropped because of CANCELLED guard
    await mixer.capture_frame(_frame())
    captured_before_reset = inner.captured

    mixer.reset()
    assert mixer.state == "NORMAL"

    # Frame now goes through
    await mixer.capture_frame(_frame())
    assert inner.captured == captured_before_reset + 1


# ---------------------------------------------------------------------------
# StreamingPipeline._on_agent_state_changed wiring (the actual bug location)
# ---------------------------------------------------------------------------


def _make_pipeline_with_mixer(initial_state: str = "NORMAL"):
    """Stub pipeline with a real OutputController in a known state."""
    from eidolon.livekit.agent.full_duplex import StreamingPipeline

    pipeline = StreamingPipeline.__new__(StreamingPipeline)

    inner = _StubInner()
    mixer = OutputController(inner, sample_rate=SAMPLE_RATE)
    if initial_state == "CANCELLED":
        mixer.cancel()
    pipeline._ducking = OutputDuckingController()
    pipeline._ducking.mixer = mixer
    pipeline._filler = None

    # super()._on_agent_state_changed needs ``_state`` attr (PipelineState mirror)
    from eidolon.livekit.agent.shared.types import PipelineState
    pipeline._state = PipelineState.IDLE

    return pipeline, mixer


def test_thinking_transition_resets_cancelled_mixer() -> None:
    """G22-fix core regression: agent_state listening→thinking with mixer
    in CANCELLED state triggers reset()."""
    pipeline, mixer = _make_pipeline_with_mixer(initial_state="CANCELLED")
    assert mixer.state == "CANCELLED"

    event = SimpleNamespace(old_state="listening", new_state="thinking")
    pipeline._on_agent_state_changed(event)

    assert mixer.state == "NORMAL"


def test_thinking_transition_noop_when_normal() -> None:
    """If mixer is already NORMAL, the thinking transition is a no-op
    (no spurious reset)."""
    pipeline, mixer = _make_pipeline_with_mixer(initial_state="NORMAL")
    assert mixer.state == "NORMAL"

    event = SimpleNamespace(old_state="listening", new_state="thinking")
    pipeline._on_agent_state_changed(event)

    # Still NORMAL — no double-reset side effects
    assert mixer.state == "NORMAL"


def test_speaking_transition_does_not_reset_cancelled() -> None:
    """Symmetric check: speaking transition handles its own counter reset
    (on_agent_started_speaking) but does NOT touch state. Reset happens
    on thinking transition because that's the EARLIEST signal of a new
    turn — TTS frames may arrive before we hit speaking."""
    pipeline, mixer = _make_pipeline_with_mixer(initial_state="CANCELLED")

    event = SimpleNamespace(old_state="listening", new_state="speaking")
    pipeline._on_agent_state_changed(event)

    # Still CANCELLED — only "thinking" transition triggers state reset
    assert mixer.state == "CANCELLED"


@pytest.mark.asyncio
async def test_full_interrupt_then_new_turn_unblocks_audio() -> None:
    """End-to-end scenario: cancel, send frame (dropped), simulate new
    turn (thinking transition), send frame (now passes through)."""
    pipeline, mixer = _make_pipeline_with_mixer(initial_state="NORMAL")

    # Stage 1: in-flight TTS gets cancelled (interrupt)
    mixer.cancel()
    assert mixer.state == "CANCELLED"

    # Stage 2: TTS keeps generating (e.g. in_flight frame arrives) — dropped
    inner = mixer._inner
    captured_after_cancel = inner.captured
    await mixer.capture_frame(_frame())
    assert inner.captured == captured_after_cancel, "frame should be dropped"

    # Stage 3: new user turn starts, agent_state → thinking
    event = SimpleNamespace(old_state="listening", new_state="thinking")
    pipeline._on_agent_state_changed(event)
    assert mixer.state == "NORMAL"

    # Stage 4: new TTS frame for the new turn — passes through
    await mixer.capture_frame(_frame())
    assert inner.captured == captured_after_cancel + 1


def test_thinking_transition_safe_without_mixer() -> None:
    """No-op when there is no mixer (headless tests, duck_enabled=False)."""
    from eidolon.livekit.agent.full_duplex import StreamingPipeline
    from eidolon.livekit.agent.shared.types import PipelineState

    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._ducking = OutputDuckingController()
    pipeline._ducking.mixer = None
    pipeline._state = PipelineState.IDLE
    pipeline._filler = None

    event = SimpleNamespace(old_state="listening", new_state="thinking")
    # Must not raise
    pipeline._on_agent_state_changed(event)
