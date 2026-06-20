"""Fast-path explicit client interrupt (PTT tap-to-stop) tests.

``_handle_explicit_client_interrupt`` is the low-latency barge-in path: it fires
on the raw client.audio_state data packet, BEFORE STT produces a transcript, and
hard-cancels the agent's TTS. This is the half_duplex tap-to-stop mechanism.

PTT is the ONLY explicit client interrupt. The device's energy-gate
``manual_interrupt`` guess was removed (it falsely tripped on residual playback
echo); open-mic full_duplex barge-in is judged server-side from the clean
transcript/attention path instead. The cut uses ``force=True`` so it works even
in half_duplex, where the session runs with allow_interruptions=False.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from eidolon.livekit.agent.client_audio_state import (
    CLIENT_AUDIO_STATE_TOPIC,
    ClientAudioState,
)
from eidolon.livekit.agent.output.ducking import OutputDuckingController
from eidolon.livekit.agent.pipeline.types import PipelineState
from eidolon.livekit.agent.streaming import StreamingPipeline
from eidolon.livekit.common.config import TurnPolicyConfig


def _pipeline(
    *,
    state: ClientAudioState | None,
    pipeline_state: PipelineState = PipelineState.SPEAKING,
    output_cancelled: bool = False,
) -> StreamingPipeline:
    p = StreamingPipeline.__new__(StreamingPipeline)
    p._turn_policy = TurnPolicyConfig()
    p._state = pipeline_state
    p._timeline = None
    if output_cancelled:
        # is_cancelled requires a mixer in CANCELLED state; stub it directly.
        p._ducking = MagicMock(is_cancelled=True)
    else:
        p._ducking = OutputDuckingController()
    # The state lookup itself is not under test (the packet was received per
    # production logs); the gate logic after it is.
    p._latest_client_audio_state = MagicMock(return_value=state)
    p._duck_cancel_and_interrupt = MagicMock()
    return p


def _packet(identity: str = "dev1") -> SimpleNamespace:
    return SimpleNamespace(
        topic=CLIENT_AUDIO_STATE_TOPIC,
        participant=SimpleNamespace(identity=identity),
    )


def _state(**kwargs) -> ClientAudioState:
    base = {
        "participant_identity": "dev1",
        "playback_state": "agent_speaking",
    }
    base.update(kwargs)
    return ClientAudioState(**base)


def test_ptt_while_speaking_force_cancels() -> None:
    # PTT is a deliberate button press → immediate hard cut, force=True so it
    # cuts through half_duplex's allow_interruptions=False.
    p = _pipeline(state=_state(ptt=True))
    p._handle_explicit_client_interrupt(_packet())
    p._duck_cancel_and_interrupt.assert_called_once_with(force=True)


def test_no_signal_does_not_cancel() -> None:
    # Plain audio_state (no ptt) must do nothing.
    p = _pipeline(state=_state())
    p._handle_explicit_client_interrupt(_packet())
    p._duck_cancel_and_interrupt.assert_not_called()


def test_manual_interrupt_alone_does_nothing() -> None:
    # The removed energy-gate signal is no longer an interrupt trigger.
    p = _pipeline(state=_state(manual_interrupt=True))
    p._handle_explicit_client_interrupt(_packet())
    p._duck_cancel_and_interrupt.assert_not_called()


def test_wrong_topic_ignored() -> None:
    p = _pipeline(state=_state(ptt=True))
    pkt = SimpleNamespace(
        topic="eidolon.something_else",
        participant=SimpleNamespace(identity="dev1"),
    )
    p._handle_explicit_client_interrupt(pkt)
    p._duck_cancel_and_interrupt.assert_not_called()


def test_no_action_when_output_already_cancelled() -> None:
    # Idempotent: if the agent output is already CANCELLED, the fast path bails.
    p = _pipeline(state=_state(ptt=True), output_cancelled=True)
    p._handle_explicit_client_interrupt(_packet())
    p._duck_cancel_and_interrupt.assert_not_called()
