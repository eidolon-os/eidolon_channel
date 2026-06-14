"""Fast-path explicit client interrupt (device manual_interrupt) tests.

`_handle_explicit_client_interrupt` is the low-latency barge-in path: it fires
on the raw client.audio_state data packet (device manual_interrupt / ptt),
BEFORE STT produces a transcript, and cancels the agent's TTS. This is separate
from the transcript/attention-admission path (which Step 1's attention.enforce
fix covers).

These tests guard the gate logic of that fast path. Note a known gap they
document: the fast path does NOT inspect transcript content, so it cannot apply
a backchannel guard at signal time (a loud "嗯" that trips the device energy gate
would cancel here). Whether that over-cancels in practice is a lifecycle/timing
question best confirmed on the real device; the policy/transcript path already
holds backchannels via the classifier.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

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


def test_manual_interrupt_while_speaking_cancels() -> None:
    p = _pipeline(state=_state(manual_interrupt=True))
    p._handle_explicit_client_interrupt(_packet())
    p._duck_cancel_and_interrupt.assert_called_once()


def test_no_signal_does_not_cancel() -> None:
    # Plain audio_state (no manual_interrupt / ptt) must not hard-cancel.
    p = _pipeline(state=_state())
    p._handle_explicit_client_interrupt(_packet())
    p._duck_cancel_and_interrupt.assert_not_called()


def test_wrong_topic_ignored() -> None:
    p = _pipeline(state=_state(manual_interrupt=True))
    pkt = SimpleNamespace(
        topic="eidolon.something_else",
        participant=SimpleNamespace(identity="dev1"),
    )
    p._handle_explicit_client_interrupt(pkt)
    p._duck_cancel_and_interrupt.assert_not_called()


def test_no_cancel_when_output_already_cancelled() -> None:
    # Idempotent: if the agent output is already CANCELLED, the fast path bails.
    p = _pipeline(state=_state(manual_interrupt=True), output_cancelled=True)
    p._handle_explicit_client_interrupt(_packet())
    p._duck_cancel_and_interrupt.assert_not_called()


def test_fast_path_ignores_transcript_no_backchannel_guard() -> None:
    # Documents the known gap: the fast path fires on the signal alone and does
    # not see the transcript, so even a backchannel-intent utterance cancels here
    # when manual_interrupt is set. The backchannel guard lives on the
    # transcript/classifier path; closing this at signal time would need a
    # duck-then-confirm design (tracked for the device e2e step).
    p = _pipeline(state=_state(manual_interrupt=True))
    p._handle_explicit_client_interrupt(_packet())
    p._duck_cancel_and_interrupt.assert_called_once()
