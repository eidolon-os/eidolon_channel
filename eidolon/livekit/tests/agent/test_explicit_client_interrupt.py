"""Fast-path explicit client interrupt (device manual_interrupt) tests.

`_handle_explicit_client_interrupt` is the low-latency barge-in path: it fires
on the raw client.audio_state data packet (device manual_interrupt / ptt),
BEFORE STT produces a transcript, and cancels the agent's TTS. This is separate
from the transcript/attention-admission path (which Step 1's attention.enforce
fix covers).

These tests guard the gate logic of that fast path. P1 (2026-06-16): the energy-
gate `manual_interrupt` is unreliable (residual playback echo trips it on-device),
so it no longer hard-cuts at signal time — it ducks (reversible) and arms the
existing suspend timeout, letting the transcript/SemanticInterrupt path confirm
(real speech → cancel, echo/backchannel → resume). `ptt` (a deliberate button)
still hard-cuts immediately.
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
    p._duck_and_arm_timeout = MagicMock()
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


def test_manual_interrupt_while_speaking_ducks_not_cancels() -> None:
    # P1: device manual_interrupt is an UNRELIABLE energy-gate signal (residual
    # echo can trip it). It must NOT hard-cut. Instead it ducks (reversible) and
    # arms the existing suspend-timeout so the evidence/semantic path confirms:
    # real speech escalates to cancel, echo/no-content resumes on timeout.
    p = _pipeline(state=_state(manual_interrupt=True))
    p._handle_explicit_client_interrupt(_packet())
    p._duck_and_arm_timeout.assert_called_once()
    p._duck_cancel_and_interrupt.assert_not_called()


def test_ptt_while_speaking_cancels() -> None:
    # PTT is a deliberate button press, not an energy guess — keep the immediate
    # hard cut.
    p = _pipeline(state=_state(ptt=True))
    p._handle_explicit_client_interrupt(_packet())
    p._duck_cancel_and_interrupt.assert_called_once()
    p._duck_and_arm_timeout.assert_not_called()


def test_no_signal_does_not_duck_or_cancel() -> None:
    # Plain audio_state (no manual_interrupt / ptt) must do nothing.
    p = _pipeline(state=_state())
    p._handle_explicit_client_interrupt(_packet())
    p._duck_cancel_and_interrupt.assert_not_called()
    p._duck_and_arm_timeout.assert_not_called()


def test_wrong_topic_ignored() -> None:
    p = _pipeline(state=_state(manual_interrupt=True))
    pkt = SimpleNamespace(
        topic="eidolon.something_else",
        participant=SimpleNamespace(identity="dev1"),
    )
    p._handle_explicit_client_interrupt(pkt)
    p._duck_cancel_and_interrupt.assert_not_called()
    p._duck_and_arm_timeout.assert_not_called()


def test_no_action_when_output_already_cancelled() -> None:
    # Idempotent: if the agent output is already CANCELLED, the fast path bails.
    p = _pipeline(state=_state(manual_interrupt=True), output_cancelled=True)
    p._handle_explicit_client_interrupt(_packet())
    p._duck_cancel_and_interrupt.assert_not_called()
    p._duck_and_arm_timeout.assert_not_called()


def test_manual_interrupt_backchannel_gap_closed() -> None:
    # Previously the fast path hard-cut on the signal alone (a loud "嗯" or echo
    # would cancel). Now manual_interrupt only ducks; the existing transcript/
    # semantic + suspend-timeout machinery resumes when there's no real speech,
    # so echo/backchannels no longer truncate the agent at signal time.
    p = _pipeline(state=_state(manual_interrupt=True))
    p._handle_explicit_client_interrupt(_packet())
    p._duck_cancel_and_interrupt.assert_not_called()
    p._duck_and_arm_timeout.assert_called_once()
