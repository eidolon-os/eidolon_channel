import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from livekit.agents.voice import AgentSession
from eidolon.livekit.agent.session.presentation_input_gate import PresentationInputGate


def session_with_input():
    # Exercise the real AgentInput attach/detach boundary, not a substitute gate.
    session = AgentSession()
    stream = Mock()
    session.input.audio = stream
    return session, stream


def state(session, value):
    session.emit("agent_state_changed", SimpleNamespace(new_state=value))


@pytest.mark.asyncio
async def test_replayed_speaker_and_user_vad_sequence_keeps_capture_closed():
    session, stream = session_with_input()
    gate = PresentationInputGate(session, tail_seconds=0.02)
    state(session, "speaking")
    assert not session.input.audio_enabled
    stream.on_detached.assert_called_once()
    # Incident sequence: reply -> echo VAD starts/stops -> repeated timeout reply.
    for user_state in ("speaking", "listening", "away"):
        session.emit("user_state_changed", SimpleNamespace(new_state=user_state))
        assert not session.input.audio_enabled
    state(session, "listening")
    assert not session.input.audio_enabled
    await asyncio.sleep(0.03)
    assert session.input.audio_enabled
    # New human turn can now enter the same input stream.
    assert stream.on_attached.call_count == 2
    gate.close()


@pytest.mark.asyncio
async def test_new_playback_cancels_previous_release_and_close_never_reopens():
    session, _ = session_with_input()
    gate = PresentationInputGate(session, tail_seconds=0.02)
    state(session, "speaking")
    state(session, "listening")
    state(session, "speaking")
    await asyncio.sleep(0.03)
    assert not session.input.audio_enabled
    state(session, "listening")
    session.emit("close", SimpleNamespace())
    await asyncio.sleep(0.03)
    assert not session.input.audio_enabled
    assert gate._release is None


@pytest.mark.asyncio
async def test_preexisting_disabled_input_is_not_enabled_by_playback_end():
    session, _ = session_with_input()
    session.input.set_audio_enabled(False)
    gate = PresentationInputGate(session, tail_seconds=0)
    state(session, "speaking")
    state(session, "listening")
    await asyncio.sleep(0.01)
    assert not session.input.audio_enabled
    gate.close()
