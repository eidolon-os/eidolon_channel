"""The selected speaker is the sole output sink of one existing voice session."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from livekit.agents.voice.room_io import RoomOptions
from eidolon.livekit.agent.session import room_io


def room(name):
    return SimpleNamespace(name=name, isconnected=lambda: True,
                           remote_participants={name: SimpleNamespace(identity=name)})


@pytest.fixture
def setup(monkeypatch):
    source, target = room("input"), room("speaker")
    session = SimpleNamespace(output=SimpleNamespace(audio=None, transcription=None), start=AsyncMock())
    outputs = []

    class Output:
        def __init__(self, actual_session, actual_room, *, options):
            assert actual_session is session
            self.room, self.options = actual_room, options
            self.start = AsyncMock()
            self.wait_for_ready = AsyncMock()
            self.aclose = AsyncMock()
            outputs.append(self)

    monkeypatch.setattr(room_io, "RoomIO", Output)
    return source, target, session, outputs


async def test_single_room_preserves_existing_start(setup):
    source, _, session, outputs = setup
    options = RoomOptions(participant_identity="input")
    assert await room_io.start_room_session(session=session, agent="agent", input_room=source,
                                           options=options) is None
    session.start.assert_awaited_once_with(agent="agent", room=source, room_options=options)
    assert outputs == []


async def test_only_selected_speaker_gets_output_and_cannot_supply_input(setup):
    source, target, session, outputs = setup
    original = RoomOptions(participant_identity="input")
    output = await room_io.start_room_session(session=session, agent="one-agent", input_room=source,
        options=original, output_room=target, output_participant_identity="speaker")
    assert output is outputs[0]
    assert output.room is target
    assert output.options.participant_identity == "speaker"
    assert output.options.get_audio_input_options() is None
    assert output.options.get_video_input_options() is None
    assert output.options.get_text_input_options() is None
    assert output.options.get_audio_output_options() is not None
    args = session.start.await_args.kwargs
    assert args["room"] is source
    assert args["room_options"].participant_identity == "input"
    assert args["room_options"].get_audio_output_options() is None
    assert args["room_options"].get_text_output_options() is None
    assert original.get_audio_output_options() is not None
    output.wait_for_ready.assert_awaited_once()


async def test_start_failure_closes_output_and_never_falls_back_to_source(setup):
    source, target, session, outputs = setup
    session.start.side_effect = RuntimeError("start failed")
    with pytest.raises(RuntimeError, match="start failed"):
        await room_io.start_room_session(session=session, agent="agent", input_room=source,
            options=RoomOptions(), output_room=target, output_participant_identity="speaker")
    outputs[0].aclose.assert_awaited_once()
    assert session.start.await_count == 1
    assert session.start.await_args.kwargs["room_options"].audio_output is False


@pytest.mark.parametrize("identity", [None, "different-device"])
async def test_missing_selected_participant_fails_before_any_start(setup, identity):
    source, target, session, outputs = setup
    with pytest.raises(ValueError):
        await room_io.start_room_session(session=session, agent="agent", input_room=source,
            options=RoomOptions(), output_room=target, output_participant_identity=identity)
    assert outputs == []
    session.start.assert_not_awaited()


async def test_native_room_io_publishes_only_to_selected_room():
    from livekit import rtc
    from livekit.agents.voice import AgentSession

    class Room(rtc.EventEmitter):
        def __init__(self, identity):
            super().__init__()
            self.name = identity
            self.remote_participants = {identity: SimpleNamespace(
                identity=identity, kind=rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
                attributes={}, track_publications={},
            )}
            self.local_participant = SimpleNamespace(
                identity="agent", publish_track=AsyncMock(return_value=SimpleNamespace(
                    wait_for_subscription=AsyncMock(), sid="audio-track",
                )), set_attributes=AsyncMock(),
            )

        def isconnected(self):
            return True

    source, target = Room("input"), Room("speaker")
    session = AgentSession()
    session.start = AsyncMock()
    output = await room_io.start_room_session(session=session, agent="agent", input_room=source,
        options=RoomOptions(text_output=False), output_room=target,
        output_participant_identity="speaker")
    try:
        assert session.input.audio is None
        assert session.output.audio is output.audio_output
        target.local_participant.publish_track.assert_awaited_once()
        source.local_participant.publish_track.assert_not_awaited()
        await session.output.audio.capture_frame(rtc.AudioFrame.create(24000, 1, 480))
        session.output.audio.flush()
    finally:
        await output.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['ptt', 'full_duplex'])
async def test_playback_stop_targets_only_presentation_device(mode):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from eidolon_sdk.biz.contracts import CONTROL_TOPIC
    from eidolon.livekit.agent.half_duplex.pipeline import HalfDuplexPttPipeline
    from eidolon.livekit.agent.full_duplex.client_control_publisher import FullDuplexClientControlPublisher
    source = SimpleNamespace(local_participant=SimpleNamespace(publish_data=AsyncMock()))
    target = SimpleNamespace(local_participant=SimpleNamespace(publish_data=AsyncMock()))
    pipeline = SimpleNamespace(_room=source, _presentation_room=target,
        _presentation_peer='speaker', _timeline=None, _pending_client_control_events=[])
    if mode == 'ptt':
        HalfDuplexPttPipeline._publish_data(pipeline, CONTROL_TOPIC, {'op': 'playback.stop'})
    else:
        FullDuplexClientControlPublisher(pipeline).publish_client_control('playback.stop', reason='cancel')
    await asyncio.sleep(0)
    source.local_participant.publish_data.assert_not_called()
    target.local_participant.publish_data.assert_awaited_once()
    assert target.local_participant.publish_data.call_args.kwargs['destination_identities'] == ['speaker']
