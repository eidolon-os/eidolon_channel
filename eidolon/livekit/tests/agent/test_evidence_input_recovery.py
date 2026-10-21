"""Real RoomIO/AgentSession lifecycle; only RTC transport and ASR are synthetic."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import websockets
from livekit import rtc
from livekit.agents import Agent, AgentSession, stt
from livekit.agents.voice.room_io import AudioInputOptions
from livekit.agents.types import APIConnectOptions

from eidolon.livekit.agent.session.evidence_room_input import EvidenceRoomInput
from eidolon.livekit.common.audio_timeline import AudioSource, AudioSpan
from eidolon.livekit.common.transcript_evidence import transcript_evidence_from_speech_event
from eidolon.livekit.plugins.stt.bailian import BailianFunASRSTT
from .._harness.headless import RecordingAudioOutput, ScriptedAudioInput


class Room(rtc.EventEmitter):
    def __init__(self):
        super().__init__()
        self.name = "synthetic-room"
        self.connected = True
        self.local_participant = SimpleNamespace(identity="agent", set_attributes=AsyncMock())
        self.publication = SimpleNamespace(
            sid="mic-1",
            source=rtc.TrackSource.SOURCE_MICROPHONE,
            track=SimpleNamespace(sid="mic-1"),
            muted=False,
            audio_features=[],
        )
        self.participant = SimpleNamespace(
            identity="selected",
            kind=rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
            attributes={},
            track_publications={"mic-1": self.publication},
        )
        self.remote_participants = {"selected": self.participant}

    def isconnected(self):
        return self.connected


class Transport:
    def __init__(self, track):
        self.track = track
        self.frames = asyncio.Queue()
        self.closed = False
        self.delivered = asyncio.Event()
        self.close_started = asyncio.Event()
        self.close_finished = asyncio.Event()
        self.fail_close_once = False
        self.allow_close = asyncio.Event()
        self.allow_close.set()

    def __aiter__(self):
        return self

    async def __anext__(self):
        value = await self.frames.get()
        if value is None:
            raise StopAsyncIteration
        self.delivered.set()
        return SimpleNamespace(frame=value)

    async def aclose(self):
        self.close_started.set()
        await self.allow_close.wait()
        if self.fail_close_once:
            self.fail_close_once = False
            raise RuntimeError("synthetic cleanup failure")
        self.closed = True
        self.close_finished.set()
        self.frames.put_nowait(None)

    def feed(self, value):
        self.frames.put_nowait(
            rtc.AudioFrame(value.to_bytes(2, "little", signed=True) * 160, 16000, 1, 160)
        )


class Recognizer(stt.STT):
    def __init__(self):
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))
        self.capture = None
        self.streams = 0
        self.received = asyncio.Queue()

    async def _recognize_impl(self, buffer, *, language, conn_options):
        raise AssertionError("streaming only")

    def bind(self, capture):
        self.capture = capture

    def stream(self, *, language=None, conn_options=None):
        self.streams += 1
        recognizer, capture = self, self.capture
        source = AudioSource(f"stream-{self.streams}", 1, 16000)
        if capture is not None:
            capture.select_stream(source.source)

        class Stream(stt.RecognizeStream):
            async def _run(self):
                if capture is not None:
                    capture.begin_run(source)
                offset = 0
                try:
                    async for frame in self._input_ch:
                        if not isinstance(frame, rtc.AudioFrame):
                            continue
                        if capture is not None:
                            capture.append(source, start=offset, pcm=bytes(frame.data))
                        end = offset + frame.samples_per_channel
                        recognizer.received.put_nowait(
                            (
                                capture,
                                AudioSpan(source.source, 1, 16000, offset, end),
                                bytes(frame.data),
                            )
                        )
                        offset = end
                finally:
                    if capture is not None:
                        capture.end_run(source)

        return Stream(stt=self, conn_options=conn_options, sample_rate=16000)


@pytest.fixture
async def running(monkeypatch):
    transports = []

    def from_track(*, track, **kwargs):
        value = Transport(track)
        transports.append(value)
        return value

    monkeypatch.setattr(rtc.AudioStream, "from_track", from_track)
    room, recognizer = Room(), Recognizer()
    session = AgentSession(aec_warmup_duration=0, session_close_transcript_timeout=0.0)
    output = RecordingAudioOutput(sample_rate=24000)
    session.output.audio = output
    await session.start(
        agent=Agent(instructions="", stt=recognizer, turn_handling={"turn_detection": "manual"})
    )
    retired = []
    owner = EvidenceRoomInput(
        session=session,
        room=room,
        participant_identity="selected",
        session_id="session",
        endpoint_id="endpoint",
        bind_capture=recognizer.bind,
        retire_candidate=retired.append,
        audio_options=AudioInputOptions(
            sample_rate=16000, auto_gain_control=False, pre_connect_audio=False
        ),
    )
    try:
        await owner.start()
        yield owner, session, room, recognizer, transports, retired, output
    finally:
        await owner.aclose()
        await session.aclose()


async def next_audio(recognizer):
    return await asyncio.wait_for(recognizer.received.get(), 2)


@pytest.mark.asyncio
async def test_pause_resume_gets_fresh_roomio_and_asr_without_old_queued_audio(running):
    owner, session, room, recognizer, transports, retired, output = running
    transports[-1].feed(11)
    first, old_span, _ = await next_audio(recognizer)
    assert first.snapshot(candidate="old", span=old_span).complete
    old_transport = transports[-1]
    old_transport.delivered.clear()
    old_transport.feed(99)  # intentionally pending when paused
    await old_transport.delivered.wait()  # now delivered into the SDK RoomIO queue
    session.input.set_audio_enabled(False)
    assert first.active_source is None and first.retained_samples == 0
    session.input.set_audio_enabled(True)
    await asyncio.wait_for(owner.wait_for_ready(), 2)
    assert old_transport.closed and transports[-1] is not old_transport
    transports[-1].feed(22)
    current, span, pcm = await next_audio(recognizer)
    # A frame already consumed by the old recognizer cannot be unsent. It must
    # stay old/invalid, never appear under the new capture after recovery.
    if current is first:
        assert pcm == b"\x63\x00" * 160
        assert not current.snapshot(candidate="late", span=span).complete
        current, span, pcm = await next_audio(recognizer)
    assert current is not first and current is owner.capture
    assert current.snapshot(candidate="new", span=span).complete
    assert pcm == b"\x16\x00" * 160
    assert not first.snapshot(candidate="late", span=old_span).complete
    assert retired and session.output.audio is output


@pytest.mark.asyncio
async def test_microphone_replacement_recovers_same_session(running):
    owner, session, room, recognizer, transports, retired, output = running
    old_transport = transports[-1]
    first = owner.capture
    room.publication = SimpleNamespace(
        sid="mic-2",
        source=rtc.TrackSource.SOURCE_MICROPHONE,
        track=SimpleNamespace(sid="mic-2"),
        muted=False,
        audio_features=[],
    )
    room.participant.track_publications = {"mic-2": room.publication}
    room.emit("track_subscribed", room.publication.track, room.publication, room.participant)
    await asyncio.wait_for(owner.wait_for_ready(), 2)
    assert old_transport.closed and transports[-1].track.sid == "mic-2"
    transports[-1].feed(33)
    current, span, pcm = await next_audio(recognizer)
    assert current is owner.capture and current is not first
    assert current.snapshot(candidate="new", span=span).complete
    assert pcm == b"\x21\x00" * 160
    assert session.output.audio is output


async def wait_state(owner, state):
    async def wait():
        while owner.state != state:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), 2)


@pytest.mark.asyncio
async def test_pause_during_old_reader_cleanup_waits_for_resume(running):
    owner, session, room, recognizer, transports, retired, output = running
    old = transports[-1]
    old.allow_close.clear()
    room.emit("track_unmuted", room.publication, room.participant)
    await asyncio.wait_for(old.close_started.wait(), 2)
    session.input.set_audio_enabled(False)
    old.allow_close.set()
    await asyncio.wait_for(old.close_finished.wait(), 2)
    await wait_state(owner, "paused")
    assert old.closed and len(transports) == 1
    session.input.set_audio_enabled(True)
    await asyncio.wait_for(owner.wait_for_ready(), 2)
    transports[-1].feed(44)
    capture, span, pcm = await next_audio(recognizer)
    assert capture is owner.capture and capture.snapshot(candidate="resume", span=span).complete
    assert pcm == b"\x2c\x00" * 160


@pytest.mark.asyncio
async def test_close_during_sdk_cleanup_finishes_same_cleanup_once(running):
    owner, session, room, recognizer, transports, retired, output = running
    old = transports[-1]
    old.allow_close.clear()
    room.emit("track_unmuted", room.publication, room.participant)
    await asyncio.wait_for(old.close_started.wait(), 2)
    closing = asyncio.create_task(owner.aclose())
    await asyncio.sleep(0)
    assert not closing.done()
    old.allow_close.set()
    await asyncio.wait_for(closing, 2)
    assert old.closed and len(transports) == 1
    assert session.input.audio is None and owner.capture is None


@pytest.mark.asyncio
async def test_room_reconnect_waits_for_selected_microphone_then_recovers(running):
    owner, session, room, recognizer, transports, retired, output = running
    old = owner.capture
    room.connected = False
    room.emit("reconnecting")
    await wait_state(owner, "waiting")
    assert old.active_source is None and all(value.closed for value in transports)
    room.connected = True
    room.emit("reconnected")
    await asyncio.wait_for(owner.wait_for_ready(), 2)
    transports[-1].feed(55)
    capture, span, pcm = await next_audio(recognizer)
    assert capture is not old and capture is owner.capture
    assert capture.snapshot(candidate="reconnected", span=span).complete
    assert pcm == b"\x37\x00" * 160


@pytest.mark.asyncio
async def test_external_input_owner_is_not_overwritten_by_recovery(running):
    owner, session, room, recognizer, transports, retired, output = running
    replacement = ScriptedAudioInput(sample_rate=16000)
    session.input.audio = replacement
    await wait_state(owner, "replaced")
    assert session.input.audio is replacement and owner.capture is None
    room.emit("track_unmuted", room.publication, room.participant)
    await asyncio.sleep(0)
    assert session.input.audio is replacement and len(transports) == 1


@pytest.mark.asyncio
async def test_external_takeover_during_cleanup_is_not_overwritten(running):
    owner, session, room, recognizer, transports, retired, output = running
    old = transports[-1]
    old.allow_close.clear()
    room.emit("track_unmuted", room.publication, room.participant)
    await asyncio.wait_for(old.close_started.wait(), 2)
    replacement = ScriptedAudioInput(sample_rate=16000)
    session.input.audio = replacement
    old.allow_close.set()
    await wait_state(owner, "replaced")
    assert session.input.audio is replacement and len(transports) == 1
    assert owner.capture is None and old.closed


@pytest.mark.asyncio
async def test_session_close_does_not_start_recovery_from_cancelled_read(running):
    owner, session, room, recognizer, transports, retired, output = running
    await session.aclose()
    await owner.aclose()
    assert owner.state == "closed" and owner.capture is None
    assert len(transports) == 1 and transports[0].closed


@pytest.mark.asyncio
async def test_failed_new_input_is_unavailable_and_does_not_fall_back_to_old_audio(
    running, monkeypatch
):
    owner, session, room, recognizer, transports, retired, output = running
    from eidolon.livekit.agent.session.evidence_room_input import RoomIO

    original = RoomIO.start

    async def fail_after_attach(io):
        await original(io)
        raise RuntimeError("synthetic setup failure")

    monkeypatch.setattr(RoomIO, "start", fail_after_attach)
    old = owner.capture
    room.emit("track_unmuted", room.publication, room.participant)
    with pytest.raises(RuntimeError, match="failed"):
        await asyncio.wait_for(owner.wait_for_ready(), 2)
    assert owner.failure == "RuntimeError"
    assert owner.capture.active_source is None and old.active_source is None
    assert session.input.audio is None
    assert all(value.closed for value in transports)
    assert session.output.audio is output
    monkeypatch.setattr(RoomIO, "start", original)
    room.emit("track_unmuted", room.publication, room.participant)
    await asyncio.wait_for(owner.wait_for_ready(), 2)
    assert owner.failure is None


@pytest.mark.asyncio
async def test_input_recovery_does_not_interrupt_current_playout(running):
    owner, session, room, recognizer, transports, retired, output = running
    started, release = asyncio.Event(), asyncio.Event()

    async def audio():
        yield rtc.AudioFrame(b"\x0a\x00" * 240, 24000, 1, 240)
        started.set()
        await release.wait()
        yield rtc.AudioFrame(b"\x14\x00" * 240, 24000, 1, 240)

    speech = session.say("", audio=audio(), allow_interruptions=False)
    try:
        await asyncio.wait_for(started.wait(), 2)
        session.input.set_audio_enabled(False)
        session.input.set_audio_enabled(True)
        await asyncio.wait_for(owner.wait_for_ready(), 2)
        assert not speech.interrupted and session.output.audio is output
    finally:
        release.set()
    await asyncio.wait_for(speech.wait_for_playout(), 2)
    assert not speech.interrupted
    assert output.collected_pcm == b"\x0a\x00" * 240 + b"\x14\x00" * 240


@pytest.mark.asyncio
async def test_recovered_roomio_uses_new_real_bailian_stream_and_sdk_resampler(monkeypatch):
    monkeypatch.setenv("BAILIAN_STT_GATE_ENABLED", "false")
    monkeypatch.setenv("BAILIAN_STT_KEEPALIVE_INTERVAL_SEC", "0")
    transports, wire = [], []

    def from_track(*, track, **kwargs):
        value = Transport(track)
        transports.append(value)
        return value

    monkeypatch.setattr(rtc.AudioStream, "from_track", from_track)

    async def server(socket):
        request = json.loads(await socket.recv())
        task_id = request["header"]["task_id"]
        await socket.send(
            json.dumps({"header": {"event": "task-started", "task_id": task_id}, "payload": {}})
        )
        emitted = False
        async for pcm in socket:
            if not isinstance(pcm, bytes) or emitted:
                continue
            emitted = True
            wire.append(pcm)
            await socket.send(
                json.dumps(
                    {
                        "header": {"event": "result-generated", "task_id": task_id},
                        "payload": {
                            "output": {
                                "sentence": {
                                    "sentence_id": 1,
                                    "text": "合成测试",
                                    "sentence_end": True,
                                    "begin_time": 0,
                                    "end_time": len(pcm) // 2 * 1000 // 16000,
                                }
                            }
                        },
                    }
                )
            )

    async with websockets.serve(server, "127.0.0.1", 0) as service:
        port = service.sockets[0].getsockname()[1]
        provider = BailianFunASRSTT(
            api_url=f"ws://127.0.0.1:{port}",
            api_key="test",
            conn_options=APIConnectOptions(max_retry=0, timeout=2),
        )
        events = asyncio.Queue()

        class ObservedAgent(Agent):
            async def stt_node(self, audio, model_settings):
                async for event in super().stt_node(audio, model_settings):
                    events.put_nowait(event)
                    yield event

        session = AgentSession(aec_warmup_duration=0)
        session.output.audio = RecordingAudioOutput()
        await session.start(
            agent=ObservedAgent(
                instructions="", stt=provider, turn_handling={"turn_detection": "manual"}
            )
        )
        owner = EvidenceRoomInput(
            session=session,
            room=Room(),
            participant_identity="selected",
            session_id="session",
            endpoint_id="endpoint",
            bind_capture=lambda value: setattr(provider, "audio_evidence_capture", value),
            retire_candidate=lambda reason: None,
            audio_options=AudioInputOptions(
                sample_rate=24000, auto_gain_control=False, pre_connect_audio=False
            ),
        )
        try:
            await owner.start()
            first = owner.capture
            transports[-1].frames.put_nowait(rtc.AudioFrame(b"\x10\x27" * 4800, 24000, 1, 4800))
            old_span = transcript_evidence_from_speech_event(
                await asyncio.wait_for(events.get(), 2)
            ).audio_span
            assert first.snapshot(candidate="first", span=old_span).pcm == wire[0]
            session.input.set_audio_enabled(False)
            session.input.set_audio_enabled(True)
            await asyncio.wait_for(owner.wait_for_ready(), 2)
            transports[-1].frames.put_nowait(rtc.AudioFrame(b"\x20\x4e" * 4800, 24000, 1, 4800))
            span = transcript_evidence_from_speech_event(
                await asyncio.wait_for(events.get(), 2)
            ).audio_span
            evidence = owner.capture.snapshot(candidate="second", span=span)
            assert span.source != old_span.source and span.start == 0
            assert evidence.complete and evidence.sample_rate == 16000
            assert evidence.pcm == wire[1] and evidence.pcm != wire[0]
            assert not first.snapshot(candidate="late", span=old_span).complete
        finally:
            await owner.aclose()
            await session.aclose()


@pytest.mark.asyncio
async def test_close_while_recovery_pending_never_reopens_input(running):
    owner, session, room, recognizer, transports, retired, output = running
    old = owner.capture
    session.input.set_audio_enabled(False)
    session.input.set_audio_enabled(True)
    await owner.aclose()
    await asyncio.sleep(0)
    assert session.input.audio is None
    assert owner.capture is None and old.active_source is None
    assert all(value.closed for value in transports)
    assert session.output.audio is output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event",
    [
        "track_subscribed",
        "track_unsubscribed",
        "track_unpublished",
        "track_muted",
        "track_unmuted",
        "participant_disconnected",
        "reconnecting",
        "disconnected",
    ],
)
async def test_selected_room_event_invalidates_evidence_synchronously(running, event):
    owner, session, room, recognizer, transports, retired, output = running
    transports[-1].feed(11)
    capture, span, pcm = await next_audio(recognizer)
    if event in {"track_subscribed", "track_unsubscribed"}:
        args = (room.publication.track, room.publication, room.participant)
    elif event.startswith("track_"):
        args = (room.publication, room.participant)
    elif event == "participant_disconnected":
        args = (room.participant,)
    else:
        args = ()
    room.emit(event, *args)
    assert capture.active_source is None and capture.retained_samples == 0
    assert not capture.snapshot(candidate="late", span=span).complete
    assert session.output.audio is output


@pytest.mark.asyncio
async def test_unrelated_participant_and_camera_do_not_restart_input(running):
    owner, session, room, recognizer, transports, retired, output = running
    first = owner.capture
    other = SimpleNamespace(identity="other")
    camera = SimpleNamespace(source=rtc.TrackSource.SOURCE_CAMERA)
    room.emit("track_subscribed", object(), room.publication, other)
    room.emit("track_unpublished", camera, room.participant)
    room.emit("participant_disconnected", other)
    await asyncio.sleep(0)
    assert owner.capture is first and not retired and len(transports) == 1


@pytest.mark.asyncio
async def test_failed_old_input_cleanup_cannot_authorize_new_capture(running):
    owner, session, room, recognizer, transports, retired, output = running
    old = owner.capture
    transports[0].fail_close_once = True
    room.emit("track_unmuted", room.publication, room.participant)
    with pytest.raises(RuntimeError, match="failed"):
        await asyncio.wait_for(owner.wait_for_ready(), 2)
    assert len(transports) == 1 and not transports[0].closed
    assert old.active_source is None
    room.emit("track_unmuted", room.publication, room.participant)
    with pytest.raises(RuntimeError, match="failed"):
        await asyncio.wait_for(owner.wait_for_ready(), 2)
    assert len(transports) == 1
    await owner.aclose()  # shutdown may retry cleanup, but never reopen input
    assert transports[0].closed
