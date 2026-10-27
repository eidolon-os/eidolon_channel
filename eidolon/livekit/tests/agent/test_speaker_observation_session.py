"""Actual SDK input owner, product commit hook and authenticated Provider HTTP.

Only RTC transport, ASR transcription and model inference are synthetic.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from aiohttp.test_utils import TestServer
from livekit import rtc
from livekit.agents import Agent, AgentSession
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.voice.room_io import AudioInputOptions, RoomOptions

from eidolon.channel_provider.http import create_app
from eidolon.channel_provider.tests.test_speaker_extraction_http import Engine, Service
from eidolon.livekit.agent.full_duplex.lifecycle import FullDuplexSessionLifecycle
from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session.speaker_extraction_client import SpeakerExtractionClient
from eidolon.livekit.agent.session.speaker_observation import SpeakerTurnObservation
from eidolon.livekit.common.audio_timeline import AudioSpan
from eidolon.livekit.common.transcript_evidence import TranscriptEvidence
from eidolon.livekit.plugins.speaker_verification.inference import InferenceBusy, ModelUnavailable
from .._harness.headless import RecordingAudioOutput
from .test_commit_guard import _make_pipeline_with_session
from .test_evidence_input_recovery import Recognizer, Room, Transport, next_audio


class MappedRecognizer(Recognizer):
    @property
    def audio_evidence_capture(self):
        return self.capture

    @audio_evidence_capture.setter
    def audio_evidence_capture(self, value):
        self.bind(value)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["extracted", "provider_busy", "provider_unavailable", "timeout", "expired"])
async def test_observation_outcome_preserves_session_input_commit_and_playout(monkeypatch, outcome):
    transports = []

    def from_track(*, track, **kwargs):
        value = Transport(track)
        transports.append(value)
        return value

    monkeypatch.setattr(rtc.AudioStream, "from_track", from_track)
    engine = Engine()
    engine.ready.set()
    extraction_started, release_extraction = asyncio.Event(), asyncio.Event()
    original_extract = engine.extract

    async def extract(pcm, min_speech_ms=1500):
        sample = await original_extract(pcm, min_speech_ms=min_speech_ms)
        extraction_started.set()
        if outcome == "provider_busy":
            raise InferenceBusy()
        if outcome == "provider_unavailable":
            raise ModelUnavailable()
        if outcome in {"timeout", "expired"}:
            await release_extraction.wait()
        return sample

    engine.extract = extract
    server = TestServer(create_app(service=Service(), bearer_token="test", speaker_engine=engine))
    await server.start_server()
    room, recognizer = Room(), MappedRecognizer()
    room.local_participant.sid = "agent-sid"
    room.local_participant.identity = "agent"
    room.local_participant.kind = rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
    room.participant.sid = "user-sid"
    room.register_byte_stream_handler = MagicMock()
    room.unregister_byte_stream_handler = MagicMock()
    room.local_participant.stream_bytes = AsyncMock(return_value=SimpleNamespace(
        write=AsyncMock(), aclose=AsyncMock()))
    records = []
    session = AgentSession(aec_warmup_duration=0, session_close_transcript_timeout=0)
    output = RecordingAudioOutput(sample_rate=24000)
    session.output.audio = output
    pipeline = _make_pipeline_with_session()
    pipeline._ensure_runtime_defaults()
    pipeline._factory.stt = SimpleNamespace(stt=recognizer)
    pipeline._factory.runtime_session_id = "session"
    pipeline._filler = None
    pipeline._warmup_stages = AsyncMock()
    pipeline.session_mark = MagicMock()
    async with httpx.AsyncClient(trust_env=False) as http:
        client = SpeakerExtractionClient(http, str(server.make_url("")), token="test")
        observation = SpeakerTurnObservation(extract=client.extract, record=records.append,
            timeout_sec=0.1 if outcome == "timeout" else 8,
            lifetime_sec=2 if outcome == "expired" else None)
        pipeline._speaker_observation = observation
        try:
            await FullDuplexSessionLifecycle(pipeline)._prepare_session(
                session=session, agent=Agent(instructions="", stt=recognizer,
                    turn_handling={"turn_detection": "manual"}), input_room=room,
                options=RoomOptions(participant_identity="selected", audio_output=False,
                    # Deterministic sample markers; SDK AGC changes constant PCM.
                    audio_input=AudioInputOptions(sample_rate=16000, auto_gain_control=False),
                    text_input=False, text_output=False, video_input=False))
            assert len(transports) == 1
            assert session.output.audio is output
            for _ in range(160):
                transports[0].feed(100)
            chunks = []
            while sum(len(pcm) for pcm in chunks) < 48000:
                capture, span, pcm = await next_audio(recognizer)
                assert capture is pipeline._evidence_input.capture
                chunks.append(pcm)
            full_span = AudioSpan(span.source, span.generation, 16000, 0, span.end)
            owner = pipeline._user_turns
            pipeline._timeline = TurnTimeline("observed-turn")
            owner.start_speech(timeline=pipeline._timeline)
            owner.add_transcript("测试完整句子", is_final=True, evidence=TranscriptEvidence(
                stream_key="asr", revision_key="final", audio_span=full_span))
            owner.note_speech_stopped(eot_score=None)
            allowed = await pipeline._ensure_turn_completion().allows_completed_turn(
                turn_ctx=ChatContext.empty(), new_message=ChatMessage(role="user", content=["测试完整句子"]))
            assert allowed
            pending = observation.pending
            await asyncio.wait_for(extraction_started.wait(), 1)
            playback_started, release_playback = asyncio.Event(), asyncio.Event()

            async def audio():
                yield rtc.AudioFrame(b"\x0a\x00" * 240, 24000, 1, 240)
                playback_started.set()
                await release_playback.wait()
                yield rtc.AudioFrame(b"\x14\x00" * 240, 24000, 1, 240)

            speech = session.say("", audio=audio(), allow_interruptions=False)
            try:
                await asyncio.wait_for(playback_started.wait(), 1)
                if outcome == "expired":
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.wait_for(asyncio.shield(pending), 3)
                    assert records == []
                    assert capture.retained_samples == 0
                    assert capture.active_source is None
                else:
                    await asyncio.wait_for(asyncio.shield(pending), 1)
                    assert records[-1].reason == outcome
                    assert records[-1].candidate_id == owner.active.candidate_id
                    assert records[-1].session_id == "session"
                assert not speech.interrupted
            finally:
                release_playback.set()
            await asyncio.wait_for(speech.wait_for_playout(), 1)
            assert output.collected_pcm == b"\x0a\x00" * 240 + b"\x14\x00" * 240
            assert engine.calls == [b"".join(chunks)]

            # The same microphone/ASR stream and next product turn remain usable.
            transports[0].feed(321)
            async with asyncio.timeout(1):
                while True:
                    next_capture, next_span, pcm = await next_audio(recognizer)
                    if pcm == (321).to_bytes(2, "little") * 160:
                        break
            assert next_capture is capture
            owner.start_speech(timeline=TurnTimeline("following-turn"))
            owner.add_transcript("下一句话", is_final=True, evidence=TranscriptEvidence(
                stream_key="asr", revision_key="next-final", audio_span=next_span))
            owner.note_speech_stopped(eot_score=None)
            assert await pipeline._ensure_turn_completion().allows_completed_turn(
                turn_ctx=ChatContext.empty(), new_message=ChatMessage(role="user", content=["下一句话"]))
            assert len(engine.calls) == 1
            if outcome == "expired":
                assert capture.retained_samples == 0
                assert records == []
            assert session.output.audio is output
            assert len(transports) == 1
        finally:
            release_extraction.set()
            await observation.aclose()
            if getattr(pipeline, "_evidence_input", None):
                await pipeline._evidence_input.aclose()
            await session.aclose()
            await server.close()
    assert all(t.closed for t in transports)
    assert recognizer.audio_evidence_capture is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mapped,microphone", [(False, True), (True, False)])
async def test_optional_observation_preserves_unsupported_audio_paths(monkeypatch, mapped, microphone):
    from eidolon.livekit.agent.session import room_io

    media = AsyncMock()
    monkeypatch.setattr(room_io, "start_room_session", media)
    pipeline = SimpleNamespace(_speaker_observation=object(), _factory=SimpleNamespace(
        stt=SimpleNamespace(stt=MappedRecognizer() if mapped else object()), runtime_session_id="session"),
        _warmup_stages=AsyncMock(), _filler=None, session_mark=MagicMock())
    options = RoomOptions(audio_input=microphone)
    await FullDuplexSessionLifecycle(pipeline)._prepare_session(options=options)
    assert media.call_args.kwargs["options"] is options
    assert not hasattr(pipeline, "_evidence_input")
