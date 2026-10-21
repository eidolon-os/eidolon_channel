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
from livekit.agents.voice.room_io import RoomOptions

from eidolon.channel_provider.http import create_app
from eidolon.channel_provider.tests.test_speaker_extraction_http import Engine, Service
from eidolon.livekit.agent.full_duplex.lifecycle import FullDuplexSessionLifecycle
from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session.speaker_extraction_client import SpeakerExtractionClient
from eidolon.livekit.agent.session.speaker_observation import SpeakerTurnObservation
from eidolon.livekit.common.audio_timeline import AudioSpan
from eidolon.livekit.common.transcript_evidence import TranscriptEvidence
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
async def test_session_installs_one_input_and_observes_normal_commit_through_provider(monkeypatch):
    transports = []

    def from_track(*, track, **kwargs):
        value = Transport(track)
        transports.append(value)
        return value

    monkeypatch.setattr(rtc.AudioStream, "from_track", from_track)
    engine = Engine()
    engine.ready.set()
    server = TestServer(create_app(service=Service(), bearer_token="test", speaker_engine=engine))
    await server.start_server()
    room, recognizer = Room(), MappedRecognizer()
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
        observation = SpeakerTurnObservation(extract=client.extract, record=records.append)
        pipeline._speaker_observation = observation
        try:
            await FullDuplexSessionLifecycle(pipeline)._prepare_session(
                session=session, agent=Agent(instructions="", stt=recognizer,
                    turn_handling={"turn_detection": "manual"}), input_room=room,
                options=RoomOptions(participant_identity="selected", audio_output=False,
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
            allowed = await pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
                turn_ctx=ChatContext.empty(), new_message=ChatMessage(role="user", content=["测试完整句子"]))
            assert allowed
            await asyncio.wait_for(asyncio.shield(observation.pending), 2)
            assert engine.calls == [b"".join(chunks)]
            assert records[-1].reason == "extracted"
            assert records[-1].candidate_id == owner.active.candidate_id
            assert records[-1].session_id == "session"
            assert session.output.audio is output
            assert len(transports) == 1
        finally:
            await observation.aclose()
            if getattr(pipeline, "_evidence_input", None):
                await pipeline._evidence_input.aclose()
            await session.aclose()
            await server.close()
    assert all(t.closed for t in transports)
    assert recognizer.audio_evidence_capture is None


@pytest.mark.asyncio
async def test_observation_rejects_unmapped_provider_before_media_start(monkeypatch):
    from eidolon.livekit.agent.session import room_io

    media = AsyncMock()
    monkeypatch.setattr(room_io, "start_room_session", media)
    pipeline = SimpleNamespace(_speaker_observation=object(), _factory=SimpleNamespace(
        stt=SimpleNamespace(stt=object()), runtime_session_id="session"))
    with pytest.raises(ValueError, match="mapped STT"):
        await FullDuplexSessionLifecycle(pipeline)._prepare_session(options=RoomOptions())
    media.assert_not_called()


def test_observation_cannot_start_legacy_independent_recorder():
    from eidolon.livekit.agent.full_duplex.pipeline import StreamingPipeline

    with pytest.raises(ValueError, match="legacy verification"):
        StreamingPipeline(SimpleNamespace(voiceprint_service=object()), speaker_observation=object())


@pytest.mark.asyncio
async def test_observation_does_not_turn_on_explicitly_disabled_microphone(monkeypatch):
    from eidolon.livekit.agent.session import room_io

    media = AsyncMock()
    monkeypatch.setattr(room_io, "start_room_session", media)
    pipeline = SimpleNamespace(_speaker_observation=object(), _factory=SimpleNamespace(
        stt=SimpleNamespace(stt=MappedRecognizer()), runtime_session_id="session"))
    with pytest.raises(ValueError, match="microphone input"):
        await FullDuplexSessionLifecycle(pipeline)._prepare_session(options=RoomOptions(audio_input=False))
    media.assert_not_called()
