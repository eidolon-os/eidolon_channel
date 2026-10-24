"""Real SDK scheduling boundary; only RTC, recognition and generation are synthetic."""

import asyncio
from types import SimpleNamespace

import pytest
from livekit import rtc
from livekit.agents import AgentSession, stt
from livekit.agents.voice.room_io import AudioInputOptions
from eidolon_sdk.biz.presentation import OutputSelection

from eidolon.livekit.agent.session.evidence_room_input import EvidenceRoomInput
from eidolon.livekit.agent.speaker_verified.agent import (
    SpeakerVerifiedAgent,
    speaker_verified_turn_handling,
)
from .._harness.headless import RecordingAudioOutput
from .._harness.mocks.mock_llm import MockLLM
from .._harness.mocks.mock_tts import MockTTS
from .._harness.mocks.mock_vad import MockVAD
from .test_evidence_input_recovery import Recognizer, Room, Transport, next_audio


class EventRecognizer(Recognizer):
    """Add controllable provider results to the existing mapped PCM recognizer."""

    def stream(self, **kwargs):
        stream = super().stream(**kwargs)
        self.current_stream = stream
        return stream

    def emit(self, text, *, final=True, metadata=None):
        self.current_stream._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT
                if final
                else stt.SpeechEventType.INTERIM_TRANSCRIPT,
                alternatives=[stt.SpeechData(language="zh", text=text, metadata=metadata)],
            )
        )


def user_texts(context):
    return [m.text_content for m in context.items if m.type == "message" and m.role == "user"]


@pytest.fixture
async def running(monkeypatch):
    transports = []

    def from_track(*, track, **kwargs):
        transport = Transport(track)
        transports.append(transport)
        return transport

    monkeypatch.setattr(rtc.AudioStream, "from_track", from_track)
    recognizer, llm = EventRecognizer(), MockLLM.echo()
    observed, public_transcripts, listeners = asyncio.Queue(), [], []

    def observe(event):
        observed.put_nowait(event)
        for listener in listeners:
            listener(event)

    session = AgentSession(
        aec_warmup_duration=0,
        session_close_transcript_timeout=0,
        turn_handling=speaker_verified_turn_handling(),
    )
    output = RecordingAudioOutput(sample_rate=24000)
    session.output.audio = output
    agent = SpeakerVerifiedAgent(
        instructions="",
        stt=recognizer,
        llm=llm,
        tts=MockTTS(),
        vad=MockVAD.from_audio(),
        outputs=OutputSelection(speech=True, dialogue_text=True),
        observe_speech=observe,
    )
    session.on("user_input_transcribed", public_transcripts.append)
    await session.start(agent=agent)
    owner = EvidenceRoomInput(
        session=session,
        room=Room(),
        participant_identity="selected",
        session_id="test",
        endpoint_id="input",
        bind_capture=recognizer.bind,
        retire_candidate=lambda reason: None,
        audio_options=AudioInputOptions(
            sample_rate=16000, auto_gain_control=False, pre_connect_audio=False
        ),
    )
    await owner.start()
    transports[0].feed(100)
    await next_audio(recognizer)
    state = SimpleNamespace(
        session=session,
        agent=agent,
        recognizer=recognizer,
        llm=llm,
        output=output,
        observed=observed,
        public_transcripts=public_transcripts,
        owner=owner,
        transports=transports,
        listeners=listeners,
    )
    try:
        yield state
    finally:
        await owner.aclose()
        await session.aclose()


@pytest.mark.asyncio
async def test_unverified_transcripts_never_enter_sdk_history_or_interrupt_playback(running):
    r = running
    chunks, supplied = asyncio.Queue(), []

    async def audio():
        while (value := await chunks.get()) is not None:
            pcm = value.to_bytes(2, "little") * 240
            supplied.append(pcm)
            yield rtc.AudioFrame(pcm, 24000, 1, 240)

    speech = r.session.say("", audio=audio(), allow_interruptions=False)
    for value in range(1, 5):
        chunks.put_nowait(value)
        r.recognizer.emit(f"未授权背景片段{value}")
        await asyncio.wait_for(r.observed.get(), 2)
        r.transports[0].feed(321)
        _, _, pcm = await next_audio(r.recognizer)
        # The SDK must retain microphone audio while speech is uninterruptible.
        assert pcm == b"\x41\x01" * 160
        assert not speech.interrupted
    chunks.put_nowait(None)
    await asyncio.wait_for(speech.wait_for_playout(), 2)
    assert r.output.collected_pcm == b"".join(supplied)
    assert not any(segment.cleared for segment in r.output.segments)
    assert r.llm.call_count == 0
    assert user_texts(r.agent.chat_ctx) == []
    assert r.public_transcripts == []
    assert len(r.transports) == 1
    await r.session.aclose()
    # SDK 1.7 may append an empty close marker; no untrusted content is admitted.
    assert not any(user_texts(r.agent.chat_ctx))


@pytest.mark.asyncio
async def test_explicit_approved_reply_has_no_rejected_or_late_transcript(running):
    r = running
    streams = r.recognizer.streams
    r.recognizer.emit("电视内容")
    await asyncio.wait_for(r.observed.get(), 2)
    approved = r.session.generate_reply(user_input="已批准的完整句子", input_modality="audio")
    await asyncio.wait_for(approved, 2)
    assert approved.exception() is None
    r.recognizer.emit("电视迟到修订")
    await asyncio.wait_for(r.observed.get(), 2)
    assert r.llm.call_count == 1
    assert user_texts(r.llm.prompts[0]) == ["已批准的完整句子"]
    assert user_texts(r.agent.chat_ctx) == ["已批准的完整句子"]
    assert r.public_transcripts == []
    assert r.recognizer.streams == streams  # no clear/reset of STT per rejected turn


@pytest.mark.asyncio
async def test_explicit_cancel_then_reply_avoids_uninterruptible_completion_guard(running):
    r = running
    started = asyncio.Event()

    async def audio():
        yield rtc.AudioFrame(b"\x01\x00" * 240, 24000, 1, 240)
        started.set()
        await asyncio.Event().wait()

    old = r.session.say("", audio=audio(), allow_interruptions=False)
    await asyncio.wait_for(started.wait(), 2)
    await asyncio.wait_for(r.session.interrupt(force=True), 2)
    reply = r.session.generate_reply(user_input="许可后的插话", input_modality="audio")
    await asyncio.wait_for(reply, 2)
    assert reply.exception() is None
    assert old.interrupted
    assert r.llm.call_count == 1
    assert user_texts(r.agent.chat_ctx) == ["许可后的插话"]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["match", "non_match", "unavailable"])
async def test_streaming_vad_eot_and_verification_release_one_reply(running, monkeypatch, outcome):
    from unittest.mock import AsyncMock
    from livekit.agents.llm import ChatContext
    from eidolon.livekit.agent.session.user_turn_coordinator import UserTurnCoordinator
    from eidolon.livekit.agent.speaker_verified.input import SpeakerVerifiedInput
    from eidolon.livekit.agent.speaker_verified.orchestration import (
        SpeakerDecision,
        SpeakerVerifiedTurns,
    )
    from eidolon.livekit.common.audio_timeline import AudioSpan
    from eidolon.livekit.common.transcript_evidence import TranscriptEvidence
    from eidolon.livekit.plugins.eot import ChineseModel

    r = running
    model = ChineseModel()
    # Use the existing EOT adapter; only its numerical inference is synthetic.
    monkeypatch.setattr(model._context_eot._base_eot, "p_complete_score", lambda text: 0.99)
    completions, commits = [], []

    async def is_complete(text):
        ctx = ChatContext.empty()
        ctx.add_message(role="user", content=text)
        completions.append(text)
        return await model.predict_end_of_turn(ctx) >= await model.unlikely_threshold("zh")

    verifying, allow = asyncio.Event(), asyncio.Event()

    async def verify(evidence):
        assert evidence.complete
        assert evidence.pcm == b"\x88\x13" * 32000
        verifying.set()
        await allow.wait()
        return SpeakerDecision(outcome, "profile", 1, "test-only")

    owner = UserTurnCoordinator(speech_merge_grace_sec=0.01)
    turns = SpeakerVerifiedTurns(
        coordinator=owner,
        capture=lambda: r.owner.capture,
        session=r.session,
        verify=verify,
        is_complete=is_complete,
        classify=AsyncMock(return_value="reply"),
        authorize=AsyncMock(return_value=True),
        committed=commits.append,
    )
    routing = SpeakerVerifiedInput(turns, coordinator=owner, settle_sec=0.01)
    r.listeners.append(routing.observe_speech)
    r.session.on("user_state_changed", routing.user_state_changed)
    started = asyncio.Event()

    async def old_audio():
        yield rtc.AudioFrame(b"\x01\x00" * 240, 24000, 1, 240)
        started.set()
        await asyncio.Event().wait()

    speech = r.session.say("", audio=old_audio(), allow_interruptions=False)
    try:
        await asyncio.wait_for(started.wait(), 2)
        first = last = None
        for _ in range(200):
            r.transports[0].feed(5000)
            _, span, _ = await next_audio(r.recognizer)
            first = span if first is None else first
            last = span
        mapped = AudioSpan(first.source, first.generation, 16000, first.start, last.end)
        r.recognizer.emit(
            "请听我说完这一整句话",
            metadata=TranscriptEvidence(
                stream_key="asr", revision_key="final-1", sequence=1, audio_span=mapped
            ).as_metadata(),
        )
        await asyncio.wait_for(r.observed.get(), 2)
        for _ in range(12):
            r.transports[0].feed(0)
            await next_audio(r.recognizer)
        await asyncio.wait_for(verifying.wait(), 2)
        assert not speech.interrupted
        assert r.llm.call_count == 0 and user_texts(r.agent.chat_ctx) == []
        allow.set()
        task = turns.pending
        await asyncio.wait_for(task, 2)
        if outcome == "match":
            async with asyncio.timeout(2):
                while r.llm.call_count == 0:
                    await asyncio.sleep(0)
            assert speech.interrupted
            assert len(commits) == 1
            assert user_texts(r.llm.prompts[0]) == ["请听我说完这一整句话"]
        else:
            assert not speech.interrupted
            assert r.llm.call_count == 0 and commits == []
            assert not user_texts(r.agent.chat_ctx)
        assert completions == ["请听我说完这一整句话"]
        assert len(r.transports) == 1
    finally:
        await routing.aclose()
        r.session.off("user_state_changed", routing.user_state_changed)
        r.listeners.remove(routing.observe_speech)
