"""Exercise the concrete assembly with real SDK lifecycle and mapped input."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from livekit import rtc
from livekit.agents.voice.room_io import RoomOptions, AudioInputOptions
from eidolon_sdk.biz.presentation import OutputSelection
from eidolon_sdk.biz.voiceprint import VoicePolicySnapshot, VoiceTemplate, ExtractedSample
from eidolon.livekit.agent.speaker_verified.session import SpeakerVerifiedSession
from eidolon.livekit.agent.speaker_verified.orchestration import SpeakerDecision
from eidolon.livekit.agent.turn_policy.intent_classifier import (
    InterruptIntent,
    IntentClassification,
)
from eidolon.livekit.common.audio_timeline import AudioSpan
from eidolon.livekit.common.transcript_evidence import TranscriptEvidence
from .test_speaker_verified_sdk_contract import EventRecognizer, user_texts
from .test_evidence_input_recovery import Room, Transport, next_audio
from .._harness.headless import RecordingAudioOutput
from .._harness.mocks.mock_llm import MockLLM
from .._harness.mocks.mock_tts import MockTTS
from .._harness.mocks.mock_vad import MockVAD


class MappedRecognizer(EventRecognizer):
    @property
    def audio_evidence_capture(self):
        return self.capture

    @audio_evidence_capture.setter
    def audio_evidence_capture(self, value):
        self.bind(value)


@pytest.fixture
async def assembled(monkeypatch, request):
    transports, commits = [], []

    def from_track(*, track, **kwargs):
        item = Transport(track)
        transports.append(item)
        return item

    monkeypatch.setattr(rtc.AudioStream, "from_track", from_track)

    # Output RoomIO RTC publication is synthetic; session, input RoomIO and
    # all admission wiring below are real. Test assertions exercise start().
    async def start_media(*, session, agent, options, **kwargs):
        assert options.audio_input is False
        assert options.participant_identity == "selected"
        await session.start(agent=agent)

    monkeypatch.setattr(
        "eidolon.livekit.agent.speaker_verified.session.start_room_session", start_media
    )
    stt, llm = MappedRecognizer(), MockLLM.echo()
    sample = ExtractedSample(
        model_id="test",
        weights_hash="a" * 64,
        preprocessing_version="test",
        vector=[1, 0],
        speech_ms=2000,
    )
    policy = VoicePolicySnapshot(
        owner_id="owner",
        companion_id="companion",
        mode="allowlist",
        revision=2,
        calibration_digest="synthetic-not-a-release",
        templates=[VoiceTemplate(profile_id="profile", revision=1, sample=sample)],
    )
    data = SimpleNamespace(
        get_voiceprint_snapshot=AsyncMock(return_value=policy),
        admit_voiceprint=AsyncMock(return_value=True),
    )
    factory = SimpleNamespace(
        stt=SimpleNamespace(stt=stt),
        llm=SimpleNamespace(llm=llm),
        tts=SimpleNamespace(tts=MockTTS()),
        vad=SimpleNamespace(vad=MockVAD.from_audio()),
        outputs=OutputSelection(speech=True),
        runtime_session_id="test",
        voiceprint_service=None,
        interrupt_classifier=SimpleNamespace(
            classify=AsyncMock(
                return_value=IntentClassification(
                    InterruptIntent.NORMAL_INTERRUPT, 0.9, "test", "test"
                )
            )
        ),
        runtime_services=SimpleNamespace(runtime=data),
    )
    verifying, release = asyncio.Event(), asyncio.Event()

    async def verify(evidence):
        assert evidence.complete and len(evidence.pcm) == 64000
        verifying.set()
        await release.wait()
        return SpeakerDecision("match", "profile", 1, "synthetic-not-a-release", policy_revision=2)

    eot = SimpleNamespace(
        score_candidate=lambda _: 0.99, unlikely_threshold=AsyncMock(return_value=0.5)
    )
    owner = SpeakerVerifiedSession(
        factory=factory,
        policy=policy,
        verify=verify,
        eot=eot,
        committed=commits.append,
        settle_sec=0.01,
        aec_warmup_duration=0,
    )
    owner.session.output.audio = RecordingAudioOutput(sample_rate=24000)
    room = Room()
    initial_input = getattr(request, "param", "available")
    if initial_input == "muted":
        room.publication.muted = True
    elif initial_input == "unpublished":
        room.participant.track_publications = {}

    async def client_received_session_started():
        # Mobile deliberately publishes/unmutes only after session_started.
        if initial_input != "available":
            assert transports == []
            assert owner._input.capture is None
            assert owner._input.state == "waiting"
            room.publication.muted = False
            room.participant.track_publications = {"mic-1": room.publication}
            if initial_input == "muted":
                room.emit("track_unmuted", room.publication, room.participant)
            else:
                room.emit("track_subscribed", room.publication.track, room.publication, room.participant)

    started = AsyncMock(side_effect=client_received_session_started)
    await asyncio.wait_for(owner.start(
        room=room,
        participant_identity="selected",
        on_started=started,
        options=RoomOptions(audio_input=AudioInputOptions(sample_rate=16000), text_output=False),
    ), 2)
    started.assert_awaited_once()
    await asyncio.wait_for(owner._input.wait_for_ready(), 2)
    transports[0].feed(100)
    await next_audio(stt)
    try:
        yield SimpleNamespace(
            owner=owner,
            stt=stt,
            llm=llm,
            commits=commits,
            data=data,
            transports=transports,
            verifying=verifying,
            release=release,
            room=room,
            factory=factory,
            policy=policy,
            eot=eot,
        )
    finally:
        await owner.aclose()
    assert stt.audio_evidence_capture is None
    assert all(t.closed for t in transports)


async def complete_candidate(r):
    first = last = None
    for _ in range(200):
        r.transports[-1].feed(5000)
        _, span, _ = await next_audio(r.stt)
        first = span if first is None else first
        last = span
    mapped = AudioSpan(first.source, first.generation, 16000, first.start, last.end)
    r.stt.emit(
        "完整的一句话",
        metadata=TranscriptEvidence(
            stream_key="asr", revision_key="one", sequence=1, audio_span=mapped
        ).as_metadata(),
    )
    for _ in range(12):
        r.transports[-1].feed(0)
        await next_audio(r.stt)
    await asyncio.wait_for(r.verifying.wait(), 2)


@pytest.mark.asyncio
async def test_assembled_candidate_uses_current_data_and_generates_once(assembled):
    r = assembled
    await complete_candidate(r)
    assert r.llm.call_count == 0
    r.release.set()
    await asyncio.wait_for(r.owner.turns.pending, 2)
    async with asyncio.timeout(2):
        while r.llm.call_count == 0:
            await asyncio.sleep(0)
    assert len(r.commits) == 1
    assert user_texts(r.llm.prompts[0]) == ["完整的一句话"]
    assert r.data.admit_voiceprint.await_count == 2
    r.data.admit_voiceprint.assert_awaited_with(
        "companion", 2, "profile", profile_revision=1, calibration_digest="synthetic-not-a-release"
    )
    assert len(r.transports) == 1


@pytest.mark.parametrize("vector,accepted", [([1, 0], True), ([0, 1], False)])
async def test_real_sdk_uses_shared_numerical_comparison_before_submission(
    assembled, vector, accepted
):
    from .test_speaker_verified_verifier import fixture as verifier_fixture
    from eidolon.livekit.agent.speaker_verified.verifier import CalibratedSpeakerVerifier

    r = assembled
    verifier, _, _, _, policy, sample = verifier_fixture()
    # Test-only frozen threshold; model output is synthetic, not accuracy evidence.
    r.data.get_voiceprint_snapshot.return_value = policy

    async def extract(pcm):
        assert len(pcm) == 64000
        r.verifying.set()
        await r.release.wait()
        return sample.model_copy(update={"vector": vector})

    extractor = SimpleNamespace(
        extract=AsyncMock(side_effect=extract),
        readiness=AsyncMock(
            return_value=SimpleNamespace(
                admission_ready=True, calibration_digest=verifier.calibration.digest
            )
        ),
    )
    r.owner.turns._verify = CalibratedSpeakerVerifier(
        data=r.data,
        extractor=extractor,
        calibration=verifier.calibration,
        owner_id="owner",
        companion_id="companion",
    )
    await complete_candidate(r)
    assert r.llm.call_count == 0
    r.release.set()
    await asyncio.wait_for(r.owner.turns.pending, 2)
    assert len(r.commits) == int(accepted)
    if accepted:
        async with asyncio.timeout(2):
            while r.llm.call_count == 0:
                await asyncio.sleep(0)
        assert user_texts(r.llm.prompts[0]) == ["完整的一句话"]
        assert r.data.admit_voiceprint.await_count == 2
    else:
        assert r.llm.call_count == 0
        r.data.admit_voiceprint.assert_not_awaited()


@pytest.mark.parametrize("replacement", [False, True])
async def test_sdk_cancel_does_not_flush_newer_playback(replacement):
    old, new = object(), object()
    owner = object.__new__(SpeakerVerifiedSession)
    owner.session = SimpleNamespace(current_speech=old)
    owner._stop_playback = AsyncMock()

    async def interrupt(**kwargs):
        owner.session.current_speech = new if replacement else None

    owner.session.interrupt = AsyncMock(side_effect=interrupt)
    await owner._cancel_output()
    owner.session.interrupt.assert_awaited_once_with(force=True)
    assert owner._stop_playback.await_count == (0 if replacement else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["stop", "close", "sdk_close", "disconnect"])
async def test_pending_identity_cannot_survive_explicit_stop_or_close(assembled, ending):
    r = assembled
    await complete_candidate(r)
    pending = r.owner.turns.pending
    if ending == "stop":
        await r.owner.stop_output()
    elif ending == "close":
        await r.owner.aclose()
    elif ending == "sdk_close":
        await r.owner.session.aclose()
    else:
        r.room.connected = False
        r.room.emit("disconnected", "test")
        async with asyncio.timeout(2):
            while not pending.done():
                await asyncio.sleep(0)
    r.release.set()
    await asyncio.gather(pending, return_exceptions=True)
    assert r.commits == [] and r.llm.call_count == 0
    r.data.admit_voiceprint.assert_not_called()


@pytest.mark.asyncio
async def test_revoke_between_extraction_and_action_prevents_reply(assembled):
    r = assembled
    await complete_candidate(r)
    r.data.get_voiceprint_snapshot.return_value = (
        r.data.get_voiceprint_snapshot.return_value.model_copy(
            update={"mode": "off", "revision": 3}
        )
    )
    r.release.set()
    await asyncio.wait_for(r.owner.turns.pending, 2)
    assert r.commits == [] and r.llm.call_count == 0
    assert r.owner.coordinator.active.reject_reason == "authorization_or_output_changed"


@pytest.mark.asyncio
async def test_explicit_text_retires_pending_voice_before_generation(assembled):
    r = assembled
    await complete_candidate(r)
    pending = r.owner.turns.pending
    await r.owner.accept_text(r.owner.session, SimpleNamespace(text="我手动输入的问题"))
    r.release.set()
    await asyncio.gather(pending, return_exceptions=True)
    async with asyncio.timeout(2):
        while not r.llm.call_count:
            await asyncio.sleep(0)
    assert r.commits == []
    assert user_texts(r.llm.prompts[0]) == ["我手动输入的问题"]
    r.data.admit_voiceprint.assert_not_called()


@pytest.mark.asyncio
async def test_start_failure_closes_media_and_does_not_play_welcome(assembled):
    r = assembled
    await r.owner.aclose()
    owner = SpeakerVerifiedSession(
        factory=r.factory,
        policy=r.policy,
        verify=AsyncMock(),
        eot=r.eot,
        committed=r.commits.append,
        welcome="准备好对话了",
        aec_warmup_duration=0,
    )
    owner.session.output.audio = RecordingAudioOutput(sample_rate=24000)
    with pytest.raises(RuntimeError, match="start acknowledgement failed"):
        await owner.start(
            room=r.room,
            participant_identity="selected",
            options=RoomOptions(audio_input=AudioInputOptions(sample_rate=16000)),
            on_started=AsyncMock(side_effect=RuntimeError("start acknowledgement failed")),
        )
    assert r.factory.tts.tts.synth_count == 0
    assert r.stt.audio_evidence_capture is None
    assert all(t.closed for t in r.transports)
    assert owner.turns.pending is None


@pytest.mark.asyncio
async def test_welcome_is_complete_after_ready_and_cannot_be_ambiently_interrupted(assembled):
    r = assembled
    await r.owner.aclose()
    owner = SpeakerVerifiedSession(
        factory=r.factory,
        policy=r.policy,
        verify=AsyncMock(),
        eot=r.eot,
        committed=r.commits.append,
        welcome="准备好对话了",
        aec_warmup_duration=0,
    )
    output = RecordingAudioOutput(sample_rate=24000)
    owner.session.output.audio = output

    async def started():
        assert r.factory.tts.tts.synth_count == 0

    try:
        await owner.start(
            room=r.room,
            participant_identity="selected",
            options=RoomOptions(audio_input=AudioInputOptions(sample_rate=16000)),
            on_started=started,
        )
        async with asyncio.timeout(2):
            while r.factory.tts.tts.synth_count == 0 or not output.collected_pcm:
                await asyncio.sleep(0)
        speech = owner.session.current_speech
        if speech is not None:
            assert not speech.allow_interruptions
            await asyncio.wait_for(speech.wait_for_playout(), 2)
        assert output.collected_pcm
        assert not any(segment.cleared for segment in output.segments)
        assert r.llm.call_count == 0
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_unconfirmed_session_cannot_release_voice_or_typed_generation(assembled):
    r = assembled
    await r.owner.aclose()
    verify = AsyncMock()
    owner = SpeakerVerifiedSession(
        factory=r.factory,
        policy=r.policy,
        verify=verify,
        eot=r.eot,
        committed=r.commits.append,
        aec_warmup_duration=0,
    )
    owner.session.output.audio = RecordingAudioOutput(sample_rate=24000)
    pending_ack, release = asyncio.Event(), asyncio.Event()

    async def acknowledge():
        pending_ack.set()
        await release.wait()

    startup = asyncio.create_task(
        owner.start(
            room=r.room,
            participant_identity="selected",
            options=RoomOptions(audio_input=AudioInputOptions(sample_rate=16000)),
            on_started=acknowledge,
        )
    )
    try:
        await asyncio.wait_for(pending_ack.wait(), 2)
        owner._user_state_changed(SimpleNamespace(old_state="listening", new_state="speaking"))
        await owner.accept_text(owner.session, SimpleNamespace(text="未确认的输入"))
        assert owner.coordinator.active is None
        verify.assert_not_called()
        assert r.llm.call_count == 0
        release.set()
        await asyncio.wait_for(startup, 2)
    finally:
        release.set()
        await asyncio.gather(startup, return_exceptions=True)
        await owner.aclose()


async def test_maximum_utterance_survives_final_transcript_delay(assembled):
    from eidolon.livekit.common.audio_timeline import AudioSource
    r = assembled
    capture = r.stt.audio_evidence_capture
    assert capture.retention_seconds == r.owner.input._timeout
    assert 30 < capture.retention_seconds <= 60
    source = AudioSource('long-candidate', 1, 16000)
    capture.select_stream(source.source)
    capture.begin_run(source)
    pcm = b'\1\0' * (16000 * 30)
    capture.append(source, start=0, pcm=pcm)
    capture.append(source, start=16000 * 30, pcm=b'\0\0' * (16000 * 2))
    evidence = capture.snapshot(candidate='long', span=AudioSpan(source.source, 1, 16000, 0, 16000 * 30))
    assert evidence.complete and evidence.pcm == pcm


@pytest.mark.parametrize("assembled", ["muted", "unpublished"], indirect=True)
async def test_mobile_waiting_for_acceptance_can_publish_after_session_started(assembled):
    r = assembled
    assert r.owner._input.state == "ready"
    assert len(r.transports) == 1
    assert r.owner._input.capture is not None
    assert r.commits == []  # A ready transport is never an identity grant.
