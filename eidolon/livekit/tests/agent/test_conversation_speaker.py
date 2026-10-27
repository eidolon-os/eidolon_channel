import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import pytest
from eidolon_sdk.biz.person import PersonProfile
from eidolon_sdk.biz.speaker_identity import (
    SpeakerCandidates,
    SpeakerCandidate,
    SpeakerRecognitionPolicy,
)
from eidolon_sdk.biz.voiceprint import (
    ExtractedSample,
    VoiceTemplate,
    SpeakerComparisonResult,
    SpeakerSimilarity,
)
from eidolon_sdk.biz.speaker_attestation import verify_observation
from eidolon.livekit.common.audio_evidence import AudioEvidence
from eidolon.livekit.agent.session.speaker_recognition import SpeakerRecognizer
from eidolon.livekit.agent.session.conversation_speaker import ConversationSpeaker

SAMPLE = ExtractedSample(
    model_id="m", weights_hash="a" * 64, preprocessing_version="v", vector=[1, 0], speech_ms=2000
)
POLICY = SpeakerRecognitionPolicy(
    version="trial",
    model_id="m",
    weights_hash="a" * 64,
    preprocessing_version="v",
    threshold=0.4,
    min_person_margin=0.1,
    min_speech_ms=1500,
)


def evidence():
    return AudioEvidence(
        source="mic",
        generation=1,
        candidate="c",
        sample_rate=16000,
        start=0,
        end=32000,
        complete=True,
        reason=None,
        pcm=b"\x01\x00" * 32000,
    )


def snapshot(people=1, profiles=1):
    return SpeakerCandidates(
        owner_id="o",
        revision="b" * 64,
        persons=[
            SpeakerCandidate(
                person_id=f"p{i}",
                person_revision=1,
                display_name=f"Person {i}",
                profile=PersonProfile(),
                templates=[
                    VoiceTemplate(profile_id=f"v{i}-{j}", revision=1, sample=SAMPLE)
                    for j in range(profiles)
                ],
            )
            for i in range(people)
        ],
    )


def client(people=1, profiles=1, score=0.8):
    async def compare(probe, templates):
        return SpeakerComparisonResult(
            model_id=probe.model_id,
            weights_hash=probe.weights_hash,
            preprocessing_version=probe.preprocessing_version,
            scores=[
                SpeakerSimilarity(
                    profile_id=t.profile_id, revision=t.revision, status="scored", similarity=score
                )
                for t in templates
            ],
        )

    return SimpleNamespace(
        candidates=AsyncMock(return_value=snapshot(people, profiles)),
        extract=AsyncMock(return_value=SAMPLE),
        compare=AsyncMock(side_effect=compare),
    )


@pytest.mark.asyncio
async def test_identification_uses_authority_and_aggregates_same_person_profiles():
    port = client(profiles=2)
    result = await SpeakerRecognizer(port, POLICY).recognize(
        evidence(), owner_id="o", session_id="s", turn_id="t"
    )
    assert result.status == "identified" and result.person_id == "p0"
    assert result.confidence == "controlled_trial"
    assert port.extract.await_count == 1 and port.candidates.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "people,score,status", [(2, 0.8, "uncertain"), (1, 0.1, "unknown"), (0, 0.8, "uncertain")]
)
async def test_no_guess_for_ambiguity_unknown_or_empty(people, score, status):
    port = client(people=people, score=score)
    result = await SpeakerRecognizer(port, POLICY).recognize(
        evidence(), owner_id="o", session_id="s", turn_id="t"
    )
    assert result.status == status and result.person_id is None
    if people == 0:
        port.extract.assert_not_awaited()


@pytest.mark.asyncio
async def test_revision_change_discards_identity():
    port = client()
    port.candidates.side_effect = [snapshot(), snapshot().model_copy(update={"revision": "c" * 64})]
    result = await SpeakerRecognizer(port, POLICY).recognize(
        evidence(), owner_id="o", session_id="s", turn_id="t"
    )
    assert result.person_id is None and result.reason == "candidates_changed"


@pytest.mark.asyncio
async def test_deadline_is_bounded_and_does_not_leak_into_next_turn():
    port = client()
    cancelled = asyncio.Event()

    async def slow(_):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()

    port.extract.side_effect = slow
    observer = ConversationSpeaker(
        recognizer=SpeakerRecognizer(port, POLICY),
        owner_id="o",
        session_id="s",
        secret="x" * 32,
        publish=MagicMock(),
        wait_ms=10,
    )
    observer.begin_evidence(evidence(), "t")
    start = time.monotonic()
    result = await observer.result("t")
    assert (
        time.monotonic() - start < 0.2
        and result.reason == "reply_deadline"
        and result.person_id is None
    )
    await asyncio.wait_for(cancelled.wait(), 0.2)
    await observer.aclose()


@pytest.mark.asyncio
async def test_signed_delivery_bound_to_exact_turn_and_text():
    observer = ConversationSpeaker(
        recognizer=SpeakerRecognizer(client(), POLICY),
        owner_id="o",
        session_id="s",
        secret="x" * 32,
        publish=MagicMock(),
        wait_ms=100,
    )
    observer.begin_evidence(evidence(), "t")
    result = await observer.result("t")
    llm = MagicMock()
    observer.deliver(result, text="hello", llm=llm, timeline=None)
    token = llm.set_speaker_observation.call_args.args[0]
    accepted = verify_observation(
        token, secret="x" * 32, owner_id="o", session_id="s", turn_id="t", text="hello"
    )
    assert accepted.person_id == "p0"
    for changed in (
        {"text": "another"},
        {"turn_id": "next"},
        {"owner_id": "other"},
        {"session_id": "next"},
    ):
        args = (
            dict(secret="x" * 32, owner_id="o", session_id="s", turn_id="t", text="hello") | changed
        )
        with pytest.raises(ValueError):
            verify_observation(token, **args)
    await observer.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("score,expected", [(0.8, "identified"), (0.1, "unknown")])
async def test_product_completed_turn_delivers_exact_audio_and_signed_context(score, expected):
    from livekit.agents.llm import ChatContext, ChatMessage
    from eidolon.livekit.common.audio_evidence import AudioEvidenceCapture
    from eidolon.livekit.common.audio_timeline import AudioSource, AudioSpan
    from eidolon.livekit.common.transcript_evidence import TranscriptEvidence
    from eidolon.livekit.agent.observability import TurnTimeline
    from .test_commit_guard import _make_pipeline_with_session

    port = client(score=score)
    published = []
    observer = ConversationSpeaker(
        recognizer=SpeakerRecognizer(port, POLICY),
        owner_id="o",
        session_id="s",
        secret="x" * 32,
        publish=published.append,
        wait_ms=100,
    )
    capture = AudioEvidenceCapture(session_id="s", endpoint_id="mic")
    source = AudioSource("mic", 1, 16000)
    capture.select_stream("mic")
    capture.begin_run(source)
    capture.append(source, start=0, pcm=evidence().pcm)
    observer.bind_capture(capture)
    pipeline = _make_pipeline_with_session()
    pipeline._ensure_runtime_defaults()
    pipeline._speaker_observation = observer
    timeline = TurnTimeline("t")
    pipeline._timeline = timeline
    pipeline._factory.llm.llm.set_speaker_observation = MagicMock()
    owner = pipeline._user_turns
    owner.start_speech(timeline=timeline)
    owner.add_transcript(
        "hello",
        is_final=True,
        evidence=TranscriptEvidence(
            stream_key="asr", revision_key="final", audio_span=AudioSpan("mic", 1, 16000, 0, 32000)
        ),
    )
    owner.note_speech_stopped(eot_score=None)
    try:
        assert await pipeline._ensure_turn_completion().allows_completed_turn(
            turn_ctx=ChatContext.empty(), new_message=ChatMessage(role="user", content=["hello"])
        )
        await observer.finish(
            owner, turn_id="t", text="hello", llm=pipeline._factory.llm.llm, timeline=timeline
        )
        assert len(published) == 1 and published[0].status == expected
        port.extract.assert_awaited_once_with(evidence().pcm)
        token = pipeline._factory.llm.llm.set_speaker_observation.call_args.args[0]
        assert (
            verify_observation(
                token, secret="x" * 32, owner_id="o", session_id="s", turn_id="t", text="hello"
            )
            == published[0]
        )
    finally:
        await observer.aclose()


@pytest.mark.asyncio
async def test_existing_provider_http_and_pure_scorer_are_reused():
    import httpx
    from aiohttp.test_utils import TestServer
    from eidolon.channel_provider.http import create_app
    from eidolon.channel_provider.tests.test_speaker_extraction_http import Engine, Service
    from eidolon.livekit.agent.session.speaker_recognition import SpeakerRecognitionClient

    engine = Engine()
    engine.ready.set()
    server = TestServer(create_app(service=Service(), bearer_token="test", speaker_engine=engine))
    await server.start_server()
    try:
        async with httpx.AsyncClient(trust_env=False) as http:
            port = SpeakerRecognitionClient(
                http,
                data_url="http://unused",
                data_token="test",
                provider_url=str(server.make_url("")).rstrip("/"),
                provider_token="test",
            )
            sample = await port.extract(evidence().pcm)
            compared = await port.compare(
                sample, [VoiceTemplate(profile_id="p", revision=1, sample=sample)]
            )
            assert compared.scores[0].similarity == pytest.approx(1)
            assert engine.calls == [evidence().pcm]
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_no_policy_and_short_audio_do_not_call_dependencies():
    port = client()
    result = await SpeakerRecognizer(port, None).recognize(
        evidence(), owner_id="o", session_id="s", turn_id="t"
    )
    assert result.status == "unavailable" and result.person_id is None
    port.candidates.assert_not_awaited()
    port.extract.assert_not_awaited()
