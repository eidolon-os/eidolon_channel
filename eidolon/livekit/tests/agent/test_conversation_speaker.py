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
    model_id="m", weights_hash="a" * 64, preprocessing_version="v", vector=[1, 0], speech_ms=2000, quality={"overlap_detected": 0}
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
        record_candidate_evidence=AsyncMock(),
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
async def test_pending_result_is_frozen_and_late_feedback_never_changes_llm():
    port = client()
    gate = asyncio.Event()
    async def slow(_, **kwargs):
        await gate.wait()
        return SAMPLE
    port.extract.side_effect = slow
    published = []
    observer = ConversationSpeaker(recognizer=SpeakerRecognizer(port, POLICY),
        owner_id="o", session_id="s", secret="x" * 32, publish=published.append)
    try:
        observer.begin_evidence(evidence(), "t")
        task = observer._task
        result = await observer.result("t")
        assert result.reason == "recognition_pending" and not task.done()
        llm = MagicMock()
        observer.deliver(result, text="hello", llm=llm, timeline=None)
        gate.set()
        await task
        await asyncio.sleep(0)
        assert published[-1].status == "identified"
        assert await observer.result("t") == result
        observer.deliver(result, text="hello", llm=llm, timeline=None)
        assert published[-1].status == "identified"
        assert llm.set_speaker_observation.call_count == 1
        wrong = await observer.result("next")
        assert wrong.person_id is None
    finally:
        await observer.aclose()


@pytest.mark.asyncio
async def test_signed_delivery_bound_to_exact_turn_and_text():
    observer = ConversationSpeaker(
        recognizer=SpeakerRecognizer(client(), POLICY),
        owner_id="o",
        session_id="s",
        secret="x" * 32,
        publish=MagicMock(),
    )
    observer.begin_evidence(evidence(), "t")
    await observer._task
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
        observer.begin(owner, "t")
        await observer._task
        await observer.finish(
            owner, turn_id="t", text="hello", llm=pipeline._factory.llm.llm, timeline=timeline
        )
        assert len(published) == 1 and published[0].status == expected
        port.extract.assert_awaited_once_with(evidence().pcm, min_speech_ms=1500)
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
            sample = await port.extract(evidence().pcm, min_speech_ms=1500)
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

@pytest.mark.asyncio
async def test_final_transcript_starts_recognition_before_framework_completion():
    from .test_commit_guard import _make_pipeline_with_session
    from eidolon.livekit.agent.full_duplex.transcript_event import FullDuplexTranscriptEvent
    from eidolon.livekit.common.transcript_evidence import TranscriptEvidence
    from eidolon.livekit.common.audio_timeline import AudioSource, AudioSpan
    from eidolon.livekit.common.audio_evidence import AudioEvidenceCapture
    from eidolon.livekit.agent.observability import TurnTimeline
    pipeline=_make_pipeline_with_session();pipeline._ensure_runtime_defaults()
    pipeline._mark_activity=MagicMock()
    port=client();observer=ConversationSpeaker(recognizer=SpeakerRecognizer(port,POLICY),owner_id='o',session_id='s',secret='x'*32,publish=MagicMock())
    pipeline._speaker_observation=observer
    capture=AudioEvidenceCapture(session_id='s',endpoint_id='mic')
    source=AudioSource('mic',1,16000);capture.select_stream('mic');capture.begin_run(source)
    capture.append(source,start=0,pcm=evidence().pcm);observer.bind_capture(capture)
    timeline=TurnTimeline('t');pipeline._timeline=timeline
    pipeline._user_turns.start_speech(timeline=timeline)
    pipeline._user_turns.note_speech_stopped(eot_score=None)
    try:
        pipeline._ensure_transcript_recorder().record(FullDuplexTranscriptEvent(
            transcript='hello',is_final=True,evidence=TranscriptEvidence(stream_key='asr',revision_key='final',audio_span=AudioSpan('mic',1,16000,0,32000))))
        assert observer._task is not None
        await observer._task
        result=await observer.result('t')
        assert result.status=='identified'
        assert pipeline._user_turns.active.state=='open'
    finally:await observer.aclose()


@pytest.mark.asyncio
async def test_long_asr_tail_expires_identity_without_aborting_delivery():
    observer=ConversationSpeaker(recognizer=SpeakerRecognizer(client(),POLICY),owner_id='o',session_id='s',secret='x'*32,publish=MagicMock())
    value=await observer.recognizer.recognize(evidence(),owner_id='o',session_id='s',turn_id='t')
    expired=value.model_copy(update={'issued_at':time.time()-20,'expires_at':time.time()-10})
    llm=MagicMock();observer.deliver(expired,text='hello',llm=llm,timeline=None)
    accepted=verify_observation(llm.set_speaker_observation.call_args.args[0],secret='x'*32,owner_id='o',session_id='s',turn_id='t',text='hello')
    assert accepted.person_id is None and accepted.reason=='observation_expired'
    assert observer.publish.call_args.args[0].status=='unavailable'
    await observer.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("score,status,reason", [(0.1,"uncertain","short_speech_inconclusive"),(0.8,"identified","voice_match")])
async def test_policy_controls_short_probe_and_unknown_requires_completed_comparison(score,status,reason):
    from dataclasses import replace
    policy = POLICY.model_copy(update={'min_speech_ms': 600})
    port = client(score=score)
    port.extract.return_value = SAMPLE.model_copy(update={'speech_ms':800})
    short = replace(evidence(), end=16000, pcm=evidence().pcm[:32000])
    value = await SpeakerRecognizer(port, policy).recognize(short, owner_id='o', session_id='s', turn_id='t')
    assert value.status == status and value.reason == reason
    assert value.audio_quality.speech_ms == 800
    port.extract.assert_awaited_once_with(short.pcm, min_speech_ms=600)
    port.compare.assert_awaited_once()
    port = client()
    too_short = replace(short, end=4800, pcm=short.pcm[:9600])
    value = await SpeakerRecognizer(port, policy).recognize(too_short, owner_id='o', session_id='s', turn_id='t')
    assert value.reason == 'audio_too_short' and value.person_id is None
    assert value.audio_quality.audio_ms == 300
    port.extract.assert_not_awaited()
    port.compare.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_upstream_validation_failure_is_not_a_voice_mismatch():
    from eidolon_sdk.core.http.errors import ServiceUpstreamError
    port = client()
    port.extract.side_effect = ServiceUpstreamError(422, 'private unexpected validation body')
    value = await SpeakerRecognizer(port, POLICY).recognize(evidence(), owner_id='o', session_id='s', turn_id='t')
    assert value.status == 'unavailable' and value.reason == 'invalid_extraction_response'
    assert 'private' not in value.model_dump_json()


@pytest.mark.asyncio
async def test_completed_observation_prepares_authority_before_turn_delivery():
    observer = ConversationSpeaker(recognizer=SpeakerRecognizer(client(), POLICY),
        owner_id="o", session_id="s", secret="x" * 32, publish=MagicMock())
    prepared = asyncio.Event()
    values = []
    def observe(value):
        values.append(value)
        prepared.set()
    observer.on_observation = observe
    try:
        observer.begin_evidence(evidence(), "t")
        await asyncio.wait_for(prepared.wait(), .2)
        assert values[0].person_id == "p0"
        observer.publish.assert_not_called()
        assert await observer.result("t") == values[0]
    finally:
        await observer.aclose()


@pytest.mark.asyncio
async def test_replaced_evidence_cannot_publish_old_late_result():
    port = client()
    entered, cancelled = asyncio.Event(), asyncio.Event()
    async def slow(_, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    port.extract.side_effect = slow
    published = []
    observer = ConversationSpeaker(recognizer=SpeakerRecognizer(port, POLICY),
        owner_id="o", session_id="s", secret="x" * 32, publish=published.append)
    try:
        observer.begin_evidence(evidence(), "old")
        await entered.wait()
        initial = await observer.result("old")
        observer.deliver(initial, text="hello", llm=MagicMock(), timeline=None)
        observer.begin_evidence(None, "new")
        await asyncio.wait_for(cancelled.wait(), .2)
        await observer._task
        await asyncio.sleep(0)
        assert published == [initial]
        assert (await observer.result("old")).person_id is None
        assert (await observer.result("new")).reason == "audio_evidence_unavailable"
    finally:
        await observer.aclose()


@pytest.mark.asyncio
async def test_candidate_lifecycle_survives_signed_turn_without_becoming_a_grant():
    from eidolon_sdk.biz.speaker_attestation import sign_observation
    port = client()
    person = port.candidates.return_value.persons[0]
    person.maturity = 'candidate'
    person.is_named = False
    result = await SpeakerRecognizer(port, POLICY).recognize(
        evidence(), owner_id='o', session_id='s', turn_id='t')
    assert result.status == 'identified' and result.subject_role == 'speaker'
    assert result.modalities == ['voice'] and result.maturity == 'candidate'
    token = sign_observation(result, secret='x' * 32, text='hello')
    verified = verify_observation(token, secret='x' * 32, owner_id='o', session_id='s',
                                  turn_id='t', text='hello')
    assert verified.maturity == 'candidate' and verified.is_named is False
    port.record_candidate_evidence.assert_awaited_once()
    assert port.record_candidate_evidence.call_args.args[0].observation == result
    assert 'memory_enabled' not in verified.model_dump()


@pytest.mark.asyncio
async def test_receipt_failure_never_changes_a_candidate_identity_match():
    port = client()
    port.candidates.return_value.persons[0].maturity = 'candidate'
    port.record_candidate_evidence.side_effect = OSError('receipt unavailable')
    result = await SpeakerRecognizer(port, POLICY).recognize(
        evidence(), owner_id='o', session_id='s', turn_id='t')
    assert result.status == 'identified' and result.maturity == 'candidate'
    port.record_candidate_evidence.assert_awaited_once()


@pytest.mark.asyncio
async def test_full_shared_pool_extracts_once_and_matches_the_last_person():
    from eidolon_sdk.biz.speaker_identity import MAX_RECOGNITION_PERSONS
    port = client(people=MAX_RECOGNITION_PERSONS)

    async def compare(probe, templates):
        return SpeakerComparisonResult(model_id=probe.model_id, weights_hash=probe.weights_hash,
            preprocessing_version=probe.preprocessing_version, scores=[SpeakerSimilarity(
                profile_id=t.profile_id, revision=t.revision, status='scored',
                similarity=0.9 if t.profile_id.startswith(f'v{MAX_RECOGNITION_PERSONS - 1}-') else 0.1)
                for t in templates])
    port.compare.side_effect = compare
    result = await SpeakerRecognizer(port, POLICY).recognize(evidence(), owner_id='o', session_id='s', turn_id='t')
    assert result.person_id == f'p{MAX_RECOGNITION_PERSONS - 1}'
    assert port.extract.await_count == 1 and port.candidates.await_count == 2
    assert port.compare.await_count == MAX_RECOGNITION_PERSONS


@pytest.mark.asyncio
@pytest.mark.parametrize('people,overlap,score,expected', [(0, 0, .1, True), (1, 0, .1, True),
    (0, None, .1, False), (0, 1, .1, False), (1, 0, .35, False), (1, 0, .9, False), (25, 0, .1, False)])
async def test_discovery_uses_same_extraction_and_never_changes_current_unknown(people, overlap, score, expected):
    from eidolon_sdk.biz.speaker_identity import CandidateDiscoveryPolicy
    port = client(people=people, score=score)
    port.discover_candidate = AsyncMock()
    port.candidates.return_value.discovery_policy = CandidateDiscoveryPolicy(
        version='discovery', recognition=POLICY, max_existing_similarity=.2)
    quality = {} if overlap is None else {'overlap_detected': overlap}
    port.extract.return_value = SAMPLE.model_copy(update={'quality': quality})
    result = await SpeakerRecognizer(port, POLICY).recognize(evidence(), owner_id='o', session_id='s', turn_id='t')
    assert port.extract.await_count == 1
    assert port.discover_candidate.await_count == int(expected)
    if expected:
        assert result.status == 'unknown' and result.person_id is None
        request = port.discover_candidate.call_args.args[0]
        assert request.observation == result and request.sample == port.extract.return_value


@pytest.mark.asyncio
async def test_discovery_policy_mismatch_never_posts_a_proposal():
    from eidolon_sdk.biz.speaker_identity import CandidateDiscoveryPolicy
    port = client(people=0)
    port.discover_candidate = AsyncMock()
    port.candidates.return_value.discovery_policy = CandidateDiscoveryPolicy(
        version='discovery', recognition=POLICY.model_copy(update={'version': 'other'}), max_existing_similarity=.2)
    port.extract.return_value = SAMPLE.model_copy(update={'quality': {'overlap_detected': 0}})
    result = await SpeakerRecognizer(port, POLICY).recognize(evidence(), owner_id='o', session_id='s', turn_id='t')
    assert result.status == 'unknown'
    port.discover_candidate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('quality', [{}, {'overlap_detected': 1}, {'overlap_detected': .5}])
async def test_candidate_match_without_single_speaker_quality_never_accumulates(quality):
    port = client()
    port.candidates.return_value.persons[0].maturity = 'candidate'
    port.extract.return_value = SAMPLE.model_copy(update={'quality': quality})
    result = await SpeakerRecognizer(port, POLICY).recognize(
        evidence(), owner_id='o', session_id='s', turn_id='t')
    assert result.status == ('identified' if not quality else 'uncertain')
    port.record_candidate_evidence.assert_not_awaited()
