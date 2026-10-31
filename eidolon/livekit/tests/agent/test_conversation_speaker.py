import asyncio
import math
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import pytest
from eidolon_sdk.biz.person import PersonProfile
from eidolon_sdk.biz.person_observation import PersonObservationFeedback
from eidolon_sdk.biz.speaker_identity import (
    CandidateDiscoveryPolicy,
    IdentityDecisions,
    IdentityPolicy,
    SpeakerCandidates,
    SpeakerCandidate,
)
from eidolon_sdk.biz.voiceprint import ExtractedSample, VoiceTemplate
from eidolon_sdk.biz.speaker_attestation import verify_observation
from eidolon.livekit.common.audio_evidence import AudioEvidence
from eidolon.livekit.agent.session.speaker_recognition import SpeakerRecognizer
from eidolon.livekit.agent.session.conversation_speaker import ConversationSpeaker, feedback_view

SAMPLE = ExtractedSample(
    model_id="m", weights_hash="a" * 64, preprocessing_version="v", vector=[1, 0], speech_ms=2000, quality={"overlap_detected": 0}
)
IDENTITY = (SAMPLE.model_id, SAMPLE.weights_hash, SAMPLE.preprocessing_version)
CHANNEL = "mobile"
POLICY = IdentityPolicy(
    version="identity-test",
    model_id="m",
    weights_hash="a" * 64,
    preprocessing_version="v",
    backend_version="backend-test",
    min_speech_ms=1500,
    min_unknown_speech_ms=1500,
    prior_unknown=0.5,
    personal_memory=0.99,
    addressing=0.9,
    speaker_label=0.9,
    candidate_evidence=0.95,
    discovery=0.99,
)
# One Person alone at this prior has posterior e^llr / (1 + e^llr): MATCH meets
# every use (0.99995), NO_MATCH leaves 0.99995 to someone not enrolled.
MATCH, NO_MATCH = 10.0, -10.0
ALL_USES = dict(personal_memory="p0", addressing="p0", speaker_label="p0")


class Backend:
    """A deterministic calibration: each template's LLR is set by the test, keyed by its vector."""

    def __init__(self, llrs, *, version=POLICY.backend_version, identity=IDENTITY):
        self.llrs, self.version, self.identity = llrs, version, identity
        self.sides = []
        self.trials = 0

    def side(self, vector, speech_ms, channel):
        self.sides.append((tuple(vector), speech_ms, channel))
        return tuple(vector)

    def llr(self, enroll, probe):
        assert probe == tuple(SAMPLE.vector)
        self.trials += 1
        return self.llrs[enroll]


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


def template_vector(person, profile):
    # A distinct unit direction per template, never the probe's.
    angle = 0.01 * (1 + 5 * person + profile)
    return [math.cos(angle), math.sin(angle)]


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
                    VoiceTemplate(profile_id=f"v{i}-{j}", revision=1,
                                  sample=ExtractedSample(**SAMPLE.model_dump() | {"vector": template_vector(i, j)}))
                    for j in range(profiles)
                ],
            )
            for i in range(people)
        ],
    )


def client(people=1, profiles=1, llr=MATCH):
    """Data and the Provider as the recognizer sees them, with the calibration in ``backend``.

    ``llr`` is one LLR for every template, or ``f(person_index, profile_index)``.
    """
    value = snapshot(people, profiles)
    score = llr if callable(llr) else (lambda i, j: llr)
    llrs = {
        tuple(template.sample.vector): score(i, j)
        for i, person in enumerate(value.persons)
        for j, template in enumerate(person.templates)
    }
    return SimpleNamespace(
        candidates=AsyncMock(return_value=value),
        extract=AsyncMock(return_value=SAMPLE),
        record_candidate_evidence=AsyncMock(),
        discover_candidate=AsyncMock(),
        backend=Backend(llrs),
    )


def recognizer(port, policy=POLICY, *, channel=CHANNEL):
    return SpeakerRecognizer(port, policy, port.backend, channel=channel)


@pytest.mark.asyncio
async def test_identification_uses_authority_and_aggregates_same_person_profiles():
    port = client(profiles=2, llr=lambda i, j: MATCH if j == 1 else NO_MATCH)
    result = await recognizer(port).recognize(
        evidence(), owner_id="o", session_id="s", turn_id="t"
    )
    assert result.status == "compared" and result.reason == "compared"
    assert result.decisions == IdentityDecisions(**ALL_USES)
    # One entry per Person, from its best-matching profile.
    assert [(e.person_id, e.profile_id, e.llr) for e in result.evidence] == [("p0", "v0-1", MATCH)]
    assert result.evidence[0].posterior + result.posterior_unknown == pytest.approx(1)
    assert result.policy_version == POLICY.version and result.backend_version == POLICY.backend_version
    assert port.extract.await_count == 1 and port.candidates.await_count == 2
    assert port.backend.trials == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "people,llr,discovery", [(2, MATCH, False), (1, NO_MATCH, True), (0, MATCH, None)]
)
async def test_no_guess_for_ambiguity_unknown_or_empty(people, llr, discovery):
    port = client(people=people, llr=llr)
    result = await recognizer(port).recognize(
        evidence(), owner_id="o", session_id="s", turn_id="t"
    )
    assert result.decisions.persons() == set() and result.display_name is None
    if people == 0:
        assert (result.status, result.reason) == ("uncertain", "no_enrolled_voices")
        port.extract.assert_not_awaited()
        return
    assert result.status == "compared" and result.decisions.discovery is discovery
    if people == 2:
        # Two Persons equally likely: each near one half, neither meets a use.
        assert [e.posterior for e in result.evidence] == pytest.approx([0.5, 0.5], abs=1e-3)


@pytest.mark.asyncio
async def test_revision_change_discards_identity():
    port = client()
    port.candidates.side_effect = [snapshot(), snapshot().model_copy(update={"revision": "c" * 64})]
    result = await recognizer(port).recognize(
        evidence(), owner_id="o", session_id="s", turn_id="t"
    )
    assert result.reason == "candidates_changed" and result.status == "uncertain"
    assert result.evidence == [] and result.decisions == IdentityDecisions()


@pytest.mark.asyncio
async def test_template_sides_are_prepared_once_per_household_revision():
    port = client(profiles=2)
    value = recognizer(port)
    templates = lambda: [s for s in port.backend.sides if s[0] != tuple(SAMPLE.vector)]
    for turn in ("t1", "t2"):
        await value.recognize(evidence(), owner_id="o", session_id="s", turn_id=turn)
    assert len(templates()) == 2
    port.candidates.return_value = port.candidates.return_value.model_copy(update={"revision": "c" * 64})
    await value.recognize(evidence(), owner_id="o", session_id="s", turn_id="t3")
    assert len(templates()) == 4


@pytest.mark.asyncio
async def test_pending_result_is_frozen_and_late_feedback_never_changes_llm():
    port = client()
    gate = asyncio.Event()
    async def slow(_, **kwargs):
        await gate.wait()
        return SAMPLE
    port.extract.side_effect = slow
    published = []
    observer = ConversationSpeaker(recognizer=recognizer(port),
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
        assert published[-1].decisions.addressing == "p0"
        assert await observer.result("t") == result
        observer.deliver(result, text="hello", llm=llm, timeline=None)
        assert published[-1].decisions.addressing == "p0"
        assert llm.set_speaker_observation.call_count == 1
        wrong = await observer.result("next")
        assert wrong.decisions == IdentityDecisions() and wrong.evidence == []
    finally:
        await observer.aclose()


@pytest.mark.asyncio
async def test_signed_delivery_bound_to_exact_turn_and_text():
    observer = ConversationSpeaker(
        recognizer=recognizer(client()),
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
    assert accepted.decided("personal_memory").person_id == "p0"
    assert accepted.source.channel == CHANNEL
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
@pytest.mark.parametrize("llr,addressing,discovery", [(MATCH, "p0", False), (NO_MATCH, None, True)])
async def test_product_completed_turn_delivers_exact_audio_and_signed_context(llr, addressing, discovery):
    from livekit.agents.llm import ChatContext, ChatMessage
    from eidolon.livekit.common.audio_evidence import AudioEvidenceCapture
    from eidolon.livekit.common.audio_timeline import AudioSource, AudioSpan
    from eidolon.livekit.common.transcript_evidence import TranscriptEvidence
    from eidolon.livekit.agent.observability import TurnTimeline
    from .test_commit_guard import _make_pipeline_with_session

    port = client(llr=llr)
    published = []
    observer = ConversationSpeaker(
        recognizer=recognizer(port),
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
        assert len(published) == 1 and published[0].status == "compared"
        assert published[0].decisions.addressing == addressing
        assert published[0].decisions.discovery is discovery
        port.extract.assert_awaited_once_with(evidence().pcm, min_speech_ms=1500)
        token = pipeline._factory.llm.llm.set_speaker_observation.call_args.args[0]
        assert (
            verify_observation(
                token, secret="x" * 32, owner_id="o", session_id="s", turn_id="t", text="hello"
            )
            == published[0]
        )
        # The trace carries evidence and decisions, never a name or profile.
        trace = timeline.attrs["speaker_observation"]
        assert trace["channel"] == CHANNEL and trace["policy_version"] == POLICY.version
        assert trace["decisions"].get("addressing") == addressing
        assert trace["evidence"][0]["person_id"] == "p0"
        assert "Person 0" not in str(trace)
    finally:
        await observer.aclose()


@pytest.mark.asyncio
async def test_existing_provider_http_extraction_is_reused():
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
            assert type(sample) is ExtractedSample and sample.vector == [1.0, 0.0]
            assert engine.calls == [evidence().pcm]
            # Recognition is calibrated here, not by the Provider's cosine comparison.
            assert not hasattr(port, "compare")
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_no_policy_and_short_audio_do_not_call_dependencies():
    port = client()
    result = await SpeakerRecognizer(port, None, port.backend, channel=CHANNEL).recognize(
        evidence(), owner_id="o", session_id="s", turn_id="t"
    )
    assert (result.status, result.reason) == ("unavailable", "recognition_policy_missing")
    assert result.decisions == IdentityDecisions() and result.evidence == []
    port.candidates.assert_not_awaited()
    port.extract.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("calibration", ["missing", "other_version", "other_model"])
async def test_a_policy_without_its_own_calibration_turns_recognition_off(calibration):
    port = client()
    backend = {
        "missing": None,
        "other_version": Backend({}, version="backend-other"),
        "other_model": Backend({}, identity=("m", "c" * 64, "v")),
    }[calibration]
    value = SpeakerRecognizer(port, POLICY, backend, channel=CHANNEL)
    assert value.policy is None and value.backend is None
    result = await value.recognize(evidence(), owner_id="o", session_id="s", turn_id="t")
    assert (result.status, result.reason) == ("unavailable", "recognition_policy_missing")
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
    port=client();observer=ConversationSpeaker(recognizer=recognizer(port),owner_id='o',session_id='s',secret='x'*32,publish=MagicMock())
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
        assert result.decisions.addressing=='p0'
        assert pipeline._user_turns.active.state=='open'
    finally:await observer.aclose()


@pytest.mark.asyncio
async def test_long_asr_tail_expires_identity_without_aborting_delivery():
    observer=ConversationSpeaker(recognizer=recognizer(client()),owner_id='o',session_id='s',secret='x'*32,publish=MagicMock())
    value=await observer.recognizer.recognize(evidence(),owner_id='o',session_id='s',turn_id='t')
    assert value.decisions.addressing=='p0'
    expired=value.model_copy(update={'issued_at':time.time()-20,'expires_at':time.time()-10})
    llm=MagicMock();observer.deliver(expired,text='hello',llm=llm,timeline=None)
    accepted=verify_observation(llm.set_speaker_observation.call_args.args[0],secret='x'*32,owner_id='o',session_id='s',turn_id='t',text='hello')
    assert accepted.decisions==IdentityDecisions() and accepted.evidence==[]
    assert accepted.reason=='observation_expired'
    assert observer.publish.call_args.args[0].status=='unavailable'
    await observer.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("llr,addressing,unknown", [(NO_MATCH, None, True), (MATCH, "p0", False)])
async def test_policy_controls_short_probe_and_unknown_requires_sufficient_speech(llr, addressing, unknown):
    from dataclasses import replace
    policy = POLICY.model_copy(update={'min_speech_ms': 600})
    port = client(llr=llr)
    port.extract.return_value = SAMPLE.model_copy(update={'speech_ms':800})
    short = replace(evidence(), end=16000, pcm=evidence().pcm[:32000])
    value = await recognizer(port, policy).recognize(short, owner_id='o', session_id='s', turn_id='t')
    assert value.status == 'compared' and value.decisions.addressing == addressing
    # A short probe is compared, but failing to match it is not evidence of someone new.
    assert (value.posterior_unknown > policy.discovery) is unknown
    assert value.decisions.discovery is False
    assert value.audio_quality.speech_ms == 800
    port.extract.assert_awaited_once_with(short.pcm, min_speech_ms=600)
    assert (tuple(SAMPLE.vector), 800, CHANNEL) in port.backend.sides and port.backend.trials == 1
    port = client()
    too_short = replace(short, end=4800, pcm=short.pcm[:9600])
    value = await recognizer(port, policy).recognize(too_short, owner_id='o', session_id='s', turn_id='t')
    assert value.reason == 'audio_too_short' and value.decisions == IdentityDecisions()
    assert value.audio_quality.audio_ms == 300
    port.extract.assert_not_awaited()
    assert port.backend.sides == [] and port.backend.trials == 0


@pytest.mark.asyncio
async def test_extraction_below_policy_floor_is_not_compared():
    port = client()
    port.extract.return_value = SAMPLE.model_copy(update={'speech_ms': 1499})
    value = await recognizer(port).recognize(evidence(), owner_id='o', session_id='s', turn_id='t')
    assert (value.status, value.reason) == ('uncertain', 'speech_too_short')
    assert value.evidence == [] and port.backend.trials == 0


@pytest.mark.asyncio
async def test_unknown_upstream_validation_failure_is_not_a_voice_mismatch():
    from eidolon_sdk.core.http.errors import ServiceUpstreamError
    port = client()
    port.extract.side_effect = ServiceUpstreamError(422, 'private unexpected validation body')
    value = await recognizer(port).recognize(evidence(), owner_id='o', session_id='s', turn_id='t')
    assert value.status == 'unavailable' and value.reason == 'invalid_extraction_response'
    assert 'private' not in value.model_dump_json()


@pytest.mark.asyncio
async def test_a_probe_from_another_model_is_never_compared():
    port = client()
    port.extract.return_value = SAMPLE.model_copy(update={'weights_hash': 'c' * 64})
    value = await recognizer(port).recognize(evidence(), owner_id='o', session_id='s', turn_id='t')
    assert (value.status, value.reason) == ('unavailable', 'policy_model_mismatch')
    assert value.evidence == [] and port.backend.trials == 0


@pytest.mark.asyncio
async def test_a_template_from_another_model_is_never_compared():
    port = client()
    template = port.candidates.return_value.persons[0].templates[0]
    template.sample = template.sample.model_copy(update={'preprocessing_version': 'other'})
    value = await recognizer(port).recognize(evidence(), owner_id='o', session_id='s', turn_id='t')
    assert (value.status, value.reason) == ('unavailable', 'model_version_mismatch')
    assert value.evidence == [] and port.backend.trials == 0


@pytest.mark.asyncio
async def test_completed_observation_prepares_authority_before_turn_delivery():
    observer = ConversationSpeaker(recognizer=recognizer(client()),
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
        assert values[0].decided("personal_memory").person_id == "p0"
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
    observer = ConversationSpeaker(recognizer=recognizer(port),
        owner_id="o", session_id="s", secret="x" * 32, publish=published.append)
    try:
        observer.begin_evidence(evidence(), "old")
        await entered.wait()
        initial = await observer.result("old")
        observer.deliver(initial, text="hello", llm=MagicMock(), timeline=None)
        observer.begin_evidence(None, "new")
        await asyncio.wait_for(cancelled.wait(), .2)
        assert observer._task is None
        await asyncio.sleep(0)
        assert published == [initial]
        old = await observer.result("old")
        assert old.status == "unavailable" and old.decisions == IdentityDecisions()
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
    result = await recognizer(port).recognize(
        evidence(), owner_id='o', session_id='s', turn_id='t')
    assert result.status == 'compared' and result.subject_role == 'speaker'
    assert result.modalities == ['voice'] and result.decisions.candidate_evidence == 'p0'
    assert result.decided('candidate_evidence').maturity == 'candidate'
    token = sign_observation(result, secret='x' * 32, text='hello')
    verified = verify_observation(token, secret='x' * 32, owner_id='o', session_id='s',
                                  turn_id='t', text='hello')
    assert verified.evidence[0].maturity == 'candidate' and verified.evidence[0].is_named is False
    port.record_candidate_evidence.assert_awaited_once()
    receipt = port.record_candidate_evidence.call_args.args[0]
    assert receipt.observation == result and receipt.snapshot_revision == 'b' * 64
    assert 'memory_enabled' not in verified.model_dump()


@pytest.mark.asyncio
async def test_receipt_failure_never_changes_a_candidate_identity_match():
    port = client()
    port.candidates.return_value.persons[0].maturity = 'candidate'
    port.record_candidate_evidence.side_effect = OSError('receipt unavailable')
    result = await recognizer(port).recognize(
        evidence(), owner_id='o', session_id='s', turn_id='t')
    assert result.status == 'compared' and result.decisions.candidate_evidence == 'p0'
    port.record_candidate_evidence.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('maturity,llr,chosen', [
    ('candidate', MATCH, 'p0'),
    ('established', MATCH, None),  # an established Person is never candidate evidence
    ('candidate', 2.5, None),      # posterior 0.924: addressed, below the candidate threshold
])
async def test_candidate_evidence_names_only_a_candidate_over_its_own_threshold(maturity, llr, chosen):
    port = client(llr=llr)
    port.candidates.return_value.persons[0].maturity = maturity
    result = await recognizer(port).recognize(evidence(), owner_id='o', session_id='s', turn_id='t')
    assert result.decisions.addressing == 'p0'
    assert result.decisions.candidate_evidence == chosen
    assert port.record_candidate_evidence.await_count == int(chosen is not None)


@pytest.mark.asyncio
async def test_a_short_candidate_match_is_not_a_receipt():
    port = client()
    port.candidates.return_value.persons[0].maturity = 'candidate'
    port.extract.return_value = SAMPLE.model_copy(update={'speech_ms': 800})
    result = await recognizer(port, POLICY.model_copy(update={'min_speech_ms': 600})).recognize(
        evidence(), owner_id='o', session_id='s', turn_id='t')
    assert result.decisions.candidate_evidence == 'p0'
    port.record_candidate_evidence.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('llr,uses', [
    (MATCH, {'personal_memory', 'addressing', 'speaker_label'}),
    (3.0, {'addressing', 'speaker_label'}),  # posterior 0.953: below personal memory
    (1.0, set()),                            # posterior 0.731: no use met
])
async def test_each_use_reads_its_own_threshold_and_names_only_when_addressed(llr, uses):
    port = client(llr=llr)
    person = port.candidates.return_value.persons[0]
    person.profile = PersonProfile(preferred_name='Ada')
    result = await recognizer(port).recognize(evidence(), owner_id='o', session_id='s', turn_id='t')
    decided = {use for use, value in result.decisions.model_dump().items() if value == 'p0'}
    assert result.status == 'compared' and decided == uses
    assert result.decisions.discovery is False
    if 'addressing' in uses:
        assert result.display_name == 'Ada' and result.profile == person.profile
    else:
        assert result.display_name is None and result.profile is None


@pytest.mark.asyncio
async def test_evidence_source_and_probe_carry_the_recognizer_channel():
    port = client()
    template = port.candidates.return_value.persons[0].templates[0]
    result = await recognizer(port, channel='esp32-s3-box-3').recognize(
        evidence(), owner_id='o', session_id='s', turn_id='t')
    assert result.source.channel == 'esp32-s3-box-3'
    # The probe is a different condition from the registration template.
    assert (tuple(SAMPLE.vector), SAMPLE.speech_ms, 'esp32-s3-box-3') in port.backend.sides
    assert (tuple(template.sample.vector), template.sample.speech_ms, 'mobile') in port.backend.sides


@pytest.mark.asyncio
async def test_a_busy_recognizer_refuses_a_second_utterance_without_dependencies():
    port = client()
    entered, gate = asyncio.Event(), asyncio.Event()
    async def slow(_, **kwargs):
        entered.set()
        await gate.wait()
        return SAMPLE
    port.extract.side_effect = slow
    value = recognizer(port)
    first = asyncio.create_task(value.recognize(evidence(), owner_id='o', session_id='s', turn_id='t1'))
    await entered.wait()
    busy = await value.recognize(evidence(), owner_id='o', session_id='s', turn_id='t2')
    assert (busy.status, busy.reason) == ('unavailable', 'recognizer_busy')
    assert port.extract.await_count == 1 and port.candidates.await_count == 1
    gate.set()
    assert (await first).decisions.addressing == 'p0'
    port.extract.side_effect = None
    assert (await value.recognize(evidence(), owner_id='o', session_id='s', turn_id='t3')).status == 'compared'


@pytest.mark.asyncio
async def test_full_shared_pool_extracts_once_and_matches_the_last_person():
    from eidolon_sdk.biz.speaker_identity import MAX_EVIDENCE, MAX_RECOGNITION_PERSONS
    last = MAX_RECOGNITION_PERSONS - 1
    port = client(people=MAX_RECOGNITION_PERSONS, llr=lambda i, j: MATCH if i == last else NO_MATCH)
    result = await recognizer(port).recognize(evidence(), owner_id='o', session_id='s', turn_id='t')
    assert result.decisions.addressing == f'p{last}' and result.evidence[0].person_id == f'p{last}'
    assert len(result.evidence) == MAX_EVIDENCE
    assert port.extract.await_count == 1 and port.candidates.await_count == 2
    assert port.backend.trials == MAX_RECOGNITION_PERSONS


@pytest.mark.asyncio
@pytest.mark.parametrize('people,overlap,llr,expected', [(0, 0, NO_MATCH, True), (1, 0, NO_MATCH, True),
    (0, None, NO_MATCH, False), (0, 1, NO_MATCH, False), (1, 0, 0.0, False), (1, 0, MATCH, False),
    (25, 0, NO_MATCH, False)])
async def test_discovery_uses_same_extraction_and_never_changes_current_unknown(people, overlap, llr, expected):
    port = client(people=people, llr=llr)
    port.candidates.return_value.discovery_policy = CandidateDiscoveryPolicy(
        version='discovery', identity_policy_version=POLICY.version)
    quality = {} if overlap is None else {'overlap_detected': overlap}
    port.extract.return_value = SAMPLE.model_copy(update={'quality': quality})
    result = await recognizer(port).recognize(evidence(), owner_id='o', session_id='s', turn_id='t')
    assert port.extract.await_count == 1
    assert port.discover_candidate.await_count == int(expected)
    if expected:
        assert result.status == 'compared' and result.decisions.discovery
        assert result.decisions.persons() == set()
        request = port.discover_candidate.call_args.args[0]
        assert request.observation == result and request.sample == port.extract.return_value
        assert request.policy_version == 'discovery'


@pytest.mark.asyncio
async def test_discovery_policy_mismatch_never_posts_a_proposal():
    port = client(people=0)
    port.candidates.return_value.discovery_policy = CandidateDiscoveryPolicy(
        version='discovery', identity_policy_version='other')
    result = await recognizer(port).recognize(evidence(), owner_id='o', session_id='s', turn_id='t')
    assert result.status == 'compared' and result.decisions.discovery
    port.discover_candidate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('quality', [{}, {'overlap_detected': 1}, {'overlap_detected': .5}])
async def test_candidate_match_without_single_speaker_quality_never_accumulates(quality):
    port = client()
    port.candidates.return_value.persons[0].maturity = 'candidate'
    port.extract.return_value = SAMPLE.model_copy(update={'quality': quality})
    result = await recognizer(port).recognize(
        evidence(), owner_id='o', session_id='s', turn_id='t')
    assert result.status == ('compared' if not quality else 'uncertain')
    if quality:
        assert result.reason == 'overlapping_speech' and result.decisions == IdentityDecisions()
    port.record_candidate_evidence.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('changes', [
    {'start': -1}, {'end': 0}, {'end': 32001}, {'start': 1},
    {'source': ''}, {'generation': -1}, {'session_id': 'different-session'},
])
async def test_malformed_audio_span_never_reaches_dependencies(changes):
    from dataclasses import replace
    port = client()
    result = await recognizer(port).recognize(
        replace(evidence(), **changes), owner_id='o', session_id='s', turn_id='t')
    assert result.status == 'unavailable'
    assert result.reason == 'invalid_audio_evidence'
    assert result.source is None and result.decisions == IdentityDecisions()
    port.candidates.assert_not_awaited()
    port.extract.assert_not_awaited()


@pytest.mark.asyncio
async def test_impossible_extracted_speech_cannot_identify_or_accumulate():
    port = client()
    port.extract.return_value = SAMPLE.model_copy(update={'speech_ms': 2001})
    result = await recognizer(port).recognize(
        evidence(), owner_id='o', session_id='s', turn_id='t')
    assert result.status == 'unavailable' and result.reason == 'invalid_extraction_response'
    assert result.decisions == IdentityDecisions() and result.evidence == []
    port.record_candidate_evidence.assert_not_awaited()
    port.discover_candidate.assert_not_awaited()


@pytest.mark.asyncio
async def test_buffer_snapshot_preserves_exact_nonzero_source_interval():
    from dataclasses import replace
    from eidolon.livekit.common.audio_evidence import AudioEvidenceBuffer

    buffer = AudioEvidenceBuffer(source='mic', generation=2, sample_rate=16000, max_samples=48000)
    buffer.append(start=1600, pcm=b'\x01\x00' * 16000)
    buffer.append(start=17600, pcm=b'\x02\x00' * 16000)
    captured = replace(buffer.snapshot(candidate='c', start=1600, end=33600, generation=2),
                       session_id='s')
    port = client()
    result = await recognizer(port).recognize(
        captured, owner_id='o', session_id='s', turn_id='t')
    assert result.decisions.addressing == 'p0'
    assert (result.source.start_sample, result.source.end_sample, result.source.generation) == (1600, 33600, 2)
    port.extract.assert_awaited_once_with(captured.pcm, min_speech_ms=POLICY.min_speech_ms)


@pytest.mark.asyncio
async def test_absent_audio_never_looks_like_pending_inference():
    port=client()
    observer=ConversationSpeaker(recognizer=recognizer(port),
        owner_id='o',session_id='s',secret='x'*32,publish=MagicMock())
    try:
        observer.begin_evidence(None,'t')
        assert observer._task is None
        value=await observer.result('t')
        assert value.reason == 'audio_evidence_unavailable'
        port.candidates.assert_not_awaited()
        port.extract.assert_not_awaited()
    finally:
        await observer.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('extra', [False, True])
async def test_eot_text_projection_reuses_only_proven_audio(extra):
    from .test_user_turn_coordinator import _mapped_pair
    from eidolon.livekit.common.audio_evidence import AudioEvidenceCapture
    coordinator, source = _mapped_pair()
    capture = AudioEvidenceCapture(session_id='s',endpoint_id='mic')
    capture.select_stream('mic')
    capture.begin_run(source)
    capture.append(source,start=0,pcm=evidence().pcm)
    observer=ConversationSpeaker(recognizer=recognizer(client()),
        owner_id='o',session_id='s',secret='x'*32,publish=MagicMock())
    observer.bind_capture(capture)
    try:
        observer.begin(coordinator,'mapped')
        original=observer._task
        await original
        text=('另外一句。'+coordinator.selected_text if extra else '你是？ 你知道我喜欢穿什么吗。')
        coordinator.mark_framework_completed(transcript=text,reason='framework_completed_turn')
        llm=MagicMock()
        await observer.finish(coordinator,turn_id='mapped',text=text,llm=llm,timeline=None)
        value=verify_observation(llm.set_speaker_observation.call_args.args[0],
            secret='x'*32,owner_id='o',session_id='s',turn_id='mapped',text=coordinator.selected_text)
        if extra:
            assert value.decisions == IdentityDecisions()
            assert value.reason == 'framework_audio_coverage_unproven'
            assert observer._task is None
        else:
            assert value.decisions.addressing == 'p0'
            assert observer._task is original
    finally:
        await observer.aclose()


# ---------------------------------------------------------------- phone feedback


def _feedback(value):
    """The phone packet, validated by its own contract."""
    return PersonObservationFeedback(sequence=1, session_id='s', conversation_id='s', turn_id=value.turn_id,
        **feedback_view(value), issued_at=value.issued_at, expires_at=value.expires_at)


@pytest.mark.asyncio
@pytest.mark.parametrize('people,llr,expected', [
    (1, MATCH, ('identified', 'voice_match', 'p0', 'Person 0', 'controlled_trial')),
    (1, 3.0, ('identified', 'voice_match', 'p0', 'Person 0', 'controlled_trial')),  # addressed, no memory
    (1, NO_MATCH, ('unknown', 'no_match', None, None, 'none')),
    (2, MATCH, ('uncertain', 'identity_not_confirmed', None, None, 'none')),
    (1, 1.0, ('uncertain', 'identity_not_confirmed', None, None, 'none')),
])
async def test_feedback_names_a_person_only_when_addressing_is_met(people, llr, expected):
    value = await recognizer(client(people=people, llr=llr)).recognize(
        evidence(), owner_id='o', session_id='s', turn_id='t')
    assert value.status == 'compared'
    packet = _feedback(value)
    assert (packet.status, packet.reason, packet.person_id, packet.display_name, packet.confidence) == expected


@pytest.mark.asyncio
async def test_feedback_passes_unresolved_outcomes_through():
    from dataclasses import replace
    port = client()
    short = await recognizer(port).recognize(replace(evidence(), end=4800, pcm=evidence().pcm[:9600]),
        owner_id='o', session_id='s', turn_id='t')
    off = await SpeakerRecognizer(port, None, None, channel=CHANNEL).recognize(
        evidence(), owner_id='o', session_id='s', turn_id='t')
    for value, status, reason in ((short, 'uncertain', 'audio_too_short'),
                                  (off, 'unavailable', 'recognition_policy_missing')):
        packet = _feedback(value)
        assert (packet.status, packet.reason, packet.person_id, packet.display_name, packet.confidence) == (
            status, reason, None, None, 'none')
