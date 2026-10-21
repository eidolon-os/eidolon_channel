import asyncio
import base64
from dataclasses import asdict

import httpx
import pytest
from eidolon_sdk.biz.voiceprint import ExtractedSample

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session.speaker_observation import SpeakerTurnObservation
from eidolon.livekit.agent.session.speaker_extraction_client import SpeakerExtractionClient
from eidolon.livekit.agent.session.user_turn_coordinator import UserTurnCoordinator, UserTurnDecision
from eidolon.livekit.common.audio_evidence import AudioEvidenceCapture
from eidolon.livekit.common.audio_timeline import AudioSource, AudioSpan
from eidolon.livekit.common.transcript_evidence import TranscriptEvidence


def sample():
    return ExtractedSample(model_id="synthetic", weights_hash="a" * 64,
        preprocessing_version="continuous-v1", vector=[1, 0], speech_ms=1500)


def turn(name="first", *, start=0, end=24000, final=True):
    owner = UserTurnCoordinator(speech_merge_grace_sec=.8)
    owner.start_speech(timeline=TurnTimeline(name))
    owner.add_transcript("完整句子", is_final=final, evidence=TranscriptEvidence(
        stream_key="stream", revision_key=name,
        audio_span=AudioSpan("stream", 1, 16000, start, end)))
    owner.note_speech_stopped(eot_score=None)
    return owner


def capture():
    value = AudioEvidenceCapture(session_id="session", endpoint_id="endpoint")
    value.select_stream("stream")
    value.begin_run(AudioSource("stream", 1, 16000))
    value.append(value.active_source, start=0, pcm=b"\x01\x00" * 48000)
    return value


def commit(observer, owner):
    decision = owner.mark_framework_completed(transcript=owner.selected_text,
        reason="framework_completed_turn")
    observer.submit(decision, owner)
    return decision


async def settle(observer):
    # Join only in tests; the reply path must never await this task.
    if observer.pending is not None:
        await asyncio.wait_for(asyncio.shield(observer.pending), 1)


@pytest.mark.asyncio
async def test_committed_turn_extracts_once_without_waiting_or_retaining_embedding():
    entered, release = asyncio.Event(), asyncio.Event()
    requests, records = [], []

    async def extract(pcm):
        requests.append(pcm)
        entered.set()
        await release.wait()
        return sample()

    observer = SpeakerTurnObservation(extract=extract, record=records.append)
    observer.bind_capture(capture())
    owner = turn(start=1600, end=25600)
    decision = commit(observer, owner)
    assert decision.action == "commit"  # model has not even started yet
    await asyncio.wait_for(entered.wait(), 1)
    assert requests == [b"\x01\x00" * 24000]
    assert records == []
    # A subsequent candidate does not steal a completed turn's observation.
    next_owner = turn("next", start=24000, end=48000)
    assert commit(observer, next_owner).action == "commit"
    assert records[-1].reason == "busy"
    release.set()
    await settle(observer)
    assert records[-1].candidate_id == decision.candidate_id
    assert records[-1].reason == "extracted"
    assert records[-1].audio_ms == 1500
    assert not {"vector", "pcm", "allowed", "score"} & asdict(records[-1]).keys()
    assert observer.pending is None
    await observer.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("case,reason", [
    ("pending_final", "transcript_not_final"), ("missing", "missing_samples"),
    ("no_source", "audio_source_unavailable"), ("short", "audio_too_short"),
])
async def test_incomplete_audio_is_not_sent_or_replaced_by_buffer_tail(case, reason):
    calls, records = [], []

    async def extract(pcm):
        calls.append(pcm)
        return sample()

    observer = SpeakerTurnObservation(extract=extract, record=records.append)
    if case != "no_source":
        observer.bind_capture(capture())
    owner = turn(final=case != "pending_final", end={"missing": 64000, "short": 1600}.get(case, 24000))
    assert commit(observer, owner).action == "commit"
    await settle(observer)
    assert not calls
    assert records[-1].reason == reason
    await observer.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["replace", "retry", "close"])
async def test_source_change_or_close_cannot_publish_late_success(change):
    entered, release = asyncio.Event(), asyncio.Event()
    records = []

    async def extract(pcm):
        entered.set()
        await release.wait()
        return sample()

    observer = SpeakerTurnObservation(extract=extract, record=records.append)
    value = capture()
    observer.bind_capture(value)
    commit(observer, turn())
    await asyncio.wait_for(entered.wait(), 1)
    if change == "replace":
        observer.bind_capture(capture())
    elif change == "retry":
        value.begin_run(AudioSource("stream", 2, 16000))
    else:
        await observer.aclose()
    release.set()
    await settle(observer)
    assert not any(r.reason == "extracted" for r in records)
    await observer.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["error", "timeout"])
async def test_model_failure_is_diagnostic_only(failure):
    records = []

    async def extract(pcm):
        if failure == "error":
            raise ValueError("must not log private provider response")
        await asyncio.Future()

    observer = SpeakerTurnObservation(extract=extract, record=records.append, timeout_sec=.01)
    observer.bind_capture(capture())
    assert commit(observer, turn()).action == "commit"
    await settle(observer)
    assert records[-1].reason == ("extraction_failed" if failure == "error" else "timeout")
    assert "private" not in repr(records)
    await observer.aclose()


@pytest.mark.asyncio
async def test_no_extraction_for_rejection_duplicate_or_wrong_candidate():
    async def extract(pcm):
        pytest.fail("not a committed observation")

    observer = SpeakerTurnObservation(extract=extract, record=lambda _: None)
    observer.bind_capture(capture())
    owner = turn()
    for action in ("none", "reject", "commit"):
        observer.submit(UserTurnDecision(action, "other"), owner)
    assert observer.pending is None
    await observer.aclose()


@pytest.mark.asyncio
async def test_http_adapter_reuses_contract_and_does_not_send_identity_or_retry():
    requests = []

    async def handler(request):
        requests.append(request)
        return httpx.Response(200, json=sample().model_dump())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = SpeakerExtractionClient(http, "http://provider", token="test-token")
        assert await client.extract(b"\x01\x00" * 24000) == sample()
    assert len(requests) == 1
    request = requests[0]
    import json
    assert json.loads(request.content) == {"pcm_base64": base64.b64encode(b"\x01\x00" * 24000).decode()}
    assert request.url.path == "/v1/voiceprints/extract"
    assert request.headers["authorization"] == "Bearer test-token"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 422, 429, 503, 504])
async def test_provider_refusals_are_not_retried_or_exposed_to_conversation(status):
    requests, records = [], []

    async def handler(request):
        requests.append(request)
        return httpx.Response(status, json={"detail": "private response"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = SpeakerExtractionClient(http, "http://provider", token="test-token")
        observer = SpeakerTurnObservation(extract=client.extract, record=records.append)
        observer.bind_capture(capture())
        assert commit(observer, turn()).action == "commit"
        await settle(observer)
        assert len(requests) == 1
        assert records[-1].reason == {
            401: "provider_unauthorized", 422: "audio_rejected", 429: "provider_busy",
            503: "provider_unavailable", 504: "provider_timeout",
        }[status]
        assert "private" not in repr(records)
        await observer.aclose()


@pytest.mark.asyncio
async def test_real_completion_hook_does_not_wait_for_extraction_and_runs_once():
    from livekit.agents.llm import ChatContext, ChatMessage
    from .test_commit_guard import _make_pipeline_with_session

    entered, release = asyncio.Event(), asyncio.Event()
    requests, records = [], []

    async def extract(pcm):
        requests.append(pcm)
        entered.set()
        await release.wait()
        return sample()

    pipeline = _make_pipeline_with_session()
    pipeline._ensure_runtime_defaults()
    pipeline._user_turns = turn()
    pipeline._timeline = pipeline._user_turns.active.timeline
    observer = SpeakerTurnObservation(extract=extract, record=records.append)
    observer.bind_capture(capture())
    pipeline._speaker_observation = observer
    completion = pipeline._ensure_turn_completion()
    message = ChatMessage(role="user", content=["完整句子"])
    try:
        assert await asyncio.wait_for(completion.voiceprint_allows_completed_turn(
            turn_ctx=ChatContext.empty(), new_message=message), 1)
        await asyncio.wait_for(entered.wait(), 1)
        assert not release.is_set()
        assert not await completion.voiceprint_allows_completed_turn(
            turn_ctx=ChatContext.empty(), new_message=message)
        release.set()
        await settle(observer)
        assert len(requests) == 1
        assert records[-1].reason == "extracted"
        pipeline._session.clear_user_turn.assert_not_called()
        pipeline._session.interrupt.assert_not_called()
    finally:
        await observer.aclose()


def test_framework_confirmation_preserves_all_mapped_segments_but_not_an_unknown_suffix():
    owner = turn(end=24000)
    owner.start_speech(timeline=owner.active.timeline)
    owner.add_transcript("还有后半句", is_final=True, evidence=TranscriptEvidence(
        stream_key="stream", revision_key="second",
        audio_span=AudioSpan("stream", 1, 16000, 32000, 48000)))
    owner.note_speech_stopped(eot_score=None)
    source = AudioSource("stream", 1, 16000)
    # The production boundary prepares then commits. Both must retain exact coverage.
    text = owner.selected_text
    owner.prepare_framework_completed(transcript=text)
    decision = owner.mark_framework_completed(transcript=text, reason="framework_completed_turn")
    reference = owner.capture_committed_audio_reference(candidate_id=decision.candidate_id, source=source)
    assert reference.span == AudioSpan("stream", 1, 16000, 0, 48000)
    assert owner.capture_audio_reference(source=source).span is None
    assert owner.capture_committed_audio_reference(candidate_id="other", source=source).span is None

    owner = turn()
    decision = owner.mark_framework_completed(transcript=owner.selected_text + "没有范围的新增文字",
        reason="framework_completed_turn")
    assert owner.capture_committed_audio_reference(candidate_id=decision.candidate_id, source=source).span is None


@pytest.mark.asyncio
async def test_reporter_failure_does_not_escape_commit_or_background_task():
    def record(_):
        raise RuntimeError("private reporting error")

    async def extract(pcm):
        return sample()

    observer = SpeakerTurnObservation(extract=extract, record=record)
    assert commit(observer, turn()).action == "commit"  # synchronous unavailable report
    observer.bind_capture(capture())
    assert commit(observer, turn("next")).action == "commit"
    await settle(observer)
    await observer.aclose()
