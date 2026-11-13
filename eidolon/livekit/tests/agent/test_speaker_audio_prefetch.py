"""Audio scheduling and ownership contracts, independent of model accuracy."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from livekit.agents.vad import VADEventType as E
from eidolon.livekit.common.audio_evidence import AudioEvidenceCapture
from eidolon.livekit.common.audio_timeline import AudioSource, AudioSpan
from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session.user_turn_coordinator import UserTurnCoordinator
from eidolon.livekit.common.transcript_evidence import TranscriptEvidence
from eidolon.livekit.agent.session.conversation_speaker import ConversationSpeaker
from .test_conversation_speaker import CHANNEL, POLICY, client
from eidolon.livekit.agent.session.speaker_recognition import SpeakerRecognizer


def event(kind, index, probability=0, frames=()):
    return SimpleNamespace(type=kind, samples_index=index, probability=probability, frames=frames)


class Stream:
    def __init__(self):
        self.closed = False
        self.frames = []

    def push_frame(self, frame):
        self.frames.append(frame)

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


def setup(prefix=1600):
    """Two seconds of speech; the VAD's start carries ``prefix`` samples before 1600."""
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=0.5)
    coordinator.start_speech(timeline=TurnTimeline("t"))
    capture = AudioEvidenceCapture(session_id="s", endpoint_id="mic")
    port = client()
    observer = ConversationSpeaker(
        recognizer=SpeakerRecognizer(port, POLICY, port.backend, channel=CHANNEL),
        owner_id="o",
        session_id="s",
        secret="x" * 32,
        publish=MagicMock(),
    )
    observer.configure_prefetch(coordinator=coordinator, stream_factory=Stream)
    observer.bind_capture(capture)
    source = AudioSource("mic", 1, 16000)
    capture.select_stream("mic")
    capture.begin_run(source)
    prefetch = observer._prefetch
    # Deliver two seconds of speech with bounded producer/consumer progress.
    for start in range(0, 32000, 1600):
        capture.append(source, start=start, pcm=b"\x01\0" * 1600)
        if start == 0:
            prefetch.consume(
                event(E.START_OF_SPEECH, 1600, frames=[SimpleNamespace(samples_per_channel=prefix)])
            )
        prefetch.consume(event(E.INFERENCE_DONE, start + 1600, 1))
    capture.append(source, start=32000, pcm=b"\0" * 3200)
    prefetch.consume(event(E.INFERENCE_DONE, 33600, 0))
    return coordinator, capture, observer, prefetch, source


@pytest.mark.asyncio
async def test_early_inference_survives_asr_pending_and_is_reused_after_final():
    c, cap, o, p, src = setup()
    try:
        await o._task
        original = o._task
        c.note_speech_stopped(eot_score=None)
        o.begin(c, "t")
        assert o._task is original  # ASR has not emitted any text yet.
        c.add_transcript(
            "hello",
            is_final=True,
            evidence=TranscriptEvidence(
                stream_key="asr", revision_key="1", audio_span=AudioSpan("mic", 1, 16000, 0, 33600)
            ),
        )
        o.begin(c, "t")
        assert o._task is original
        assert (await o.result("t")).decisions.addressing == "p0"
        assert o.recognizer.client.extract.await_count == 1
    finally:
        await o.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("start,end,generation", [(0, 35200, 1), (0, 33600, 2), (34000, 35000, 1)])
async def test_unprocessed_tail_source_mismatch_and_nonoverlap_do_not_reuse(start, end, generation):
    c, cap, o, p, src = setup()
    try:
        assert p.resolve(c.active, AudioSpan("mic", generation, 16000, start, end)) is None
    finally:
        await o.aclose()


@pytest.mark.asyncio
async def test_new_voice_invalidates_prefix_before_rate_limiter_permits_recompute():
    c, cap, o, p, src = setup()
    try:
        await o._task
        assert p.record is not None
        cap.append(src, start=33600, pcm=b"\x01\0" * 800)
        p.consume(event(E.INFERENCE_DONE, 34400, 1))
        assert p.record is None and o._task is None
        assert (await o.result("t")).decisions.persons() == set()
        cap.append(src, start=34400, pcm=b"\0" * 3200)
        p.consume(event(E.INFERENCE_DONE, 36000, 0))
        assert o._task is None  # bound frequency, do not reuse the preceding person.
    finally:
        await o.aclose()


@pytest.mark.asyncio
async def test_trailing_verified_silence_can_extend_coverage_but_not_pcm():
    c, cap, o, p, src = setup()
    try:
        pcm_end = p.record.evidence.end
        cap.append(src, start=33600, pcm=b"\0" * 3200)
        p.consume(event(E.INFERENCE_DONE, 35200, 0))
        resolved = p.resolve(c.active, AudioSpan("mic", 1, 16000, 0, 35200))
        assert resolved is not None and resolved.end == pcm_end
    finally:
        await o.aclose()


@pytest.mark.asyncio
async def test_new_candidate_never_inherits_previous_audio_or_identity():
    c, cap, o, p, src = setup()
    try:
        await o._task
        old = c.active.candidate_id
        c.reject_active(reason="test")
        c.start_speech(timeline=TurnTimeline("next"))
        assert c.active.candidate_id != old
        assert p.resolve(c.active, AudioSpan("mic", 1, 16000, 0, 33600)) is None
        assert (await o.result("next")).decisions.persons() == set()
    finally:
        await o.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["invalidate", "end", "overrun", "gap"])
async def test_input_retirement_cancels_all_adoption(operation):
    c, cap, o, p, src = setup()
    try:
        await o._task
        if operation == "invalidate":
            cap.invalidate("reconnect")
        elif operation == "end":
            cap.end_run(src)
        elif operation == "overrun":
            cap.append(src, start=33600, pcm=b"\0" * 64000)
        else:
            cap.append(src, start=35000, pcm=b"\0" * 1000)
        assert p.record is None and o._task is None
        assert (await o.result("t")).decisions.persons() == set()
    finally:
        await o.aclose()


@pytest.mark.asyncio
async def test_completed_speculation_is_not_adopted_without_committed_audio_mapping():
    c, cap, o, p, src = setup()
    try:
        await o._task
        c.active.state = "committed"
        llm = MagicMock()
        await o.finish(c, turn_id="t", text="hello", llm=llm, timeline=None)
        assert o._frozen.decisions.persons() == set()
    finally:
        await o.aclose()


@pytest.mark.asyncio
async def test_short_new_voice_before_start_confirmation_invalidates_old_result():
    c, cap, o, p, src = setup()
    try:
        await o._task
        p.consume(event(E.END_OF_SPEECH, 33600))
        cap.append(src, start=33600, pcm=b"\x01\0" * 160)
        p.consume(event(E.INFERENCE_DONE, 33760, 1))
        assert p.record is None and o._task is None
    finally:
        await o.aclose()


@pytest.mark.asyncio
async def test_after_endpoint_only_observed_silence_extends_range():
    c, cap, o, p, src = setup()
    try:
        p.consume(event(E.END_OF_SPEECH, 33600))
        cap.append(src, start=33600, pcm=b"\0" * 6400)
        p.consume(event(E.INFERENCE_DONE, 36800, 0))
        assert p.resolve(c.active, AudioSpan("mic", 1, 16000, 0, 36800)) is not None
    finally:
        await o.aclose()


@pytest.mark.asyncio
async def test_listener_failure_preserves_normal_capture_and_detaches_optional_path():
    c, cap, o, p, src = setup()
    try:
        p.append = MagicMock(side_effect=RuntimeError("test optional failure"))
        cap.append(src, start=33600, pcm=b"\0" * 3200)
        assert cap.listener is None
        assert cap.snapshot(candidate="x", span=AudioSpan("mic", 1, 16000, 0, 35200)).complete
        assert p.record is None
    finally:
        await o.aclose()


@pytest.mark.asyncio
async def test_rebinding_same_capture_preserves_work_and_replacing_it_cannot_cancel_new_work():
    c, cap, o, p, src = setup()
    try:
        original = o._task
        o.bind_capture(cap)
        assert o._task is original and o._prefetch is p
        fresh = AudioEvidenceCapture(session_id="s", endpoint_id="new")
        o.bind_capture(fresh)
        source = AudioSource("new", 2, 16000)
        fresh.select_stream("new")
        fresh.begin_run(source)
        from .test_conversation_speaker import evidence

        o.begin_evidence(evidence(), "new-turn")
        new_task = o._task
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert o._task is new_task
        await new_task
        assert (await o.result("new-turn")).decisions.addressing == "p0"
    finally:
        await o.aclose()


@pytest.mark.asyncio
async def test_audio_invalidation_never_reopens_an_already_delivered_turn():
    c, cap, o, p, src = setup()
    try:
        await o._task
        result = await o.result("t")
        llm = MagicMock()
        o.deliver(result, text="hello", llm=llm, timeline=None)
        cap.invalidate("reconnect")
        c.active.state = "committed"
        await o.finish(c, turn_id="t", text="hello", llm=llm, timeline=None)
        o.deliver(
            o.fallback("t", "audio_evidence_unavailable"), text="hello", llm=llm, timeline=None
        )
        assert llm.set_speaker_observation.call_count == 1
        assert o.publish.call_count == 1
    finally:
        await o.aclose()


@pytest.mark.asyncio
async def test_voice_listeners_follow_the_prefetch_vad_and_its_source():
    """The interaction facts read this VAD state: it hears what STT hears."""
    c, cap, o, p, src = setup()  # setup's utterance has started and not ended
    try:
        heard = []
        remove = o.add_voice_listener(heard.append)
        assert heard == [True]  # registered mid-utterance: told at once
        p.consume(event(E.END_OF_SPEECH, 34000))
        p.consume(event(E.START_OF_SPEECH, 40000, frames=[SimpleNamespace(samples_per_channel=1600)]))
        p.consume(event(E.INFERENCE_DONE, 40512, 1))
        assert heard == [True, False, True]
        # A new STT run restarts this VAD state; the old one stops hearing.
        cap.begin_run(AudioSource("mic", 2, 16000))
        assert heard == [True, False, True, False] and o._prefetch.voice is False

        def broken(speaking):
            raise RuntimeError("listener bug")

        o.add_voice_listener(broken)
        o._prefetch.consume(event(E.START_OF_SPEECH, 1600, frames=[SimpleNamespace(samples_per_channel=1600)]))
        assert heard[-1] is True  # one failing listener does not stop the others
        remove()
        o._prefetch.consume(event(E.END_OF_SPEECH, 8000))
        assert heard[-1] is True
    finally:
        await o.aclose()


@pytest.mark.asyncio
async def test_closing_the_observation_ends_an_utterance_it_was_hearing():
    c, cap, o, p, src = setup()
    heard = []
    o.add_voice_listener(heard.append)
    await o.aclose()
    assert heard == [True, False]


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix,span,reason", [
    (1600, AudioSpan("mic", 1, 16000, 0, 33600), None),
    # ASR heard the onset 50 ms before the VAD's start (prefix included).
    (800, AudioSpan("mic", 1, 16000, 0, 33600), "asr_starts_50ms_before_prefetch"),
    (1600, AudioSpan("mic", 1, 16000, 0, 35200), "asr_ends_100ms_after_verified_silence"),
    (1600, AudioSpan("mic", 2, 16000, 0, 33600), "source_changed"),
])
async def test_why_prefetched_audio_is_not_reused_is_named(prefix, span, reason):
    c, cap, o, p, src = setup(prefix)
    try:
        assert p.mismatch(c.active, span) == reason
        assert (p.resolve(c.active, span) is None) is (reason is not None)
    finally:
        await o.aclose()


@pytest.mark.asyncio
async def test_a_prefetch_thrown_away_at_the_final_is_logged_once_with_its_reason(caplog):
    c, cap, o, p, src = setup(800)
    try:
        await o._task
        c.note_speech_stopped(eot_score=None)
        c.add_transcript("hello", is_final=True, evidence=TranscriptEvidence(
            stream_key="asr", revision_key="1", audio_span=AudioSpan("mic", 1, 16000, 0, 33600)))
        with caplog.at_level("INFO"):
            o.begin(c, "t")
            o.begin(c, "t")  # the turn's completion asks again: same audio, same miss
        misses = [r.getMessage() for r in caplog.records if "prefetch not reused" in r.getMessage()]
        assert misses == ["Speaker prefetch not reused turn_id=t reason=asr_starts_50ms_before_prefetch"]
        assert o.recognizer.client.extract.await_count >= 1
    finally:
        await o.aclose()
