"""Exercise the public SDK input boundary; no room, model or personal audio."""

import asyncio

import pytest
from livekit import rtc
from livekit.agents import Agent, AgentSession
from livekit.agents import stt
from livekit.agents.voice.io import AudioInput

from eidolon.livekit.tests._harness.audio_evidence_stream import AudioEvidenceStream
from .._harness.headless import RecordingAudioOutput


class Source(AudioInput):
    def __init__(self):
        super().__init__(label="synthetic-microphone")
        self.frames = asyncio.Queue()
        self.attaches = self.detaches = 0

    async def __anext__(self):
        return await self.frames.get()

    def on_attached(self):
        self.attaches += 1

    def on_detached(self):
        self.detaches += 1


def frame(value, samples=240, rate=24000, channels=1):
    return rtc.AudioFrame(
        data=value.to_bytes(2, "little", signed=True) * samples * channels,
        sample_rate=rate,
        num_channels=channels,
        samples_per_channel=samples,
    )


def observed(source, **kwargs):
    return AudioEvidenceStream(
        source=source,
        session_id="session",
        endpoint_id="endpoint",
        sample_rate=24000,
        max_seconds=1,
        **kwargs,
    )


class Recognizer(stt.STT):
    """Replace only the model; SDK owns input, default node and stream plumbing."""

    def __init__(self):
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))
        self.received = asyncio.Queue()

    async def _recognize_impl(self, buffer, *, language, conn_options):
        raise AssertionError("streaming expected")

    def stream(self, *, language=None, conn_options=None):
        recognizer = self

        class Stream(stt.RecognizeStream):
            async def _run(self):
                async for item in self._input_ch:
                    if isinstance(item, rtc.AudioFrame):
                        recognizer.received.put_nowait(item)

        return Stream(stt=self, conn_options=conn_options)


@pytest.mark.asyncio
@pytest.mark.parametrize("playing", [False, True])
async def test_real_session_passes_identical_frames_to_stt_and_clears_on_close(playing):
    source = Source()
    recognizer = Recognizer()
    tap_ready = asyncio.Event()
    tap = None

    class ObservedAgent(Agent):
        async def stt_node(self, audio, model_settings):
            nonlocal tap
            tap = observed(audio)
            tap_ready.set()
            try:
                async for event in super().stt_node(tap, model_settings):
                    yield event
            finally:
                tap.close()

    agent = ObservedAgent(
        instructions="", stt=recognizer, turn_handling={"turn_detection": "manual"}
    )
    session = AgentSession(aec_warmup_duration=0)
    session.input.audio = source
    session.output.audio = RecordingAudioOutput()
    output_started, release_output = asyncio.Event(), asyncio.Event()

    async def output():
        yield frame(99)
        output_started.set()
        await release_output.wait()

    try:
        await session.start(agent=agent)
        await asyncio.wait_for(tap_ready.wait(), 2)
        if playing:
            session.say("", audio=output(), allow_interruptions=False)
            await asyncio.wait_for(output_started.wait(), 2)
        start = tap.position
        originals = [frame(11, 113), frame(22, 217)]
        recognized = []
        for item in originals:
            source.frames.put_nowait(item)
            received = await asyncio.wait_for(recognizer.received.get(), 2)
            recognized.append(received)
        end = tap.position
        evidence = tap.snapshot(candidate="turn", start=start, end=end)
        assert evidence.complete
        assert evidence.pcm == b"".join(bytes(item.data) for item in recognized)
        if playing:
            assert evidence.pcm == bytes(330 * 2)
        else:
            assert all(a is b for a, b in zip(originals, recognized))
        assert (evidence.session_id, evidence.endpoint_id) == ("session", "endpoint")
        assert evidence.sample_rate == 24000  # no second resampler before ASR
        assert tap.retained_samples == 330
    finally:
        release_output.set()
        await session.aclose()
    assert tap.retained_samples == 0
    assert not tap.snapshot(candidate="late", start=start, end=end).complete
    assert source.attaches >= 1 and source.detaches >= 1


@pytest.mark.asyncio
async def test_explicit_source_reset_and_new_stream_cannot_reuse_old_evidence():
    source = Source()
    tap = observed(source)
    before = tap.position
    source.frames.put_nowait(frame(1))
    await anext(tap)
    old_end = tap.position
    tap.reset_source()
    assert tap.retained_samples == 0
    after_pause = tap.position
    source.frames.put_nowait(frame(2))
    await anext(tap)
    assert not tap.snapshot(candidate="old", start=before, end=old_end).complete
    assert not tap.snapshot(candidate="mixed", start=before, end=tap.position).complete
    assert tap.snapshot(candidate="new", start=after_pause, end=tap.position).complete
    other = observed(Source())  # same endpoint/session cannot alias an old input
    tap.close()
    assert tap.retained_samples == 0
    assert not other.snapshot(candidate="wrong-input", start=after_pause, end=tap.position).complete
    other.close()


@pytest.mark.asyncio
async def test_reset_during_pending_read_marks_evidence_unverifiable_without_dropping_audio():
    source = Source()
    tap = observed(source)
    read = asyncio.create_task(anext(tap))
    await asyncio.sleep(0)
    tap.reset_source()
    start = tap.position
    original = frame(3)
    source.frames.put_nowait(original)
    assert await read is original
    source.frames.put_nowait(frame(4))
    await anext(tap)
    evidence = tap.snapshot(candidate="race", start=start, end=tap.position)
    assert not evidence.complete and evidence.reason == "input_changed_during_read"
    assert evidence.pcm == b""


@pytest.mark.asyncio
@pytest.mark.parametrize("rate,channels", [(16000, 1), (24000, 2)])
async def test_format_change_invalidates_evidence_but_preserves_media(rate, channels):
    source = Source()
    tap = observed(source)
    start = tap.position
    for item in [frame(1), frame(2, rate=rate, channels=channels), frame(3)]:
        source.frames.put_nowait(item)
        assert await anext(tap) is item
    evidence = tap.snapshot(candidate="bad-format", start=start, end=tap.position)
    assert not evidence.complete and evidence.reason == "audio_format_changed"
    assert evidence.pcm == b""


@pytest.mark.asyncio
async def test_candidate_snapshot_includes_silence_but_excludes_following_turn():
    source = Source()
    tap = observed(source)
    start = tap.position
    for value in [1, 0, 2]:  # pauses remain continuous; no VAD concatenation
        source.frames.put_nowait(frame(value))
        await anext(tap)
    end = tap.position
    source.frames.put_nowait(frame(9))
    await anext(tap)
    evidence = tap.snapshot(candidate="multi-segment", start=start, end=end)
    assert evidence.pcm == b"\x01\x00" * 240 + b"\0\0" * 240 + b"\x02\x00" * 240
    assert evidence.end == 720
    assert "pcm=" not in repr(evidence)
    # No explicit sample boundary means no verification, never "all audio so far".
    with pytest.raises(TypeError):
        tap.snapshot(candidate="missing-boundary")


@pytest.mark.asyncio
async def test_overlong_candidate_never_becomes_a_valid_truncated_suffix():
    source = Source()
    tap = observed(source)
    start = tap.position
    source.frames.put_nowait(frame(5, samples=30000))
    await anext(tap)
    assert tap.retained_samples == 24000
    evidence = tap.snapshot(candidate="long", start=start, end=tap.position)
    assert not evidence.complete and evidence.reason == "range_exceeds_limit"
    assert evidence.pcm == b""


@pytest.mark.asyncio
async def test_cancel_pending_read_clears_evidence_and_does_not_reopen_stream():
    source = Source()
    tap = observed(source)
    start = tap.position
    source.frames.put_nowait(frame(7))
    await anext(tap)
    end = tap.position
    read = asyncio.create_task(anext(tap))
    await asyncio.sleep(0)
    read.cancel()
    with pytest.raises(asyncio.CancelledError):
        await read
    assert tap.retained_samples == 0
    assert not tap.snapshot(candidate="cancelled", start=start, end=end).complete
    tap.reset_source()  # reset must never reopen a closed upstream
    with pytest.raises(StopAsyncIteration):
        await anext(tap)


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_exhaustion_or_error_releases_audio_and_keeps_upstream_semantics(failed):
    pulled = 0

    async def source():
        nonlocal pulled
        pulled += 1
        yield frame(6)
        if failed:
            raise RuntimeError("source failure")

    tap = observed(source())
    assert pulled == 0  # no independent subscription, task or read-ahead
    start = tap.position
    await anext(tap)
    end = tap.position
    assert pulled == 1 and tap.retained_samples == 240
    with pytest.raises(RuntimeError if failed else StopAsyncIteration):
        await anext(tap)
    assert tap.retained_samples == 0
    assert not tap.snapshot(candidate="ended", start=start, end=end).complete
