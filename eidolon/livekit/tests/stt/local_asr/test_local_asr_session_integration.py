"""What the framework does to this plugin, rather than what we hoped it does.

The tests beside this file drive the plugin directly: they call
``push_frame`` and then ``end_input`` themselves, so they answer "does the
plugin close an utterance when someone flushes it" — and it does. What
they cannot answer is the question a Host actually asks: *does anyone ever
flush it?*

So these tests hand the plugin to the framework and let the framework drive:
the node under test is LiveKit's own ``Agent.default.stt_node``, and it is
iterated by LiveKit's own ``_STTPipeline`` — the same two pieces that carry
audio on a Host. The test decides only what a microphone decides: which
frames arrive, and at what rate. It never sends a flush, because the
framework never sends one, and that is the whole point.
"""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from aiohttp import web
from eidolon_sdk.biz.contracts import local_asr as contract
from livekit.agents import stt as lk_stt
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.voice.agent import Agent, ModelSettings
from livekit.agents.voice.audio_recognition import _STTPipeline

from eidolon.livekit.plugins.stt.local_asr import LocalAsrSTT, LocalAsrSTTConfig

#: What LiveKit's room input hands the pipeline. Not a choice this test
#: makes: ``AudioInputOptions.sample_rate`` defaults to 24 kHz, and nothing
#: in eidolon_channel overrides it.
ROOM_SAMPLE_RATE = 24_000

#: The rate the service assumes when it counts an utterance's seconds:
#: ``max_utterance_seconds * 32000`` bytes, i.e. 16 kHz mono PCM16.
SERVICE_BYTES_PER_SECOND = 32_000


class _FakeHostAsr:
    """The service's half of the contract, including its utterance limit.

    Counts seconds the way ``eidolon_models_asr.service`` counts them — from
    accumulated bytes at 16 kHz — because that accounting is exactly what the
    plugin's audio has to survive.
    """

    def __init__(
        self,
        *,
        max_utterance_seconds: float = 60.0,
        interim_every: int | None = None,
    ) -> None:
        self.max_audio_bytes = round(max_utterance_seconds * SERVICE_BYTES_PER_SECOND)
        #: Emit a partial every N audio frames, the way the real backend does:
        #: ``feed_pcm16`` returns results as the words arrive, without waiting
        #: for the utterance to be closed.
        self.interim_every = interim_every
        self.audio_frames = 0
        self.interims_sent = 0
        self.connections = 0
        self.utterances_started = 0
        self.utterances_ended = 0
        self.audio_bytes = 0
        #: Every byte this service was ever sent, across utterances.
        self.total_audio_bytes = 0
        self.rejected = False

    async def handler(self, request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        self.connections += 1
        await socket.send_json(
            {
                "type": contract.CONNECTED,
                contract.PROTOCOL_VERSION_FIELD: contract.LOCAL_ASR_PROTOCOL_VERSION,
                "backend": "fake",
                "model_id": "fake-streaming",
                "offline_model_id": "fake-offline",
            }
        )
        async for message in socket:
            if message.type is web.WSMsgType.BINARY:
                self.audio_bytes += len(message.data)
                self.total_audio_bytes += len(message.data)
                self.audio_frames += 1
                if self.interim_every and self.audio_frames % self.interim_every == 0:
                    self.interims_sent += 1
                    await socket.send_json(
                        {
                            "type": contract.TRANSCRIPT,
                            "text": "半句",
                            contract.IS_FINAL_FIELD: False,
                        }
                    )
                if self.audio_bytes > self.max_audio_bytes:
                    # service.py:325-334 — said once, not retryable, and the
                    # socket goes with it.
                    self.rejected = True
                    await socket.send_json(
                        {
                            "type": contract.ERROR,
                            "code": contract.ERROR_UTTERANCE_TOO_LONG,
                            "message": "utterance exceeds the limit",
                            "retryable": False,
                        }
                    )
                    await socket.close(code=1009, message=b"utterance too long")
                    return socket
                continue
            if message.type is not web.WSMsgType.TEXT:
                continue
            payload = json.loads(message.data)
            kind = payload.get("type")
            if kind in {contract.START_UTTERANCE, contract.START_UTTERANCE_LEGACY}:
                self.utterances_started += 1
                # Per utterance, as `service.py` counts it: `_audio_bytes`
                # lives on the utterance object, so the limit does not carry
                # across utterances or across streams.
                self.audio_bytes = 0
                await socket.send_json(
                    {
                        "type": contract.UTTERANCE_STARTED,
                        "stream_id": payload["stream_id"],
                        "utterance_id": payload["utterance_id"],
                    }
                )
            elif kind == contract.END_UTTERANCE:
                self.utterances_ended += 1
                await socket.send_json(
                    {
                        "type": contract.TRANSCRIPT,
                        "text": "一句话。",
                        contract.IS_FINAL_FIELD: True,
                    }
                )
            elif kind == contract.CLOSE_STREAM:
                await socket.close()
        return socket


async def _serve(service: _FakeHostAsr):
    app = web.Application()
    app.add_routes([web.get(contract.LOCAL_ASR_STREAM_PATH, service.handler)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, runner.addresses[0][1]


def _room_frame(*, samples: int = 1200):
    """One 50 ms frame as the room input produces them: 24 kHz, mono."""

    from livekit import rtc

    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=ROOM_SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=samples,
    )


def _framework_pipeline(stt: LocalAsrSTT) -> _STTPipeline:
    """The plugin, wired the way a Host wires it.

    ``Agent.default.stt_node`` is the framework's own node — the thing that
    decides whether ``push_frame``, ``flush`` or ``end_input`` is called —
    and ``_STTPipeline`` is the framework's own owner of the node's
    lifecycle, including whether a dead stream is rebuilt. Both come from
    livekit-agents; the test supplies only the activity they read from.
    """

    activity = SimpleNamespace(
        stt=stt,
        vad=None,
        session=SimpleNamespace(
            conn_options=SimpleNamespace(stt_conn_options=DEFAULT_API_CONNECT_OPTIONS),
            _recorder_io=None,
            _started_at=time.time(),
        ),
        _audio_recognition=None,
    )
    agent = SimpleNamespace(_get_activity_or_raise=lambda: activity)

    def node(audio, model_settings: ModelSettings):
        return Agent.default.stt_node(agent, audio, model_settings)

    return _STTPipeline(node, is_closing=lambda: False)


async def _drain_events(pipeline: _STTPipeline, sink: list) -> None:
    async for event in pipeline.event_ch:
        sink.append(event)


@pytest.mark.asyncio
async def test_the_framework_alone_never_closes_an_utterance() -> None:
    """Why this provider needs `end_utterance` at all.

    Characterisation, not a wish: LiveKit flushes a *non*-streaming STT by
    wrapping it in `StreamAdapter`, and this plugin declares `streaming`, so
    the default node only ever calls `push_frame`. Ten seconds of room audio
    spanning two spoken stretches and a silence therefore close nothing —
    which is why the Host accumulated a whole session into one utterance and
    why the boundary has to come from the pipeline's VAD instead.

    If a future livekit-agents does start flushing streaming plugins, this
    test fails, and that is the point of it.
    """

    service = _FakeHostAsr(max_utterance_seconds=600.0)
    runner, port = await _serve(service)
    stt = LocalAsrSTT(config=LocalAsrSTTConfig(port=port))
    pipeline = _framework_pipeline(stt)
    events: list = []
    reader = asyncio.create_task(_drain_events(pipeline, events))
    try:
        # 2 s spoken, 6 s quiet, 2 s spoken — 50 ms frames, as the room sends
        # them. Nothing here flushes, because nothing in the framework does.
        for _ in range(int(10 / 0.05)):
            pipeline.audio_ch.send_nowait(_room_frame())
            await asyncio.sleep(0)
        await asyncio.sleep(0.5)

        assert service.utterances_started == 1, (
            "the utterance opens on the first frame the room delivers"
        )
        assert service.utterances_ended == 0, (
            f"the framework closed {service.utterances_ended} utterance(s) on its "
            "own; if livekit-agents now flushes streaming plugins, the pipeline's "
            "own end_utterance call is no longer the only boundary and this "
            "plugin's endpointing should be reconsidered"
        )
    finally:
        reader.cancel()
        await pipeline.aclose()
        await stt.aclose()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_room_audio_is_resampled_to_the_rate_the_service_reads() -> None:
    """The service reads 16 kHz. The room sends 24 kHz. Someone must convert.

    ``LocalAsrSTTConfig.sample_rate`` says 16 kHz and its own docstring says
    it is there "so the pipeline can be told what it is rather than
    assuming" — but the stream never passes it to ``RecognizeStream``, which
    is the only place that resamples. The siblings do pass it
    (``bailian``/``sensetime`` both hand ``sample_rate=`` to ``super()``).

    The cost is paid twice: the model hears speech 1.5x too fast, and the
    service's byte accounting reaches its 60 s limit after 40 s of wall
    clock — which is exactly the 40.06 s measured on the Host between
    ``local_asr connected`` and ``utterance exceeds 60 seconds``.
    """

    service = _FakeHostAsr(max_utterance_seconds=600.0)
    runner, port = await _serve(service)
    stt = LocalAsrSTT(config=LocalAsrSTTConfig(port=port))
    pipeline = _framework_pipeline(stt)
    events: list = []
    reader = asyncio.create_task(_drain_events(pipeline, events))
    try:
        seconds = 4.0
        frames = int(seconds / 0.05)
        for _ in range(frames):
            pipeline.audio_ch.send_nowait(_room_frame())
            await asyncio.sleep(0)
        await asyncio.sleep(0.5)

        pushed_24k_bytes = frames * 1200 * 2
        expected_16k_bytes = round(seconds * 16_000 * 2)
        # A frame's worth of tolerance: the plugin's AudioByteStream holds a
        # partial frame back until something flushes it.
        assert service.total_audio_bytes <= expected_16k_bytes + 3200, (
            f"{seconds:g} s of 24 kHz room audio reached the service as "
            f"{service.total_audio_bytes} bytes, which it will read as "
            f"{service.total_audio_bytes / SERVICE_BYTES_PER_SECOND:.1f} s of 16 kHz "
            f"audio; resampled it would be about {expected_16k_bytes} bytes "
            f"(raw, unresampled: {pushed_24k_bytes})"
        )
    finally:
        reader.cancel()
        await pipeline.aclose()
        await stt.aclose()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_one_over_long_utterance_does_not_end_recognition_for_the_session() -> None:
    """``utterance_too_long`` is "not retryable as sent", not "give up".

    The contract's own wording is about the utterance, not the stream: the
    words already sent are gone, the next sentence is fine. But the plugin
    answers with a bare ``RuntimeError``, and both framework layers that
    could recover are looking for an ``APIError`` —
    ``RecognizeStream._main_task`` for a retry and ``_STTPipeline._stt_pump``
    for a rebuild. A ``RuntimeError`` reaches neither, so the pump dies, its
    done-callback closes ``event_ch``, and nothing the user says for the rest
    of the session is ever transcribed.
    """

    service = _FakeHostAsr(max_utterance_seconds=1.0)
    runner, port = await _serve(service)
    stt = LocalAsrSTT(config=LocalAsrSTTConfig(port=port))
    pipeline = _framework_pipeline(stt)
    events: list = []
    reader = asyncio.create_task(_drain_events(pipeline, events))
    try:
        # Enough to trip the limit, then keep talking as a user would.
        for _ in range(int(4.0 / 0.05)):
            pipeline.audio_ch.send_nowait(_room_frame())
            await asyncio.sleep(0)
        # Well past _STT_RECONNECT_INTERVAL (0.5 s), so a rebuild would show.
        await asyncio.sleep(1.5)
        assert service.rejected, "the fake service never reached its limit"

        for _ in range(int(2.0 / 0.05)):
            pipeline.audio_ch.send_nowait(_room_frame())
            await asyncio.sleep(0)
        await asyncio.sleep(1.5)

        assert service.connections >= 2, (
            "after the service refused one over-long utterance and closed the "
            f"socket, the plugin opened {service.connections} connection(s) in "
            "total: recognition is gone for the rest of the session and only a "
            "restart brings it back"
        )
        assert not pipeline.event_ch.closed, (
            "the STT event channel is closed, so no later utterance in this "
            "session can produce a transcript"
        )
    finally:
        reader.cancel()
        await pipeline.aclose()
        await stt.aclose()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_interims_reach_the_session_while_the_words_are_still_arriving() -> None:
    """The two-pass shape's first pass, which is the plugin's stated reason to exist.

    Its own header promises that "interim answers arrive from a streaming
    model while the words are still being spoken". Delivering them is
    ``_drain``'s job, and ``_drain`` asks for them with
    ``asyncio.wait_for(socket.receive(), timeout=0)``. A zero timeout does not
    mean "take whatever has already arrived": ``wait_for`` schedules the
    coroutine and, finding it not yet done, cancels it — so the branch is
    unreachable for every message, including ones already queued on the
    socket.

    With no flush to reach ``_await_final`` either, this is why a Host that
    is streaming audio correctly still shows ``transcript_interim_first_ms =
    null`` on every turn.
    """

    service = _FakeHostAsr(max_utterance_seconds=600.0, interim_every=4)
    runner, port = await _serve(service)
    stt = LocalAsrSTT(config=LocalAsrSTTConfig(port=port))
    pipeline = _framework_pipeline(stt)
    events: list = []
    reader = asyncio.create_task(_drain_events(pipeline, events))
    try:
        for _ in range(int(3.0 / 0.05)):
            pipeline.audio_ch.send_nowait(_room_frame())
            await asyncio.sleep(0)
        await asyncio.sleep(0.5)

        assert service.interims_sent >= 1, "the fake service sent no partials"
        interims = [
            event for event in events if event.type is lk_stt.SpeechEventType.INTERIM_TRANSCRIPT
        ]
        assert interims, (
            f"the service sent {service.interims_sent} partial(s) and the "
            f"session received {len(interims)}: _drain's zero timeout cancels "
            "every receive before it can return one"
        )
    finally:
        reader.cancel()
        await pipeline.aclose()
        await stt.aclose()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_the_pipeline_closing_an_utterance_yields_a_final_and_starts_the_next() -> None:
    """The boundary the pipeline now sends, driven through the framework.

    This calls exactly what `StreamingPipeline._close_stt_utterance` calls,
    at the moment `handle_stopped` calls it — the plugin's own
    `end_utterance()` — and nothing else. It never reaches into
    `RecognizeStream` to fabricate a `_FlushSentinel`, because that is the
    step that made the earlier "it works on the Host" claim untrue: a test
    that supplies its own boundary is testing the half that was never
    broken.

    Two spoken stretches, told apart the way the Host tells them apart, are
    two utterances with a final each.
    """

    service = _FakeHostAsr(max_utterance_seconds=600.0)
    runner, port = await _serve(service)
    stt = LocalAsrSTT(config=LocalAsrSTTConfig(port=port))
    pipeline = _framework_pipeline(stt)
    events: list = []
    reader = asyncio.create_task(_drain_events(pipeline, events))
    try:
        for _ in range(2):
            for _ in range(int(1.5 / 0.05)):
                pipeline.audio_ch.send_nowait(_room_frame())
                await asyncio.sleep(0)
            # VAD says the speaker stopped; the pipeline passes that on.
            assert stt.end_utterance() is True
            await asyncio.sleep(0.4)

        assert service.utterances_started == 2, (
            f"{service.utterances_started} utterance(s) opened for two spoken "
            "stretches"
        )
        assert service.utterances_ended == 2, (
            f"{service.utterances_ended} utterance(s) closed for two spoken "
            "stretches"
        )
        finals = [
            event
            for event in events
            if event.type is lk_stt.SpeechEventType.FINAL_TRANSCRIPT
        ]
        assert len(finals) == 2, (
            f"two closed utterances produced {len(finals)} final transcript(s)"
        )
    finally:
        reader.cancel()
        await pipeline.aclose()
        await stt.aclose()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_end_utterance_is_asked_for_by_name_and_stays_quiet_otherwise() -> None:
    """No stream yet, or a finished one, is normal rather than an error.

    `_close_stt_utterance` calls this between sessions and on providers that
    never made a stream, and a False is how it says "nothing to close" — the
    same shape as `observe_next_audio_for_turn` next door.
    """

    stt = LocalAsrSTT(config=LocalAsrSTTConfig(port=1))
    assert stt.end_utterance() is False
    await stt.aclose()


@pytest.mark.asyncio
async def test_the_sentence_after_an_over_long_one_is_still_transcribed() -> None:
    """The property the Host actually needs, stated end to end.

    Someone talking past the limit costs that utterance — the service says
    so and hangs up, and it is right to. What must survive is everything
    after it. This is the shape of the real failure: three turns in a row
    answered with a fallback line, and a service restart as the only way
    back.
    """

    # A 1 s limit against a 0.2 s sentence stands in for the Host's 60 s
    # against a normal one: the over-long utterance is the exception and what
    # follows it is comfortably inside.
    service = _FakeHostAsr(max_utterance_seconds=1.0)
    runner, port = await _serve(service)
    stt = LocalAsrSTT(config=LocalAsrSTTConfig(port=port))
    pipeline = _framework_pipeline(stt)
    events: list = []
    reader = asyncio.create_task(_drain_events(pipeline, events))
    try:
        # Someone talks past what this Host will hold for one utterance.
        for _ in range(int(1.2 / 0.05)):
            pipeline.audio_ch.send_nowait(_room_frame())
            await asyncio.sleep(0)
        await asyncio.sleep(0.5)
        assert service.rejected

        # The framework reconnects on its own backoff, without needing audio
        # to do it. Wait for the stream rather than guessing at the schedule.
        deadline = asyncio.get_running_loop().time() + 15.0
        while (
            service.connections < 2
            and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.05)
        assert service.connections >= 2, "recognition was never re-established"

        # Now a short sentence, ended the way the pipeline ends one.
        for _ in range(int(0.2 / 0.05)):
            pipeline.audio_ch.send_nowait(_room_frame())
            await asyncio.sleep(0)
        assert stt.end_utterance() is True
        await asyncio.sleep(1.0)

        finals = [
            event
            for event in events
            if event.type is lk_stt.SpeechEventType.FINAL_TRANSCRIPT
        ]
        assert finals, (
            "nothing said after the over-long utterance was transcribed; "
            f"the service saw {service.connections} connection(s) and "
            f"{service.utterances_ended} closed utterance(s)"
        )
    finally:
        reader.cancel()
        await pipeline.aclose()
        await stt.aclose()
        await runner.cleanup()
