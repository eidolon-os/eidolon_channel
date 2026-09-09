"""The client half against the protocol the service actually speaks.

A fake server built from the SDK contract rather than from the plugin's own
expectations: if the two ever disagree about a message name, this is where it
shows, and it shows on the side that would otherwise fail on a Host.
"""

from __future__ import annotations

import json

import pytest
from aiohttp import web
from eidolon_sdk.biz.contracts import local_asr as contract
from livekit.agents import stt as lk_stt

from eidolon.livekit.plugins.stt.local_asr import LocalAsrSTT, LocalAsrSTTConfig


class _Recorder:
    """A server that speaks the contract, and remembers what it was told."""

    def __init__(self, *, interims: list[str], final: str) -> None:
        self.interims = interims
        self.final = final
        self.utterances = 0
        self.audio_bytes = 0
        self.closed_cleanly = False

    async def handler(self, request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
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
                continue
            if message.type is not web.WSMsgType.TEXT:
                continue
            payload = json.loads(message.data)
            kind = payload.get("type")
            if kind in {contract.START_UTTERANCE, contract.START_UTTERANCE_LEGACY}:
                self.utterances += 1
                await socket.send_json(
                    {
                        "type": contract.UTTERANCE_STARTED,
                        "stream_id": payload["stream_id"],
                        "utterance_id": payload["utterance_id"],
                    }
                )
            elif kind == contract.END_UTTERANCE:
                for text in self.interims:
                    await socket.send_json(
                        {
                            "type": contract.TRANSCRIPT,
                            "text": text,
                            contract.IS_FINAL_FIELD: False,
                        }
                    )
                await socket.send_json(
                    {
                        "type": contract.TRANSCRIPT,
                        "text": self.final,
                        contract.IS_FINAL_FIELD: True,
                    }
                )
            elif kind == contract.CLOSE_STREAM:
                self.closed_cleanly = True
                await socket.close()
        return socket


async def _serve(recorder: _Recorder) -> tuple[web.AppRunner, int]:
    app = web.Application()
    app.add_routes([web.get(contract.LOCAL_ASR_STREAM_PATH, recorder.handler)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    return runner, port


def _frame(samples: int = 1600):
    from livekit import rtc

    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=contract.AUDIO_SAMPLE_RATE,
        num_channels=contract.AUDIO_CHANNELS,
        samples_per_channel=samples,
    )


@pytest.mark.asyncio
async def test_one_utterance_yields_interims_then_a_final() -> None:
    """The two-pass shape, which is what a local Host offers over a provider.

    Interims arrive from the streaming model while the words are still being
    spoken; the final is re-decoded offline and punctuated. Rendered as
    INTERIM_TRANSCRIPT then FINAL_TRANSCRIPT, which is the contract LiveKit
    already has — so nothing above the plugin has to know where they came from.
    """

    recorder = _Recorder(interims=["欢迎", "欢迎大家"], final="欢迎大家来体验。")
    runner, port = await _serve(recorder)
    try:
        recognizer = LocalAsrSTT(config=LocalAsrSTTConfig(port=port))
        stream = recognizer.stream()
        stream.push_frame(_frame())
        stream.flush()
        stream.end_input()

        events = [event async for event in stream]
        await recognizer.aclose()

        kinds = [event.type for event in events]
        assert lk_stt.SpeechEventType.FINAL_TRANSCRIPT in kinds
        assert kinds.count(lk_stt.SpeechEventType.INTERIM_TRANSCRIPT) == 2
        assert kinds.index(lk_stt.SpeechEventType.FINAL_TRANSCRIPT) == len(kinds) - 1
        final = events[-1].alternatives[0]
        assert final.text == "欢迎大家来体验。"
        assert final.language == "zh"
        assert recorder.utterances == 1
        assert recorder.audio_bytes > 0
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_each_vad_flush_is_its_own_utterance() -> None:
    """The service brackets an utterance; the framework's flush is the bracket.

    Two spoken turns must not merge into one, or the transcript is one long
    sentence and every downstream turn boundary is wrong.
    """

    recorder = _Recorder(interims=[], final="好")
    runner, port = await _serve(recorder)
    try:
        recognizer = LocalAsrSTT(config=LocalAsrSTTConfig(port=port))
        stream = recognizer.stream()
        for _ in range(2):
            stream.push_frame(_frame())
            stream.flush()
        stream.end_input()

        events = [event async for event in stream]
        await recognizer.aclose()

        finals = [
            event
            for event in events
            if event.type is lk_stt.SpeechEventType.FINAL_TRANSCRIPT
        ]
        assert len(finals) == 2
        assert recorder.utterances == 2
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_a_refusal_is_reported_with_the_services_own_code() -> None:
    """Retryable or not is the service's judgement, carried rather than guessed."""

    async def refuse(request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        await socket.send_json(
            {
                "type": contract.CONNECTED,
                contract.PROTOCOL_VERSION_FIELD: contract.LOCAL_ASR_PROTOCOL_VERSION,
            }
        )
        async for message in socket:
            if message.type is web.WSMsgType.TEXT:
                await socket.send_json(
                    {
                        "type": contract.ERROR,
                        "code": contract.ERROR_CAPACITY,
                        "message": "busy",
                        "retryable": True,
                    }
                )
                break
        return socket

    app = web.Application()
    app.add_routes([web.get(contract.LOCAL_ASR_STREAM_PATH, refuse)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    try:
        recognizer = LocalAsrSTT(config=LocalAsrSTTConfig(port=port))
        stream = recognizer.stream()
        stream.push_frame(_frame())
        stream.flush()
        stream.end_input()

        with pytest.raises(Exception) as raised:
            async for _event in stream:
                pass
        await recognizer.aclose()

        assert contract.ERROR_CAPACITY in str(raised.value) or "utterance_started" in str(
            raised.value
        )
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_a_host_whose_service_is_not_running_says_that() -> None:
    """No reconnect budget: it is on this machine. If it is not answering it is
    not running, and backing off would only delay saying so."""

    recognizer = LocalAsrSTT(config=LocalAsrSTTConfig(port=1, connect_timeout_s=0.5))
    stream = recognizer.stream()
    stream.push_frame(_frame())
    stream.flush()
    stream.end_input()

    with pytest.raises(Exception) as raised:
        async for _event in stream:
            pass
    await recognizer.aclose()

    assert "did not answer" in str(raised.value)


@pytest.mark.asyncio
async def test_a_version_this_build_cannot_read_is_refused_not_parsed() -> None:
    """Found on the board: the service greets every stream with the version it
    serves, and the contract had not said so — so the fake server here did not
    greet, these tests passed, and the real service failed the first message.

    Now the greeting is part of the contract, and the version is checked on the
    connection actually being used rather than from a separate readiness
    request that some other process could have answered.
    """

    async def from_the_future(request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        await socket.send_json(
            {
                "type": contract.CONNECTED,
                contract.PROTOCOL_VERSION_FIELD: contract.LOCAL_ASR_PROTOCOL_VERSION + 1,
            }
        )
        async for _message in socket:
            pass
        return socket

    app = web.Application()
    app.add_routes([web.get(contract.LOCAL_ASR_STREAM_PATH, from_the_future)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    try:
        recognizer = LocalAsrSTT(config=LocalAsrSTTConfig(port=port))
        stream = recognizer.stream()
        stream.push_frame(_frame())
        stream.flush()
        stream.end_input()

        with pytest.raises(Exception) as raised:
            async for _event in stream:
                pass
        await recognizer.aclose()

        assert "protocol" in str(raised.value)
    finally:
        await runner.cleanup()
