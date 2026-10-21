"""Public AgentSession -> SDK resampler -> Bailian -> loopback -> exact PCM.

All audio is synthetic. This exercises real SDK suppression/lifecycle, not a
parallel room recorder or a mocked recognition queue.
"""

import asyncio
import json
from array import array

import pytest
import websockets
from livekit import rtc
from livekit.agents import Agent, AgentSession
from livekit.agents.types import APIConnectOptions

from eidolon.livekit.agent.session.audio_evidence_input import EvidenceAudioInput
from eidolon.livekit.common.audio_evidence import AudioEvidenceCapture
from eidolon.livekit.common.transcript_evidence import transcript_evidence_from_speech_event
from eidolon.livekit.plugins.stt.bailian import BailianFunASRSTT
from .._harness.headless import RecordingAudioOutput, ScriptedAudioInput


async def start_task(socket):
    request = json.loads(await socket.recv())
    task_id = request["header"]["task_id"]
    await socket.send(
        json.dumps(
            {
                "header": {"event": "task-started", "task_id": task_id},
                "payload": {},
            }
        )
    )
    return task_id


async def final(socket, task_id, start, end):
    await socket.send(
        json.dumps(
            {
                "header": {"event": "result-generated", "task_id": task_id},
                "payload": {
                    "output": {
                        "sentence": {
                            "sentence_id": start,
                            "text": "合成测试",
                            "sentence_end": True,
                            "begin_time": start * 1000 // 16000,
                            "end_time": end * 1000 // 16000,
                        }
                    }
                },
            }
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("playing", [False, True])
async def test_normalized_capture_matches_sdk_suppression_and_pause_preserves_media(
    monkeypatch, playing
):
    monkeypatch.setenv("BAILIAN_STT_GATE_ENABLED", "false")
    monkeypatch.setenv("BAILIAN_STT_KEEPALIVE_INTERVAL_SEC", "0")
    sent = {}

    async def server(socket):
        task = await start_task(socket)
        offset = 0
        async for pcm in socket:
            if isinstance(pcm, bytes):
                end = offset + len(pcm) // 2
                sent[(offset, end)] = pcm
                await final(socket, task, offset, end)
                offset = end

    async with websockets.serve(server, "127.0.0.1", 0) as service:
        port = service.sockets[0].getsockname()[1]
        capture = AudioEvidenceCapture(session_id="session", endpoint_id="input")
        provider = BailianFunASRSTT(
            api_url=f"ws://127.0.0.1:{port}",
            api_key="test",
            conn_options=APIConnectOptions(max_retry=0, timeout=2),
            audio_evidence_capture=capture,
        )
        observed = asyncio.Queue()

        class ObservedAgent(Agent):
            async def stt_node(self, audio, model_settings):
                async for event in super().stt_node(audio, model_settings):
                    observed.put_nowait(event)
                    yield event

        raw = ScriptedAudioInput(sample_rate=24000)
        guarded = EvidenceAudioInput(source=raw, capture=capture)
        session = AgentSession(aec_warmup_duration=0)
        session.input.audio = guarded
        session.output.audio = RecordingAudioOutput()
        output_started, release_output = asyncio.Event(), asyncio.Event()

        async def output():
            yield rtc.AudioFrame(b"\x01\x00" * 240, 24000, 1, 240)
            output_started.set()
            await release_output.wait()

        try:
            await session.start(
                agent=ObservedAgent(
                    instructions="",
                    stt=provider,
                    turn_handling={"turn_detection": "manual"},
                )
            )
            if playing:
                session.say("", audio=output(), allow_interruptions=False)
                await asyncio.wait_for(output_started.wait(), 2)
            raw.feed_pcm(b"\x10\x27" * 4800)
            event = await asyncio.wait_for(observed.get(), 2)
            span = transcript_evidence_from_speech_event(event).audio_span
            assert span is not None
            evidence = capture.snapshot(candidate="candidate", span=span)
            assert evidence.complete and evidence.sample_rate == 16000
            assert evidence.pcm == sent[(span.start, span.end)]
            peak = max(abs(sample) for sample in array("h", evidence.pcm))
            if playing:
                # SDK resampling dithers silence by at most a few PCM16 units.
                assert peak <= 2
            else:
                assert peak > 9000

            session.input.set_audio_enabled(False)
            assert capture.active_source is None and capture.retained_samples == 0
            session.input.set_audio_enabled(True)
            # Resuming media is not permission to relabel queued PCM as a new source.
            raw.feed_pcm(b"\x20\x4e" * 4800)
            resumed = await asyncio.wait_for(observed.get(), 2)
            assert resumed.alternatives[0].text == "合成测试"
            assert session.input.audio_enabled
            assert not capture.snapshot(candidate="late", span=span).complete
        finally:
            release_output.set()
            await session.aclose()
        assert capture.active_source is None and capture.retained_samples == 0


@pytest.mark.asyncio
async def test_real_provider_retry_clears_old_partial_chunk_and_audio(monkeypatch):
    monkeypatch.setenv("BAILIAN_STT_GATE_ENABLED", "false")
    monkeypatch.setenv("BAILIAN_STT_KEEPALIVE_INTERVAL_SEC", "0")
    reconnect_ready, drop = asyncio.Event(), asyncio.Event()
    attempts = 0
    second_pcm = []

    async def server(socket):
        nonlocal attempts
        attempts += 1
        attempt = attempts
        task = await start_task(socket)
        if attempt == 2:
            reconnect_ready.set()
        pcm = await socket.recv()
        if attempt == 2:
            second_pcm.append(pcm)
        await final(socket, task, 0, len(pcm) // 2)
        if attempt == 1:
            await drop.wait()
            await socket.close(code=1011, reason="synthetic reconnect")
        else:
            await socket.wait_closed()

    async with websockets.serve(server, "127.0.0.1", 0) as service:
        port = service.sockets[0].getsockname()[1]
        capture = AudioEvidenceCapture(session_id="session", endpoint_id="input")
        provider = BailianFunASRSTT(
            api_url=f"ws://127.0.0.1:{port}",
            api_key="test",
            conn_options=APIConnectOptions(max_retry=1, retry_interval=0, timeout=2),
            audio_evidence_capture=capture,
        )
        stream = provider.stream()
        try:
            stream.push_frame(rtc.AudioFrame(b"\x09\x00" * 2400, 16000, 1, 2400))
            first = transcript_evidence_from_speech_event(
                await asyncio.wait_for(anext(stream), 2)
            ).audio_span
            assert capture.snapshot(candidate="first", span=first).complete
            drop.set()
            await asyncio.wait_for(reconnect_ready.wait(), 2)
            assert capture.retained_samples == 0
            assert not capture.snapshot(candidate="late", span=first).complete
            stream.push_frame(rtc.AudioFrame(b"\x07\x00" * 1600, 16000, 1, 1600))
            second = transcript_evidence_from_speech_event(
                await asyncio.wait_for(anext(stream), 2)
            ).audio_span
            assert second.generation == first.generation + 1
            assert second_pcm == [b"\x07\x00" * 1600]
            evidence = capture.snapshot(candidate="second", span=second)
            assert evidence.complete and evidence.pcm == second_pcm[0]
        finally:
            drop.set()
            await stream.aclose()
        assert capture.retained_samples == 0 and capture.active_source is None
