"""Real SDK resampling and loopback WebSocket, no speech service or microphone."""

import asyncio
import json
from dataclasses import replace

import pytest
import websockets
from livekit import rtc
from livekit.agents.types import APIConnectOptions

from eidolon.livekit.common.transcript_evidence import transcript_evidence_from_speech_event
from eidolon.livekit.common.audio_timeline import AudioSource
from eidolon.livekit.common.audio_evidence import AudioEvidenceCapture
from eidolon.livekit.agent.full_duplex.transcript_evidence_buffer import TranscriptEvidenceBuffer
from eidolon.livekit.agent.session.user_turn_coordinator import UserTurnCoordinator
from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.plugins.stt.bailian import BailianFunASRSTT
from eidolon.livekit.plugins.stt.bailian.models import FunASRResultGenerated, FunASRSentence


@pytest.mark.asyncio
@pytest.mark.parametrize("gate_enabled", [False, True])
@pytest.mark.parametrize("capture_enabled", [False, True])
async def test_provider_wire_time_maps_to_resampled_input_after_keepalive(
    monkeypatch, gate_enabled, capture_enabled
):
    monkeypatch.setenv("BAILIAN_STT_GATE_ENABLED", str(gate_enabled).lower())
    monkeypatch.setenv("BAILIAN_STT_KEEPALIVE_INTERVAL_SEC", ".02")
    monkeypatch.setenv("BAILIAN_STT_GATE_KEEPALIVE_INTERVAL_SEC", ".02")
    alive = asyncio.Event()
    sent_samples = []
    sent_pcm = []
    capture = AudioEvidenceCapture(session_id="session", endpoint_id="input")

    async def server(socket):
        run = json.loads(await socket.recv())
        task = run["header"]["task_id"]
        await socket.send(
            json.dumps({"header": {"event": "task-started", "task_id": task}, "payload": {}})
        )
        offset = 0
        emitted = False
        async for data in socket:
            if not isinstance(data, bytes):
                continue
            size = len(data) // 2
            if not any(data):
                alive.set()
            elif not emitted:
                emitted = True
                sent_samples.append(size)
                sent_pcm.append(data)
                await socket.send(
                    json.dumps(
                        {
                            "header": {"event": "result-generated", "task_id": task},
                            "payload": {
                                "output": {
                                    "sentence": {
                                        "sentence_id": 1,
                                        "text": "测试",
                                        "sentence_end": True,
                                        "begin_time": offset * 1000 // 16000,
                                        "end_time": (offset + size) * 1000 // 16000,
                                    }
                                }
                            },
                        }
                    )
                )
            offset += size

    async with websockets.serve(server, "127.0.0.1", 0) as service:
        port = service.sockets[0].getsockname()[1]
        stt = BailianFunASRSTT(
            api_url=f"ws://127.0.0.1:{port}",
            api_key="test",
            conn_options=APIConnectOptions(max_retry=0, timeout=2),
            audio_evidence_capture=capture if capture_enabled else None,
        )
        # The gate config uses its own environment spelling; make this test's
        # bounded cadence explicit on the existing provider configuration.
        stt._config.gate_keepalive_interval_sec = 0.02
        stream = stt.stream()
        try:
            await asyncio.wait_for(alive.wait(), 2)
            # SDK resamples normal RoomIO-rate audio before provider rechunking.
            stream.push_frame(
                rtc.AudioFrame(
                    data=b"\x10\x27" * 4800,
                    sample_rate=24000,
                    num_channels=1,
                    samples_per_channel=4800,
                )
            )
            event = await asyncio.wait_for(anext(stream), 2)
            evidence = transcript_evidence_from_speech_event(event)
            assert evidence.audio_span is not None
            assert evidence.audio_span.sample_rate == 16000
            assert (evidence.audio_span.start, evidence.audio_span.end) == (0, sent_samples[0])
            assert evidence.audio_span.generation == 1
            bridge = TranscriptEvidenceBuffer()
            assert bridge.observe_speech_event(event)
            turns = UserTurnCoordinator(speech_merge_grace_sec=0.8)
            turns.start_speech(timeline=TurnTimeline("wire-test"))
            turns.add_transcript("测试", is_final=True, evidence=bridge.take("测试", is_final=True))
            turns.note_speech_stopped(eot_score=None)
            current_source = AudioSource(stream._stream_id, stream._audio_generation, 16000)
            reference = turns.capture_audio_reference(source=current_source)
            assert reference.span == evidence.audio_span
            if capture_enabled:
                assert capture.active_source == current_source
                frozen = capture.snapshot(candidate=reference.candidate_id, span=reference.span)
                assert frozen.complete and frozen.pcm == sent_pcm[0]
                assert frozen.sample_rate == 16000
                assert (frozen.session_id, frozen.endpoint_id) == ("session", "input")
            else:
                assert capture.active_source is None and capture.retained_samples == 0
            assert not turns.is_audio_reference_current(
                reference, source=replace(current_source, generation=2)
            )
            sentence = FunASRSentence(sentence_id=1, begin_time=100, end_time=200)
            old = FunASRResultGenerated(
                task_id="previous-connection", request_id="", sentences=[sentence]
            )
            assert (
                "audio_span"
                not in stream._transcript_evidence_metadata(old, sentence)[
                    "eidolon.transcript_evidence"
                ]
            )
            # A new connection's generation is a different scope even with the same samples.
            assert evidence.audio_span != replace(evidence.audio_span, generation=2)
        finally:
            await stream.aclose()
        assert capture.active_source is None and capture.retained_samples == 0
