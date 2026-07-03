from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass
from dataclasses import replace
from types import SimpleNamespace

import pytest
from eidolon_sdk.biz.contracts import CLIENT_AUDIO_STATE_TOPIC, WIRE_SCHEMA_VERSION
from livekit.agents.voice import AgentSession
from livekit.agents.voice.room_io import RoomOptions

from eidolon.livekit.agent.half_duplex import (
    HalfDuplexPttPipeline,
    HalfDuplexPttTurnController,
    PttAudioSegmentConfig,
    PttAudioSegmentRecorder,
    PttSegmentTranscriber,
    PttSegmentTranscriberConfig,
)
from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.pipeline.types import PipelineState
from eidolon.livekit.common.config import ObservabilityConfig, TurnPolicyConfig


@dataclass
class _Frame:
    data: bytes


class _FakeSttStage:
    def __init__(
        self,
        *,
        offline_text: str = "",
        streaming_text: str = "",
        offline_supported: bool = True,
        offline_raises: bool = False,
        offline_error: Exception | None = None,
        streaming_error: Exception | None = None,
    ) -> None:
        self.stt = SimpleNamespace(
            capabilities=SimpleNamespace(offline_recognize=offline_supported)
        )
        self.offline_text = offline_text
        self.streaming_text = streaming_text
        self.offline_raises = offline_raises
        self.offline_error = offline_error
        self.streaming_error = streaming_error
        self.recognize_calls: list[bytes] = []
        self.recognize_streaming_calls: list[bytes] = []

    async def recognize(self, audio: bytes) -> str:
        self.recognize_calls.append(audio)
        if self.offline_error is not None:
            raise self.offline_error
        if self.offline_raises:
            raise NotImplementedError("offline unsupported")
        return self.offline_text

    async def recognize_streaming(self, audio: bytes) -> str:
        self.recognize_streaming_calls.append(audio)
        if self.streaming_error is not None:
            raise self.streaming_error
        return self.streaming_text


def _controller(
    stt: _FakeSttStage,
    *,
    agent_output_active: bool = False,
    preemptions: list[str] | None = None,
    min_audio_duration_sec: float = 0.01,
    tap_to_stop_max_audio_sec: float = 0.35,
    strategy: str = "streaming",
) -> HalfDuplexPttTurnController:
    preemptions = preemptions if preemptions is not None else []
    return HalfDuplexPttTurnController(
        recorder=PttAudioSegmentRecorder(
            config=PttAudioSegmentConfig(max_duration_sec=2.0)
        ),
        transcriber=PttSegmentTranscriber(
            stt,
            config=PttSegmentTranscriberConfig(
                strategy=strategy,
                min_audio_duration_sec=min_audio_duration_sec,
            ),
        ),
        agent_output_active=lambda: agent_output_active,
        preempt_agent_output=lambda: preemptions.append("cancelled"),
        tap_to_stop_max_audio_sec=tap_to_stop_max_audio_sec,
    )


def _pcm(duration_ms: int = 120, sample: int = 1000) -> bytes:
    samples = int(16_000 * duration_ms / 1000)
    return int(sample).to_bytes(2, "little", signed=True) * samples


class _FakeLocalParticipant:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []

    async def publish_data(self, data: bytes, *, reliable: bool, topic: str) -> None:
        del reliable
        self.published.append((topic, json.loads(data.decode("utf-8"))))


class _FakeSession:
    def __init__(self) -> None:
        self.generate_reply_calls: list[dict] = []
        self.interrupt_calls: list[dict] = []

    def generate_reply(self, **kwargs):
        self.generate_reply_calls.append(kwargs)

    def interrupt(self, *, force: bool):
        self.interrupt_calls.append({"force": force})
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(None)
        return fut


class _FakeFactory:
    def __init__(self, stt: _FakeSttStage) -> None:
        self.stt = stt


def _packet(*, ptt: bool):
    payload = {
        "schema_v": WIRE_SCHEMA_VERSION,
        "type": "client.audio_state",
        "seq": 1,
        "input_mode": "ptt",
        "playback_state": "idle",
        "mic_muted": not ptt,
        "ptt": ptt,
        "rms": 0.1 if ptt else 0.0,
        "client_ts_ms": 1,
    }
    return SimpleNamespace(
        topic=CLIENT_AUDIO_STATE_TOPIC,
        data=json.dumps(payload).encode("utf-8"),
        participant=SimpleNamespace(identity="bench-user"),
    )


def _segment_policy() -> TurnPolicyConfig:
    base = TurnPolicyConfig()
    return replace(
        base,
        ptt=replace(
            base.ptt,
            turn_owner="segment",
            segment_min_audio_ms=10,
            segment_max_audio_ms=2_000,
        ),
    )


def _segment_pipeline(
    stt: _FakeSttStage,
    *,
    observability: ObservabilityConfig | None = None,
) -> HalfDuplexPttPipeline:
    pipeline = HalfDuplexPttPipeline(
        _FakeFactory(stt),
        turn_policy=_segment_policy(),
        observability=observability,
    )
    pipeline._room = SimpleNamespace(
        name="bench-phase_a_ptt_normal_release_commits_001",
        local_participant=_FakeLocalParticipant(),
    )
    pipeline._session = _FakeSession()
    return pipeline


def test_agent_session_supports_segment_ptt_text_reply_entrypoint() -> None:
    generate_reply = inspect.signature(AgentSession.generate_reply)
    assert "user_input" in generate_reply.parameters
    assert "input_modality" in generate_reply.parameters

    room_options = RoomOptions(audio_input=False, audio_output=True, text_output=True)
    assert room_options.audio_input is False
    assert room_options.audio_output is True
    assert room_options.text_output is True


def test_server_uses_segment_pipeline_only_for_half_duplex_segment_owner() -> None:
    from eidolon.livekit.agent.server import _use_segment_ptt_pipeline
    from eidolon_sdk.biz.contracts import INTERACTION_MODE_FULL_DUPLEX, INTERACTION_MODE_HALF_DUPLEX

    segment = _segment_policy()
    streaming = replace(segment, ptt=replace(segment.ptt, turn_owner="streaming"))

    assert _use_segment_ptt_pipeline(INTERACTION_MODE_HALF_DUPLEX, segment) is True
    assert _use_segment_ptt_pipeline(INTERACTION_MODE_HALF_DUPLEX, streaming) is False
    assert _use_segment_ptt_pipeline(INTERACTION_MODE_FULL_DUPLEX, segment) is False


@pytest.mark.asyncio
async def test_empty_ptt_tap_rejects_without_stt_call() -> None:
    stt = _FakeSttStage(offline_text="不应该调用")
    controller = _controller(stt, strategy="auto")

    controller.press()
    result = await controller.release()

    assert result.action == "reject"
    assert result.reason == "empty_audio"
    assert stt.recognize_calls == []
    assert stt.recognize_streaming_calls == []


@pytest.mark.asyncio
async def test_release_transcribes_complete_ptt_audio_once() -> None:
    stt = _FakeSttStage(offline_text="讲个小笑话")
    controller = _controller(stt, strategy="offline")

    controller.press()
    controller.push_frame(_Frame(_pcm(80, sample=900)))
    controller.push_frame(_Frame(_pcm(80, sample=1200)))
    result = await controller.release()

    assert result.action == "commit"
    assert result.transcript == "讲个小笑话"
    assert result.reason == "segment_transcribed"
    assert result.stt_mode == "offline"
    assert len(stt.recognize_calls) == 1
    assert stt.recognize_streaming_calls == []
    assert len(stt.recognize_calls[0]) == len(_pcm(160, sample=900))


@pytest.mark.asyncio
async def test_provider_without_offline_recognize_uses_one_shot_streaming() -> None:
    stt = _FakeSttStage(
        streaming_text="告诉我时间",
        offline_supported=False,
    )
    controller = _controller(stt, strategy="auto")

    controller.press()
    controller.push_frame(_Frame(_pcm(120, sample=1100)))
    result = await controller.release()

    assert result.action == "commit"
    assert result.transcript == "告诉我时间"
    assert result.stt_mode == "streaming"
    assert stt.recognize_calls == []
    assert len(stt.recognize_streaming_calls) == 1


@pytest.mark.asyncio
async def test_default_strategy_uses_one_shot_streaming() -> None:
    stt = _FakeSttStage(
        offline_text="不应该走 offline",
        streaming_text="默认走 streaming",
        offline_supported=True,
    )
    controller = _controller(stt)

    controller.press()
    controller.push_frame(_Frame(_pcm(120, sample=1100)))
    result = await controller.release()

    assert result.action == "commit"
    assert result.transcript == "默认走 streaming"
    assert result.stt_mode == "streaming"
    assert stt.recognize_calls == []
    assert len(stt.recognize_streaming_calls) == 1


@pytest.mark.asyncio
async def test_auto_strategy_falls_back_to_streaming_when_offline_raises() -> None:
    stt = _FakeSttStage(
        streaming_text="走 fallback",
        offline_supported=True,
        offline_raises=True,
    )
    controller = _controller(stt, strategy="auto")

    controller.press()
    controller.push_frame(_Frame(_pcm(120, sample=1200)))
    result = await controller.release()

    assert result.action == "commit"
    assert result.transcript == "走 fallback"
    assert result.stt_mode == "streaming"
    assert len(stt.recognize_calls) == 1
    assert len(stt.recognize_streaming_calls) == 1


@pytest.mark.asyncio
async def test_press_during_agent_output_preempts_and_records_new_turn() -> None:
    preemptions: list[str] = []
    stt = _FakeSttStage(offline_text="不听笑话了，告诉我时间")
    controller = _controller(
        stt,
        agent_output_active=True,
        preemptions=preemptions,
        tap_to_stop_max_audio_sec=0.05,
        strategy="offline",
    )

    pressed = controller.press()
    controller.push_frame(_Frame(_pcm(90, sample=1000)))
    controller.push_frame(_Frame(_pcm(120, sample=1300)))
    result = await controller.release()

    assert pressed.preempted_agent_output is True
    assert preemptions == ["cancelled"]
    assert result.action == "commit"
    assert result.preempted_agent_output is True
    assert result.transcript == "不听笑话了，告诉我时间"
    assert len(stt.recognize_calls) == 1


@pytest.mark.asyncio
async def test_press_during_agent_output_spoken_text_commits() -> None:
    preemptions: list[str] = []
    stt = _FakeSttStage(offline_text="停，不要说了。")
    controller = _controller(
        stt,
        agent_output_active=True,
        preemptions=preemptions,
        tap_to_stop_max_audio_sec=0.05,
        strategy="offline",
    )

    pressed = controller.press()
    controller.push_frame(_Frame(_pcm(320, sample=1200)))
    result = await controller.release()

    assert pressed.preempted_agent_output is True
    assert preemptions == ["cancelled"]
    assert result.action == "commit"
    assert result.reason == "segment_transcribed"
    assert result.transcript == "停，不要说了。"
    assert len(stt.recognize_calls) == 1


@pytest.mark.asyncio
async def test_preempted_short_hold_rejects_as_tap_to_stop_without_stt() -> None:
    preemptions: list[str] = []
    stt = _FakeSttStage(offline_text="不应该调用")
    controller = _controller(
        stt,
        agent_output_active=True,
        preemptions=preemptions,
        tap_to_stop_max_audio_sec=0.35,
    )

    pressed = controller.press()
    controller.push_frame(_Frame(_pcm(180, sample=1200)))
    result = await controller.release()

    assert pressed.preempted_agent_output is True
    assert preemptions == ["cancelled"]
    assert result.action == "reject"
    assert result.reason == "tap_to_stop"
    assert result.preempted_agent_output is True
    assert stt.recognize_calls == []
    assert stt.recognize_streaming_calls == []


@pytest.mark.asyncio
async def test_preempted_buffered_backchannel_rejects_as_tap_to_stop() -> None:
    stt = _FakeSttStage(offline_text="嗯嗯")
    controller = _controller(
        stt,
        agent_output_active=True,
        tap_to_stop_max_audio_sec=0.9,
    )

    controller.press()
    controller.push_frame(_Frame(_pcm(670, sample=1200)))
    result = await controller.release()

    assert result.action == "reject"
    assert result.reason == "tap_to_stop"
    assert result.audio_duration_sec == pytest.approx(0.67, abs=0.01)
    assert stt.recognize_calls == []
    assert stt.recognize_streaming_calls == []


@pytest.mark.asyncio
async def test_pipeline_release_commits_segment_text_to_agent_session() -> None:
    stt = _FakeSttStage(streaming_text="告诉我时间")
    pipeline = _segment_pipeline(stt)

    packet_down = _packet(ptt=True)
    pipeline._room_data.handle_packet(packet_down)
    pipeline._on_room_packet(packet_down)
    pipeline._ptt_controller.push_frame(_Frame(_pcm(120, sample=1200)))

    packet_up = _packet(ptt=False)
    pipeline._room_data.handle_packet(packet_up)
    pipeline._on_room_packet(packet_up)
    await asyncio.gather(*pipeline._turn_tasks)

    session = pipeline._session
    assert isinstance(session, _FakeSession)
    assert session.generate_reply_calls == [
        {"user_input": "告诉我时间", "input_modality": "audio"}
    ]
    local = pipeline._room.local_participant
    outcomes = [
        payload["payload"]["outcome"]
        for topic, payload in local.published
        if payload.get("op") == "ptt.turn_status"
    ]
    assert outcomes[-2:] == ["finalizing", "committed"]


@pytest.mark.asyncio
async def test_pipeline_writes_segment_timeline_on_commit(tmp_path) -> None:
    timeline_path = tmp_path / "turns.jsonl"
    stt = _FakeSttStage(streaming_text="告诉我时间")
    pipeline = _segment_pipeline(
        stt,
        observability=ObservabilityConfig(timeline_debug_path=str(timeline_path)),
    )

    packet_down = _packet(ptt=True)
    pipeline._room_data.handle_packet(packet_down)
    pipeline._on_room_packet(packet_down)
    pipeline._ptt_controller.push_frame(_Frame(_pcm(120, sample=1200)))
    packet_up = _packet(ptt=False)
    pipeline._room_data.handle_packet(packet_up)
    pipeline._on_room_packet(packet_up)
    await asyncio.gather(*pipeline._turn_tasks)

    row = json.loads(timeline_path.read_text(encoding="utf-8").strip())
    attrs = row["attrs"]
    assert attrs["pipeline"] == "half_duplex_ptt_segment"
    assert attrs["ptt_turn_owner"] == "segment"
    assert attrs["ptt_segment_terminal"]["action"] == "commit"
    assert attrs["ptt_segment"]["stt_mode"] == "streaming"
    assert row["timestamps"]["turn_committed_at"] >= row["timestamps"]["speech_stopped_at"]
    ops = [event["op"] for event in attrs["client_control_events"]]
    assert ops[-2:] == ["ptt.turn_status", "ptt.turn_status"]


@pytest.mark.asyncio
async def test_pipeline_rejects_transcription_error_and_resets_controller(tmp_path) -> None:
    timeline_path = tmp_path / "turns.jsonl"
    stt = _FakeSttStage(streaming_error=RuntimeError("stt down"))
    pipeline = _segment_pipeline(
        stt,
        observability=ObservabilityConfig(timeline_debug_path=str(timeline_path)),
    )

    packet_down = _packet(ptt=True)
    pipeline._room_data.handle_packet(packet_down)
    pipeline._on_room_packet(packet_down)
    pipeline._ptt_controller.push_frame(_Frame(_pcm(120, sample=1200)))
    packet_up = _packet(ptt=False)
    pipeline._room_data.handle_packet(packet_up)
    pipeline._on_room_packet(packet_up)
    await asyncio.gather(*pipeline._turn_tasks)

    assert pipeline._ptt_controller.state == "idle"
    local = pipeline._room.local_participant
    outcomes = [
        payload["payload"]["outcome"]
        for _, payload in local.published
        if payload.get("op") == "ptt.turn_status"
    ]
    assert outcomes[-1] == "rejected:transcription_error"
    row = json.loads(timeline_path.read_text(encoding="utf-8").strip())
    assert row["attrs"]["ptt_segment_terminal"] == {
        "action": "reject",
        "reason": "transcription_error",
    }


@pytest.mark.asyncio
async def test_pipeline_records_playback_stop_in_segment_timeline(tmp_path) -> None:
    timeline_path = tmp_path / "turns.jsonl"
    stt = _FakeSttStage()
    pipeline = _segment_pipeline(
        stt,
        observability=ObservabilityConfig(timeline_debug_path=str(timeline_path)),
    )
    pipeline._state = PipelineState.SPEAKING

    packet_down = _packet(ptt=True)
    pipeline._room_data.handle_packet(packet_down)
    pipeline._on_room_packet(packet_down)
    packet_up = _packet(ptt=False)
    pipeline._room_data.handle_packet(packet_up)
    pipeline._on_room_packet(packet_up)
    await asyncio.gather(*pipeline._turn_tasks)

    row = json.loads(timeline_path.read_text(encoding="utf-8").strip())
    attrs = row["attrs"]
    assert attrs["decision"]["action"] == "cancel"
    assert attrs["decision"]["reason"] == "explicit_client_ptt"
    assert attrs["ptt_segment_terminal"]["action"] == "reject"
    assert attrs["ptt_segment_terminal"]["reason"] == "tap_to_stop"
    events = attrs["client_control_events"]
    assert any(event["op"] == "playback.stop" for event in events)


def test_pipeline_observes_existing_subscribed_audio_tracks() -> None:
    stt = _FakeSttStage()
    pipeline = _segment_pipeline(stt)
    track = SimpleNamespace(kind=1)
    participant = SimpleNamespace(
        identity="bench-user",
        track_publications={
            "audio": SimpleNamespace(subscribed=True, track=track),
            "pending": SimpleNamespace(subscribed=False, track=SimpleNamespace(kind=1)),
        },
    )
    observed: list[tuple[object, object]] = []
    pipeline._maybe_start_audio_stream = (  # type: ignore[method-assign]
        lambda track, participant: observed.append((track, participant))
    )

    pipeline._observe_existing_audio_tracks(
        SimpleNamespace(remote_participants={"bench-user": participant})
    )

    assert observed == [(track, participant)]


def test_segment_client_control_without_local_participant_is_not_recorded() -> None:
    stt = _FakeSttStage()
    pipeline = _segment_pipeline(stt)
    pipeline._room = SimpleNamespace(name="room", local_participant=None)
    pipeline._timeline = TurnTimeline("turn-no-local")

    pipeline._publish_client_control("playback.stop", reason="explicit_client_ptt")

    assert "client_control_events" not in pipeline._timeline.attrs


@pytest.mark.asyncio
async def test_pipeline_ptt_press_during_output_interrupts_and_stops_playback() -> None:
    stt = _FakeSttStage(offline_text="")
    pipeline = _segment_pipeline(stt)
    pipeline._state = PipelineState.SPEAKING

    packet_down = _packet(ptt=True)
    pipeline._room_data.handle_packet(packet_down)
    pipeline._on_room_packet(packet_down)
    await asyncio.sleep(0)

    session = pipeline._session
    assert isinstance(session, _FakeSession)
    assert session.interrupt_calls == [{"force": True}]
    local = pipeline._room.local_participant
    ops = [payload.get("op") for _, payload in local.published]
    assert "playback.stop" in ops
