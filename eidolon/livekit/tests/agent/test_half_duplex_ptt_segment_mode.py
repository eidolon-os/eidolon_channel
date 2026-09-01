from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from eidolon_sdk.biz.contracts import (
    CLIENT_AUDIO_STATE_TOPIC,
    LIVEKIT_TRANSCRIPTION_TOPIC,
    SESSION_END_IDLE_NORMAL,
    SESSION_END_PROACTIVE_DONE,
    SESSION_INTENT_PRESENCE,
    SESSION_INTENT_PROACTIVE,
    WIRE_SCHEMA_VERSION,
)
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
from eidolon.livekit.agent.shared.types import PipelineState
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
        delay_sec: float = 0.0,
    ) -> None:
        self.stt = SimpleNamespace(
            capabilities=SimpleNamespace(offline_recognize=offline_supported)
        )
        self.offline_text = offline_text
        self.streaming_text = streaming_text
        self.offline_raises = offline_raises
        self.offline_error = offline_error
        self.streaming_error = streaming_error
        self.delay_sec = delay_sec
        self.recognize_calls: list[bytes] = []
        self.recognize_streaming_calls: list[bytes] = []

    async def recognize(self, audio: bytes) -> str:
        self.recognize_calls.append(audio)
        if self.delay_sec > 0:
            await asyncio.sleep(self.delay_sec)
        if self.offline_error is not None:
            raise self.offline_error
        if self.offline_raises:
            raise NotImplementedError("offline unsupported")
        return self.offline_text

    async def recognize_streaming(self, audio: bytes) -> str:
        self.recognize_streaming_calls.append(audio)
        if self.delay_sec > 0:
            await asyncio.sleep(self.delay_sec)
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
        recorder=PttAudioSegmentRecorder(config=PttAudioSegmentConfig(max_duration_sec=2.0)),
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
        self.turn_decisions: list[dict[str, object]] = []
        self.llm = SimpleNamespace(
            llm=SimpleNamespace(
                set_turn_decision_metadata=self.turn_decisions.append,
            )
        )


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


def _finish_fake_agent_output(pipeline: HalfDuplexPttPipeline) -> None:
    timeline = pipeline._agent_output.active_timeline
    assert timeline is not None
    timeline.mark("brain_request_started_at")
    pipeline._on_agent_state_changed(SimpleNamespace(old_state="speaking", new_state="listening"))


def test_agent_session_supports_segment_ptt_text_reply_entrypoint() -> None:
    generate_reply = inspect.signature(AgentSession.generate_reply)
    assert "user_input" in generate_reply.parameters
    assert "input_modality" in generate_reply.parameters

    room_options = RoomOptions(audio_input=False, audio_output=True, text_output=True)
    assert room_options.audio_input is False
    assert room_options.audio_output is True
    assert room_options.text_output is True


def test_livekit_transcription_topic_matches_sdk_contract() -> None:
    from livekit.agents.types import TOPIC_TRANSCRIPTION

    assert TOPIC_TRANSCRIPTION == LIVEKIT_TRANSCRIPTION_TOPIC


def test_server_uses_ptt_pipeline_only_for_ptt_mode() -> None:
    from eidolon.livekit.agent.server import _use_ptt_pipeline
    from eidolon_sdk.biz.contracts import (
        INTERACTION_MODE_FULL_DUPLEX,
        INTERACTION_MODE_HALF_DUPLEX,
        INTERACTION_MODE_PTT,
    )

    # Only ptt uses the button-driven segment pipeline; half_duplex and
    # full_duplex both run the streaming (EOT-commit) pipeline.
    assert _use_ptt_pipeline(INTERACTION_MODE_PTT) is True
    assert _use_ptt_pipeline(INTERACTION_MODE_HALF_DUPLEX) is False
    assert _use_ptt_pipeline(INTERACTION_MODE_FULL_DUPLEX) is False


def test_half_duplex_idle_policy_uses_session_intent() -> None:
    policy = _segment_policy()
    pipeline = HalfDuplexPttPipeline(
        _FakeFactory(_FakeSttStage()),
        turn_policy=policy,
        session_intent=SESSION_INTENT_PROACTIVE,
    )

    assert pipeline._idle_timeout_sec == (policy.idle.proactive_disconnect_after_idle_ms / 1000.0)
    assert pipeline._idle_end_reason == SESSION_END_PROACTIVE_DONE


def test_half_duplex_uses_shared_session_opening_policy() -> None:
    proactive = HalfDuplexPttPipeline(
        _FakeFactory(_FakeSttStage()),
        welcome_message="Welcome",
        session_intent=SESSION_INTENT_PROACTIVE,
    )
    presence = HalfDuplexPttPipeline(
        _FakeFactory(_FakeSttStage()),
        welcome_message="Welcome",
        session_intent=SESSION_INTENT_PRESENCE,
    )

    assert proactive._welcome_on_enter_text() is None
    assert presence._welcome_on_enter_text() == "Welcome"


@pytest.mark.asyncio
async def test_half_duplex_idle_watchdog_notifies_and_deletes_room() -> None:
    policy = _segment_policy()
    policy = replace(
        policy,
        idle=replace(
            policy.idle,
            disconnect_after_idle_ms=50,
            disconnect_grace_ms=0,
        ),
    )
    on_session_end = AsyncMock()
    on_idle_disconnect = AsyncMock()
    pipeline = HalfDuplexPttPipeline(
        _FakeFactory(_FakeSttStage()),
        turn_policy=policy,
        on_session_end=on_session_end,
        on_idle_disconnect=on_idle_disconnect,
    )
    pipeline._room = SimpleNamespace(
        name="ptt-room",
        local_participant=_FakeLocalParticipant(),
    )
    pipeline._session = SimpleNamespace(agent_state="idle", user_state="listening")

    pipeline._start_idle_watchdog()
    await asyncio.wait_for(pipeline._idle_watchdog_controller.task, timeout=2.0)

    on_session_end.assert_awaited_once_with(SESSION_END_IDLE_NORMAL)
    on_idle_disconnect.assert_awaited_once()
    assert pipeline._session_closed_event.is_set()
    assert pipeline._idle_disconnect_started is True


@pytest.mark.asyncio
async def test_half_duplex_idle_watchdog_rearms_while_ptt_turn_is_busy() -> None:
    policy = _segment_policy()
    policy = replace(
        policy,
        idle=replace(
            policy.idle,
            disconnect_after_idle_ms=50,
            disconnect_grace_ms=0,
        ),
    )
    on_idle_disconnect = AsyncMock()
    pipeline = HalfDuplexPttPipeline(
        _FakeFactory(_FakeSttStage()),
        turn_policy=policy,
        on_idle_disconnect=on_idle_disconnect,
    )
    pipeline._room = SimpleNamespace(
        name="ptt-room",
        local_participant=_FakeLocalParticipant(),
    )
    pipeline._session = SimpleNamespace(agent_state="idle", user_state="listening")

    pipeline._ptt_controller.press()
    pipeline._start_idle_watchdog()
    await asyncio.sleep(0.2)

    assert pipeline._idle_watchdog_controller.task is not None
    assert not pipeline._idle_watchdog_controller.task.done()
    on_idle_disconnect.assert_not_awaited()

    await pipeline._ptt_controller.release()
    await asyncio.wait_for(pipeline._idle_watchdog_controller.task, timeout=2.0)
    on_idle_disconnect.assert_awaited_once()


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
async def test_duplicate_press_while_recording_is_idempotent() -> None:
    preemptions: list[str] = []
    stt = _FakeSttStage(offline_text="第一段和第二段都应该保留")
    controller = _controller(
        stt,
        agent_output_active=True,
        preemptions=preemptions,
        tap_to_stop_max_audio_sec=0.05,
        strategy="offline",
    )

    first_press = controller.press()
    controller.push_frame(_Frame(_pcm(80, sample=900)))
    duplicate_press = controller.press()
    controller.push_frame(_Frame(_pcm(90, sample=1300)))
    result = await controller.release()

    assert first_press.preempted_agent_output is True
    assert duplicate_press.reason == "already_recording"
    assert duplicate_press.preempted_agent_output is True
    assert preemptions == ["cancelled"]
    assert result.action == "commit"
    assert len(stt.recognize_calls) == 1
    assert len(stt.recognize_calls[0]) == len(_pcm(170, sample=900))


@pytest.mark.asyncio
async def test_press_while_transcribing_rejects_busy_without_new_segment() -> None:
    stt = _FakeSttStage(streaming_text="旧问题", delay_sec=0.01)
    controller = _controller(stt, strategy="streaming")

    controller.press()
    controller.push_frame(_Frame(_pcm(120, sample=1200)))
    release_task = asyncio.create_task(controller.release())
    await asyncio.sleep(0)

    busy = controller.press()

    assert busy.action == "reject"
    assert busy.reason == "busy_transcribing"
    assert busy.state == "transcribing"
    assert controller.push_frame(_Frame(_pcm(120, sample=1300))) is False

    result = await release_task
    assert result.action == "commit"
    assert result.transcript == "旧问题"
    assert len(stt.recognize_streaming_calls) == 1


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
async def test_preempted_backchannel_with_control_lead_rejects_as_tap_to_stop() -> None:
    stt = _FakeSttStage(streaming_text="嗯嗯")
    controller = _controller(
        stt,
        agent_output_active=True,
        tap_to_stop_max_audio_sec=0.9,
    )

    controller.press()
    controller.push_frame(_Frame(_pcm(640, sample=0) + _pcm(670, sample=1200)))
    result = await controller.release()

    assert result.action == "reject"
    assert result.reason == "tap_to_stop"
    assert result.audio_duration_sec == pytest.approx(1.31, abs=0.02)
    assert result.audio_effective_duration_sec == pytest.approx(0.67, abs=0.02)
    assert result.audio_leading_silence_sec == pytest.approx(0.64, abs=0.02)
    assert stt.recognize_calls == []
    assert stt.recognize_streaming_calls == []


@pytest.mark.asyncio
async def test_preempted_long_speech_after_control_lead_commits() -> None:
    stt = _FakeSttStage(streaming_text="继续讲这个方案")
    controller = _controller(
        stt,
        agent_output_active=True,
        tap_to_stop_max_audio_sec=0.9,
    )

    controller.press()
    controller.push_frame(_Frame(_pcm(640, sample=0) + _pcm(1000, sample=1200)))
    result = await controller.release()

    assert result.action == "commit"
    assert result.reason == "segment_transcribed"
    assert result.transcript == "继续讲这个方案"
    assert result.audio_duration_sec == pytest.approx(1.64, abs=0.02)
    assert result.audio_effective_duration_sec == pytest.approx(1.0, abs=0.02)
    assert result.audio_leading_silence_sec == pytest.approx(0.64, abs=0.02)
    assert len(stt.recognize_streaming_calls) == 1


@pytest.mark.asyncio
async def test_tap_to_stop_then_next_ptt_commit_uses_clean_segment() -> None:
    preemptions: list[str] = []
    agent_output_active = True
    stt = _FakeSttStage(streaming_text="告诉我时间")
    controller = HalfDuplexPttTurnController(
        recorder=PttAudioSegmentRecorder(config=PttAudioSegmentConfig(max_duration_sec=2.0)),
        transcriber=PttSegmentTranscriber(
            stt,
            config=PttSegmentTranscriberConfig(
                strategy="streaming",
                min_audio_duration_sec=0.01,
            ),
        ),
        agent_output_active=lambda: agent_output_active,
        preempt_agent_output=lambda: preemptions.append("cancelled"),
        tap_to_stop_max_audio_sec=0.35,
    )

    first_press = controller.press()
    controller.push_frame(_Frame(_pcm(160, sample=1200)))
    first = await controller.release()

    assert first_press.preempted_agent_output is True
    assert first.action == "reject"
    assert first.reason == "tap_to_stop"
    assert controller.state == "idle"
    assert preemptions == ["cancelled"]
    assert stt.recognize_streaming_calls == []

    agent_output_active = False
    second_press = controller.press()
    controller.push_frame(_Frame(_pcm(120, sample=1100)))
    second = await controller.release()

    assert second_press.preempted_agent_output is False
    assert second.action == "commit"
    assert second.transcript == "告诉我时间"
    assert second.preempted_agent_output is False
    assert controller.state == "idle"
    assert len(stt.recognize_streaming_calls) == 1


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
    assert session.generate_reply_calls == [{"user_input": "告诉我时间", "input_modality": "audio"}]
    turn_decisions = pipeline._factory.turn_decisions
    assert len(turn_decisions) == 1
    assert turn_decisions[0]["decision"] == "commit"
    assert turn_decisions[0]["evidence"]["boundary"] == "ptt_segment_commit"
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

    assert not timeline_path.exists()
    _finish_fake_agent_output(pipeline)

    row = json.loads(timeline_path.read_text(encoding="utf-8").strip())
    attrs = row["attrs"]
    assert attrs["pipeline"] == "half_duplex_ptt_segment"
    assert attrs["interaction_mode"] == "ptt"
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


@pytest.mark.asyncio
async def test_pipeline_tap_to_stop_then_next_ptt_commit_has_clean_timeline(tmp_path) -> None:
    timeline_path = tmp_path / "turns.jsonl"
    stt = _FakeSttStage(streaming_text="告诉我时间")
    pipeline = _segment_pipeline(
        stt,
        observability=ObservabilityConfig(timeline_debug_path=str(timeline_path)),
    )

    pipeline._state = PipelineState.SPEAKING
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
    assert session.generate_reply_calls == []
    assert stt.recognize_streaming_calls == []

    pipeline._state = PipelineState.IDLE
    packet_down = _packet(ptt=True)
    pipeline._room_data.handle_packet(packet_down)
    pipeline._on_room_packet(packet_down)
    pipeline._ptt_controller.push_frame(_Frame(_pcm(140, sample=1100)))
    packet_up = _packet(ptt=False)
    pipeline._room_data.handle_packet(packet_up)
    pipeline._on_room_packet(packet_up)
    await asyncio.gather(*pipeline._turn_tasks)

    assert session.generate_reply_calls == [{"user_input": "告诉我时间", "input_modality": "audio"}]
    _finish_fake_agent_output(pipeline)
    rows = [json.loads(line) for line in timeline_path.read_text(encoding="utf-8").splitlines()]
    assert [row["attrs"]["ptt_segment_terminal"]["action"] for row in rows] == [
        "reject",
        "commit",
    ]
    assert rows[0]["attrs"]["ptt_segment_terminal"]["reason"] == "tap_to_stop"
    second_events = rows[1]["attrs"]["client_control_events"]
    assert not any(event["op"] == "playback.stop" for event in second_events)


@pytest.mark.asyncio
async def test_new_ptt_segment_does_not_steal_committed_output_owner(tmp_path) -> None:
    timeline_path = tmp_path / "turns.jsonl"
    stt = _FakeSttStage(streaming_text="告诉我时间")
    pipeline = _segment_pipeline(
        stt,
        observability=ObservabilityConfig(timeline_debug_path=str(timeline_path)),
    )

    first_down = _packet(ptt=True)
    pipeline._room_data.handle_packet(first_down)
    pipeline._on_room_packet(first_down)
    pipeline._ptt_controller.push_frame(_Frame(_pcm(500, sample=1200)))
    first_up = _packet(ptt=False)
    pipeline._room_data.handle_packet(first_up)
    pipeline._on_room_packet(first_up)
    await asyncio.gather(*pipeline._turn_tasks)

    output = pipeline._agent_output.active_timeline
    assert output is not None
    assert pipeline._timeline is None

    pipeline._state = PipelineState.SPEAKING
    second_down = _packet(ptt=True)
    pipeline._room_data.handle_packet(second_down)
    pipeline._on_room_packet(second_down)

    candidate = pipeline._timeline
    assert candidate is not None
    assert candidate is not output
    assert pipeline._agent_output.active_timeline is output

    pipeline._ptt_controller.push_frame(_Frame(_pcm(120, sample=1200)))
    second_up = _packet(ptt=False)
    pipeline._room_data.handle_packet(second_up)
    pipeline._on_room_packet(second_up)
    await asyncio.gather(*pipeline._turn_tasks)

    assert pipeline._agent_output.active_timeline is output
    output.mark("brain_cancelled_at")
    pipeline._on_agent_state_changed(SimpleNamespace(old_state="speaking", new_state="listening"))

    rows = [json.loads(line) for line in timeline_path.read_text().splitlines()]
    assert {row["turn_id"] for row in rows} == {candidate.turn_id, output.turn_id}
    assert pipeline._agent_output.active_timeline is None


@pytest.mark.asyncio
async def test_pipeline_busy_press_during_finalizing_does_not_fake_release(
    tmp_path,
) -> None:
    timeline_path = tmp_path / "turns.jsonl"
    stt = _FakeSttStage(streaming_text="旧问题", delay_sec=0.01)
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
    await asyncio.sleep(0)

    busy_press = _packet(ptt=True)
    pipeline._room_data.handle_packet(busy_press)
    pipeline._on_room_packet(busy_press)
    busy_release = _packet(ptt=False)
    pipeline._room_data.handle_packet(busy_release)
    pipeline._on_room_packet(busy_release)
    await asyncio.gather(*pipeline._turn_tasks)

    local = pipeline._room.local_participant
    outcomes = [
        payload["payload"]["outcome"]
        for _, payload in local.published
        if payload.get("op") == "ptt.turn_status"
    ]
    assert outcomes.count("finalizing") == 1
    assert "rejected:busy_transcribing" in outcomes
    assert outcomes[-1] == "committed"
    assert len(stt.recognize_streaming_calls) == 1


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
