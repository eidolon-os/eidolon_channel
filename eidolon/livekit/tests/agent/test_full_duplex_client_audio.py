"""Full-duplex client audio-state view tests."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from eidolon_sdk.biz.contracts import CLIENT_AUDIO_STATE_TOPIC, PLAYBACK_STATE_AGENT_SPEAKING

from eidolon.livekit.agent.full_duplex.client_audio import (
    FullDuplexClientAudioStateView,
    FullDuplexRoomDataBridge,
)
from eidolon.livekit.agent.integration.client_audio_state import ClientAudioState
from eidolon.livekit.agent.pipeline.types import PipelineState
from eidolon.livekit.agent.session.room_data import RoomDataHandler
from eidolon.livekit.common.config import TurnPolicyConfig


def _view(
    *,
    room_data: RoomDataHandler,
    pipeline_state: PipelineState = PipelineState.IDLE,
    ducking: object | None = None,
) -> FullDuplexClientAudioStateView:
    return FullDuplexClientAudioStateView(
        ensure_runtime_defaults=lambda: None,
        ensure_room_data=lambda: None,
        ensure_ducking=lambda: None,
        get_room_data=lambda: room_data,
        get_turn_policy=TurnPolicyConfig,
        get_pipeline_state=lambda: pipeline_state,
        get_ducking=lambda: ducking
        or SimpleNamespace(is_cancelled=False, is_suspended=False),
    )


def test_pipeline_speaking_counts_as_active_output() -> None:
    room_data = RoomDataHandler(get_timeline=lambda: None)
    view = _view(room_data=room_data, pipeline_state=PipelineState.SPEAKING)

    assert view.agent_output_active_for_interrupts() is True


def test_cancelled_ducking_suppresses_active_output() -> None:
    room_data = RoomDataHandler(get_timeline=lambda: None)
    view = _view(
        room_data=room_data,
        pipeline_state=PipelineState.SPEAKING,
        ducking=SimpleNamespace(is_cancelled=True, is_suspended=False),
    )

    assert view.agent_output_active_for_interrupts() is False


def test_fresh_client_playback_counts_as_active_output() -> None:
    room_data = RoomDataHandler(get_timeline=lambda: None)
    room_data.client_audio_states["dev1"] = ClientAudioState(
        participant_identity="dev1",
        playback_state=PLAYBACK_STATE_AGENT_SPEAKING,
        received_at=time.monotonic(),
    )
    view = _view(room_data=room_data, pipeline_state=PipelineState.IDLE)

    assert view.agent_output_active_for_interrupts(participant_identity="dev1") is True
    assert view.latest_state(participant_identity="dev1") is room_data.client_audio_states[
        "dev1"
    ]


class _FakeRoom:
    def __init__(self) -> None:
        self._handlers: dict[str, list] = {}

    def on(self, event: str):
        def _register(fn):
            self._handlers.setdefault(event, []).append(fn)
            return fn

        return _register

    def emit(self, event: str, packet) -> None:
        for fn in self._handlers.get(event, []):
            fn(packet)


def test_room_data_bridge_calls_preempt_after_room_data_stores_state() -> None:
    room_data = RoomDataHandler(get_timeline=lambda: None)
    preempts = SimpleNamespace(on_client_room_packet=MagicMock())
    bridge = FullDuplexRoomDataBridge(
        ensure_room_data=lambda: None,
        ensure_client_preempts=lambda: None,
        get_room_data=lambda: room_data,
        get_client_preempts=lambda: preempts,
    )
    room = _FakeRoom()
    packet = SimpleNamespace(
        topic=CLIENT_AUDIO_STATE_TOPIC,
        data=b'{"type":"client.audio_state","playback_state":"agent_speaking"}',
        participant=SimpleNamespace(identity="dev1"),
    )

    bridge.install(room)
    room.emit("data_received", packet)

    assert room_data.latest_client_audio_state(participant_identity="dev1") is not None
    preempts.on_client_room_packet.assert_called_once_with(packet)
