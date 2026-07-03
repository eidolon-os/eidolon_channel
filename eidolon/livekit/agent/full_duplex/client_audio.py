"""Full-duplex client audio-state view and room-data bridge."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from eidolon_sdk.biz.contracts import PLAYBACK_STATE_AGENT_SPEAKING

from eidolon.livekit.agent.integration.client_audio_state import ClientAudioState
from eidolon.livekit.agent.pipeline.types import PipelineState
from eidolon.livekit.common.config import TurnPolicyConfig

from ..session.room_data import RoomDataHandler
from .client_preempt import ExplicitClientPreemptHandler


class FullDuplexClientAudioStateView:
    """Read full-duplex client audio hints with one freshness contract."""

    def __init__(
        self,
        *,
        ensure_runtime_defaults: Callable[[], None],
        ensure_room_data: Callable[[], None],
        ensure_ducking: Callable[[], None],
        get_room_data: Callable[[], RoomDataHandler],
        get_turn_policy: Callable[[], TurnPolicyConfig],
        get_pipeline_state: Callable[[], PipelineState],
        get_ducking: Callable[[], Any],
    ) -> None:
        self._ensure_runtime_defaults = ensure_runtime_defaults
        self._ensure_room_data = ensure_room_data
        self._ensure_ducking = ensure_ducking
        self._get_room_data = get_room_data
        self._get_turn_policy = get_turn_policy
        self._get_pipeline_state = get_pipeline_state
        self._get_ducking = get_ducking

    def latest_state(
        self,
        *,
        participant_identity: str | None = None,
    ) -> ClientAudioState | None:
        self._ensure_runtime_defaults()
        max_age_sec = self._get_turn_policy().attention.client_state_max_age_ms / 1000.0
        self._ensure_room_data()
        return self._get_room_data().latest_client_audio_state(
            participant_identity=participant_identity,
            max_age_sec=max_age_sec,
        )

    def agent_output_active_for_interrupts(
        self,
        *,
        participant_identity: str | None = None,
    ) -> bool:
        """Return true when user speech should be evaluated as an interrupt."""

        self._ensure_ducking()
        ducking = self._get_ducking()
        if bool(getattr(ducking, "is_cancelled", False)):
            return False
        if self._get_pipeline_state() == PipelineState.SPEAKING or bool(
            getattr(ducking, "is_suspended", False)
        ):
            return True

        self._ensure_room_data()
        room_data = self._get_room_data()
        states = room_data.client_audio_states
        if not states:
            return False

        max_age_sec = self._get_turn_policy().attention.client_state_max_age_ms / 1000.0
        now = time.monotonic()
        if participant_identity:
            client = states.get(participant_identity)
            if (
                client is not None
                and client.is_fresh(now=now, max_age_sec=max_age_sec)
                and client.playback_state == PLAYBACK_STATE_AGENT_SPEAKING
            ):
                return True
        return any(
            state.is_fresh(now=now, max_age_sec=max_age_sec)
            and state.playback_state == PLAYBACK_STATE_AGENT_SPEAKING
            for state in states.values()
        )


class FullDuplexRoomDataBridge:
    """Wire room data packets to full-duplex client-control side effects."""

    def __init__(
        self,
        *,
        ensure_room_data: Callable[[], None],
        ensure_client_preempts: Callable[[], None],
        get_room_data: Callable[[], RoomDataHandler],
        get_client_preempts: Callable[[], ExplicitClientPreemptHandler],
    ) -> None:
        self._ensure_room_data = ensure_room_data
        self._ensure_client_preempts = ensure_client_preempts
        self._get_room_data = get_room_data
        self._get_client_preempts = get_client_preempts

    def install(self, room: Any) -> None:
        self._ensure_room_data()
        self._ensure_client_preempts()
        self._get_room_data().install(
            room,
            on_packet=self._get_client_preempts().on_client_room_packet,
        )
