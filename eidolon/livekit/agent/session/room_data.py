"""Room data packet handling for a LiveKit streaming session."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from eidolon_sdk.biz.contracts import CLIENT_AUDIO_STATE_TOPIC

from eidolon.livekit.agent.integration.client_audio_state import (
    ClientAudioState,
    parse_client_audio_state,
)
from eidolon.livekit.agent.observability import TurnTimeline

logger = logging.getLogger("agent.session.room_data")


def participant_identity_from_packet(packet: Any) -> str:
    participant = getattr(packet, "participant", None)
    return getattr(participant, "identity", "") or "unknown"


class RoomDataHandler:
    """Observe room data packets and maintain client-side audio state hints."""

    def __init__(
        self,
        *,
        get_timeline: Callable[[], TurnTimeline | None],
    ) -> None:
        self._get_timeline = get_timeline
        self.client_audio_states: dict[str, ClientAudioState] = {}
        self.room_data_packet_count = 0
        self.client_audio_state_packet_count = 0

    def install(
        self,
        room: Any,
        *,
        on_packet: Callable[[Any], None] | None = None,
    ) -> None:
        logger.info(
            "[RoomDataHandler] installed for topic=%s",
            CLIENT_AUDIO_STATE_TOPIC,
        )

        @room.on("data_received")
        def _on_data_received(packet: Any) -> None:
            self.handle_packet(packet)
            # Post-handle observer (e.g. the explicit-interrupt fast path). Runs
            # after the latest client audio state is stored. Never let it break
            # packet handling.
            if on_packet is not None:
                try:
                    on_packet(packet)
                except Exception:  # noqa: BLE001 - defensive: observer must not crash the room callback
                    logger.exception("[RoomDataHandler] on_packet observer failed")

    def handle_packet(self, packet: Any) -> None:
        topic = getattr(packet, "topic", None)
        self.room_data_packet_count += 1
        packet_count = self.room_data_packet_count
        timeline = self._get_timeline()
        if timeline is not None:
            events = list(timeline.attrs.get("room_data_events") or ())
            events.append(
                {
                    "topic": topic,
                    "participant_identity": participant_identity_from_packet(packet),
                    "bytes": len(getattr(packet, "data", b"") or b""),
                }
            )
            timeline.set_attr("room_data_events", events[-12:])
            timeline.set_attr("room_data_packet_count", packet_count)
        if packet_count <= 3 or topic == CLIENT_AUDIO_STATE_TOPIC:
            logger.debug(
                "[RoomDataHandler] room data received topic=%s identity=%s bytes=%d count=%d",
                topic,
                participant_identity_from_packet(packet),
                len(getattr(packet, "data", b"") or b""),
                packet_count,
            )
        if topic != CLIENT_AUDIO_STATE_TOPIC:
            return
        identity = participant_identity_from_packet(packet)
        try:
            state = parse_client_audio_state(
                getattr(packet, "data", b""),
                participant_identity=identity,
            )
        except ValueError:
            logger.debug(
                "[RoomDataHandler] ignored malformed client.audio_state packet",
                exc_info=True,
            )
            return
        self.client_audio_states[identity] = state
        self.client_audio_state_packet_count += 1
        client_packet_count = self.client_audio_state_packet_count
        if timeline is not None:
            timeline.set_attr(
                "client_audio_state",
                state.as_timeline_attr(),
            )
            timeline.set_attr(
                "client_audio_state_packet_count",
                client_packet_count,
            )
        logger.info(
            "[RoomDataHandler] client.audio_state received identity=%s "
            "playback=%s mic_muted=%s manual_interrupt=%s ptt=%s count=%d",
            state.participant_identity,
            state.playback_state,
            state.mic_muted,
            state.manual_interrupt,
            state.ptt,
            client_packet_count,
        )

    def latest_client_audio_state(
        self,
        *,
        participant_identity: str | None = None,
        max_age_sec: float = 2.0,
        now: float | None = None,
    ) -> ClientAudioState | None:
        if not self.client_audio_states:
            return None
        ref = time.monotonic() if now is None else now
        if participant_identity:
            state = self.client_audio_states.get(participant_identity)
            if state is not None and state.is_fresh(
                now=ref,
                max_age_sec=max_age_sec,
            ):
                return state
        states = [
            state
            for state in self.client_audio_states.values()
            if state.is_fresh(now=ref, max_age_sec=max_age_sec)
        ]
        if not states:
            return None
        return max(states, key=lambda state: state.received_at)
