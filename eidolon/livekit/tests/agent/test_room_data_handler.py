"""RoomDataHandler tests."""

from __future__ import annotations

import time
from dataclasses import replace
from types import SimpleNamespace

from eidolon.livekit.agent.client_audio_state import CLIENT_AUDIO_STATE_TOPIC
from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session import RoomDataHandler


def _packet(
    *,
    topic: str = CLIENT_AUDIO_STATE_TOPIC,
    identity: str = "alice",
    data: bytes | None = None,
):
    payload = data or (
        b'{"type":"client.audio_state","input_mode":"auto",'
        b'"playback_state":"agent_speaking","mic_muted":true}'
    )
    return SimpleNamespace(
        topic=topic,
        data=payload,
        participant=SimpleNamespace(identity=identity),
    )


def test_room_data_handler_records_client_audio_state_into_timeline() -> None:
    timeline = TurnTimeline("turn-room-data")
    handler = RoomDataHandler(get_timeline=lambda: timeline)
    packet = _packet(identity="alice")

    handler.handle_packet(packet)

    state = handler.client_audio_states["alice"]
    snap = timeline.snapshot()
    assert state.mic_muted is True
    assert snap["attrs"]["client_audio_state"]["participant_identity"] == "alice"
    assert snap["attrs"]["client_audio_state_packet_count"] == 1
    assert snap["attrs"]["room_data_events"][-1] == {
        "topic": CLIENT_AUDIO_STATE_TOPIC,
        "participant_identity": "alice",
        "bytes": len(packet.data),
    }


def test_room_data_handler_ignores_malformed_client_audio_state() -> None:
    timeline = TurnTimeline("turn-room-data-bad")
    handler = RoomDataHandler(get_timeline=lambda: timeline)

    handler.handle_packet(_packet(data=b"{bad json"))

    assert handler.client_audio_states == {}
    assert "client_audio_state" not in timeline.attrs
    assert timeline.attrs["room_data_packet_count"] == 1


def test_room_data_handler_prefers_fresh_matching_participant() -> None:
    handler = RoomDataHandler(get_timeline=lambda: None)
    now = time.monotonic()
    handler.handle_packet(
        _packet(
            identity="alice",
            data=(
                b'{"type":"client.audio_state","playback_state":"agent_speaking"}'
            ),
        )
    )
    handler.handle_packet(
        _packet(
            identity="bob",
            data=b'{"type":"client.audio_state","playback_state":"idle"}',
        )
    )
    handler.client_audio_states["alice"] = replace(
        handler.client_audio_states["alice"],
        received_at=now - 0.1,
    )
    handler.client_audio_states["bob"] = replace(
        handler.client_audio_states["bob"],
        received_at=now,
    )

    state = handler.latest_client_audio_state(
        participant_identity="alice",
        max_age_sec=2.0,
        now=now,
    )

    assert state is handler.client_audio_states["alice"]


def test_room_data_handler_falls_back_to_newest_fresh_state() -> None:
    handler = RoomDataHandler(get_timeline=lambda: None)
    now = time.monotonic()
    handler.handle_packet(_packet(identity="alice"))
    handler.handle_packet(_packet(identity="bob"))
    handler.client_audio_states["alice"] = replace(
        handler.client_audio_states["alice"],
        received_at=now - 0.1,
    )
    handler.client_audio_states["bob"] = replace(
        handler.client_audio_states["bob"],
        received_at=now,
    )

    state = handler.latest_client_audio_state(
        participant_identity="missing",
        max_age_sec=2.0,
        now=now,
    )

    assert state is handler.client_audio_states["bob"]


class _FakeRoom:
    """Minimal stand-in for a LiveKit Room that records on(...) handlers."""

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


def test_install_invokes_on_packet_after_handle() -> None:
    # Regression: the explicit-interrupt fast path is wired through this on_packet
    # hook. It must run on every data_received packet, AFTER handle_packet has
    # stored the latest state (so the interrupt handler sees it).
    handler = RoomDataHandler(get_timeline=lambda: None)
    room = _FakeRoom()
    seen: list = []

    handler.install(
        room,
        on_packet=lambda pkt: seen.append("alice" in handler.client_audio_states),
    )
    room.emit("data_received", _packet(identity="alice"))

    assert "alice" in handler.client_audio_states  # handle_packet ran
    assert seen == [True]  # on_packet ran once, after the state was stored


def test_install_without_on_packet_still_handles() -> None:
    handler = RoomDataHandler(get_timeline=lambda: None)
    room = _FakeRoom()

    handler.install(room)
    room.emit("data_received", _packet(identity="bob"))

    assert "bob" in handler.client_audio_states


def test_install_on_packet_exception_does_not_break_handling() -> None:
    handler = RoomDataHandler(get_timeline=lambda: None)
    room = _FakeRoom()

    def _boom(_pkt) -> None:
        raise RuntimeError("observer blew up")

    handler.install(room, on_packet=_boom)
    room.emit("data_received", _packet(identity="carol"))  # must not raise

    assert "carol" in handler.client_audio_states
