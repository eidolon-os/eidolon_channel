from __future__ import annotations

from types import SimpleNamespace

from eidolon.livekit.agent.full_duplex.user_state_event import (
    FullDuplexUserStateEvent,
)


def test_user_state_event_normalizes_livekit_event_shape() -> None:
    event = SimpleNamespace(old_state="listening", new_state="speaking")

    normalized = FullDuplexUserStateEvent.from_event(event)

    assert normalized.old_state == "listening"
    assert normalized.new_state == "speaking"
    assert normalized.started_speaking is True
    assert normalized.stopped_speaking is False


def test_user_state_event_detects_speech_stop() -> None:
    normalized = FullDuplexUserStateEvent.from_event(
        SimpleNamespace(old_state="speaking", new_state="listening")
    )

    assert normalized.started_speaking is False
    assert normalized.stopped_speaking is True


def test_user_state_event_handles_missing_fields() -> None:
    normalized = FullDuplexUserStateEvent.from_event(SimpleNamespace())

    assert normalized.old_state == ""
    assert normalized.new_state == ""
    assert normalized.started_speaking is False
    assert normalized.stopped_speaking is False


def test_user_state_event_from_event_is_idempotent() -> None:
    event = FullDuplexUserStateEvent(old_state="away", new_state="listening")

    assert FullDuplexUserStateEvent.from_event(event) is event
