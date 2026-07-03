from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from eidolon.livekit.agent.full_duplex.user_state_handler import (
    FullDuplexUserStateHandler,
)


def _handler():
    calls = SimpleNamespace(
        publish_companion_ui_state=MagicMock(),
        signal_stt_user_away=MagicMock(),
        signal_stt_user_present=MagicMock(),
        handle_speaking_started=MagicMock(),
        handle_speaking_stopped=MagicMock(),
    )
    return (
        FullDuplexUserStateHandler(
            publish_companion_ui_state=calls.publish_companion_ui_state,
            signal_stt_user_away=calls.signal_stt_user_away,
            signal_stt_user_present=calls.signal_stt_user_present,
            handle_speaking_started=calls.handle_speaking_started,
            handle_speaking_stopped=calls.handle_speaking_stopped,
        ),
        calls,
    )


def _event(old: str, new: str) -> SimpleNamespace:
    return SimpleNamespace(old_state=old, new_state=new)


def test_user_state_handler_routes_speech_start() -> None:
    handler, calls = _handler()

    normalized = handler.handle(_event("listening", "speaking"))

    assert normalized.started_speaking is True
    calls.publish_companion_ui_state.assert_called_once_with(
        "listening",
        "user_state:speaking",
    )
    calls.handle_speaking_started.assert_called_once_with()
    calls.handle_speaking_stopped.assert_not_called()
    calls.signal_stt_user_away.assert_not_called()
    calls.signal_stt_user_present.assert_not_called()


def test_user_state_handler_routes_speech_stop() -> None:
    handler, calls = _handler()

    normalized = handler.handle(_event("speaking", "listening"))

    assert normalized.stopped_speaking is True
    calls.publish_companion_ui_state.assert_called_once_with(
        "listening",
        "user_state:listening",
    )
    calls.handle_speaking_stopped.assert_called_once_with()
    calls.handle_speaking_started.assert_not_called()


def test_user_state_handler_signals_stt_away() -> None:
    handler, calls = _handler()

    handler.handle(_event("listening", "away"))

    calls.publish_companion_ui_state.assert_called_once_with("idle", "user_state:away")
    calls.signal_stt_user_away.assert_called_once_with()
    calls.signal_stt_user_present.assert_not_called()
    calls.handle_speaking_started.assert_not_called()
    calls.handle_speaking_stopped.assert_not_called()


def test_user_state_handler_signals_stt_present_after_away() -> None:
    handler, calls = _handler()

    handler.handle(_event("away", "listening"))

    calls.signal_stt_user_present.assert_called_once_with()
    calls.signal_stt_user_away.assert_not_called()
    calls.publish_companion_ui_state.assert_not_called()
    calls.handle_speaking_started.assert_not_called()
    calls.handle_speaking_stopped.assert_not_called()


def test_user_state_handler_ignores_non_boundary_transition() -> None:
    handler, calls = _handler()

    handler.handle(_event("listening", "listening"))

    calls.publish_companion_ui_state.assert_not_called()
    calls.signal_stt_user_away.assert_not_called()
    calls.signal_stt_user_present.assert_not_called()
    calls.handle_speaking_started.assert_not_called()
    calls.handle_speaking_stopped.assert_not_called()
