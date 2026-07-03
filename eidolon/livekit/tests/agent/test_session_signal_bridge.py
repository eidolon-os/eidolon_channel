"""SessionSignalBridge boundary tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from eidolon.livekit.agent.session.signals import SessionSignalBridge


class _FakeVAD:
    def __init__(self) -> None:
        self.callback = None

    def register_inference_callback(self, callback) -> None:
        self.callback = callback


def test_register_vad_callback_forwards_probability_to_eot_and_stt_gate() -> None:
    vad = _FakeVAD()
    stt = SimpleNamespace(notify_vad_state=MagicMock())
    factory = SimpleNamespace(
        vad=SimpleNamespace(vad=vad),
        stt=SimpleNamespace(stt=stt),
    )
    eot = SimpleNamespace(update_vad_probability=MagicMock())
    bridge = SessionSignalBridge(factory=factory, get_eot_model=lambda: eot)

    assert bridge.register_vad_inference_callback() is True
    assert vad.callback is not None

    vad.callback(0.73, True)

    eot.update_vad_probability.assert_called_once_with(0.73)
    stt.notify_vad_state.assert_called_once_with(0.73, 0.0)


def test_register_vad_callback_noops_without_hook() -> None:
    factory = SimpleNamespace(vad=SimpleNamespace(vad=object()), stt=None)
    bridge = SessionSignalBridge(factory=factory, get_eot_model=lambda: object())

    assert bridge.register_vad_inference_callback() is False


def test_signal_stt_user_away_and_present_use_private_stt_slot() -> None:
    stt = SimpleNamespace(
        signal_user_away=MagicMock(),
        signal_user_present=MagicMock(),
    )
    factory = SimpleNamespace(stt=SimpleNamespace(_stt=stt))
    bridge = SessionSignalBridge(factory=factory, get_eot_model=lambda: object())

    bridge.signal_stt_user_away()
    bridge.signal_stt_user_present()

    stt.signal_user_away.assert_called_once_with()
    stt.signal_user_present.assert_called_once_with()


def test_signal_stt_hooks_noop_when_missing() -> None:
    factory = SimpleNamespace(stt=SimpleNamespace(stt=object()))
    bridge = SessionSignalBridge(factory=factory, get_eot_model=lambda: object())

    bridge.signal_stt_user_away()
    bridge.signal_stt_user_present()
