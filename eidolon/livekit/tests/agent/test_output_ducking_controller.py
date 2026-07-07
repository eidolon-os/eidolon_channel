"""OutputDuckingController boundary tests."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from livekit import rtc
from livekit.agents.voice import io as lk_io

from eidolon.livekit.agent.output import OutputController, OutputDuckingController


class _FakeInnerOutput(lk_io.AudioOutput):
    def __init__(self) -> None:
        super().__init__(
            label="fake-inner",
            capabilities=lk_io.AudioOutputCapabilities(pause=True),
            sample_rate=16000,
        )

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)

    def flush(self) -> None:
        super().flush()

    def clear_buffer(self) -> None:
        pass


def _cfg(**overrides):
    values = {
        "duck_enabled": True,
        "duck_fade_ms": 30,
        "duck_fade_in_ms": 30,
        "duck_suspend_volume": 0.0,
        "duck_buffer_max_sec": 2.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _session(inner=None):
    return SimpleNamespace(output=SimpleNamespace(audio=inner))


def test_install_wraps_session_audio_output() -> None:
    inner = _FakeInnerOutput()
    session = _session(inner)
    controller = OutputDuckingController()

    mixer = controller.install(session, _cfg())

    assert isinstance(mixer, OutputController)
    assert controller.mixer is mixer
    assert session.output.audio is mixer
    assert controller.installed is True


def test_install_skips_when_ducking_disabled() -> None:
    inner = _FakeInnerOutput()
    session = _session(inner)
    controller = OutputDuckingController()

    mixer = controller.install(session, _cfg(duck_enabled=False))

    assert mixer is None
    assert controller.mixer is None
    assert session.output.audio is inner


@pytest.mark.asyncio
async def test_cancel_timeout_cancels_active_task() -> None:
    controller = OutputDuckingController()
    controller.timeout_task = asyncio.create_task(asyncio.sleep(10))

    controller.cancel_timeout()
    await asyncio.sleep(0)

    assert controller.timeout_task is None


def test_cancel_and_reset_state_transitions() -> None:
    controller = OutputDuckingController()
    controller.install(_session(_FakeInnerOutput()), _cfg())

    controller.cancel_output()
    assert controller.is_cancelled is True

    assert controller.reset_if_cancelled() is True
    assert controller.is_cancelled is False


def test_unduck_only_when_suspended() -> None:
    controller = OutputDuckingController()
    controller.install(_session(_FakeInnerOutput()), _cfg())

    assert controller.unduck_if_suspended() is False

    assert controller.duck(now=123.0) is True
    assert controller.is_suspended is True
    assert controller.unduck_if_suspended(drop_buffered=True) is True
    assert controller.is_suspended is False
    assert controller.last_unduck_time > 0


def test_suspended_passthrough_only_enables_while_suspended() -> None:
    controller = OutputDuckingController()
    controller.install(_session(_FakeInnerOutput()), _cfg())

    assert controller.enable_suspended_passthrough(volume=0.25) is False

    assert controller.duck(now=123.0) is True
    assert controller.enable_suspended_passthrough(volume=0.25) is True
    assert controller.mixer is not None
    assert controller.mixer.suspended_passthrough_volume == 0.25


def test_duck_returns_false_when_cancelled_output_cannot_suspend() -> None:
    controller = OutputDuckingController()
    controller.install(_session(_FakeInnerOutput()), _cfg())

    controller.cancel_output()

    assert controller.duck(now=123.0) is False
    assert controller.is_cancelled is True
