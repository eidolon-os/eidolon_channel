"""SoftInterruptController tests."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from eidolon.livekit.agent.session import SoftInterruptController


@pytest.mark.asyncio
async def test_soft_interrupt_controller_cancel_stops_pending_timeout() -> None:
    on_timeout = MagicMock()
    controller = SoftInterruptController(timeout_sec=0.05, on_timeout=on_timeout)

    controller.enter()
    controller.cancel()
    await asyncio.sleep(0.08)

    assert controller.active is False
    assert controller.task is None
    on_timeout.assert_not_called()


@pytest.mark.asyncio
async def test_soft_interrupt_controller_timeout_upgrades() -> None:
    on_timeout = MagicMock()
    controller = SoftInterruptController(timeout_sec=0.01, on_timeout=on_timeout)

    controller.enter()
    assert controller.task is not None
    await asyncio.wait_for(controller.task, timeout=1.0)

    assert controller.active is False
    assert controller.task is None
    on_timeout.assert_called_once()
