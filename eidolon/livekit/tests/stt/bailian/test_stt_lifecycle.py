from __future__ import annotations

import asyncio

import pytest
from livekit import rtc

from eidolon.livekit.plugins.stt.bailian.connection_manager import (
    BailianConnectionError,
    BailianConnectionManager,
)
from eidolon.livekit.plugins.stt.bailian.stt import BailianFunASRSTT


class _FakeWebSocket:
    def __init__(self) -> None:
        self.closed = False
        self.close_code: int | None = None

    async def close(self, code: int) -> None:
        self.closed = True
        self.close_code = code


@pytest.mark.asyncio
async def test_close_closes_ws_even_before_handshake_connected() -> None:
    conn = BailianConnectionManager(
        api_url="ws://example.invalid",
        api_key="test-key",
        model="fun-asr-realtime-2026-02-28",
    )
    ws = _FakeWebSocket()
    conn._ws = ws  # noqa: SLF001
    conn._connected = False  # noqa: SLF001

    await conn.close(code=1000)

    assert ws.closed is True
    assert ws.close_code == 1000
    assert conn._ws is None  # noqa: SLF001
    assert conn._connected is False  # noqa: SLF001


@pytest.mark.asyncio
async def test_batch_recognize_closes_connection_when_send_fails(monkeypatch) -> None:
    import eidolon.livekit.plugins.stt.bailian.stt as stt_module

    class FakeConnection:
        instances: list["FakeConnection"] = []

        def __init__(self, **_: object) -> None:
            self._connected = False
            self.closed = False
            self.recv_started = asyncio.Event()
            self.recv_cancelled = False
            FakeConnection.instances.append(self)

        async def connect(self, message_callback=None) -> None:  # noqa: ANN001
            self._connected = True

        async def receive_loop(self, message_cb=None) -> None:  # noqa: ANN001
            self.recv_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.recv_cancelled = True
                raise

        async def send_audio(self, _: bytes) -> None:
            await asyncio.wait_for(self.recv_started.wait(), timeout=1.0)
            raise BailianConnectionError("send failed", recoverable=True)

        async def finish(self) -> None:
            raise AssertionError("finish should not be called after send failure")

        async def close(self) -> None:
            self.closed = True
            self._connected = False

    monkeypatch.setattr(stt_module, "BailianConnectionManager", FakeConnection)
    plugin = BailianFunASRSTT(api_url="ws://example.invalid", api_key="test-key")
    frame = rtc.AudioFrame(
        data=bytes(320),
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=160,
    )

    with pytest.raises(BailianConnectionError):
        await plugin._recognize_impl(  # noqa: SLF001
            [frame],
            language="zh",
            conn_options=plugin.conn_options,
        )

    [conn] = FakeConnection.instances
    assert conn.recv_cancelled is True
    assert conn.closed is True
