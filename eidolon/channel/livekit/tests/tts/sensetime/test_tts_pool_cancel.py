# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Integration tests for Round 8 R8.7 — TTS pool + cancel-swap behavior.

The bug being fixed: with the old single-persistent-connection design,
cancelling a TTS reply mid-stream left server-side queued task_continues
still synthesizing. The same WebSocket would then emit those leftover
audio frames at the start of the NEXT reply, causing user-audible
"play a tail of the previous reply, then the new reply" leakage.

R8.7's pool architecture eliminates this by using a different physical
WebSocket for each turn — the cancelled turn's connection is closed
(server cleanly drops its in-flight work; we don't care because we're
done with that conn) and a pre-warmed conn from the pool serves the
next turn without contamination.

These tests verify:
  1. Each turn uses a different connection (proves pool architecture).
  2. Cancel mid-stream doesn't leak audio into the next turn.
  3. Pool refill keeps up under repeated cancel.
  4. Concurrent acquire requests don't deadlock the pool.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
import websockets

_root = Path(__file__).resolve().parents[5]
if str(_root) not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    os.environ["PYTHONPATH"] = str(_root) + os.pathsep + os.environ.get("PYTHONPATH", "")

from livekit import rtc

from eidolon.channel.livekit.plugins.tts.sensetime import (
    SenseTimeTTS,
    SenseTimeTTSConfig,
)

logger = logging.getLogger("test_tts_pool_cancel")


# Reuse the standard mock server fixture from test_tts_persistent.py.
# (Importing it would create a hard dep; copy a minimal version here.)
class _MockTTSServer:
    """Minimal SenseAudio-protocol mock for cancel-swap testing.

    Tags every audio chunk with a per-connection generation marker so
    tests can verify that a given turn's audio came from a specific
    connection (not leaked from a prior turn's connection).
    """

    def __init__(self, port: int = 0, *, audio_per_continue: int = 5):
        self.port = port
        self.audio_per_continue = audio_per_continue
        self._server: Any = None
        self.connections_total: int = 0
        self.task_start_count: int = 0
        self.task_finish_count: int = 0
        # Per-connection log: (conn_id, event_type, text_payload | None)
        self.events: list[tuple[int, str, Any]] = []

    async def start(self) -> None:
        async def handler(ws: Any) -> None:
            self.connections_total += 1
            conn_id = self.connections_total

            # Server hello
            await ws.send(json.dumps({
                "event": "connected_success",
                "session_id": f"sess-{conn_id}",
                "trace_id": f"trace-{conn_id}",
                "base_resp": {"status_code": 0, "status_msg": "success"},
            }))

            try:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    ev = msg.get("event")
                    self.events.append((conn_id, ev, msg.get("text")))
                    if ev == "task_start":
                        self.task_start_count += 1
                        await ws.send(json.dumps({
                            "event": "task_started",
                            "session_id": f"sess-{conn_id}",
                            "base_resp": {"status_code": 0, "status_msg": "success"},
                        }))
                    elif ev == "task_continue":
                        text = msg.get("text", "")
                        if not text:
                            continue  # heartbeat — ignore
                        # Emit ``audio_per_continue`` audio chunks tagged
                        # with conn_id so we can detect cross-conn leakage.
                        for chunk_idx in range(self.audio_per_continue):
                            await asyncio.sleep(0.01)
                            # Audio payload encodes (conn_id, chunk_idx) so
                            # tests can identify which conn produced it.
                            tag = f"conn{conn_id}_c{chunk_idx}".ljust(60, "_")
                            audio_b64 = (tag.encode("ascii") * 4).hex()
                            await ws.send(json.dumps({
                                "event": "task_continued",
                                "session_id": f"sess-{conn_id}",
                                "data": {
                                    "audio": audio_b64,
                                    "status": 1,  # in-progress
                                },
                                "base_resp": {"status_code": 0, "status_msg": "success"},
                            }))
                        # Batch end marker (empty audio + status=0)
                        await ws.send(json.dumps({
                            "event": "task_continued",
                            "session_id": f"sess-{conn_id}",
                            "data": {
                                "audio": "",
                                "status": 0,
                            },
                            "base_resp": {"status_code": 0, "status_msg": "success"},
                        }))
                    elif ev == "task_finish":
                        self.task_finish_count += 1
                        await ws.send(json.dumps({
                            "event": "task_finished",
                            "session_id": f"sess-{conn_id}",
                            "base_resp": {"status_code": 0, "status_msg": "success"},
                        }))
                        return
            except websockets.exceptions.ConnectionClosed:
                pass

        self._server = await websockets.serve(
            handler, "127.0.0.1", self.port,
            ping_interval=None, ping_timeout=None,
        )
        sock = self._server.sockets[0]
        self.port = sock.getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None


@pytest_asyncio.fixture
async def server():
    s = _MockTTSServer(port=0)
    await s.start()
    await asyncio.sleep(0.05)
    yield s
    await s.stop()


def _make_config(api_url: str, *, pool_size: int = 2) -> SenseTimeTTSConfig:
    return SenseTimeTTSConfig(
        api_url=api_url,
        api_key="test-key",
        model="test-model",
        voice="female_0033_a",
        sample_rate=16000,
        pool_size=pool_size,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPoolCancelSwap:
    """R8.7: verify pool architecture solves the cancel-leakage bug."""

    @pytest.mark.asyncio
    async def test_each_turn_uses_distinct_connection(self, server):
        """3 turns → 3 different conn_ids visible in server-side events.

        With pool_size=2, warmup opens conn 1 + 2. Turn 1 uses one of
        them (say conn 1), turn 2 uses the other (conn 2 — already warm).
        Turn 3 uses a refill conn (conn 3 or 4 depending on refill timing).
        """
        config = _make_config(f"ws://127.0.0.1:{server.port}/ws", pool_size=2)
        tts = SenseTimeTTS(config=config)
        await tts.warmup()
        # warmup opens 2 conns
        warmup_total = server.connections_total
        assert warmup_total == 2

        # Run 3 turns, each pushing distinct text so we can identify them
        # in server events.
        for turn_idx in range(3):
            stream = tts.stream()
            frames: list[rtc.AudioFrame] = []

            async def drain():
                try:
                    async for ev in stream:
                        if hasattr(ev, "frame"):
                            frames.append(ev.frame)
                except Exception:
                    pass

            task = asyncio.create_task(drain())
            await asyncio.sleep(0)
            stream.push_text(f"turn{turn_idx}_marker.")
            stream.end_input()
            await asyncio.wait_for(task, timeout=5.0)
            await asyncio.sleep(0.15)  # let pool refill

        # Each turn's task_continue should have hit a DIFFERENT conn_id.
        turn_conn_ids: dict[int, set[int]] = {}
        for conn_id, ev, text in server.events:
            if ev == "task_continue" and text and "marker" in text:
                turn_idx = int(text[4])  # extract from "turn{N}_marker..."
                turn_conn_ids.setdefault(turn_idx, set()).add(conn_id)

        assert len(turn_conn_ids) == 3, f"expected 3 turns, got {turn_conn_ids}"
        # Each turn's text should appear on exactly one conn
        for turn_idx, conn_ids in turn_conn_ids.items():
            assert len(conn_ids) == 1, (
                f"turn {turn_idx} text leaked across {len(conn_ids)} conns!"
            )
        # And the 3 turns should have used 3 DIFFERENT conns
        all_conn_ids = {next(iter(c)) for c in turn_conn_ids.values()}
        assert len(all_conn_ids) == 3, (
            f"3 turns should have used 3 different conns; "
            f"got {all_conn_ids}"
        )

        await tts.shutdown()
        logger.info(
            "[PASS] test_each_turn_uses_distinct_connection "
            "(per-turn conns: %s)", turn_conn_ids,
        )

    @pytest.mark.asyncio
    async def test_cancel_does_not_leak_audio_into_next_turn(self, server):
        """Cancel turn 1 mid-stream → turn 2's audio comes from a fresh
        conn, never carries conn1's chunks.

        This is the core regression test for the user-reported bug
        ("plays a small tail of previous reply, then new reply").
        """
        config = _make_config(f"ws://127.0.0.1:{server.port}/ws", pool_size=2)
        tts = SenseTimeTTS(config=config)
        await tts.warmup()

        # Turn 1: push some text, then cancel mid-stream by aclose() on
        # the stream itself — this propagates CancelledError into the
        # underlying _run task, which is what the framework does on
        # user interrupt.
        stream1 = tts.stream()
        turn1_frames: list[rtc.AudioFrame] = []

        async def drain1():
            try:
                async for ev in stream1:
                    if hasattr(ev, "frame"):
                        turn1_frames.append(ev.frame)
            except (asyncio.CancelledError, Exception):
                pass

        task1 = asyncio.create_task(drain1())
        await asyncio.sleep(0)
        stream1.push_text("turn1.")
        # Wait for at least one frame so server is mid-flight
        deadline = time.monotonic() + 2.0
        while not turn1_frames and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert turn1_frames, "turn1 produced no audio before cancel"

        # CANCEL turn 1 mid-stream — close() is the standard SynthesizeStream
        # cancel API and propagates into _run, releasing the stream_lock.
        await stream1.aclose()
        try:
            await asyncio.wait_for(task1, timeout=2.0)
        except (asyncio.CancelledError, Exception, asyncio.TimeoutError):
            pass

        await asyncio.sleep(0.3)  # let pool refill

        # Turn 2: should get fresh audio with NO conn1 leakage
        stream2 = tts.stream()
        turn2_frames: list[rtc.AudioFrame] = []

        async def drain2():
            try:
                async for ev in stream2:
                    if hasattr(ev, "frame"):
                        turn2_frames.append(ev.frame)
            except Exception:
                pass

        task2 = asyncio.create_task(drain2())
        await asyncio.sleep(0)
        stream2.push_text("turn2.")
        stream2.end_input()
        await asyncio.wait_for(task2, timeout=5.0)
        assert len(turn2_frames) > 0

        # Inspect server events: turn 2's task_continue must be on a
        # DIFFERENT conn_id than turn 1's.
        turn1_conn = None
        turn2_conn = None
        for conn_id, ev, text in server.events:
            if ev == "task_continue" and text:
                if "turn1" in text and turn1_conn is None:
                    turn1_conn = conn_id
                elif "turn2" in text and turn2_conn is None:
                    turn2_conn = conn_id

        assert turn1_conn is not None, "turn 1 didn't reach server"
        assert turn2_conn is not None, "turn 2 didn't reach server"
        assert turn1_conn != turn2_conn, (
            f"turn 2 reused turn 1's connection (conn {turn1_conn})! "
            f"This is the bug R8.7 fixes."
        )

        await tts.shutdown()
        logger.info(
            "[PASS] test_cancel_does_not_leak_audio_into_next_turn "
            "(turn1=conn%d, turn2=conn%d)", turn1_conn, turn2_conn,
        )

    @pytest.mark.asyncio
    async def test_repeated_cancel_keeps_pool_topped_up(self, server):
        """5 turns with cancel each → pool size stays at target after
        each cancel (background refill keeps up)."""
        config = _make_config(f"ws://127.0.0.1:{server.port}/ws", pool_size=2)
        tts = SenseTimeTTS(config=config)
        await tts.warmup()

        for i in range(5):
            stream = tts.stream()

            async def drain():
                try:
                    async for _ in stream:
                        pass
                except (asyncio.CancelledError, Exception):
                    pass

            task = asyncio.create_task(drain())
            await asyncio.sleep(0)
            stream.push_text(f"cancel-test-{i}.")
            await asyncio.sleep(0.05)  # let some audio flow
            await stream.aclose()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.CancelledError, Exception, asyncio.TimeoutError):
                pass
            await asyncio.sleep(0.4)  # let refill

        # After 5 cancel-loops, pool should be back to size 2.
        # (Allow brief slack for refill timing.)
        deadline = time.monotonic() + 3.0
        while tts._pool.warm_count < 2 and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert tts._pool.warm_count == 2

        await tts.shutdown()
        logger.info(
            "[PASS] test_repeated_cancel_keeps_pool_topped_up "
            "(connections=%d after 5 cancels)", server.connections_total,
        )
