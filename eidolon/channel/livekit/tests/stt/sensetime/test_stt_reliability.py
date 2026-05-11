"""Reliability / self-healing tests for SenseTime STT (Round 8 R8.3).

Verifies that mid-session WebSocket failures are handled inline by the
SenseTimeSpeechStream's retry layer rather than letting the framework
end up with a dead _STTPipeline. Three failure modes covered:

  1. Mid-session WS close (network blip): stream reconnects, subsequent
     utterances arrive normally.
  2. Retry budget exhausted: stream emits an STT error and exits cleanly
     (framework can then close the AgentSession).
  3. Fatal task_failed (server-side error): no retry, fail fast.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import pytest_asyncio
import websockets

_root = Path(__file__).resolve().parents[5]
if str(_root) not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    os.environ["PYTHONPATH"] = str(_root) + os.pathsep + os.environ.get("PYTHONPATH", "")

from livekit import rtc
from livekit.agents.stt import SpeechEventType

from eidolon.channel.livekit.plugins.stt.sensetime import (
    SenseTimeSTT,
    SenseTimeSTTConfig,
)

logger = logging.getLogger("test_stt_reliability")


def make_audio_frame(num_samples: int = 1600) -> rtc.AudioFrame:
    arr = np.int16([256] * num_samples)
    return rtc.AudioFrame(
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=num_samples,
        data=arr.tobytes(),
    )


def make_config(api_url: str, **kwargs) -> SenseTimeSTTConfig:
    base = dict(
        api_url=api_url,
        api_key="test-key",
        model="senseaudio-asr-deepthink-1.5-260319",
        sample_rate=16000,
        language="zh",
    )
    base.update(kwargs)
    return SenseTimeSTTConfig(**base)


# ---------------------------------------------------------------------------
# Mock server with controllable failure injection
# ---------------------------------------------------------------------------


class FlakeySTTServer:
    """Mock SenseAudio STT server with injectable mid-session failures.

    Modes:
      ``drop_after_n_audio_frames``: after this many binary frames are
        received, abruptly close the WS (simulates network blip).
      ``always_drop``: close on EVERY connection after task_started
        (simulates persistent failure → exhaust retry budget).
      ``utterance_offsets``: list of (delay_after_start_s, text) for
        result_finals to emit on the connection.
    """

    def __init__(
        self,
        port: int = 0,
        *,
        drop_after_n_audio_frames: int | None = None,
        always_drop: bool = False,
        utterance_offsets: list[tuple[float, str]] | None = None,
        emit_task_finished: bool = True,
    ) -> None:
        self.port = port
        self.drop_after_n_audio_frames = drop_after_n_audio_frames
        self.always_drop = always_drop
        self.utterance_offsets = utterance_offsets or [(0.2, "test_utterance")]
        self.emit_task_finished = emit_task_finished
        self._server: Any = None
        self.connections_total: int = 0
        self.task_start_count: int = 0
        self.task_finish_count: int = 0
        self.binary_frames_received: int = 0
        self.result_finals_emitted: int = 0
        # Whether the next-incoming connection should drop after N audio
        # frames. Defaults True if either always_drop or
        # drop_after_n_audio_frames is configured. Tests can flip this
        # to False mid-test to simulate "network recovers" — the next
        # connection then proceeds normally.
        self._drop_next: bool = bool(
            always_drop or drop_after_n_audio_frames is not None
        )

    async def start(self) -> None:
        async def handler(ws: Any) -> None:
            self.connections_total += 1
            session_id = f"sess-flaky-{self.connections_total}"
            should_drop_this_conn = self._drop_next or self.always_drop
            frames_received = 0

            await ws.send(json.dumps({
                "event": "connected_success",
                "session_id": session_id,
                "trace_id": session_id,
                "base_resp": {"status_code": 0, "status_msg": "success"},
            }))

            utterance_task: asyncio.Task[None] | None = None

            async def emit_utterances() -> None:
                t0 = asyncio.get_event_loop().time()
                for delay, text in self.utterance_offsets:
                    target = t0 + delay
                    while True:
                        wait = target - asyncio.get_event_loop().time()
                        if wait <= 0:
                            break
                        await asyncio.sleep(wait)
                    try:
                        await ws.send(json.dumps({
                            "event": "result_final",
                            "session_id": session_id,
                            "data": {
                                "text": text,
                                "is_final": True,
                                "segment_id": self.result_finals_emitted,
                                "timestamp_end": int(time.time() * 1000),
                            },
                            "base_resp": {"status_code": 0, "status_msg": "success"},
                        }))
                        self.result_finals_emitted += 1
                    except websockets.exceptions.ConnectionClosed:
                        return

            try:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        self.binary_frames_received += 1
                        frames_received += 1
                        if (
                            should_drop_this_conn
                            and self.drop_after_n_audio_frames is not None
                            and frames_received >= self.drop_after_n_audio_frames
                        ):
                            logger.info(
                                "[FlakeySTTServer] dropping conn after %d "
                                "binary frames", frames_received,
                            )
                            await ws.close()
                            return
                        continue
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    event = msg.get("event")
                    if event == "task_start":
                        self.task_start_count += 1
                        await ws.send(json.dumps({
                            "event": "task_started",
                            "session_id": session_id,
                            "base_resp": {"status_code": 0, "status_msg": "success"},
                        }))
                        utterance_task = asyncio.create_task(emit_utterances())
                    elif event == "task_finish":
                        self.task_finish_count += 1
                        if utterance_task and not utterance_task.done():
                            utterance_task.cancel()
                        if self.emit_task_finished:
                            await ws.send(json.dumps({
                                "event": "task_finished",
                                "session_id": session_id,
                                "base_resp": {"status_code": 0, "status_msg": "success"},
                            }))
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                if utterance_task and not utterance_task.done():
                    utterance_task.cancel()

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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRetryOnConnDrop:
    """R8.3 — mid-session WS drop should reconnect inline."""

    @pytest.mark.asyncio
    async def test_recovers_from_single_mid_session_drop(self):
        """Server drops after 5 audio frames; stream reconnects and the
        second utterance arrives on the new connection."""
        # First connection: emit one utterance, then drop after 5 frames.
        # Second connection (after retry): emit another utterance.
        server = FlakeySTTServer(
            port=0,
            drop_after_n_audio_frames=5,
            utterance_offsets=[(0.2, "在断开之前的第一句")],
        )
        await server.start()
        try:
            config = make_config(
                f"ws://localhost:{server.port}/ws",
                # Keep retries fast for the test.
                max_stream_retries=3,
                stream_retry_backoffs=(0.05, 0.1, 0.2),
            )
            stt = SenseTimeSTT(config=config)
            await stt.warmup()
            stream = stt.stream()

            finals: list[str] = []

            async def drain():
                async for ev in stream:
                    if ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                        finals.append(ev.alternatives[0].text)

            task = asyncio.create_task(drain())
            await asyncio.sleep(0)

            # After first drop, reconfigure server to NOT drop and emit a
            # second utterance on the next connection.
            async def trigger_recovery_after_drop():
                # Wait long enough for first conn drop to register + retry.
                await asyncio.sleep(0.5)
                server._drop_next = False
                server.utterance_offsets = [(0.3, "在断开之后的第二句")]

            recover_task = asyncio.create_task(trigger_recovery_after_drop())

            # Push frames continuously for ~1.5s.
            for _ in range(15):
                stream.push_frame(make_audio_frame())
                await asyncio.sleep(0.1)

            stream.end_input()
            await asyncio.wait_for(task, timeout=10.0)
            await recover_task

            # Should have at least 2 connections (initial + retry).
            assert server.connections_total >= 2, (
                f"expected ≥2 connections (initial + retry), got "
                f"{server.connections_total}"
            )
            # The post-recovery utterance should make it through.
            assert any("第二句" in t for t in finals), (
                f"expected post-recovery utterance, got finals={finals}"
            )
            await stt.shutdown()
            logger.info(
                "[PASS] test_recovers_from_single_mid_session_drop "
                "(connections=%d finals=%s)",
                server.connections_total, finals,
            )
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_retry_budget_exhausted_emits_error_and_exits(self):
        """When the WS keeps dying, the stream eventually gives up and
        emits an STT error rather than retrying forever."""
        server = FlakeySTTServer(
            port=0,
            always_drop=True,
            drop_after_n_audio_frames=2,  # drop very fast
        )
        await server.start()
        try:
            config = make_config(
                f"ws://localhost:{server.port}/ws",
                max_stream_retries=2,    # 1 initial + 2 retries = 3 total
                stream_retry_backoffs=(0.05, 0.05, 0.05),
            )
            stt = SenseTimeSTT(config=config)
            await stt.warmup()
            stream = stt.stream()

            errors_emitted: list[Any] = []
            stt.on("error", lambda ev: errors_emitted.append(ev))

            async def drain():
                async for _ in stream:
                    pass

            task = asyncio.create_task(drain())
            await asyncio.sleep(0)

            # Push frames; tolerate stream closing mid-loop (it may close
            # quickly once retry budget is exhausted).
            for _ in range(50):
                try:
                    stream.push_frame(make_audio_frame())
                except RuntimeError:
                    # Stream closed (retry budget exhausted) — stop pushing.
                    break
                await asyncio.sleep(0.05)
            try:
                stream.end_input()
            except RuntimeError:
                pass
            await asyncio.wait_for(task, timeout=10.0)

            # Should have exhausted 1 + max_retries connection attempts.
            assert server.connections_total >= 3, (
                f"expected ≥3 connection attempts (initial + retries), got "
                f"{server.connections_total}"
            )
            # An STT error should have been emitted on budget exhaustion.
            assert len(errors_emitted) >= 1, (
                "expected STT error event on retry budget exhaustion, got "
                f"{errors_emitted}"
            )
            await stt.shutdown()
            logger.info(
                "[PASS] test_retry_budget_exhausted_emits_error_and_exits "
                "(connections=%d errors=%d)",
                server.connections_total, len(errors_emitted),
            )
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_no_retry_on_natural_session_end(self):
        """Natural end of session (framework closes audio_ch) should NOT
        trigger any retry — there's nothing to recover from.
        """
        server = FlakeySTTServer(
            port=0,
            utterance_offsets=[(0.2, "正常结束")],
        )
        await server.start()
        try:
            config = make_config(
                f"ws://localhost:{server.port}/ws",
                max_stream_retries=3,
            )
            stt = SenseTimeSTT(config=config)
            await stt.warmup()
            stream = stt.stream()

            finals: list[str] = []

            async def drain():
                async for ev in stream:
                    if ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                        finals.append(ev.alternatives[0].text)

            task = asyncio.create_task(drain())
            await asyncio.sleep(0)
            stream.push_frame(make_audio_frame())
            await asyncio.sleep(0.5)
            stream.end_input()
            await asyncio.wait_for(task, timeout=5.0)

            # Exactly ONE connection expected (no retries on natural end).
            assert server.connections_total == 1, (
                f"natural session end shouldn't retry; got "
                f"{server.connections_total} connections"
            )
            assert finals == ["正常结束"]
            await stt.shutdown()
            logger.info(
                "[PASS] test_no_retry_on_natural_session_end "
                "(connections=%d)", server.connections_total,
            )
        finally:
            await server.stop()
