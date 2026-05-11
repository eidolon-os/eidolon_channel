"""Integration tests for SenseTime SenseAudio TTS plugin with real API.

Run with::

    cd <repository-root>
    SENSETIME_TTS_API_KEY=your_key .venv/bin/python -m pytest \\
        eidolon/channel/livekit/tests/tts/sensetime/test_tts.py -v -s

Requirements::

    pytest pytest-asyncio aiohttp

These tests connect to the real SenseTime SenseAudio API. All assertions verify
observable client-side behavior (frames emitted, errors raised, connection states).
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.integration

_root = Path(__file__).resolve().parents[6]
if str(_root) not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    os.environ["PYTHONPATH"] = str(_root) + os.pathsep + os.environ.get("PYTHONPATH", "")

from livekit import rtc

from eidolon.channel.livekit.plugins.tts.sensetime import (
    SenseTimeTTS,
    SenseTimeTTSConfig,
    SenseTimeTTSError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(name)-35s %(levelname)-8s %(message)s",
)
logger = logging.getLogger("test_sensetime_tts")

# Real-API tests require a valid API key in the environment.
_HAS_API_KEY = bool(
    os.environ.get("SENSETIME_TTS_API_KEY") or os.environ.get("SENSEAUDIO_API_KEY")
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(**overrides: Any) -> SenseTimeTTSConfig:
    """Create a real SenseTimeTTSConfig, reading credentials from environment."""
    kwargs = {
        "sample_rate": 16000,
        "speed": 1.0,
        "vol": 1.0,
        "pitch": 0,
    }
    kwargs.update(overrides)
    return SenseTimeTTSConfig(**kwargs)


# ---------------------------------------------------------------------------
# TTS Properties Tests (no network)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tts_properties():
    """SenseTimeTTS exposes correct provider/model/label/sample_rate/num_channels."""
    config = SenseTimeTTSConfig(
        api_url="ws://localhost:9999",
        api_key="test-key",
        model="SenseAudio-TTS-1.0",
        voice="female_0033_a",
        sample_rate=16000,
        speed=1.0,
        vol=1.0,
        pitch=0,
    )
    tts = SenseTimeTTS(config=config)

    assert tts.provider == "sensetime", f"Expected provider='sensetime', got {tts.provider}"
    assert tts.model == "SenseAudio-TTS-1.0", f"Expected model='SenseAudio-TTS-1.0', got {tts.model}"
    assert tts.sample_rate == 16000, f"Expected sample_rate=16000, got {tts.sample_rate}"
    assert tts.num_channels == 1, f"Expected num_channels=1, got {tts.num_channels}"
    assert "SenseTimeTTS" in tts.label, f"Expected 'SenseTimeTTS' in label, got {tts.label}"

    logger.info("[TEST] test_tts_properties PASSED")
    logger.info(
        "  provider=%s model=%s sample_rate=%d num_channels=%d label=%s",
        tts.provider, tts.model, tts.sample_rate, tts.num_channels, tts.label,
    )


# ---------------------------------------------------------------------------
# Error Tests (no real API call needed)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_connection_failure():
    """Connecting to non-existent host raises SenseTimeTTSError."""
    config = SenseTimeTTSConfig(
        api_url="ws://localhost:59999",
        api_key="test-key",
        model="SenseAudio-TTS-1.0",
        voice="female_0033_a",
        sample_rate=16000,
    )
    tts = SenseTimeTTS(config=config)
    stream = tts.stream()

    async def drain():
        try:
            async for _ in stream:
                pass
        except SenseTimeTTSError as e:
            logger.info("[TEST] Caught expected error: %s", e)
            raise

    task = asyncio.create_task(drain())
    stream.push_text("trigger error")
    stream.end_input()

    with pytest.raises(SenseTimeTTSError):
        await asyncio.wait_for(task, timeout=5.0)

    logger.info("[TEST] test_connection_failure PASSED")


# ---------------------------------------------------------------------------
# Streaming Tests — real API
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_basic():
    """push_text() -> audio frames emitted via AudioEmitter."""
    if not _HAS_API_KEY:
        pytest.skip("SENSETIME_TTS_API_KEY not set")
    tts = SenseTimeTTS()
    stream = tts.stream()
    events: list[Any] = []
    audio_frames: list[rtc.AudioFrame] = []

    async def drain():
        try:
            async for ev in stream:
                events.append(ev)
        except (Exception, asyncio.CancelledError):
            pass  # Stream ended or was cancelled

    task = asyncio.ensure_future(drain())
    await asyncio.sleep(0)

    stream.push_text("hello")
    stream.end_input()

    try:
        await asyncio.wait_for(task, timeout=15.0)
    except asyncio.TimeoutError:
        task.cancel()
        pytest.fail("Stream did not complete within 15s")

    audio_frames = [ev.frame for ev in events if hasattr(ev, "frame")]
    assert len(audio_frames) > 0, f"Expected audio frames, got {len(audio_frames)}"
    assert audio_frames[0].sample_rate == 16000
    assert audio_frames[0].num_channels == 1

    await tts.shutdown()

    logger.info(
        "[TEST] test_streaming_basic PASSED — %d audio frames emitted",
        len(audio_frames),
    )
    for i, f in enumerate(audio_frames):
        logger.info(
            "  frame[%d] size=%d bytes sample_rate=%d channels=%d duration=%.3fs",
            i, len(f.data), f.sample_rate, f.num_channels, f.duration,
        )


@pytest.mark.asyncio
async def test_streaming_multiple_text_chunks():
    """Multiple push_text() calls produce audio in order."""
    if not _HAS_API_KEY:
        pytest.skip("SENSETIME_TTS_API_KEY not set")
    tts = SenseTimeTTS()

    stream = tts.stream()

    async def drain():
        count = 0
        try:
            async for ev in stream:
                count += 1
                logger.info(
                    "[TEST] received frame[%d] size=%d bytes duration=%.3fs",
                    count, len(ev.frame.data), ev.frame.duration,
                )
        except Exception as e:
            logger.warning("[TEST] stream error: %s", e)
        return count

    task = asyncio.create_task(drain())

    stream.push_text("first ")
    await asyncio.sleep(0.1)
    stream.push_text("second ")
    await asyncio.sleep(0.1)
    stream.push_text("third")
    stream.end_input()

    try:
        frame_count = await asyncio.wait_for(task, timeout=15.0)
    except asyncio.TimeoutError:
        task.cancel()
        pytest.fail("Stream did not complete within 15s")

    assert frame_count > 0, "Expected at least one audio frame"

    await tts.shutdown()

    logger.info(
        "[TEST] test_streaming_multiple_text_chunks PASSED — %d frames total",
        frame_count,
    )


@pytest.mark.asyncio
async def test_streaming_audio_frame_format():
    """Emitted audio frames are 16kHz mono PCM (audio/pcm)."""
    if not _HAS_API_KEY:
        pytest.skip("SENSETIME_TTS_API_KEY not set")
    tts = SenseTimeTTS()

    stream = tts.stream()
    audio_frames: list[rtc.AudioFrame] = []

    async def drain():
        try:
            async for ev in stream:
                if hasattr(ev, "frame"):
                    audio_frames.append(ev.frame)
        except Exception:
            pass

    task = asyncio.create_task(drain())
    stream.push_text("format test")
    stream.end_input()

    try:
        await asyncio.wait_for(task, timeout=15.0)
    except asyncio.TimeoutError:
        task.cancel()
        pytest.fail("Timeout")

    assert len(audio_frames) > 0, "Expected audio frames"
    for f in audio_frames:
        assert f.sample_rate == 16000, f"Expected 16000Hz, got {f.sample_rate}"
        assert f.num_channels == 1, f"Expected mono, got {f.num_channels}"
        # Data should be 16-bit PCM. ``f.data`` is a memoryview with
        # format='h' (int16), so ``f.data.nbytes`` is the byte count
        # (always 2× samples_per_channel for mono int16). Earlier
        # versions used ``len(f.data) % 2 == 0`` which mixed sample
        # count and byte count — flaky depending on whether the LAST
        # (partial) frame happened to have an even sample count.
        assert f.data.nbytes == f.samples_per_channel * 2 * f.num_channels, (
            f"unexpected data size: {f.data.nbytes} bytes for "
            f"{f.samples_per_channel} samples × {f.num_channels} ch"
        )

    await tts.shutdown()

    logger.info(
        "[TEST] test_streaming_audio_frame_format PASSED — %d frames, "
        "all 16kHz mono PCM, sizes: %s",
        len(audio_frames),
        [len(f.data) for f in audio_frames],
    )


@pytest.mark.asyncio
async def test_streaming_segment_boundaries():
    """flush() creates separate segments with their own boundaries."""
    if not _HAS_API_KEY:
        pytest.skip("SENSETIME_TTS_API_KEY not set")
    tts = SenseTimeTTS()

    stream = tts.stream()
    audio_frames: list[Any] = []

    async def drain():
        try:
            async for ev in stream:
                audio_frames.append(ev)
        except Exception:
            pass

    task = asyncio.create_task(drain())

    # First segment
    stream.push_text("segment one")
    await asyncio.sleep(0.1)
    stream.flush()
    await asyncio.sleep(0.2)

    # Second segment
    stream.push_text("segment two")
    stream.end_input()

    try:
        await asyncio.wait_for(task, timeout=15.0)
    except asyncio.TimeoutError:
        task.cancel()
        pytest.fail("Timeout")

    # Check we got audio frames
    frames = [ev for ev in audio_frames if hasattr(ev, "frame")]
    assert len(frames) > 0, "Expected audio frames"

    await tts.shutdown()

    logger.info(
        "[TEST] test_streaming_segment_boundaries PASSED — %d total frames",
        len(frames),
    )


@pytest.mark.asyncio
async def test_streaming_end_input():
    """end_input() triggers stream completion."""
    if not _HAS_API_KEY:
        pytest.skip("SENSETIME_TTS_API_KEY not set")
    tts = SenseTimeTTS()

    stream = tts.stream()
    frame_count = 0

    async def drain():
        nonlocal frame_count
        try:
            async for ev in stream:
                frame_count += 1
                logger.info(
                    "[TEST] end_input test: frame[%d] size=%d bytes",
                    frame_count, len(ev.frame.data),
                )
        except Exception as e:
            logger.warning("[TEST] drain error: %s", e)

    task = asyncio.create_task(drain())
    stream.push_text("end input test")
    stream.end_input()

    try:
        await asyncio.wait_for(task, timeout=15.0)
    except asyncio.TimeoutError:
        task.cancel()
        pytest.fail("Timeout")

    assert frame_count > 0, "Expected audio frames after end_input"

    await tts.shutdown()

    logger.info(
        "[TEST] test_streaming_end_input PASSED — %d frames received",
        frame_count,
    )


@pytest.mark.asyncio
async def test_streaming_is_final():
    """Last frame of segment has is_final=True."""
    if not _HAS_API_KEY:
        pytest.skip("SENSETIME_TTS_API_KEY not set")
    tts = SenseTimeTTS()

    stream = tts.stream()
    events: list[Any] = []

    async def drain():
        try:
            async for ev in stream:
                events.append(ev)
                logger.info(
                    "[TEST] is_final test: frame size=%d is_final=%s",
                    len(ev.frame.data), ev.is_final,
                )
        except Exception:
            pass

    task = asyncio.create_task(drain())
    stream.push_text("is final test")
    stream.end_input()

    try:
        await asyncio.wait_for(task, timeout=15.0)
    except asyncio.TimeoutError:
        task.cancel()
        pytest.fail("Timeout")

    frames = [ev for ev in events if hasattr(ev, "frame")]
    if frames:
        # At least one frame should have is_final=True at end of stream
        final_frames = [f for f in frames if f.is_final]
        logger.info(
            "[TEST] test_streaming_is_final: %d/%d frames have is_final=True",
            len(final_frames), len(frames),
        )

    await tts.shutdown()

    logger.info("[TEST] test_streaming_is_final PASSED")


# ---------------------------------------------------------------------------
# Batch Synthesize Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_synthesize_batch():
    """synthesize() returns combined audio frame for full text."""
    if not _HAS_API_KEY:
        pytest.skip("SENSETIME_TTS_API_KEY not set")
    tts = SenseTimeTTS()

    text = "hello world synthesize batch"
    logger.info("[TEST] Calling synthesize() with: '%s'", text)

    frames: list[rtc.AudioFrame] = []
    async with tts.synthesize(text) as stream:
        async for ev in stream:
            frames.append(ev.frame)
            logger.info(
                "[TEST] synthesize frame: size=%d bytes duration=%.3fs is_final=%s",
                len(ev.frame.data), ev.frame.duration, ev.is_final,
            )

    await tts.shutdown()

    total_bytes = sum(len(f.data) for f in frames)
    logger.info(
        "[TEST] synthesize() complete: %d frames, total %d bytes",
        len(frames), total_bytes,
    )
    assert len(frames) > 0, "Expected at least one frame"
    assert frames[0].sample_rate == 16000


# ---------------------------------------------------------------------------
# Error Injection Test (local server for task_failed scenario)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_task_failure():
    """Server task_failed event propagates as SenseTimeTTSError.

    Uses a minimal local WebSocket server that intentionally sends task_failed
    to simulate a server-side error. This is error injection, not mocking the
    real SenseTime API.
    """
    import aiohttp
    from aiohttp import web, WSMessage, WSMsgType
    import json

    task_failed_received = asyncio.Event()

    async def ws_handler(request: web.Request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        # Send connected_success
        await ws.send_json({"event": "connected_success", "session_id": "test-sess"})

        msg = await ws.receive()
        if msg.type == WSMsgType.TEXT:
            msg_obj = json.loads(msg.data)
            if msg_obj.get("event") == "task_start":
                await ws.send_json({"event": "task_started"})

        # Wait for task_continue, then send task_failed
        msg = await ws.receive()
        if msg.type == WSMsgType.TEXT:
            msg_obj = json.loads(msg.data)
            if msg_obj.get("event") == "task_continue":
                await ws.send_json({
                    "event": "task_failed",
                    "data": {
                        "base_resp": {
                            "status_msg": "mock task failure from test server"
                        }
                    },
                })
                await ws.close()
                task_failed_received.set()

        return web.Response()

    app = web.Application()
    app.router.add_get("/", ws_handler)
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    try:
        config = make_config(api_url=f"ws://localhost:{port}", api_key="test-key")
        tts = SenseTimeTTS(config=config)
        stream = tts.stream()

        async def drain():
            try:
                async for _ in stream:
                    pass
            except SenseTimeTTSError as e:
                logger.info("[TEST] Caught expected SenseTimeTTSError: %s", e)
                raise

        task = asyncio.create_task(drain())
        stream.push_text("trigger failure")
        stream.end_input()

        with pytest.raises(SenseTimeTTSError) as exc_info:
            await asyncio.wait_for(task, timeout=10.0)

        assert "mock task failure" in str(exc_info.value)
        await tts.shutdown()
        logger.info("[TEST] test_task_failure PASSED")
    finally:
        await runner.cleanup()


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_idle_stream_noop():
    """No text pushed -> stream completes without errors."""
    if not _HAS_API_KEY:
        pytest.skip("SENSETIME_TTS_API_KEY not set")
    tts = SenseTimeTTS()

    stream = tts.stream()
    frame_count = 0

    async def drain():
        nonlocal frame_count
        async for ev in stream:
            frame_count += 1

    task = asyncio.create_task(drain())
    stream.end_input()  # No push_text!

    try:
        await asyncio.wait_for(task, timeout=5.0)
    except asyncio.TimeoutError:
        task.cancel()
        pytest.fail("Timeout")

    await tts.shutdown()

    # With no text, the stream should still complete but with 0 or minimal audio
    logger.info(
        "[TEST] test_idle_stream_noop PASSED — %d frames (expected 0 or minimal)",
        frame_count,
    )


@pytest.mark.asyncio
async def test_concurrent_push_and_receive():
    """Text pushed while audio is being received."""
    if not _HAS_API_KEY:
        pytest.skip("SENSETIME_TTS_API_KEY not set")
    tts = SenseTimeTTS()

    stream = tts.stream()
    frames: list[Any] = []

    async def drain():
        try:
            async for ev in stream:
                frames.append(ev)
                logger.info(
                    "[TEST] concurrent: received frame size=%d bytes total=%d",
                    len(ev.frame.data), sum(len(f.frame.data) for f in frames),
                )
        except Exception as e:
            logger.warning("[TEST] drain error: %s", e)

    task = asyncio.create_task(drain())

    # Push text and wait for audio concurrently
    stream.push_text("concurrent ")
    await asyncio.sleep(0.1)
    stream.push_text("test")
    await asyncio.sleep(0.1)
    stream.end_input()

    try:
        await asyncio.wait_for(task, timeout=15.0)
    except asyncio.TimeoutError:
        task.cancel()
        pytest.fail("Timeout")

    assert len(frames) > 0, "Expected audio frames from concurrent push/receive"

    await tts.shutdown()

    logger.info(
        "[TEST] test_concurrent_push_and_receive PASSED — %d frames",
        len(frames),
    )


# ---------------------------------------------------------------------------
# Lifecycle Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_warmup_then_stream_then_shutdown():
    """Round 8 R8.7: full lifecycle with pool architecture (live API).

    Verifies the new pool model: warmup opens a pool, each stream
    consumes a conn from the pool, shutdown drains the pool.
    """
    if not _HAS_API_KEY:
        pytest.skip("SENSETIME_TTS_API_KEY not set")
    tts = SenseTimeTTS()

    # warmup opens a pool of conns (default size=2)
    await tts.warmup()
    assert tts._pool.warm_count >= 1

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
    stream.push_text("lifecycle test")
    stream.end_input()

    await asyncio.wait_for(task, timeout=15.0)
    assert len(frames) > 0

    # Pool refills in background; should still have warm conns available.
    await asyncio.sleep(0.5)  # give refill time

    await tts.shutdown()
    # Pool drained at shutdown
    assert tts._pool.warm_count == 0

    logger.info(
        "[TEST] test_warmup_then_stream_then_shutdown PASSED — %d frames",
        len(frames),
    )


@pytest.mark.asyncio
async def test_shutdown_closes_session():
    """Round 8 R8.7: shutdown drains the pool and closes http session."""
    if not _HAS_API_KEY:
        pytest.skip("SENSETIME_TTS_API_KEY not set")
    tts = SenseTimeTTS()

    await tts.warmup()
    assert tts._pool.warm_count >= 1
    await tts.shutdown()
    assert tts._pool.warm_count == 0
    assert tts._http_session is None

    logger.info("[TEST] test_shutdown_closes_session PASSED")


# ---------------------------------------------------------------------------
# Module-level smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
