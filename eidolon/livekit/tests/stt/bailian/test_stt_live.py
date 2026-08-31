"""Live integration tests for the Bailian FunASR STT plugin.

These tests connect to the real Bailian / DashScope FunASR WebSocket API
using a checked-in 16 kHz speech sample from the benchmark corpus.

Run with::

    BAILIAN_STT_API_KEY=<your-key> python -m pytest \\
        eidolon/livekit/tests/stt/bailian/test_stt_live.py -v -s

``DASHSCOPE_API_KEY`` remains supported as the shared credential fallback.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import av
import pytest

from livekit import rtc
from livekit.agents import stt as lk_stt

from eidolon.livekit.plugins.stt.bailian import (
    BailianConnectionManager,
    BailianFunASRSTT,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_bailian_live")

_API_KEY = os.environ.get("BAILIAN_STT_API_KEY") or os.environ.get(
    "DASHSCOPE_API_KEY", ""
)
_API_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"

# Skip the entire module if no API key is configured. Avoids leaking a
# hardcoded credential into the repo while keeping the live tests opt-in.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _API_KEY,
        reason="BAILIAN_STT_API_KEY/DASHSCOPE_API_KEY not set; skipping Bailian live tests",
    ),
]

# test_stt_live.py -> parents[5] = repository root.
_AUDIO_FILE = (
    Path(__file__).resolve().parents[5]
    / "benchmark"
    / "audio"
    / "generated"
    / "normal_followup.wav"
)


def load_audio_chunks(
    path: Path,
    chunk_samples: int = 1600,
    target_sr: int = 16000,
) -> list[rtc.AudioFrame]:
    """Decode an audio file and return audio as LiveKit AudioFrame chunks.

    Converts to 16 kHz mono 16-bit PCM and splits into ``chunk_samples``-size
    pieces (default 1600 samples = 100 ms at 16 kHz). Partial chunks are
    included.
    """
    ctx = av.open(str(path))
    stream = ctx.streams.audio[0]
    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono",
        rate=target_sr,
    )
    frames: list[rtc.AudioFrame] = []

    for packet in ctx.demux(stream):
        for frame in packet.decode():
            resampled = resampler.resample(frame)
            if resampled is not None:
                for rf in resampled:
                    samples = rf.to_ndarray()
                    total = samples.shape[-1]
                    for offset in range(0, total, chunk_samples):
                        chunk = samples[..., offset : offset + chunk_samples]
                        if chunk.size == 0:
                            continue
                        frames.append(
                            rtc.AudioFrame(
                                data=bytearray(chunk.tobytes()),
                                sample_rate=target_sr,
                                num_channels=1,
                                samples_per_channel=chunk.shape[-1],
                            )
                        )

    flushed = resampler.resample(None)
    if flushed is not None:
        for rf in flushed:
            samples = rf.to_ndarray()
            for offset in range(0, samples.shape[-1], chunk_samples):
                chunk = samples[..., offset : offset + chunk_samples]
                if chunk.size == 0:
                    continue
                frames.append(
                    rtc.AudioFrame(
                        data=bytearray(chunk.tobytes()),
                        sample_rate=target_sr,
                        num_channels=1,
                        samples_per_channel=chunk.shape[-1],
                    )
                )

    ctx.close()
    return frames


@pytest.fixture(scope="module")
def audio_chunks() -> list[rtc.AudioFrame]:
    """Load the m4a file and return pre-split AudioFrame chunks at 16 kHz."""
    if not _AUDIO_FILE.exists():
        pytest.skip(f"Audio file not found: {_AUDIO_FILE}")
    chunks = load_audio_chunks(_AUDIO_FILE, chunk_samples=1600, target_sr=16000)
    if not chunks:
        pytest.skip("Audio file produced no frames")
    total_dur = sum(c.samples_per_channel for c in chunks) / 16000
    logger.info("Loaded %d audio chunks (%.1f s) from %s", len(chunks), total_dur, _AUDIO_FILE)
    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _wait_for(
    condition: callable,
    timeout: float = 30.0,
    poll: float = 0.1,
) -> None:
    """Poll until ``condition()`` returns truthy, or raise TimeoutError."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if condition():
            return
        await asyncio.sleep(poll)
    raise asyncio.TimeoutError(f"Condition not met within {timeout}s")


async def _drain_stream(stream, events: list[lk_stt.SpeechEvent], errors: list[Exception]) -> None:
    """Consume all events from a stream, collecting errors."""
    try:
        async for ev in stream:
            events.append(ev)
    except Exception as e:
        errors.append(e)


# ---------------------------------------------------------------------------
# STT Class Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stt_properties():
    """BailianFunASRSTT exposes correct provider/model/label."""
    stt = BailianFunASRSTT(
        api_url=_API_URL,
        api_key=_API_KEY,
        model="fun-asr-realtime-2026-02-28",
        language="zh",
    )
    assert stt.provider == "bailian"
    assert stt.model == "fun-asr-realtime-2026-02-28"
    assert "Bailian" in stt.label


# ---------------------------------------------------------------------------
# Connection Manager Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_connection_manager_connect():
    """connect() completes the run-task handshake and returns task_id."""
    conn = BailianConnectionManager(
        api_url=_API_URL,
        api_key=_API_KEY,
        model="fun-asr-realtime-2026-02-28",
    )
    received: list[dict[str, Any]] = []
    await conn.connect(message_callback=lambda m: received.append(m))
    assert conn._connected, "Connection should be marked as connected"
    assert conn._task_id != "", "task_id should be set"
    assert len(received) == 1, f"Should have received task-started, got {len(received)}"
    assert received[0]["header"]["event"] == "task-started"
    await conn.close()
    logger.info("test_connection_manager_connect PASSED")


@pytest.mark.asyncio
async def test_connection_manager_send_audio_and_finish(audio_chunks):
    """send_audio() delivers bytes and finish() triggers task-finished."""
    conn = BailianConnectionManager(
        api_url=_API_URL,
        api_key=_API_KEY,
    )
    await conn.connect()

    finish_event = asyncio.Event()

    async def on_msg(msg: dict[str, Any]) -> None:
        if msg.get("header", {}).get("event") == "task-finished":
            finish_event.set()

    recv_task = asyncio.create_task(conn.receive_loop(message_cb=on_msg))

    for chunk in audio_chunks:
        await conn.send_audio(bytes(chunk.data))

    await conn.finish()

    try:
        await asyncio.wait_for(finish_event.wait(), timeout=60.0)
    except asyncio.TimeoutError:
        pytest.fail("task-finished event not received within 60s")
    finally:
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass
        await conn.close()

    logger.info("test_connection_manager_send_audio_and_finish PASSED")


# ---------------------------------------------------------------------------
# Streaming Tests — use BailianFunASRSpeechStream directly
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_basic(audio_chunks):
    """Push real audio frames and verify INTERIM then FINAL_TRANSCRIPT."""
    from eidolon.livekit.plugins.stt.bailian.speech_stream import (
        BailianFunASRSpeechStream,
    )

    stt = BailianFunASRSTT(api_url=_API_URL, api_key=_API_KEY)
    stream = BailianFunASRSpeechStream(
        stt=stt,
        sample_rate=16000,
        language="zh",
    )

    events: list[lk_stt.SpeechEvent] = []
    errors: list[Exception] = []

    task = asyncio.create_task(_drain_stream(stream, events, errors))

    # RecognizeStream's public contract accepts frames while its provider
    # connection starts asynchronously; do not inspect private connection state.
    for frame in audio_chunks:
        stream.push_frame(frame)
        await asyncio.sleep(0.001)

    stream.end_input()

    try:
        await _wait_for(
            lambda: any(e.type == lk_stt.SpeechEventType.FINAL_TRANSCRIPT for e in events),
            timeout=60.0,
        )
    except asyncio.TimeoutError:
        pytest.fail(
            f"Timeout waiting for FINAL_TRANSCRIPT. "
            f"Events: {[e.type for e in events]}. "
            f"Errors: {errors}"
        )
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await stream.aclose()

    event_types = [e.type for e in events]
    assert lk_stt.SpeechEventType.INTERIM_TRANSCRIPT in event_types, \
        f"Should have received INTERIM_TRANSCRIPT. Got: {event_types}"
    assert lk_stt.SpeechEventType.FINAL_TRANSCRIPT in event_types, \
        f"Should have received FINAL_TRANSCRIPT. Got: {event_types}"
    assert not errors, f"Should not have errors: {errors}"

    final_ev = next(e for e in events if e.type == lk_stt.SpeechEventType.FINAL_TRANSCRIPT)
    text = final_ev.alternatives[0].text
    assert text != "", f"Transcript should not be empty: {text!r}"
    logger.info("test_streaming_basic PASSED — transcript: %s", text)


@pytest.mark.asyncio
async def test_streaming_end_of_speech(audio_chunks):
    """After end_input() the stream emits END_OF_SPEECH."""
    from eidolon.livekit.plugins.stt.bailian.speech_stream import (
        BailianFunASRSpeechStream,
    )

    stt = BailianFunASRSTT(api_url=_API_URL, api_key=_API_KEY)
    stream = BailianFunASRSpeechStream(stt=stt, sample_rate=16000, language="zh")
    events: list[lk_stt.SpeechEvent] = []

    task = asyncio.create_task(_drain_stream(stream, events, []))

    # Use the complete utterance: the sample starts with a short silent lead-in,
    # so a small prefix alone is correctly rejected by Bailian as EmptyAudio.
    for frame in audio_chunks:
        stream.push_frame(frame)

    stream.end_input()

    try:
        await _wait_for(
            lambda: any(e.type == lk_stt.SpeechEventType.END_OF_SPEECH for e in events),
            timeout=60.0,
        )
    except asyncio.TimeoutError:
        pytest.fail(f"END_OF_SPEECH not received. Events: {[e.type for e in events]}")
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await stream.aclose()

    logger.info("test_streaming_end_of_speech PASSED")


@pytest.mark.asyncio
async def test_streaming_word_timestamps(audio_chunks):
    """FINAL_TRANSCRIPT should include word-level timing data."""
    from eidolon.livekit.plugins.stt.bailian.speech_stream import (
        BailianFunASRSpeechStream,
    )

    stt = BailianFunASRSTT(api_url=_API_URL, api_key=_API_KEY)
    stream = BailianFunASRSpeechStream(stt=stt, sample_rate=16000, language="zh")
    events: list[lk_stt.SpeechEvent] = []

    task = asyncio.create_task(_drain_stream(stream, events, []))

    for frame in audio_chunks:
        stream.push_frame(frame)

    stream.end_input()

    try:
        await _wait_for(
            lambda: any(e.type == lk_stt.SpeechEventType.FINAL_TRANSCRIPT for e in events),
            timeout=60.0,
        )
    except asyncio.TimeoutError:
        pytest.fail(f"FINAL_TRANSCRIPT not received. Events: {[e.type for e in events]}")
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await stream.aclose()

    final_ev = next(e for e in events if e.type == lk_stt.SpeechEventType.FINAL_TRANSCRIPT)
    words = final_ev.alternatives[0].words
    assert words, "FINAL_TRANSCRIPT should contain word-level data"
    logger.info("test_streaming_word_timestamps PASSED — %d words, text: %s",
                 len(words), final_ev.alternatives[0].text)


# ---------------------------------------------------------------------------
# Batch Recognition Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_batch_recognize(audio_chunks):
    """recognize() (batch mode) returns a FINAL_TRANSCRIPT with text."""
    stt = BailianFunASRSTT(api_url=_API_URL, api_key=_API_KEY)

    # Concatenate all chunks into one frame (trim to even sample count)
    total_samples = sum(c.samples_per_channel for c in audio_chunks)
    total_samples = total_samples - (total_samples % 2)
    total_bytes = total_samples * 2
    combined = bytearray(total_bytes)
    offset = 0
    for c in audio_chunks:
        chunk_bytes = min(c.samples_per_channel * 2, total_bytes - offset)
        if chunk_bytes <= 0:
            break
        combined[offset : offset + chunk_bytes] = c.data[:chunk_bytes]
        offset += chunk_bytes

    frame = rtc.AudioFrame(
        data=combined,
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=total_samples,
    )

    event = await stt.recognize(frame)
    assert event.type == lk_stt.SpeechEventType.FINAL_TRANSCRIPT, \
        f"Expected FINAL_TRANSCRIPT, got {event.type}"
    assert len(event.alternatives) > 0
    text = event.alternatives[0].text
    assert text != "", f"Transcript text should not be empty, got: {text!r}"
    logger.info("test_batch_recognize PASSED — transcript: %s", text)


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_connection_failure_bad_api_key():
    """Connecting with an invalid key should raise an error."""
    conn = BailianConnectionManager(
        api_url=_API_URL,
        api_key="invalid-key-12345",
        model="fun-asr-realtime-2026-02-28",
    )
    try:
        await conn.connect()
        pytest.fail("Expected an exception for invalid API key")
    except Exception as e:
        logger.info("Expected error for bad key: %s: %s", type(e).__name__, e)
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Module-level smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
