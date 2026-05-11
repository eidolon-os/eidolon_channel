# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the FireRedChat pVAD VAD plugin.

These tests use real model files and the real audio file ``vad.m4a``.
They do NOT require network access or an API key.

Run with::

    python -m pytest eidolon/livekit/plugins/vad/firered/test_vad.py -v

Requirements::

    pytest pytest-asyncio av numpy onnxruntime
    pip install speechbrain  # optional — speaker embedding tests degrade gracefully without it
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import av
import numpy as np
import pytest

# Ensure the package is importable from the repo root
_root = Path(__file__).resolve().parents[5]
if str(_root) not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    os.environ["PYTHONPATH"] = str(_root) + os.pathsep + os.environ.get("PYTHONPATH", "")

from livekit import rtc

# ---------------------------------------------------------------------------
# Audio fixture
# ---------------------------------------------------------------------------

# eidolon/livekit/plugins/vad/firered/test_vad.py → parents[4] = eidolon/
_AUDIO_FILE = _root / "pipeline" / "data" / "vad.m4a"


def load_audio_chunks(
    path: Path,
    chunk_samples: int = 1600,
    target_sr: int = 16000,
) -> list[rtc.AudioFrame]:
    """Decode an m4a file and return audio as LiveKit AudioFrame chunks.

    Converts to 16 kHz mono 16-bit PCM and splits into ``chunk_samples``-size
    pieces (default 1600 samples = 100 ms at 16 kHz).
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

    logging.getLogger("test_firered_vad").info(
        "Loaded %d audio chunks (%.1f s) from %s",
        len(chunks),
        sum(c.samples_per_channel for c in chunks) / 16000,
        _AUDIO_FILE,
    )
    return chunks


# ---------------------------------------------------------------------------
# Processor tests
# ---------------------------------------------------------------------------

class TestPvadProcessor:
    """Tests for the low-level PvadProcessor and SpeakerEmbExtractor."""

    def test_processor_load(self):
        """PvadProcessor loads successfully and initializes state buffers."""
        from eidolon.livekit.plugins.vad.firered import PvadProcessor

        proc = PvadProcessor(force_cpu=True)

        # Check state buffers are initialized to the correct shapes
        assert proc.mel_buffer.shape == (1, 80, 15), f"mel_buffer shape mismatch: {proc.mel_buffer.shape}"
        assert proc.gru_buffer.shape == (2, 1, 256), f"gru_buffer shape mismatch: {proc.gru_buffer.shape}"
        assert proc.spkemb.shape == (1, 192), f"spkemb shape mismatch: {proc.spkemb.shape}"
        assert proc.window_size_samples == 160, f"window_size_samples should be 160, got {proc.window_size_samples}"
        assert proc.sample_rate == 16000

        # ONNX session is loaded lazily — access it to trigger loading
        _ = proc.onnx_session
        assert True  # if we got here, no exception was raised

    def test_processor_inference_shape(self):
        """Single-frame inference returns a probability in [0, 1]."""
        from eidolon.livekit.plugins.vad.firered import PvadProcessor

        proc = PvadProcessor(force_cpu=True)
        # Synthetic silent frame: all zeros
        silent_frame = np.zeros(160, dtype=np.float32)
        prob = proc(silent_frame)

        assert isinstance(prob, float), f"Expected float, got {type(prob)}"
        assert 0.0 <= prob <= 1.0, f"Probability {prob} is out of [0, 1] range"

        # Synthetic speech-like frame: some energy
        speech_frame = np.random.randn(160).astype(np.float32) * 0.1
        prob2 = proc(speech_frame)
        assert 0.0 <= prob2 <= 1.0

        logging.getLogger("test_firered_vad").info(
            "test_processor_inference_shape: silent_prob=%.4f, noise_prob=%.4f",
            prob,
            prob2,
        )

    def test_processor_inference_invalid_shape(self):
        """Frames with incorrect length return 0.0 without crashing."""
        from eidolon.livekit.plugins.vad.firered import PvadProcessor

        proc = PvadProcessor(force_cpu=True)

        # Too short
        short_frame = np.zeros(80, dtype=np.float32)
        result = proc(short_frame)
        assert result == 0.0

        # Too long
        long_frame = np.zeros(320, dtype=np.float32)
        result = proc(long_frame)
        assert result == 0.0

    def test_processor_reset(self):
        """reset() zeros all state buffers."""
        from eidolon.livekit.plugins.vad.firered import PvadProcessor

        proc = PvadProcessor(force_cpu=True)

        # Run inference to mutate state
        frame = np.random.randn(160).astype(np.float32) * 0.1
        proc(frame)
        proc(frame)

        # Verify state was mutated
        assert not np.allclose(proc.mel_buffer, 0.0), "mel_buffer should be non-zero after inference"
        assert not np.allclose(proc.gru_buffer, 0.0), "gru_buffer should be non-zero after inference"

        # Reset
        proc.reset()

        # Verify state is back to zeros
        assert np.allclose(proc.mel_buffer, 0.0), "mel_buffer should be zero after reset"
        assert np.allclose(proc.gru_buffer, 0.0), "gru_buffer should be zero after reset"
        assert np.allclose(proc.spkemb, 0.0), "spkemb should be zero after reset"

    def test_speaker_embedding_extraction(self):
        """SpeakerEmbExtractor loads and produces non-zero embeddings when model is present."""
        from eidolon.livekit.plugins.vad.firered import PvadProcessor

        proc = PvadProcessor(force_cpu=True)

        # Skip if speechbrain is not installed or model is missing
        if not proc.spk_extractor.is_loaded:
            pytest.skip("speechbrain or ECAPA model not available — graceful degradation works")

        # 1 second of 16kHz audio
        audio = np.random.randn(16000).astype(np.float32) * 0.01
        emb = proc.spk_extractor.get_embedding(audio)

        assert emb.shape == (1, 192), f"Embedding shape should be (1, 192), got {emb.shape}"
        assert not np.allclose(emb, 0.0), "Embedding should not be all zeros"

        # Verify L2 normalization: ||emb||_2 ≈ 1.0
        norm = np.linalg.norm(emb)
        assert 0.99 < norm < 1.01, f"Embedding norm should be ~1.0, got {norm}"

        logging.getLogger("test_firered_vad").info(
            "test_speaker_embedding_extraction: norm=%.4f, loaded=%s",
            norm,
            proc.spk_extractor.is_loaded,
        )

    def test_speaker_embedding_graceful_degrade(self):
        """Without speechbrain, SpeakerEmbExtractor returns zero vector without crashing."""
        from eidolon.livekit.plugins.vad.firered.processor import SpeakerEmbExtractor

        # Load with a non-existent path to force the graceful-degraded path
        extractor = SpeakerEmbExtractor(ckpt_path="/nonexistent/path/to/model")

        # Should not raise — just log a warning
        audio = np.random.randn(16000).astype(np.float32)
        emb = extractor.get_embedding(audio)

        assert emb.shape == (1, 192)
        assert np.allclose(emb, 0.0), "Without speechbrain, should return zero vector"
        assert extractor.is_loaded is False

    def test_processor_update_speaker_embedding(self):
        """update_speaker_embedding() mutates spkemb without crashing."""
        from eidolon.livekit.plugins.vad.firered import PvadProcessor

        proc = PvadProcessor(force_cpu=True)

        # 1 second of audio
        audio = np.random.randn(16000).astype(np.float32) * 0.01
        proc.update_speaker_embedding(audio)

        # spkemb should have been updated (zero if speechbrain unavailable, non-zero otherwise)
        assert proc.spkemb.shape == (1, 192)

        # Running inference after update should not crash
        frame = np.zeros(160, dtype=np.float32)
        prob = proc(frame)
        assert 0.0 <= prob <= 1.0

    def test_processor_shared_state_isolation(self):
        """Two processors maintain independent state."""
        from eidolon.livekit.plugins.vad.firered import PvadProcessor

        proc_a = PvadProcessor(force_cpu=True)
        proc_b = PvadProcessor(force_cpu=True)

        frame_a = np.random.randn(160).astype(np.float32) * 0.1
        frame_b = np.random.randn(160).astype(np.float32) * 0.1

        prob_a = proc_a(frame_a)
        prob_b = proc_b(frame_b)

        # State buffers should differ because inference modified them differently
        # (unless by coincidence the random vectors produced identical outputs)
        assert not np.allclose(proc_a.gru_buffer, proc_b.gru_buffer), \
            "Two processors should have independent state"

        # Reset one should not affect the other
        proc_a.reset()
        assert np.allclose(proc_b.gru_buffer, 0.0) is False, \
            "Resetting proc_a should not affect proc_b"


# ---------------------------------------------------------------------------
# VAD class tests
# ---------------------------------------------------------------------------

class TestVadClass:
    """Tests for the VAD class (load, options, stream creation)."""

    def test_vad_load_defaults(self):
        """VAD.load() succeeds with default parameters."""
        from eidolon.livekit.plugins.vad.firered import FireredPvadVAD

        # Reset the class-level processor so we get a fresh one
        FireredPvadVAD._processor = None

        vad = FireredPvadVAD.load()
        assert vad is not None
        assert hasattr(vad, "stream")
        assert hasattr(vad, "update_options")

        # Re-reset for other tests
        FireredPvadVAD._processor = None

    def test_vad_load_custom_params(self):
        """VAD.load() accepts custom parameters."""
        from eidolon.livekit.plugins.vad.firered import FireredPvadVAD

        FireredPvadVAD._processor = None

        vad = FireredPvadVAD.load(
            activation_threshold=0.6,
            min_speech_duration=0.2,
            min_silence_duration=0.5,
            prefix_padding_duration=0.3,
            max_buffered_speech=30.0,
            force_cpu=True,
        )
        assert vad._opts.activation_threshold == 0.6
        assert vad._opts.min_speech_duration == 0.2
        assert vad._opts.min_silence_duration == 0.5
        assert vad._opts.prefix_padding_duration == 0.3
        assert vad._opts.max_buffered_speech == 30.0

        FireredPvadVAD._processor = None

    def test_vad_load_invalid_sample_rate(self):
        """VAD.load() raises ValueError for unsupported sample rates."""
        from eidolon.livekit.plugins.vad.firered import FireredPvadVAD

        FireredPvadVAD._processor = None

        with pytest.raises(ValueError, match="16 kHz"):
            FireredPvadVAD.load(sample_rate=8000)

        FireredPvadVAD._processor = None

    @pytest.mark.asyncio
    async def test_vad_update_options(self):
        """update_options() changes options and propagates to streams."""
        from eidolon.livekit.plugins.vad.firered import FireredPvadVAD

        FireredPvadVAD._processor = None

        vad = FireredPvadVAD.load(activation_threshold=0.5)
        stream = vad.stream()

        assert stream._opts.activation_threshold == 0.5

        vad.update_options(activation_threshold=0.7)
        assert stream._opts.activation_threshold == 0.7

        await stream.aclose()
        FireredPvadVAD._processor = None


# ---------------------------------------------------------------------------
# VADStream tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestVadStream:
    """Async tests for VADStream behavior."""

    async def test_vad_stream_basic(self, audio_chunks):
        """Creating a stream and pushing audio produces INFERENCE_DONE events."""
        from eidolon.livekit.plugins.vad.firered import FireredPvadVAD
        from livekit.agents import vad as lk_vad

        logger = logging.getLogger("test_firered_vad")
        logger.setLevel(logging.INFO)

        FireredPvadVAD._processor = None
        vad = FireredPvadVAD.load()
        stream = vad.stream()

        events = []
        inference_done_count = 0
        probs: list[float] = []

        async def consume():
            nonlocal inference_done_count
            async for ev in stream:
                events.append(ev)
                if ev.type == lk_vad.VADEventType.INFERENCE_DONE:
                    inference_done_count += 1
                    probs.append(ev.probability)
                    if inference_done_count <= 3:
                        logger.info(
                            "  [frame %d] ts=%.3fs prob=%.4f speech_dur=%.3fs silence_dur=%.3fs",
                            inference_done_count,
                            ev.timestamp,
                            ev.probability,
                            ev.speech_duration,
                            ev.silence_duration,
                        )

        task = asyncio.create_task(consume())

        # Push 10 frames (1 second of audio)
        logger.info("Pushing 10 frames (1s)...")
        for frame in audio_chunks[:10]:
            stream.push_frame(frame)
            await asyncio.sleep(0.001)

        stream.end_input()
        await asyncio.wait_for(task, timeout=10.0)

        assert inference_done_count > 0, f"Expected INFERENCE_DONE events, got none"
        assert len(events) > 0

        logger.info(
            "Result: %d INFERENCE_DONE events, probs: min=%.4f max=%.4f mean=%.4f",
            inference_done_count,
            min(probs) if probs else 0,
            max(probs) if probs else 0,
            sum(probs) / len(probs) if probs else 0,
        )

        FireredPvadVAD._processor = None

    async def test_vad_stream_speech_detection(self, audio_chunks):
        """Real audio triggers START_OF_SPEECH and END_OF_SPEECH events."""
        from eidolon.livekit.plugins.vad.firered import FireredPvadVAD
        from livekit.agents import vad as lk_vad

        logger = logging.getLogger("test_firered_vad")
        logger.setLevel(logging.INFO)

        FireredPvadVAD._processor = None
        vad = FireredPvadVAD.load(
            activation_threshold=0.5,
            min_speech_duration=0.05,  # 50ms to be more sensitive
            min_silence_duration=0.3,
        )
        stream = vad.stream()

        event_types = []
        probs: list[float] = []

        async def consume():
            async for ev in stream:
                event_types.append(ev.type)
                if ev.type == lk_vad.VADEventType.INFERENCE_DONE:
                    probs.append(ev.probability)

        task = asyncio.create_task(consume())

        logger.info("test_vad_stream_speech_detection: pushing all %d frames...", len(audio_chunks))
        for i, frame in enumerate(audio_chunks):
            stream.push_frame(frame)
            if i % 100 == 0:
                logger.info("  [pushed %d/%d frames]", i, len(audio_chunks))
            await asyncio.sleep(0.001)

        stream.end_input()

        try:
            await asyncio.wait_for(task, timeout=15.0)
        except asyncio.TimeoutError:
            pytest.fail("Timeout waiting for stream to finish — END_OF_SPEECH may not have fired")

        has_start = lk_vad.VADEventType.START_OF_SPEECH in event_types
        has_end = lk_vad.VADEventType.END_OF_SPEECH in event_types
        has_inference = lk_vad.VADEventType.INFERENCE_DONE in event_types

        assert has_inference, f"Expected INFERENCE_DONE events, got {set(event_types)}"

        logger.info(
            "test_vad_stream_speech_detection RESULT: "
            "total_events=%d, INFERENCE_DONE=%d, "
            "START_OF_SPEECH=%s, END_OF_SPEECH=%s",
            len(event_types),
            len(probs),
            has_start,
            has_end,
        )
        if probs:
            logger.info(
            "  probability stats: min=%.4f, max=%.4f, mean=%.4f",
                min(probs), max(probs), sum(probs) / len(probs)
            )

        FireredPvadVAD._processor = None

    async def test_vad_stream_update_speaker(self, audio_chunks):
        """update_speaker() does not crash and processes audio without error."""
        from eidolon.livekit.plugins.vad.firered import FireredPvadVAD

        FireredPvadVAD._processor = None
        vad = FireredPvadVAD.load()
        stream = vad.stream()

        async def consume():
            async for _ in stream:
                pass

        task = asyncio.create_task(consume())

        # Push some audio, then update speaker
        for frame in audio_chunks[:5]:
            stream.push_frame(frame)
            await asyncio.sleep(0.001)

        # Build a float32 audio array for update_speaker
        # Take first 1 second (16000 samples) from the first chunk
        first_chunk_data = np.frombuffer(bytes(audio_chunks[0].data), dtype=np.int16)
        audio_f32 = (first_chunk_data.astype(np.float32) / 32768.0)[:16000]

        # This should not raise
        stream.update_speaker(audio_f32)

        # Continue pushing
        for frame in audio_chunks[5:15]:
            stream.push_frame(frame)
            await asyncio.sleep(0.001)

        stream.end_input()
        await asyncio.wait_for(asyncio.shield(task), timeout=10.0)

        # If we got here without exception, the test passes
        FireredPvadVAD._processor = None

    async def test_vad_stream_reset(self, audio_chunks):
        """reset() allows a stream to be restarted."""
        from eidolon.livekit.plugins.vad.firered import FireredPvadVAD

        FireredPvadVAD._processor = None
        vad = FireredPvadVAD.load()
        stream = vad.stream()

        # First use: push a few frames
        events1 = []

        async def consume1():
            async for ev in stream:
                events1.append(ev)
                if len(events1) >= 3:
                    break

        task1 = asyncio.create_task(consume1())
        for frame in audio_chunks[:3]:
            stream.push_frame(frame)
            await asyncio.sleep(0.001)

        await asyncio.wait_for(task1, timeout=5.0)

        # Reset the stream (close it)
        await stream.aclose()

        # Create a new stream from the same VAD
        stream2 = vad.stream()
        events2 = []

        async def consume2():
            async for ev in stream2:
                events2.append(ev)
                if len(events2) >= 3:
                    break

        task2 = asyncio.create_task(consume2())
        for frame in audio_chunks[:3]:
            stream2.push_frame(frame)
            await asyncio.sleep(0.001)

        await asyncio.wait_for(task2, timeout=5.0)

        assert len(events2) >= 3, "New stream should produce events after reset"

        FireredPvadVAD._processor = None

    async def test_vad_stream_update_options_runtime(self):
        """update_options() called at runtime changes VAD behavior."""
        from eidolon.livekit.plugins.vad.firered import FireredPvadVAD

        FireredPvadVAD._processor = None
        vad = FireredPvadVAD.load(activation_threshold=0.5)
        stream = vad.stream()

        # Update options at runtime
        vad.update_options(activation_threshold=0.8)

        assert stream._opts.activation_threshold == 0.8
        assert vad._opts.activation_threshold == 0.8

        await stream.aclose()
        FireredPvadVAD._processor = None

    async def test_vad_stream_end_without_speech(self):
        """Pushing only silence does not trigger START_OF_SPEECH."""
        from eidolon.livekit.plugins.vad.firered import FireredPvadVAD
        from livekit.agents import vad as lk_vad

        FireredPvadVAD._processor = None
        vad = FireredPvadVAD.load(
            activation_threshold=0.5,
            min_speech_duration=0.1,
        )
        stream = vad.stream()

        speech_events = []

        async def consume():
            async for ev in stream:
                if ev.type in (lk_vad.VADEventType.START_OF_SPEECH, lk_vad.VADEventType.END_OF_SPEECH):
                    speech_events.append(ev.type)

        task = asyncio.create_task(consume())

        # Push 1 second of silence (16000 samples of zeros = 1 frame at 16kHz)
        silent_frame = rtc.AudioFrame(
            data=bytes(16000 * 2),  # 16000 int16 samples = 0 bytes
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=16000,
        )
        stream.push_frame(silent_frame)
        await asyncio.sleep(0.1)
        stream.end_input()
        await asyncio.wait_for(task, timeout=5.0)

        assert lk_vad.VADEventType.START_OF_SPEECH not in speech_events, \
            "Silence should not trigger START_OF_SPEECH"

        FireredPvadVAD._processor = None

    async def test_vad_event_probability_field(self, audio_chunks):
        """INFERENCE_DONE events carry a probability in [0, 1]."""
        from eidolon.livekit.plugins.vad.firered import FireredPvadVAD
        from livekit.agents import vad as lk_vad

        FireredPvadVAD._processor = None
        vad = FireredPvadVAD.load()
        stream = vad.stream()

        inference_probs = []

        async def consume():
            async for ev in stream:
                if ev.type == lk_vad.VADEventType.INFERENCE_DONE:
                    inference_probs.append(ev.probability)

        task = asyncio.create_task(consume())

        for frame in audio_chunks[:10]:
            stream.push_frame(frame)
            await asyncio.sleep(0.001)

        stream.end_input()
        await asyncio.wait_for(task, timeout=10.0)

        assert len(inference_probs) > 0, "Should have received INFERENCE_DONE events"
        for p in inference_probs:
            assert 0.0 <= p <= 1.0, f"Probability {p} is out of [0, 1]"

        logging.getLogger("test_firered_vad").info(
            "test_vad_event_probability_field: min=%.4f, max=%.4f, mean=%.4f",
            min(inference_probs),
            max(inference_probs),
            sum(inference_probs) / len(inference_probs),
        )

        FireredPvadVAD._processor = None


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRealAudioIntegration:
    """End-to-end integration test with real audio."""

    async def test_real_audio_full_pipeline(self, audio_chunks):
        """Process vad.m4a end-to-end: detect speech and emit events with per-frame logging."""
        from eidolon.livekit.plugins.vad.firered import FireredPvadVAD
        from livekit.agents import vad as lk_vad

        logger = logging.getLogger("test_firered_vad")
        logger.setLevel(logging.INFO)

        # Ensure INFO logs are visible in pytest output
        _handler = logging.StreamHandler()
        _handler.setLevel(logging.INFO)
        _handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(_handler)

        FireredPvadVAD._processor = None

        vad = FireredPvadVAD.load(
            activation_threshold=0.5,
            min_speech_duration=0.05,
            min_silence_duration=0.3,
        )
        stream = vad.stream()

        event_types: list[lk_vad.VADEventType] = []
        inference_probs: list[float] = []
        speech_durations: list[float] = []
        silence_durations: list[float] = []

        # Track speech/silence segments for detailed logging
        segment_log: list[dict] = []
        current_segment_start = 0
        current_is_speech = False

        async def consume():
            frame_idx = 0
            async for ev in stream:
                event_types.append(ev.type)

                if ev.type == lk_vad.VADEventType.INFERENCE_DONE:
                    inference_probs.append(ev.probability)
                    speech_durations.append(ev.speech_duration)
                    silence_durations.append(ev.silence_duration)

                    # Print every 10th frame for visibility
                    is_speech = ev.speech_duration >= vad._opts.min_speech_duration
                    marker = "SPEECH" if is_speech else "silent"
                    ts = ev.timestamp
                    if frame_idx % 10 == 0:
                        logger.info(
                            "  [frame %4d] ts=%.3fs prob=%.4f speech_dur=%.3fs "
                            "silence_dur=%.3fs → %s",
                            frame_idx,
                            ts,
                            ev.probability,
                            ev.speech_duration,
                            ev.silence_duration,
                            marker,
                        )

                    frame_idx += 1

                elif ev.type == lk_vad.VADEventType.START_OF_SPEECH:
                    logger.info(
                        ">>> SPEECH_START at frame %d, ts=%.3fs, speech_dur=%.3fs",
                        len(inference_probs),
                        ev.timestamp,
                        ev.speech_duration,
                    )
                    # Print a mini summary of the preceding silent segment
                    if silence_durations:
                        last_sil = silence_durations[-1]
                        logger.info(
                            "    [pre-speech silence: %.3fs]",
                            last_sil,
                        )

                elif ev.type == lk_vad.VADEventType.END_OF_SPEECH:
                    logger.info(
                        "<<< SPEECH_END   at frame %d, ts=%.3fs, speech_dur=%.3fs, "
                        "silence_dur=%.3fs",
                        len(inference_probs),
                        ev.timestamp,
                        ev.speech_duration,
                        ev.silence_duration,
                    )
                    # Print mini summary of the speech segment
                    if inference_probs:
                        probs_since_start = inference_probs[-20:] if len(inference_probs) > 20 else inference_probs
                        logger.info(
                            "    [segment max_prob=%.4f, mean_prob=%.4f, frames=%d]",
                            max(probs_since_start),
                            sum(probs_since_start) / len(probs_since_start),
                            len(probs_since_start),
                        )

        task = asyncio.create_task(consume())

        logger.info("=" * 60)
        logger.info("VAD REAL-AUDIO TEST: pushing %d audio frames (total %.1fs)...",
                    len(audio_chunks), sum(c.samples_per_channel for c in audio_chunks) / 16000)
        logger.info("Config: threshold=0.5, min_speech=0.05s, min_silence=0.3s")
        logger.info("=" * 60)

        # Push all audio
        for i, frame in enumerate(audio_chunks):
            stream.push_frame(frame)
            if i % 50 == 0:
                logger.info("  [pushed frame %d/%d]", i, len(audio_chunks))
            await asyncio.sleep(0.001)

        stream.end_input()
        logger.info("  [all frames pushed, waiting for stream to finish...]")

        try:
            await asyncio.wait_for(task, timeout=30.0)
        except asyncio.TimeoutError:
            pytest.fail(
                f"Timeout after {len(event_types)} events. "
                f"Last 5 types: {event_types[-5:]}"
            )

        logger.info("=" * 60)
        logger.info("RESULT: %d INFERENCE_DONE events, %d START_OF_SPEECH, %d END_OF_SPEECH",
                    sum(1 for t in event_types if t == lk_vad.VADEventType.INFERENCE_DONE),
                    sum(1 for t in event_types if t == lk_vad.VADEventType.START_OF_SPEECH),
                    sum(1 for t in event_types if t == lk_vad.VADEventType.END_OF_SPEECH))

        if inference_probs:
            max_p = max(inference_probs)
            mean_p = sum(inference_probs) / len(inference_probs)
            logger.info("Probability stats: max=%.4f, mean=%.4f, min=%.4f",
                        max_p, mean_p, min(inference_probs))

            # Find speech segments
            speech_frames = [p for p in inference_probs if p >= 0.5]
            logger.info("Speech frames (prob >= 0.5): %d / %d (%.1f%%)",
                        len(speech_frames), len(inference_probs),
                        100 * len(speech_frames) / len(inference_probs))
        logger.info("=" * 60)

        # Verify we got real inference results
        assert len(inference_probs) > 0, "Should have produced inference results"
        assert len(event_types) > 0, "Should have produced events"

        # At least some frames should have non-trivial probability
        max_prob = max(inference_probs) if inference_probs else 0.0
        assert max_prob > 0.01, f"Expected at least some speech energy, max_prob={max_prob}"

        FireredPvadVAD._processor = None


# ---------------------------------------------------------------------------
# VadStage minimal wrapper tests (mirrors SttStage / TtsStage)
# ---------------------------------------------------------------------------


class TestVadStageWrapper:
    """Smoke tests for the minimal VadStage DI wrapper."""

    @pytest.mark.asyncio
    async def test_vadstage_holds_vad_instance(self):
        """VadStage.vad property returns the wrapped instance unchanged."""
        from eidolon.livekit.agent.pipeline.vad import VadStage
        from eidolon.livekit.plugins.vad.firered import FireredPvadVAD

        underlying = FireredPvadVAD.load()
        stage = VadStage(underlying)
        assert stage.vad is underlying

        FireredPvadVAD._processor = None

    @pytest.mark.asyncio
    async def test_vadstage_warmup_shutdown_no_op_when_plugin_lacks_methods(self):
        """warmup/shutdown are safe no-ops when the underlying plugin
        has no warmup() / shutdown() (the FireRed case today)."""
        from eidolon.livekit.agent.pipeline.vad import VadStage
        from eidolon.livekit.plugins.vad.firered import FireredPvadVAD

        stage = VadStage(FireredPvadVAD.load())
        # Should not raise even though FireredPvadVAD has no warmup/shutdown.
        await stage.warmup()
        await stage.shutdown()

        FireredPvadVAD._processor = None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
