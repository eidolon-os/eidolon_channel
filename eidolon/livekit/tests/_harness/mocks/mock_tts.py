# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MockTTS — deterministic, in-process TTS stand-in.

Conforms to ``livekit.agents.tts.TTS`` (chunked synth API). Generates
synthetic PCM proportional to the input text length:

    duration_sec = len(text) * char_seconds   (default 0.08s/char)

The PCM signal is a low-amplitude voiced buzz (``synth_voiced``) so
downstream RecordingAudioOutput can verify "TTS output is non-silent
and within expected duration" without an audio decoder.

Designed to mimic the per-frame streaming behavior of real TTS: PCM
is emitted in 200 ms chunks via ``AudioEmitter.push``, with optional
inter-chunk delays for testing back-pressure / interrupt-mid-synth.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Optional

from livekit.agents.tts import (
    TTS,
    AudioEmitter,
    ChunkedStream,
    TTSCapabilities,
)
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

from ..audio import DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE, synth_voiced


class MockTTS(TTS):
    """Deterministic TTS mock.

    Examples:
        >>> tts = MockTTS()                    # 0.08s/char
        >>> tts = MockTTS(char_seconds=0.05)   # faster
        >>> tts = MockTTS(first_chunk_delay_ms=200)  # simulate slow first byte
        >>> tts = MockTTS.errors_with(RuntimeError("upstream"))
    """

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        num_channels: int = DEFAULT_CHANNELS,
        char_seconds: float = 0.08,
        chunk_ms: int = 200,
        first_chunk_delay_ms: int = 0,
        chunk_delay_ms: int = 0,
        always_raises: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            capabilities=TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        self._char_seconds = max(0.001, char_seconds)
        self._chunk_ms = max(1, chunk_ms)
        self._first_chunk_delay_ms = max(0, first_chunk_delay_ms)
        self._chunk_delay_ms = max(0, chunk_delay_ms)
        self._always_raises = always_raises
        # Test counters.
        self.synth_count = 0
        self.synth_texts: list[str] = []

    # ─────────────────────────────────────── factories

    @classmethod
    def errors_with(cls, exc: Exception, **kwargs) -> "MockTTS":
        return cls(always_raises=exc, **kwargs)

    # ─────────────────────────────────────── protocol

    @property
    def model(self) -> str:
        return "mock-tts"

    @property
    def provider(self) -> str:
        return "eidolon-test"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        self.synth_count += 1
        self.synth_texts.append(text)
        return _MockTTSChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )


class _MockTTSChunkedStream(ChunkedStream):
    """Streaming impl. Synth PCM is computed once at start and
    pushed in ``chunk_ms`` slices."""

    async def _run(self, output_emitter: AudioEmitter) -> None:
        mock: MockTTS = self._tts  # type: ignore[assignment]
        if mock._always_raises is not None:
            raise mock._always_raises

        # Compute one big PCM blob proportional to text length.
        # We deliberately ignore text content — tests should not
        # assume MockTTS PCM resembles speech of the input.
        text = self._input_text or ""
        duration_sec = max(
            0.05, min(60.0, len(text) * mock._char_seconds)
        ) if text else 0.05
        pcm = synth_voiced(
            duration_sec, sample_rate=mock.sample_rate, amplitude=0.3
        )

        # Initialize the emitter (must happen before any push()).
        request_id = f"mock-tts-{uuid.uuid4().hex[:8]}"
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=mock.sample_rate,
            num_channels=mock.num_channels,
            mime_type="audio/pcm",
            frame_size_ms=mock._chunk_ms,
            stream=False,
        )

        if mock._first_chunk_delay_ms:
            await asyncio.sleep(mock._first_chunk_delay_ms / 1000.0)

        # Slice PCM into chunk_ms pieces and push each.
        bytes_per_chunk = (
            mock.sample_rate
            * mock._chunk_ms
            * 2  # 16-bit
            * mock.num_channels
            // 1000
        )
        for offset in range(0, len(pcm), bytes_per_chunk):
            chunk = pcm[offset : offset + bytes_per_chunk]
            output_emitter.push(chunk)
            if mock._chunk_delay_ms:
                await asyncio.sleep(mock._chunk_delay_ms / 1000.0)
        # _main_task will call output_emitter.end_input() / join()
        # after _run returns.
