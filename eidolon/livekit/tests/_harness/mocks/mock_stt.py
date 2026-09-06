# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MockSTT — scripted in-process STT stand-in.

Conforms to ``livekit.agents.stt.STT`` (streaming + batch). Useful for
scenario tests where you want to drive specific transcripts at chosen
points in time without bringing up MockSTTServer / MockFunASRServer.

Two driving modes:

1. **Time-based scripts** (default): a list of ``ScriptedTranscript``
   entries, each with a ``trigger_after_ms`` from stream start. The
   stream emits interim/final events on schedule regardless of audio
   content. Audio frames are still consumed (no back-pressure on the
   pipeline) but ignored. Best for deterministic scenario tests.

2. **Audio-byte-threshold scripts**: each entry fires after N total
   PCM bytes have been pushed in. Useful when the test cares about
   "transcript appears after the user has spoken N seconds of audio".

Each ``ScriptedTranscript`` produces:
   * START_OF_SPEECH on first event of a segment (auto)
   * INTERIM_TRANSCRIPT(s) per ``interims`` list (optional)
   * FINAL_TRANSCRIPT with ``text`` (or skip if ``final`` is None)
   * END_OF_SPEECH after the final (auto)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from livekit.agents.stt import (
    STT,
    RecognizeStream,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    STTCapabilities,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)


@dataclass
class ScriptedTranscript:
    """One scripted utterance the mock will emit."""

    text: str
    """The final transcript text. If None, only interims emit
    (useful to test "user is talking but never lands a final")."""

    trigger_after_ms: int = 0
    """Wall-time milliseconds from stream start to wait before
    emitting this entry. 0 = emit ASAP."""

    after_pcm_bytes: int = 0
    """Alternative trigger: total PCM bytes pushed to the stream.
    If both ``trigger_after_ms`` and ``after_pcm_bytes`` are set,
    whichever fires last wins."""

    interims: list[str] = field(default_factory=list)
    """Interim transcripts emitted before the final, in order. Each
    is sent as a separate INTERIM_TRANSCRIPT event."""

    interim_gap_ms: int = 50
    """Gap between successive interim events (and between the last
    interim and the final). Real STTs typically do 50–150 ms."""

    language: str = "zh"

    confidence: float = 0.95

    metadata: dict[str, Any] | None = None
    """Optional public provider evidence, forwarded without interpreting it."""

    final: Optional[bool] = True
    """When False, only interims are emitted (no FINAL_TRANSCRIPT
    or END_OF_SPEECH). Test stale-interim handling."""


class MockSTT(STT):
    """Deterministic streaming STT mock.

    Examples:
        >>> stt = MockSTT.scripted([
        ...     ScriptedTranscript(text="你好", trigger_after_ms=300),
        ...     ScriptedTranscript(text="你好吗", trigger_after_ms=800,
        ...                        interims=["你好"]),
        ... ])
        >>> stream = stt.stream()
        >>> stream.push_frame(some_audio_frame)
        >>> async for event in stream:
        ...     ...   # observe START_OF_SPEECH / INTERIM / FINAL / END
    """

    def __init__(
        self,
        *,
        scripts: Iterable[ScriptedTranscript] | None = None,
        always_raises: Optional[Exception] = None,
        sample_rate: int = 16_000,
    ) -> None:
        super().__init__(
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=True,
                offline_recognize=True,
            )
        )
        self._scripts: list[ScriptedTranscript] = list(scripts or [])
        self._always_raises = always_raises
        self._sample_rate = sample_rate
        # Recorded events for test assertions.
        self.streams_opened: int = 0
        self.bytes_pushed: int = 0

    # ─────────────────────────────────────── factories

    @classmethod
    def scripted(
        cls,
        entries: Iterable[ScriptedTranscript | tuple[str, int]],
        **kwargs,
    ) -> "MockSTT":
        """Build from full ScriptedTranscript or (text, trigger_after_ms) pairs."""
        normalized: list[ScriptedTranscript] = []
        for e in entries:
            if isinstance(e, ScriptedTranscript):
                normalized.append(e)
            elif isinstance(e, tuple) and len(e) == 2:
                normalized.append(
                    ScriptedTranscript(text=e[0], trigger_after_ms=e[1])
                )
            else:
                raise TypeError(f"unsupported script entry: {type(e)}")
        return cls(scripts=normalized, **kwargs)

    @classmethod
    def errors_with(cls, exc: Exception, **kwargs) -> "MockSTT":
        return cls(always_raises=exc, **kwargs)

    # ─────────────────────────────────────── protocol

    @property
    def model(self) -> str:
        return "mock-stt"

    @property
    def provider(self) -> str:
        return "eidolon-test"

    async def _recognize_impl(
        self,
        buffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechEvent:
        """Batch recognize — return the first scripted entry's text
        (or empty), wrapped as FINAL_TRANSCRIPT.

        Sufficient for component tests that need one-shot recognition; not
        intended to model real-world batch latency.
        """
        if self._always_raises is not None:
            raise self._always_raises
        text = self._scripts[0].text if self._scripts else ""
        lang_str: str = (
            language if isinstance(language, str) and language else "zh"
        )
        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            request_id=f"mock-stt-{uuid.uuid4().hex[:8]}",
            alternatives=[
                SpeechData(language=lang_str, text=text, confidence=0.95)
            ],
        )

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> RecognizeStream:
        self.streams_opened += 1
        return _MockSTTStream(stt=self, conn_options=conn_options)


class _MockSTTStream(RecognizeStream):
    """Streaming impl. Drives scripted events on schedule."""

    def __init__(
        self,
        *,
        stt: MockSTT,
        conn_options: APIConnectOptions,
    ) -> None:
        self._mock = stt
        self._stream_started = time.monotonic()
        self._bytes_pushed = 0
        super().__init__(stt=stt, conn_options=conn_options)

    async def _run(self) -> None:
        if self._mock._always_raises is not None:
            raise self._mock._always_raises

        # Drive two concurrent loops:
        #   1. consume incoming audio frames (track bytes pushed)
        #   2. emit scripted events on schedule (time- or byte-triggered)
        # We don't care about audio content — just account bytes for
        # ``after_pcm_bytes`` trigger.
        consume_task = asyncio.create_task(
            self._consume_audio_loop(), name="MockSTT._consume"
        )
        emit_task = asyncio.create_task(
            self._emit_events_loop(), name="MockSTT._emit"
        )
        try:
            await asyncio.gather(emit_task, return_exceptions=False)
        finally:
            # Drain audio loop so input ch closes cleanly.
            if not consume_task.done():
                consume_task.cancel()
                try:
                    await consume_task
                except (asyncio.CancelledError, BaseException):
                    pass

    async def _consume_audio_loop(self) -> None:
        """Read frames from input channel; just count bytes."""
        try:
            async for item in self._input_ch:
                if isinstance(item, RecognizeStream._FlushSentinel):
                    continue
                # rtc.AudioFrame.data is a memoryview/bytes-like.
                self._bytes_pushed += len(bytes(item.data))
                self._mock.bytes_pushed += len(bytes(item.data))
        except asyncio.CancelledError:
            pass

    async def _emit_events_loop(self) -> None:
        """Emit each scripted entry's events on schedule."""
        scripts = list(self._mock._scripts)
        already_in_segment = False

        for entry in scripts:
            # Wait for both triggers (if specified).
            target_t = self._stream_started + entry.trigger_after_ms / 1000.0
            now = time.monotonic()
            if target_t > now:
                await asyncio.sleep(target_t - now)
            if entry.after_pcm_bytes > 0:
                while self._bytes_pushed < entry.after_pcm_bytes:
                    await asyncio.sleep(0.005)

            # ── Segment START
            if not already_in_segment:
                self._send(
                    SpeechEvent(
                        type=SpeechEventType.START_OF_SPEECH,
                        request_id=self._req_id(),
                    )
                )
                already_in_segment = True

            # ── Interims
            for interim in entry.interims:
                self._send(
                    SpeechEvent(
                        type=SpeechEventType.INTERIM_TRANSCRIPT,
                        request_id=self._req_id(),
                        alternatives=[
                            SpeechData(
                                language=entry.language,
                                text=interim,
                                confidence=entry.confidence,
                                metadata=entry.metadata,
                            )
                        ],
                    )
                )
                if entry.interim_gap_ms:
                    await asyncio.sleep(entry.interim_gap_ms / 1000.0)

            # ── Final (optional)
            if entry.final:
                self._send(
                    SpeechEvent(
                        type=SpeechEventType.FINAL_TRANSCRIPT,
                        request_id=self._req_id(),
                        alternatives=[
                            SpeechData(
                                language=entry.language,
                                text=entry.text,
                                confidence=entry.confidence,
                                metadata=entry.metadata,
                            )
                        ],
                    )
                )
                self._send(
                    SpeechEvent(
                        type=SpeechEventType.END_OF_SPEECH,
                        request_id=self._req_id(),
                    )
                )
                already_in_segment = False

    # ── helpers

    def _send(self, event: SpeechEvent) -> None:
        if not self._event_ch.closed:
            self._event_ch.send_nowait(event)

    def _req_id(self) -> str:
        return f"mock-stt-{uuid.uuid4().hex[:8]}"
