# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Headless AgentSession driver — runs the full pipeline in-memory.

Three building blocks:

  * ``ScriptedAudioInput`` — subclass of livekit-agents' ``AudioInput``.
    Test code feeds PCM bytes via ``feed_pcm`` / ``feed_silence`` and
    the AgentSession pulls AudioFrames via ``__anext__``. When all
    queued audio is consumed and ``end()`` has been called, the
    iterator raises ``StopAsyncIteration``.

  * ``RecordingAudioOutput`` — subclass of ``AudioOutput``. Captures
    every ``capture_frame`` call into an in-memory buffer. Exposes
    ``collected_pcm`` (raw bytes) and ``flushes`` (segment markers)
    plus an asyncio.Event that fires on first non-silent capture for
    "wait until agent starts speaking" assertions.

  * ``HeadlessSession`` — async context manager that wires everything
    together: AgentSession + Agent + mock plugins + ScriptedAudioInput
    + RecordingAudioOutput + EventRecorder. Yields a tuple of
    ``(session, audio_in, audio_out, events)`` to the test.

Critical: AgentSession.start() is called WITHOUT a ``room=`` argument.
The framework's lifecycle code skips RoomIO setup when both
``input.audio`` and ``output.audio`` are pre-populated, so we never
need a real LiveKit server.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional

from livekit import rtc
from livekit.agents import llm as lk_llm
from livekit.agents import stt as lk_stt
from livekit.agents import tts as lk_tts
from livekit.agents import vad as lk_vad
from livekit.agents.voice import (
    Agent,
    AgentSession,
)
from livekit.agents.voice.events import (
    AgentStateChangedEvent,
    CloseEvent,
    ConversationItemAddedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
)
from livekit.agents.voice.io import (
    AudioInput,
    AudioOutput,
    AudioOutputCapabilities,
)

from .audio import (
    DEFAULT_CHANNELS,
    DEFAULT_SAMPLE_RATE,
    SAMPLE_WIDTH_BYTES,
    frames_from_pcm,
)


# ─────────────────────────────────────────────────────────────────
#  ScriptedAudioInput
# ─────────────────────────────────────────────────────────────────


class ScriptedAudioInput(AudioInput):
    """In-memory AudioInput. Test code pushes PCM; AgentSession pulls
    AudioFrames.

    Usage:
        >>> audio_in = ScriptedAudioInput()
        >>> audio_in.feed_pcm(synth_voiced(0.5))
        >>> audio_in.feed_silence(0.2)
        >>> # ... pipeline consumes frames ...
        >>> audio_in.end()  # iterator raises StopAsyncIteration once drained
    """

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        num_channels: int = DEFAULT_CHANNELS,
        frame_ms: int = 20,
        label: str = "ScriptedAudioInput",
    ) -> None:
        super().__init__(label=label)
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._frame_ms = frame_ms
        self._queue: deque[rtc.AudioFrame] = deque()
        self._has_data: asyncio.Event = asyncio.Event()
        self._ended: bool = False
        # Track total bytes pushed for assertions / coordination.
        self._bytes_pushed: int = 0
        self._frames_consumed: int = 0
        # Pacing (default: real-time). Tests can disable via ``pacing=False``
        # for deterministic, fast scenario execution.
        self._real_time_pacing: bool = False
        self._next_frame_at: Optional[float] = None

    # ── public API ───────────────────────────────────────────────

    def feed_pcm(self, pcm: bytes) -> None:
        """Push raw 16-bit PCM bytes. Will be chunked into frames."""
        if self._ended:
            raise RuntimeError("ScriptedAudioInput already ended()")
        if not pcm:
            return
        for f in frames_from_pcm(
            pcm,
            sample_rate=self._sample_rate,
            frame_ms=self._frame_ms,
            num_channels=self._num_channels,
        ):
            self._queue.append(f)
        self._bytes_pushed += len(pcm)
        self._has_data.set()

    def feed_silence(self, duration_sec: float) -> None:
        """Push ``duration_sec`` of silence."""
        if self._ended:
            raise RuntimeError("ScriptedAudioInput already ended()")
        n_bytes = int(duration_sec * self._sample_rate) * SAMPLE_WIDTH_BYTES * self._num_channels
        # Round to whole frames.
        bytes_per_frame = (
            self._sample_rate
            * self._frame_ms
            * SAMPLE_WIDTH_BYTES
            * self._num_channels
            // 1000
        )
        n_bytes = (n_bytes // bytes_per_frame) * bytes_per_frame
        if n_bytes == 0:
            return
        self.feed_pcm(b"\x00" * n_bytes)

    def end(self) -> None:
        """Signal end of input. Iterator drains and raises
        StopAsyncIteration when queue empties."""
        self._ended = True
        # Wake any pending __anext__ so it can re-check ended state.
        self._has_data.set()

    @property
    def bytes_pushed(self) -> int:
        return self._bytes_pushed

    @property
    def frames_consumed(self) -> int:
        return self._frames_consumed

    def set_real_time_pacing(self, enabled: bool) -> None:
        """When True, frames are released at ``frame_ms`` real-time
        intervals — match production audio capture rate. Default off
        for fast scenario tests."""
        self._real_time_pacing = enabled
        self._next_frame_at = None

    async def wait_drained(self, *, timeout: float = 5.0) -> None:
        """Wait until the consumer has drained the queue."""
        deadline = time.monotonic() + timeout
        while self._queue:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"queue not drained within {timeout}s ({len(self._queue)} frames left)"
                )
            await asyncio.sleep(0.01)

    # ── AudioInput protocol ──────────────────────────────────────

    async def __anext__(self) -> rtc.AudioFrame:
        # Block until we have a frame or the input is ended.
        while not self._queue:
            if self._ended:
                raise StopAsyncIteration
            self._has_data.clear()
            await self._has_data.wait()

        if self._real_time_pacing:
            now = time.monotonic()
            if self._next_frame_at is None:
                self._next_frame_at = now
            else:
                wait = self._next_frame_at - now
                if wait > 0:
                    await asyncio.sleep(wait)
            self._next_frame_at += self._frame_ms / 1000.0

        frame = self._queue.popleft()
        self._frames_consumed += 1
        if not self._queue and not self._ended:
            self._has_data.clear()
        return frame


# ─────────────────────────────────────────────────────────────────
#  RecordingAudioOutput
# ─────────────────────────────────────────────────────────────────


@dataclass
class _Segment:
    """A single playback segment (between flush()/clear_buffer())."""

    pcm_chunks: list[bytes] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)
    ended_at: Optional[float] = None
    cleared: bool = False  # True if clear_buffer() was called (interrupt)

    @property
    def pcm(self) -> bytes:
        return b"".join(self.pcm_chunks)

    @property
    def duration_bytes(self) -> int:
        return sum(len(c) for c in self.pcm_chunks)


class RecordingAudioOutput(AudioOutput):
    """In-memory AudioOutput. Captures every ``capture_frame`` call
    into a buffer organized into segments.

    Usage:
        >>> audio_out = RecordingAudioOutput()
        >>> # ... pipeline produces TTS ...
        >>> assert len(audio_out.collected_pcm) > 0
        >>> assert audio_out.segment_count == 1

    Each ``flush()`` finalizes a segment and emits a synthetic
    ``playback_finished`` event so AgentSession's playback-tracking
    state machine advances correctly.
    """

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        num_channels: int = DEFAULT_CHANNELS,
        label: str = "RecordingAudioOutput",
    ) -> None:
        super().__init__(
            label=label,
            capabilities=AudioOutputCapabilities(pause=False),
            sample_rate=sample_rate,
        )
        self._num_channels = num_channels
        self._segments: list[_Segment] = []
        self._current: Optional[_Segment] = None
        self._first_audio_event: asyncio.Event = asyncio.Event()
        self._captured_bytes: int = 0

    # ── public observables ───────────────────────────────────────

    @property
    def collected_pcm(self) -> bytes:
        """All captured PCM concatenated across segments."""
        return b"".join(s.pcm for s in self._segments) + (
            self._current.pcm if self._current else b""
        )

    @property
    def segments(self) -> list[_Segment]:
        return list(self._segments)

    @property
    def segment_count(self) -> int:
        return len(self._segments)

    @property
    def captured_bytes(self) -> int:
        return self._captured_bytes

    @property
    def first_audio_event(self) -> asyncio.Event:
        """Set when the first non-empty audio frame is captured."""
        return self._first_audio_event

    async def wait_for_first_audio(self, *, timeout: float = 5.0) -> None:
        """Wait until any audio data has been captured."""
        await asyncio.wait_for(self._first_audio_event.wait(), timeout=timeout)

    # ── AudioOutput protocol ─────────────────────────────────────

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        new_segment = self._current is None
        if new_segment:
            self._current = _Segment()
        data = bytes(frame.data)
        self._current.pcm_chunks.append(data)
        self._captured_bytes += len(data)
        if data and not self._first_audio_event.is_set():
            self._first_audio_event.set()
        # Emit playback_started on the first frame of each segment.
        # The framework's _AudioOutput wrapper subscribes to this and
        # uses it to resolve first_frame_fut → drives the agent state
        # transition to "speaking".
        if new_segment:
            self.on_playback_started(created_at=time.time())

    def flush(self) -> None:
        super().flush()
        if self._current is None:
            return
        self._current.ended_at = time.monotonic()
        self._segments.append(self._current)
        # Compute playback position (seconds of audio in this segment).
        sr = self.sample_rate or DEFAULT_SAMPLE_RATE
        bytes_per_sec = sr * SAMPLE_WIDTH_BYTES * self._num_channels
        if bytes_per_sec > 0:
            playback_position = self._current.duration_bytes / bytes_per_sec
        else:
            playback_position = 0.0
        self._current = None
        # Notify AgentSession that this segment finished playing.
        self.on_playback_finished(
            playback_position=playback_position,
            interrupted=False,
        )

    def clear_buffer(self) -> None:
        # Mark current segment as cleared (interrupt) and finalize.
        if self._current is None:
            return
        self._current.cleared = True
        self._current.ended_at = time.monotonic()
        self._segments.append(self._current)
        sr = self.sample_rate or DEFAULT_SAMPLE_RATE
        bytes_per_sec = sr * SAMPLE_WIDTH_BYTES * self._num_channels
        playback_position = (
            self._current.duration_bytes / bytes_per_sec if bytes_per_sec > 0 else 0.0
        )
        self._current = None
        self.on_playback_finished(
            playback_position=playback_position, interrupted=True
        )


# ─────────────────────────────────────────────────────────────────
#  EventRecorder
# ─────────────────────────────────────────────────────────────────


@dataclass
class RecordedEvent:
    """One snapshot of a session event (chronological log entry)."""

    type: str
    """The event type from livekit-agents EventTypes literal."""

    payload: Any
    """The original event object (UserStateChangedEvent / etc.)."""

    timestamp: float = field(default_factory=time.monotonic)
    """Monotonic time when we observed the event (test-relative)."""


class EventRecorder:
    """Subscribes to all AgentSession events and records them.

    Useful for scenario tests:
        >>> # Did the agent enter "speaking" state?
        >>> assert any(
        ...     e.type == "agent_state_changed" and e.payload.new_state == "speaking"
        ...     for e in events.all
        ... )
        >>> # Was there exactly one user-final transcript?
        >>> finals = events.user_finals()
        >>> assert len(finals) == 1
    """

    # Events we subscribe to. Skipping noisy/internal ones.
    EVENT_TYPES = (
        "user_state_changed",
        "agent_state_changed",
        "user_input_transcribed",
        "conversation_item_added",
        "agent_false_interruption",
        "speech_created",
        "error",
        "close",
    )

    def __init__(self, session: AgentSession) -> None:
        self._session = session
        self._events: list[RecordedEvent] = []
        self._listeners: list[tuple[str, Callable]] = []
        for et in self.EVENT_TYPES:
            cb = self._make_listener(et)
            session.on(et, cb)
            self._listeners.append((et, cb))

    def _make_listener(self, event_type: str) -> Callable:
        def _listener(payload: Any) -> None:
            self._events.append(RecordedEvent(type=event_type, payload=payload))

        return _listener

    # ── observability helpers ───────────────────────────────────

    @property
    def all(self) -> list[RecordedEvent]:
        return list(self._events)

    def of_type(self, event_type: str) -> list[RecordedEvent]:
        return [e for e in self._events if e.type == event_type]

    def user_finals(self) -> list[UserInputTranscribedEvent]:
        return [
            e.payload
            for e in self._events
            if e.type == "user_input_transcribed" and e.payload.is_final
        ]

    def user_interims(self) -> list[UserInputTranscribedEvent]:
        return [
            e.payload
            for e in self._events
            if e.type == "user_input_transcribed" and not e.payload.is_final
        ]

    def agent_state_history(self) -> list[str]:
        """All agent_state transitions, in order. e.g.
        ['initializing', 'idle', 'listening', 'thinking', 'speaking']"""
        states = []
        for e in self._events:
            if e.type == "agent_state_changed":
                if not states or states[-1] != e.payload.new_state:
                    states.append(e.payload.new_state)
        return states

    def user_state_history(self) -> list[str]:
        states = []
        for e in self._events:
            if e.type == "user_state_changed":
                if not states or states[-1] != e.payload.new_state:
                    states.append(e.payload.new_state)
        return states

    def agent_messages(self) -> list[str]:
        """Text content of every assistant ChatMessage added to the
        conversation."""
        out: list[str] = []
        for e in self._events:
            if e.type != "conversation_item_added":
                continue
            item = e.payload.item
            if getattr(item, "role", None) != "assistant":
                continue
            text = getattr(item, "text_content", None)
            if text:
                out.append(text)
        return out

    def user_messages(self) -> list[str]:
        out: list[str] = []
        for e in self._events:
            if e.type != "conversation_item_added":
                continue
            item = e.payload.item
            if getattr(item, "role", None) != "user":
                continue
            text = getattr(item, "text_content", None)
            if text:
                out.append(text)
        return out

    async def wait_for(
        self,
        predicate: Callable[[RecordedEvent], bool],
        *,
        timeout: float = 5.0,
        poll_ms: int = 10,
    ) -> RecordedEvent:
        """Block until an event matching ``predicate`` arrives.

        The predicate is evaluated against the cumulative event log
        (i.e. matches both already-recorded events and future ones).
        """
        deadline = time.monotonic() + timeout
        while True:
            for ev in self._events:
                if predicate(ev):
                    return ev
            if time.monotonic() >= deadline:
                snapshot = ", ".join(e.type for e in self._events[-10:])
                raise TimeoutError(
                    f"event predicate not satisfied within {timeout}s. "
                    f"recent events: [{snapshot}]"
                )
            await asyncio.sleep(poll_ms / 1000.0)

    async def wait_for_state(
        self,
        *,
        agent: Optional[str] = None,
        user: Optional[str] = None,
        timeout: float = 5.0,
    ) -> RecordedEvent:
        """Wait for a specific agent_state or user_state transition."""
        if agent is not None:
            return await self.wait_for(
                lambda e: e.type == "agent_state_changed"
                and e.payload.new_state == agent,
                timeout=timeout,
            )
        if user is not None:
            return await self.wait_for(
                lambda e: e.type == "user_state_changed"
                and e.payload.new_state == user,
                timeout=timeout,
            )
        raise ValueError("must specify agent= or user=")


# ─────────────────────────────────────────────────────────────────
#  HeadlessSession factory
# ─────────────────────────────────────────────────────────────────


@dataclass
class HeadlessHandle:
    """Returned to test code from ``headless_session()``."""

    session: AgentSession
    audio_in: ScriptedAudioInput
    audio_out: RecordingAudioOutput
    events: EventRecorder
    agent: Agent


@asynccontextmanager
async def headless_session(
    *,
    llm: Optional[lk_llm.LLM] = None,
    stt: Optional[lk_stt.STT] = None,
    tts: Optional[lk_tts.TTS] = None,
    vad: Optional[lk_vad.VAD] = None,
    instructions: str = "You are a helpful test assistant.",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    real_time_audio: bool = False,
    user_away_timeout: Optional[float] = None,
    aec_warmup_duration: Optional[float] = None,
    extra_session_kwargs: Optional[dict] = None,
    extra_agent_kwargs: Optional[dict] = None,
) -> AsyncIterator[HeadlessHandle]:
    """Async context manager wiring AgentSession + mocks + scripted IO.

    On entry: builds AgentSession, installs ScriptedAudioInput +
    RecordingAudioOutput, calls ``session.start(agent)`` (no room).
    On exit: closes the session cleanly so transient tasks shut down.

    Defaults are tuned for fast tests:
      * ``user_away_timeout=None`` — disable the 15s away timer (noisy
        in scenarios that pause)
      * ``aec_warmup_duration=None`` — agent does not refuse to be
        interrupted during a 3s warm-up
      * ``real_time_audio=False`` — frames release as fast as the
        consumer can pull them

    Pass ``extra_session_kwargs`` / ``extra_agent_kwargs`` to override
    or add more AgentSession / Agent constructor arguments.
    """
    if llm is None or stt is None or tts is None or vad is None:
        raise ValueError(
            "headless_session requires all four mock plugins: llm, stt, tts, vad"
        )

    audio_in = ScriptedAudioInput(sample_rate=sample_rate)
    audio_in.set_real_time_pacing(real_time_audio)
    audio_out = RecordingAudioOutput(sample_rate=sample_rate)

    session_kwargs: dict[str, Any] = {
        "stt": stt,
        "vad": vad,
        "llm": llm,
        "tts": tts,
        "user_away_timeout": user_away_timeout,
        "aec_warmup_duration": aec_warmup_duration,
    }
    if extra_session_kwargs:
        session_kwargs.update(extra_session_kwargs)
    session = AgentSession(**session_kwargs)
    session.input.audio = audio_in
    session.output.audio = audio_out

    agent_kwargs: dict[str, Any] = {"instructions": instructions}
    if extra_agent_kwargs:
        agent_kwargs.update(extra_agent_kwargs)
    agent = Agent(**agent_kwargs)

    events = EventRecorder(session)

    try:
        await session.start(agent)
        yield HeadlessHandle(
            session=session,
            audio_in=audio_in,
            audio_out=audio_out,
            events=events,
            agent=agent,
        )
    finally:
        try:
            audio_in.end()
        except Exception:
            pass
        try:
            await asyncio.wait_for(session.aclose(), timeout=5.0)
        except (asyncio.TimeoutError, Exception):
            # Best-effort cleanup. Tests can re-raise if they care.
            pass
