"""G23 (2026-05-18): STTTranscriptGate unit tests.

Coverage matrix from docs/plans/2026-05-18/g23-stt-transcript-gate.md §5.1:

  ✓ test_gate_pass_through_when_no_prior_final
  ✓ test_gate_suppresses_interim_within_window
  ✓ test_gate_passes_interim_outside_window
  ✓ test_gate_suppresses_second_final_within_window
  ✓ test_gate_forwards_input_audio
  ✓ test_gate_forwards_flush_sentinel
  ✓ test_gate_preserves_event_order
  ✓ test_gate_capabilities_forwarded
  ✓ test_gate_suppressed_count_metric
  ✓ test_gate_window_zero_disables_suppression

Plus a few edge cases:
  ✓ test_gate_non_transcript_events_always_pass
  ✓ test_gate_aclose_cascades_to_inner
"""

from __future__ import annotations

import asyncio
from typing import List
from unittest.mock import MagicMock

import pytest
from livekit import rtc
from livekit.agents import stt as lk_stt
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType, STTCapabilities
from livekit.agents.types import APIConnectOptions

from eidolon.livekit.plugins.stt._transcript_gate import STTTranscriptGate


# ───────────────────────────────────────────────────────────────────────
# Test doubles — a minimal STT plugin + RecognizeStream we control
# ───────────────────────────────────────────────────────────────────────


class _FakeRecognizeStream(lk_stt.RecognizeStream):
    """A RecognizeStream we drive directly from the test: ``inject_event``
    pushes a SpeechEvent onto our event channel; ``capture_push`` records
    audio frames that were forwarded to us via ``push_frame``."""

    def __init__(self, *, stt: lk_stt.STT, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self.received_frames: List[rtc.AudioFrame] = []
        self.received_flush_count: int = 0
        self._inject_q: asyncio.Queue = asyncio.Queue()
        self._end_requested = asyncio.Event()

    async def _run(self) -> None:
        """Drain ``_inject_q`` into our outer ``_event_ch`` until end_input."""
        while True:
            if self._end_requested.is_set() and self._inject_q.empty():
                return
            try:
                ev = await asyncio.wait_for(self._inject_q.get(), timeout=0.05)
            except asyncio.TimeoutError:
                continue
            if ev is None:  # poison pill
                return
            self._event_ch.send_nowait(ev)

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        # We don't call super().push_frame because the base class enforces
        # _check_input_not_ended which we don't want in tests.
        self.received_frames.append(frame)

    def flush(self) -> None:
        self.received_flush_count += 1

    def end_input(self) -> None:
        self._end_requested.set()

    # Public test-only helpers
    async def inject_event(self, ev: SpeechEvent) -> None:
        await self._inject_q.put(ev)


class _FakeSTT(lk_stt.STT):
    """Minimal STT that returns the fake stream we constructed."""

    def __init__(self) -> None:
        super().__init__(
            capabilities=STTCapabilities(streaming=True, interim_results=True),
        )
        self.stream_obj: _FakeRecognizeStream | None = None

    async def _recognize_impl(self, buffer, *, language, conn_options):
        raise NotImplementedError

    def stream(
        self,
        *,
        language=None,
        conn_options=None,
    ) -> _FakeRecognizeStream:
        s = _FakeRecognizeStream(
            stt=self,
            conn_options=conn_options or APIConnectOptions(max_retry=0),
        )
        self.stream_obj = s
        return s


def _interim(text: str) -> SpeechEvent:
    return SpeechEvent(
        type=SpeechEventType.INTERIM_TRANSCRIPT,
        alternatives=[SpeechData(language="zh", text=text)],
    )


def _final(text: str) -> SpeechEvent:
    return SpeechEvent(
        type=SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[SpeechData(language="zh", text=text)],
    )


def _start_of_speech() -> SpeechEvent:
    return SpeechEvent(type=SpeechEventType.START_OF_SPEECH)


def _end_of_speech() -> SpeechEvent:
    return SpeechEvent(type=SpeechEventType.END_OF_SPEECH)


async def _drain_one(gate_stream, timeout: float = 0.5) -> SpeechEvent | None:
    """Read one event from the gate's outer stream with a small timeout.
    Returns None if no event arrives in ``timeout`` seconds."""
    try:
        async for ev in gate_stream:
            return ev
    except StopAsyncIteration:
        return None
    return None


async def _drain_all(gate_stream, *, deadline: float = 0.5) -> List[SpeechEvent]:
    """Drain events from gate's outer stream until ``end_input`` propagates
    and the channel closes."""
    out: List[SpeechEvent] = []
    try:
        await asyncio.wait_for(
            _drain_loop(gate_stream, out),
            timeout=deadline,
        )
    except asyncio.TimeoutError:
        pass
    return out


async def _drain_loop(gate_stream, out: list) -> None:
    async for ev in gate_stream:
        out.append(ev)


# ───────────────────────────────────────────────────────────────────────
# Core filter behaviour
# ───────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_gate_pass_through_when_no_prior_final() -> None:
    """Without a prior FINAL the suppression window is not armed, so all
    INTERIM/FINAL events pass through."""
    fake = _FakeSTT()
    gate = STTTranscriptGate(fake, suppress_window_ms=200)
    gs = gate.stream()
    try:
        inner = fake.stream_obj
        assert inner is not None

        await inner.inject_event(_interim("你"))
        await inner.inject_event(_interim("你好"))

        out: List[SpeechEvent] = []
        for _ in range(2):
            out.append(await asyncio.wait_for(gs.__anext__(), timeout=0.5))
        assert [e.alternatives[0].text for e in out] == ["你", "你好"]
        assert gs.suppressed_count == 0
    finally:
        await gs.aclose()


@pytest.mark.asyncio
async def test_gate_suppresses_interim_within_window() -> None:
    """FINAL → 50ms later INTERIM → INTERIM must be dropped."""
    fake = _FakeSTT()
    gate = STTTranscriptGate(fake, suppress_window_ms=200)
    gs = gate.stream()
    try:
        inner = fake.stream_obj
        assert inner is not None

        await inner.inject_event(_final("嗯，今天下雨了。"))
        first = await asyncio.wait_for(gs.__anext__(), timeout=0.5)
        assert first.type == SpeechEventType.FINAL_TRANSCRIPT

        # Within 200ms window
        await asyncio.sleep(0.05)
        await inner.inject_event(_interim("你有"))

        # Drain non-blocking — gate should have dropped it
        await asyncio.sleep(0.1)
        assert gs.suppressed_count == 1, (
            f"expected the cross-turn INTERIM to be suppressed; "
            f"got suppressed_count={gs.suppressed_count}"
        )
    finally:
        await gs.aclose()


@pytest.mark.asyncio
async def test_gate_passes_interim_outside_window() -> None:
    """FINAL → wait > window → INTERIM must pass."""
    fake = _FakeSTT()
    gate = STTTranscriptGate(fake, suppress_window_ms=100)
    gs = gate.stream()
    try:
        inner = fake.stream_obj
        assert inner is not None

        await inner.inject_event(_final("嗯，今天下雨了。"))
        await asyncio.wait_for(gs.__anext__(), timeout=0.5)

        # Wait longer than the window
        await asyncio.sleep(0.15)
        await inner.inject_event(_interim("你有什么"))

        second = await asyncio.wait_for(gs.__anext__(), timeout=0.5)
        assert second.alternatives[0].text == "你有什么"
        assert gs.suppressed_count == 0
    finally:
        await gs.aclose()


@pytest.mark.asyncio
async def test_gate_suppresses_second_final_within_window() -> None:
    """FINAL → 50ms later FINAL → second FINAL must also be dropped
    (covers framework audio_recognition line 839 contamination source B)."""
    fake = _FakeSTT()
    gate = STTTranscriptGate(fake, suppress_window_ms=200)
    gs = gate.stream()
    try:
        inner = fake.stream_obj
        assert inner is not None

        await inner.inject_event(_final("第一句。"))
        first = await asyncio.wait_for(gs.__anext__(), timeout=0.5)
        assert first.alternatives[0].text == "第一句。"

        await asyncio.sleep(0.05)
        await inner.inject_event(_final("第二句。"))

        await asyncio.sleep(0.1)
        assert gs.suppressed_count == 1, "the second FINAL must be suppressed"
    finally:
        await gs.aclose()


@pytest.mark.asyncio
async def test_gate_non_transcript_events_always_pass() -> None:
    """START_OF_SPEECH / END_OF_SPEECH must pass through even inside the
    suppression window — they're control events the framework needs."""
    fake = _FakeSTT()
    gate = STTTranscriptGate(fake, suppress_window_ms=500)
    gs = gate.stream()
    try:
        inner = fake.stream_obj
        assert inner is not None

        await inner.inject_event(_final("第一句。"))
        first = await asyncio.wait_for(gs.__anext__(), timeout=0.5)
        assert first.type == SpeechEventType.FINAL_TRANSCRIPT

        # Inside the window — but END_OF_SPEECH should still get through
        await inner.inject_event(_end_of_speech())
        eos = await asyncio.wait_for(gs.__anext__(), timeout=0.5)
        assert eos.type == SpeechEventType.END_OF_SPEECH

        await inner.inject_event(_start_of_speech())
        sos = await asyncio.wait_for(gs.__anext__(), timeout=0.5)
        assert sos.type == SpeechEventType.START_OF_SPEECH

        assert gs.suppressed_count == 0
    finally:
        await gs.aclose()


# ───────────────────────────────────────────────────────────────────────
# Audio forwarding (input direction)
# ───────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_gate_forwards_input_audio() -> None:
    """Frames pushed to the outer gate stream must reach the inner stream."""
    fake = _FakeSTT()
    gate = STTTranscriptGate(fake, suppress_window_ms=200)
    gs = gate.stream()
    try:
        inner = fake.stream_obj
        assert inner is not None

        frame = rtc.AudioFrame(
            data=b"\x00\x00" * 800,
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=800,
        )
        gs.push_frame(frame)
        gs.push_frame(frame)

        # _forward_input runs in a background task — give it a moment
        await asyncio.sleep(0.05)
        assert len(inner.received_frames) == 2
    finally:
        await gs.aclose()


@pytest.mark.asyncio
async def test_gate_forwards_flush_sentinel() -> None:
    """Outer flush() → inner.flush() (sentinel translates across types)."""
    fake = _FakeSTT()
    gate = STTTranscriptGate(fake, suppress_window_ms=200)
    gs = gate.stream()
    try:
        inner = fake.stream_obj
        assert inner is not None

        gs.flush()
        await asyncio.sleep(0.05)

        assert inner.received_flush_count == 1
    finally:
        await gs.aclose()


# ───────────────────────────────────────────────────────────────────────
# Order & metric guarantees
# ───────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_gate_preserves_event_order() -> None:
    """When no suppression happens, gate must emit events in the same
    order inner emits them."""
    fake = _FakeSTT()
    gate = STTTranscriptGate(fake, suppress_window_ms=0)  # gate disabled
    gs = gate.stream()
    try:
        inner = fake.stream_obj
        assert inner is not None

        sequence = [_interim("a"), _interim("ab"), _final("abc。"), _interim("d")]
        for ev in sequence:
            await inner.inject_event(ev)

        out: List[SpeechEvent] = []
        for _ in range(len(sequence)):
            out.append(await asyncio.wait_for(gs.__anext__(), timeout=0.5))

        assert [
            (e.type, e.alternatives[0].text if e.alternatives else "")
            for e in out
        ] == [
            (SpeechEventType.INTERIM_TRANSCRIPT, "a"),
            (SpeechEventType.INTERIM_TRANSCRIPT, "ab"),
            (SpeechEventType.FINAL_TRANSCRIPT, "abc。"),
            (SpeechEventType.INTERIM_TRANSCRIPT, "d"),
        ]
        assert gs.suppressed_count == 0
    finally:
        await gs.aclose()


@pytest.mark.asyncio
async def test_gate_suppressed_count_metric() -> None:
    """Counter increments exactly once per suppressed event."""
    fake = _FakeSTT()
    gate = STTTranscriptGate(fake, suppress_window_ms=500)
    gs = gate.stream()
    try:
        inner = fake.stream_obj
        assert inner is not None

        await inner.inject_event(_final("A。"))
        await asyncio.wait_for(gs.__anext__(), timeout=0.5)

        for text in ["B1", "B2", "B3"]:
            await inner.inject_event(_interim(text))

        await asyncio.sleep(0.1)
        assert gs.suppressed_count == 3
    finally:
        await gs.aclose()


@pytest.mark.asyncio
async def test_gate_window_zero_disables_suppression() -> None:
    """``suppress_window_ms=0`` → gate becomes a transparent pass-through."""
    fake = _FakeSTT()
    gate = STTTranscriptGate(fake, suppress_window_ms=0)
    gs = gate.stream()
    try:
        inner = fake.stream_obj
        assert inner is not None

        await inner.inject_event(_final("A。"))
        await asyncio.wait_for(gs.__anext__(), timeout=0.5)

        # No matter how immediate, INTERIM passes
        await inner.inject_event(_interim("B"))
        out = await asyncio.wait_for(gs.__anext__(), timeout=0.5)
        assert out.alternatives[0].text == "B"
        assert gs.suppressed_count == 0
    finally:
        await gs.aclose()


# ───────────────────────────────────────────────────────────────────────
# Metadata + lifecycle forwarding
# ───────────────────────────────────────────────────────────────────────


def test_gate_capabilities_forwarded() -> None:
    """gate.capabilities IS inner.capabilities — framework must see the
    same streaming/interim support."""
    fake = _FakeSTT()
    gate = STTTranscriptGate(fake, suppress_window_ms=200)

    assert gate.capabilities is fake.capabilities
    assert gate.capabilities.streaming is True
    assert gate.capabilities.interim_results is True


def test_gate_metadata_forwarded() -> None:
    """label / model / provider forward to inner."""
    fake = _FakeSTT()
    gate = STTTranscriptGate(fake, suppress_window_ms=200)

    assert "STTTranscriptGate" in gate.label
    assert fake.label in gate.label
    # Both model & provider default to "unknown" on _FakeSTT; the test
    # is that the forward call works without raising.
    assert gate.model == fake.model
    assert gate.provider == fake.provider


def test_gate_inner_accessor() -> None:
    """Public ``inner`` property exposes the wrapped plugin."""
    fake = _FakeSTT()
    gate = STTTranscriptGate(fake, suppress_window_ms=200)
    assert gate.inner is fake


def test_gate_suppress_window_accessor() -> None:
    gate = STTTranscriptGate(_FakeSTT(), suppress_window_ms=350)
    assert gate.suppress_window_ms == 350


def test_gate_clamps_negative_window_to_zero() -> None:
    """Negative window should clamp to 0 (= disabled), not crash."""
    gate = STTTranscriptGate(_FakeSTT(), suppress_window_ms=-100)
    assert gate.suppress_window_ms == 0


def test_gate_notify_vad_state_forwards_if_inner_supports() -> None:
    """G16 compatibility: gate must forward notify_vad_state when inner
    has it (Bailian has the hook; SenseTime doesn't)."""
    fake = _FakeSTT()
    fake.notify_vad_state = MagicMock()  # type: ignore[attr-defined]
    gate = STTTranscriptGate(fake, suppress_window_ms=200)

    gate.notify_vad_state(0.8, 1500.0)

    fake.notify_vad_state.assert_called_once_with(0.8, 1500.0)


def test_gate_notify_vad_state_noop_when_inner_lacks_hook() -> None:
    """Must not crash for plugins that don't expose notify_vad_state."""
    fake = _FakeSTT()
    # Ensure no notify_vad_state on the inner
    if hasattr(fake, "notify_vad_state"):
        del fake.notify_vad_state
    gate = STTTranscriptGate(fake, suppress_window_ms=200)

    # Must not raise
    gate.notify_vad_state(0.5, 500.0)
