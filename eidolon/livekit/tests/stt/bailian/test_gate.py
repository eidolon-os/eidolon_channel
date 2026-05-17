"""G16 (2026-05-17): SttGate state machine tests.

Coverage matrix from docs/plans/2026-05-17/g10-g16-reliability-and-cost.md:

  ✓ test_gate_idle_no_forward            — GATED never sends real audio
  ✓ test_gate_keepalive_1hz              — keepalive fires at config rate
  ✓ test_gate_preroll_flush              — start_of_speech flushes ring
  ✓ test_gate_tail_window                — end_of_speech + tail → GATED
  ✓ test_gate_tail_window_cancel         — start within tail cancels timer
  ✓ test_gate_energy_fallback            — VAD silent + high RMS → forward
  ✓ test_gate_hysteresis_no_thrash       — grey-zone probability is stable
  ✓ test_gate_long_speech_buffer_drops_old  — ring respects maxlen
  ✓ test_gate_finalizes_metrics          — get_metrics() reports state
"""

from __future__ import annotations

import asyncio
from typing import List

import numpy as np
import pytest

from eidolon.livekit.plugins.stt.bailian._gate import SttGate, _rms_int16


def _silence(samples: int) -> bytes:
    return b"\x00\x00" * samples


def _tone(samples: int, amplitude: int = 8000) -> bytes:
    """Synthesize a loud tone for RMS testing."""
    arr = (np.ones(samples, dtype=np.int16) * amplitude)
    return arr.tobytes()


@pytest.fixture
def sent() -> List[bytes]:
    """Collector for chunks the gate forwards."""
    return []


@pytest.fixture
async def gate_factory(sent: List[bytes]):
    """Build gate variants conveniently."""
    gates: List[SttGate] = []

    async def _make(**kwargs) -> SttGate:
        defaults = dict(
            sample_rate=16000,
            sender=lambda b: _async_append(sent, b),
            preroll_ms=200,
            tail_window_ms=300,
            keepalive_interval_sec=0.1,
            keepalive_frame_ms=20,
            chunk_ms=100,
            vad_high_threshold=0.6,
            vad_low_threshold=0.3,
            rms_threshold=1000.0,
        )
        defaults.update(kwargs)
        g = SttGate(**defaults)
        await g.start()
        gates.append(g)
        return g

    yield _make
    for g in gates:
        await g.stop()


async def _async_append(sent: List[bytes], b: bytes) -> None:
    sent.append(b)


# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_idle_no_forward(gate_factory, sent: List[bytes]) -> None:
    g = await gate_factory()
    chunk = _silence(1600)  # 100ms silent chunk
    for _ in range(3):
        await g.feed(chunk)
    assert g.state == "GATED"
    # No real audio forwarded. (Keepalive task may have fired silence — that
    # IS forwarded, so check via the explicit "real audio" counter.)
    assert g._total_forwarded_bytes == 0


@pytest.mark.asyncio
async def test_gate_keepalive_1hz(gate_factory, sent: List[bytes]) -> None:
    g = await gate_factory(
        keepalive_interval_sec=0.05, keepalive_frame_ms=20
    )
    # Wait a few keepalive cycles
    await asyncio.sleep(0.2)
    assert g._total_keepalive_sent >= 3, (
        f"expected ≥3 keepalive sends in 200ms at 50ms interval; "
        f"got {g._total_keepalive_sent}"
    )


@pytest.mark.asyncio
async def test_gate_preroll_flush(gate_factory, sent: List[bytes]) -> None:
    g = await gate_factory(preroll_ms=200)  # ring capacity = 2 chunks
    # Push 2 chunks while GATED — should fill preroll ring
    chunk_a = _silence(1600)
    chunk_b = _tone(1600, 1)  # use distinguishable content
    await g.feed(chunk_a)
    await g.feed(chunk_b)
    assert g.state == "GATED"
    # Now VAD triggers
    g.notify_vad_state(0.9, 0.0)
    # Feed a new chunk — gate should transition and flush preroll first
    chunk_c = _tone(1600, 2)
    await g.feed(chunk_c)
    assert g.state == "FORWARDING"
    # The keepalive task may have sent silence before this — filter to
    # find the preroll+current chunks among `sent`.
    # Identify by content: chunk_a / chunk_b / chunk_c are all in `sent`.
    assert chunk_a in sent
    assert chunk_b in sent
    assert chunk_c in sent
    # Order should be: preroll (a, b), then current (c)
    a_idx = sent.index(chunk_a)
    b_idx = sent.index(chunk_b)
    c_idx = sent.index(chunk_c)
    assert a_idx < b_idx < c_idx


@pytest.mark.asyncio
async def test_gate_tail_window(gate_factory, sent: List[bytes]) -> None:
    g = await gate_factory(tail_window_ms=300)
    g.notify_vad_state(0.9, 0.0)
    await g.feed(_silence(1600))
    assert g.state == "FORWARDING"
    # VAD goes silent
    g.notify_vad_state(0.05, 0.0)
    # Within tail window: stays FORWARDING
    await g.feed(_silence(1600))
    assert g.state == "FORWARDING"
    # Wait past tail window
    await asyncio.sleep(0.4)
    # New feed evaluates transition
    g.notify_vad_state(0.05, 0.0)
    await g.feed(_silence(1600))
    assert g.state == "GATED"


@pytest.mark.asyncio
async def test_gate_tail_window_cancel(gate_factory, sent: List[bytes]) -> None:
    g = await gate_factory(tail_window_ms=300)
    # Enter forwarding
    g.notify_vad_state(0.9, 0.0)
    await g.feed(_silence(1600))
    # Drop to low
    g.notify_vad_state(0.05, 0.0)
    await asyncio.sleep(0.1)  # less than tail window
    # Speak again before tail expires
    g.notify_vad_state(0.9, 0.0)
    await g.feed(_silence(1600))
    # Should stay FORWARDING (tail cancelled by new high-VAD)
    assert g.state == "FORWARDING"
    # Now even after waiting past original tail window, still FORWARDING
    # (because last_above_low_time was just updated)
    await asyncio.sleep(0.2)
    g.notify_vad_state(0.9, 0.0)
    await g.feed(_silence(1600))
    assert g.state == "FORWARDING"


@pytest.mark.asyncio
async def test_gate_energy_fallback(gate_factory, sent: List[bytes]) -> None:
    """VAD says silent but audio is loud — should still forward
    (energy-fallback gate path)."""
    g = await gate_factory(rms_threshold=2000.0)
    # VAD low but a LOUD chunk comes in
    g.notify_vad_state(0.05, 0.0)  # low VAD
    loud_chunk = _tone(1600, amplitude=20000)  # very loud
    # Sanity check the RMS
    assert _rms_int16(loud_chunk) > 2000.0
    await g.feed(loud_chunk)
    # Gate's internal _latest_rms picked up the loud RMS → energy fallback
    # should transition to FORWARDING
    assert g.state == "FORWARDING", (
        f"expected FORWARDING on energy fallback; got {g.state}, "
        f"latest_rms={g._latest_rms}"
    )


@pytest.mark.asyncio
async def test_gate_hysteresis_no_thrash(gate_factory, sent: List[bytes]) -> None:
    """VAD probability bouncing in [vad_low, vad_high) should NOT cause
    repeated state transitions."""
    g = await gate_factory(
        vad_high_threshold=0.6, vad_low_threshold=0.3, tail_window_ms=200
    )
    transitions_before = g._state_transitions
    # Bounce between 0.35 and 0.55 several times — all in the grey zone
    for prob in [0.35, 0.55, 0.40, 0.50, 0.35, 0.55]:
        g.notify_vad_state(prob, 0.0)
        await g.feed(_silence(1600))
    # Should still be in initial GATED state (never crossed high threshold)
    assert g.state == "GATED"
    assert g._state_transitions == transitions_before


@pytest.mark.asyncio
async def test_gate_long_speech_buffer_drops_old(gate_factory, sent: List[bytes]) -> None:
    """Ring buffer with preroll_ms=200, chunk_ms=100 → capacity 2.
    Pushing 5 chunks should leave only the LAST 2 in ring."""
    g = await gate_factory(preroll_ms=200)
    chunks = [_tone(1600, i + 1) for i in range(5)]
    for c in chunks:
        await g.feed(c)
    # Ring should hold only the last 2 chunks (chunks[3], chunks[4])
    assert len(g._preroll_ring) == 2
    ring_contents = list(g._preroll_ring)
    assert ring_contents[0] == chunks[3]
    assert ring_contents[1] == chunks[4]


@pytest.mark.asyncio
async def test_gate_finalizes_metrics(gate_factory, sent: List[bytes]) -> None:
    g = await gate_factory()
    await g.feed(_silence(1600))
    g.notify_vad_state(0.9, 0.0)
    await g.feed(_silence(1600))
    metrics = g.get_metrics()
    assert metrics["state"] in ("GATED", "FORWARDING")
    assert metrics["state_transitions"] >= 1
    assert "forwarded_bytes" in metrics
    assert "preroll_flushed_bytes" in metrics


@pytest.mark.asyncio
async def test_gate_input_validation() -> None:
    """Constructor rejects invalid configs."""
    async def noop(_): pass

    with pytest.raises(ValueError):
        SttGate(sample_rate=16000, sender=noop, preroll_ms=0)
    with pytest.raises(ValueError):
        SttGate(
            sample_rate=16000,
            sender=noop,
            vad_low_threshold=0.7,
            vad_high_threshold=0.5,
        )
    with pytest.raises(ValueError):
        SttGate(sample_rate=16000, sender=noop, chunk_ms=0)
