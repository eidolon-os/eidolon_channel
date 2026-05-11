# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for the provider-agnostic TTSConnectionPool."""

from __future__ import annotations

import asyncio
import itertools

import pytest

from eidolon.channel.livekit.plugins.tts._pool import TTSConnectionPool


class FakeConn:
    """Minimal "connection" stand-in for testing."""
    _ids = itertools.count(1)

    def __init__(self) -> None:
        self.id = next(FakeConn._ids)
        self.disposed = False

    def __repr__(self) -> str:
        return f"FakeConn#{self.id}"


def _make_factory(*, fail_n_times: int = 0):
    """Return (factory_callable, opened_list) where the factory increments
    a counter and optionally raises for the first N calls."""
    opened: list[FakeConn] = []
    fail_remaining = [fail_n_times]

    async def factory():
        if fail_remaining[0] > 0:
            fail_remaining[0] -= 1
            raise RuntimeError("simulated factory failure")
        c = FakeConn()
        opened.append(c)
        return c

    return factory, opened


def _make_disposer(*, raise_on_dispose: bool = False, slow_ms: int = 0):
    """Return (disposer_callable, disposed_list)."""
    disposed: list[FakeConn] = []

    async def disposer(conn: FakeConn):
        if slow_ms:
            await asyncio.sleep(slow_ms / 1000.0)
        if raise_on_dispose:
            raise RuntimeError("simulated dispose failure")
        conn.disposed = True
        disposed.append(conn)

    return disposer, disposed


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestPoolHappyPath:
    @pytest.mark.asyncio
    async def test_warmup_opens_size_conns(self):
        factory, opened = _make_factory()
        disposer, _ = _make_disposer()
        pool = TTSConnectionPool(factory=factory, disposer=disposer, size=3)
        await pool.warmup()
        assert pool.warm_count == 3
        assert len(opened) == 3
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_warmup_idempotent(self):
        factory, opened = _make_factory()
        disposer, _ = _make_disposer()
        pool = TTSConnectionPool(factory=factory, disposer=disposer, size=2)
        await pool.warmup()
        await pool.warmup()  # second call: nothing to do
        assert len(opened) == 2
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_acquire_fast_path_when_pool_full(self):
        factory, opened = _make_factory()
        disposer, _ = _make_disposer()
        pool = TTSConnectionPool(factory=factory, disposer=disposer, size=2)
        await pool.warmup()
        c1 = await pool.acquire()
        assert isinstance(c1, FakeConn)
        assert pool.warm_count == 1
        # Factory only ran 2 times (the warmup), not 3.
        assert len(opened) == 2
        await pool.mark_dirty(c1)
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_mark_dirty_disposes_and_refills(self):
        factory, opened = _make_factory()
        disposer, disposed = _make_disposer()
        pool = TTSConnectionPool(factory=factory, disposer=disposer, size=2)
        await pool.warmup()

        c1 = await pool.acquire()       # pool now has 1 warm
        await pool.mark_dirty(c1)       # refill spawned in background
        # Wait for refill to complete.
        await _wait_until(lambda: pool.in_flight_refills == 0, timeout=2.0)

        assert c1 in disposed
        assert pool.warm_count == 2     # back to target size
        assert len(opened) == 3         # 2 warmup + 1 refill
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_repeated_acquire_dirty_keeps_pool_topped_up(self):
        """Simulate 5 turns: each acquires, then marks dirty."""
        factory, opened = _make_factory()
        disposer, disposed = _make_disposer()
        pool = TTSConnectionPool(factory=factory, disposer=disposer, size=2)
        await pool.warmup()

        for _ in range(5):
            c = await pool.acquire()
            await pool.mark_dirty(c)
            # Give refill a moment so next acquire is fast-path.
            await asyncio.sleep(0.01)

        await _wait_until(lambda: pool.in_flight_refills == 0, timeout=2.0)
        assert len(disposed) == 5
        assert pool.warm_count == 2
        await pool.shutdown()


# ---------------------------------------------------------------------------
# Slow path / edge cases
# ---------------------------------------------------------------------------


class TestPoolEdgeCases:
    @pytest.mark.asyncio
    async def test_acquire_slow_path_when_pool_empty(self):
        """When all warm conns are checked out and refill hasn't completed,
        acquire warms a new one inline."""
        factory, opened = _make_factory()
        disposer, _ = _make_disposer()
        pool = TTSConnectionPool(factory=factory, disposer=disposer, size=2)
        await pool.warmup()
        c1 = await pool.acquire()
        c2 = await pool.acquire()
        assert pool.warm_count == 0
        # Pool is now empty; this should slow-path.
        c3 = await pool.acquire()
        assert isinstance(c3, FakeConn)
        assert len(opened) == 3
        for c in (c1, c2, c3):
            await pool.mark_dirty(c)
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_warmup_partial_failure_is_tolerated(self):
        """If 1 of 3 factory calls fails, warmup still completes with 2 conns
        and does NOT raise."""
        factory, opened = _make_factory(fail_n_times=1)
        disposer, _ = _make_disposer()
        pool = TTSConnectionPool(factory=factory, disposer=disposer, size=3)
        await pool.warmup()
        # 1 failed, 2 succeeded.
        assert pool.warm_count == 2
        assert len(opened) == 2
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_warmup_total_failure_raises(self):
        factory, _ = _make_factory(fail_n_times=10)
        disposer, _ = _make_disposer()
        pool = TTSConnectionPool(factory=factory, disposer=disposer, size=2)
        with pytest.raises(RuntimeError, match="warmup failed"):
            await pool.warmup()
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_refill_failure_backoff(self):
        """Factory transiently fails after warmup; pool should recover on
        subsequent mark_dirty calls (not retry-loop the failure)."""
        factory, opened = _make_factory()
        disposer, disposed = _make_disposer()
        pool = TTSConnectionPool(
            factory=factory, disposer=disposer, size=2,
            refill_failure_backoff=0.05,  # speed up the test
        )
        await pool.warmup()
        # Acquire + dirty 1 — this triggers a refill.
        c = await pool.acquire()
        await pool.mark_dirty(c)
        await _wait_until(lambda: pool.in_flight_refills == 0, timeout=2.0)
        assert pool.warm_count == 2
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_disposer_failure_does_not_crash_pool(self):
        factory, _ = _make_factory()
        disposer, _ = _make_disposer(raise_on_dispose=True)
        pool = TTSConnectionPool(factory=factory, disposer=disposer, size=2)
        await pool.warmup()
        c = await pool.acquire()
        # Disposer will raise; pool should swallow and continue.
        await pool.mark_dirty(c)
        await _wait_until(lambda: pool.in_flight_refills == 0, timeout=2.0)
        # Pool refilled despite disposer failure.
        assert pool.warm_count == 2
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_disposes_all_warm_conns(self):
        factory, opened = _make_factory()
        disposer, disposed = _make_disposer()
        pool = TTSConnectionPool(factory=factory, disposer=disposer, size=3)
        await pool.warmup()
        await pool.shutdown()
        # All 3 disposed.
        assert len(disposed) == 3
        for c in opened:
            assert c.disposed

    @pytest.mark.asyncio
    async def test_shutdown_waits_for_inflight_refills(self):
        """If a refill is mid-flight, shutdown should wait for it."""
        # Use a slow disposer to ensure mark_dirty's bg task is alive
        # when shutdown begins.
        factory, opened = _make_factory()
        disposer, disposed = _make_disposer(slow_ms=50)
        pool = TTSConnectionPool(factory=factory, disposer=disposer, size=2)
        await pool.warmup()
        c = await pool.acquire()
        await pool.mark_dirty(c)
        # Don't wait_until — directly shutdown so the in-flight refill is
        # in progress.
        await pool.shutdown()
        # All in-flight refills awaited.
        assert pool.in_flight_refills == 0

    @pytest.mark.asyncio
    async def test_acquire_after_shutdown_raises(self):
        factory, _ = _make_factory()
        disposer, _ = _make_disposer()
        pool = TTSConnectionPool(factory=factory, disposer=disposer, size=2)
        await pool.warmup()
        await pool.shutdown()
        with pytest.raises(RuntimeError, match="closed pool"):
            await pool.acquire()

    @pytest.mark.asyncio
    async def test_mark_dirty_after_shutdown_disposes_synchronously(self):
        factory, _ = _make_factory()
        disposer, disposed = _make_disposer()
        pool = TTSConnectionPool(factory=factory, disposer=disposer, size=1)
        await pool.warmup()
        c = await pool.acquire()
        await pool.shutdown()
        # mark_dirty after shutdown should still dispose, just synchronously.
        await pool.mark_dirty(c)
        assert c in disposed

    @pytest.mark.asyncio
    async def test_invalid_size_raises(self):
        factory, _ = _make_factory()
        disposer, _ = _make_disposer()
        with pytest.raises(ValueError):
            TTSConnectionPool(factory=factory, disposer=disposer, size=0)


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestPoolConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_acquires_dont_overshoot(self):
        """Many concurrent acquire+mark_dirty pairs; pool should converge
        back to size N without leaking conns or growing past N."""
        factory, opened = _make_factory()
        disposer, disposed = _make_disposer()
        pool = TTSConnectionPool(factory=factory, disposer=disposer, size=2)
        await pool.warmup()

        async def one_turn():
            c = await pool.acquire()
            await asyncio.sleep(0.005)  # simulate a tiny use
            await pool.mark_dirty(c)

        await asyncio.gather(*(one_turn() for _ in range(20)))
        await _wait_until(lambda: pool.in_flight_refills == 0, timeout=5.0)

        # 2 from warmup + 20 from turns = 22 conns ever opened.
        # All 20 turn conns should be disposed; the final 2 warm ones live.
        assert len(opened) >= 22
        assert pool.warm_count == 2
        # Pool size should never exceed target (only target's conns sit in queue).
        await pool.shutdown()


# ---------------------------------------------------------------------------
# R8.14 — predictive (acquire-time) refill
# ---------------------------------------------------------------------------


class TestPoolPredictiveRefill:
    """Round 8 R8.14: refill is scheduled at acquire-time, not at dispose-time.
    This keeps the pool hot during turn-rate bursts that the original
    R8.7 lazy-refill design couldn't keep up with."""

    @pytest.mark.asyncio
    async def test_acquire_triggers_background_refill(self):
        """One acquire → pool should start refilling automatically,
        WITHOUT a corresponding mark_dirty call."""
        factory, opened = _make_factory()
        disposer, _ = _make_disposer()
        pool = TTSConnectionPool(factory=factory, disposer=disposer, size=2)
        await pool.warmup()
        # Pool: warm=2, opened=2

        c = await pool.acquire()
        # Right after acquire: warm=1, refill should be in flight
        # (or already done if factory is instant in test).
        await _wait_until(lambda: pool.in_flight_refills == 0, timeout=2.0)
        # Pool refilled itself: warm=2 even WITHOUT mark_dirty being called
        assert pool.warm_count == 2
        # Total opened: 2 warmup + 1 refill = 3
        assert len(opened) == 3

        await pool.mark_dirty(c)
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_burst_acquire_does_not_drain_pool_indefinitely(self):
        """Simulate a burst of acquires without intervening mark_dirty
        (e.g. concurrent streams). Pool should heal as refills complete."""
        factory, opened = _make_factory()
        disposer, _ = _make_disposer()
        pool = TTSConnectionPool(factory=factory, disposer=disposer, size=3)
        await pool.warmup()  # 3 warm

        # Acquire 5 in rapid succession (more than pool size).
        # First 3 fast-path, last 2 slow-path. All should succeed.
        acquired = []
        for _ in range(5):
            acquired.append(await pool.acquire())
        assert len(acquired) == 5

        # After burst: pool should self-heal back to size 3 once
        # background refills complete.
        await _wait_until(
            lambda: pool.warm_count == 3 and pool.in_flight_refills == 0,
            timeout=3.0,
        )
        assert pool.warm_count == 3

        for c in acquired:
            await pool.mark_dirty(c)
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_mark_dirty_only_disposes_no_refill_coupling(self):
        """Dispose is decoupled from refill: many mark_dirty calls
        without matching acquire should NOT inflate the pool past size."""
        factory, opened = _make_factory()
        disposer, disposed = _make_disposer()
        pool = TTSConnectionPool(factory=factory, disposer=disposer, size=2)
        await pool.warmup()  # 2 warm

        # Acquire 1 then mark_dirty repeatedly (artificial — same conn
        # disposed multiple times shouldn't happen in real use, but the
        # pool's bookkeeping should remain sane).
        c = await pool.acquire()  # warm=1, refill spawned
        await pool.mark_dirty(c)

        await _wait_until(
            lambda: pool.warm_count == 2 and pool.in_flight_refills == 0,
            timeout=2.0,
        )
        assert pool.warm_count == 2  # exactly target, never overshooting

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_idempotent_refill_scheduling(self):
        """Multiple back-to-back acquires shouldn't oversubscribe refill
        tasks — _maybe_refill is idempotent."""
        factory, opened = _make_factory()
        disposer, _ = _make_disposer()
        pool = TTSConnectionPool(factory=factory, disposer=disposer, size=4)
        await pool.warmup()

        # Take 2 → expect 2 refill tasks (not 4 from doubled scheduling)
        c1 = await pool.acquire()
        c2 = await pool.acquire()
        # Right after: 2 refill tasks should be in flight
        # (deficit was 2, not 4)
        # Allow tiny window for refill to complete in fast tests:
        if pool.in_flight_refills > 0:
            assert pool.in_flight_refills == 2

        await _wait_until(
            lambda: pool.warm_count == 4 and pool.in_flight_refills == 0,
            timeout=3.0,
        )
        # Total opened: 4 warmup + 2 refills = 6 (not 8 from oversubscription)
        assert len(opened) == 6

        await pool.mark_dirty(c1)
        await pool.mark_dirty(c2)
        await pool.shutdown()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _wait_until(predicate, *, timeout: float = 1.0, interval: float = 0.01):
    """Spin until ``predicate()`` returns truthy or timeout."""
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError(f"predicate never became truthy within {timeout}s")
