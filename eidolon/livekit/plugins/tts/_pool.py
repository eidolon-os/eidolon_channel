# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Provider-agnostic WebSocket connection pool for streaming TTS plugins.

============================================================================
ARCHITECTURE DECISION (ADR — why this isn't using livekit-agents primitives)
============================================================================

The framework offers ``tts.StreamAdapter`` for adapting non-streaming TTS
plugins to streaming. It does NOT offer connection pooling. Why we need
our own:

1. **SenseAudio TTS handshake is heavy** (~1.5-2s per fresh conn:
   WS handshake + ``task_start`` + server-side GPU slot allocation).
   Naive single-connection-per-reply would block every cancel for 2s
   while a new connection is set up.

2. **StreamAdapter doesn't help** — it wraps a one-shot ``synthesize()``
   plugin into a streaming one. SenseTime TTS is ALREADY streaming
   (token-level), so StreamAdapter would actually break batching by
   buffering whole sentences for one-shot TTS.

3. **No public framework API for pre-warming** — ``TTS.prewarm()``
   exists but manages a single instance. Cancel-and-reconnect cycles
   need multiple pre-warmed sockets to avoid "every cancel pays full
   factory cost" (Round 8 R8.7 spike: pool 0ms vs reconnect 2s).

If livekit-agents ever adds connection pooling primitives, this module
should be re-evaluated; until then it's the cleanest extension point.

============================================================================
DESIGN
============================================================================

Designed for "use once, discard" semantics:
  - Each turn (one TTS reply) acquires a warm connection from the pool.
  - When the turn completes (normally OR via cancel), the connection is
    marked dirty and discarded.
  - The pool spawns a background replacement so a warm connection is
    ready for the next turn.

Why "discard every turn" rather than "reuse"?

Because the only reliable way to guarantee no cross-turn audio leakage on
cancel is to physically separate the WebSockets. Trying to reuse a conn
after cancel risks server-side residual state (queued ``task_continue``
batches still being synthesized) bleeding into the next turn — a race
that ``Round 8 R8.2 SentenceAggregator`` exposed in production.

Spike measurement (``scripts/spike_tts_pool_vs_reconnect.py``):
  Single-conn + reconnect-on-cancel : ~3500ms cancel→first new audio
  Pool (size=2) + swap-on-cancel    : ~1335ms cancel→first new audio
  → 62% reduction; the entire connection setup cost (~2s on SenseAudio)
    is eliminated by pre-warming.

The pool is **provider-agnostic** — provider-specific warm-up logic
(e.g. ``task_start`` + waiting for ``task_started`` ack on SenseAudio)
lives in the ``factory`` callable. Provider-specific cleanup logic
(e.g. ``disconnect``) lives in the ``disposer`` callable. The pool
itself only manages: acquire, mark_dirty, background refill, shutdown.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Generic, Optional, TypeVar

logger = logging.getLogger("plugins.tts.pool")

T = TypeVar("T")


class TTSConnectionPool(Generic[T]):
    """Maintain ``size`` pre-warmed connections; one-shot acquire-and-discard.

    Lifecycle::

        pool = TTSConnectionPool(
            factory=my_warm_conn_factory,    # opens + warms one conn
            disposer=my_close_conn,          # cleanly closes one conn
            size=2,
        )
        await pool.warmup()                  # open ``size`` conns in parallel

        # Per-turn:
        conn = await pool.acquire()          # 0 ms if pool has a warm one
        try:
            ... use conn for one turn ...
        finally:
            await pool.mark_dirty(conn)      # discard + trigger background refill

        # On shutdown:
        await pool.shutdown()                # close all warm conns

    Concurrency safety:
      - ``acquire`` / ``mark_dirty`` / ``shutdown`` are all thread-safe at
        the asyncio task level (no shared mutable state outside the queue
        and a refill semaphore).
      - Concurrent ``mark_dirty`` calls each spawn their own background
        refill task; each refill checks pool size before inserting to
        avoid overshooting the target.
      - If a refill races against a concurrent ``acquire`` that triggered
        an inline warmup (pool was empty), the refill may produce a
        surplus conn — it's disposed instead of inserted.
    """

    def __init__(
        self,
        *,
        factory: Callable[[], Awaitable[T]],
        disposer: Callable[[T], Awaitable[None]],
        size: int = 2,
        label: str = "TTSConnectionPool",
        refill_failure_backoff: float = 1.0,
        acquire_wait_timeout: float | None = None,
        enable_inline_slow_path: bool = True,
    ) -> None:
        if size < 1:
            raise ValueError(f"pool size must be ≥ 1, got {size}")
        self._factory = factory
        self._disposer = disposer
        self._size = size
        self._label = label
        self._refill_failure_backoff = refill_failure_backoff
        self._acquire_wait_timeout = acquire_wait_timeout
        self._enable_inline_slow_path = enable_inline_slow_path

        # The "ready" queue holds warm conns. We cap it generously to avoid
        # spurious blocking on transient overshoot (race between concurrent
        # mark_dirty calls and inline acquire-warm); the size invariant is
        # maintained by checking ``qsize()`` before inserting.
        self._ready: asyncio.Queue[T] = asyncio.Queue(maxsize=size + 4)

        self._closed: bool = False
        # Track in-flight refill tasks so shutdown can wait for them.
        self._refill_tasks: set[asyncio.Task[None]] = set()

    # ── Lifecycle ───────────────────────────────────────────────

    async def warmup(self) -> None:
        """Open ``size`` connections in parallel. Idempotent.

        Raises only if ALL ``size`` factories fail. Partial failures are
        logged and the pool starts with whatever connections succeeded;
        ``acquire`` will warm more inline as needed.
        """
        if self._closed:
            raise RuntimeError(f"[{self._label}] cannot warmup a closed pool")

        already_ready = self._ready.qsize()
        needed = max(0, self._size - already_ready)
        if needed == 0:
            return

        logger.info(
            "[%s] warmup: opening %d connection(s) in parallel "
            "(target size=%d, already=%d)",
            self._label, needed, self._size, already_ready,
        )

        results = await asyncio.gather(
            *(self._factory() for _ in range(needed)),
            return_exceptions=True,
        )

        succeeded = 0
        for r in results:
            if isinstance(r, BaseException):
                logger.warning("[%s] warmup: one factory call failed: %s",
                               self._label, r)
                continue
            try:
                self._ready.put_nowait(r)
                succeeded += 1
            except asyncio.QueueFull:
                # Shouldn't happen because we sized the queue with headroom,
                # but be safe: dispose the surplus.
                await self._safe_dispose(r)

        if succeeded == 0:
            raise RuntimeError(
                f"[{self._label}] warmup failed: 0/{needed} connections opened"
            )

        logger.info(
            "[%s] warmup complete: %d/%d connections ready",
            self._label, succeeded, needed,
        )

    async def shutdown(self) -> None:
        """Close all warm connections + wait for in-flight refills to finish."""
        if self._closed:
            return
        self._closed = True
        logger.info(
            "[%s] shutdown: %d warm conns + %d in-flight refills",
            self._label, self._ready.qsize(), len(self._refill_tasks),
        )

        # Wait for any in-flight refill tasks to complete (or fail) so we
        # don't leak background work or fight with them on conn close.
        if self._refill_tasks:
            await asyncio.gather(*self._refill_tasks, return_exceptions=True)
            self._refill_tasks.clear()

        # Drain and dispose remaining warm conns.
        while not self._ready.empty():
            try:
                conn = self._ready.get_nowait()
            except asyncio.QueueEmpty:
                break
            await self._safe_dispose(conn)

    # ── Per-turn API ────────────────────────────────────────────

    async def acquire(self) -> T:
        """Return a warm connection. Fast path: synchronous queue pop (≤ 1 ms).
        Slow path: pool empty → warm one inline (factory cost).

        Round 8 R8.14: ``_maybe_refill`` is called at acquire time so
        the next acquire stays fast. The original R8.7 design only
        triggered refill at ``mark_dirty`` time — equivalent to
        "lazily refill after dispose, ~2 s real-time gap". In production
        with frequent turns + pre-R8.12.b cancel-retry chaos, every
        acquire fell to slow path because refill hadn't completed.
        Acquire-time eager refill makes the pool actively self-healing.

        Raises if the pool is closed.
        """
        if self._closed:
            raise RuntimeError(f"[{self._label}] acquire on a closed pool")

        try:
            conn = self._ready.get_nowait()
            logger.debug(
                "[%s] acquire: fast path (%d remaining in pool, "
                "%d refills in flight)",
                self._label, self._ready.qsize(), self.in_flight_refills,
            )
            self._maybe_refill()
            return conn
        except asyncio.QueueEmpty:
            pass

        # Hard-cap mode: don't create inline overflow connections.
        # Wait for a refilled warm connection within timeout.
        if not self._enable_inline_slow_path:
            timeout = self._acquire_wait_timeout
            try:
                if timeout is None:
                    conn = await self._ready.get()
                else:
                    conn = await asyncio.wait_for(self._ready.get(), timeout=timeout)
                logger.debug(
                    "[%s] acquire: waited for refill (pool=%d in_flight=%d)",
                    self._label, self._ready.qsize(), self.in_flight_refills,
                )
                self._maybe_refill()
                return conn
            except asyncio.TimeoutError as e:
                raise RuntimeError(
                    f"[{self._label}] acquire timeout waiting for warm connection "
                    f"(timeout={timeout}s, in_flight={self.in_flight_refills})"
                ) from e

        # Legacy mode: pool is empty -> warm one inline.
        logger.warning(
            "[%s] acquire: SLOW PATH — pool empty (refills_in_flight=%d), "
            "warming inline. Subsequent acquires will recover via "
            "background refill. Consider raising pool size if this "
            "happens often.",
            self._label, self.in_flight_refills,
        )
        try:
            conn = await self._factory()
        finally:
            self._maybe_refill()
        return conn

    async def mark_dirty(self, conn: T) -> None:
        """Discard ``conn``; refill is handled by ``acquire``-time scheduling.

        Round 8 R8.14: split from refill — under the new design, refill
        is scheduled at acquire-time (proactively), not at dispose-time
        (reactively). ``mark_dirty`` only handles the dispose half.

        Returns immediately; dispose runs asynchronously so the caller
        (a TTS stream) can move on without waiting.
        """
        if self._closed:
            # Pool already shutting down; just dispose synchronously.
            await self._safe_dispose(conn)
            return

        # Belt + suspenders: also run _maybe_refill here. Acquire-time
        # already handles the common case, but if the caller does many
        # ``mark_dirty`` without ``acquire`` interleaved (unusual but
        # possible), we still want the pool to recover.
        task = asyncio.create_task(self._safe_dispose(conn))
        self._refill_tasks.add(task)
        task.add_done_callback(self._refill_tasks.discard)
        self._maybe_refill()

    def _maybe_refill(self) -> None:
        """Idempotent refill scheduler — spawn refill tasks until
        ``qsize + in_flight_refills ≥ target``.

        Safe to call from any path (acquire / mark_dirty / monitor).
        Multiple invocations don't oversubscribe because each call
        only spawns the deficit between current state and target.

        ``in_flight_refills`` only counts refill tasks (not dispose
        tasks); dispose tasks are also tracked in ``_refill_tasks``
        for shutdown bookkeeping but are excluded from the deficit
        calculation via the ``_is_refill`` marker.
        """
        if self._closed:
            return
        target = self._size
        in_flight = sum(
            1 for t in self._refill_tasks
            if not t.done() and getattr(t, "_is_refill", False)
        )
        current = self._ready.qsize() + in_flight
        deficit = target - current
        if deficit <= 0:
            return
        for _ in range(deficit):
            task = asyncio.create_task(self._opportunistic_refill())
            task._is_refill = True  # type: ignore[attr-defined]
            self._refill_tasks.add(task)
            task.add_done_callback(self._refill_tasks.discard)
        if deficit > 0:
            logger.debug(
                "[%s] refill scheduled %d task(s) (pool=%d target=%d "
                "in_flight=%d)",
                self._label, deficit, self._ready.qsize(), target,
                in_flight + deficit,
            )

    @property
    def warm_count(self) -> int:
        """Number of pre-warmed connections currently sitting in the pool."""
        return self._ready.qsize()

    @property
    def in_flight_refills(self) -> int:
        """Number of background refill tasks currently running."""
        return len(self._refill_tasks)

    # ── Internals ───────────────────────────────────────────────

    async def _opportunistic_refill(self) -> None:
        """Background refill task: open one new conn and add it to pool.

        Round 8 R8.14: replaces the old ``_dispose_and_refill`` —
        dispose is now decoupled (handled directly by ``mark_dirty``);
        this task is purely "create + insert into pool".

        Called by ``_maybe_refill`` which ensures we don't oversubscribe.
        Race protection: re-checks ``qsize >= target`` before insertion
        (a concurrent refill may have completed first), and disposes the
        surplus conn cleanly if so.
        """
        if self._closed:
            return

        try:
            new_conn = await self._factory()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(
                "[%s] refill: factory failed: %s; pool size now %d/%d",
                self._label, e, self._ready.qsize(), self._size,
            )
            # Don't immediately retry on failure — back off briefly so
            # the next acquire can re-trigger refill via _maybe_refill.
            # A tight retry loop here would amplify network outages.
            await asyncio.sleep(self._refill_failure_backoff)
            return

        if self._closed:
            await self._safe_dispose(new_conn)
            return

        # Re-check size after the (possibly slow) factory call: another
        # concurrent refill may have completed first.
        if self._ready.qsize() >= self._size:
            logger.debug(
                "[%s] refill: pool already full (%d), disposing surplus",
                self._label, self._ready.qsize(),
            )
            await self._safe_dispose(new_conn)
            return

        try:
            self._ready.put_nowait(new_conn)
            logger.debug(
                "[%s] refill: added 1 conn (pool now %d/%d)",
                self._label, self._ready.qsize(), self._size,
            )
        except asyncio.QueueFull:
            # Defensive — queue has headroom so this shouldn't fire.
            await self._safe_dispose(new_conn)

    async def _safe_dispose(self, conn: T) -> None:
        """Call ``self._disposer(conn)``, swallow any exception."""
        try:
            await self._disposer(conn)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("[%s] disposer raised (ignoring): %s",
                           self._label, e)
