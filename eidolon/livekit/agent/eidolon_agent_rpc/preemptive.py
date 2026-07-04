"""Preemptive (speculative) brain warm-up on a stabilizing partial transcript.

While the user is still speaking, fire one *speculative* turn at the brain with
the current partial transcript to warm the path (connection + prompt compile +
KV-cache) so the real turn's first response lands sooner. Speculative turns are
ephemeral server-side (no history / memory / persona side effects — see the
agent StartTurn.speculative contract), so an unused guess is harmless.

The real turn (LiveKit chat() on the final transcript) supersedes the warm-up:
the caller calls :meth:`discard` at real-turn start, cancelling the in-flight
speculative turn.

Built-in gating keeps this from spamming the brain on every interim keystroke:
a minimum length, dedup on identical text, and single-in-flight (a new warm
cancels the previous). It never raises into the caller — warming is best effort.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

logger = logging.getLogger("agent")


class PreemptiveWarmer:
    """Owns at most one in-flight speculative turn against a brain session."""

    def __init__(
        self,
        session: Any,
        *,
        spawn: Callable[..., Any],
        min_chars: int = 6,
    ) -> None:
        # ``session`` is an EidolonAgentSession; ``spawn`` schedules a background
        # task the session keeps a strong ref to (session.spawn), used to drain
        # the speculative stream without blocking the caller.
        self._session = session
        self._spawn = spawn
        self._min_chars = min_chars
        self._active_turn_id: str | None = None
        self._active_text: str = ""

    async def warm(
        self,
        text: str,
        *,
        conversation_id: str,
        trace_id: str | None = None,
    ) -> None:
        """Speculatively start a turn on a partial transcript (best effort)."""
        text = (text or "").strip()
        if len(text) < self._min_chars or text == self._active_text:
            return
        await self.discard()
        try:
            turn_id, payloads = await self._session.start_turn(
                text=text,
                conversation_id=conversation_id,
                trace_id=trace_id,
                speculative=True,
            )
        except Exception as exc:  # noqa: BLE001 — warming must never break the turn
            logger.debug("[PreemptiveWarmer] warm(%r) skipped: %r", text[:40], exc)
            return
        self._active_turn_id = turn_id
        self._active_text = text
        self._spawn(self._drain(payloads), name=f"speculative-drain-{turn_id}")

    async def discard(self) -> None:
        """Cancel the in-flight speculative turn, if any (real turn supersedes)."""
        turn_id = self._active_turn_id
        self._active_turn_id = None
        self._active_text = ""
        if turn_id is not None:
            try:
                await self._session.cancel_turn(turn_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("[PreemptiveWarmer] discard(%s) ignored: %r", turn_id, exc)

    @staticmethod
    async def _drain(payloads: AsyncIterator[Any]) -> None:
        # Consume + discard the speculative stream; its only value is warming
        # the brain path. Output is never surfaced to the user.
        try:
            async for _ in payloads:
                pass
        except Exception:  # noqa: BLE001 — a cancelled/failed speculative stream is expected
            pass
