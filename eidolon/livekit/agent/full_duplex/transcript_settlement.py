"""Bounded transcript settlement for LiveKit framework-completed turns."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from ..session.user_turn_coordinator import (
    FrameworkCompletionReadiness,
    UserTurnCoordinator,
)

TranscriptSettlementOutcome = Literal["ready", "deadline", "candidate_replaced"]


@dataclass(frozen=True)
class TranscriptSettlementResult:
    outcome: TranscriptSettlementOutcome
    readiness: FrameworkCompletionReadiness
    elapsed_sec: float
    evidence_updates: int


class TranscriptSettlementLease:
    """Wait for evidence changes up to one hard product deadline.

    The lease is driven by coordinator notifications, not polling and not a
    hoped-for second LiveKit completion callback.
    """

    def __init__(
        self,
        coordinator: UserTurnCoordinator,
        *,
        timeout_sec: float,
    ) -> None:
        self._coordinator = coordinator
        self._timeout_sec = max(0.0, float(timeout_sec))

    async def settle(
        self,
        *,
        candidate_id: str | None,
        framework_transcript: str,
    ) -> TranscriptSettlementResult:
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        deadline = started_at + self._timeout_sec
        updates = 0
        while True:
            # Capture the notification version before evaluating readiness.
            # If evidence lands during the evaluation, wait_for_transcript_change
            # observes the version mismatch immediately instead of sleeping until
            # the deadline for a change that already happened.
            version = self._coordinator.transcript_change_version
            active = self._coordinator.active
            if candidate_id is not None and (
                active is None or active.candidate_id != candidate_id
            ):
                return TranscriptSettlementResult(
                    outcome="candidate_replaced",
                    readiness=FrameworkCompletionReadiness(
                        ready=False,
                        reason="candidate_replaced_during_settlement",
                    ),
                    elapsed_sec=loop.time() - started_at,
                    evidence_updates=updates,
                )

            readiness = self._coordinator.framework_completion_readiness(
                framework_transcript
            )
            if readiness.ready:
                return TranscriptSettlementResult(
                    outcome="ready",
                    readiness=readiness,
                    elapsed_sec=loop.time() - started_at,
                    evidence_updates=updates,
                )

            remaining = deadline - loop.time()
            if remaining <= 0:
                return TranscriptSettlementResult(
                    outcome="deadline",
                    readiness=readiness,
                    elapsed_sec=loop.time() - started_at,
                    evidence_updates=updates,
                )
            changed = await self._coordinator.wait_for_transcript_change(
                after_version=version,
                timeout_sec=remaining,
            )
            if not changed:
                return TranscriptSettlementResult(
                    outcome="deadline",
                    readiness=readiness,
                    elapsed_sec=loop.time() - started_at,
                    evidence_updates=updates,
                )
            updates += 1
