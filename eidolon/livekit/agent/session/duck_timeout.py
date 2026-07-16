"""Duck suspend-window timeout handling."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from eidolon.livekit.agent.output import DuckingStats
from eidolon.livekit.agent.turn_policy import (
    Action,
    Decision,
    InterruptIntent,
    TurnPolicyRuntime,
)

logger = logging.getLogger("agent.session.duck_timeout")


class DuckSuspendTimeoutHandler:
    """Resolve the duck suspend-window deadline through turn policy.

    VAD start immediately ducks output. This handler owns the later deadline:
    by then we may have STT text, an EOT score, or only VAD. The policy runtime
    remains the authority for the decision; this class only handles deadline
    bookkeeping, max-suspend rollback and applying the resolved decision.
    """

    def __init__(
        self,
        *,
        turn_runtime: TurnPolicyRuntime,
        sleep: Callable[[float], Awaitable[None]],
        create_task: Callable[[Awaitable[None]], asyncio.Task],
        get_duck_suspended: Callable[[], bool],
        get_duck_stats: Callable[[], DuckingStats],
        get_suspend_start: Callable[[], float],
        set_timeout_task: Callable[[asyncio.Task], None],
        get_latest_asr_text: Callable[[], str],
        get_vad_active: Callable[[], bool],
        get_eot_model: Callable[[], Any],
        apply_decision: Callable[..., None],
        should_hold_for_evidence: Callable[[], bool] | None = None,
        get_max_suspend_sec: Callable[[], float] | None = None,
        deadline_decision: Callable[..., Decision] | None = None,
    ) -> None:
        self._turn_runtime = turn_runtime
        self._sleep = sleep
        self._create_task = create_task
        self._get_duck_suspended = get_duck_suspended
        self._get_duck_stats = get_duck_stats
        self._get_suspend_start = get_suspend_start
        self._set_timeout_task = set_timeout_task
        self._get_latest_asr_text = get_latest_asr_text
        self._get_vad_active = get_vad_active
        self._get_eot_model = get_eot_model
        self._apply_decision = apply_decision
        self._should_hold_for_evidence = should_hold_for_evidence or (lambda: False)
        self._get_max_suspend_sec = get_max_suspend_sec or (lambda: 0.0)
        self._deadline_decision = deadline_decision or turn_runtime.deadline_decision

    async def run(self, timeout_sec: float) -> None:
        """Wait for the decision budget, then resolve or re-arm ducking."""
        try:
            await self._sleep(timeout_sec)
            if not self._get_duck_suspended():
                return

            stats = self._get_duck_stats()
            vad_still_active = self._get_vad_active()
            latest_asr_text = self._get_latest_asr_text().strip()
            eot_model = self._get_eot_model()
            decision = self._deadline_decision(
                vad_still_active,
                has_transcript=bool(latest_asr_text),
                transcript=latest_asr_text,
                eot_score=eot_model.current_eot_score,
            )
            if decision.action is Action.ROLLBACK and self._should_hold_for_evidence():
                decision = Decision(
                    action=Action.HOLD,
                    reason=(
                        "deadline_wait_for_active_speech_evidence"
                        if vad_still_active
                        else "deadline_wait_for_post_speech_evidence"
                    ),
                    intent=decision.intent,
                    intent_source=decision.intent_source or "timeout",
                    intent_confidence=decision.intent_confidence,
                )
            if decision.action is Action.HOLD:
                decision = self._resolve_hold_or_rearm(
                    decision,
                    timeout_sec=timeout_sec,
                    eot_model=eot_model,
                )
            logger.info(
                "[DuckSuspendTimeoutHandler] duck deadline  reason=deadline  "
                "decision=%s decider_reason=%s  has_transcript=%s  "
                "suspend_ms=%.0f  buffered=%d frames (%.3fs)  timeout=%.2fs",
                decision.action.value,
                decision.reason,
                bool(latest_asr_text),
                stats.suspend_ms,
                stats.buffered_frames,
                stats.buffered_sec,
                timeout_sec,
            )
            self._apply_decision(
                decision,
                resolved_reason="timeout",
                transcript=latest_asr_text,
                vad_active=vad_still_active,
            )
        except asyncio.CancelledError:
            pass

    def _resolve_hold_or_rearm(
        self,
        decision: Decision,
        *,
        timeout_sec: float,
        eot_model: Any,
    ) -> Decision:
        eot_config = getattr(eot_model, "_config", None)
        owner_max_suspend_sec = self._get_max_suspend_sec()
        if owner_max_suspend_sec > 0:
            max_suspend_sec = max(timeout_sec, owner_max_suspend_sec)
        else:
            max_suspend_sec = max(
                timeout_sec,
                getattr(eot_config, "duck_buffer_max_sec", timeout_sec),
            )
        suspend_sec = time.monotonic() - self._get_suspend_start()
        if suspend_sec >= max_suspend_sec:
            rollback = Decision(
                action=Action.ROLLBACK,
                reason=(
                    "deadline_hold_max_suspend_elapsed "
                    f"suspend={suspend_sec:.2f}s>={max_suspend_sec:.2f}s "
                    f"last_reason={decision.reason}"
                ),
                rollback_drop_buffered=True,
                intent=InterruptIntent.UNCERTAIN,
                intent_source="timeout",
                intent_confidence=0.0,
            )
            return self._turn_runtime.tiers.annotate_decision(rollback)

        next_timeout = max(0.0, max_suspend_sec - suspend_sec)
        if decision.hold_recheck_ms is not None:
            next_timeout = min(
                next_timeout,
                max(0.0, decision.hold_recheck_ms / 1000.0),
            )
        self._set_timeout_task(
            self._create_task(self.run(min(timeout_sec, next_timeout)))
        )
        return decision
