"""Streaming-mode half-duplex PTT owner adapter.

``PttTurnOwner`` is the pure state machine.  This adapter owns the small amount
of runtime glue around it: press/release/VAD/STT events, release-finalization
timers, and terminal decision callbacks.  LiveKit side effects stay in
``StreamingPipeline``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from eidolon.livekit.common.config import PttPolicyConfig

from .ptt_turn import PttTurnDecision, PttTurnOwner, PttTurnOwnerConfig

logger = logging.getLogger("agent.session.ptt_manual")


def ptt_turn_owner_config_from_policy(policy: PttPolicyConfig) -> PttTurnOwnerConfig:
    return PttTurnOwnerConfig(
        empty_probe_sec=policy.empty_probe_ms / 1000.0,
        finalization_timeout_sec=policy.finalization_timeout_ms / 1000.0,
        post_vad_settle_sec=policy.post_vad_settle_ms / 1000.0,
        stable_interim_sec=policy.stable_interim_ms / 1000.0,
    )


class PttManualTurnHandler:
    """Own the streaming/manual PTT lifecycle around ``PttTurnOwner``."""

    def __init__(
        self,
        *,
        config: PttTurnOwnerConfig,
        on_decision: Callable[[PttTurnDecision], None],
        on_commit: Callable[[PttTurnDecision], None],
        on_reject: Callable[[PttTurnDecision], None],
        sleep: Callable[[float], Awaitable[None]] | None = None,
        create_task: Callable[[Awaitable[None]], asyncio.Task] | None = None,
    ) -> None:
        self._owner = PttTurnOwner(config=config)
        self._on_decision = on_decision
        self._on_commit = on_commit
        self._on_reject = on_reject
        self._sleep = sleep or asyncio.sleep
        self._create_task = create_task or (lambda coro: asyncio.create_task(coro))
        self._finalize_task: asyncio.Task | None = None

    @property
    def state(self) -> str:
        return self._owner.state

    @property
    def speech_detected(self) -> bool:
        return self._owner.speech_detected

    def press(self, *, preempted_agent_output: bool = False) -> None:
        self.cancel_finalize_task("new_ptt_press")
        self._handle_decision(
            self._owner.press(preempted_agent_output=preempted_agent_output)
        )

    def release(self) -> None:
        self._handle_decision(self._owner.release())

    def vad_started(self) -> None:
        self._handle_decision(self._owner.vad_started())

    def vad_stopped(self) -> None:
        self._handle_decision(self._owner.vad_stopped())

    def transcript(self, text: str, *, is_final: bool) -> None:
        self._handle_decision(self._owner.transcript(text, is_final=is_final))

    def resolve(self) -> None:
        self._handle_decision(self._owner.resolve())

    def cancel_finalize_task(self, reason: str) -> None:
        task = self._finalize_task
        if task is not None and not task.done():
            task.cancel()
        self._finalize_task = None
        if reason:
            logger.debug("[ptt-manual] finalize timer cancelled reason=%s", reason)

    def _handle_decision(self, decision: PttTurnDecision) -> None:
        self._on_decision(decision)
        if decision.action == "commit":
            self.cancel_finalize_task("ptt_terminal_commit")
            self._on_commit(decision)
            return
        if decision.action == "reject":
            self.cancel_finalize_task("ptt_terminal_reject")
            self._on_reject(decision)
            return
        if decision.state == "released_finalizing":
            self._schedule_finalize(decision.next_delay_sec)

    def _schedule_finalize(self, delay_sec: float | None) -> None:
        self.cancel_finalize_task("replace_ptt_finalize_timer")
        delay = max(0.0, float(delay_sec or 0.0))
        coro = self._run_finalize_after(delay)
        try:
            self._finalize_task = self._create_task(coro)
        except RuntimeError:
            coro.close()
            logger.debug("[ptt-manual] no running event loop; finalize timer not scheduled")

    async def _run_finalize_after(self, delay_sec: float) -> None:
        try:
            if delay_sec > 0:
                await self._sleep(delay_sec)
            self.resolve()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[ptt-manual] finalization task failed")
        finally:
            if self._finalize_task is asyncio.current_task():
                self._finalize_task = None
