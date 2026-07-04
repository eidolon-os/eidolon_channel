"""Full-duplex interruption output side effects."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.output import OutputDuckingController
from eidolon.livekit.agent.pipeline.types import PipelineCallbacks
from eidolon.livekit.agent.session.interruption import SoftInterruptController
from eidolon.livekit.agent.turn_policy import Decision
from eidolon.livekit.agent.turn_policy.constants import STABLE_SIGNAL_WAIT_REASON_PREFIX

logger = logging.getLogger("agent.full_duplex.interruption_effects")


class FullDuplexInterruptionEffects:
    """Apply full-duplex interruption effects to output and AgentSession.

    Turn policy, semantic classification, and user-turn commit remain separate
    owners. This component owns the immediate output/framework effects after a
    decision says cancel, rollback, hold, or force-preempt.
    """

    def __init__(
        self,
        *,
        ducking: OutputDuckingController,
        callbacks: PipelineCallbacks,
        get_session: Callable[[], Any],
        allow_interruptions: Callable[[], bool],
        get_eot_model: Callable[[], Any],
        get_timeline: Callable[[], TurnTimeline | None],
        get_latest_asr_text: Callable[[], str],
        get_state_label: Callable[[], str],
        get_interruption_orchestrator: Callable[[], Any],
        publish_playback_stop: Callable[[str], None],
        snapshot_interrupted_context: Callable[[], None],
        commit_post_speech_interruption_candidate: Callable[[str, str], bool],
        reject_post_speech_interruption_candidate: Callable[[str], None],
        cancel_residual_commit_suppress_sec: Callable[[], float],
        semantic_interrupt_run: Callable[[str], None],
        correction_topic_stability_window_ms: Callable[[], int],
        set_interrupt_cancel_suppression: Callable[[bool, float], None],
        soft_interrupt_timeout_sec: Callable[[], float],
    ) -> None:
        self._ducking = ducking
        self._callbacks = callbacks
        self._get_session = get_session
        self._allow_interruptions = allow_interruptions
        self._get_eot_model = get_eot_model
        self._get_timeline = get_timeline
        self._get_latest_asr_text = get_latest_asr_text
        self._get_state_label = get_state_label
        self._get_interruption_orchestrator = get_interruption_orchestrator
        self._publish_playback_stop = publish_playback_stop
        self._snapshot_context = snapshot_interrupted_context
        self._commit_post_speech_interruption_candidate = (
            commit_post_speech_interruption_candidate
        )
        self._reject_post_speech_interruption_candidate = (
            reject_post_speech_interruption_candidate
        )
        self._cancel_residual_commit_suppress_sec = cancel_residual_commit_suppress_sec
        self._semantic_interrupt_run = semantic_interrupt_run
        self._correction_topic_stability_window_ms = (
            correction_topic_stability_window_ms
        )
        self._set_interrupt_cancel_suppression = set_interrupt_cancel_suppression
        self._soft_interrupt_timeout_sec = soft_interrupt_timeout_sec
        self._soft_interrupt = SoftInterruptController(
            timeout_sec=self._soft_interrupt_timeout_sec(),
            on_timeout=lambda: self.interrupt_current_turn(),
        )
        self._stable_signal_timer: asyncio.Task | None = None

    def interrupt_current_turn(
        self,
        *,
        force: bool = False,
        allow_cancelled_output: bool = False,
    ) -> None:
        """Interrupt the current AgentSession turn."""

        if not force and not self._allow_interruptions():
            return
        if not force and self._ducking.is_cancelled and not allow_cancelled_output:
            logger.debug(
                "[FullDuplexInterruptionEffects] interrupt skipped; output already CANCELLED"
            )
            return

        session = self._get_session()
        if session is not None:
            session.interrupt(force=force)

        self._get_eot_model().update_vad(False)
        logger.info("[FullDuplexInterruptionEffects] turn interrupted")

    def enter_soft_interrupt(self) -> None:
        self._soft_interrupt.timeout_sec = self._soft_interrupt_timeout_sec()
        self._soft_interrupt.enter()

    def cancel_soft_interrupt(self) -> None:
        self._soft_interrupt.cancel()

    def soft_interrupt_active(self) -> bool:
        return self._soft_interrupt.active

    def handle_hold_decision(
        self,
        decision: Decision,
        transcript: str,
        eot_score: float | None,
        vad_active: bool | None,
    ) -> None:
        if decision.hold_recheck_ms is None and not decision.reason.startswith(
            STABLE_SIGNAL_WAIT_REASON_PREFIX
        ):
            return
        if not self._ducking.is_suspended:
            return
        if not transcript.strip():
            return
        recheck_ms = decision.hold_recheck_ms
        if recheck_ms is None:
            recheck_ms = self._correction_topic_stability_window_ms()
        timeout_sec = max(0.0, recheck_ms) / 1000.0
        self.cancel_stable_signal_timer()
        logger.info(
            "[FullDuplexInterruptionEffects] stable-signal recheck armed "
            "timeout=%.3fs reason=%s text=%r eot_score=%s vad_active=%s",
            timeout_sec,
            decision.reason,
            transcript[:80],
            f"{eot_score:.2f}" if eot_score is not None else "None",
            vad_active,
        )
        self._stable_signal_timer = asyncio.create_task(
            self._stable_signal_recheck(timeout_sec, transcript)
        )

    async def _stable_signal_recheck(self, timeout_sec: float, transcript: str) -> None:
        current_task = asyncio.current_task()
        try:
            await asyncio.sleep(timeout_sec)
            if not self._ducking.is_suspended:
                return
            latest = (self._get_latest_asr_text() or transcript).strip()
            if not latest:
                return
            logger.info(
                "[FullDuplexInterruptionEffects] stable-signal recheck firing text=%r",
                latest[:80],
            )
            self._semantic_interrupt_run(latest)
        except asyncio.CancelledError:
            return
        finally:
            if self._stable_signal_timer is current_task:
                self._stable_signal_timer = None

    def cancel_stable_signal_timer(self) -> None:
        task = self._stable_signal_timer
        if task is not None and not task.done():
            task.cancel()
        self._stable_signal_timer = None

    def cancel_and_interrupt(self, *, force: bool = False) -> None:
        """Confirm an interruption by dropping buffered output and cancelling TTS."""

        if self._ducking.is_cancelled:
            logger.debug(
                "[FullDuplexInterruptionEffects] duplicate duck cancel ignored; "
                "output already CANCELLED"
            )
            return
        orchestrator = self._get_interruption_orchestrator()
        collect_confirmed_cancel_turn = (
            orchestrator.should_collect_after_confirmed_cancel()
        )
        commit_post_speech_candidate = (
            False
            if collect_confirmed_cancel_turn
            else orchestrator.should_commit_after_confirmed_cancel()
        )
        post_speech_transcript = (
            orchestrator.current_transcript if commit_post_speech_candidate else ""
        )
        stats = self._ducking.stats()
        logger.info(
            "[FullDuplexInterruptionEffects] duck resolved reason=eot_cancel "
            "action=cancel suspend_ms=%.0f discarded=%d frames (%.3fs)",
            stats.suspend_ms,
            stats.buffered_frames,
            stats.buffered_sec,
        )
        self.cancel_stable_signal_timer()
        self._snapshot_context()
        self._publish_playback_stop("interrupt_cancel")
        self._ducking.cancel_output()
        if collect_confirmed_cancel_turn:
            orchestrator.mark_confirmed_cancel_collecting_turn()
        else:
            orchestrator.resolve(action="cancel", reason="eot_cancel")
        self._callbacks.on_duck_resolved("cancel")
        timeline = self._get_timeline()
        if timeline is not None:
            self._record_duck_event(
                timeline,
                "duck_cancelled",
                reason="eot_cancel",
                suspend_ms=stats.suspend_ms,
                buffered_frames=stats.buffered_frames,
                buffered_sec=stats.buffered_sec,
                drop_buffered=True,
            )
            timeline.mark("interrupt_resolved_at")
            timeline.set_attr("cancel_reason", "eot_cancel")

        self._set_interrupt_cancel_suppression(
            True,
            time.monotonic() + self._cancel_residual_commit_suppress_sec(),
        )
        self.interrupt_current_turn(
            force=force,
            allow_cancelled_output=True,
        )
        if commit_post_speech_candidate:
            committed = self._commit_post_speech_interruption_candidate(
                "post_speech_confirmed_cancel",
                post_speech_transcript,
            )
            if committed:
                self._set_interrupt_cancel_suppression(False, 0.0)

    def rollback_if_suspended(
        self,
        reason: str = "user_silent",
        *,
        drop_buffered: bool = False,
    ) -> None:
        """Resume output if a suspended interruption candidate is rejected."""

        if not self._ducking.installed:
            return
        if not self._ducking.is_suspended:
            return

        orchestrator = self._get_interruption_orchestrator()
        waiting_post_speech_evidence = orchestrator.awaiting_post_speech_evidence
        stats = self._ducking.stats()
        logger.info(
            "[FullDuplexInterruptionEffects] duck resolved reason=%s "
            "action=unduck(drop_buffered=%s) suspend_ms=%.0f "
            "buffered=%d frames (%.3fs)",
            reason,
            drop_buffered,
            stats.suspend_ms,
            stats.buffered_frames,
            stats.buffered_sec,
        )
        self.cancel_stable_signal_timer()
        self._ducking.unduck_if_suspended(drop_buffered=drop_buffered)
        orchestrator.resolve(action="rollback", reason=reason)
        self._callbacks.on_duck_resolved("unduck")
        timeline = self._get_timeline()
        if timeline is not None:
            self._record_duck_event(
                timeline,
                "duck_unducked",
                reason=reason,
                suspend_ms=stats.suspend_ms,
                buffered_frames=stats.buffered_frames,
                buffered_sec=stats.buffered_sec,
                drop_buffered=drop_buffered,
            )
            timeline.mark("interrupt_resolved_at")
            timeline.set_attr("interrupt_action", "rollback")
            timeline.set_attr("rollback_drop_buffered", drop_buffered)
            timeline.set_attr("rollback_reason", reason)
        if waiting_post_speech_evidence:
            reject_reason = (
                "post_speech_evidence_timeout"
                if reason == "timeout"
                else f"post_speech_false_interruption:{reason}"
            )
            self._reject_post_speech_interruption_candidate(reject_reason)

    def cancel_silent_generation_for_explicit_preempt(self) -> None:
        """Cancel a non-audible active agent generation for explicit client control."""

        self.cancel_stable_signal_timer()
        self._ducking.cancel_output()
        timeline = self._get_timeline()
        if timeline is not None:
            timeline.set_attr(
                "explicit_client_generation_preempt",
                {
                    "reason": "explicit_client_ptt",
                    "agent_state": self._get_state_label(),
                    "playback_state": "not_audible",
                },
            )
        logger.info(
            "[FullDuplexInterruptionEffects] explicit client preempted silent "
            "agent generation state=%s",
            self._get_state_label(),
        )
        self.interrupt_current_turn(force=True)

    @staticmethod
    def _record_duck_event(
        timeline: TurnTimeline,
        event: str,
        **fields: object,
    ) -> None:
        payload = {"event": event, **fields}
        events = list(timeline.attrs.get("duck_events") or ())
        events.append(payload)
        timeline.set_attr("duck_events", events)
        timeline.set_attr("duck_last_event", payload)
