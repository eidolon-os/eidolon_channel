"""Full-duplex user-turn completion and framework commit gate."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from ..observability import TurnTimeline
from ..session.voiceprint_reasons import (
    is_voiceprint_inconclusive_reason,
    voiceprint_blocked_reason,
    voiceprint_error_reason,
    voiceprint_inconclusive_reason,
)
from ..turn_policy import TranscriptEvidenceGate
from .framework_completed_turn import FullDuplexFrameworkCompletedTurnGate
from .post_speech_interruption import FullDuplexPostSpeechInterruptionCommitter
from .session_turn_boundary import FullDuplexSessionTurnBoundary

if TYPE_CHECKING:
    from .pipeline import StreamingPipeline

logger = logging.getLogger("agent")


class FullDuplexTurnCompletion:
    """Own the final gate before a full-duplex user turn reaches the LLM."""

    def __init__(self, pipeline: StreamingPipeline) -> None:
        self._pipeline = pipeline
        self._session_turns = FullDuplexSessionTurnBoundary(pipeline)
        self._framework_completed_turn = FullDuplexFrameworkCompletedTurnGate(
            pipeline,
            completion=self,
            session_turns=self._session_turns,
        )
        self._post_speech_interruption = FullDuplexPostSpeechInterruptionCommitter(
            pipeline,
            cancel_deferred_low_eot_commit=self.cancel_deferred_low_eot_commit,
            clear_session_user_turn=self.clear_session_user_turn,
            candidate_voiceprint_gate_task=self.candidate_voiceprint_gate_task,
            schedule_voiceprint_gated_commit=self.schedule_voiceprint_gated_commit,
            reset_candidate_voiceprint_tasks=self.reset_candidate_voiceprint_tasks,
        )

    def cancel_pending_voiceprint_commits(self, reason: str) -> None:
        owner = self._pipeline
        tasks = getattr(owner, "_pending_voiceprint_commit_tasks", set())
        for task in list(tasks):
            if not task.done():
                logger.info(
                    "[StreamingPipeline] cancelling pending voiceprint-gated commit reason=%s",
                    reason,
                )
                task.cancel()

    def reset_candidate_voiceprint_tasks(self) -> None:
        self._pipeline._candidate_voiceprint_tasks = []

    def remember_candidate_voiceprint_task(self, task: asyncio.Task | None) -> None:
        owner = self._pipeline
        if task is None:
            return
        if not hasattr(owner, "_candidate_voiceprint_tasks"):
            owner._candidate_voiceprint_tasks = []
        owner._candidate_voiceprint_tasks.append(task)

    def candidate_voiceprint_gate_task(self) -> asyncio.Task | None:
        owner = self._pipeline
        tasks = list(getattr(owner, "_candidate_voiceprint_tasks", []))
        owner._candidate_voiceprint_tasks = []
        if not tasks:
            return None
        if len(tasks) == 1:
            return tasks[0]
        return asyncio.create_task(self._combine_candidate_voiceprint_results(tasks))

    async def _combine_candidate_voiceprint_results(
        self,
        tasks: list[asyncio.Task],
    ) -> Any:
        results = await asyncio.gather(*tasks)
        inconclusive = None
        for result in results:
            if not bool(getattr(result, "commit_allowed", False)):
                if _voiceprint_result_is_inconclusive(result):
                    inconclusive = inconclusive or result
                    continue
                return result
        if results and bool(getattr(results[-1], "commit_allowed", False)):
            return results[-1]
        if inconclusive is not None:
            return inconclusive
        return results[-1]

    def cancel_deferred_low_eot_commit(self, reason: str) -> None:
        owner = self._pipeline
        task = getattr(owner, "_deferred_low_eot_commit_task", None)
        if task is None or task.done():
            owner._deferred_low_eot_commit_task = None
            return
        logger.info(
            "[StreamingPipeline] cancelling deferred low-EOT commit reason=%s",
            reason,
        )
        task.cancel()
        owner._deferred_low_eot_commit_task = None

    def should_defer_low_eot_commit(self, *, transcript: str, eot_model: Any) -> bool:
        owner = self._pipeline
        owner._ensure_runtime_defaults()
        if not transcript.strip():
            return False
        score = float(
            getattr(
                eot_model,
                "current_eot_score",
                getattr(eot_model, "_current_eot_score", 1.0),
            )
            or 0.0
        )
        if score < float(owner._turn_policy.eot.eot_unlikely_threshold):
            return True
        return self._looks_like_short_statement_continuation(transcript)

    def playback_low_evidence_reject_reason(
        self,
        *,
        transcript: str,
        eot_model: Any,
    ) -> str:
        owner = self._pipeline
        if not transcript.strip():
            return ""
        timeline = getattr(owner, "_timeline", None)
        if timeline is None:
            return ""
        if self._has_confirmed_redirect_decision(timeline):
            return ""
        events = timeline.attrs.get("attention_admission_events") or ()
        playback_observed = any(
            isinstance(event, dict)
            and event.get("action") == "observe"
            and str(event.get("reason") or "").startswith(
                (
                    "client_playback_active_without_direct_signal",
                    "playback_low_evidence_transcript",
                )
            )
            for event in events
        )
        if not playback_observed:
            return ""
        score = float(
            getattr(
                eot_model,
                "current_eot_score",
                getattr(eot_model, "_current_eot_score", 0.0),
            )
            or 0.0
        )
        evidence = TranscriptEvidenceGate(owner._turn_policy.interrupt).evaluate_attention(
            transcript, eot_score=score
        )
        if evidence.allow_decision:
            return ""
        return f"playback_low_evidence_artifact:{evidence.reason}"

    @staticmethod
    def _has_confirmed_redirect_decision(timeline: TurnTimeline) -> bool:
        decision = timeline.attrs.get("decision")
        if not isinstance(decision, dict):
            return False
        if decision.get("action") != "cancel":
            return False
        if bool(decision.get("topic_switch_hint")) or bool(
            decision.get("correction_hint")
        ):
            return True
        reason = str(decision.get("reason") or "")
        return reason.startswith(("intent:topic_switch", "intent:correction"))

    def _looks_like_short_statement_continuation(self, transcript: str) -> bool:
        owner = self._pipeline
        text = transcript.strip()
        if not text:
            return False
        if any(mark in text for mark in ("？", "?", "！", "!")):
            return False
        cjk_chars = _count_cjk_chars(text)
        if cjk_chars <= 0:
            return False
        if cjk_chars > owner._turn_policy.eot.short_statement_defer_max_cjk_chars:
            return False
        if text.startswith(("帮我", "请", "麻烦", "换个话题", "换一个话题")):
            return False
        return text.endswith(("。", "，", ",", "、", "的", "了", "呢", "吧"))

    def schedule_deferred_low_eot_commit(
        self,
        *,
        verify_task: asyncio.Task | None,
        eot_model: Any,
        transcript: str,
        timeline: TurnTimeline | None,
        delay_sec: float | None = None,
    ) -> None:
        owner = self._pipeline
        self.cancel_deferred_low_eot_commit("replace_deferred_commit")
        if delay_sec is None:
            delay = min(
                max(owner._turn_policy.eot.tail_hang_silence_ms / 1000.0, 0.0),
                owner._low_eot_commit_grace_max_sec(),
            )
        else:
            delay = max(float(delay_sec), 0.0)
        task = asyncio.create_task(
            self._run_deferred_low_eot_commit(
                delay=delay,
                verify_task=verify_task,
                eot_model=eot_model,
                transcript=transcript,
                timeline=timeline,
            )
        )
        owner._deferred_low_eot_commit_task = task
        logger.info(
            "[StreamingPipeline] deferred low-EOT commit delay=%.3fs score=%s transcript=%r",
            delay,
            getattr(
                eot_model,
                "current_eot_score",
                getattr(eot_model, "_current_eot_score", None),
            ),
            transcript[:80],
        )

    async def _run_deferred_low_eot_commit(
        self,
        *,
        delay: float,
        verify_task: asyncio.Task | None,
        eot_model: Any,
        transcript: str,
        timeline: TurnTimeline | None,
    ) -> None:
        owner = self._pipeline
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            owner._ensure_user_turn_coordinator()
            decision = owner._user_turns.deferred_ready()
            if decision.action != "commit":
                logger.info(
                    "[StreamingPipeline] deferred low-EOT commit skipped reason=%s",
                    decision.reason,
                )
                return
            final_transcript = (
                decision.transcript.strip() or owner._latest_asr_text.strip() or transcript
            )
            self.schedule_voiceprint_gated_commit(
                verify_task=self.candidate_voiceprint_gate_task() or verify_task,
                eot_model=eot_model,
                transcript=final_transcript,
                timeline=timeline,
            )
            owner._latest_asr_text = ""
        except asyncio.CancelledError:
            raise
        finally:
            if owner._deferred_low_eot_commit_task is asyncio.current_task():
                owner._deferred_low_eot_commit_task = None

    def schedule_voiceprint_gated_commit(
        self,
        *,
        verify_task: asyncio.Task | None,
        eot_model: Any,
        transcript: str,
        timeline: TurnTimeline | None,
    ) -> None:
        owner = self._pipeline
        if verify_task is None:
            self._commit_user_turn_now(
                eot_model=eot_model,
                transcript=transcript,
                timeline=timeline,
            )
            return
        owner._completed_turn_voiceprint_task = verify_task
        owner._completed_turn_voiceprint_result = None
        owner._completed_turn_voiceprint_timeline = timeline
        task = asyncio.create_task(
            self._finalize_voiceprint_gated_commit(
                verify_task=verify_task,
                eot_model=eot_model,
                transcript=transcript,
                timeline=timeline,
            )
        )
        owner._pending_voiceprint_commit_tasks.add(task)
        task.add_done_callback(owner._pending_voiceprint_commit_tasks.discard)

    async def _finalize_voiceprint_gated_commit(
        self,
        *,
        verify_task: asyncio.Task,
        eot_model: Any,
        transcript: str,
        timeline: TurnTimeline | None,
    ) -> None:
        owner = self._pipeline
        try:
            result = await verify_task
        except asyncio.CancelledError:
            eot_model.reset()
            raise
        except Exception as exc:
            eot_model.reset()
            self.clear_session_user_turn("voiceprint_error")
            self._record_voiceprint_commit_gate(
                timeline,
                allowed=False,
                reason=voiceprint_error_reason(type(exc).__name__),
            )
            owner._flush_turn_timeline(timeline, "voiceprint_commit_blocked")
            logger.exception("[StreamingPipeline] voiceprint gate failed")
            return

        owner._completed_turn_voiceprint_result = result
        owner._ensure_user_turn_coordinator()
        allowed = bool(getattr(result, "commit_allowed", False))
        raw_reason = str(getattr(result, "commit_reason", "") or "unknown")
        if not allowed and self._should_keep_waiting_merge_after_inconclusive_voiceprint(
            result,
            transcript=transcript,
        ):
            self._defer_inconclusive_voiceprint_result(
                transcript=transcript,
                timeline=timeline,
                reason=raw_reason,
            )
            self._record_voiceprint_commit_gate(
                timeline,
                allowed=False,
                reason=raw_reason,
            )
            return

        decision = owner._user_turns.apply_voiceprint_result(result)
        allowed = decision.action == "commit"
        reason = decision.reason or str(getattr(result, "commit_reason", "") or "unknown")
        self._record_voiceprint_commit_gate(timeline, allowed=allowed, reason=reason)
        if not allowed:
            eot_model.reset()
            owner._suppress_transcripts_until_next_speech = True
            self.clear_session_user_turn(voiceprint_blocked_reason(reason))
            owner._flush_turn_timeline(timeline, "voiceprint_commit_blocked")
            logger.info(
                "[StreamingPipeline] voiceprint gate blocked commit reason=%s transcript=%r",
                reason,
                transcript[:80],
            )
            return

        self._commit_user_turn_now(
            eot_model=eot_model,
            transcript=decision.transcript or transcript,
            timeline=timeline,
        )

    def _commit_user_turn_now(
        self,
        *,
        eot_model: Any,
        transcript: str,
        timeline: TurnTimeline | None,
    ) -> bool:
        owner = self._pipeline
        if owner._session is None:
            eot_model.reset()
            return False
        if timeline is not None:
            timeline.set_attr(
                "framework_commit_request",
                {
                    "transcript_preview": transcript[:120],
                    "transcript_length": len(transcript),
                },
            )
        self._session_turns.publish_canonical_user_text(
            transcript,
            source="user_turn_coordinator",
            timeline=timeline,
        )
        owner._ensure_turn_committer()
        committed = owner._turn_committer.commit_or_skip(
            session=owner._session,
            eot_model=eot_model,
            transcript=transcript,
            transcript_timeout=owner._stt_commit_transcript_timeout,
            timeline=timeline,
            inject_interrupted_context=lambda: owner._ensure_context_ledger().inject(),
            filler=owner._filler,
        )
        if not committed:
            owner._ensure_user_turn_coordinator()
            owner._user_turns.reject_active("empty_transcript")
            self.clear_session_user_turn("empty_transcript")
        else:
            owner._ensure_user_turn_coordinator()
            owner._user_turns.mark_committed(
                transcript=transcript,
                reason="framework_commit_user_turn",
            )
        return committed

    def clear_session_user_turn(self, reason: str) -> None:
        self._session_turns.clear_session_user_turn(reason)

    async def voiceprint_allows_completed_turn(self, *, new_message: Any) -> bool:
        return await self._framework_completed_turn.allows_completed_turn(
            new_message=new_message
        )

    def _should_keep_waiting_merge_after_inconclusive_voiceprint(
        self,
        result: Any,
        *,
        transcript: str,
    ) -> bool:
        owner = self._pipeline
        if not _voiceprint_result_is_inconclusive(result):
            return False
        owner._ensure_user_turn_coordinator()
        candidate = owner._user_turns.active
        if candidate is None:
            return False
        if candidate.state == "waiting_merge":
            return True
        if candidate.state in {"committed", "rejected"}:
            return False
        selected = candidate.selected_text or transcript
        return self._looks_like_short_statement_continuation(selected)

    def _defer_inconclusive_voiceprint_result(
        self,
        *,
        transcript: str,
        timeline: TurnTimeline | None,
        reason: str,
    ) -> None:
        owner = self._pipeline
        defer_reason = voiceprint_inconclusive_reason(reason)
        self.clear_session_user_turn(defer_reason)
        owner._ensure_user_turn_coordinator()
        decision = owner._user_turns.defer_voiceprint_inconclusive(
            transcript=transcript,
            reason=defer_reason,
            timeline=timeline,
        )
        if timeline is not None:
            timeline.set_attr(
                "voiceprint_deferred",
                {
                    "reason": defer_reason,
                    "state": "waiting_merge",
                    "text_preview": decision.transcript[:120],
                    "text_length": len(decision.transcript),
                },
            )
            owner._append_turn_timeline_snapshot(timeline, "voiceprint_waiting_merge")
        logger.info(
            "[StreamingPipeline] deferred inconclusive voiceprint result reason=%s transcript=%r",
            reason,
            transcript[:80],
        )

    def commit_post_speech_interruption_candidate(
        self,
        reason: str,
        *,
        transcript_override: str = "",
    ) -> bool:
        return self._post_speech_interruption.commit_candidate(
            reason,
            transcript_override=transcript_override,
        )

    def reject_post_speech_interruption_candidate(self, reason: str) -> None:
        self._post_speech_interruption.reject_candidate(reason)

    def _record_voiceprint_commit_gate(
        self,
        timeline: TurnTimeline | None,
        *,
        allowed: bool,
        reason: str,
    ) -> None:
        if timeline is None:
            return
        timeline.set_attr(
            "voiceprint_commit_gate",
            {
                "allowed": allowed,
                "reason": reason,
            },
        )


def _count_cjk_chars(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def _voiceprint_result_is_inconclusive(result: Any) -> bool:
    reason = str(getattr(result, "commit_reason", "") or "")
    return is_voiceprint_inconclusive_reason(reason)
