"""Full-duplex user-turn completion and framework commit gate."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from ..observability import TurnTimeline
from ..pipeline.types import generate_turn_id
from ..session.messages import message_text
from ..turn_policy import TranscriptEvidenceGate

if TYPE_CHECKING:
    from .pipeline import StreamingPipeline

logger = logging.getLogger("agent")


class FullDuplexTurnCompletion:
    """Own the final gate before a full-duplex user turn reaches the LLM."""

    def __init__(self, pipeline: StreamingPipeline) -> None:
        self._pipeline = pipeline

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
                reason=f"voiceprint_error:{type(exc).__name__}",
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
            self.clear_session_user_turn(f"voiceprint_blocked:{reason}")
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
        self._publish_canonical_user_text(
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

    def _publish_canonical_user_text(
        self,
        transcript: str,
        *,
        source: str,
        timeline: TurnTimeline | None,
    ) -> None:
        owner = self._pipeline
        stripped = transcript.strip()
        if not stripped:
            return
        try:
            llm_plugin = getattr(getattr(owner._factory, "llm", None), "llm", None)
            setter = getattr(llm_plugin, "set_next_user_text", None)
            if setter is None:
                return
            setter(stripped, source=source)
            if timeline is not None:
                timeline.set_attr(
                    "canonical_user_text",
                    {
                        "source": source,
                        "text_preview": stripped[:120],
                        "text_length": len(stripped),
                    },
                )
        except Exception:
            logger.debug(
                "[StreamingPipeline] failed to publish canonical user text",
                exc_info=True,
            )

    def _clear_pending_canonical_user_text(self, reason: str) -> None:
        owner = self._pipeline
        try:
            factory = getattr(owner, "_factory", None)
            llm_plugin = getattr(getattr(factory, "llm", None), "llm", None)
            clearer = getattr(llm_plugin, "clear_next_user_text", None)
            if clearer is not None:
                clearer(reason=reason)
                return
            setter = getattr(llm_plugin, "set_next_user_text", None)
            if setter is not None:
                setter("", source=f"clear:{reason}")
        except Exception:
            logger.debug(
                "[StreamingPipeline] failed to clear canonical user text",
                exc_info=True,
            )

    def clear_session_user_turn(self, reason: str) -> None:
        owner = self._pipeline
        self._clear_pending_canonical_user_text(reason)
        session = getattr(owner, "_session", None)
        if session is None:
            return
        clear_user_turn = getattr(session, "clear_user_turn", None)
        if clear_user_turn is None:
            return
        try:
            clear_user_turn()
            logger.info("[StreamingPipeline] cleared user turn reason=%s", reason)
        except Exception:
            logger.exception(
                "[StreamingPipeline] failed to clear user turn reason=%s",
                reason,
            )
        if "context_error" in (reason or ""):
            self._notify_context_error_once(reason)

    def _notify_context_error_once(self, reason: str) -> None:
        owner = self._pipeline
        if getattr(owner, "_context_error_notified", False):
            return
        owner._context_error_notified = True
        logger.error(
            "[StreamingPipeline] conversation blocked: session context unresolved "
            "(reason=%s). The user/device likely references a missing agent "
            "binding; turns are dropped until it is rebound in admin.",
            reason,
        )
        session = getattr(owner, "_session", None)
        say = getattr(session, "say", None) if session is not None else None
        if not callable(say):
            return
        try:
            say(
                "抱歉，我暂时无法连接到你的助手，请检查账号绑定或联系管理员。",
                allow_interruptions=True,
            )
        except Exception:
            logger.exception("[StreamingPipeline] context-error fallback announcement failed")
            return
        try:
            owner._mark_activity()
        except Exception:
            logger.debug(
                "[StreamingPipeline] mark_activity after context-error say failed",
                exc_info=True,
            )

    async def voiceprint_allows_completed_turn(self, *, new_message: Any) -> bool:
        owner = self._pipeline
        owner._ensure_runtime_defaults()
        task = getattr(owner, "_completed_turn_voiceprint_task", None)
        result = getattr(owner, "_completed_turn_voiceprint_result", None)
        timeline = getattr(owner, "_completed_turn_voiceprint_timeline", None) or getattr(
            owner, "_timeline", None
        )
        completed_transcript = message_text(new_message)
        if timeline is not None:
            timeline.set_attr(
                "framework_completed_turn",
                {
                    "text_preview": completed_transcript[:120],
                    "text_length": len(completed_transcript),
                },
            )
        if self._stop_active_interruption_framework_completed_turn(
            completed_transcript,
            timeline=timeline,
        ):
            return False
        if task is None and result is None:
            if self._stop_non_semantic_framework_completed_turn(
                completed_transcript,
                timeline=timeline,
            ):
                return False
            if self._should_defer_framework_completed_turn(completed_transcript):
                self._defer_framework_completed_turn(
                    completed_transcript=completed_transcript,
                    timeline=timeline,
                    voiceprint_reason="",
                )
                return False
            self._align_framework_completed_turn(
                completed_transcript,
                timeline=timeline,
                voiceprint_reason="",
            )
            return True
        if result is None and task is not None:
            try:
                result = await task
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._record_voiceprint_commit_gate(
                    timeline,
                    allowed=False,
                    reason=f"voiceprint_error:{type(exc).__name__}",
                )
                self.clear_session_user_turn(f"voiceprint_error:{type(exc).__name__}")
                owner._flush_turn_timeline(timeline, "voiceprint_commit_blocked")
                logger.exception("[StreamingPipeline] voiceprint gate failed in turn hook")
                return False
            owner._completed_turn_voiceprint_result = result

        allowed = bool(getattr(result, "commit_allowed", False))
        reason = str(getattr(result, "commit_reason", "") or "unknown")
        self._record_voiceprint_commit_gate(timeline, allowed=allowed, reason=reason)
        if allowed:
            if self._stop_active_interruption_framework_completed_turn(
                completed_transcript,
                timeline=timeline,
            ):
                return False
            if self._stop_non_semantic_framework_completed_turn(
                completed_transcript,
                timeline=timeline,
            ):
                return False
            if self._should_defer_framework_completed_turn(completed_transcript):
                self._defer_framework_completed_turn(
                    completed_transcript=completed_transcript,
                    timeline=timeline,
                    voiceprint_reason=reason,
                )
                return False
            self._align_framework_completed_turn(
                completed_transcript,
                timeline=timeline,
                voiceprint_reason=reason,
            )
            return True

        if self._should_keep_waiting_merge_after_inconclusive_voiceprint(
            result,
            transcript=completed_transcript,
        ):
            self._defer_inconclusive_voiceprint_result(
                transcript=completed_transcript,
                timeline=timeline,
                reason=reason,
            )
            return False

        owner._suppress_transcripts_until_next_speech = True
        self.clear_session_user_turn(f"voiceprint_blocked:{reason}")
        owner._flush_turn_timeline(timeline, "voiceprint_commit_blocked")
        logger.info(
            "[StreamingPipeline] voiceprint gate stopped completed turn reason=%s transcript=%r",
            reason,
            completed_transcript[:80],
        )
        return False

    def _eot_thinks_turn_complete(self) -> bool:
        owner = self._pipeline
        eot_model = owner._get_eot_model()
        if eot_model is None:
            return True
        score = float(
            getattr(
                eot_model,
                "current_eot_score",
                getattr(eot_model, "_current_eot_score", 1.0),
            )
            or 0.0
        )
        return score >= float(owner._turn_policy.eot.eot_unlikely_threshold)

    def _should_defer_framework_completed_turn(self, transcript: str) -> bool:
        owner = self._pipeline
        owner._ensure_user_turn_coordinator()
        candidate = owner._user_turns.active
        if candidate is None:
            return False
        if candidate.state in {"committed", "rejected"}:
            return False
        if owner._user_turns.should_wait_for_deferred_voiceprint_merge():
            return True
        if owner._user_turns.should_wait_for_statement_sequence_merge():
            return True
        if self._eot_thinks_turn_complete():
            return False
        selected = candidate.selected_text or transcript
        return self._looks_like_short_statement_continuation(selected)

    def _defer_framework_completed_turn(
        self,
        *,
        completed_transcript: str,
        timeline: TurnTimeline | None,
        voiceprint_reason: str,
    ) -> None:
        owner = self._pipeline
        self.clear_session_user_turn("framework_completed_wait_for_continuation")
        owner._ensure_user_turn_coordinator()
        decision = owner._user_turns.defer_framework_completed(
            transcript=completed_transcript,
            reason="framework_completed_wait_for_continuation",
            timeline=timeline,
            voiceprint_reason=voiceprint_reason,
        )
        if timeline is not None:
            timeline.set_attr(
                "framework_completed_deferred",
                {
                    "reason": decision.reason,
                    "state": "waiting_merge",
                    "text_preview": decision.transcript[:120],
                    "text_length": len(decision.transcript),
                },
            )
            owner._append_turn_timeline_snapshot(
                timeline,
                "framework_completed_waiting_merge",
            )
        self.schedule_deferred_low_eot_commit(
            verify_task=None,
            eot_model=owner._get_eot_model(),
            transcript=decision.transcript or completed_transcript,
            timeline=timeline,
            delay_sec=decision.delay_sec,
        )
        owner._completed_turn_voiceprint_task = None
        owner._completed_turn_voiceprint_result = None
        owner._completed_turn_voiceprint_timeline = None
        logger.info(
            "[StreamingPipeline] deferred framework completed turn "
            "for continuation transcript=%r voiceprint_reason=%s",
            completed_transcript[:80],
            voiceprint_reason,
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
        defer_reason = f"voiceprint_inconclusive:{reason}"
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

    def _align_framework_completed_turn(
        self,
        completed_transcript: str,
        *,
        timeline: TurnTimeline | None,
        voiceprint_reason: str,
    ) -> None:
        owner = self._pipeline
        self.cancel_deferred_low_eot_commit("framework_completed_turn")
        owner._ensure_user_turn_coordinator()
        decision = owner._user_turns.mark_framework_completed(
            transcript=completed_transcript,
            reason="framework_completed_turn",
            timeline=timeline,
            voiceprint_reason=voiceprint_reason,
        )
        canonical = decision.transcript or completed_transcript
        self._publish_canonical_user_text(
            canonical,
            source="framework_completed_turn",
            timeline=timeline,
        )

    @staticmethod
    def _non_semantic_completed_turn_reason(
        timeline: TurnTimeline | None,
    ) -> str:
        if timeline is None:
            return ""
        decision = timeline.attrs.get("decision")
        if not isinstance(decision, dict):
            return ""
        action = str(decision.get("action") or "")
        intent = str(decision.get("intent") or "")
        if action == "rollback":
            return f"non_semantic_completed_turn:{intent or action}"
        if intent in {"backchannel", "noise", "hard_stop"}:
            return f"non_semantic_completed_turn:{intent}"
        return ""

    def _stop_non_semantic_framework_completed_turn(
        self,
        completed_transcript: str,
        *,
        timeline: TurnTimeline | None,
    ) -> bool:
        owner = self._pipeline
        stop_reason = self._non_semantic_completed_turn_reason(timeline)
        if not stop_reason:
            return False
        self.cancel_deferred_low_eot_commit(stop_reason)
        owner._ensure_user_turn_coordinator()
        owner._user_turns.reject_active(stop_reason)
        self.clear_session_user_turn(stop_reason)
        owner._flush_turn_timeline(timeline, stop_reason)
        logger.info(
            "[StreamingPipeline] stopped framework completed turn reason=%s transcript=%r",
            stop_reason,
            completed_transcript[:80],
        )
        return True

    def _stop_active_interruption_framework_completed_turn(
        self,
        completed_transcript: str,
        *,
        timeline: TurnTimeline | None,
    ) -> bool:
        owner = self._pipeline
        interruption_owner = getattr(owner, "_interruption_orchestrator", None)
        if (
            interruption_owner is None
            or not interruption_owner.blocks_framework_completed_turn()
        ):
            return False
        reason = "interruption_owner_waiting_for_evidence"
        self.cancel_deferred_low_eot_commit(reason)
        self.clear_session_user_turn(reason)
        if timeline is not None:
            timeline.set_attr(
                "framework_completed_blocked_by_interruption_owner",
                {
                    "reason": reason,
                    "state": interruption_owner.state.value,
                    "text_preview": completed_transcript[:120],
                    "text_length": len(completed_transcript),
                },
            )
            owner._append_turn_timeline_snapshot(timeline, reason)
        logger.info(
            "[StreamingPipeline] blocked framework completed turn while "
            "interruption owner waits reason=%s state=%s transcript=%r",
            reason,
            interruption_owner.state.value,
            completed_transcript[:80],
        )
        return True

    def commit_post_speech_interruption_candidate(
        self,
        reason: str,
        *,
        transcript_override: str = "",
    ) -> bool:
        owner = self._pipeline
        interruption_owner = getattr(owner, "_interruption_orchestrator", None)
        owner._ensure_user_turn_coordinator()
        owner_transcript = (
            interruption_owner.current_transcript if interruption_owner is not None else ""
        )
        transcript = (
            transcript_override
            or owner_transcript
            or owner._user_turns.selected_text
            or owner._latest_asr_text
        ).strip()
        if not transcript:
            return False
        timeline = getattr(owner, "_timeline", None)
        self.cancel_deferred_low_eot_commit(reason)
        if owner._user_turns.active is None:
            if owner._timeline is None:
                owner._timeline = TurnTimeline(generate_turn_id())
                owner._timeline_debug_flushed = False
                timeline = owner._timeline
            owner._user_turns.start_speech(timeline=owner._timeline)
            owner._apply_pending_explicit_client_preempt(owner._timeline)
            owner._apply_pending_client_control_events(owner._timeline)
        owner._user_turns.add_transcript(transcript, is_final=True)
        eot_model = owner._get_eot_model()
        decision = owner._user_turns.finish_speech(
            eot_score=getattr(
                eot_model,
                "current_eot_score",
                getattr(eot_model, "_current_eot_score", None),
            ),
            should_defer=False,
        )
        if decision.action == "reject":
            self.clear_session_user_turn(decision.reason)
            return False
        committed_text = decision.transcript or transcript
        if timeline is not None:
            timeline.set_attr(
                "post_speech_interruption_candidate_committed",
                {
                    "reason": reason,
                    "transcript_preview": committed_text[:120],
                    "text_length": len(committed_text),
                },
            )
        self.schedule_voiceprint_gated_commit(
            verify_task=self.candidate_voiceprint_gate_task(),
            eot_model=eot_model,
            transcript=committed_text,
            timeline=timeline,
        )
        owner._latest_asr_text = ""
        logger.info(
            "[StreamingPipeline] committed post-speech interruption candidate "
            "reason=%s transcript=%r",
            reason,
            committed_text[:80],
        )
        return True

    def reject_post_speech_interruption_candidate(self, reason: str) -> None:
        owner = self._pipeline
        timeline = getattr(owner, "_timeline", None)
        self.cancel_deferred_low_eot_commit(reason)
        owner._ensure_user_turn_coordinator()
        decision = owner._user_turns.reject_active(reason)
        eot_model = owner._get_eot_model()
        try:
            eot_model.reset()
        except Exception:
            logger.debug(
                "[StreamingPipeline] EOT reset failed while rejecting "
                "post-speech interruption candidate",
                exc_info=True,
            )
        task = getattr(owner, "_completed_turn_voiceprint_task", None)
        if task is not None and not task.done():
            task.cancel()
        owner._completed_turn_voiceprint_task = None
        owner._completed_turn_voiceprint_result = None
        owner._completed_turn_voiceprint_timeline = None
        self.reset_candidate_voiceprint_tasks()
        self.clear_session_user_turn(reason)
        owner._latest_asr_text = ""
        if timeline is not None:
            timeline.set_attr(
                "post_speech_interruption_candidate_rejected",
                {
                    "reason": reason,
                    "transcript_preview": decision.transcript[:120],
                    "text_length": len(decision.transcript),
                },
            )
            owner._flush_turn_timeline(timeline, reason)
        logger.info(
            "[StreamingPipeline] rejected post-speech interruption candidate "
            "reason=%s transcript=%r",
            reason,
            decision.transcript[:80],
        )

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
    reason = str(getattr(result, "commit_reason", "") or "").lower()
    return reason in {"audio_too_short", "insufficient_audio", "too_short"}
