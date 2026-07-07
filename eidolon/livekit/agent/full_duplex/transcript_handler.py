"""Full-duplex transcript entry workflow."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .semantic_interrupt_gate import evaluate_semantic_interrupt_gate
from .transcript_admission import TranscriptAdmissionGate
from .transcript_event import FullDuplexTranscriptEvent

logger = logging.getLogger("agent")


class FullDuplexTranscriptHandler:
    """Route accepted STT transcripts into turn/evidence logic.

    This handler owns the STT event entry workflow after LiveKit event
    normalization. It does not own EOT, commit, cancel, or resume decisions;
    those remain with the turn runtime and semantic interruption owner.
    """

    def __init__(
        self,
        *,
        admission_gate: Callable[[], TranscriptAdmissionGate],
        record_accepted_event: Callable[[FullDuplexTranscriptEvent], None],
        allow_interruptions: Callable[[], bool],
        native_adaptive_owner: Callable[[], bool],
        agent_output_active: Callable[[str | None], bool],
        interrupt_window_active: Callable[[], bool],
        decision_suppressed: Callable[[], bool],
        attention_allows_eot_check: Callable[[str, str | None], bool],
        run_semantic_interrupt: Callable[[str, bool], None],
        reject_agent_echo: Callable[[str], None],
        forward_to_base: Callable[[Any], None],
        record_transcript_ingress_event: Callable[[dict[str, Any]], None]
        | None = None,
        record_transcript_admission_event: Callable[[dict[str, Any]], None]
        | None = None,
        record_semantic_gate_event: Callable[[dict[str, Any]], None] | None = None,
        warm_preemptive: Callable[[str], None] | None = None,
    ) -> None:
        self._admission_gate = admission_gate
        self._record_accepted_event = record_accepted_event
        self._allow_interruptions = allow_interruptions
        self._native_adaptive_owner = native_adaptive_owner
        self._agent_output_active = agent_output_active
        self._interrupt_window_active = interrupt_window_active
        self._decision_suppressed = decision_suppressed
        self._attention_allows_eot_check = attention_allows_eot_check
        self._run_semantic_interrupt = run_semantic_interrupt
        self._reject_agent_echo = reject_agent_echo
        self._forward_to_base = forward_to_base
        self._record_transcript_ingress_event = record_transcript_ingress_event
        self._record_transcript_admission_event = record_transcript_admission_event
        self._record_semantic_gate_event = record_semantic_gate_event
        # Optional: preemptively warm the brain on a stabilizing partial
        # transcript (fired for accepted non-final events). No-op when unset.
        self._warm_preemptive = warm_preemptive

    def handle(self, event: Any) -> None:
        transcript_event = FullDuplexTranscriptEvent.from_event(event)
        self._record_ingress_event(transcript_event, event)
        admission = self._admission_gate().evaluate(transcript_event)
        self._record_admission_event(admission)
        if not admission.accepted:
            if admission.reason == "suppressed_until_next_speech":
                logger.info(
                    "[StreamingPipeline] dropping post-turn transcript after "
                    "voiceprint ownership gate transcript=%r final=%s",
                    admission.transcript[:80],
                    admission.is_final,
                )
            elif admission.reason == "agent_echo":
                logger.info(
                    "[StreamingPipeline] dropping agent-echo transcript during playback "
                    "transcript=%r",
                    admission.transcript[:80],
                )
                self._reject_agent_echo(admission.transcript)
            return

        self._record_accepted_event(transcript_event)

        # Preemptive warm-up: on a stabilizing partial (non-final) transcript,
        # speculatively warm the brain so the real turn's first response lands
        # sooner. Ephemeral server-side; superseded by the real turn. Best
        # effort — never let it disturb transcript routing.
        if self._warm_preemptive is not None and not transcript_event.is_final:
            try:
                self._warm_preemptive(transcript_event.transcript)
            except Exception:  # noqa: BLE001
                logger.debug("[StreamingPipeline] preemptive warm hook failed", exc_info=True)

        agent_output_active = self._agent_output_active(transcript_event.speaker_id)
        interrupt_window_active = self._interrupt_window_active()
        decision_suppressed = self._decision_suppressed()
        allow_interruptions = self._allow_interruptions()
        native_adaptive_owner = self._native_adaptive_owner()

        semantic_gate = evaluate_semantic_interrupt_gate(
            allow_interruptions=allow_interruptions,
            native_adaptive=native_adaptive_owner,
            transcript=transcript_event.transcript,
            agent_output_active=agent_output_active,
            interrupt_window_active=interrupt_window_active,
            decision_suppressed=decision_suppressed,
        )
        self._record_gate_event(
            transcript_event,
            semantic_gate,
            stage="initial",
            allow_interruptions=allow_interruptions,
            native_adaptive_owner=native_adaptive_owner,
            agent_output_active=agent_output_active,
            interrupt_window_active=interrupt_window_active,
            decision_suppressed=decision_suppressed,
        )
        if semantic_gate.should_forward_and_stop:
            if semantic_gate.reason == "decision_suppressed":
                logger.debug("[StreamingPipeline] interrupt decision suppressed after cancel")
            self._forward_to_base(event)
            return

        if semantic_gate.needs_attention:
            attention_allowed = self._attention_allows_eot_check(
                transcript_event.transcript,
                transcript_event.speaker_id,
            )
            semantic_gate = semantic_gate.with_attention_result(
                attention_allowed,
            )
            self._record_gate_event(
                transcript_event,
                semantic_gate,
                stage="attention",
                allow_interruptions=allow_interruptions,
                native_adaptive_owner=native_adaptive_owner,
                agent_output_active=agent_output_active,
                interrupt_window_active=interrupt_window_active,
                decision_suppressed=decision_suppressed,
                attention_allowed=attention_allowed,
            )
            if semantic_gate.should_forward_and_stop:
                self._forward_to_base(event)
                return

        if semantic_gate.should_run:
            self._run_semantic_interrupt(
                transcript_event.transcript,
                transcript_event.is_final,
            )

        self._forward_to_base(event)

    def _record_admission_event(self, admission: Any) -> None:
        if self._record_transcript_admission_event is None:
            return
        self._record_transcript_admission_event(
            {
                "accepted": bool(admission.accepted),
                "reason": admission.reason,
                "transcript_preview": admission.transcript[:120],
                "text_length": len(admission.transcript),
                "is_final": admission.is_final,
                "speaker_id": admission.speaker_id,
            }
        )

    def _record_ingress_event(
        self,
        transcript_event: FullDuplexTranscriptEvent,
        raw_event: Any,
    ) -> None:
        if self._record_transcript_ingress_event is None:
            return
        self._record_transcript_ingress_event(
            {
                "transcript_preview": transcript_event.transcript[:120],
                "text_length": len(transcript_event.transcript),
                "is_final": transcript_event.is_final,
                "speaker_id": transcript_event.speaker_id,
                "event_type": type(raw_event).__name__,
            }
        )

    def _record_gate_event(
        self,
        transcript_event: FullDuplexTranscriptEvent,
        semantic_gate: Any,
        *,
        stage: str,
        allow_interruptions: bool,
        native_adaptive_owner: bool,
        agent_output_active: bool,
        interrupt_window_active: bool,
        decision_suppressed: bool,
        attention_allowed: bool | None = None,
    ) -> None:
        if self._record_semantic_gate_event is None:
            return
        payload: dict[str, Any] = {
            "stage": stage,
            "action": semantic_gate.action,
            "reason": semantic_gate.reason,
            "transcript_preview": transcript_event.transcript[:120],
            "text_length": len(transcript_event.transcript),
            "is_final": transcript_event.is_final,
            "speaker_id": transcript_event.speaker_id,
            "allow_interruptions": allow_interruptions,
            "native_adaptive_owner": native_adaptive_owner,
            "agent_output_active": agent_output_active,
            "interrupt_window_active": interrupt_window_active,
            "decision_suppressed": decision_suppressed,
        }
        if attention_allowed is not None:
            payload["attention_allowed"] = attention_allowed
        self._record_semantic_gate_event(payload)
