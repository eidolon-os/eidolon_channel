"""Fallback wiring for focused tests that bypass ``StreamingPipeline.__init__``."""

from __future__ import annotations

from typing import Any

from eidolon_sdk.biz.contracts import INTERACTION_MODE_FULL_DUPLEX

from eidolon.livekit.common.config import (
    ObservabilityConfig,
    TurnPolicyConfig,
)

from ..shared.types import PipelineCallbacks, PipelineState
from ..observability import ChannelTurnEventSink
from ..session.agent_output_coordinator import AgentOutputCoordinator
from .transcript_ingress_ledger import FullDuplexTranscriptIngressLedger
from .transcript_evidence_buffer import TranscriptEvidenceBuffer
from ..turn_policy import TurnPolicyRuntime


def ensure_full_duplex_runtime_defaults(pipeline: Any) -> None:
    """Ensure runtime helpers exist on test-built full-duplex pipeline objects.

    Production construction still lives in ``StreamingPipeline.__init__``. This
    function exists for focused unit tests that instantiate the class via
    ``__new__`` to avoid LiveKit setup while exercising full-duplex owners.
    """

    if not hasattr(pipeline, "_turn_policy"):
        pipeline._turn_policy = TurnPolicyConfig()
    if not hasattr(pipeline, "_turn_runtime"):
        pipeline._turn_runtime = TurnPolicyRuntime(pipeline._turn_policy)
    if not hasattr(pipeline, "_stt_commit_transcript_timeout"):
        pipeline._stt_commit_transcript_timeout = (
            pipeline._turn_policy.interrupt.stt_commit_transcript_timeout_ms / 1000.0
        )
    if not hasattr(pipeline, "_callbacks"):
        pipeline._callbacks = PipelineCallbacks()
    if not hasattr(pipeline, "_allow_interruptions"):
        pipeline._allow_interruptions = True
    if not hasattr(pipeline, "_state"):
        pipeline._state = PipelineState.IDLE
    if not hasattr(pipeline, "_observability"):
        pipeline._observability = ObservabilityConfig()
    if not hasattr(pipeline, "_turn_events"):
        pipeline._turn_events = ChannelTurnEventSink(
            observer=pipeline._observe_channel_event
        )
    if not hasattr(pipeline, "_timeline"):
        pipeline._timeline = None
    if not hasattr(pipeline, "_timeline_debug_flushed"):
        pipeline._timeline_debug_flushed = False
    if not hasattr(pipeline, "_flushed_timeline_ids"):
        pipeline._flushed_timeline_ids = set()
    if not hasattr(pipeline, "_agent_output"):
        pipeline._agent_output = AgentOutputCoordinator()
    if not hasattr(pipeline, "_pending_client_control_events"):
        pipeline._pending_client_control_events = []
    if not hasattr(pipeline, "_full_duplex_irreversible_side_effects"):
        pipeline._full_duplex_irreversible_side_effects = set()
    if not hasattr(pipeline, "_skip_commit_after_interrupt_cancel"):
        pipeline._skip_commit_after_interrupt_cancel = False
    if not hasattr(pipeline, "_suppress_commit_after_interrupt_until"):
        pipeline._suppress_commit_after_interrupt_until = 0.0
    if not hasattr(pipeline, "_latest_asr_text"):
        pipeline._latest_asr_text = ""
    pipeline._ensure_full_duplex_state_machine()
    pipeline._ensure_user_turn_coordinator()
    pipeline._ensure_turn_completion()
    if not hasattr(pipeline, "_interaction_mode"):
        pipeline._interaction_mode = INTERACTION_MODE_FULL_DUPLEX
    if not hasattr(pipeline, "_suppress_transcripts_until_next_speech"):
        pipeline._suppress_transcripts_until_next_speech = False
    pipeline._ensure_transcript_admission_gate()
    if not hasattr(pipeline, "_transcript_ingress_ledger"):
        pipeline._transcript_ingress_ledger = FullDuplexTranscriptIngressLedger()
    if not hasattr(pipeline, "_transcript_evidence_buffer"):
        pipeline._transcript_evidence_buffer = TranscriptEvidenceBuffer()
    pipeline._ensure_ducking_controller()
    pipeline._ensure_output_flow()
    pipeline._ensure_interruption_effects()
    pipeline._ensure_decision_effect_applier()
    pipeline._ensure_explicit_client_preempt_ledger()
    pipeline._ensure_interruption_orchestrator()
    pipeline._ensure_attention_effect_handler()
    pipeline._ensure_session_signal_bridge()
    pipeline._ensure_client_audio_state_view()
    pipeline._ensure_client_preempt_handler()
    pipeline._ensure_room_data_bridge()
    pipeline._ensure_agent_state_effect_handler()
    pipeline._ensure_semantic_interrupt_handler()
    pipeline._ensure_duck_suspend_timeout_handler()
    pipeline._ensure_room_data_handler()
