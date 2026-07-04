"""Fallback wiring for focused tests that bypass ``StreamingPipeline.__init__``."""

from __future__ import annotations

from typing import Any

from eidolon_sdk.biz.contracts import INTERACTION_MODE_FULL_DUPLEX

from eidolon.livekit.common.config import (
    ObservabilityConfig,
    TurnPolicyConfig,
    VoiceprintConfig,
)

from ..shared.types import PipelineCallbacks, PipelineState
from ..session.voiceprint import VoiceprintTurnObserver
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
    if not hasattr(pipeline, "_callbacks"):
        pipeline._callbacks = PipelineCallbacks()
    if not hasattr(pipeline, "_allow_interruptions"):
        pipeline._allow_interruptions = True
    if not hasattr(pipeline, "_state"):
        pipeline._state = PipelineState.IDLE
    if not hasattr(pipeline, "_observability"):
        pipeline._observability = ObservabilityConfig()
    if not hasattr(pipeline, "_voiceprint_config"):
        pipeline._voiceprint_config = VoiceprintConfig()
    if not hasattr(pipeline, "_timeline"):
        pipeline._timeline = None
    if not hasattr(pipeline, "_timeline_debug_flushed"):
        pipeline._timeline_debug_flushed = False
    if not hasattr(pipeline, "_pending_client_control_events"):
        pipeline._pending_client_control_events = []
    if not hasattr(pipeline, "_skip_commit_after_interrupt_cancel"):
        pipeline._skip_commit_after_interrupt_cancel = False
    if not hasattr(pipeline, "_suppress_commit_after_interrupt_until"):
        pipeline._suppress_commit_after_interrupt_until = 0.0
    if not hasattr(pipeline, "_latest_asr_text"):
        pipeline._latest_asr_text = ""
    if not hasattr(pipeline, "_pending_voiceprint_commit_tasks"):
        pipeline._pending_voiceprint_commit_tasks = set()
    if not hasattr(pipeline, "_candidate_voiceprint_tasks"):
        pipeline._candidate_voiceprint_tasks = []
    if not hasattr(pipeline, "_deferred_low_eot_commit_task"):
        pipeline._deferred_low_eot_commit_task = None
    pipeline._ensure_user_turn_coordinator()
    pipeline._ensure_turn_completion()
    if not hasattr(pipeline, "_interaction_mode"):
        pipeline._interaction_mode = INTERACTION_MODE_FULL_DUPLEX
    if not hasattr(pipeline, "_suppress_transcripts_until_next_speech"):
        pipeline._suppress_transcripts_until_next_speech = False
    pipeline._ensure_transcript_admission_gate()
    if not hasattr(pipeline, "_completed_turn_voiceprint_task"):
        pipeline._completed_turn_voiceprint_task = None
    if not hasattr(pipeline, "_completed_turn_voiceprint_result"):
        pipeline._completed_turn_voiceprint_result = None
    if not hasattr(pipeline, "_completed_turn_voiceprint_timeline"):
        pipeline._completed_turn_voiceprint_timeline = None
    if not hasattr(pipeline, "_voiceprint_turns"):
        factory = getattr(pipeline, "_factory", None)
        pipeline._voiceprint_turns = VoiceprintTurnObserver(
            service=getattr(factory, "voiceprint_service", None),
            runtime_admin=getattr(factory, "runtime_admin", None),
            sample_rate=getattr(pipeline, "_audio_sample_rate", 16000),
            max_audio_ms=pipeline._voiceprint_config.turn_max_audio_ms,
            accept_cache_ttl_sec=(pipeline._voiceprint_config.accept_cache_ttl_ms / 1000.0),
            accept_cache_short_audio_max_ms=(
                pipeline._voiceprint_config.accept_cache_short_audio_max_ms
            ),
            commit_threshold=pipeline._voiceprint_config.owner_commit_threshold,
            owner_short_audio_bypass_ms=(
                pipeline._voiceprint_config.owner_short_audio_bypass_ms
            ),
            trust_paired_devices=getattr(
                factory,
                "voiceprint_trust_paired_devices",
                True,
            ),
        )
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
    pipeline._ensure_turn_committer()
    pipeline._ensure_agent_state_effect_handler()
    pipeline._ensure_semantic_interrupt_handler()
    pipeline._ensure_duck_suspend_timeout_handler()
    pipeline._ensure_room_data_handler()
