"""AttentionAdmission policy tests."""

from __future__ import annotations

import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

from eidolon.livekit.agent.integration.client_audio_state import ClientAudioState
from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.output.ducking import OutputDuckingController
from eidolon.livekit.agent.pipeline.types import PipelineState
from eidolon.livekit.agent.full_duplex import StreamingPipeline
from eidolon.livekit.agent.session.room_data import RoomDataHandler
from eidolon.livekit.agent.turn_policy import (
    AdmissionAction,
    AttentionAdmission,
    AttentionInput,
    TurnPolicyRuntime,
)
from eidolon.livekit.common.config import (
    AttentionPolicyConfig,
    InterruptPolicyConfig,
    TurnPolicyConfig,
)


def _client_state(**kwargs) -> ClientAudioState:
    base = {
        "participant_identity": "alice",
        "input_mode": "auto",
        "playback_state": "agent_speaking",
        "received_at": time.monotonic(),
    }
    base.update(kwargs)
    return ClientAudioState(**base)


def test_attention_preserves_existing_path_without_client_state() -> None:
    admission = AttentionAdmission(TurnPolicyConfig())

    decision = admission.decide(
        AttentionInput(agent_speaking=True, client_state=None, transcript="")
    )

    assert decision.action is AdmissionAction.DUCK_AND_DECIDE
    assert decision.reason == "no_client_state"


def test_attention_observes_substantive_overlap_during_playback_without_eot() -> None:
    admission = AttentionAdmission(TurnPolicyConfig())

    decision = admission.decide(
        AttentionInput(
            agent_speaking=True,
            client_state=_client_state(),
            transcript="那它的主要风险是什么",
        )
    )

    assert decision.action is AdmissionAction.OBSERVE
    assert decision.reason == "playback_low_evidence_transcript:substantive_cjk_transcript"


def test_attention_soft_ducks_playback_speech_start_when_configured() -> None:
    policy = replace(
        TurnPolicyConfig(),
        attention=replace(
            AttentionPolicyConfig(),
            soft_duck_on_playback_speech_start=True,
        ),
    )
    admission = AttentionAdmission(policy)

    decision = admission.decide(
        AttentionInput(
            agent_speaking=True,
            client_state=_client_state(),
            transcript="",
            speech_started=True,
        )
    )

    assert decision.action is AdmissionAction.DUCK_AND_DECIDE
    assert decision.reason == "playback_speech_start_soft_duck"


def test_attention_soft_ducks_playback_speech_start_by_default() -> None:
    admission = AttentionAdmission(TurnPolicyConfig())

    decision = admission.decide(
        AttentionInput(
            agent_speaking=True,
            client_state=_client_state(),
            transcript="",
            speech_started=True,
        )
    )

    assert decision.action is AdmissionAction.DUCK_AND_DECIDE
    assert decision.reason == "playback_speech_start_soft_duck"


def test_attention_keeps_blank_transcript_observe_without_speech_start() -> None:
    policy = replace(
        TurnPolicyConfig(),
        attention=replace(
            AttentionPolicyConfig(),
            soft_duck_on_playback_speech_start=True,
        ),
    )
    admission = AttentionAdmission(policy)

    decision = admission.decide(
        AttentionInput(
            agent_speaking=True,
            client_state=_client_state(),
            transcript="",
        )
    )

    assert decision.action is AdmissionAction.OBSERVE
    assert decision.reason == "playback_empty_transcript"


def test_attention_allows_high_eot_overlap_during_playback() -> None:
    admission = AttentionAdmission(TurnPolicyConfig())

    decision = admission.decide(
        AttentionInput(
            agent_speaking=True,
            client_state=_client_state(),
            transcript="那它的主要风险是什么",
            eot_score=0.82,
        )
    )

    assert decision.action is AdmissionAction.DUCK_AND_DECIDE
    assert decision.reason == "transcript_evidence:high_eot_transcript"


def test_attention_hard_stop_upgrades_during_playback() -> None:
    admission = AttentionAdmission(TurnPolicyConfig())

    decision = admission.decide(
        AttentionInput(
            agent_speaking=True,
            client_state=_client_state(),
            transcript="别说了",
        )
    )

    assert decision.action is AdmissionAction.HARD_INTERRUPT
    assert decision.reason == "transcript_hard_stop"


def test_attention_hard_stop_homophone_upgrades_during_playback() -> None:
    admission = AttentionAdmission(TurnPolicyConfig())

    decision = admission.decide(
        AttentionInput(
            agent_speaking=True,
            client_state=_client_state(),
            transcript="亭",
        )
    )

    assert decision.action is AdmissionAction.HARD_INTERRUPT
    assert decision.reason == "transcript_hard_stop"


def test_attention_allows_topic_switch_intent_during_playback() -> None:
    admission = AttentionAdmission(
        replace(
            TurnPolicyConfig(),
            interrupt=replace(InterruptPolicyConfig(), fast_lexical_intents=True),
        )
    )

    decision = admission.decide(
        AttentionInput(
            agent_speaking=True,
            client_state=_client_state(),
            transcript="我们聊点别的。",
        )
    )

    assert decision.action is AdmissionAction.DUCK_AND_DECIDE
    assert decision.reason == "transcript_intent:topic_switch"


def test_attention_allows_correction_intent_during_playback() -> None:
    admission = AttentionAdmission(
        replace(
            TurnPolicyConfig(),
            interrupt=replace(InterruptPolicyConfig(), fast_lexical_intents=True),
        )
    )

    decision = admission.decide(
        AttentionInput(
            agent_speaking=True,
            client_state=_client_state(),
            transcript="不是。",
        )
    )

    assert decision.action is AdmissionAction.DUCK_AND_DECIDE
    assert decision.reason == "transcript_intent:correction"


def test_attention_observes_short_prefix_during_playback() -> None:
    admission = AttentionAdmission(TurnPolicyConfig())

    decision = admission.decide(
        AttentionInput(
            agent_speaking=True,
            client_state=_client_state(),
            transcript="换个",
        )
    )

    assert decision.action is AdmissionAction.OBSERVE
    assert decision.reason == "playback_low_evidence_transcript:insufficient_transcript_evidence"


def test_attention_observes_single_char_prefix_during_playback() -> None:
    admission = AttentionAdmission(TurnPolicyConfig())

    decision = admission.decide(
        AttentionInput(
            agent_speaking=True,
            client_state=_client_state(),
            transcript="换",
        )
    )

    assert decision.action is AdmissionAction.OBSERVE
    assert decision.reason == "playback_low_evidence_transcript:insufficient_transcript_evidence"


def test_attention_ptt_is_hard_interrupt() -> None:
    # PTT (deliberate button) stays an immediate hard cut.
    admission = AttentionAdmission(TurnPolicyConfig())

    decision = admission.decide(
        AttentionInput(
            agent_speaking=True,
            client_state=_client_state(ptt=True),
        )
    )

    assert decision.action is AdmissionAction.HARD_INTERRUPT
    assert decision.reason == "explicit_client_ptt"


def test_attention_manual_interrupt_without_evidence_does_not_hard_cut() -> None:
    # P1: manual_interrupt is an unreliable energy-gate signal (echo trips it).
    # With no transcript evidence it must NOT hard-cut — it falls through to the
    # evidence-gated path (observe / duck-and-decide), so echo can't truncate.
    admission = AttentionAdmission(TurnPolicyConfig())

    decision = admission.decide(
        AttentionInput(
            agent_speaking=True,
            client_state=_client_state(manual_interrupt=True),
        )
    )

    assert decision.action is not AdmissionAction.HARD_INTERRUPT


def _turn_policy(
    *,
    enforce: bool,
    soft_duck_on_playback_speech_start: bool = False,
) -> TurnPolicyConfig:
    return replace(
        TurnPolicyConfig(),
        attention=replace(
            AttentionPolicyConfig(),
            enforce=enforce,
            soft_duck_on_playback_speech_start=soft_duck_on_playback_speech_start,
        ),
    )


def _pipeline_with_client_state(
    state: ClientAudioState | None,
    *,
    enforce: bool = True,
    pipeline_state: PipelineState = PipelineState.SPEAKING,
    soft_duck_on_playback_speech_start: bool = False,
) -> StreamingPipeline:
    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._turn_policy = _turn_policy(
        enforce=enforce,
        soft_duck_on_playback_speech_start=soft_duck_on_playback_speech_start,
    )
    pipeline._turn_runtime = TurnPolicyRuntime(pipeline._turn_policy)
    pipeline._state = pipeline_state
    pipeline._ducking = OutputDuckingController()
    pipeline._ducking.mixer = None
    pipeline._timeline = TurnTimeline("turn-1")
    pipeline._room_data = RoomDataHandler(get_timeline=lambda: pipeline._timeline)
    if state is not None:
        pipeline._room_data.client_audio_states[state.participant_identity] = state
    pipeline._output_flow = SimpleNamespace(duck_and_arm_timeout=MagicMock())
    pipeline._callbacks = MagicMock()
    return pipeline


def _allows_eot(
    pipeline: StreamingPipeline,
    transcript: str,
    *,
    speaker_id: str | None = None,
) -> bool:
    pipeline._ensure_runtime_defaults()
    return pipeline._attention_effects.allows_eot_check(
        transcript,
        speaker_id=speaker_id,
    )


def test_pipeline_attention_observes_substantive_playback_speech_without_eot() -> None:
    pipeline = _pipeline_with_client_state(_client_state())

    allowed = _allows_eot(pipeline, "那它的主要风险是什么")

    assert allowed is False
    pipeline._output_flow.duck_and_arm_timeout.assert_not_called()
    assert pipeline._timeline.attrs["attention_admission"]["action"] == "observe"


def test_pipeline_attention_soft_ducks_on_playback_speech_start() -> None:
    pipeline = _pipeline_with_client_state(
        _client_state(),
        soft_duck_on_playback_speech_start=True,
    )

    pipeline._ensure_runtime_defaults()
    pipeline._attention_effects.handle_speaking_started()

    pipeline._output_flow.duck_and_arm_timeout.assert_called_once()
    assert (
        pipeline._timeline.attrs["attention_admission"]["action"]
        == "duck_and_decide"
    )
    assert (
        pipeline._timeline.attrs["attention_admission"]["reason"]
        == "playback_speech_start_soft_duck"
    )


def test_pipeline_attention_routes_low_evidence_transcript_after_soft_duck() -> None:
    pipeline = _pipeline_with_client_state(
        _client_state(),
        soft_duck_on_playback_speech_start=True,
    )

    pipeline._ensure_runtime_defaults()
    pipeline._attention_effects.handle_speaking_started()
    pipeline._ducking.mixer = SimpleNamespace(state="SUSPENDED")
    pipeline._output_flow.duck_and_arm_timeout.reset_mock()
    allowed = _allows_eot(pipeline, "嗯嗯")

    assert allowed is True
    pipeline._output_flow.duck_and_arm_timeout.assert_not_called()
    assert pipeline._timeline.attrs["attention_admission"]["action"] == "observe"


def test_pipeline_attention_allows_high_eot_playback_speech() -> None:
    pipeline = _pipeline_with_client_state(_client_state())
    eot = MagicMock()
    eot.current_eot_score = 0.82
    pipeline._get_eot_model = MagicMock(return_value=eot)

    allowed = _allows_eot(pipeline, "那它的主要风险是什么")

    assert allowed is True
    pipeline._output_flow.duck_and_arm_timeout.assert_called_once()
    assert pipeline._timeline.attrs["attention_admission"]["action"] == "duck_and_decide"
    assert (
        pipeline._timeline.attrs["attention_admission"]["reason"]
        == "transcript_evidence:high_eot_transcript"
    )


def test_pipeline_attention_allows_hard_stop_during_playback() -> None:
    pipeline = _pipeline_with_client_state(_client_state())

    allowed = _allows_eot(pipeline, "别说了")

    assert allowed is True
    pipeline._output_flow.duck_and_arm_timeout.assert_not_called()
    assert pipeline._timeline.attrs["attention_admission"]["action"] == "hard_interrupt"


def test_pipeline_attention_allows_topic_switch_intent_during_playback() -> None:
    pipeline = _pipeline_with_client_state(_client_state())
    pipeline._turn_policy = replace(
        pipeline._turn_policy,
        interrupt=replace(
            pipeline._turn_policy.interrupt,
            fast_lexical_intents=True,
        ),
    )
    pipeline._turn_runtime = TurnPolicyRuntime(pipeline._turn_policy)

    allowed = _allows_eot(pipeline, "我们聊点别的。")

    assert allowed is True
    pipeline._output_flow.duck_and_arm_timeout.assert_called_once()
    assert pipeline._timeline.attrs["attention_admission"]["action"] == "duck_and_decide"
    assert (
        pipeline._timeline.attrs["attention_admission"]["reason"]
        == "transcript_intent:topic_switch"
    )


def test_pipeline_attention_uses_client_playback_when_internal_state_idle() -> None:
    pipeline = _pipeline_with_client_state(
        _client_state(),
        pipeline_state=PipelineState.IDLE,
    )

    allowed = _allows_eot(pipeline, "停一下")

    assert allowed is True
    pipeline._output_flow.duck_and_arm_timeout.assert_not_called()
    assert pipeline._timeline.attrs["attention_admission"]["action"] == "hard_interrupt"


def test_user_transcript_runs_semantic_when_client_playback_active_but_state_idle() -> None:
    pipeline = _pipeline_with_client_state(
        _client_state(participant_identity="manson"),
        pipeline_state=PipelineState.IDLE,
    )
    eot = MagicMock()
    eot.update_asr = MagicMock()
    pipeline._get_eot_model = MagicMock(return_value=eot)
    pipeline._semantic_interrupts = SimpleNamespace(
        run=MagicMock(),
        _turn_runtime=pipeline._turn_runtime,
    )
    pipeline._allow_interruptions = True
    pipeline._mark_activity = MagicMock()

    pipeline._on_user_transcribed(
        SimpleNamespace(
            transcript="停一下",
            is_final=False,
            speaker_id="manson",
        )
    )

    pipeline._semantic_interrupts.run.assert_called_once_with(
        "停一下",
        is_final=False,
    )


def test_user_transcript_suppresses_semantic_during_post_cancel_residual_window() -> None:
    pipeline = _pipeline_with_client_state(
        _client_state(participant_identity="manson"),
        pipeline_state=PipelineState.IDLE,
    )
    eot = MagicMock()
    eot.update_asr = MagicMock()
    pipeline._get_eot_model = MagicMock(return_value=eot)
    pipeline._semantic_interrupts = SimpleNamespace(
        run=MagicMock(),
        _turn_runtime=pipeline._turn_runtime,
    )
    pipeline._allow_interruptions = True
    pipeline._mark_activity = MagicMock()
    pipeline._suppress_commit_after_interrupt_until = time.monotonic() + 1.0

    pipeline._on_user_transcribed(
        SimpleNamespace(
            transcript="换个话题，我们聊点别的",
            is_final=False,
            speaker_id="manson",
        )
    )

    eot.update_asr.assert_called_once_with(
        "换个话题，我们聊点别的",
        is_final=False,
    )
    pipeline._semantic_interrupts.run.assert_not_called()


def test_user_transcript_ignores_stale_client_playback_after_output_cancelled() -> None:
    pipeline = _pipeline_with_client_state(
        _client_state(participant_identity="manson"),
        pipeline_state=PipelineState.IDLE,
    )
    pipeline._ensure_runtime_defaults()
    pipeline._ducking.mixer = SimpleNamespace(state="CANCELLED")
    eot = MagicMock()
    eot.update_asr = MagicMock()
    pipeline._get_eot_model = MagicMock(return_value=eot)
    pipeline._semantic_interrupts = SimpleNamespace(
        run=MagicMock(),
        _turn_runtime=pipeline._turn_runtime,
    )
    pipeline._allow_interruptions = True
    pipeline._mark_activity = MagicMock()

    pipeline._on_user_transcribed(
        SimpleNamespace(
            transcript="我们聊点别的。",
            is_final=False,
            speaker_id="manson",
        )
    )

    eot.update_asr.assert_called_once_with(
        "我们聊点别的。",
        is_final=False,
    )
    pipeline._semantic_interrupts.run.assert_not_called()


def test_user_transcript_does_not_use_timeline_marker_as_interrupt_window() -> None:
    pipeline = _pipeline_with_client_state(None, pipeline_state=PipelineState.IDLE)
    pipeline._ensure_runtime_defaults()
    pipeline._timeline.mark("interrupt_started_at")
    eot = MagicMock()
    eot.update_asr = MagicMock()
    pipeline._get_eot_model = MagicMock(return_value=eot)
    pipeline._semantic_interrupts = SimpleNamespace(
        run=MagicMock(),
        _turn_runtime=pipeline._turn_runtime,
    )
    pipeline._allow_interruptions = True
    pipeline._mark_activity = MagicMock()

    pipeline._on_user_transcribed(
        SimpleNamespace(
            transcript="我们聊点别的。",
            is_final=False,
            speaker_id="manson",
        )
    )

    eot.update_asr.assert_called_once_with(
        "我们聊点别的。",
        is_final=False,
    )
    pipeline._semantic_interrupts.run.assert_not_called()


def test_pipeline_attention_observes_short_prefix_during_playback() -> None:
    pipeline = _pipeline_with_client_state(_client_state())

    allowed = _allows_eot(pipeline, "换个")

    assert allowed is False
    pipeline._output_flow.duck_and_arm_timeout.assert_not_called()
    assert pipeline._timeline.attrs["attention_admission"]["action"] == "observe"
    assert (
        pipeline._timeline.attrs["attention_admission"]["reason"]
        == "playback_low_evidence_transcript:insufficient_transcript_evidence"
    )


def test_pipeline_attention_observes_single_char_prefix_during_playback() -> None:
    pipeline = _pipeline_with_client_state(_client_state())

    allowed = _allows_eot(pipeline, "换")

    assert allowed is False
    pipeline._output_flow.duck_and_arm_timeout.assert_not_called()
    assert pipeline._timeline.attrs["attention_admission"]["action"] == "observe"
    assert (
        pipeline._timeline.attrs["attention_admission"]["reason"]
        == "playback_low_evidence_transcript:insufficient_transcript_evidence"
    )


def test_pipeline_attention_preserves_old_path_without_client_state() -> None:
    pipeline = _pipeline_with_client_state(None)

    allowed = _allows_eot(pipeline, "那它的主要风险是什么")

    assert allowed is True
    pipeline._output_flow.duck_and_arm_timeout.assert_called_once()
    assert pipeline._timeline.attrs["attention_admission"]["action"] == "duck_and_decide"


def test_pipeline_attention_defaults_to_observe_only_rollout() -> None:
    pipeline = _pipeline_with_client_state(_client_state(), enforce=False)

    allowed = _allows_eot(pipeline, "那它的主要风险是什么")

    assert allowed is True
    pipeline._output_flow.duck_and_arm_timeout.assert_not_called()
    assert pipeline._timeline.attrs["attention_admission"]["action"] == "observe"
    assert pipeline._timeline.attrs["attention_admission"]["enforced"] is False


def test_pipeline_attention_records_decision_history() -> None:
    pipeline = _pipeline_with_client_state(_client_state())

    _allows_eot(pipeline, "那它的主要风险是什么")
    _allows_eot(pipeline, "别说了")

    events = pipeline._timeline.attrs["attention_admission_events"]
    assert [event["action"] for event in events] == ["observe", "hard_interrupt"]
    assert pipeline._timeline.attrs["attention_admission"]["action"] == "hard_interrupt"


def test_pipeline_rejects_playback_low_evidence_artifact_before_commit() -> None:
    pipeline = _pipeline_with_client_state(_client_state())
    pipeline._timeline.set_attr(
        "attention_admission_events",
        [
            {
                "action": "observe",
                "reason": (
                    "playback_low_evidence_transcript:"
                    "insufficient_transcript_evidence"
                ),
            }
        ],
    )
    eot = MagicMock()
    eot.current_eot_score = 0.0

    reason = pipeline._ensure_turn_completion().playback_low_evidence_reject_reason(
        transcript="所",
        eot_model=eot,
    )

    assert reason == (
        "playback_low_evidence_artifact:insufficient_transcript_evidence"
    )


def test_confirmed_redirect_bypasses_playback_low_evidence_reject() -> None:
    pipeline = _pipeline_with_client_state(_client_state())
    pipeline._timeline.set_attr(
        "decision",
        {
            "action": "cancel",
            "reason": "intent:correction",
            "intent": "normal_interrupt",
            "correction_hint": True,
            "topic_switch_hint": False,
        },
    )
    pipeline._timeline.set_attr(
        "attention_admission_events",
        [
            {
                "action": "observe",
                "reason": (
                    "playback_low_evidence_transcript:"
                    "insufficient_transcript_evidence"
                ),
            }
        ],
    )
    eot = MagicMock()
    eot.current_eot_score = 0.0

    reason = pipeline._ensure_turn_completion().playback_low_evidence_reject_reason(
        transcript="不是",
        eot_model=eot,
    )

    assert reason == ""


def test_pipeline_attention_prefers_speaker_client_state() -> None:
    now = time.monotonic()
    alice = _client_state(
        participant_identity="alice",
        playback_state="agent_speaking",
        received_at=now,
    )
    bob = _client_state(
        participant_identity="bob",
        playback_state="idle",
        received_at=now - 0.2,
    )
    pipeline = _pipeline_with_client_state(alice)
    pipeline._room_data.client_audio_states[bob.participant_identity] = bob

    allowed = _allows_eot(
        pipeline,
        "那它的主要风险是什么",
        speaker_id="bob",
    )

    assert allowed is False
    pipeline._output_flow.duck_and_arm_timeout.assert_not_called()
    assert (
        pipeline._timeline.attrs["attention_admission"]["reason"]
        == "playback_low_evidence_transcript:substantive_cjk_transcript"
    )
