"""Voice benchmark infrastructure tests."""

from __future__ import annotations

import asyncio
import contextlib
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import jwt
import pytest

from benchmark.compare import compare_metrics
from benchmark.dashboard import DashboardRunner, write_dashboard
from benchmark.realcall import (
    apply_real_call_verification,
    preflight_runtime_identity,
    verify_provider_config,
    verify_real_call,
)
from benchmark.report import (
    aggregate,
    aggregate_runs,
    write_repeated_reports,
)
from benchmark.schema import CaseResult, RunResult, load_suite
from benchmark.livekit_room_runner import (
    LiveKitRoomOptions,
    _agent_audio_wait_mode,
    _agent_audio_wait_timeout_sec,
    _transcription_role,
)
from benchmark.device_envelope import (
    audio_state_interval_sec,
    render_device_envelope_mic_pcm,
)
from benchmark.hil_barge_in import analyze_hil_barge_in
from eidolon.livekit.tests._harness.audio import pcm_rms
from benchmark.policy_runner import run_policy_suite
from benchmark.slo import (
    DEFAULT_SLO_GATES,
    SloGate,
    enforcement_failures,
    evaluate_slo_gates,
)
from eidolon.livekit.common.config import AttentionPolicyConfig, TurnPolicyConfig
from benchmark.timeline import (
    TimelineCapture,
    load_timeline_records,
    summarize_timeline_records,
)
from benchmark.timeline_expectations import apply_timeline_expectations
from scripts.bench_voice import _default_cases, _participant_metadata as _bench_participant_metadata
from scripts.bench_barge_in_ab import DEFAULT_CASES as DEFAULT_BARGE_IN_AB_CASES


def test_load_core_benchmark_suite() -> None:
    suite = load_suite("benchmark/cases/core.yaml")

    assert suite.suite_id == "core_voice_baseline"
    assert {case.case_id for case in suite.cases} >= {
        "normal_single_turn_001",
        "hard_interrupt_001",
        "backchannel_001",
    }
    hard = next(case for case in suite.cases if case.case_id == "hard_interrupt_001")
    assert hard.expectations.max_interrupt_decision_ms == 650
    topic = next(case for case in suite.cases if case.case_id == "topic_switch_001")
    assert topic.expectations.max_interrupt_resolution_after_started_ms == 250
    backchannel = next(case for case in suite.cases if case.case_id == "backchannel_001")
    assert "observe" in backchannel.expectations.allow_attention_actions


def test_load_attention_admission_benchmark_suite() -> None:
    suite = load_suite("benchmark/cases/attention_admission_baseline.yaml")

    assert suite.suite_id == "attention_admission_baseline"
    assert {case.case_id for case in suite.cases} >= {
        "owner_hard_stop_while_agent_speaking_001",
        "ambient_normal_speech_currently_interrupts_001",
        "normal_user_turn_when_agent_idle_001",
        "cough_noise_then_ambient_speech_holds_001",
    }
    ambient_case = next(
        case
        for case in suite.cases
        if case.case_id == "ambient_normal_speech_currently_interrupts_001"
    )
    assert ambient_case.expectations.action == "cancel"
    assert "known_gap" in ambient_case.tags
    cough_case = next(
        case for case in suite.cases if case.case_id == "cough_noise_then_ambient_speech_holds_001"
    )
    assert cough_case.expectations.action == "any"
    assert "cancel" in cough_case.expectations.forbid_actions
    assert "known_gap" not in cough_case.tags
    backchannel_case = next(
        case for case in suite.cases if case.case_id == "short_backchannel_rolls_back_001"
    )
    # fast_lexical_intents defaults off after the interrupt_mode axis was retired,
    # so a short backchannel is held (evidence gate) rather than fast-rolled-back.
    assert backchannel_case.expectations.action == "hold"
    assert "cancel" in backchannel_case.expectations.forbid_actions


def test_load_attention_admission_enforced_suite() -> None:
    suite = load_suite("benchmark/cases/attention_admission_enforced.yaml")

    assert suite.suite_id == "attention_admission_enforced"
    assert {case.case_id for case in suite.cases} == {
        "enforced_ambient_playback_speech_does_not_cancel_001",
        "enforced_hard_stop_still_cancels_001",
        "enforced_no_client_state_preserves_legacy_path_001",
    }
    ambient_case = next(
        case
        for case in suite.cases
        if case.case_id == "enforced_ambient_playback_speech_does_not_cancel_001"
    )
    assert ambient_case.user_steps[0].client_playback_state == "agent_speaking"
    assert ambient_case.expectations.action == "any"
    assert ambient_case.expectations.decision_action == "hold"
    assert "cancel" in ambient_case.expectations.forbid_actions


def test_default_voice_benchmark_cases_skip_enforced_suites() -> None:
    cases = [path.rsplit("/", 1)[-1] for path in _default_cases()]

    assert "attention_admission_baseline.yaml" in cases
    assert "attention_admission_enforced.yaml" not in cases
    assert "v1_interrupt_tiers_enforced.yaml" not in cases
    assert "v1_realistic_interaction_flows_enforced.yaml" not in cases
    assert "dogfood_box3_audio_first_enforced.yaml" not in cases


def test_load_dogfood_box3_audio_first_suite() -> None:
    suite = load_suite("benchmark/cases/dogfood_box3_audio_first_enforced.yaml")

    assert suite.suite_id == "dogfood_box3_audio_first_enforced"
    followup = next(
        case
        for case in suite.cases
        if case.case_id == "dogfood_box3_owner_followup_during_playback_001"
    )
    assert followup.device_envelope.enabled is True
    assert followup.device_envelope.device.model == "esp32_box_3"
    assert followup.device_envelope.device.mode == "full_duplex"
    assert followup.device_envelope.device.audio_state_hz == 10
    assert followup.device_envelope.acoustics.echo.enabled is True
    assert followup.expectations.max_speech_start_to_suspend_ms == 120
    assert followup.expectations.playback_stop_sent is True
    assert followup.expectations.agent_audio_response == "after_user_done"
    assert followup.expectations.max_user_done_to_agent_audio_ms is None
    backchannel = next(
        case
        for case in suite.cases
        if case.case_id == "dogfood_box3_backchannel_during_playback_001"
    )
    assert len(backchannel.user_steps) == 2
    assert backchannel.user_steps[0].agent_speaking is False
    assert backchannel.user_steps[1].agent_speaking is True
    assert backchannel.expectations.agent_audio_response == "after_user_done"
    assert backchannel.expectations.min_user_finals == 2
    assert backchannel.expectations.min_agent_messages == 1


def test_load_offline_policy_regression_suite() -> None:
    suite = load_suite("benchmark/cases/offline_policy_regression_enforced.yaml")
    cases = {case.case_id: case for case in suite.cases}

    assert suite.suite_id == "offline_policy_regression_enforced"
    assert set(cases) == {
        "opr_waveshare_ptt_idle_tap_no_policy_decision_001",
        "opr_waveshare_ptt_playback_tap_cancels_001",
        "opr_fullduplex_compound_backchannel_holds_001",
        "opr_fullduplex_false_start_holds_001",
        "opr_fullduplex_echo_like_agent_words_holds_001",
        "opr_fullduplex_followup_after_low_prefix_cancels_001",
        "opr_fullduplex_backchannel_then_followup_cancels_001",
        "opr_fullduplex_false_start_then_followup_cancels_001",
        "opr_fullduplex_echo_then_followup_cancels_001",
        "opr_fullduplex_echo_then_hard_stop_cancels_001",
    }

    idle_tap = cases["opr_waveshare_ptt_idle_tap_no_policy_decision_001"]
    assert idle_tap.device_envelope.device.mode == "ptt"
    assert idle_tap.user_steps[0].client_ptt is True
    assert idle_tap.user_steps[0].agent_speaking is False
    assert idle_tap.expectations.decision_action == "none"

    playback_tap = cases["opr_waveshare_ptt_playback_tap_cancels_001"]
    assert playback_tap.device_envelope.device.model == "waveshare_esp32_s3_touch_amoled_2_06"
    assert playback_tap.expectations.decision_action == "cancel"
    assert playback_tap.expectations.decision_intent == "hard_stop"

    compound_backchannel = cases["opr_fullduplex_compound_backchannel_holds_001"]
    assert compound_backchannel.device_envelope.device.mode == "full_duplex"
    assert compound_backchannel.expectations.decision_action == "hold"
    assert "cancel" in compound_backchannel.expectations.forbid_actions
    assert compound_backchannel.expectations.no_full_assistant_context_commit is True

    followup = cases["opr_fullduplex_followup_after_low_prefix_cancels_001"]
    assert followup.expectations.decision_action == "cancel"
    assert followup.expectations.decision_intent == "normal_interrupt"

    continuity = cases["opr_fullduplex_echo_then_followup_cancels_001"]
    assert len(continuity.user_steps) == 2
    assert continuity.user_steps[0].text == "我会先讲系统结构"
    assert continuity.user_steps[1].text == "那你现在能帮我做什么"
    assert continuity.expectations.decision_action == "cancel"


def test_full_duplex_gate_guard_cases_use_matching_audio_assets() -> None:
    suite = load_suite("benchmark/cases/full_duplex/gate_enforced.yaml")
    cases = {case.case_id: case for case in suite.cases}

    false_start = cases["fd_gate_false_start_holds_001"]
    assert false_start.audio_clips[0].id == "reaction_start"
    assert false_start.audio_clips[0].text == "我觉得"
    assert false_start.audio_clips[0].path.endswith("/reaction_start.wav")

    echo_guard = cases["fd_gate_echo_like_agent_words_holds_001"]
    assert echo_guard.audio_clips[0].id == "welcome_echo_words"
    assert echo_guard.audio_clips[0].text == "我是你的 AI 助手"
    assert echo_guard.audio_clips[0].path.endswith("/welcome_echo_words.wav")
    assert echo_guard.user_steps[0].text == "我是你的 AI 助手"


def test_policy_runner_offline_policy_continuity_cases_continue_after_hold() -> None:
    suite = load_suite("benchmark/cases/offline_policy_regression_enforced.yaml")
    run = run_policy_suite(
        [suite],
        turn_policy=TurnPolicyConfig(
            attention=replace(AttentionPolicyConfig(), enforce=True),
        ),
        run_id="test",
    )
    cases = {case.case_id: case for case in run.cases}

    assert all(case.passed for case in run.cases)
    for case_id in (
        "opr_fullduplex_backchannel_then_followup_cancels_001",
        "opr_fullduplex_false_start_then_followup_cancels_001",
        "opr_fullduplex_echo_then_followup_cancels_001",
    ):
        result = cases[case_id]
        assert result.metrics["actual_action"] == "cancel"
        assert result.metrics["actual_decision_action"] == "cancel"
        assert any(
            (decision.get("decision") or {}).get("action") == "hold"
            for decision in result.decisions
        )


def test_barge_in_ab_default_cases_include_offline_policy_regression_suite() -> None:
    assert (
        "benchmark/cases/shared/offline_policy_regression_enforced.yaml"
        in DEFAULT_BARGE_IN_AB_CASES
    )


def test_device_envelope_audio_state_interval_uses_device_cadence() -> None:
    suite = load_suite("benchmark/cases/dogfood_box3_audio_first_enforced.yaml")
    case = suite.cases[0]

    assert audio_state_interval_sec(case) == pytest.approx(0.1)


def test_device_envelope_mic_render_adds_echo_and_noise() -> None:
    suite = load_suite("benchmark/cases/dogfood_box3_audio_first_enforced.yaml")
    case = suite.cases[0]
    step = case.user_steps[1]
    pcm = b"\x00\x00" * 1600

    rendered = render_device_envelope_mic_pcm(case, step, pcm, sample_rate=16000)

    assert len(rendered) == len(pcm)
    assert pcm_rms(rendered) > pcm_rms(pcm)


def test_livekit_room_device_envelope_input_mode_tracks_device_mode() -> None:
    from benchmark.livekit_room_runner import _case_input_mode, _step_input_mode

    # full_duplex auto-records: input_mode "auto" (NOT ptt) after the 3-mode split.
    suite = load_suite("benchmark/cases/dogfood_box3_audio_first_enforced.yaml")
    assert _case_input_mode(suite.cases[0]) == "auto"

    # A ptt-mode device reports input_mode "ptt" at the case level.
    ptt_mode_suite = load_suite("benchmark/cases/half_duplex_ptt_phase_a_enforced.yaml")
    assert _case_input_mode(ptt_mode_suite.cases[0]) == "ptt"

    # A full_duplex case still reports "ptt" per-step when the step is an
    # explicit client PTT preempt (driven by client_ptt, not device.mode).
    ptt_suite = load_suite("benchmark/cases/full_duplex/explicit_control_enforced.yaml")
    ptt_case = {case.case_id: case for case in ptt_suite.cases}[
        "fd_explicit_ptt_preempts_without_speech_001"
    ]
    assert _case_input_mode(ptt_case) == "auto"
    assert _step_input_mode(ptt_case, ptt_case.user_steps[0]) == "ptt"


def test_half_duplex_ptt_phase_a_suite_is_mode_specific() -> None:
    suite = load_suite("benchmark/cases/half_duplex_ptt_phase_a_enforced.yaml")
    cases = {case.case_id: case for case in suite.cases}

    # push-to-talk is its own interaction_mode; the suite must route to it.
    assert suite.suite_mode == "ptt"

    normal = cases["phase_a_ptt_normal_release_commits_001"]
    assert normal.device_envelope.device.model == "waveshare_esp32_s3_touch_amoled_2_06"
    assert normal.device_envelope.device.mode == "ptt"
    assert normal.user_steps[0].client_ptt is True
    assert normal.expectations.agent_audio_response == "after_user_done"

    tap_to_stop = cases["phase_a_ptt_tap_to_stop_cancels_001"]
    assert tap_to_stop.device_envelope.device.mode == "ptt"
    assert tap_to_stop.expectations.decision_action == "cancel"
    assert tap_to_stop.expectations.playback_stop_sent is True
    assert tap_to_stop.expectations.agent_audio_response == "none"


@pytest.mark.asyncio
async def test_livekit_room_ptt_step_publishes_release_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmark import livekit_room_runner as runner

    suite = load_suite("benchmark/cases/full_duplex/explicit_control_enforced.yaml")
    case = {item.case_id: item for item in suite.cases}[
        "fd_explicit_ptt_preempts_without_speech_001"
    ]
    local_participant = AsyncMock()
    events: list[dict] = []

    async def fake_wait_for_agent_speaking(*_args, **_kwargs) -> bool:
        return True

    async def fake_capture_pcm(*_args, **_kwargs) -> int:
        return 200

    monkeypatch.setattr(runner, "_wait_for_agent_speaking", fake_wait_for_agent_speaking)
    monkeypatch.setattr(runner, "_capture_pcm", fake_capture_pcm)
    monkeypatch.setattr(runner, "load_clip_pcm", lambda *_args, **_kwargs: (b"\0\0" * 160, 16000))
    monkeypatch.setattr(
        runner, "render_device_envelope_mic_pcm", lambda _case, _step, pcm, **_kw: pcm
    )

    await runner._feed_case_audio(
        object(),
        case=case,
        root=Path("."),
        events=events,
        started=0.0,
        state=object(),
        options=LiveKitRoomOptions(),
        local_participant=local_participant,
    )

    payloads = [json.loads(call.args[0]) for call in local_participant.publish_data.await_args_list]
    assert payloads[0]["input_mode"] == "ptt"
    assert payloads[0]["ptt"] is True
    assert payloads[-1]["input_mode"] == "ptt"
    assert payloads[-1]["ptt"] is False
    assert payloads[-1]["mic_muted"] is True


@pytest.mark.asyncio
async def test_livekit_room_agent_speaking_primes_client_state_before_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmark import livekit_room_runner as runner

    suite = load_suite("benchmark/cases/full_duplex/explicit_control_enforced.yaml")
    case = {item.case_id: item for item in suite.cases}[
        "fd_explicit_ptt_preempts_without_speech_001"
    ]
    local_participant = AsyncMock()
    events: list[dict] = []

    async def fake_wait_for_agent_speaking(*_args, **_kwargs) -> bool:
        return True

    async def fake_capture_pcm(*_args, **_kwargs) -> int:
        return 120

    monkeypatch.setattr(runner, "_wait_for_agent_speaking", fake_wait_for_agent_speaking)
    monkeypatch.setattr(runner, "_capture_pcm", fake_capture_pcm)
    monkeypatch.setattr(runner, "load_clip_pcm", lambda *_args, **_kwargs: (b"\0\0" * 160, 16000))
    monkeypatch.setattr(
        runner, "render_device_envelope_mic_pcm", lambda _case, _step, pcm, **_kw: pcm
    )

    await runner._feed_case_audio(
        object(),
        case=case,
        root=Path("."),
        events=events,
        started=0.0,
        state=object(),
        options=LiveKitRoomOptions(agent_speaking_client_state_lead_ms=120),
        local_participant=local_participant,
    )

    event_types = [event["type"] for event in events]
    published_index = event_types.index("client_audio_state_published")
    lead_index = event_types.index("client_audio_state_lead_wait")
    audio_started_index = event_types.index("user_audio_started")
    assert published_index < lead_index < audio_started_index
    assert events[lead_index]["duration_ms"] == 120

    first_payload = json.loads(local_participant.publish_data.await_args_list[0].args[0])
    assert first_payload["playback_state"] == "agent_speaking"
    assert first_payload["input_mode"] == "ptt"
    assert local_participant.publish_data.await_args_list[0].kwargs["reliable"] is True


@pytest.mark.asyncio
async def test_livekit_room_agent_speaking_step_fails_fast_when_no_agent_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmark import livekit_room_runner as runner

    suite = load_suite("benchmark/cases/full_duplex/explicit_control_enforced.yaml")
    case = {item.case_id: item for item in suite.cases}[
        "fd_explicit_ptt_preempts_without_speech_001"
    ]
    local_participant = AsyncMock()
    events: list[dict] = []

    async def fake_wait_for_agent_speaking(*_args, **_kwargs) -> bool:
        return False

    async def fake_capture_pcm(*_args, **_kwargs) -> int:
        return 0

    monkeypatch.setattr(runner, "_wait_for_agent_speaking", fake_wait_for_agent_speaking)
    monkeypatch.setattr(runner, "_capture_pcm", fake_capture_pcm)
    monkeypatch.setattr(runner, "load_clip_pcm", lambda *_args, **_kwargs: (b"\0\0" * 160, 16000))
    monkeypatch.setattr(
        runner, "render_device_envelope_mic_pcm", lambda _case, _step, pcm, **_kw: pcm
    )

    with pytest.raises(RuntimeError, match="active agent audio"):
        await runner._feed_case_audio(
            object(),
            case=case,
            root=Path("."),
            events=events,
            started=0.0,
            state=object(),
            options=LiveKitRoomOptions(agent_speaking_wait_sec=0.01),
            local_participant=local_participant,
        )

    assert any(event["type"] == "agent_speaking_wait_timeout" for event in events)
    assert all(event["type"] != "user_audio_started" for event in events)
    local_participant.publish_data.assert_not_awaited()


def test_synthetic_default_voiceprint_suite_declares_room_audio_semantics() -> None:
    suite = load_suite("benchmark/cases/synthetic_default_voiceprint_e2e.yaml")
    cases = {case.case_id: case for case in suite.cases}

    assert (
        cases["synthetic_default_owner_normal_001"].expectations.agent_audio_response
        == "after_user_done"
    )
    assert (
        cases["synthetic_default_topic_switch_001"].expectations.agent_audio_response
        == "after_user_done"
    )
    assert cases["synthetic_default_backchannel_001"].expectations.brain == "any"
    assert (
        cases["synthetic_default_backchannel_001"].expectations.rejected_turn_brain == "forbidden"
    )
    assert cases["synthetic_default_backchannel_001"].expectations.action == "any"
    assert cases["synthetic_default_backchannel_001"].expectations.decision_action == "rollback"
    assert cases["synthetic_default_backchannel_001"].expectations.decision_intent == "backchannel"
    assert cases["synthetic_default_backchannel_001"].expectations.agent_audio_response == "first"
    assert (
        cases["synthetic_default_non_owner_rejected_001"].expectations.agent_audio_response
        == "none"
    )


def test_livekit_room_audio_wait_mode_uses_explicit_expectation() -> None:
    suite = load_suite("benchmark/cases/synthetic_default_voiceprint_e2e.yaml")
    cases = {case.case_id: case for case in suite.cases}

    assert _agent_audio_wait_mode(cases["synthetic_default_topic_switch_001"]) == "after_user_done"
    assert _agent_audio_wait_mode(cases["synthetic_default_backchannel_001"]) == "first"
    assert _agent_audio_wait_mode(cases["synthetic_default_non_owner_rejected_001"]) == "none"


def test_livekit_room_audio_wait_timeout_respects_long_case_timeout() -> None:
    suite = load_suite("benchmark/cases/synthetic_default_voiceprint_e2e.yaml")
    case = next(
        case for case in suite.cases if case.case_id == "synthetic_default_topic_switch_001"
    )

    assert _agent_audio_wait_timeout_sec(case, LiveKitRoomOptions(timeout_sec=45.0)) == 90.0


def test_policy_runner_attention_enforced_suite() -> None:
    suite = load_suite("benchmark/cases/attention_admission_enforced.yaml")
    policy = TurnPolicyConfig(
        attention=replace(AttentionPolicyConfig(), enforce=True),
    )

    run = run_policy_suite([suite], turn_policy=policy, run_id="test")

    assert {case.case_id: case.passed for case in run.cases} == {
        "enforced_ambient_playback_speech_does_not_cancel_001": True,
        "enforced_hard_stop_still_cancels_001": True,
        "enforced_no_client_state_preserves_legacy_path_001": True,
    }
    ambient = next(
        case
        for case in run.cases
        if case.case_id == "enforced_ambient_playback_speech_does_not_cancel_001"
    )
    assert ambient.metrics["actual_action"] == "hold"
    assert ambient.metrics["actual_decision_action"] == "hold"
    assert ambient.decisions[0]["attention_admission"]["action"] == "duck_and_decide"
    assert ambient.decisions[0]["attention_admission"]["client_state_used"] is True
    assert ambient.decisions[0]["decision"] is None


def test_policy_runner_enforced_ambient_playback_is_observed_not_cancelled() -> None:
    # Regression: the retired responsive mode used to override attention_enforce
    # =False, so ambient playback speech was first_signal-cancelled even when the
    # operator set enforce=true. With the interrupt_mode axis gone and enforce
    # =True, the ambient overlap is observed (not cancelled). Speed for genuine
    # barge-in comes from channel-owned VAD-start ducking plus transcript/EOT evidence.
    suite = load_suite("benchmark/cases/attention_admission_enforced.yaml")
    policy = TurnPolicyConfig(
        attention=replace(AttentionPolicyConfig(), enforce=True),
    )

    run = run_policy_suite([suite], turn_policy=policy, run_id="test")

    ambient = next(
        case
        for case in run.cases
        if case.case_id == "enforced_ambient_playback_speech_does_not_cancel_001"
    )
    assert run.profile == "balanced_semantic"
    assert ambient.passed is True
    assert ambient.metrics["actual_action"] == "hold"
    assert ambient.metrics["actual_decision_action"] == "hold"
    assert ambient.decisions[0]["attention_admission"]["action"] == "duck_and_decide"
    assert ambient.decisions[0]["decision"] is None


def test_load_v1_interrupt_tiers_enforced_suite() -> None:
    suite = load_suite("benchmark/cases/v1_interrupt_tiers_enforced.yaml")

    assert suite.suite_id == "v1_interrupt_tiers_enforced"
    assert {case.case_id for case in suite.cases} == {
        "tier0_hard_stop_enforced_cancels_001",
        "tier1_topic_switch_enforced_cancels_after_stability_001",
        "tier1_correction_enforced_cancels_after_stability_001",
        "tier2_normal_interrupt_no_client_state_stable_cancel_001",
        "tier3_backchannel_no_client_state_rolls_back_001",
        "tier3_noise_no_client_state_rolls_back_001",
        "tier4_ambient_speech_enforced_observes_001",
    }
    ambient = next(
        case for case in suite.cases if case.case_id == "tier4_ambient_speech_enforced_observes_001"
    )
    assert ambient.user_steps[0].client_playback_state == "agent_speaking"
    assert ambient.expectations.action == "any"
    assert ambient.expectations.decision_action == "hold"
    assert "cancel" in ambient.expectations.forbid_actions


def test_policy_runner_v1_interrupt_tiers_enforced_suite() -> None:
    suite = load_suite("benchmark/cases/v1_interrupt_tiers_enforced.yaml")
    policy = TurnPolicyConfig(
        attention=replace(AttentionPolicyConfig(), enforce=True),
    )

    run = run_policy_suite([suite], turn_policy=policy, run_id="test")

    assert {case.case_id: case.passed for case in run.cases} == {
        "tier0_hard_stop_enforced_cancels_001": True,
        "tier1_topic_switch_enforced_cancels_after_stability_001": True,
        "tier1_correction_enforced_cancels_after_stability_001": True,
        "tier2_normal_interrupt_no_client_state_stable_cancel_001": True,
        "tier3_backchannel_no_client_state_rolls_back_001": True,
        "tier3_noise_no_client_state_rolls_back_001": True,
        "tier4_ambient_speech_enforced_observes_001": True,
    }
    tier4 = next(
        case for case in run.cases if case.case_id == "tier4_ambient_speech_enforced_observes_001"
    )
    assert tier4.metrics["actual_action"] == "hold"
    assert tier4.metrics["actual_decision_action"] == "hold"
    assert tier4.decisions[0]["attention_admission"]["action"] == "duck_and_decide"
    assert tier4.decisions[0]["decision"] is None


def test_load_v1_realistic_interaction_flows_enforced_suite() -> None:
    suite = load_suite("benchmark/cases/v1_realistic_interaction_flows_enforced.yaml")

    assert suite.suite_id == "v1_realistic_interaction_flows_enforced"
    assert len(suite.cases) == 13
    assert {case.case_id for case in suite.cases} >= {
        "flow_normal_question_then_agent_reply_001",
        "flow_normal_question_then_hard_stop_001",
        "flow_normal_question_then_hard_stop_delayed_phrase_001",
        "flow_normal_question_then_topic_switch_001",
        "flow_normal_question_then_correction_001",
        "flow_normal_question_then_followup_no_client_state_001",
        "flow_owner_followup_during_playback_enforced_001",
        "flow_language_switch_during_playback_enforced_001",
        "flow_short_pause_directive_during_playback_enforced_001",
        "flow_normal_question_then_backchannel_001",
        "flow_normal_question_then_noise_001",
        "flow_ambient_speech_during_agent_playback_001",
        "flow_mic_muted_hard_stop_is_ignored_001",
    }
    multi_step = [
        case for case in suite.cases if case.case_id != "flow_normal_question_then_agent_reply_001"
    ]
    assert all(len(case.user_steps) == 2 for case in multi_step)
    assert all(case.user_steps[0].agent_speaking is False for case in multi_step)
    assert all(case.user_steps[1].agent_speaking is True for case in multi_step)


def test_policy_runner_v1_realistic_interaction_flows_enforced_suite() -> None:
    suite = load_suite("benchmark/cases/v1_realistic_interaction_flows_enforced.yaml")
    policy = TurnPolicyConfig(
        attention=replace(AttentionPolicyConfig(), enforce=True),
    )

    run = run_policy_suite([suite], turn_policy=policy, run_id="test")

    pass_by_case = {case.case_id: case.passed for case in run.cases}
    assert pass_by_case == {
        "flow_normal_question_then_agent_reply_001": True,
        "flow_normal_question_then_hard_stop_001": True,
        "flow_normal_question_then_hard_stop_delayed_phrase_001": True,
        "flow_normal_question_then_topic_switch_001": True,
        "flow_normal_question_then_correction_001": True,
        "flow_normal_question_then_followup_no_client_state_001": True,
        "flow_owner_followup_during_playback_enforced_001": True,
        "flow_language_switch_during_playback_enforced_001": True,
        "flow_short_pause_directive_during_playback_enforced_001": True,
        "flow_normal_question_then_backchannel_001": True,
        "flow_normal_question_then_noise_001": True,
        "flow_ambient_speech_during_agent_playback_001": True,
        "flow_mic_muted_hard_stop_is_ignored_001": True,
    }
    followup = next(
        case
        for case in run.cases
        if case.case_id == "flow_normal_question_then_followup_no_client_state_001"
    )
    assert followup.metrics["actual_action"] == "cancel"
    owner_followup = next(
        case
        for case in run.cases
        if case.case_id == "flow_owner_followup_during_playback_enforced_001"
    )
    assert owner_followup.metrics["actual_action"] == "cancel"
    assert owner_followup.decisions[-1]["attention_admission"]["action"] == "duck_and_decide"
    language_switch = next(
        case
        for case in run.cases
        if case.case_id == "flow_language_switch_during_playback_enforced_001"
    )
    assert language_switch.metrics["actual_action"] == "cancel"
    assert language_switch.decisions[-1]["attention_admission"]["action"] == "duck_and_decide"
    pause_directive = next(
        case
        for case in run.cases
        if case.case_id == "flow_short_pause_directive_during_playback_enforced_001"
    )
    assert pause_directive.metrics["actual_action"] == "cancel"
    assert pause_directive.decisions[-1]["attention_admission"]["action"] == "duck_and_decide"
    ambient = next(
        case
        for case in run.cases
        if case.case_id == "flow_ambient_speech_during_agent_playback_001"
    )
    assert ambient.metrics["actual_action"] in {"hold", "none"}
    assert ambient.metrics["actual_action"] != "cancel"
    muted = next(
        case for case in run.cases if case.case_id == "flow_mic_muted_hard_stop_is_ignored_001"
    )
    assert muted.metrics["actual_action"] == "none"
    assert muted.decisions[-1]["attention_admission"]["action"] == "ignore"


def test_load_v1_realistic_extended_suite() -> None:
    suite = load_suite("benchmark/cases/v1_realistic_extended.yaml")

    assert suite.suite_id == "v1_realistic_extended"
    assert {case.case_id for case in suite.cases} == {
        "extended_normal_question_brain_proof_001",
        "extended_hard_stop_latin_artifact_then_correct_001",
        "extended_hard_stop_homophone_then_correct_001",
        "extended_topic_switch_prefix_confusion_001",
        "extended_backchannel_en_during_agent_reply_001",
        "extended_late_ambient_followup_observed_001",
    }
    homophone = next(
        case
        for case in suite.cases
        if case.case_id == "extended_hard_stop_homophone_then_correct_001"
    )
    assert homophone.user_steps[1].interims[0] == "亭"
    assert homophone.expectations.intent == "hard_stop"
    ambient = next(
        case
        for case in suite.cases
        if case.case_id == "extended_late_ambient_followup_observed_001"
    )
    assert ambient.user_steps[1].client_playback_state == "agent_speaking"
    assert "cancel" in ambient.expectations.forbid_actions


def test_policy_runner_v1_realistic_extended_suite() -> None:
    suite = load_suite("benchmark/cases/v1_realistic_extended.yaml")
    policy = TurnPolicyConfig(
        attention=replace(AttentionPolicyConfig(), enforce=True),
    )

    run = run_policy_suite([suite], turn_policy=policy, run_id="test")

    assert {case.case_id: case.passed for case in run.cases} == {
        "extended_normal_question_brain_proof_001": True,
        "extended_hard_stop_latin_artifact_then_correct_001": True,
        "extended_hard_stop_homophone_then_correct_001": True,
        "extended_topic_switch_prefix_confusion_001": True,
        "extended_backchannel_en_during_agent_reply_001": True,
        "extended_late_ambient_followup_observed_001": True,
    }
    topic = next(
        case for case in run.cases if case.case_id == "extended_topic_switch_prefix_confusion_001"
    )
    assert topic.metrics["actual_action"] == "cancel"
    assert topic.metrics["actual_intent"] == "normal_interrupt"
    assert topic.metrics["topic_switch_hint"] is False
    backchannel = next(
        case
        for case in run.cases
        if case.case_id == "extended_backchannel_en_during_agent_reply_001"
    )
    assert backchannel.metrics["actual_action"] in {"hold", "rollback", "none"}


def test_load_barge_in_ab_matrix_suite() -> None:
    suite = load_suite("benchmark/cases/barge_in_ab_matrix_enforced.yaml")

    assert suite.suite_id == "barge_in_ab_matrix_enforced"
    assert {case.case_id for case in suite.cases} == {
        "ab_tier0_hard_stop_tingyixia_fast_cancel_001",
        "ab_tier0_productive_hard_stop_buyaojiangle_001",
        "ab_backchannel_mm_does_not_cancel_001",
        "ab_ack_haode_does_not_cancel_001",
        "ab_false_start_wojuede_waits_for_more_evidence_001",
        "ab_normal_interrupt_high_eot_cancels_001",
        "ab_low_evidence_playback_transcript_observes_001",
        "ab_echo_like_agent_words_observes_001",
        "ab_mic_muted_hard_stop_is_ignored_001",
    }
    backchannel = next(
        case for case in suite.cases if case.case_id == "ab_backchannel_mm_does_not_cancel_001"
    )
    assert backchannel.expectations.decision_action == "rollback"
    false_start = next(
        case
        for case in suite.cases
        if case.case_id == "ab_false_start_wojuede_waits_for_more_evidence_001"
    )
    assert false_start.expectations.decision_action == "hold"
    muted = next(
        case for case in suite.cases if case.case_id == "ab_mic_muted_hard_stop_is_ignored_001"
    )
    assert muted.expectations.decision_action == "none"


def test_policy_runner_barge_in_ab_matrix_suite() -> None:
    suite = load_suite("benchmark/cases/barge_in_ab_matrix_enforced.yaml")
    policy = TurnPolicyConfig(
        attention=replace(AttentionPolicyConfig(), enforce=True),
    )

    run = run_policy_suite([suite], turn_policy=policy, run_id="test")

    assert all(case.passed for case in run.cases)
    hard_stop = next(
        case
        for case in run.cases
        if case.case_id == "ab_tier0_productive_hard_stop_buyaojiangle_001"
    )
    assert hard_stop.metrics["actual_action"] == "cancel"
    assert hard_stop.metrics["actual_intent"] == "hard_stop"
    false_start = next(
        case
        for case in run.cases
        if case.case_id == "ab_false_start_wojuede_waits_for_more_evidence_001"
    )
    assert false_start.metrics["actual_action"] != "cancel"
    assert false_start.metrics["actual_decision_action"] == "hold"
    backchannel = next(
        case for case in run.cases if case.case_id == "ab_backchannel_mm_does_not_cancel_001"
    )
    assert backchannel.metrics["actual_decision_action"] == "rollback"
    normal_interrupt = next(
        case for case in run.cases if case.case_id == "ab_normal_interrupt_high_eot_cancels_001"
    )
    assert normal_interrupt.metrics["actual_action"] == "cancel"
    assert normal_interrupt.metrics["actual_decision_action"] == "cancel"


def test_aggregate_case_metrics() -> None:
    summary = aggregate(
        [
            CaseResult(
                case_id="a",
                suite="s",
                runner="policy",
                passed=True,
                metrics={"interrupt_decision_ms": 100},
            ),
            CaseResult(
                case_id="b",
                suite="s",
                runner="policy",
                passed=False,
                metrics={"interrupt_decision_ms": 200},
            ),
        ]
    )

    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["metrics"]["interrupt_decision_ms"]["p50"] == 150


def _repeat_runs() -> list[RunResult]:
    def run(m_a: float, b_passed: bool, m_b: float) -> RunResult:
        return RunResult(
            run_id="r",
            git_sha="sha",
            runner="policy",
            profile="p",
            cases=[
                CaseResult("a", "s", "policy", True, metrics={"m": m_a}),
                CaseResult(
                    "b",
                    "s",
                    "policy",
                    b_passed,
                    metrics={"m": m_b},
                    errors=[] if b_passed else ["boom"],
                ),
            ],
        )

    return [run(100, True, 200), run(110, False, 260)]


def test_aggregate_runs_tracks_per_case_jitter_and_flakiness() -> None:
    summary = aggregate_runs(_repeat_runs())

    assert summary["repeats"] == 2
    assert summary["total"] == 2
    # only case "a" passed in every repeat
    assert summary["passed"] == 1
    assert summary["flaky"] == 1

    case_a = summary["per_case"]["a"]
    assert case_a["runs"] == 2
    assert case_a["pass_rate"] == 1.0
    assert case_a["metrics"]["m"]["stdev"] == statistics.stdev([100, 110])

    case_b = summary["per_case"]["b"]
    assert case_b["passed"] == 1
    assert case_b["pass_rate"] == 0.5
    assert "boom" in case_b["errors"]

    # pooled distribution carries every (case x repeat) sample for SLO gates
    assert summary["metrics"]["m"]["count"] == 4


def test_write_repeated_reports_emits_stability_view(tmp_path) -> None:
    out = tmp_path / "policy"
    payload = write_repeated_reports(_repeat_runs(), out)

    assert payload["summary"]["repeats"] == 2
    assert len(payload["cases"]) == 4
    assert (out / "metrics.json").exists()
    markdown = (out / "report.md").read_text(encoding="utf-8")
    assert "重复次数：`2`" in markdown
    assert "用例稳定性" in markdown


def test_write_repeated_reports_emits_interrupt_latency_breakdown(tmp_path) -> None:
    out = tmp_path / "livekit_room"
    run = RunResult(
        run_id="interrupt-report",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                "hard_interrupt_001",
                "interrupt",
                "livekit_room",
                True,
                metrics={
                    "expected_action": "cancel",
                    "actual_action": "cancel",
                    "expected_intent": "hard_stop",
                    "actual_intent": "hard_stop",
                    "timeline_interrupted_context_count": 1,
                    "timeline_interrupted_context_source": "tts_in_flight",
                    "timeline_interrupted_context_played_seconds": 1.2,
                    "timeline_interrupted_context_preview": "上一轮被打断的回答",
                    "timeline_interrupt_speech_to_first_transcript_ms": 240.0,
                    "timeline_interrupt_first_transcript_to_resolved_ms": 20.0,
                    "timeline_interrupt_speech_to_resolved_ms": 260.0,
                },
                decisions=[
                    {
                        "interim_text": "停一下",
                        "attention_admission": {
                            "action": "hard_interrupt",
                            "reason": "transcript_hard_stop",
                        },
                        "decision": {
                            "action": "cancel",
                            "intent": "hard_stop",
                            "reason": "intent:hard_stop",
                        },
                    }
                ],
            )
        ],
    )

    write_repeated_reports([run], out)

    markdown = (out / "report.md").read_text(encoding="utf-8")
    assert "打断延迟拆解" in markdown
    assert "`timeline_interrupt_speech_to_first_transcript_ms`" in markdown
    assert "VAD 起声 -> 首次转写" in markdown
    assert "判定摘要" in markdown
    assert "被打断上下文来源" in markdown
    assert "上一轮被打断的回答" in markdown
    assert "关键指标" in markdown
    assert "决策路径" in markdown
    assert "该用例完成 `hard_stop` 打断" in markdown


def test_dashboard_flags_flaky_case_from_repeats(tmp_path) -> None:
    run_dir = tmp_path / "policy"
    write_repeated_reports(_repeat_runs(), run_dir)

    payload = write_dashboard(
        runners=[DashboardRunner(name="policy", candidate=run_dir)],
        output_path=tmp_path / "dashboard.html",
    )

    assert any("flaky case b" in item["text"] for item in payload["findings"])
    html = (tmp_path / "dashboard.html").read_text(encoding="utf-8")
    assert "2 repeats per case" in html


def test_verify_provider_config_rejects_mock() -> None:
    assert verify_provider_config({"brain": "eidolon_agent", "stt": "bailian"}) == []
    failures = verify_provider_config({"brain": "mock_llm", "stt": "bailian"})
    assert any("brain" in f for f in failures)


def test_verify_real_call_component_tts_bytes() -> None:
    cfg = {"brain": "eidolon_agent", "stt": "bailian", "tts": "bailian", "vad": "firered"}
    low = CaseResult("c:tts", "s:tts", "component", True, metrics={"tts_audio_bytes": 10})
    assert verify_real_call(low, runner="component", provider_config=cfg)
    ok = CaseResult("c:tts", "s:tts", "component", True, metrics={"tts_audio_bytes": 50_000})
    assert verify_real_call(ok, runner="component", provider_config=cfg) == []


@pytest.mark.asyncio
async def test_runtime_identity_preflight_resolves_device_boundary() -> None:
    resolver = AsyncMock()
    resolver.resolve_room.return_value = SimpleNamespace(
        owner_id="owner-1",
        companion_id="companion-1",
        device_id="device-1",
    )

    result = await preflight_runtime_identity(
        identity="device-1",
        kind="device",
        owner_id="owner-1",
        resolver=resolver,
    )

    assert result == {
        "ok": True,
        "kind": "device",
        "identity": "device-1",
        "owner_id": "owner-1",
        "companion_id": "companion-1",
        "device_id": "device-1",
    }
    resolver.resolve_room.assert_awaited_once()


@pytest.mark.asyncio
async def test_runtime_identity_preflight_surfaces_unregistered_identity() -> None:
    resolver = AsyncMock()
    resolver.resolve_room.side_effect = RuntimeError("device not found")

    result = await preflight_runtime_identity(
        identity="bench-device",
        kind="device",
        owner_id="owner-1",
        resolver=resolver,
    )

    assert result["ok"] is False
    assert result["identity"] == "bench-device"
    assert result["error"] == "RuntimeError: device not found"


def test_verify_real_call_component_stt_empty_allowed_only_for_noise() -> None:
    cfg = {"brain": "eidolon_agent", "stt": "bailian", "tts": "bailian", "vad": "firered"}
    normal_empty = CaseResult("c:stt", "s:stt", "component", True, metrics={"stt_nonempty": False})
    assert verify_real_call(normal_empty, runner="component", provider_config=cfg)

    noise_empty = CaseResult(
        "c:stt",
        "s:stt",
        "component",
        True,
        metrics={"stt_nonempty": False, "stt_empty_allowed": True},
    )
    assert verify_real_call(noise_empty, runner="component", provider_config=cfg) == []


def test_verify_real_call_room_per_case_checks_stt_and_audio() -> None:
    cfg = {"brain": "eidolon_agent", "stt": "bailian", "tts": "bailian", "vad": "firered"}
    # interrupt-style turn: no brain marks, but real STT stream + agent audio.
    rec = {
        "attrs": {
            "room_name": "voice-bench-x-12345678",
            "stt_stream": {"provider": "bailian"},
            "interrupt_action": "cancel",
        },
        "timestamps": {},
    }
    ok = CaseResult("x", "s", "livekit_room", True, metrics={"agent_audio_bytes": 40_000})
    # per-case no longer requires brain marks (those are run-level)
    assert (
        verify_real_call(ok, runner="livekit_room", provider_config=cfg, case_records=[rec]) == []
    )
    # low audio fails per-case
    low = CaseResult("x", "s", "livekit_room", True, metrics={"agent_audio_bytes": 10})
    assert verify_real_call(low, runner="livekit_room", provider_config=cfg, case_records=[rec])
    # Rejected/non-semantic room turns may intentionally produce no agent audio.
    no_reply = CaseResult(
        "x",
        "s",
        "livekit_room",
        True,
        metrics={
            "agent_audio_bytes": 0,
            "expected_agent_audio_response": "none",
        },
    )
    assert (
        verify_real_call(
            no_reply,
            runner="livekit_room",
            provider_config=cfg,
            case_records=[rec],
        )
        == []
    )
    # wrong/absent STT stream fails per-case
    no_stt = {"attrs": {"room_name": "voice-bench-x-12345678"}, "timestamps": {}}
    assert verify_real_call(ok, runner="livekit_room", provider_config=cfg, case_records=[no_stt])
    # Immediate control turns may flush before stt_stream provider attrs are
    # copied; transcript timeline + decision transcript is still real room STT
    # evidence.
    transcript_mark = {
        "attrs": {
            "room_name": "voice-bench-x-12345678",
            "decision": {"transcript_preview": "停一下"},
        },
        "timestamps": {"transcript_interim_first_at": 1.0},
    }
    assert (
        verify_real_call(
            ok,
            runner="livekit_room",
            provider_config=cfg,
            case_records=[transcript_mark],
        )
        == []
    )


def test_verify_real_call_room_accepts_half_duplex_ptt_segment_evidence() -> None:
    cfg = {"brain": "eidolon_agent", "stt": "bailian", "tts": "bailian", "vad": "firered"}
    committed = CaseResult(
        "phase_a_ptt_normal_release_commits_001",
        "half_duplex_ptt_phase_a",
        "livekit_room",
        True,
        metrics={"agent_audio_bytes": 40_000},
    )
    commit_record = {
        "attrs": {
            "room_name": "voice-bench-phase_a_ptt_normal_release_commits_001-12345678",
            "pipeline": "half_duplex_ptt_segment",
            "ptt_segment": {
                "terminal": {"action": "commit", "reason": "segment_transcribed"},
                "stt_mode": "streaming",
                "transcript_preview": "帮我详细介绍一下这个方案。",
            },
        },
        "timestamps": {},
    }

    assert (
        verify_real_call(
            committed,
            runner="livekit_room",
            provider_config=cfg,
            case_records=[commit_record],
        )
        == []
    )


def test_verify_real_call_room_accepts_half_duplex_ptt_reject_without_stt() -> None:
    cfg = {"brain": "eidolon_agent", "stt": "bailian", "tts": "bailian", "vad": "firered"}
    rejected = CaseResult(
        "phase_a_ptt_tap_to_stop_cancels_001",
        "half_duplex_ptt_phase_a",
        "livekit_room",
        True,
        metrics={
            "agent_audio_bytes": 0,
            "expected_agent_audio_response": "none",
        },
    )
    reject_record = {
        "attrs": {
            "room_name": "voice-bench-phase_a_ptt_tap_to_stop_cancels_001-12345678",
            "pipeline": "half_duplex_ptt_segment",
            "ptt_segment": {
                "terminal": {"action": "reject", "reason": "tap_to_stop"},
                "stt_mode": "none",
            },
        },
        "timestamps": {},
    }

    assert (
        verify_real_call(
            rejected,
            runner="livekit_room",
            provider_config=cfg,
            case_records=[reject_record],
        )
        == []
    )


def test_apply_real_call_verification_run_level_brain(tmp_path) -> None:
    cfg = {"brain": "eidolon_agent", "stt": "bailian", "tts": "bailian", "vad": "firered"}

    def make_run() -> RunResult:
        return RunResult(
            run_id="r",
            git_sha="s",
            runner="livekit_room",
            profile="p",
            cases=[
                CaseResult(
                    "normal_x", "s", "livekit_room", True, metrics={"agent_audio_bytes": 40_000}
                ),
                CaseResult(
                    "hard_y", "s", "livekit_room", True, metrics={"agent_audio_bytes": 40_000}
                ),
            ],
            provider_config=cfg,
        )

    stt = '"stt_stream":{"provider":"bailian"}'
    brain = '"brain_rpc":{"provider":"eidolon_agent_rpc","request_id":"eidolon-abc"}'
    marks = '"brain_request_sent_at":1.0,"brain_first_delta_at":1.2'

    # one normal turn proves the brain; the interrupt turn has none -> all pass
    good = tmp_path / "good_turn_timeline.jsonl"
    good.write_text(
        f'{{"attrs":{{"room_name":"voice-bench-normal_x-12345678",{brain},{stt}}},"timestamps":{{{marks}}}}}\n'
        f'{{"attrs":{{"room_name":"voice-bench-hard_y-abcdef12",{stt}}},"timestamps":{{}}}}\n',
        encoding="utf-8",
    )
    run = make_run()
    apply_real_call_verification(run, strict=True, timeline_path=good)
    assert all(c.passed for c in run.cases)
    assert all(c.metrics["real_call_verified"] for c in run.cases)

    # brain never streams anywhere -> run-level failure on every case
    bad = tmp_path / "bad_turn_timeline.jsonl"
    bad.write_text(
        f'{{"attrs":{{"room_name":"voice-bench-normal_x-12345678",{stt}}},"timestamps":{{}}}}\n'
        f'{{"attrs":{{"room_name":"voice-bench-hard_y-abcdef12",{stt}}},"timestamps":{{}}}}\n',
        encoding="utf-8",
    )
    run2 = make_run()
    apply_real_call_verification(run2, strict=True, timeline_path=bad)
    assert all(not c.passed for c in run2.cases)
    assert all("brain RPC evidence in the whole run" in " ".join(c.errors) for c in run2.cases)

    run3 = make_run()
    apply_real_call_verification(
        run3,
        strict=True,
        timeline_path=bad,
        require_brain_evidence=False,
    )
    assert all(c.passed for c in run3.cases)
    assert all(c.metrics["real_call_verified"] for c in run3.cases)


def test_apply_real_call_verification_strict_fails_mocked_run() -> None:
    run = RunResult(
        run_id="r",
        git_sha="sha",
        runner="component",
        profile="p",
        cases=[
            CaseResult("c:tts", "s:tts", "component", True, metrics={"tts_audio_bytes": 50_000})
        ],
        provider_config={"brain": "mock_llm", "stt": "bailian", "tts": "bailian"},
    )
    apply_real_call_verification(run, strict=True)
    assert run.cases[0].passed is False
    assert run.cases[0].metrics["real_call_verified"] is False
    assert any(e.startswith("real-call:") for e in run.cases[0].errors)


def test_dashboard_flags_unverified_real_call(tmp_path) -> None:
    run = RunResult(
        run_id="rc",
        git_sha="sha",
        runner="component",
        profile="real_components",
        cases=[CaseResult("c:tts", "s:tts", "component", True, metrics={"tts_audio_bytes": 5})],
        provider_config={"brain": "eidolon_agent", "stt": "bailian", "tts": "bailian"},
    )
    apply_real_call_verification(run, strict=True)

    run_dir = tmp_path / "component"
    write_repeated_reports([run], run_dir)
    payload = write_dashboard(
        runners=[DashboardRunner(name="component", candidate=run_dir)],
        output_path=tmp_path / "dashboard.html",
    )
    html = (tmp_path / "dashboard.html").read_text(encoding="utf-8")

    assert any("real call NOT verified" in item["text"] for item in payload["findings"])
    assert "providers: brain=eidolon_agent" in html


def test_compare_detects_p95_regression() -> None:
    baseline = {
        "summary": {
            "failed": 0,
            "metrics": {"latency_ms": {"p95": 100}},
        }
    }
    candidate = {
        "summary": {
            "failed": 0,
            "metrics": {"latency_ms": {"p95": 130}},
        }
    }

    result = compare_metrics(baseline, candidate, max_p95_regression_pct=10)

    assert result["passed"] is False
    assert result["failures"]


def test_component_runner_name_is_supported() -> None:
    summary = aggregate(
        [
            CaseResult(
                case_id="normal_single_turn_001:vad",
                suite="core_latency:vad",
                runner="component",
                passed=True,
                metrics={"vad_elapsed_ms": 42, "vad_detected_speech": True},
            )
        ]
    )

    assert summary["passed"] == 1
    assert summary["metrics"]["vad_elapsed_ms"]["p95"] == 42


def test_livekit_room_state_marks_agent_connected() -> None:
    from benchmark.livekit_room_runner import _RoomCaseState

    state = _RoomCaseState(started=0.0, events=[])
    assert not state.agent_connected.is_set()

    state.mark("participant_connected_at")

    assert state.agent_connected.is_set()


@pytest.mark.asyncio
async def test_livekit_room_wait_for_agent_speaking_requires_audio_after_previous_user_step() -> (
    None
):
    from benchmark.livekit_room_runner import _RoomCaseState, _wait_for_agent_speaking

    state = _RoomCaseState(started=0.0, events=[])
    state.last_agent_audio_monotonic = time.monotonic()
    state.agent_audio_frame_timestamps.append(100)

    assert (
        await _wait_for_agent_speaking(
            state,
            timeout_sec=0.01,
            after_elapsed_ms=200,
        )
        is False
    )

    state.agent_audio_frame_timestamps.append(240)
    state.last_agent_audio_monotonic = time.monotonic()

    assert (
        await _wait_for_agent_speaking(
            state,
            timeout_sec=0.01,
            after_elapsed_ms=200,
        )
        is True
    )


def test_livekit_room_retries_only_pre_audio_infrastructure_failures() -> None:
    from benchmark.livekit_room_runner import _should_retry_room_case

    missing_agent = CaseResult(
        case_id="topic_switch_001",
        suite="semantic_control",
        runner="livekit_room",
        passed=False,
        errors=["timed out waiting for agent participant before user audio"],
    )
    semantic_failure = CaseResult(
        case_id="topic_switch_001",
        suite="semantic_control",
        runner="livekit_room",
        passed=False,
        errors=["timeline expected intent=topic_switch, got ['<none>']"],
    )
    connect_transient = CaseResult(
        case_id="topic_switch_001",
        suite="semantic_control",
        runner="livekit_room",
        passed=False,
        metrics={"room_connected_ms": None, "user_audio_done_ms": None},
        errors=[
            "ConnectError: engine: signal failure: client error: "
            "401 Unauthorized - no permissions to access the room"
        ],
    )
    after_audio_failure = CaseResult(
        case_id="topic_switch_001",
        suite="semantic_control",
        runner="livekit_room",
        passed=False,
        metrics={"room_connected_ms": 100, "user_audio_done_ms": 900},
        errors=[
            "ConnectError: engine: signal failure: client error: "
            "401 Unauthorized - no permissions to access the room"
        ],
    )

    assert _should_retry_room_case(missing_agent) is True
    assert _should_retry_room_case(connect_transient) is True
    assert _should_retry_room_case(after_audio_failure) is False
    assert _should_retry_room_case(semantic_failure) is False


@pytest.mark.asyncio
async def test_livekit_room_case_retry_records_previous_attempt(monkeypatch) -> None:
    from benchmark import livekit_room_runner
    from benchmark.livekit_room_runner import (
        LiveKitRoomOptions,
        _run_room_case_with_retries,
    )

    calls = 0

    async def fake_run_room_case(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return CaseResult(
                case_id="topic_switch_001",
                suite="semantic_control",
                runner="livekit_room",
                passed=False,
                metrics={"room_name": "room-failed"},
                errors=["timed out waiting for agent participant before user audio"],
            )
        return CaseResult(
            case_id="topic_switch_001",
            suite="semantic_control",
            runner="livekit_room",
            passed=True,
            metrics={"room_name": "room-retry"},
        )

    monkeypatch.setattr(livekit_room_runner, "_run_room_case", fake_run_room_case)
    suite = load_suite("benchmark/cases/core.yaml")
    case = next(case for case in suite.cases if case.case_id == "topic_switch_001")

    result = await _run_room_case_with_retries(
        case,
        root=Path("."),
        options=LiveKitRoomOptions(agent_missing_retry_count=1),
        livekit_url="ws://127.0.0.1:7880",
        api_key="devkey",
        api_secret="devkey_secret",
    )

    assert calls == 2
    assert result.passed is True
    assert result.metrics["retry_attempts"] == 1
    assert result.events[0]["type"] == "case_retry"
    assert result.events[0]["previous_room_name"] == "room-failed"


@pytest.mark.asyncio
async def test_livekit_room_case_retry_records_room_connect_transient(
    monkeypatch,
) -> None:
    from benchmark import livekit_room_runner
    from benchmark.livekit_room_runner import (
        LiveKitRoomOptions,
        _run_room_case_with_retries,
    )

    calls = 0

    async def fake_run_room_case(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return CaseResult(
                case_id="topic_switch_001",
                suite="semantic_control",
                runner="livekit_room",
                passed=False,
                metrics={"room_name": "room-connect-failed"},
                errors=[
                    "ConnectError: engine: signal failure: client error: "
                    "401 Unauthorized - no permissions to access the room"
                ],
            )
        return CaseResult(
            case_id="topic_switch_001",
            suite="semantic_control",
            runner="livekit_room",
            passed=True,
            metrics={"room_name": "room-retry"},
        )

    monkeypatch.setattr(livekit_room_runner, "_run_room_case", fake_run_room_case)
    suite = load_suite("benchmark/cases/core.yaml")
    case = next(case for case in suite.cases if case.case_id == "topic_switch_001")

    result = await _run_room_case_with_retries(
        case,
        root=Path("."),
        options=LiveKitRoomOptions(agent_missing_retry_count=1),
        livekit_url="ws://127.0.0.1:7880",
        api_key="devkey",
        api_secret="devkey_secret",
    )

    assert calls == 2
    assert result.passed is True
    assert result.metrics["retry_attempts"] == 1
    assert result.events[0]["reason"] == "room_connect_transient"
    assert result.events[0]["previous_room_name"] == "room-connect-failed"


@pytest.mark.asyncio
async def test_livekit_room_publishes_client_audio_state() -> None:
    from eidolon_sdk.biz.contracts import CLIENT_AUDIO_STATE_TOPIC, WIRE_SCHEMA_VERSION
    from benchmark.livekit_room_runner import (
        _publish_client_audio_state,
    )

    local_participant = AsyncMock()
    events: list[dict] = []

    await _publish_client_audio_state(
        local_participant,
        events=events,
        started=0.0,
        playback_state="agent_speaking",
        ptt=True,
        manual_interrupt=True,
        mic_muted=True,
    )

    local_participant.publish_data.assert_awaited_once()
    payload = json.loads(local_participant.publish_data.await_args.args[0])
    assert payload["type"] == "client.audio_state"
    assert payload["schema_v"] == WIRE_SCHEMA_VERSION
    assert payload["input_mode"] == "auto"
    assert payload["playback_state"] == "agent_speaking"
    assert payload["ptt"] is True
    assert payload["manual_interrupt"] is True
    assert payload["mic_muted"] is True
    assert local_participant.publish_data.await_args.kwargs == {
        "reliable": True,
        "topic": CLIENT_AUDIO_STATE_TOPIC,
    }
    assert events[-1]["type"] == "client_audio_state_published"
    assert events[-1]["schema_v"] == WIRE_SCHEMA_VERSION
    assert events[-1]["input_mode"] == "auto"
    assert events[-1]["reliable"] is True
    assert events[-1]["ptt"] is True
    assert events[-1]["manual_interrupt"] is True
    assert events[-1]["mic_muted"] is True


@pytest.mark.asyncio
async def test_livekit_room_refreshes_client_audio_state_periodically() -> None:
    from benchmark.livekit_room_runner import (
        _refresh_client_audio_state,
    )

    local_participant = AsyncMock()
    events: list[dict] = []

    task = asyncio.create_task(
        _refresh_client_audio_state(
            local_participant,
            events=events,
            started=0.0,
            playback_state="agent_speaking",
            interval_sec=0.01,
        )
    )
    deadline = asyncio.get_running_loop().time() + 0.2
    while (
        local_participant.publish_data.await_count < 2
        and asyncio.get_running_loop().time() < deadline
    ):
        await asyncio.sleep(0.005)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert local_participant.publish_data.await_count >= 2
    assert {event["playback_state"] for event in events} == {"agent_speaking"}


def test_livekit_room_transcription_attribution_excludes_agent_tts() -> None:
    assert (
        _transcription_role(
            source_identity="device-1",
            benchmark_identity="device-1",
        )
        == "user"
    )
    assert (
        _transcription_role(
            source_identity="agent-AJ_123",
            benchmark_identity="device-1",
        )
        == "agent"
    )


@pytest.mark.asyncio
async def test_wait_for_agent_quiet_requires_a_new_reply_after_previous_user_step() -> None:
    from benchmark.livekit_room_runner import _RoomCaseState, _wait_for_agent_quiet

    state = _RoomCaseState(started=time.monotonic(), events=[])
    state.agent_audio_frame_timestamps.append(10)
    state.last_agent_audio_monotonic = time.monotonic() - 1.0

    wait = asyncio.create_task(
        _wait_for_agent_quiet(
            state,
            quiet_ms=10,
            timeout_sec=0.5,
            after_elapsed_ms=20,
        )
    )
    await asyncio.sleep(0.06)
    assert wait.done() is False

    state.agent_audio_frame_timestamps.append(30)
    state.last_agent_audio_monotonic = time.monotonic()
    assert await wait is True


@pytest.mark.asyncio
async def test_wait_for_agent_quiet_times_out_without_next_reply() -> None:
    from benchmark.livekit_room_runner import _RoomCaseState, _wait_for_agent_quiet

    state = _RoomCaseState(started=time.monotonic(), events=[])
    state.agent_audio_frame_timestamps.append(10)
    state.last_agent_audio_monotonic = time.monotonic() - 1.0

    assert await _wait_for_agent_quiet(
        state,
        quiet_ms=10,
        timeout_sec=0.06,
        after_elapsed_ms=20,
    ) is False


def test_livekit_dispatch_token_includes_participant_metadata() -> None:
    from benchmark.livekit_room_runner import (
        LiveKitRoomOptions,
        _make_dispatch_token,
        _participant_metadata,
    )

    metadata = _participant_metadata(
        LiveKitRoomOptions(
            participant_identity="manson",
            participant_kind="user",
            participant_metadata={"client": "bench"},
        )
    )
    token = _make_dispatch_token(
        api_key="devkey",
        api_secret="test-secret-with-enough-entropy-32-bytes",
        room_name="room-1",
        participant="manson",
        agent_name="eidolon",
        metadata=metadata,
    )

    payload = jwt.decode(
        token,
        "test-secret-with-enough-entropy-32-bytes",
        algorithms=["HS256"],
    )

    assert payload["sub"] == "manson"
    assert payload["roomConfig"]["agents"][0]["agentName"] == "eidolon"
    assert json.loads(payload["roomConfig"]["agents"][0]["metadata"]) == {
        "conversation_id": "bench:b629ec01f1ef5f85855baac825022053"
    }
    assert json.loads(payload["metadata"]) == {
        "client": "bench",
        "kind": "user",
    }


def test_bench_voice_derives_full_duplex_participant_route_from_suite() -> None:
    suite = load_suite("benchmark/cases/full_duplex/dogfood_box3_audio_first_enforced.yaml")
    args = type(
        "Args",
        (),
        {
            "livekit_interaction_mode": None,
            "livekit_session_intent": "user_initiated",
        },
    )()

    metadata = _bench_participant_metadata(args, [suite])

    assert metadata == {
        "interaction_mode": "full_duplex",
        "session_intent": "user_initiated",
    }


def test_bench_voice_explicit_route_overrides_suite_mode() -> None:
    suite = load_suite("benchmark/cases/full_duplex/dogfood_box3_audio_first_enforced.yaml")
    args = type(
        "Args",
        (),
        {
            "livekit_interaction_mode": "half_duplex",
            "livekit_session_intent": "user_initiated",
        },
    )()

    metadata = _bench_participant_metadata(args, [suite])

    assert metadata["interaction_mode"] == "half_duplex"


def test_timeline_records_are_summarized(tmp_path) -> None:
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        "\n".join(
            [
                (
                    '{"turn_id":"t1","durations_ms":{"commit_to_tts_first_audio":100},'
                    '"timestamps":{"speech_started_at":1.0,"transcript_final_at":1.2},'
                    '"attrs":{"decision_reason":"intent:hard_stop",'
                    '"decision":{"hold_recheck_ms":40},'
                    '"room_name":"voice-bench-hard_interrupt_001-1234abcd",'
                    '"timeline_flush_reason":"interrupt_cancel",'
                    '"provider_latency_ms":{"stt_final_ms":200}}}'
                ),
                (
                    '{"turn_id":"t2","durations_ms":{"commit_to_tts_first_audio":300},'
                    '"timestamps":{"speech_started_at":2.0,"transcript_final_at":2.4},'
                    '"attrs":{"room_name":"voice-bench-normal_single_turn_001-deadbeef",'
                    '"timeline_flush_reason":"agent_audio_playback_done",'
                    '"provider_latency_ms":{"stt_final_ms":400}}}'
                ),
            ]
        ),
        encoding="utf-8",
    )

    summary = summarize_timeline_records(load_timeline_records(tmp_path))

    assert summary["count"] == 2
    assert summary["latencies"]["commit_to_tts_first_audio"]["p95"] == 290
    assert summary["latencies"]["stt_final_ms"]["p50"] == 300
    assert summary["latencies"]["decision_hold_recheck_ms"]["p50"] == 40
    assert round(summary["provider_segments"]["stt_final"]["p50"]) == 300
    assert summary["record_summaries"][0]["case_id"] == "hard_interrupt_001"
    assert summary["decision_reasons"]["intent:hard_stop"] == 1
    assert summary["flush_reasons"]["agent_audio_playback_done"] == 1
    assert summary["cases"]["hard_interrupt_001"] == 1
    assert summary["cases"]["normal_single_turn_001"] == 1


def test_timeline_capture_writes_only_new_lines(tmp_path) -> None:
    source = tmp_path / "worker_timeline.jsonl"
    source.write_text('{"turn_id":"old"}\n', encoding="utf-8")
    capture = TimelineCapture.start(str(source))

    with source.open("a", encoding="utf-8") as f:
        f.write('{"turn_id":"new-1"}\n')
        f.write('{"turn_id":"new-2"}\n')

    output = tmp_path / "run" / "turn_timeline.jsonl"
    written = capture.write_new_lines(output)

    assert written == 2
    assert "old" not in output.read_text(encoding="utf-8")
    assert len(load_timeline_records(output)) == 2


def test_timeline_capture_expands_home_config_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    source = tmp_path / "eidolon" / "logs" / "turn-timeline.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text('{"turn_id":"old"}\n', encoding="utf-8")
    capture = TimelineCapture.start("~/eidolon/logs/turn-timeline.jsonl")

    with source.open("a", encoding="utf-8") as f:
        f.write('{"turn_id":"new"}\n')

    output = tmp_path / "captured" / "turn_timeline.jsonl"
    assert capture.write_new_lines(output) == 1
    assert load_timeline_records(output)[0]["turn_id"] == "new"


def test_dashboard_includes_timeline_section(tmp_path) -> None:
    from benchmark.report import write_reports

    run_dir = tmp_path / "candidate"
    write_reports(
        RunResult(
            run_id="timeline-test",
            git_sha="abc123",
            runner="policy",
            profile="test",
            cases=[
                CaseResult(
                    case_id="case-1",
                    suite="suite",
                    runner="policy",
                    passed=True,
                    metrics={"elapsed_ms": 1},
                )
            ],
        ),
        run_dir,
    )
    (run_dir / "turn_timeline.jsonl").write_text(
        (
            '{"turn_id":"t1","durations_ms":{"commit_to_tts_first_audio":100},'
            '"attrs":{"timeline_flush_reason":"agent_audio_playback_done"}}\n'
        ),
        encoding="utf-8",
    )

    payload = write_dashboard(
        runners=[DashboardRunner(name="policy", candidate=run_dir)],
        output_path=tmp_path / "dashboard.html",
    )
    html = (tmp_path / "dashboard.html").read_text(encoding="utf-8")

    assert payload["runners"][0]["timeline"]["count"] == 1
    assert "policy Turn Timeline" in html
    assert "commit_to_tts_first_audio" in html
    assert "Case Coverage" in html
    assert "Provider Latency Breakdown" in html
    assert "Per-Turn Provider Segments" in html
    assert "STT Audio" in html


def test_slo_gate_detects_timeline_regression() -> None:
    runner = {
        "name": "livekit_room",
        "summary": {"metrics": {}},
        "timeline": {
            "latencies": {
                "vad_start_to_interrupt_resolved": {
                    "p95": 800.0,
                }
            }
        },
    }

    results = evaluate_slo_gates(
        runner,
        gates=(
            SloGate(
                name="interrupt",
                runner="livekit_room",
                source="timeline",
                metric="vad_start_to_interrupt_resolved",
                statistic="p95",
                max_value=650.0,
            ),
        ),
    )

    assert results[0]["passed"] is False
    assert results[0]["value"] == 800.0


def test_slo_gate_advisory_below_min_samples() -> None:
    gate = SloGate(
        name="g",
        runner="livekit_room",
        source="summary",
        metric="m",
        statistic="p95",
        max_value=650.0,
        tier="target",
        min_samples=20,
    )
    low = {"name": "livekit_room", "summary": {"metrics": {"m": {"p95": 900.0, "count": 3}}}}
    res_low = evaluate_slo_gates(low, gates=(gate,))[0]
    assert res_low["advisory"] is True
    assert res_low["passed"] is True
    assert res_low["tier"] == "target"

    enough = {"name": "livekit_room", "summary": {"metrics": {"m": {"p95": 900.0, "count": 25}}}}
    res_full = evaluate_slo_gates(enough, gates=(gate,))[0]
    assert res_full["advisory"] is False
    assert res_full["passed"] is False


def test_enforcement_failures_only_required_hard_fails() -> None:
    # The default ratchet enforces the two summary experience ceilings.
    enforced = {g.name for g in DEFAULT_SLO_GATES if g.required and g.source == "summary"}
    assert "room_user_done_to_next_audio" in enforced
    assert "room_publish_to_first_audio" in enforced

    # required gate that hard-fails -> enforcement failure
    runner = {
        "name": "livekit_room",
        "summary": {
            "metrics": {
                "user_done_to_agent_audio_after_user_done_ms": {"p95": 1800.0, "count": 25},
                "publish_to_agent_audio_first_ms": {"p95": 1000.0, "count": 25},
            }
        },
    }
    results = evaluate_slo_gates(runner)
    fails = enforcement_failures(results)
    assert [f["name"] for f in fails] == ["room_user_done_to_next_audio"]

    # an advisory (too few samples) required gate must NOT block
    advisory = evaluate_slo_gates(
        {
            "name": "livekit_room",
            "summary": {"metrics": {}},
            "timeline": {
                "latencies": {
                    "commit_to_tts_first_audio": {"p50": 9999.0, "p95": 9999.0, "count": 3},
                }
            },
        }
    )
    assert enforcement_failures(advisory) == []


def test_tts_ttfb_slo_uses_first_text_to_provider_audio() -> None:
    tts_gate = next(g for g in DEFAULT_SLO_GATES if g.name == "tts_ttfb")

    assert tts_gate.metric == "tts_first_text_sent_to_provider_first_audio_ms"


def test_timeline_preemptive_outcomes(tmp_path) -> None:
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        "\n".join(
            [
                # reused: brain started before commit, not cancelled
                '{"turn_id":"a","timestamps":{"brain_request_started_at":1.0,'
                '"turn_committed_at":2.0},"attrs":{}}',
                # discarded: brain started before commit, then cancelled
                '{"turn_id":"b","timestamps":{"brain_request_started_at":1.0,'
                '"turn_committed_at":2.0,"brain_cancelled_at":1.5},"attrs":{}}',
                # not preemptive: brain started after commit
                '{"turn_id":"c","timestamps":{"turn_committed_at":1.0,'
                '"brain_request_started_at":2.0},"attrs":{}}',
            ]
        ),
        encoding="utf-8",
    )

    summary = summarize_timeline_records(load_timeline_records(timeline_path))
    pre = summary["preemptive"]
    assert pre["triggered"] == 2
    assert pre["reused"] == 1
    assert pre["discarded"] == 1
    assert pre["reuse_rate"] == 0.5
    assert pre["waste_rate"] == 0.5
    outcomes = {r["turn_id"]: r["preemptive"] for r in summary["record_summaries"]}
    assert outcomes == {"a": "reused", "b": "discarded", "c": ""}


def test_timeline_exposes_tts_and_endpoint_segments(tmp_path) -> None:
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        (
            '{"turn_id":"t1","timestamps":{'
            '"turn_committed_at":1.0,"stt_provider_final_at":1.8,'
            '"tts_stream_started_at":1.95,"tts_connection_acquired_at":2.0,'
            '"tts_request_started_at":2.0,"tts_first_text_sent_at":2.05,'
            '"tts_provider_first_audio_at":2.15,'
            '"tts_first_audio_at":2.20},'
            '"attrs":{"room_name":"voice-bench-normal_single_turn_001-1234abcd"}}\n'
        ),
        encoding="utf-8",
    )

    summary = summarize_timeline_records(load_timeline_records(timeline_path))
    segments = summary["provider_segments"]

    assert round(segments["tts_request_to_first_audio"]["p50"]) == 150
    assert round(segments["tts_pool_acquire"]["p50"]) == 50
    assert round(segments["tts_request_to_first_text"]["p50"]) == 50
    assert round(segments["tts_first_text_to_provider_audio"]["p50"]) == 100
    assert round(segments["tts_provider_to_agent_audio"]["p50"]) == 50
    assert round(segments["stt_final_after_commit"]["p50"]) == 800


def test_dashboard_findings_include_slo_failure(tmp_path) -> None:
    from benchmark.report import write_reports

    run_dir = tmp_path / "livekit_room"
    write_reports(
        RunResult(
            run_id="slo-test",
            git_sha="abc123",
            runner="livekit_room",
            profile="test",
            cases=[
                CaseResult(
                    case_id="case-1",
                    suite="suite",
                    runner="livekit_room",
                    passed=True,
                    metrics={"user_done_to_agent_audio_after_user_done_ms": 2000},
                )
            ],
        ),
        run_dir,
    )

    payload = write_dashboard(
        runners=[DashboardRunner(name="livekit_room", candidate=run_dir)],
        output_path=tmp_path / "dashboard.html",
    )

    assert payload["findings"]
    assert any("SLO" in item["text"] for item in payload["findings"])


def test_dashboard_renders_slo_gate_table(tmp_path) -> None:
    from benchmark.report import write_reports

    run_dir = tmp_path / "livekit_room"
    write_reports(
        RunResult(
            run_id="slo-table-test",
            git_sha="abc123",
            runner="livekit_room",
            profile="test",
            cases=[
                CaseResult(
                    case_id="case-1",
                    suite="suite",
                    runner="livekit_room",
                    passed=True,
                    metrics={"user_done_to_agent_audio_after_user_done_ms": 900},
                )
            ],
        ),
        run_dir,
    )

    write_dashboard(
        runners=[DashboardRunner(name="livekit_room", candidate=run_dir)],
        output_path=tmp_path / "dashboard.html",
    )
    html = (tmp_path / "dashboard.html").read_text(encoding="utf-8")

    assert "livekit_room SLO Gates" in html
    assert "room_user_done_to_next_audio" in html
    assert "PASS" in html


def test_dashboard_warns_when_livekit_room_timeline_coverage_is_partial(
    tmp_path,
) -> None:
    from benchmark.report import write_reports

    run_dir = tmp_path / "livekit_room"
    write_reports(
        RunResult(
            run_id="coverage-test",
            git_sha="abc123",
            runner="livekit_room",
            profile="test",
            cases=[
                CaseResult(
                    case_id="covered",
                    suite="suite",
                    runner="livekit_room",
                    passed=True,
                    metrics={},
                ),
                CaseResult(
                    case_id="missing",
                    suite="suite",
                    runner="livekit_room",
                    passed=True,
                    metrics={},
                ),
            ],
        ),
        run_dir,
    )
    (run_dir / "turn_timeline.jsonl").write_text(
        ('{"turn_id":"t1","attrs":{"room_name":"voice-bench-covered-1234abcd"}}\n'),
        encoding="utf-8",
    )

    payload = write_dashboard(
        runners=[DashboardRunner(name="livekit_room", candidate=run_dir)],
        output_path=tmp_path / "dashboard.html",
    )

    assert any("timeline coverage missing" in item["text"] for item in payload["findings"])
    assert any("missing" in item["text"] for item in payload["findings"])


def test_livekit_room_timeline_expectations_fail_unexpected_cancel(
    tmp_path,
) -> None:
    suite = load_suite("benchmark/cases/core.yaml")
    run = RunResult(
        run_id="expectation-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id="normal_single_turn_001",
                suite="core_latency",
                runner="livekit_room",
                passed=True,
            )
        ],
    )
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        (
            '{"turn_id":"t1","attrs":{"room_name":'
            '"voice-bench-normal_single_turn_001-1234abcd",'
            '"interrupt_action":"cancel","decision":{"intent":"normal_interrupt"}},'
            '"timestamps":{"interrupt_resolved_at":10.2}}\n'
        ),
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].passed is False
    assert "expected no interrupt action" in run.cases[0].errors[0]


def test_livekit_room_timeline_expectations_pass_expected_hard_stop(
    tmp_path,
) -> None:
    suite = load_suite("benchmark/cases/core.yaml")
    run = RunResult(
        run_id="expectation-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id="hard_interrupt_001",
                suite="interrupt",
                runner="livekit_room",
                passed=True,
            )
        ],
    )
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        (
            '{"turn_id":"t1","attrs":{"room_name":'
            '"voice-bench-hard_interrupt_001-1234abcd",'
            '"interrupt_action":"cancel","decision":{"intent":"hard_stop"}},'
            '"durations_ms":{"vad_start_to_interrupt_resolved":320}}\n'
        ),
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].passed is True
    assert not run.cases[0].errors


def test_livekit_room_timeline_interrupt_slo_ignores_setup_commit(
    tmp_path,
) -> None:
    suite = load_suite("benchmark/cases/full_duplex/gate_enforced.yaml")
    run = RunResult(
        run_id="expectation-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id="fd_gate_topic_switch_cancels_and_replies_001",
                suite="full_duplex_gate",
                runner="livekit_room",
                passed=True,
            )
        ],
    )
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        (
            '{"turn_id":"setup","attrs":{"room_name":'
            '"voice-bench-fd_gate_topic_switch_cancels_and_replies_001-1234abcd"},'
            '"timestamps":{"speech_started_at":1.0,"turn_committed_at":3.196}}\n'
            '{"turn_id":"cancel","attrs":{"room_name":'
            '"voice-bench-fd_gate_topic_switch_cancels_and_replies_001-1234abcd",'
            '"interrupt_action":"cancel",'
            '"decision":{"action":"cancel","intent":"normal_interrupt",'
            '"topic_switch_hint":true},'
            '"interrupted_context":{"source":"tts_in_flight",'
            '"played_seconds":1.2,"text_preview":"上一轮回答"},'
            '"client_control_events":[{"op":"playback.stop"}]},'
            '"timestamps":{"speech_started_at":10.0,'
            '"interrupt_cancel_resolved_at":10.49,'
            '"interrupt_resolved_at":10.49,'
            '"transcript_actionable_first_at":10.48},'
            '"durations_ms":{"vad_start_to_interrupt_cancel_resolved":490}}\n'
        ),
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].passed is True
    assert not run.cases[0].errors


def test_livekit_room_topic_switch_uses_first_yield_for_slo(
    tmp_path,
) -> None:
    suite = load_suite("benchmark/cases/full_duplex/gate_enforced.yaml")
    run = RunResult(
        run_id="expectation-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id="fd_gate_topic_switch_cancels_and_replies_001",
                suite="full_duplex_gate",
                runner="livekit_room",
                passed=True,
            )
        ],
    )
    room = "voice-bench-fd_gate_topic_switch_cancels_and_replies_001-1234abcd"
    timeline_path = tmp_path / "turn_timeline.jsonl"
    records = [
        {
            "turn_id": "setup",
            "attrs": {"room_name": room},
            "timestamps": {"speech_started_at": 1.0, "turn_committed_at": 2.0},
        },
        {
            "turn_id": "first-yield",
            "attrs": {
                "room_name": room,
                "interrupt_action": "cancel",
                "decision": {
                    "action": "cancel",
                    "intent": "normal_interrupt",
                    "topic_switch_hint": True,
                },
                "interrupted_context": {
                    "source": "tts_in_flight",
                    "played_seconds": 1.2,
                    "text_preview": "上一轮回答",
                },
                "client_control_events": [{"op": "playback.stop"}],
            },
            "timestamps": {
                "speech_started_at": 10.0,
                "transcript_actionable_first_at": 10.550,
                "playback_stop_sent_at": 10.5511,
                "interrupt_cancel_resolved_at": 10.5512,
                "interrupt_resolved_at": 10.5512,
            },
            "durations_ms": {
                "vad_start_to_interrupt_cancel_resolved": 551.2,
                "vad_start_to_interrupt_resolved": 551.2,
                "vad_start_to_playback_stop_sent": 551.1,
            },
        },
        {
            "turn_id": "collect-new-topic",
            "attrs": {
                "room_name": room,
                "interrupt_action": "cancel",
                "decision": {
                    "action": "cancel",
                    "intent": "normal_interrupt",
                    "topic_switch_hint": True,
                },
                "interrupted_context": {
                    "source": "tts_in_flight",
                    "played_seconds": 1.2,
                    "text_preview": "上一轮回答",
                },
                "client_control_events": [{"op": "playback.stop"}],
            },
            "timestamps": {
                "speech_started_at": 20.0,
                "transcript_actionable_first_at": 21.2476,
                "playback_stop_sent_at": 21.2477,
                "interrupt_cancel_resolved_at": 21.2478,
                "interrupt_resolved_at": 21.2478,
            },
            "durations_ms": {
                "vad_start_to_interrupt_cancel_resolved": 1247.8,
                "vad_start_to_interrupt_resolved": 1247.8,
                "vad_start_to_playback_stop_sent": 1247.7,
            },
        },
    ]
    timeline_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    metrics = run.cases[0].metrics
    assert run.cases[0].passed is True
    assert not run.cases[0].errors
    assert metrics["timeline_yield_old_output_ms"] == 551.2
    assert metrics["timeline_yield_old_output_playback_stop_ms"] == 551.1
    assert metrics["timeline_vad_start_to_interrupt_cancel_resolved"] == 1247.8
    assert metrics["timeline_interrupt_speech_to_playback_stop_ms"] == 1247.7
    assert metrics["timeline_cancel_then_collect"] == 1.0
    assert metrics["timeline_cancel_then_collect_count"] == 2.0
    assert metrics["timeline_collect_new_topic_turn_cancel_ms"] == 1247.8
    assert metrics["timeline_collect_new_topic_turn_playback_stop_ms"] == 1247.7


def test_livekit_room_topic_switch_fails_when_first_yield_is_slow(
    tmp_path,
) -> None:
    suite = load_suite("benchmark/cases/full_duplex/gate_enforced.yaml")
    run = RunResult(
        run_id="expectation-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id="fd_gate_topic_switch_cancels_and_replies_001",
                suite="full_duplex_gate",
                runner="livekit_room",
                passed=True,
            )
        ],
    )
    room = "voice-bench-fd_gate_topic_switch_cancels_and_replies_001-1234abcd"
    timeline_path = tmp_path / "turn_timeline.jsonl"
    records = [
        {
            "turn_id": "first-yield",
            "attrs": {
                "room_name": room,
                "interrupt_action": "cancel",
                "decision": {
                    "action": "cancel",
                    "intent": "normal_interrupt",
                    "topic_switch_hint": True,
                },
            },
            "timestamps": {
                "speech_started_at": 10.0,
                "interrupt_cancel_resolved_at": 10.9,
                "interrupt_resolved_at": 10.9,
            },
            "durations_ms": {
                "vad_start_to_interrupt_cancel_resolved": 900.0,
                "vad_start_to_interrupt_resolved": 900.0,
            },
        },
        {
            "turn_id": "later-fast-cancel",
            "attrs": {
                "room_name": room,
                "interrupt_action": "cancel",
                "decision": {
                    "action": "cancel",
                    "intent": "normal_interrupt",
                    "topic_switch_hint": True,
                },
            },
            "timestamps": {
                "speech_started_at": 20.0,
                "interrupt_cancel_resolved_at": 20.3,
                "interrupt_resolved_at": 20.3,
            },
            "durations_ms": {
                "vad_start_to_interrupt_cancel_resolved": 300.0,
                "vad_start_to_interrupt_resolved": 300.0,
            },
        },
    ]
    timeline_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].passed is False
    assert any("timeline interrupt decision exceeded" in error for error in run.cases[0].errors)


def test_livekit_room_timeline_expectations_fail_wrong_ptt_terminal(
    tmp_path,
) -> None:
    suite = load_suite("benchmark/cases/half_duplex_ptt_phase_a_enforced.yaml")
    run = RunResult(
        run_id="expectation-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id="phase_a_ptt_tap_to_stop_cancels_001",
                suite="half_duplex_ptt_phase_a",
                runner="livekit_room",
                passed=True,
            )
        ],
    )
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        (
            '{"turn_id":"t1","attrs":{"room_name":'
            '"voice-bench-phase_a_ptt_tap_to_stop_cancels_001-1234abcd",'
            '"interrupt_action":"cancel",'
            '"decision":{"action":"cancel","intent":"hard_stop"},'
            '"client_control_events":[{"op":"playback.stop"}],'
            '"ptt_segment_terminal":{"action":"commit","reason":"segment_transcribed"}},'
            '"timestamps":{"speech_started_at":1.0,'
            '"interrupt_started_at":1.0,'
            '"interrupt_resolved_at":1.0,'
            '"transcript_actionable_first_at":1.0}}\n'
        ),
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].passed is False
    assert any("PTT terminal action=reject" in err for err in run.cases[0].errors)


def test_livekit_room_timeline_expectations_pass_ptt_terminal_reject(
    tmp_path,
) -> None:
    suite = load_suite("benchmark/cases/half_duplex_ptt_phase_a_enforced.yaml")
    run = RunResult(
        run_id="expectation-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id="phase_a_ptt_tap_to_stop_cancels_001",
                suite="half_duplex_ptt_phase_a",
                runner="livekit_room",
                passed=True,
            )
        ],
    )
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        (
            '{"turn_id":"t1","attrs":{"room_name":'
            '"voice-bench-phase_a_ptt_tap_to_stop_cancels_001-1234abcd",'
            '"interrupt_action":"cancel",'
            '"decision":{"action":"cancel","intent":"hard_stop"},'
            '"client_control_events":[{"op":"playback.stop"}],'
            '"ptt_segment_terminal":{"action":"reject","reason":"tap_to_stop"}},'
            '"timestamps":{"speech_started_at":1.0,'
            '"interrupt_started_at":1.0,'
            '"interrupt_resolved_at":1.0,'
            '"transcript_actionable_first_at":1.0}}\n'
        ),
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].passed is True
    assert not run.cases[0].errors


def test_livekit_room_timeline_actions_count_only_resolved_interrupts(
    tmp_path,
) -> None:
    suite = load_suite("benchmark/cases/core.yaml")
    run = RunResult(
        run_id="expectation-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id="topic_switch_001",
                suite="semantic_control",
                runner="livekit_room",
                passed=True,
            )
        ],
    )
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        (
            '{"turn_id":"resolved","attrs":{"room_name":'
            '"voice-bench-topic_switch_001-1234abcd",'
            '"interrupt_action":"cancel",'
            '"decision":{"intent":"topic_switch","topic_switch_hint":true}},'
            '"timestamps":{"interrupt_started_at":10.0,'
            '"interrupt_resolved_at":10.14}}\n'
            '{"turn_id":"residual","attrs":{"room_name":'
            '"voice-bench-topic_switch_001-1234abcd",'
            '"interrupt_action":"cancel",'
            '"decision":{"intent":"topic_switch","topic_switch_hint":true},'
            '"timeline_flush_reason":"agent_audio_playback_done"},'
            '"timestamps":{"interrupt_started_at":10.6}}\n'
        ),
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].passed is True
    assert run.cases[0].metrics["timeline_actions"] == "cancel"
    assert run.cases[0].metrics["timeline_intents"] == "topic_switch"
    assert run.cases[0].metrics["timeline_decision_actions"] == "cancel,cancel"
    assert run.cases[0].metrics["timeline_decision_intents"] == ("topic_switch,topic_switch")
    assert not run.cases[0].errors


def test_livekit_room_timeline_rejected_turn_brain_forbidden_allows_setup_brain(
    tmp_path,
) -> None:
    suite = load_suite("benchmark/cases/synthetic_default_voiceprint_e2e.yaml")
    run = RunResult(
        run_id="expectation-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id="synthetic_default_backchannel_001",
                suite="synthetic_default_voiceprint_e2e",
                runner="livekit_room",
                passed=True,
            )
        ],
    )
    timeline_path = tmp_path / "turn_timeline.jsonl"
    room_name = "voice-bench-synthetic_default_backchannel_001-1234abcd"
    timeline_path.write_text(
        (
            '{"turn_id":"setup","attrs":{"room_name":"'
            + room_name
            + '","voiceprint_commit_gate":{"allowed":true}},'
            '"timestamps":{"brain_request_sent_at":1.0}}\n'
            '{"turn_id":"backchannel","attrs":{"room_name":"'
            + room_name
            + '","interrupt_action":"rollback",'
            '"decision":{"intent":"backchannel"},'
            '"voiceprint_commit_gate":{"allowed":true},'
            '"user_turn_coordinator":{"state":"rejected"}},'
            '"timestamps":{"interrupt_resolved_at":2.0}}\n'
        ),
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].passed is True
    assert not run.cases[0].errors


def test_livekit_room_backchannel_accepts_decision_rollback_without_resolved_interrupt(
    tmp_path,
) -> None:
    suite = load_suite("benchmark/cases/synthetic_default_voiceprint_e2e.yaml")
    run = RunResult(
        run_id="expectation-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id="synthetic_default_backchannel_001",
                suite="synthetic_default_voiceprint_e2e",
                runner="livekit_room",
                passed=True,
            )
        ],
    )
    timeline_path = tmp_path / "turn_timeline.jsonl"
    room_name = "voice-bench-synthetic_default_backchannel_001-1234abcd"
    timeline_path.write_text(
        (
            '{"turn_id":"setup","attrs":{"room_name":"'
            + room_name
            + '","voiceprint_commit_gate":{"allowed":true}},'
            '"timestamps":{"brain_request_sent_at":1.0}}\n'
            '{"turn_id":"backchannel","attrs":{"room_name":"'
            + room_name
            + '","interrupt_action":"rollback",'
            '"decision":{"intent":"backchannel"},'
            '"voiceprint_commit_gate":{"allowed":true},'
            '"user_turn_coordinator":{"state":"rejected"}},'
            '"timestamps":{}}\n'
        ),
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].metrics["timeline_actions"] == ""
    assert run.cases[0].metrics["timeline_decision_actions"] == "rollback"
    assert run.cases[0].passed is True
    assert not run.cases[0].errors


def test_livekit_room_timeline_rejected_turn_brain_forbidden_fails_leak(
    tmp_path,
) -> None:
    suite = load_suite("benchmark/cases/synthetic_default_voiceprint_e2e.yaml")
    run = RunResult(
        run_id="expectation-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id="synthetic_default_backchannel_001",
                suite="synthetic_default_voiceprint_e2e",
                runner="livekit_room",
                passed=True,
            )
        ],
    )
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        (
            '{"turn_id":"backchannel","attrs":{"room_name":'
            '"voice-bench-synthetic_default_backchannel_001-1234abcd",'
            '"interrupt_action":"rollback",'
            '"voiceprint_commit_gate":{"allowed":true},'
            '"user_turn_coordinator":{"state":"rejected"}},'
            '"timestamps":{"interrupt_resolved_at":2.0,'
            '"brain_request_sent_at":2.1}}\n'
        ),
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].passed is False
    assert any("rejected turns" in error for error in run.cases[0].errors)


def test_livekit_room_timeline_expectations_export_latency_metrics(
    tmp_path,
) -> None:
    suite = load_suite("benchmark/cases/core.yaml")
    run = RunResult(
        run_id="expectation-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id="hard_interrupt_001",
                suite="interrupt",
                runner="livekit_room",
                passed=True,
            )
        ],
    )
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        (
            '{"turn_id":"t1","attrs":{"room_name":'
            '"voice-bench-hard_interrupt_001-1234abcd",'
            '"interrupt_action":"cancel",'
            '"decision":{"intent":"hard_stop","hold_recheck_ms":40},'
            '"decision_events":['
            '{"action":"hold",'
            '"reason":"stable_signal_wait intent=topic_switch age_ms=0 window_ms=120",'
            '"intent":"topic_switch",'
            '"source":"turn_policy",'
            '"hold_recheck_ms":120,'
            '"transcript_preview":"换个话"},'
            '{"action":"cancel",'
            '"reason":"intent:topic_switch",'
            '"intent":"normal_interrupt",'
            '"source":"framework_completed_playback_evidence",'
            '"resolved_reason":"framework_completed_playback_evidence",'
            '"transcript_preview":"换个话 换个话题"}],'
            '"interrupted_context":{'
            '"source":"tts_in_flight",'
            '"played_seconds":1.2,'
            '"text_preview":"上一轮被打断的回答"},'
            '"transcript_ingress_events":['
            '{"transcript_preview":"换个话",'
            '"is_final":false,'
            '"event_type":"UserInputTranscribedEvent"},'
            '{"transcript_preview":"换个话题",'
            '"is_final":true,'
            '"event_type":"UserInputTranscribedEvent"},'
            '{"transcript_preview":"我们聊点别的",'
            '"is_final":false,'
            '"event_type":"UserInputTranscribedEvent"}],'
            '"transcript_ingress_recent_events":['
            '{"transcript_preview":"换个话题",'
            '"is_final":true,'
            '"event_type":"UserInputTranscribedEvent",'
            '"timeline_present":false},'
            '{"transcript_preview":"我们聊点别的",'
            '"is_final":false,'
            '"event_type":"UserInputTranscribedEvent",'
            '"timeline_present":true}],'
            '"transcript_ingress_pre_timeline_events":['
            '{"transcript_preview":"换个话题",'
            '"is_final":true,'
            '"event_type":"UserInputTranscribedEvent",'
            '"timeline_present":false}],'
            '"transcript_ingress_recent_cross_turn_events":['
            '{"transcript_preview":"换个话",'
            '"is_final":false,'
            '"event_type":"UserInputTranscribedEvent",'
            '"timeline_turn_id":"previous-turn"}],'
            '"transcript_admission_events":['
            '{"accepted":false,"reason":"agent_echo",'
            '"transcript_preview":"助手自己的回声"},'
            '{"accepted":true,"reason":"accepted",'
            '"transcript_preview":"换个话题"},'
            '{"accepted":true,"reason":"accepted",'
            '"transcript_preview":"我们聊点别的"}],'
            '"attention_admission_events":['
            '{"action":"duck_and_decide",'
            '"reason":"playback_speech_start_soft_duck",'
            '"transcript_preview":""},'
            '{"action":"observe",'
            '"reason":"playback_low_evidence_transcript:substantive_cjk_transcript",'
            '"transcript_preview":"换个话",'
            '"tier":"tier4_attention",'
            '"tier_reason":"playback_low_evidence_transcript:substantive_cjk_transcript",'
            '"state":{"agent_speaking":true,'
            '"duck_active":true,'
            '"speech_started":false,'
            '"client_state_present":true,'
            '"client_state_fresh":true,'
            '"client_playback_state":"agent_speaking",'
            '"client_state_age_ms":120,'
            '"eot_score":0.0}},'
            '{"action":"duck_and_decide",'
            '"reason":"transcript_intent:topic_switch",'
            '"transcript_preview":"换个话题",'
            '"state":{"agent_speaking":true,'
            '"duck_active":true,'
            '"speech_started":false,'
            '"client_state_present":true,'
            '"client_state_fresh":true,'
            '"client_playback_state":"agent_speaking",'
            '"client_state_age_ms":42,'
            '"eot_score":0.72}}],'
            '"semantic_interrupt_gate_events":['
            '{"stage":"initial","action":"inactive",'
            '"reason":"no_interrupt_window",'
            '"transcript_preview":"换个话题"},'
            '{"stage":"initial","action":"needs_attention",'
            '"reason":"needs_attention",'
            '"transcript_preview":"我们聊点别的"},'
            '{"stage":"attention","action":"run",'
            '"reason":"eligible",'
            '"transcript_preview":"我们聊点别的"}],'
            '"framework_completed_gate_events":['
            '{"stage":"received","action":"observe",'
            '"reason":"framework_completed_turn",'
            '"transcript_preview":"换个话题"},'
            '{"stage":"playback_check","action":"continue",'
            '"reason":"playback_active",'
            '"transcript_preview":"换个话题"},'
            '{"stage":"framework_completed_playback_evidence",'
            '"action":"cancel","reason":"intent:topic_switch",'
            '"transcript_preview":"换个话题"}],'
            '"provider_latency_ms":{'
            '"interrupt_speech_to_first_transcript_ms":310,'
            '"stt_speech_to_actionable_transcript_ms":360,'
            '"stt_first_transcript_to_actionable_transcript_ms":50,'
            '"interrupt_first_transcript_to_resolved_ms":70,'
            '"interrupt_actionable_transcript_to_resolved_ms":20,'
            '"interrupt_intent_admitted_to_resolved_ms":35,'
            '"stt_provider_partial_to_livekit_interim_ms":22,'
            '"framework_completed_after_speech_ms":1200,'
            '"framework_completed_to_cancel_resolved_ms":80,'
            '"framework_playback_evidence_to_cancel_resolved_ms":25}},'
            '"durations_ms":{"vad_start_to_interrupt_resolved":380,'
            '"interrupt_intent_admitted_to_resolved":35}}\n'
        ),
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    metrics = run.cases[0].metrics
    assert metrics["timeline_interrupt_speech_to_first_transcript_ms"] == 310
    assert metrics["timeline_stt_speech_to_actionable_transcript_ms"] == 360
    assert metrics["timeline_stt_first_transcript_to_actionable_transcript_ms"] == 50
    assert metrics["timeline_interrupt_first_transcript_to_resolved_ms"] == 70
    assert metrics["timeline_interrupt_actionable_transcript_to_resolved_ms"] == 20
    assert metrics["timeline_interrupt_intent_admitted_to_resolved_ms"] == 35
    assert metrics["timeline_decision_hold_recheck_ms"] == 40
    assert metrics["timeline_decision_event_count"] == 2
    assert metrics["timeline_decision_event_last_reason"] == "intent:topic_switch"
    assert metrics["timeline_decision_event_last_preview"] == "换个话 换个话题"
    assert metrics["timeline_decision_event_hold_chain"] == (
        "hold:stable_signal_wait intent=topic_switch age_ms=0 window_ms=120:"
        "topic_switch:turn_policy:换个话"
    )
    assert metrics["timeline_decision_stable_signal_wait_count"] == 1
    assert metrics["timeline_decision_stable_signal_last_intent"] == "topic_switch"
    assert metrics["timeline_decision_stable_signal_last_age_ms"] == 0
    assert metrics["timeline_decision_stable_signal_last_window_ms"] == 120
    assert metrics["timeline_decision_stable_signal_last_recheck_ms"] == 120
    assert metrics["timeline_decision_stable_signal_last_preview"] == "换个话"
    assert metrics["timeline_decision_stable_signal_chain"] == (
        "topic_switch:age=0:window=120:recheck=120:换个话"
    )
    assert metrics["timeline_decision_event_terminal_chain"] == (
        "cancel:intent:topic_switch:normal_interrupt:"
        "framework_completed_playback_evidence:换个话 换个话题"
    )
    assert metrics["timeline_stt_provider_partial_to_livekit_interim_ms"] == 22
    assert metrics["timeline_vad_start_to_interrupt_resolved"] == 380
    assert metrics["timeline_interrupted_context_count"] == 1
    assert metrics["timeline_interrupted_context_source"] == "tts_in_flight"
    assert metrics["timeline_interrupted_context_played_seconds"] == 1.2
    assert metrics["timeline_interrupted_context_preview"] == "上一轮被打断的回答"
    assert metrics["timeline_transcript_ingress_event_count"] == 3
    assert metrics["timeline_transcript_ingress_last_preview"] == "我们聊点别的"
    assert metrics["timeline_transcript_ingress_last_event_type"] == ("UserInputTranscribedEvent")
    assert metrics["timeline_transcript_ingress_last_final"] is False
    assert metrics["timeline_transcript_ingress_chain"] == (
        "interim:换个话 ; final:换个话题 ; interim:我们聊点别的"
    )
    assert metrics["timeline_transcript_ingress_recent_event_count"] == 2
    assert metrics["timeline_transcript_ingress_recent_chain"] == (
        "final:换个话题 ; interim:我们聊点别的"
    )
    assert metrics["timeline_transcript_ingress_pre_timeline_event_count"] == 1
    assert metrics["timeline_transcript_ingress_pre_timeline_chain"] == ("final:换个话题")
    assert metrics["timeline_transcript_ingress_recent_cross_turn_event_count"] == 1
    assert metrics["timeline_transcript_ingress_recent_cross_turn_chain"] == ("interim:换个话")
    assert metrics["timeline_transcript_admission_event_count"] == 3
    assert metrics["timeline_transcript_admission_last_reason"] == "accepted"
    assert metrics["timeline_transcript_admission_last_preview"] == "我们聊点别的"
    assert metrics["timeline_transcript_admission_last_accepted"] is True
    assert metrics["timeline_transcript_admission_rejected_chain"] == ("agent_echo:助手自己的回声")
    assert metrics["timeline_attention_admission_event_count"] == 3
    assert metrics["timeline_attention_admission_last_reason"] == ("transcript_intent:topic_switch")
    assert metrics["timeline_attention_admission_last_preview"] == "换个话题"
    assert metrics["timeline_attention_admission_last_agent_speaking"] is True
    assert metrics["timeline_attention_admission_last_duck_active"] is True
    assert metrics["timeline_attention_admission_last_playback_state"] == ("agent_speaking")
    assert metrics["timeline_attention_admission_last_client_state_age_ms"] == 42
    assert metrics["timeline_attention_admission_last_client_state_fresh"] is True
    assert metrics["timeline_attention_admission_last_eot_score"] == 0.72
    assert metrics["timeline_attention_admission_blocked_chain"] == (
        "observe:playback_low_evidence_transcript:substantive_cjk_transcript:换个话"
    )
    assert metrics["timeline_attention_admission_blocked_state_chain"] == (
        "observe:playback_low_evidence_transcript:substantive_cjk_transcript:"
        "agent=true:duck=true:playback=agent_speaking:age=120ms:eot=0.00:换个话"
    )
    assert metrics["timeline_semantic_gate_event_count"] == 3
    assert metrics["timeline_semantic_gate_last_stage"] == "attention"
    assert metrics["timeline_semantic_gate_last_action"] == "run"
    assert metrics["timeline_semantic_gate_last_reason"] == "eligible"
    assert metrics["timeline_semantic_gate_last_preview"] == "我们聊点别的"
    assert metrics["timeline_semantic_gate_blocked_chain"] == (
        "initial:inactive:no_interrupt_window:换个话题"
    )
    assert metrics["timeline_framework_completed_after_speech_ms"] == 1200
    assert metrics["timeline_framework_completed_to_cancel_resolved_ms"] == 80
    assert metrics["timeline_framework_playback_evidence_to_cancel_resolved_ms"] == 25
    assert metrics["timeline_framework_completed_gate_event_count"] == 3
    assert metrics["timeline_framework_completed_gate_last_stage"] == (
        "framework_completed_playback_evidence"
    )
    assert metrics["timeline_framework_completed_gate_last_action"] == "cancel"
    assert metrics["timeline_framework_completed_gate_last_reason"] == ("intent:topic_switch")
    assert metrics["timeline_framework_completed_gate_last_preview"] == "换个话题"
    assert metrics["timeline_framework_completed_gate_chain"] == (
        "received:observe:framework_completed_turn:换个话题 ; "
        "playback_check:continue:playback_active:换个话题 ; "
        "framework_completed_playback_evidence:cancel:intent:topic_switch:换个话题"
    )


def test_livekit_room_timeline_expectations_export_interruption_owner_metrics(
    tmp_path,
) -> None:
    suite = load_suite("benchmark/cases/core.yaml")
    run = RunResult(
        run_id="owner-metrics-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id="backchannel_001",
                suite="false_interrupt",
                runner="livekit_room",
                passed=True,
                metrics={"room_name": "voice-bench-backchannel_001-a1b2c3d4"},
            )
        ],
    )
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        (
            '{"turn_id":"t1","timestamps":{'
            '"speech_started_at":10.0,'
            '"speech_stopped_at":10.645,'
            '"interrupt_resolved_at":10.646,'
            '"interrupt_rollback_resolved_at":10.646},'
            '"attrs":{"room_name":"voice-bench-backchannel_001-a1b2c3d4",'
            '"interrupt_action":"rollback",'
            '"decision":{"intent":"backchannel"},'
            '"interruption_orchestrator_events":['
            '{"event":"candidate_started","state":"suspended_waiting_evidence",'
            '"elapsed_ms":0},'
            '{"event":"turn_policy_decision",'
            '"state":"suspended_waiting_evidence",'
            '"action":"hold",'
            '"reason":"intent:backchannel_await_more_speech",'
            '"intent":"backchannel",'
            '"transcript_preview":"好",'
            '"elapsed_ms":410,'
            '"since_last_event_ms":410},'
            '{"event":"short_false_interruption_fast_resume",'
            '"state":"suspended_waiting_evidence",'
            '"transcript_preview":"好",'
            '"last_policy_action":"hold",'
            '"last_policy_reason":"intent:backchannel_await_more_speech",'
            '"elapsed_ms":645,'
            '"since_last_event_ms":235},'
            '{"event":"candidate_resolved",'
            '"state":"confirmed_false_resume",'
            '"action":"rollback",'
            '"reason":"user_silent",'
            '"elapsed_ms":646,'
            '"since_last_event_ms":1}],'
            '"provider_latency_ms":{'
            '"interrupt_speech_to_rollback_resolved_ms":646,'
            '"speech_stop_to_rollback_resolved_ms":1}},'
            '"durations_ms":{'
            '"vad_start_to_interrupt_rollback_resolved":646,'
            '"speech_stop_to_interrupt_rollback_resolved":1}}\n'
        ),
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    metrics = run.cases[0].metrics
    assert metrics["timeline_interrupt_speech_to_rollback_resolved_ms"] == 646
    assert metrics["timeline_speech_stop_to_rollback_resolved_ms"] == 1
    assert metrics["timeline_interruption_owner_event_count"] == 4
    assert metrics["timeline_interruption_owner_last_event"] == "candidate_resolved"
    assert metrics["timeline_interruption_owner_last_reason"] == "user_silent"
    assert metrics["timeline_interruption_owner_last_elapsed_ms"] == 646
    assert metrics["timeline_interruption_owner_last_since_previous_ms"] == 1
    assert metrics["timeline_interruption_owner_fast_resume_elapsed_ms"] == 645
    assert metrics["timeline_interruption_owner_last_backchannel_hold_elapsed_ms"] == 410
    assert metrics["timeline_interruption_owner_backchannel_hold_to_resume_ms"] == 236
    assert metrics["timeline_interruption_owner_last_backchannel_hold_preview"] == "好"
    assert metrics["timeline_interruption_owner_resolved_elapsed_ms"] == 646
    assert metrics["timeline_interruption_owner_wait_chain"] == (
        "turn_policy_decision@410ms:hold:intent:backchannel_await_more_speech:好 ; "
        "short_false_interruption_fast_resume@645ms:hold:"
        "intent:backchannel_await_more_speech:好"
    )


def test_livekit_room_timeline_expectations_use_latest_backchannel_hold_gap(
    tmp_path,
) -> None:
    suite = load_suite("benchmark/cases/core.yaml")
    run = RunResult(
        run_id="owner-latest-backchannel-gap-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id="backchannel_001",
                suite="false_interrupt",
                runner="livekit_room",
                passed=True,
                metrics={"room_name": "voice-bench-backchannel_001-a1b2c3d4"},
            )
        ],
    )
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        json.dumps(
            {
                "turn_id": "t1",
                "timestamps": {
                    "speech_started_at": 10.0,
                    "speech_stopped_at": 10.651,
                    "interrupt_resolved_at": 10.651,
                    "interrupt_rollback_resolved_at": 10.651,
                },
                "attrs": {
                    "room_name": "voice-bench-backchannel_001-a1b2c3d4",
                    "interrupt_action": "rollback",
                    "decision": {"intent": "backchannel"},
                    "interruption_orchestrator_events": [
                        {
                            "event": "candidate_started",
                            "state": "suspended_waiting_evidence",
                            "elapsed_ms": 0,
                        },
                        {
                            "event": "turn_policy_decision",
                            "state": "suspended_waiting_evidence",
                            "action": "hold",
                            "reason": "intent:backchannel_await_more_speech",
                            "intent": "backchannel",
                            "transcript_preview": "啊",
                            "elapsed_ms": 307,
                        },
                        {
                            "event": "turn_policy_decision",
                            "state": "suspended_waiting_evidence",
                            "action": "hold",
                            "reason": "intent:backchannel_await_more_speech",
                            "intent": "backchannel",
                            "transcript_preview": "好",
                            "elapsed_ms": 553,
                        },
                        {
                            "event": "short_false_interruption_fast_resume",
                            "state": "suspended_waiting_evidence",
                            "transcript_preview": "好",
                            "last_policy_action": "hold",
                            "last_policy_reason": "intent:backchannel_await_more_speech",
                            "elapsed_ms": 651,
                        },
                        {
                            "event": "candidate_resolved",
                            "state": "confirmed_false_resume",
                            "action": "rollback",
                            "reason": "user_silent",
                            "elapsed_ms": 651,
                        },
                    ],
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    metrics = run.cases[0].metrics
    assert metrics["timeline_interruption_owner_last_backchannel_hold_elapsed_ms"] == 553
    assert metrics["timeline_interruption_owner_backchannel_hold_to_resume_ms"] == 98
    assert metrics["timeline_interruption_owner_last_backchannel_hold_preview"] == "好"


def test_livekit_room_timeline_expectations_report_suspended_passthrough(
    tmp_path,
) -> None:
    suite = load_suite("benchmark/cases/core.yaml")
    run = RunResult(
        run_id="owner-passthrough-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id="backchannel_001",
                suite="false_interrupt",
                runner="livekit_room",
                passed=True,
                metrics={"room_name": "voice-bench-backchannel_001-a1b2c3d4"},
            )
        ],
    )
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        json.dumps(
            {
                "turn_id": "t1",
                "timestamps": {
                    "speech_started_at": 10.0,
                    "interrupt_started_at": 10.05,
                },
                "attrs": {
                    "room_name": "voice-bench-backchannel_001-a1b2c3d4",
                    "duck_events": [
                        {"event": "duck_started", "vad_to_duck_ms": 50},
                        {
                            "event": "duck_suspended_passthrough_enabled",
                            "reason": "intent:backchannel_await_more_speech",
                            "volume": 0.2,
                            "suspend_ms": 430,
                            "buffered_frames": 2,
                            "buffered_sec": 0.02,
                        },
                        {
                            "event": "duck_unducked",
                            "reason": "timeout",
                            "suspended_passthrough_frames": 3,
                            "buffer_frames_dropped_on_passthrough": 2,
                            "suspended_passthrough_enabled_ms": 430,
                            "suspended_passthrough_first_frame_ms": 455,
                            "suspended_passthrough_last_frame_ms": 475,
                        },
                    ],
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    metrics = run.cases[0].metrics
    assert metrics["timeline_duck_suspended_passthrough_count"] == 1
    assert (
        metrics["timeline_duck_suspended_passthrough_last_reason"]
        == "intent:backchannel_await_more_speech"
    )
    assert metrics["timeline_duck_suspended_passthrough_last_volume"] == 0.2
    assert metrics["timeline_duck_suspended_passthrough_since_duck_ms"] == 430
    assert metrics["timeline_duck_speech_to_suspended_passthrough_ms"] == 480
    assert metrics["timeline_duck_suspended_passthrough_buffered_frames"] == 2
    assert metrics["timeline_duck_suspended_passthrough_buffered_sec"] == 0.02
    assert metrics["timeline_duck_suspended_passthrough_forwarded_frames"] == 3
    assert metrics["timeline_duck_suspended_passthrough_dropped_frames"] == 2
    assert metrics["timeline_duck_suspended_passthrough_first_frame_since_duck_ms"] == 455
    assert metrics["timeline_duck_speech_to_suspended_passthrough_first_frame_ms"] == 505
    assert metrics["timeline_duck_suspended_passthrough_enabled_to_first_frame_ms"] == 25
    assert metrics["timeline_duck_suspended_passthrough_last_frame_since_duck_ms"] == 475


def test_livekit_room_timeline_expectations_skip_backchannel_resume_gap_for_cancel(
    tmp_path,
) -> None:
    suite = load_suite("benchmark/cases/core.yaml")
    run = RunResult(
        run_id="owner-cancel-after-backchannel-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id="backchannel_001",
                suite="false_interrupt",
                runner="livekit_room",
                passed=True,
                metrics={"room_name": "voice-bench-backchannel_001-a1b2c3d4"},
            )
        ],
    )
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        json.dumps(
            {
                "turn_id": "t1",
                "timestamps": {
                    "speech_started_at": 10.0,
                    "interrupt_resolved_at": 10.689,
                    "interrupt_cancel_resolved_at": 10.689,
                },
                "attrs": {
                    "room_name": "voice-bench-backchannel_001-a1b2c3d4",
                    "interrupt_action": "cancel",
                    "decision": {"intent": "normal_interrupt"},
                    "interruption_orchestrator_events": [
                        {
                            "event": "turn_policy_decision",
                            "state": "suspended_waiting_evidence",
                            "action": "hold",
                            "reason": "intent:backchannel_await_more_speech",
                            "intent": "backchannel",
                            "transcript_preview": "是",
                            "elapsed_ms": 264,
                        },
                        {
                            "event": "candidate_resolved",
                            "state": "confirmed_cancelled",
                            "action": "cancel",
                            "reason": "confirmed_cancel_turn_committed",
                            "elapsed_ms": 689,
                        },
                    ],
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    metrics = run.cases[0].metrics
    assert "timeline_interruption_owner_backchannel_hold_to_resume_ms" not in metrics
    assert "timeline_interruption_owner_last_backchannel_hold_elapsed_ms" not in metrics
    assert "timeline_interruption_owner_last_backchannel_hold_preview" not in metrics


def test_livekit_room_timeline_expectations_ignore_stale_retry_room(
    tmp_path,
) -> None:
    suite = load_suite("benchmark/cases/core.yaml")
    run = RunResult(
        run_id="expectation-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id="hard_interrupt_001",
                suite="interrupt",
                runner="livekit_room",
                passed=True,
                metrics={"room_name": "voice-bench-hard_interrupt_001-retry123"},
                events=[
                    {
                        "type": "case_retry",
                        "previous_room_name": ("voice-bench-hard_interrupt_001-stale123"),
                    }
                ],
            )
        ],
    )
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        (
            '{"turn_id":"stale","attrs":{"room_name":'
            '"voice-bench-hard_interrupt_001-stale123",'
            '"timeline_flush_reason":"session_closed"}}\n'
            '{"turn_id":"retry","attrs":{"room_name":'
            '"voice-bench-hard_interrupt_001-retry123",'
            '"interrupt_action":"cancel","decision":{"intent":"hard_stop"}},'
            '"durations_ms":{"vad_start_to_interrupt_resolved":320}}\n'
        ),
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].passed is True
    assert run.cases[0].metrics["timeline_record_count"] == 1
    assert run.cases[0].metrics["timeline_actions"] == "cancel"
    assert run.cases[0].metrics["timeline_intents"] == "hard_stop"
    assert not run.cases[0].errors


def test_livekit_room_timeline_expectations_fail_slow_interrupt(
    tmp_path,
) -> None:
    suite = load_suite("benchmark/cases/core.yaml")
    run = RunResult(
        run_id="expectation-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id="hard_interrupt_001",
                suite="interrupt",
                runner="livekit_room",
                passed=True,
            )
        ],
    )
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        (
            '{"turn_id":"t1","attrs":{"room_name":'
            '"voice-bench-hard_interrupt_001-1234abcd",'
            '"interrupt_action":"cancel","decision":{"intent":"hard_stop"}},'
            '"durations_ms":{"vad_start_to_interrupt_resolved":900}}\n'
        ),
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].passed is False
    assert any("exceeded 650" in error for error in run.cases[0].errors)


def test_livekit_room_timeline_expectations_pass_fast_tier1_after_start(
    tmp_path,
) -> None:
    suite = load_suite("benchmark/cases/core.yaml")
    run = RunResult(
        run_id="expectation-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id="topic_switch_001",
                suite="semantic_control",
                runner="livekit_room",
                passed=True,
            )
        ],
    )
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        (
            '{"turn_id":"t1","attrs":{"room_name":'
            '"voice-bench-topic_switch_001-1234abcd",'
            '"interrupt_action":"cancel",'
            '"decision":{"intent":"topic_switch","topic_switch_hint":true}},'
            '"durations_ms":{"vad_start_to_interrupt_resolved":960},'
            '"timestamps":{"interrupt_started_at":10.0,'
            '"interrupt_resolved_at":10.14}}\n'
        ),
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].passed is True
    assert not run.cases[0].errors


def test_livekit_room_timeline_expectations_use_direct_intent_admission_time(
    tmp_path,
) -> None:
    suite = load_suite("benchmark/cases/core.yaml")
    run = RunResult(
        run_id="expectation-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id="topic_switch_001",
                suite="semantic_control",
                runner="livekit_room",
                passed=True,
            )
        ],
    )
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        (
            '{"turn_id":"t1","attrs":{"room_name":'
            '"voice-bench-topic_switch_001-1234abcd",'
            '"interrupt_action":"cancel",'
            '"decision":{"intent":"topic_switch","topic_switch_hint":true}},'
            '"timestamps":{"interrupt_started_at":10.0,'
            '"interrupt_intent_admitted_at":10.6,'
            '"interrupt_resolved_at":10.74}}\n'
        ),
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].passed is True
    assert not run.cases[0].errors


def test_livekit_room_timeline_expectations_fail_slow_tier1_after_start(
    tmp_path,
) -> None:
    suite = load_suite("benchmark/cases/core.yaml")
    run = RunResult(
        run_id="expectation-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id="topic_switch_001",
                suite="semantic_control",
                runner="livekit_room",
                passed=True,
            )
        ],
    )
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        (
            '{"turn_id":"t1","attrs":{"room_name":'
            '"voice-bench-topic_switch_001-1234abcd",'
            '"interrupt_action":"cancel",'
            '"decision":{"intent":"topic_switch","topic_switch_hint":true}},'
            '"timestamps":{"interrupt_started_at":10.0,'
            '"interrupt_resolved_at":10.4}}\n'
        ),
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].passed is False
    assert any("resolution-after-start exceeded 250" in error for error in run.cases[0].errors)


def test_livekit_room_timeline_expectations_use_cancel_specific_resolution(
    tmp_path,
) -> None:
    suite = load_suite("benchmark/cases/core.yaml")
    run = RunResult(
        run_id="expectation-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id="topic_switch_001",
                suite="semantic_control",
                runner="livekit_room",
                passed=True,
            )
        ],
    )
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        (
            '{"turn_id":"t1","attrs":{"room_name":'
            '"voice-bench-topic_switch_001-1234abcd",'
            '"interrupt_action":"cancel",'
            '"decision":{"intent":"topic_switch","topic_switch_hint":true}},'
            '"timestamps":{"speech_started_at":10.0,'
            '"interrupt_started_at":10.0,'
            '"interrupt_resolved_at":10.1,'
            '"interrupt_rollback_resolved_at":10.1,'
            '"interrupt_cancel_resolved_at":10.4}}\n'
        ),
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].passed is False
    assert any("resolution-after-start exceeded 250" in error for error in run.cases[0].errors)


def test_livekit_room_timeline_expectations_accept_allowed_attention_observe(
    tmp_path,
) -> None:
    suite = load_suite("benchmark/cases/core.yaml")
    run = RunResult(
        run_id="expectation-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id="backchannel_001",
                suite="false_interrupt",
                runner="livekit_room",
                passed=True,
            )
        ],
    )
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        (
            '{"turn_id":"t1","attrs":{"room_name":'
            '"voice-bench-backchannel_001-1234abcd",'
            '"attention_admission":{"action":"observe"},'
            '"attention_admission_events":[{"action":"observe"}]}}\n'
        ),
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].passed is True
    assert not run.cases[0].errors


def test_dogfood_timeline_expectations_pass_cancel_chain(tmp_path) -> None:
    suite = load_suite("benchmark/cases/dogfood_box3_audio_first_enforced.yaml")
    case_id = "dogfood_box3_owner_followup_during_playback_001"
    run = RunResult(
        run_id="dogfood-expectation-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id=case_id,
                suite="dogfood_box3_audio_first",
                runner="livekit_room",
                passed=True,
            )
        ],
    )
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        (
            '{"turn_id":"t1","attrs":{"room_name":'
            '"voice-bench-dogfood_box3_owner_followup_during_playback_001-1234abcd",'
            '"interrupt_action":"cancel",'
            '"decision":{"intent":"normal_interrupt"},'
            '"client_control_events":[{"op":"playback.stop",'
            '"reason":"interrupt_cancel"}],'
            '"interrupted_context":{"source":"tts_in_flight",'
            '"played_seconds":1.1,"text_preview":"我会先讲系统结构"}},'
            '"timestamps":{"speech_started_at":10.0,'
            '"interrupt_started_at":10.08,'
            '"interrupt_resolved_at":10.42}}\n'
        ),
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].passed is True
    assert not run.cases[0].errors


def test_dogfood_timeline_expectations_fail_missing_suspend(tmp_path) -> None:
    suite = load_suite("benchmark/cases/dogfood_box3_audio_first_enforced.yaml")
    case_id = "dogfood_box3_owner_followup_during_playback_001"
    run = RunResult(
        run_id="dogfood-expectation-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id=case_id,
                suite="dogfood_box3_audio_first",
                runner="livekit_room",
                passed=True,
            )
        ],
    )
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        (
            '{"turn_id":"t1","attrs":{"room_name":'
            '"voice-bench-dogfood_box3_owner_followup_during_playback_001-1234abcd",'
            '"attention_admission":{"action":"observe"}},'
            '"timestamps":{"speech_started_at":10.0}}\n'
        ),
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].passed is False
    assert any("speech-start-to-suspend" in error for error in run.cases[0].errors)


def test_dogfood_timeline_expectations_fail_missing_stop_and_context(
    tmp_path,
) -> None:
    suite = load_suite("benchmark/cases/dogfood_box3_audio_first_enforced.yaml")
    case_id = "dogfood_box3_owner_followup_during_playback_001"
    run = RunResult(
        run_id="dogfood-expectation-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id=case_id,
                suite="dogfood_box3_audio_first",
                runner="livekit_room",
                passed=True,
            )
        ],
    )
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        (
            '{"turn_id":"t1","attrs":{"room_name":'
            '"voice-bench-dogfood_box3_owner_followup_during_playback_001-1234abcd",'
            '"interrupt_action":"cancel",'
            '"decision":{"intent":"normal_interrupt"}},'
            '"timestamps":{"speech_started_at":10.0,'
            '"interrupt_started_at":10.04,'
            '"interrupt_resolved_at":10.28}}\n'
        ),
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].passed is False
    assert any("playback.stop" in error for error in run.cases[0].errors)
    assert any("interrupted_context" in error for error in run.cases[0].errors)


def test_dogfood_timeline_expectations_pass_false_interrupt_resume(
    tmp_path,
) -> None:
    suite = load_suite("benchmark/cases/dogfood_box3_audio_first_enforced.yaml")
    case_id = "dogfood_box3_backchannel_during_playback_001"
    run = RunResult(
        run_id="dogfood-expectation-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id=case_id,
                suite="dogfood_box3_audio_first",
                runner="livekit_room",
                passed=True,
            )
        ],
    )
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        (
            '{"turn_id":"t1","attrs":{"room_name":'
            '"voice-bench-dogfood_box3_backchannel_during_playback_001-1234abcd",'
            '"rollback_reason":"backchannel",'
            '"duck_events":[{"event":"duck_started","vad_to_duck_ms":55},'
            '{"event":"duck_unducked","reason":"backchannel"}]},'
            '"timestamps":{"speech_started_at":10.0,'
            '"interrupt_resolved_at":10.44}}\n'
        ),
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].passed is True
    assert not run.cases[0].errors


def test_hil_barge_in_analyzer_passes_cancel_chain(tmp_path) -> None:
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        (
            '{"turn_id":"r1","attrs":{"room_name":"real-box3-room",'
            '"timeline_flush_reason":"interrupted_by_user",'
            '"interrupted_context":{"played_seconds":1.2}}}\n'
            '{"turn_id":"t1","attrs":{"room_name":"real-box3-room",'
            '"interruption_target":{"response_turn_id":"r1"},'
            '"attention_admission":{"action":"duck_and_decide",'
            '"reason":"playback_speech_start_soft_duck"},'
            '"interrupt_action":"cancel",'
            '"client_control_events":[{"op":"playback.stop"}],'
            '"duck_events":[{"event":"duck_started","vad_to_duck_ms":45},'
            '{"event":"duck_cancelled"}]},'
            '"timestamps":{"speech_started_at":10.0,'
            '"interrupt_started_at":10.04,'
            '"interrupt_resolved_at":10.32}}\n'
        ),
        encoding="utf-8",
    )

    report = analyze_hil_barge_in(
        timeline_path,
        room_contains="box3",
        latest=2,
        require_cancel=True,
    )

    assert report.passed is True
    assert report.evidence["speech_start_to_suspend_ms"] == pytest.approx(40.0)
    assert report.evidence["speech_start_to_cancel_ms"] == pytest.approx(320.0)
    assert report.evidence["target_response_turn_ids"] == ["r1"]
    assert not report.findings


def test_hil_barge_in_analyzer_fails_observe_only_path(tmp_path) -> None:
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        (
            '{"turn_id":"r1","attrs":{"room_name":"real-box3-room"}}\n'
            '{"turn_id":"t1","attrs":{"room_name":"real-box3-room",'
            '"interruption_target":{"response_turn_id":"r1"},'
            '"attention_admission":{"action":"observe",'
            '"reason":"client_playback_active_without_direct_signal"}},'
            '"timestamps":{"speech_started_at":10.0}}\n'
        ),
        encoding="utf-8",
    )

    report = analyze_hil_barge_in(timeline_path, room_contains="box3")

    assert report.passed is False
    assert "missing playback_speech_start_soft_duck admission" in report.findings
    assert any("old blocked path" in finding for finding in report.findings)


def test_hil_barge_in_analyzer_passes_resume_chain(tmp_path) -> None:
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        (
            '{"turn_id":"r1","attrs":{"room_name":"real-box3-room"}}\n'
            '{"turn_id":"t1","attrs":{"room_name":"real-box3-room",'
            '"interruption_target":{"response_turn_id":"r1"},'
            '"attention_admission":{"action":"duck_and_decide",'
            '"reason":"playback_speech_start_soft_duck"},'
            '"rollback_reason":"backchannel",'
            '"duck_events":[{"event":"duck_started","vad_to_duck_ms":50},'
            '{"event":"duck_unducked"}]},'
            '"timestamps":{"speech_started_at":10.0,'
            '"interrupt_started_at":10.05,'
            '"interrupt_resolved_at":10.58}}\n'
        ),
        encoding="utf-8",
    )

    report = analyze_hil_barge_in(
        timeline_path,
        room_contains="box3",
        require_resume=True,
    )

    assert report.passed is True
    assert report.evidence["speech_start_to_resume_ms"] == pytest.approx(580.0)


def test_hil_barge_in_ignores_unscoped_stop_after_response_completion(tmp_path) -> None:
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        (
            '{"turn_id":"r1","attrs":{"room_name":"real-box3-room",'
            '"timeline_flush_reason":"interrupted_by_user",'
            '"interrupted_context":{"played_seconds":1.2}}}\n'
            '{"turn_id":"cancel-in-playback","attrs":{"room_name":"real-box3-room",'
            '"interruption_target":{"response_turn_id":"r1"},'
            '"attention_admission":{"reason":"playback_speech_start_soft_duck"},'
            '"interrupt_action":"cancel",'
            '"client_control_events":[{"op":"playback.stop"}],'
            '"duck_events":[{"event":"duck_started"}]},'
            '"durations_ms":{"vad_start_to_interrupt_cancel_resolved":240}}\n'
            '{"turn_id":"post-playback-stop","attrs":{"room_name":"real-box3-room",'
            '"interrupt_action":"cancel",'
            '"client_control_events":[{"op":"playback.stop"}],'
            '"duck_events":[{"event":"duck_started"}]},'
            '"durations_ms":{"vad_start_to_interrupt_cancel_resolved":550}}\n'
        ),
        encoding="utf-8",
    )

    report = analyze_hil_barge_in(
        timeline_path,
        room_contains="box3",
        latest=2,
        require_cancel=True,
    )

    assert report.passed is True
    assert report.evidence["speech_start_to_cancel_ms"] == pytest.approx(240.0)
    assert report.evidence["qualified_interrupt_count"] == 1
    assert report.evidence["unscoped_cancel_count"] == 1


def test_load_conversation_turn_taking_suite() -> None:
    suite = load_suite("benchmark/cases/conversation_turn_taking.yaml")

    assert suite.suite_id == "conversation_turn_taking"
    assert {case.case_id for case in suite.cases} == {
        "pause_mid_utterance_single_turn_001",
        "hesitation_filler_start_001",
        "quick_followup_merge_001",
        "eot_prompt_commit_001",
        "multi_turn_three_rounds_001",
        "backchannel_resume_001",
        "interrupt_then_new_turn_001",
    }
    pause = next(
        case for case in suite.cases if case.case_id == "pause_mid_utterance_single_turn_001"
    )
    assert pause.expectations.max_brain_requests == 1
    multi = next(case for case in suite.cases if case.case_id == "multi_turn_three_rounds_001")
    assert multi.expectations.min_brain_requests == 3
    eot = next(case for case in suite.cases if case.case_id == "eot_prompt_commit_001")
    assert eot.expectations.max_speech_stop_to_commit_ms == 1500
    assert eot.expectations.max_user_done_to_agent_audio_ms == 4000
    resume = next(case for case in suite.cases if case.case_id == "backchannel_resume_001")
    assert resume.expectations.rejected_turn_brain == "forbidden"
    assert resume.expectations.max_user_done_to_agent_audio_ms == 2500


def _expectation_run(case_id: str, suite_name: str) -> RunResult:
    return RunResult(
        run_id="expectation-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id=case_id,
                suite=suite_name,
                runner="livekit_room",
                passed=True,
            )
        ],
    )


def _committed_turn_record(case_id: str, *, turn_id: str) -> str:
    return json.dumps(
        {
            "turn_id": turn_id,
            "attrs": {
                "room_name": f"voice-bench-{case_id}-1234abcd",
                "canonical_user_text": {"text_preview": "实时语音方案的风险和天气"},
            },
            "timestamps": {"brain_request_sent_at": 10.0},
        }
    )


def test_timeline_expectations_fail_split_turn_with_max_brain_requests(
    tmp_path,
) -> None:
    suite = load_suite("benchmark/cases/conversation_turn_taking.yaml")
    run = _expectation_run("pause_mid_utterance_single_turn_001", "turn_boundary")
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        _committed_turn_record("pause_mid_utterance_single_turn_001", turn_id="t1")
        + "\n"
        + _committed_turn_record("pause_mid_utterance_single_turn_001", turn_id="t2")
        + "\n",
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].passed is False
    assert any("<=1 brain requests, got 2" in error for error in run.cases[0].errors)


def test_timeline_expectations_require_min_brain_requests(tmp_path) -> None:
    suite = load_suite("benchmark/cases/conversation_turn_taking.yaml")
    run = _expectation_run("multi_turn_three_rounds_001", "conversation_flow")
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        _committed_turn_record("multi_turn_three_rounds_001", turn_id="t1")
        + "\n"
        + _committed_turn_record("multi_turn_three_rounds_001", turn_id="t2")
        + "\n",
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].passed is False
    assert any(">=3 brain requests, got 2" in error for error in run.cases[0].errors)


def test_timeline_expectations_pass_single_merged_turn(tmp_path) -> None:
    suite = load_suite("benchmark/cases/conversation_turn_taking.yaml")
    run = _expectation_run("pause_mid_utterance_single_turn_001", "turn_boundary")
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        _committed_turn_record("pause_mid_utterance_single_turn_001", turn_id="t1") + "\n",
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].passed is True
    assert not run.cases[0].errors


def test_timeline_expectations_fail_slow_speech_stop_to_commit(tmp_path) -> None:
    suite = load_suite("benchmark/cases/conversation_turn_taking.yaml")
    run = _expectation_run("eot_prompt_commit_001", "eot_latency")
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        json.dumps(
            {
                "turn_id": "t1",
                "attrs": {
                    "room_name": "voice-bench-eot_prompt_commit_001-1234abcd",
                    "provider_latency_ms": {"speech_stop_to_commit_ms": 2400.0},
                },
                "timestamps": {"brain_request_sent_at": 10.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].passed is False
    assert any("speech-stop-to-commit exceeded 1500" in error for error in run.cases[0].errors)


def test_timeline_expectations_speech_stop_to_commit_timestamp_fallback(
    tmp_path,
) -> None:
    suite = load_suite("benchmark/cases/conversation_turn_taking.yaml")
    run = _expectation_run("eot_prompt_commit_001", "eot_latency")
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        json.dumps(
            {
                "turn_id": "t1",
                "attrs": {"room_name": "voice-bench-eot_prompt_commit_001-1234abcd"},
                "timestamps": {
                    "speech_stopped_at": 5.0,
                    "turn_committed_at": 5.4,
                    "brain_request_sent_at": 5.5,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].passed is True
    assert not run.cases[0].errors


def test_room_user_done_audio_latency_bound() -> None:
    from benchmark.livekit_room_runner import (
        _user_done_audio_latency_errors,
    )

    suite = load_suite("benchmark/cases/conversation_turn_taking.yaml")
    case = next(case for case in suite.cases if case.case_id == "backchannel_resume_001")

    in_bound = {"user_done_to_agent_audio_after_user_done_ms": 1200}
    assert _user_done_audio_latency_errors(case, in_bound) == []

    too_slow = {"user_done_to_agent_audio_after_user_done_ms": 3000}
    assert any("too slow" in e for e in _user_done_audio_latency_errors(case, too_slow))

    assert any("missing" in e for e in _user_done_audio_latency_errors(case, {}))

    unbounded = next(case for case in suite.cases if case.case_id == "multi_turn_three_rounds_001")
    assert _user_done_audio_latency_errors(unbounded, {}) == []


def test_event_recorder_waits_for_multiple_agent_messages() -> None:
    from types import SimpleNamespace

    from eidolon.livekit.tests._harness.headless import EventRecorder, RecordedEvent

    recorder = EventRecorder(SimpleNamespace(on=lambda event_type, callback: None))

    def assistant_message(text: str) -> RecordedEvent:
        item = SimpleNamespace(role="assistant", text_content=text)
        return RecordedEvent(type="conversation_item_added", payload=SimpleNamespace(item=item))

    async def scenario() -> None:
        recorder._events.append(assistant_message("first"))
        wait = asyncio.create_task(recorder.wait_for_agent_messages(2, timeout=2.0, poll_ms=5))
        await asyncio.sleep(0.05)
        assert not wait.done()
        recorder._events.append(assistant_message("second"))
        await asyncio.wait_for(wait, timeout=1.0)

        with pytest.raises(TimeoutError, match="expected 3 agent messages"):
            await recorder.wait_for_agent_messages(3, timeout=0.05, poll_ms=5)

    asyncio.run(scenario())


def test_synthesize_composite_pcm_inserts_silence() -> None:
    from benchmark.audio_assets import (
        silence_pcm,
        synthesize_composite_pcm,
    )

    sample_rate = 16_000
    speech = b"\x01\x02" * sample_rate  # 1s of non-zero PCM

    class FakeFrame:
        def __init__(self, data: bytes, rate: int) -> None:
            self.data = data
            self.sample_rate = rate

    async def fake_synthesize(text: str):
        yield FakeFrame(speech, sample_rate)

    async def scenario() -> None:
        pcm, rate = await synthesize_composite_pcm(
            fake_synthesize, [("你好。", 500), ("再见。", 0)]
        )
        assert rate == sample_rate
        expected_silence = silence_pcm(500, sample_rate=sample_rate)
        assert len(pcm) == len(speech) * 2 + len(expected_silence)
        assert pcm[len(speech) : len(speech) + len(expected_silence)] == expected_silence

    asyncio.run(scenario())
