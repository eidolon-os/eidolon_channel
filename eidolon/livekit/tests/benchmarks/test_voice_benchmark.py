"""Voice benchmark infrastructure tests."""

from __future__ import annotations

import asyncio
import contextlib
import json
import statistics
from dataclasses import replace
from unittest.mock import AsyncMock

import jwt
import pytest

from eidolon.livekit.benchmarks.compare import compare_metrics
from eidolon.livekit.benchmarks.dashboard import DashboardRunner, write_dashboard
from eidolon.livekit.benchmarks.realcall import (
    apply_real_call_verification,
    verify_provider_config,
    verify_real_call,
)
from eidolon.livekit.benchmarks.report import (
    aggregate,
    aggregate_runs,
    write_repeated_reports,
)
from eidolon.livekit.benchmarks.schema import CaseResult, RunResult, load_suite
from eidolon.livekit.benchmarks.policy_runner import run_policy_suite
from eidolon.livekit.benchmarks.slo import (
    DEFAULT_SLO_GATES,
    SloGate,
    enforcement_failures,
    evaluate_slo_gates,
)
from eidolon.livekit.common.config import AttentionPolicyConfig, TurnPolicyConfig
from eidolon.livekit.benchmarks.timeline import (
    TimelineCapture,
    load_timeline_records,
    summarize_timeline_records,
)
from eidolon.livekit.benchmarks.timeline_expectations import apply_timeline_expectations
from scripts.bench_voice import _default_cases


def test_load_core_benchmark_suite() -> None:
    suite = load_suite("benchmarks/cases/core.yaml")

    assert suite.suite_id == "core_voice_baseline"
    assert {case.case_id for case in suite.cases} >= {
        "normal_single_turn_001",
        "hard_interrupt_001",
        "backchannel_001",
    }


def test_load_attention_admission_benchmark_suite() -> None:
    suite = load_suite("benchmarks/cases/attention_admission_baseline.yaml")

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
        case
        for case in suite.cases
        if case.case_id == "cough_noise_then_ambient_speech_holds_001"
    )
    assert cough_case.expectations.action == "any"
    assert "cancel" in cough_case.expectations.forbid_actions
    assert "known_gap" not in cough_case.tags
    backchannel_case = next(
        case
        for case in suite.cases
        if case.case_id == "short_backchannel_rolls_back_001"
    )
    assert backchannel_case.expectations.action == "rollback"
    assert "cancel" in backchannel_case.expectations.forbid_actions


def test_load_attention_admission_enforced_suite() -> None:
    suite = load_suite("benchmarks/cases/attention_admission_enforced.yaml")

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
    assert ambient_case.expectations.action == "none"
    assert "cancel" in ambient_case.expectations.forbid_actions


def test_default_voice_benchmark_cases_skip_enforced_suites() -> None:
    cases = [path.rsplit("/", 1)[-1] for path in _default_cases()]

    assert "attention_admission_baseline.yaml" in cases
    assert "attention_admission_enforced.yaml" not in cases
    assert "v1_interrupt_tiers_enforced.yaml" not in cases
    assert "v1_realistic_interaction_flows_enforced.yaml" not in cases


def test_policy_runner_attention_enforced_suite() -> None:
    suite = load_suite("benchmarks/cases/attention_admission_enforced.yaml")
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
    assert ambient.metrics["actual_action"] == "none"
    assert ambient.decisions[0]["attention_admission"]["action"] == "observe"
    assert ambient.decisions[0]["attention_admission"]["client_state_used"] is True
    assert ambient.decisions[0]["decision"] is None


def test_load_v1_interrupt_tiers_enforced_suite() -> None:
    suite = load_suite("benchmarks/cases/v1_interrupt_tiers_enforced.yaml")

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
        case
        for case in suite.cases
        if case.case_id == "tier4_ambient_speech_enforced_observes_001"
    )
    assert ambient.user_steps[0].client_playback_state == "agent_speaking"
    assert ambient.expectations.action == "none"
    assert "cancel" in ambient.expectations.forbid_actions


def test_policy_runner_v1_interrupt_tiers_enforced_suite() -> None:
    suite = load_suite("benchmarks/cases/v1_interrupt_tiers_enforced.yaml")
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
    assert tier4.metrics["actual_action"] == "none"
    assert tier4.decisions[0]["attention_admission"]["action"] == "observe"
    assert tier4.decisions[0]["decision"] is None


def test_load_v1_realistic_interaction_flows_enforced_suite() -> None:
    suite = load_suite("benchmarks/cases/v1_realistic_interaction_flows_enforced.yaml")

    assert suite.suite_id == "v1_realistic_interaction_flows_enforced"
    assert len(suite.cases) == 9
    assert {case.case_id for case in suite.cases} >= {
        "flow_normal_question_then_agent_reply_001",
        "flow_normal_question_then_hard_stop_001",
        "flow_normal_question_then_topic_switch_001",
        "flow_normal_question_then_correction_001",
        "flow_normal_question_then_followup_no_client_state_001",
        "flow_normal_question_then_backchannel_001",
        "flow_normal_question_then_noise_001",
        "flow_ambient_speech_during_agent_playback_001",
        "flow_mic_muted_hard_stop_is_ignored_001",
    }
    multi_step = [
        case
        for case in suite.cases
        if case.case_id != "flow_normal_question_then_agent_reply_001"
    ]
    assert all(len(case.user_steps) == 2 for case in multi_step)
    assert all(case.user_steps[0].agent_speaking is False for case in multi_step)
    assert all(case.user_steps[1].agent_speaking is True for case in multi_step)


def test_policy_runner_v1_realistic_interaction_flows_enforced_suite() -> None:
    suite = load_suite("benchmarks/cases/v1_realistic_interaction_flows_enforced.yaml")
    policy = TurnPolicyConfig(
        attention=replace(AttentionPolicyConfig(), enforce=True),
    )

    run = run_policy_suite([suite], turn_policy=policy, run_id="test")

    pass_by_case = {case.case_id: case.passed for case in run.cases}
    assert pass_by_case == {
        "flow_normal_question_then_agent_reply_001": True,
        "flow_normal_question_then_hard_stop_001": True,
        "flow_normal_question_then_topic_switch_001": True,
        "flow_normal_question_then_correction_001": True,
        # Current known gap: Tier2 ordinary follow-up is held by the
        # weak-signal follow-up window in pure policy simulation.
        "flow_normal_question_then_followup_no_client_state_001": False,
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
    assert any("expected action=cancel" in error for error in followup.errors)
    ambient = next(
        case
        for case in run.cases
        if case.case_id == "flow_ambient_speech_during_agent_playback_001"
    )
    assert ambient.metrics["actual_action"] == "none"
    assert ambient.decisions[-1]["attention_admission"]["action"] == "observe"
    muted = next(
        case
        for case in run.cases
        if case.case_id == "flow_mic_muted_hard_stop_is_ignored_001"
    )
    assert muted.metrics["actual_action"] == "none"
    assert muted.decisions[-1]["attention_admission"]["action"] == "ignore"


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
    assert "repeats: `2`" in markdown
    assert "Per-Case Stability" in markdown


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
    low = CaseResult(
        "c:tts", "s:tts", "component", True, metrics={"tts_audio_bytes": 10}
    )
    assert verify_real_call(low, runner="component", provider_config=cfg)
    ok = CaseResult(
        "c:tts", "s:tts", "component", True, metrics={"tts_audio_bytes": 50_000}
    )
    assert verify_real_call(ok, runner="component", provider_config=cfg) == []


def test_verify_real_call_component_stt_empty_allowed_only_for_noise() -> None:
    cfg = {"brain": "eidolon_agent", "stt": "bailian", "tts": "bailian", "vad": "firered"}
    normal_empty = CaseResult(
        "c:stt", "s:stt", "component", True, metrics={"stt_nonempty": False}
    )
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
    assert verify_real_call(ok, runner="livekit_room", provider_config=cfg, case_records=[rec]) == []
    # low audio fails per-case
    low = CaseResult("x", "s", "livekit_room", True, metrics={"agent_audio_bytes": 10})
    assert verify_real_call(low, runner="livekit_room", provider_config=cfg, case_records=[rec])
    # wrong/absent STT stream fails per-case
    no_stt = {"attrs": {"room_name": "voice-bench-x-12345678"}, "timestamps": {}}
    assert verify_real_call(ok, runner="livekit_room", provider_config=cfg, case_records=[no_stt])


def test_apply_real_call_verification_run_level_brain(tmp_path) -> None:
    cfg = {"brain": "eidolon_agent", "stt": "bailian", "tts": "bailian", "vad": "firered"}

    def make_run() -> RunResult:
        return RunResult(
            run_id="r", git_sha="s", runner="livekit_room", profile="p",
            cases=[
                CaseResult("normal_x", "s", "livekit_room", True, metrics={"agent_audio_bytes": 40_000}),
                CaseResult("hard_y", "s", "livekit_room", True, metrics={"agent_audio_bytes": 40_000}),
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
        cases=[
            CaseResult("c:tts", "s:tts", "component", True, metrics={"tts_audio_bytes": 5})
        ],
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
    from eidolon.livekit.benchmarks.livekit_room_runner import _RoomCaseState

    state = _RoomCaseState(started=0.0, events=[])
    assert not state.agent_connected.is_set()

    state.mark("participant_connected_at")

    assert state.agent_connected.is_set()


@pytest.mark.asyncio
async def test_livekit_room_publishes_client_audio_state() -> None:
    from eidolon.livekit.agent.client_audio_state import CLIENT_AUDIO_STATE_TOPIC
    from eidolon.livekit.benchmarks.livekit_room_runner import (
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
    assert payload["playback_state"] == "agent_speaking"
    assert payload["ptt"] is True
    assert payload["manual_interrupt"] is True
    assert payload["mic_muted"] is True
    assert local_participant.publish_data.await_args.kwargs == {
        "reliable": False,
        "topic": CLIENT_AUDIO_STATE_TOPIC,
    }
    assert events[-1]["type"] == "client_audio_state_published"
    assert events[-1]["ptt"] is True
    assert events[-1]["manual_interrupt"] is True
    assert events[-1]["mic_muted"] is True


@pytest.mark.asyncio
async def test_livekit_room_refreshes_client_audio_state_periodically() -> None:
    from eidolon.livekit.benchmarks.livekit_room_runner import (
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
    await asyncio.sleep(0.035)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert local_participant.publish_data.await_count >= 2
    assert {event["playback_state"] for event in events} == {"agent_speaking"}


def test_livekit_dispatch_token_includes_participant_metadata() -> None:
    from eidolon.livekit.benchmarks.livekit_room_runner import (
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
    assert json.loads(payload["metadata"]) == {
        "client": "bench",
        "kind": "user",
    }


def test_timeline_records_are_summarized(tmp_path) -> None:
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        "\n".join(
            [
                (
                    '{"turn_id":"t1","durations_ms":{"commit_to_tts_first_audio":100},'
                    '"timestamps":{"speech_started_at":1.0,"transcript_final_at":1.2},'
                    '"attrs":{"decision_reason":"intent:hard_stop",'
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


def test_dashboard_includes_timeline_section(tmp_path) -> None:
    from eidolon.livekit.benchmarks.report import write_reports

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
    enforced = {
        g.name for g in DEFAULT_SLO_GATES if g.required and g.source == "summary"
    }
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
        {"name": "livekit_room", "summary": {"metrics": {}}, "timeline": {"latencies": {
            "commit_to_tts_first_audio": {"p50": 9999.0, "p95": 9999.0, "count": 3},
        }}}
    )
    assert enforcement_failures(advisory) == []


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
            '"tts_request_started_at":2.0,"tts_provider_first_audio_at":2.15,'
            '"tts_first_audio_at":2.20},'
            '"attrs":{"room_name":"voice-bench-normal_single_turn_001-1234abcd"}}\n'
        ),
        encoding="utf-8",
    )

    summary = summarize_timeline_records(load_timeline_records(timeline_path))
    segments = summary["provider_segments"]

    assert round(segments["tts_request_to_first_audio"]["p50"]) == 150
    assert round(segments["tts_provider_to_agent_audio"]["p50"]) == 50
    assert round(segments["stt_final_after_commit"]["p50"]) == 800


def test_dashboard_findings_include_slo_failure(tmp_path) -> None:
    from eidolon.livekit.benchmarks.report import write_reports

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
    from eidolon.livekit.benchmarks.report import write_reports

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
    from eidolon.livekit.benchmarks.report import write_reports

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
        (
            '{"turn_id":"t1","attrs":'
            '{"room_name":"voice-bench-covered-1234abcd"}}\n'
        ),
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
    suite = load_suite("benchmarks/cases/core.yaml")
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
            '"interrupt_action":"cancel","decision":{"intent":"normal_interrupt"}}}\n'
        ),
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].passed is False
    assert "expected no interrupt action" in run.cases[0].errors[0]


def test_livekit_room_timeline_expectations_pass_expected_hard_stop(
    tmp_path,
) -> None:
    suite = load_suite("benchmarks/cases/core.yaml")
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


def test_livekit_room_timeline_expectations_fail_slow_interrupt(
    tmp_path,
) -> None:
    suite = load_suite("benchmarks/cases/core.yaml")
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
    assert any("exceeded 500" in error for error in run.cases[0].errors)
