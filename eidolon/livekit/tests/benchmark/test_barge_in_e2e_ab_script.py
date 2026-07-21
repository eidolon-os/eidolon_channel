"""Tests for the real-room barge-in A/B orchestration script."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.bench_barge_in_e2e_ab import (
    DEFAULT_CASES,
    DEFAULT_SUITE_SET,
    _artifact_case_labels,
    _fmt_ms,
    _case_paths_for_args,
    _filter_suites_by_case_ids,
    _overlay_payload,
    _participant_metadata,
    _profile_brief,
    _recommendation,
    _suites_for_livekit_mode,
    _summary_stat,
    _validate_room_case_expectations,
)
from benchmark.schema import BenchmarkCase, BenchmarkSuite, Expectations


def test_overlay_payload_isolates_owner_port_and_timeline() -> None:
    payload = _overlay_payload(
        interruption_owner="channel",
        server_port=18766,
        timeline_path=Path("runs/channel/worker-turn-timeline.jsonl"),
    )

    assert payload == {
        "core": {"port": 18766},
        "worker": {"num_idle_processes": 0},
        "turn_policy": {
            "interruption_owner": "channel",
            "attention": {"enforce": True},
        },
        "observability": {
            "timeline_debug_path": "runs/channel/worker-turn-timeline.jsonl",
        },
    }


def test_overlay_payload_can_enable_suspended_passthrough() -> None:
    payload = _overlay_payload(
        interruption_owner="channel",
        suspended_passthrough_enabled=True,
        suspended_passthrough_volume=0.2,
        server_port=18766,
        timeline_path=Path("runs/channel/worker-turn-timeline.jsonl"),
    )

    assert payload["turn_policy"]["ducking"] == {
        "suspended_passthrough_enabled": True,
        "suspended_passthrough_volume": 0.2,
    }
    assert payload["turn_policy"]["attention"] == {"enforce": True}


def test_summary_stat_reads_report_metric_distribution() -> None:
    payload = {
        "summary": {
            "metrics": {
                "timeline_interrupt_speech_to_resolved_ms": {
                    "p50": 110.0,
                    "p95": 180.0,
                }
            }
        }
    }

    assert _summary_stat(payload, "timeline_interrupt_speech_to_resolved_ms") == 180.0
    assert (
        _summary_stat(
            payload,
            "timeline_interrupt_speech_to_resolved_ms",
            "p50",
        )
        == 110.0
    )
    assert _summary_stat(payload, "missing") is None


def test_participant_metadata_defaults_to_full_duplex_user_initiated() -> None:
    class Args:
        livekit_interaction_mode = "full_duplex"
        livekit_session_intent = "user_initiated"

    assert _participant_metadata(Args()) == {
        "interaction_mode": "full_duplex",
        "session_intent": "user_initiated",
    }


def test_participant_metadata_allows_half_duplex_override() -> None:
    class Args:
        livekit_interaction_mode = "half_duplex"
        livekit_session_intent = ""

    assert _participant_metadata(Args()) == {"interaction_mode": "half_duplex"}


def test_participant_metadata_allows_ptt_override() -> None:
    class Args:
        livekit_interaction_mode = "ptt"
        livekit_session_intent = ""

    assert _participant_metadata(Args()) == {"interaction_mode": "ptt"}


def test_default_suite_set_is_full_duplex_gate_without_legacy() -> None:
    assert DEFAULT_SUITE_SET == "full_duplex_gate"
    assert list(DEFAULT_CASES) == [
        "benchmark/cases/full_duplex/gate_enforced.yaml",
        "benchmark/cases/full_duplex/explicit_control_enforced.yaml",
    ]
    assert all("/legacy/" not in path for path in DEFAULT_CASES)


def test_case_paths_use_suite_set_when_cases_omitted() -> None:
    class Args:
        cases = None
        suite_set = "half_duplex_ptt_phase_a"

    assert _case_paths_for_args(Args()) == [
        "benchmark/cases/half_duplex/ptt_phase_a_enforced.yaml"
    ]


def test_livekit_mode_rejects_half_full_mixed_explicit_cases() -> None:
    suites = [
        BenchmarkSuite("full", "full_duplex", ()),
        BenchmarkSuite("half", "half_duplex", ()),
    ]

    with pytest.raises(SystemExit, match="suite_mode does not match"):
        _suites_for_livekit_mode(
            suites,
            interaction_mode="full_duplex",
            allow_suite_set_filter=False,
        )


def test_all_suite_set_can_filter_to_requested_mode() -> None:
    suites = [
        BenchmarkSuite("full", "full_duplex", ()),
        BenchmarkSuite("half", "half_duplex", ()),
        BenchmarkSuite("shared", "shared", ()),
    ]

    selected = _suites_for_livekit_mode(
        suites,
        interaction_mode="half_duplex",
        allow_suite_set_filter=True,
    )

    assert [suite.suite_id for suite in selected] == ["half", "shared"]


def test_livekit_mode_ptt_selects_ptt_suite() -> None:
    suites = [
        BenchmarkSuite("full", "full_duplex", ()),
        BenchmarkSuite("ptt", "ptt", ()),
        BenchmarkSuite("shared", "shared", ()),
    ]

    selected = _suites_for_livekit_mode(
        suites,
        interaction_mode="ptt",
        allow_suite_set_filter=True,
    )

    assert [suite.suite_id for suite in selected] == ["ptt", "shared"]


def test_case_id_filter_selects_cases_without_copying_yaml() -> None:
    keep = BenchmarkCase(
        case_id="keep",
        suite="test",
        description="",
        audio_clips=(),
        user_steps=(),
        expectations=Expectations(agent_audio_response="first"),
    )
    skip = BenchmarkCase(
        case_id="skip",
        suite="test",
        description="",
        audio_clips=(),
        user_steps=(),
        expectations=Expectations(agent_audio_response="first"),
    )
    suites = [BenchmarkSuite("suite", "full_duplex", (keep, skip))]

    selected = _filter_suites_by_case_ids(suites, ("keep",))

    assert len(selected) == 1
    assert [case.case_id for case in selected[0].cases] == ["keep"]


def test_case_id_filter_reports_unknown_ids() -> None:
    case = BenchmarkCase(
        case_id="known",
        suite="test",
        description="",
        audio_clips=(),
        user_steps=(),
        expectations=Expectations(agent_audio_response="first"),
    )

    with pytest.raises(SystemExit, match="unknown --case-id"):
        _filter_suites_by_case_ids(
            [BenchmarkSuite("suite", "full_duplex", (case,))],
            ("missing",),
        )


def test_artifact_case_labels_use_case_ids_after_filter() -> None:
    case = BenchmarkCase(
        case_id="fd_gate_backchannel_resumes_001",
        suite="test",
        description="",
        audio_clips=(),
        user_steps=(),
        expectations=Expectations(agent_audio_response="first"),
    )

    assert _artifact_case_labels(
        suites=[BenchmarkSuite("suite", "full_duplex", (case,))],
        case_paths=["benchmark/cases/full_duplex/gate_enforced.yaml"],
        filtered=True,
    ) == ["fd_gate_backchannel_resumes_001"]
    assert _artifact_case_labels(
        suites=[BenchmarkSuite("suite", "full_duplex", (case,))],
        case_paths=["benchmark/cases/full_duplex/gate_enforced.yaml"],
        filtered=False,
    ) == ["benchmark/cases/full_duplex/gate_enforced.yaml"]


def test_room_case_expectations_require_explicit_agent_audio_response() -> None:
    case = BenchmarkCase(
        case_id="implicit",
        suite="test",
        description="",
        audio_clips=(),
        user_steps=(),
        expectations=Expectations(agent_audio_response="auto"),
    )
    suite = BenchmarkSuite("suite", "full_duplex", (case,))

    with pytest.raises(SystemExit, match="agent_audio_response"):
        _validate_room_case_expectations([suite])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "-"),
        (12, "12.0"),
        (12.345, "12.3"),
    ],
)
def test_fmt_ms(raw: object, expected: str) -> None:
    assert _fmt_ms(raw) == expected


def test_profile_brief_extracts_failed_cases_and_key_latencies(tmp_path: Path) -> None:
    class Result:
        profile = "channel"
        output_dir = tmp_path
        overlay_path = tmp_path / "overlay.yaml"
        worker_log_path = tmp_path / "worker.log"
        timeline = {"total_records": 3}
        metrics = {
            "summary": {
                "total": 2,
                "passed": 1,
                "failed": 1,
                "flaky": 0,
                "pass_rate": 0.5,
                "metrics": {
                    "timeline_interrupt_speech_to_resolved_ms": {"p95": 210.0}
                },
                "per_case": {
                    "ok_case": {"pass_rate": 1.0},
                    "bad_case": {"pass_rate": 0.0},
                },
            }
        }

    brief = _profile_brief(Result())  # type: ignore[arg-type]

    assert brief["profile"] == "channel"
    assert brief["failed_cases"] == ["bad_case"]
    assert (
        brief["key_latencies_p95"]["timeline_interrupt_speech_to_resolved_ms"]
        == 210.0
    )


def test_recommendation_keeps_channel_when_native_has_lower_pass_rate() -> None:
    recommendation = _recommendation(
        [
            {
                "profile": "channel",
                "pass_rate": 1.0,
                "key_latencies_p95": {
                    "timeline_interrupt_speech_to_resolved_ms": 300.0
                },
            },
            {
                "profile": "livekit_native_adaptive",
                "pass_rate": 0.8,
                "key_latencies_p95": {
                    "timeline_interrupt_speech_to_resolved_ms": 150.0
                },
            },
        ]
    )

    assert recommendation["decision"] == "keep_channel_owner"


def test_recommendation_requires_channel_baseline_before_switch() -> None:
    recommendation = _recommendation(
        [
            {
                "profile": "channel",
                "pass_rate": 0.9,
                "key_latencies_p95": {
                    "timeline_interrupt_speech_to_resolved_ms": 200.0
                },
            },
            {
                "profile": "livekit_native_adaptive",
                "pass_rate": 0.9,
                "key_latencies_p95": {
                    "timeline_interrupt_speech_to_resolved_ms": 190.0
                },
            },
        ]
    )

    assert recommendation["decision"] == "fix_channel_baseline_first"


def test_recommendation_marks_native_candidate_only_after_equal_green() -> None:
    recommendation = _recommendation(
        [
            {
                "profile": "channel",
                "pass_rate": 1.0,
                "key_latencies_p95": {
                    "timeline_interrupt_speech_to_resolved_ms": 200.0
                },
            },
            {
                "profile": "livekit_native_adaptive",
                "pass_rate": 1.0,
                "key_latencies_p95": {
                    "timeline_interrupt_speech_to_resolved_ms": 205.0
                },
            },
        ]
    )

    assert (
        recommendation["decision"]
        == "native_candidate_needs_context_ledger_review"
    )
