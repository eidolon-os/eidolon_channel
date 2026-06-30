"""Tests for the real-room barge-in A/B orchestration script."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.bench_barge_in_e2e_ab import (
    _fmt_ms,
    _overlay_payload,
    _profile_brief,
    _recommendation,
    _summary_stat,
)


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
