"""CLI exit status must preserve failed cases and incomplete release evidence."""

import json
import sys

import pytest

from scripts import bench_voice


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("total", "failed", "enforce", "expected"),
    [(1, 0, False, 0), (1, 1, False, 1), (0, 0, False, 1), (1, 0, True, 1)],
)
async def test_room_cli_exit_reflects_cases_and_required_evidence(
    tmp_path, monkeypatch, total, failed, enforce, expected
):
    (tmp_path / "metrics.json").write_text(
        json.dumps({"summary": {"total": total, "failed": failed, "metrics": {}}})
    )

    async def run_room(args, suites):
        return tmp_path

    monkeypatch.setattr(bench_voice, "_run_livekit_room", run_room)
    monkeypatch.setattr(bench_voice, "load_suites", lambda paths: [])
    args = ["bench_voice", "--runner", "livekit_room", "--cases", "fixture.yaml"]
    if enforce:
        args.append("--enforce-slo")
    monkeypatch.setattr(sys, "argv", args)
    assert await bench_voice._main() == expected


def test_explicit_slo_enforcement_rejects_missing_required_metrics(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"summary": {"metrics": {}}}))
    failures = bench_voice._slo_enforcement_failures(tmp_path)
    assert {row["name"] for row in failures} == {
        "room_user_done_to_next_audio",
        "room_publish_to_first_audio",
        "commit_to_first_audio_p50",
        "commit_to_first_audio_p95",
    }
    assert all(row["missing"] for row in failures)
