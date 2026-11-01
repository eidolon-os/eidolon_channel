"""One conversation's way in, from the request the Provider heard to the start."""

from __future__ import annotations

import pytest

from eidolon.livekit.common.entry_timing import (
    action_to_send_ms,
    dispatch_entry,
    entry_record,
    format_entry,
)


def test_a_warm_start_is_timed_from_the_request() -> None:
    metadata = {"entry": dispatch_entry(requested_at_ms=100_000, action_to_send_ms=60)}

    record = entry_record(metadata, job_started_at=100.25, warmed_at=40.0, now=101.0)

    assert record == {
        "job_to_started_ms": 750, "request_to_started_ms": 1000, "request_to_job_ms": 250,
        "warm_process": True, "action_to_send_ms": 60, "action_to_started_ms": 1060,
    }
    assert format_entry(record).startswith("job_to_started_ms=750 request_to_started_ms=1000")
    assert "warm_process=true" in format_entry(record)


def test_a_process_built_for_the_conversation_is_called_cold() -> None:
    # 11-01: a second conversation two seconds after the first waited 8.6 s.
    record = entry_record(
        {"entry": {"requested_at_ms": 100_000}}, job_started_at=108.7, warmed_at=108.6,
        now=109.5,
    )
    assert record["warm_process"] is False
    assert record["request_to_job_ms"] == 8700
    assert "action_to_send_ms" not in record


@pytest.mark.parametrize("metadata", [{}, {"entry": "x"}, {"entry": {"requested_at_ms": "1"}},
                                      {"entry": {"requested_at_ms": True}}])
def test_without_a_request_time_only_the_jobs_own_part_is_known(metadata) -> None:
    assert entry_record(metadata, job_started_at=1.0, warmed_at=0.5, now=1.5) == {
        "job_to_started_ms": 500,
    }


@pytest.mark.parametrize(
    ("raw", "kept"),
    [(0, 0), (12.7, 12), (60_000, 60_000), (60_001, None), (-5, None), (float("nan"), None),
     (False, None), ("10", None), (None, None)],
)
def test_only_a_plausible_device_figure_is_kept(raw, kept) -> None:
    assert action_to_send_ms(raw) == kept
