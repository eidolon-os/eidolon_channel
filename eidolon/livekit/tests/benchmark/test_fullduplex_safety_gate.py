"""CI-pure full-duplex decision-safety gate (plan Phase 0a v1).

Guards the Phase 2 work: 2a 做减法 (removing compensating heuristics) and 2c
owner 收敛 both edit behavioral interrupt/turn decision code. This gate replays
the enforced FD safety corpus through the deterministic ``policy`` runner and
fails on any *new* per-case correctness regression, plus a hard
zero-false-cancel invariant.

It runs offline (no LiveKit room / real providers), so it belongs in CI and
should be run before AND after every 做减法 / 收敛 change. It does NOT cover
acoustic self-interrupt (echo → real VAD/STT → cancel) — that is Phase 0a-v2;
see ``benchmark/fd_safety.py``.
"""

from __future__ import annotations

from benchmark.fd_safety import (
    BASELINE_KNOWN_FAILURES,
    false_cancel_violations,
    new_regressions,
    run_fd_safety_corpus,
)


def test_fd_safety_corpus_has_no_false_cancels() -> None:
    run = run_fd_safety_corpus()
    violations = false_cancel_violations(run)
    assert violations == [], (
        "Phase 2 false-cancel regression: these cases forbid cancel but the "
        "policy hard-cancelled (a false barge-in): " + ", ".join(violations)
    )


def test_fd_safety_corpus_no_new_correctness_regression() -> None:
    run = run_fd_safety_corpus()
    regressions = new_regressions(run)
    assert regressions == [], (
        "Phase 2 correctness regression in the FD safety corpus:\n"
        + "\n".join(f"  {cid}: {errs}" for cid, errs in regressions)
    )


def test_fd_safety_baseline_allowlist_is_not_stale() -> None:
    # If a known-failing case starts passing, that is good — but the allowlist
    # must be tightened so it never silently masks a case that later regresses.
    run = run_fd_safety_corpus()
    failing = {case.case_id for case in run.cases if not case.passed}
    stale = sorted(set(BASELINE_KNOWN_FAILURES) - failing)
    assert stale == [], (
        "stale baseline allowlist — these cases now PASS, remove them from "
        "BASELINE_KNOWN_FAILURES: " + ", ".join(stale)
    )
