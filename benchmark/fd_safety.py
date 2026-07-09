"""Full-duplex decision-layer safety corpus + invariants (plan Phase 0a v1).

The Phase 2 work — 2a 做减法 (removing compensating heuristics) and 2c owner
收敛 — edits behavioral interrupt/turn decision code. This module wires the
existing ``*_enforced`` full-duplex suites into a deterministic, CI-pure
correctness gate so a change that flips a hold-case into a false barge-in (or
drops a real turn) fails CI instead of only surfacing in dogfood.

Scope (v1): the DECISION layer only. Scripted interims / eot_scores /
client-state are replayed through the turn policy (the ``policy`` runner). It
catches decision regressions — false cancel, missed commit, wrong hold/resume —
deterministically across machines. It does NOT run real VAD/STT on acoustically
rendered audio, so it cannot catch a true acoustic self-interrupt
(echo → VAD → STT → cancel). That is Phase 0a-v2 and needs the component/room
runner + real device audio; see the plan's Phase 0a note.

Reused as-is (not rebuilt): ``load_suite`` / ``run_policy_suite`` (deterministic
replay), the per-case ``passed`` expectation check, and the ``*_enforced`` FD
corpus that already covers echo_safety/guard, mic_muted, manual_interrupt,
backchannel-holds, normal-turn commit, topic-switch/correction/hard-stop.
"""

from __future__ import annotations

from benchmark.policy_runner import run_policy_suite
from benchmark.schema import RunResult, load_suite

# Enforced FD suites carrying the 2a/2c decision expectations, plus the shared
# offline regression set. These are "explicit-only" (``*_enforced.yaml``) so the
# default benchmark run does not pull them in — this gate opts into them on
# purpose as the Phase 2 safety net.
FD_SAFETY_SUITES: tuple[str, ...] = (
    "benchmark/cases/full_duplex/barge_in_ab_matrix_enforced.yaml",
    "benchmark/cases/full_duplex/barge_in_probe_enforced.yaml",
    "benchmark/cases/full_duplex/dogfood_box3_audio_first_enforced.yaml",
    "benchmark/cases/full_duplex/explicit_control_enforced.yaml",
    "benchmark/cases/full_duplex/gate_enforced.yaml",
    "benchmark/cases/full_duplex/v1_interrupt_tiers_enforced.yaml",
    "benchmark/cases/full_duplex/v1_realistic_interaction_flows_enforced.yaml",
    "benchmark/cases/shared/offline_policy_regression_enforced.yaml",
)

# Cases that do not pass under the deterministic policy runner today, with the
# reason. Each is a NON-CANCEL outcome (safety-neutral) whose expectation is
# pinned to a more specific ``decision_action`` than the policy currently emits
# — NOT a false cancel. Documented so the gate flags *new* regressions without
# being blocked by a known, safety-neutral mismatch. Revisit when tightening
# decision_action semantics (or in 2b when EOT cut paths are simplified).
BASELINE_KNOWN_FAILURES: dict[str, str] = {
    "fd_gate_echo_like_agent_words_holds_001": (
        "echo-like words: case expects decision_action=rollback, policy emits "
        "hold; both are non-cancel (safe). Safety-neutral decision_action nuance."
    ),
}


def run_fd_safety_corpus(run_id: str = "fd-safety") -> RunResult:
    """Replay the FD safety corpus through the deterministic policy runner."""

    suites = [load_suite(path) for path in FD_SAFETY_SUITES]
    return run_policy_suite(suites, run_id=run_id)


def false_cancel_violations(run: RunResult) -> list[str]:
    """Cases that hard-cancelled while their expectation forbids cancel.

    This is the highest-severity Phase 2 regression: a compensating-heuristic
    removal (2a) or owner refactor (2c) turning a hold-case into a false
    barge-in. Independent of the per-case pass check so the invariant reads as
    an explicit safety property.
    """

    violations: list[str] = []
    for case in run.cases:
        forbid = str(case.metrics.get("forbid_actions") or "").split(",")
        if case.metrics.get("actual_action") == "cancel" and "cancel" in forbid:
            violations.append(case.case_id)
    return violations


def new_regressions(run: RunResult) -> list[tuple[str, list[str]]]:
    """Failing cases that are NOT in the documented baseline (i.e. new)."""

    return [
        (case.case_id, list(case.errors))
        for case in run.cases
        if not case.passed and case.case_id not in BASELINE_KNOWN_FAILURES
    ]


def correctness_summary(run: RunResult) -> dict[str, object]:
    """Aggregate correctness view for tracking (not a gate by itself)."""

    total = len(run.cases)
    passed = sum(1 for case in run.cases if case.passed)
    return {
        "cases": total,
        "passed": passed,
        "failed": total - passed,
        "false_cancels": false_cancel_violations(run),
        "new_regressions": [cid for cid, _ in new_regressions(run)],
        "known_baseline_failures": sorted(BASELINE_KNOWN_FAILURES),
    }
