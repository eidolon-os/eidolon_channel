"""CI-pure SLO gate for the offline barge-in decision path (plan Track C3).

The real first-response / barge-in LATENCY SLOs live on the ``livekit_room``
runner and need real STT/LLM/TTS — they cannot honestly run in CI. What CAN run
in CI is the DECISION path: the ``policy`` runner replays scripted interrupt
cases through the turn-policy and records ``interrupt_decision_ms``, which is
driven by the policy's configured stabilization/decision windows (deterministic
across machines, not wall-clock compute). This test runs that offline suite
through the SAME dashboard → ``evaluate_slo_gates`` plumbing production uses and
fails on a gross regression of the decision path.
"""

from __future__ import annotations

from pathlib import Path

from eidolon.livekit.benchmarks.dashboard import DashboardRunner, write_dashboard
from eidolon.livekit.benchmarks.policy_runner import run_policy_suite
from eidolon.livekit.benchmarks.report import write_repeated_reports
from eidolon.livekit.benchmarks.schema import load_suite
from eidolon.livekit.benchmarks.slo import enforcement_failures
from scripts.bench_voice import _default_cases


def _policy_runner_payload(tmp_path: Path) -> dict:
    suites = [load_suite(path) for path in _default_cases()]
    run = run_policy_suite(suites, run_id="offline-slo")
    run_dir = tmp_path / "policy"
    run_dir.mkdir(parents=True)
    write_repeated_reports([run], run_dir)
    payload = write_dashboard(
        runners=[DashboardRunner(name="policy", candidate=run_dir)],
        output_path=tmp_path / "dashboard.html",
    )
    return next(r for r in payload["runners"] if r["name"] == "policy")


def test_offline_policy_decision_slo_gate_is_wired_and_evaluated(tmp_path: Path) -> None:
    policy = _policy_runner_payload(tmp_path)
    gates = policy["slo_gates"]
    decision = [g for g in gates if g["metric"] == "interrupt_decision_ms"]
    # The offline gate must actually be wired for the policy runner...
    assert decision, "offline policy decision-latency SLO gate is not wired"
    # ...and the metric must be present in the policy summary (not missing/advisory),
    # otherwise the 'gate' would be a no-op that can never catch a regression.
    assert all(not g["missing"] for g in decision)
    assert any(g["required"] for g in decision)


def test_offline_slo_gates_have_no_enforced_failures(tmp_path: Path) -> None:
    policy = _policy_runner_payload(tmp_path)
    failures = enforcement_failures(policy["slo_gates"])
    assert failures == [], (
        "offline SLO gate regression: "
        + ", ".join(
            f"{f['name']} {f['statistic']}={f['value']} > {f['max_value']}"
            for f in failures
        )
    )
