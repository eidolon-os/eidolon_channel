"""Pure turn-policy benchmark runner."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from eidolon.livekit.agent.turn_policy import Action, TurnPolicyRuntime
from eidolon.livekit.common.config import TurnPolicyConfig, load_effective_config

from .schema import BenchmarkSuite, CaseResult, RunResult


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def run_policy_suite(
    suites: list[BenchmarkSuite],
    *,
    turn_policy: TurnPolicyConfig | None = None,
    run_id: str | None = None,
) -> RunResult:
    cfg = load_effective_config()
    policy = turn_policy or cfg.turn_policy
    runtime = TurnPolicyRuntime(policy)
    results: list[CaseResult] = []

    for suite in suites:
        for case in suite.cases:
            started = time.monotonic()
            errors: list[str] = []
            decisions: list[dict] = []
            action = Action.NONE
            intent = "uncertain"
            topic_switch_hint = False
            correction_hint = False
            decision_start_ms: int | None = None
            decision_at_ms: int | None = None

            for step in case.user_steps:
                if not step.agent_speaking:
                    continue
                text = step.interims[0] if step.interims else step.text
                decision = runtime.decide_from_transcript(
                    text,
                    0.0,
                    vad_active=True,
                    agent_speaking=step.agent_speaking,
                )
                decisions.append(
                    {
                        "step_text": step.text,
                        "interim_text": text,
                        "decision": {
                            "action": decision.action.value,
                            "reason": decision.reason,
                            "intent": decision.intent.value if decision.intent else None,
                            "intent_source": decision.intent_source,
                            "intent_confidence": decision.intent_confidence,
                            "topic_switch_hint": decision.topic_switch_hint,
                            "correction_hint": decision.correction_hint,
                            "rollback_drop_buffered": decision.rollback_drop_buffered,
                        },
                    }
                )
                if decision.action is not Action.HOLD:
                    action = decision.action
                    intent = decision.intent.value if decision.intent else "unknown"
                    topic_switch_hint = decision.topic_switch_hint
                    correction_hint = decision.correction_hint
                    decision_start_ms = step.start_ms
                    decision_at_ms = step.start_ms + step.final_delay_ms
                    break

            if action.value != case.expectations.action:
                errors.append(
                    f"expected action={case.expectations.action}, got {action.value}"
                )
            if case.expectations.intent not in ("", "uncertain") and intent != case.expectations.intent:
                errors.append(f"expected intent={case.expectations.intent}, got {intent}")
            if topic_switch_hint != case.expectations.topic_switch_hint:
                errors.append(
                    "expected topic_switch_hint="
                    f"{case.expectations.topic_switch_hint}, got {topic_switch_hint}"
                )
            if correction_hint != case.expectations.correction_hint:
                errors.append(
                    "expected correction_hint="
                    f"{case.expectations.correction_hint}, got {correction_hint}"
                )

            decision_latency_ms = (
                decision_at_ms - decision_start_ms
                if decision_at_ms is not None and decision_start_ms is not None
                else None
            )
            if (
                case.expectations.max_interrupt_decision_ms is not None
                and decision_latency_ms is not None
                and decision_latency_ms > case.expectations.max_interrupt_decision_ms
            ):
                errors.append(
                    "interrupt decision too slow: "
                    f"{decision_latency_ms}>{case.expectations.max_interrupt_decision_ms}"
                )

            metrics = {
                "elapsed_ms": round((time.monotonic() - started) * 1000),
                "interrupt_decision_ms": decision_latency_ms,
                "expected_action": case.expectations.action,
                "actual_action": action.value,
                "expected_intent": case.expectations.intent,
                "actual_intent": intent,
                "topic_switch_hint": topic_switch_hint,
                "correction_hint": correction_hint,
            }
            results.append(
                CaseResult(
                    case_id=case.case_id,
                    suite=case.suite,
                    runner="policy",
                    passed=not errors,
                    metrics=metrics,
                    decisions=decisions,
                    errors=errors,
                )
            )

    return RunResult(
        run_id=run_id or time.strftime("%Y%m%d-%H%M%S"),
        git_sha=_git_sha(),
        runner="policy",
        profile=policy.profile,
        cases=results,
    )


def write_policy_outputs(run: RunResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run.write_jsonl(output_dir / "policy_results.jsonl")
