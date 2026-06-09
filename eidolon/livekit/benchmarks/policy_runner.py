"""Pure turn-policy benchmark runner."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from eidolon.livekit.agent.client_audio_state import ClientAudioState
from eidolon.livekit.agent.turn_policy import (
    AdmissionAction,
    AttentionInput,
    Action,
    TurnPolicyRuntime,
)
from eidolon.livekit.agent.turn_policy.modes import effective_attention_enforce
from eidolon.livekit.common.config import TurnPolicyConfig, load_effective_config

from .schema import BenchmarkSuite, CaseResult, RunResult, UserStep


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
    attention_enforced = effective_attention_enforce(policy)
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
                texts = step.interims or (step.text,)
                for index, text in enumerate(texts):
                    is_final = index == len(texts) - 1
                    event_time_ms = step.start_ms + index * 80
                    eot_score = (
                        step.eot_scores[index]
                        if index < len(step.eot_scores)
                        else 0.0
                    )
                    attention = runtime.admit_attention(
                        AttentionInput(
                            agent_speaking=step.agent_speaking,
                            client_state=_client_audio_state(step),
                            transcript=text,
                            eot_score=eot_score,
                        )
                    )
                    decision_record = {
                        "step_text": step.text,
                        "interim_index": index,
                        "interim_text": text,
                        "attention_admission": {
                            "action": attention.action.value,
                            "reason": attention.reason,
                            "client_state_used": attention.client_state_used,
                        },
                        "decision": None,
                    }
                    if attention_enforced and attention.action in (
                        AdmissionAction.IGNORE,
                        AdmissionAction.OBSERVE,
                    ):
                        decisions.append(decision_record)
                        continue
                    if (
                        attention_enforced
                        and attention.action is AdmissionAction.HARD_INTERRUPT
                        and not text.strip()
                    ):
                        action = Action.CANCEL
                        intent = "hard_stop"
                        decision_start_ms = step.start_ms
                        decision_at_ms = event_time_ms
                        decision_record["decision"] = {
                            "action": action.value,
                            "reason": attention.reason,
                            "intent": intent,
                            "intent_source": "attention_admission",
                            "intent_confidence": 1.0,
                            "topic_switch_hint": False,
                            "correction_hint": False,
                            "rollback_drop_buffered": False,
                        }
                        decisions.append(decision_record)
                        break
                    decision = runtime.decide_from_transcript(
                        text,
                        eot_score,
                        vad_active=True,
                        agent_speaking=step.agent_speaking,
                        is_final=is_final,
                        event_time_ms=event_time_ms,
                    )
                    decision_record["decision"] = {
                        "action": decision.action.value,
                        "reason": decision.reason,
                        "intent": decision.intent.value if decision.intent else None,
                        "intent_source": decision.intent_source,
                        "intent_confidence": decision.intent_confidence,
                        "topic_switch_hint": decision.topic_switch_hint,
                        "correction_hint": decision.correction_hint,
                        "rollback_drop_buffered": decision.rollback_drop_buffered,
                    }
                    decisions.append(decision_record)
                    if decision.action is not Action.HOLD:
                        action = decision.action
                        intent = decision.intent.value if decision.intent else "unknown"
                        topic_switch_hint = decision.topic_switch_hint
                        correction_hint = decision.correction_hint
                        decision_start_ms = step.start_ms
                        decision_at_ms = event_time_ms
                        break
                    if action is Action.NONE:
                        action = Action.HOLD
                        intent = decision.intent.value if decision.intent else "unknown"
                if action is not Action.NONE:
                    break

            if action.value in case.expectations.forbid_actions:
                errors.append(f"forbidden action={action.value}")
            if (
                case.expectations.action not in ("", "any")
                and action.value != case.expectations.action
            ):
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
                "forbid_actions": ",".join(case.expectations.forbid_actions),
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


def _client_audio_state(step: UserStep) -> ClientAudioState | None:
    if (
        step.client_playback_state in ("unknown", "none")
        and not step.client_ptt
        and not step.client_manual_interrupt
        and not step.client_mic_muted
    ):
        return None
    return ClientAudioState(
        participant_identity="benchmark-user",
        playback_state=(
            step.client_playback_state
            if step.client_playback_state in ("idle", "agent_speaking")
            else "unknown"
        ),
        ptt=step.client_ptt,
        manual_interrupt=step.client_manual_interrupt,
        mic_muted=step.client_mic_muted,
        received_at=time.monotonic(),
    )


def write_policy_outputs(run: RunResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run.write_jsonl(output_dir / "policy_results.jsonl")
