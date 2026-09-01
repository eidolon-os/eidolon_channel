"""Pure turn-policy benchmark runner."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from eidolon.livekit.agent.full_duplex.transcript_admission import (
    TranscriptAdmissionGate,
)
from eidolon.livekit.agent.integration.client_audio_state import ClientAudioState
from eidolon.livekit.agent.session.assistant_speech import AssistantSpeechLedger
from eidolon.livekit.agent.session.transcript_echo import TranscriptEchoGate
from eidolon.livekit.agent.turn_policy import (
    AdmissionAction,
    AttentionInput,
    Action,
    TurnPolicyRuntime,
)
from eidolon.livekit.common.config import TurnPolicyConfig, load_effective_config

from .device_envelope import device_envelope_metrics
from .schema import BenchmarkCase, BenchmarkSuite, CaseResult, RunResult, UserStep


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
    attention_enforced = policy.attention.enforce
    runtime = TurnPolicyRuntime(policy)
    results: list[CaseResult] = []

    for suite in suites:
        for case in suite.cases:
            started = time.monotonic()
            errors: list[str] = []
            decisions: list[dict] = []
            assistant_speech = _assistant_speech_ledger(
                case,
                welcome_message=cfg.behavior.welcome_message,
            )
            echo_gate = TranscriptEchoGate(
                get_agent_text=lambda: _latest_assistant_text(assistant_speech),
                min_normalized_chars=policy.attention.echo_min_normalized_chars,
            )
            echo_rejection_count = 0
            action = Action.NONE
            intent = "uncertain"
            decision_action = "none"
            decision_intent = "uncertain"
            topic_switch_hint = False
            correction_hint = False
            decision_start_ms: int | None = None
            decision_at_ms: int | None = None

            for step in case.user_steps:
                if not step.agent_speaking:
                    _record_matching_agent_reply(assistant_speech, case, step)
                    continue
                texts = step.interims or (step.text,)
                duck_active = False
                speech_start_attention = runtime.admit_attention(
                    AttentionInput(
                        agent_speaking=step.agent_speaking,
                        client_state=_client_audio_state(step),
                        transcript="",
                        eot_score=0.0,
                        speech_started=True,
                    )
                )
                decisions.append(
                    {
                        "step_text": step.text,
                        "interim_index": -1,
                        "interim_text": "",
                        "speech_started": True,
                        "attention_admission": {
                            "action": speech_start_attention.action.value,
                            "reason": speech_start_attention.reason,
                            "client_state_used": (
                                speech_start_attention.client_state_used
                            ),
                        },
                        "decision": None,
                    }
                )
                if attention_enforced:
                    if speech_start_attention.action is AdmissionAction.IGNORE:
                        continue
                    if speech_start_attention.action is AdmissionAction.HARD_INTERRUPT:
                        action = Action.CANCEL
                        intent = "hard_stop"
                        decision_action = action.value
                        decision_intent = intent
                        decision_start_ms = step.start_ms
                        decision_at_ms = step.start_ms
                        decisions[-1]["decision"] = {
                            "action": action.value,
                            "reason": speech_start_attention.reason,
                            "intent": intent,
                            "intent_source": "attention_admission",
                            "intent_confidence": 1.0,
                            "topic_switch_hint": False,
                            "correction_hint": False,
                            "rollback_drop_buffered": False,
                        }
                        break
                    duck_active = (
                        speech_start_attention.action
                        is AdmissionAction.DUCK_AND_DECIDE
                    )
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
                    transcript_admission = TranscriptAdmissionGate(
                        suppress_until_next_speech=lambda: False,
                        agent_output_active=lambda _speaker_id: step.agent_speaking,
                        echo_gate=lambda: echo_gate,
                    ).evaluate(
                        SimpleNamespace(
                            transcript=text,
                            is_final=is_final,
                            speaker_id=None,
                        )
                    )
                    decision_record["transcript_admission"] = {
                        "accepted": transcript_admission.accepted,
                        "reason": transcript_admission.reason,
                        "assistant_text_source": (
                            assistant_speech.latest.source
                            if assistant_speech.latest is not None
                            else ""
                        ),
                    }
                    if not transcript_admission.accepted:
                        possible_echo = (
                            transcript_admission.reason == "possible_agent_echo_hold"
                        )
                        if not possible_echo:
                            echo_rejection_count += 1
                        action = Action.HOLD if possible_echo else Action.ROLLBACK
                        intent = "uncertain"
                        decision_action = action.value
                        decision_intent = intent
                        decision_start_ms = step.start_ms
                        decision_at_ms = event_time_ms
                        decision_record["decision"] = {
                            "action": action.value,
                            "reason": transcript_admission.reason,
                            "intent": None,
                            "intent_source": "transcript_admission",
                            "intent_confidence": 1.0,
                            "topic_switch_hint": False,
                            "correction_hint": False,
                            "rollback_drop_buffered": False,
                        }
                        decisions.append(decision_record)
                        break
                    if attention_enforced and attention.action in (
                        AdmissionAction.IGNORE,
                        AdmissionAction.OBSERVE,
                    ):
                        if not (
                            attention.action is AdmissionAction.OBSERVE
                            and duck_active
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
                        decision_action = action.value
                        decision_intent = intent
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
                    decision_action = decision.action.value
                    decision_intent = (
                        decision.intent.value if decision.intent else "unknown"
                    )
                    if decision.action is not Action.HOLD:
                        action = decision.action
                        intent = decision_intent
                        topic_switch_hint = decision.topic_switch_hint
                        correction_hint = decision.correction_hint
                        decision_start_ms = step.start_ms
                        decision_at_ms = event_time_ms
                        break
                    if action is Action.NONE:
                        action = Action.HOLD
                        intent = decision_intent
                # HOLD/ROLLBACK are non-terminal for a scripted multi-step
                # policy case: they model false-resume / wait-for-evidence
                # before a later user utterance. CANCEL is terminal because it
                # stops the active assistant turn.
                if action is Action.CANCEL:
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
            expected_decision_action = case.expectations.decision_action
            if expected_decision_action == "none":
                if decision_action != "none":
                    errors.append(
                        f"expected no decision_action, got {decision_action}"
                    )
            elif expected_decision_action not in ("", "any"):
                if decision_action != expected_decision_action:
                    errors.append(
                        "expected decision_action="
                        f"{expected_decision_action}, got {decision_action}"
                    )
            expected_decision_intent = case.expectations.decision_intent
            if (
                expected_decision_action != "none"
                and expected_decision_intent not in ("", "uncertain")
                and decision_intent != expected_decision_intent
            ):
                errors.append(
                    "expected decision_intent="
                    f"{expected_decision_intent}, got {decision_intent}"
                )
            if (
                case.expectations.topic_switch_hint
                and topic_switch_hint != case.expectations.topic_switch_hint
            ):
                errors.append(
                    "expected topic_switch_hint="
                    f"{case.expectations.topic_switch_hint}, got {topic_switch_hint}"
                )
            if (
                case.expectations.correction_hint
                and correction_hint != case.expectations.correction_hint
            ):
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
                "expected_decision_action": case.expectations.decision_action,
                "actual_decision_action": decision_action,
                "forbid_actions": ",".join(case.expectations.forbid_actions),
                "expected_intent": case.expectations.intent,
                "actual_intent": intent,
                "expected_decision_intent": case.expectations.decision_intent,
                "actual_decision_intent": decision_intent,
                "topic_switch_hint": topic_switch_hint,
                "correction_hint": correction_hint,
                "echo_rejection_count": echo_rejection_count,
                **device_envelope_metrics(case),
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


def _assistant_speech_ledger(
    case: BenchmarkCase,
    *,
    welcome_message: str,
) -> AssistantSpeechLedger:
    ledger = AssistantSpeechLedger()
    synthetic_text = case.device_envelope.agent.speaking_text.strip()
    if synthetic_text:
        ledger.record(synthetic_text, source="benchmark_device_envelope")
    elif welcome_message.strip():
        ledger.record(welcome_message, source="benchmark_welcome")
    return ledger


def _latest_assistant_text(ledger: AssistantSpeechLedger) -> str:
    latest = ledger.latest
    return latest.text if latest is not None else ""


def _record_matching_agent_reply(
    ledger: AssistantSpeechLedger,
    case: BenchmarkCase,
    step: UserStep,
) -> None:
    normalized_step = step.text.strip()
    for reply in case.agent_replies:
        trigger = reply.when.strip()
        if trigger and (trigger in normalized_step or normalized_step in trigger):
            ledger.record(reply.reply, source="benchmark_agent_reply")
            return


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
