from __future__ import annotations

from types import SimpleNamespace

from eidolon.livekit.agent.full_duplex.transcript_admission import (
    TranscriptAdmissionDecision,
)
from eidolon.livekit.agent.full_duplex.transcript_handler import (
    FullDuplexTranscriptHandler,
)


class _AdmissionGate:
    def __init__(self, decision: TranscriptAdmissionDecision | None = None) -> None:
        self.decision = decision
        self.events = []

    def evaluate(self, event):
        self.events.append(event)
        if self.decision is not None:
            return self.decision
        return TranscriptAdmissionDecision(
            accepted=True,
            transcript=event.transcript,
            speaker_id=event.speaker_id,
            is_final=event.is_final,
        )


def _event(text: str = "停一下", *, final: bool = False, speaker_id: str = "user"):
    return SimpleNamespace(transcript=text, is_final=final, speaker_id=speaker_id)


def _handler(
    *,
    admission_gate: _AdmissionGate | None = None,
    allow_interruptions: bool = True,
    native_adaptive: bool = False,
    agent_output_active: bool = False,
    interrupt_window_active: bool = False,
    decision_suppressed: bool = False,
    attention_allowed: bool = True,
    policy_transcript: str | None = None,
):
    gate = admission_gate or _AdmissionGate()
    calls = {
        "recorded": [],
        "semantic": [],
        "forwarded": [],
        "attention": [],
        "speaker_checks": [],
        "echo_rejected": [],
        "ingress_events": [],
        "admission_events": [],
        "semantic_gate_events": [],
    }

    return FullDuplexTranscriptHandler(
        admission_gate=lambda: gate,
        record_accepted_event=lambda event: (
            calls["recorded"].append(event) or policy_transcript
        ),
        allow_interruptions=lambda: allow_interruptions,
        native_adaptive_owner=lambda: native_adaptive,
        agent_output_active=lambda speaker_id: (
            calls["speaker_checks"].append(speaker_id) or agent_output_active
        ),
        interrupt_window_active=lambda: interrupt_window_active,
        decision_suppressed=lambda: decision_suppressed,
        attention_allows_eot_check=lambda transcript, speaker_id: (
            calls["attention"].append((transcript, speaker_id)) or attention_allowed
        ),
        run_semantic_interrupt=lambda transcript, is_final: calls["semantic"].append(
            (transcript, is_final)
        ),
        reject_agent_echo=lambda transcript: calls["echo_rejected"].append(transcript),
        forward_to_base=lambda event: calls["forwarded"].append(event),
        record_transcript_ingress_event=lambda payload: calls[
            "ingress_events"
        ].append(payload),
        record_transcript_admission_event=lambda payload: calls[
            "admission_events"
        ].append(payload),
        record_semantic_gate_event=lambda payload: calls["semantic_gate_events"].append(
            payload
        ),
    ), calls, gate


def test_transcript_handler_records_and_forwards_non_interrupt_transcript() -> None:
    handler, calls, gate = _handler()
    event = _event("你好", final=True)

    handler.handle(event)

    assert gate.events[0].transcript == "你好"
    assert calls["ingress_events"] == [
        {
            "transcript_preview": "你好",
            "text_length": 2,
            "is_final": True,
            "speaker_id": "user",
            "event_type": "SimpleNamespace",
        }
    ]
    assert calls["recorded"][0].transcript == "你好"
    assert calls["semantic"] == []
    assert calls["forwarded"] == [event]
    assert calls["admission_events"] == [
        {
            "accepted": True,
            "reason": "accepted",
            "transcript_preview": "你好",
            "text_length": 2,
            "is_final": True,
            "speaker_id": "user",
        }
    ]
    assert calls["semantic_gate_events"] == [
        {
            "stage": "initial",
            "action": "inactive",
            "reason": "no_interrupt_window",
            "transcript_preview": "你好",
            "text_length": 2,
            "is_final": True,
            "speaker_id": "user",
            "allow_interruptions": True,
            "native_adaptive_owner": False,
            "agent_output_active": False,
            "interrupt_window_active": False,
            "decision_suppressed": False,
        }
    ]


def test_transcript_handler_runs_semantic_interrupt_when_attention_allows() -> None:
    handler, calls, _ = _handler(
        agent_output_active=True,
        attention_allowed=True,
    )
    event = _event("停一下", final=False, speaker_id="owner")

    handler.handle(event)

    assert calls["speaker_checks"] == ["owner"]
    assert calls["attention"] == [("停一下", "owner")]
    assert calls["semantic"] == [("停一下", False)]
    assert calls["forwarded"] == [event]
    assert calls["semantic_gate_events"] == [
        {
            "stage": "initial",
            "action": "needs_attention",
            "reason": "needs_attention",
            "transcript_preview": "停一下",
            "text_length": 3,
            "is_final": False,
            "speaker_id": "owner",
            "allow_interruptions": True,
            "native_adaptive_owner": False,
            "agent_output_active": True,
            "interrupt_window_active": False,
            "decision_suppressed": False,
        },
        {
            "stage": "attention",
            "action": "run",
            "reason": "eligible",
            "transcript_preview": "停一下",
            "text_length": 3,
            "is_final": False,
            "speaker_id": "owner",
            "allow_interruptions": True,
            "native_adaptive_owner": False,
            "agent_output_active": True,
            "interrupt_window_active": False,
            "decision_suppressed": False,
            "attention_allowed": True,
        },
    ]


def test_transcript_handler_uses_canonical_turn_for_semantic_policy() -> None:
    handler, calls, _ = _handler(
        agent_output_active=True,
        attention_allowed=True,
        policy_transcript="换个话题",
    )
    event = _event("话题", final=False, speaker_id="owner")

    handler.handle(event)

    assert calls["attention"] == [("换个话题", "owner")]
    assert calls["semantic"] == [("换个话题", False)]
    assert calls["forwarded"] == [event]
    assert calls["semantic_gate_events"][-1]["transcript_preview"] == "换个话题"
    assert calls["semantic_gate_events"][-1]["event_transcript_preview"] == "话题"


def test_transcript_handler_forwards_without_semantic_run_when_attention_blocks() -> None:
    handler, calls, _ = _handler(
        agent_output_active=True,
        attention_allowed=False,
    )
    event = _event("背景声音")

    handler.handle(event)

    assert calls["recorded"][0].transcript == "背景声音"
    assert calls["attention"] == [("背景声音", "user")]
    assert calls["semantic"] == []
    assert calls["forwarded"] == [event]
    assert calls["semantic_gate_events"][-1] == {
        "stage": "attention",
        "action": "forward_and_stop",
        "reason": "attention_blocked",
        "transcript_preview": "背景声音",
        "text_length": 4,
        "is_final": False,
        "speaker_id": "user",
        "allow_interruptions": True,
        "native_adaptive_owner": False,
        "agent_output_active": True,
        "interrupt_window_active": False,
        "decision_suppressed": False,
        "attention_allowed": False,
    }


def test_transcript_handler_forwards_and_stops_when_decision_suppressed() -> None:
    handler, calls, _ = _handler(
        agent_output_active=True,
        decision_suppressed=True,
    )
    event = _event("残留字幕")

    handler.handle(event)

    assert calls["recorded"][0].transcript == "残留字幕"
    assert calls["attention"] == []
    assert calls["semantic"] == []
    assert calls["forwarded"] == [event]
    assert calls["semantic_gate_events"][-1]["reason"] == "decision_suppressed"


def test_transcript_handler_drops_rejected_admission() -> None:
    gate = _AdmissionGate(
        TranscriptAdmissionDecision(
            accepted=False,
            reason="agent_echo",
            transcript="你好",
            speaker_id="agent",
            is_final=False,
        )
    )
    handler, calls, _ = _handler(admission_gate=gate)

    handler.handle(_event("你好", speaker_id="agent"))

    assert calls["ingress_events"] == [
        {
            "transcript_preview": "你好",
            "text_length": 2,
            "is_final": False,
            "speaker_id": "agent",
            "event_type": "SimpleNamespace",
        }
    ]
    assert calls["recorded"] == []
    assert calls["attention"] == []
    assert calls["semantic"] == []
    assert calls["forwarded"] == []
    assert calls["echo_rejected"] == ["你好"]
    assert calls["admission_events"] == [
        {
            "accepted": False,
            "reason": "agent_echo",
            "transcript_preview": "你好",
            "text_length": 2,
            "is_final": False,
            "speaker_id": "agent",
        }
    ]
    assert calls["semantic_gate_events"] == []


def test_transcript_handler_does_not_echo_rollback_for_other_rejections() -> None:
    gate = _AdmissionGate(
        TranscriptAdmissionDecision(
            accepted=False,
            reason="suppressed_until_next_speech",
            transcript="迟到字幕",
            speaker_id="user",
            is_final=True,
        )
    )
    handler, calls, _ = _handler(admission_gate=gate)

    handler.handle(_event("迟到字幕", final=True))

    assert calls["recorded"] == []
    assert calls["forwarded"] == []
    assert calls["echo_rejected"] == []
    assert calls["admission_events"][0]["reason"] == "suppressed_until_next_speech"
    assert calls["semantic_gate_events"] == []
