from __future__ import annotations

from types import SimpleNamespace

from eidolon.livekit.agent.full_duplex.transcript_admission import (
    TranscriptAdmissionGate,
)
from eidolon.livekit.agent.session.transcript_echo import EchoEvidence


class _EchoGate:
    def __init__(
        self,
        result: bool = False,
        *,
        evidence: EchoEvidence | None = None,
    ) -> None:
        self.evidence = evidence or (
            EchoEvidence.CONFIRMED if result else EchoEvidence.NONE
        )
        self.calls: list[str] = []

    def classify(self, transcript: str) -> EchoEvidence:
        self.calls.append(transcript)
        return self.evidence


def _event(
    transcript: str,
    *,
    speaker_id: str | None = "user-1",
    is_final: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        transcript=transcript,
        speaker_id=speaker_id,
        is_final=is_final,
    )


def test_rejects_post_turn_residual_transcript_before_echo_gate() -> None:
    echo_gate = _EchoGate(result=True)
    gate = TranscriptAdmissionGate(
        suppress_until_next_speech=lambda: True,
        agent_output_active=lambda _speaker_id: True,
        echo_gate=lambda: echo_gate,
    )

    decision = gate.evaluate(_event("上一轮残留转写", is_final=True))

    assert not decision.accepted
    assert decision.reason == "suppressed_until_next_speech"
    assert decision.transcript == "上一轮残留转写"
    assert decision.is_final is True
    assert echo_gate.calls == []


def test_allows_empty_transcript_during_post_turn_suppression() -> None:
    echo_gate = _EchoGate(result=True)
    gate = TranscriptAdmissionGate(
        suppress_until_next_speech=lambda: True,
        agent_output_active=lambda _speaker_id: True,
        echo_gate=lambda: echo_gate,
    )

    decision = gate.evaluate(_event(""))

    assert decision.accepted
    assert decision.reason == "accepted"
    assert echo_gate.calls == []


def test_rejects_agent_echo_during_output() -> None:
    echo_gate = _EchoGate(result=True)
    gate = TranscriptAdmissionGate(
        suppress_until_next_speech=lambda: False,
        agent_output_active=lambda speaker_id: speaker_id == "user-1",
        echo_gate=lambda: echo_gate,
    )

    decision = gate.evaluate(_event("这是 AI 正在说的话"))

    assert not decision.accepted
    assert decision.reason == "agent_echo"
    assert decision.speaker_id == "user-1"
    assert echo_gate.calls == ["这是 AI 正在说的话"]


def test_holds_possible_echo_prefix_without_confirming_rejection() -> None:
    echo_gate = _EchoGate(evidence=EchoEvidence.POSSIBLE)
    gate = TranscriptAdmissionGate(
        suppress_until_next_speech=lambda: False,
        agent_output_active=lambda _speaker_id: True,
        echo_gate=lambda: echo_gate,
    )

    decision = gate.evaluate(_event("我是。", is_final=True))

    assert not decision.accepted
    assert decision.reason == "possible_agent_echo_hold"
    assert echo_gate.calls == ["我是。"]


def test_rejects_committed_turn_revision_before_echo_gate() -> None:
    echo_gate = _EchoGate(result=True)
    absorbed: list[tuple[str, bool]] = []
    gate = TranscriptAdmissionGate(
        suppress_until_next_speech=lambda: False,
        agent_output_active=lambda _speaker_id: True,
        echo_gate=lambda: echo_gate,
        absorb_committed_turn_revision=lambda transcript, is_final: (
            absorbed.append((transcript, is_final)) or True
        ),
    )

    decision = gate.evaluate(_event("那你现在能帮我做什么。", is_final=True))

    assert not decision.accepted
    assert decision.reason == "committed_turn_revision"
    assert absorbed == [("那你现在能帮我做什么。", True)]
    assert echo_gate.calls == []


def test_allows_echo_like_transcript_when_agent_output_is_inactive() -> None:
    echo_gate = _EchoGate(result=True)
    gate = TranscriptAdmissionGate(
        suppress_until_next_speech=lambda: False,
        agent_output_active=lambda _speaker_id: False,
        echo_gate=lambda: echo_gate,
    )

    decision = gate.evaluate(_event("这句话像 AI 回复"))

    assert decision.accepted
    assert echo_gate.calls == []


def test_allows_non_echo_transcript_during_output() -> None:
    echo_gate = _EchoGate(result=False)
    gate = TranscriptAdmissionGate(
        suppress_until_next_speech=lambda: False,
        agent_output_active=lambda _speaker_id: True,
        echo_gate=lambda: echo_gate,
    )

    decision = gate.evaluate(_event("用户真正插话"))

    assert decision.accepted
    assert echo_gate.calls == ["用户真正插话"]
