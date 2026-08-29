from __future__ import annotations

from eidolon.livekit.agent.full_duplex.state_machine import (
    FullDuplexPhase,
    FullDuplexStateMachine,
)
from eidolon.livekit.agent.observability import TurnTimeline


class _Clock:
    def __init__(self) -> None:
        self.value = 10.0

    def __call__(self) -> float:
        return self.value


def test_full_duplex_state_machine_projects_transitions_to_timeline() -> None:
    clock = _Clock()
    machine = FullDuplexStateMachine(clock=clock)
    timeline = TurnTimeline("turn-state")

    machine.transition(
        FullDuplexPhase.USER_SPEECH_OPEN,
        event="speech_started",
        reason="new_speech_started",
        timeline=timeline,
    )
    clock.value = 10.25
    machine.transition(
        FullDuplexPhase.PROVISIONAL_DUCK,
        event="duck_started",
        reason="vad_started",
        side_effect="reversible",
        transcript="那你有性别吗",
        timeline=timeline,
        details={"timeout_sec": 2.0},
    )

    assert machine.phase is FullDuplexPhase.PROVISIONAL_DUCK
    assert machine.snapshot()["transition_count"] == 2
    assert timeline.attrs["full_duplex_state"]["phase"] == "provisional_duck"
    assert timeline.attrs["full_duplex_state"]["last"] == {
        "phase": "provisional_duck",
        "event": "duck_started",
        "reason": "vad_started",
        "side_effect": "reversible",
        "at": 10.25,
        "transcript_preview": "那你有性别吗",
        "details": {"timeout_sec": 2.0},
    }
    assert len(timeline.attrs["full_duplex_state_transitions"]) == 2


# ---------------------------------------------------------------------------
# F1 guardrail (2026-07): unexpected-transition observability. Never blocks —
# only records sequences outside the expected graph so anomalies surface.
# ---------------------------------------------------------------------------


def _machine() -> FullDuplexStateMachine:
    return FullDuplexStateMachine(clock=_Clock())


def test_expected_forward_transitions_are_not_flagged() -> None:
    machine = _machine()
    timeline = TurnTimeline("turn-ok")
    machine.transition(
        FullDuplexPhase.USER_SPEECH_OPEN,
        event="speech_started",
        reason="new_speech",
        timeline=timeline,
    )
    machine.transition(
        FullDuplexPhase.PROVISIONAL_DUCK,
        event="duck_started",
        reason="vad_started",
        timeline=timeline,
    )
    assert machine.unexpected_transition_count == 0
    assert "full_duplex_unexpected_transitions" not in timeline.attrs


def test_unexpected_transition_is_flagged_but_still_applied() -> None:
    # IDLE -> USER_TURN_COMMITTED skips speech/pending: unexpected. It must be
    # recorded (observability) but still applied (recorder never blocks).
    machine = _machine()
    timeline = TurnTimeline("turn-weird")
    machine.transition(
        FullDuplexPhase.USER_TURN_COMMITTED,
        event="odd",
        reason="skipped_states",
        timeline=timeline,
    )
    assert machine.phase is FullDuplexPhase.USER_TURN_COMMITTED  # applied
    assert machine.unexpected_transition_count == 1
    flagged = timeline.attrs["full_duplex_unexpected_transitions"]
    assert flagged[-1] == {
        "from": "idle",
        "to": "user_turn_committed",
        "event": "odd",
        "reason": "skipped_states",
    }
    assert timeline.attrs["full_duplex_unexpected_transition_count"] == 1


def test_reset_and_reject_allowed_from_any_phase() -> None:
    machine = _machine()
    machine.transition(FullDuplexPhase.USER_SPEECH_OPEN, event="s", reason="r")
    machine.transition(FullDuplexPhase.USER_TURN_REJECTED, event="rej", reason="r")
    machine.transition(FullDuplexPhase.IDLE, event="reset", reason="r")
    assert machine.unexpected_transition_count == 0


def test_self_transition_not_flagged() -> None:
    machine = _machine()
    machine.transition(FullDuplexPhase.USER_SPEECH_OPEN, event="s", reason="r")
    machine.transition(FullDuplexPhase.USER_SPEECH_OPEN, event="s2", reason="r")
    assert machine.unexpected_transition_count == 0


def test_each_timeline_has_an_independent_phase() -> None:
    machine = _machine()
    first = TurnTimeline("turn-first")
    second = TurnTimeline("turn-second")
    machine.transition(
        FullDuplexPhase.USER_SPEECH_OPEN,
        event="speech_started",
        reason="first",
        timeline=first,
    )
    machine.transition(
        FullDuplexPhase.USER_TURN_PENDING,
        event="speech_stopped",
        reason="first_pending",
        timeline=first,
    )

    machine.transition(
        FullDuplexPhase.USER_SPEECH_OPEN,
        event="speech_started",
        reason="second",
        timeline=second,
    )

    assert machine.phase_for(first) is FullDuplexPhase.USER_TURN_PENDING
    assert machine.phase_for(second) is FullDuplexPhase.USER_SPEECH_OPEN
    assert machine.unexpected_transition_count == 0


def test_unexpected_counts_are_scoped_to_each_timeline() -> None:
    machine = _machine()
    first = TurnTimeline("turn-first-invalid")
    second = TurnTimeline("turn-second-invalid")

    machine.transition(
        FullDuplexPhase.USER_TURN_COMMITTED,
        event="odd_first",
        reason="skipped",
        timeline=first,
    )
    machine.transition(
        FullDuplexPhase.USER_TURN_COMMITTED,
        event="odd_second",
        reason="skipped",
        timeline=second,
    )

    assert machine.unexpected_transition_count == 2
    assert first.attrs["full_duplex_unexpected_transition_count"] == 1
    assert second.attrs["full_duplex_unexpected_transition_count"] == 1


def test_reentrant_multisegment_turn_edges_are_expected() -> None:
    machine = _machine()
    timeline = TurnTimeline("turn-multisegment")
    phases = (
        FullDuplexPhase.USER_SPEECH_OPEN,
        FullDuplexPhase.USER_TURN_PENDING,
        FullDuplexPhase.USER_SPEECH_OPEN,
        FullDuplexPhase.PROVISIONAL_DUCK,
        FullDuplexPhase.EVIDENCE_ARBITRATION,
        FullDuplexPhase.REJECTED_INTERRUPTION,
        FullDuplexPhase.ACCEPTED_INTERRUPTION,
        FullDuplexPhase.USER_TURN_PENDING,
        FullDuplexPhase.USER_TURN_COMMITTED,
    )
    for index, phase in enumerate(phases):
        machine.transition(phase, event=f"event_{index}", reason="observed", timeline=timeline)

    assert machine.unexpected_transition_count == 0
