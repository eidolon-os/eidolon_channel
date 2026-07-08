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
