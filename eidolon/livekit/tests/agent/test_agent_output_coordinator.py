from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.output import OutputDuckingController
from eidolon.livekit.agent.session.agent_state import AgentStateEffectHandler
from eidolon.livekit.agent.session.agent_output_coordinator import AgentOutputCoordinator


def test_agent_state_speaking_without_provider_audio_is_visible() -> None:
    timeline = TurnTimeline("turn-output")
    handler = AgentStateEffectHandler(
        get_timeline=lambda: timeline,
        mark_activity=MagicMock(),
        cancel_soft_interrupt=MagicMock(),
        soft_interrupt_active=lambda: False,
        ducking=OutputDuckingController(),
        get_filler=lambda: None,
        flush_timeline_debug=MagicMock(),
    )

    handler.handle(SimpleNamespace(old_state="thinking", new_state="speaking"))

    assert timeline.attrs["agent_output"]["framework_agent_state"] == "speaking"
    assert (
        timeline.attrs["agent_output"]["phase"]
        == "framework_speaking_without_provider_audio"
    )
    assert timeline.attrs["agent_output"]["risk"] == "awaiting_tts_provider_audio"


def test_interruption_candidate_links_to_active_response_owner() -> None:
    coordinator = AgentOutputCoordinator()
    response = TurnTimeline("response-turn")
    candidate = TurnTimeline("candidate-turn")
    coordinator.claim(response)

    linked_turn_id = coordinator.link_interruption_candidate(candidate)

    assert linked_turn_id == "response-turn"
    assert candidate.attrs["interruption_target"]["response_turn_id"] == ("response-turn")


def test_idle_candidate_does_not_invent_interruption_target() -> None:
    coordinator = AgentOutputCoordinator()
    candidate = TurnTimeline("candidate-turn")

    assert coordinator.link_interruption_candidate(candidate) is None
    assert "interruption_target" not in candidate.attrs
