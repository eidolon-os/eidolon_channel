"""UI follows output decisions even when LiveKit keeps the same speech handle."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from eidolon.livekit.agent.full_duplex.client_control_publisher import (
    FullDuplexClientControlPublisher,
)
from eidolon.livekit.agent.full_duplex.interruption_effects import FullDuplexInterruptionEffects
from eidolon.livekit.agent.output.ducking import OutputDuckingController


def setup_projection():
    ducking = OutputDuckingController()
    mixer = SimpleNamespace(state="SPEAKING")
    mixer.unduck = lambda **kw: setattr(mixer, "state", "SPEAKING")
    mixer.cancel = lambda: setattr(mixer, "state", "CANCELLED")
    mixer.get_metrics = lambda: {}
    mixer.buffered_frames = 8
    mixer.buffered_sec = 1.2
    ducking.mixer = mixer
    local = SimpleNamespace(publish_data=AsyncMock())
    pipeline = SimpleNamespace(_ducking=ducking, _room=SimpleNamespace(local_participant=local))
    publisher = FullDuplexClientControlPublisher(pipeline)
    effects = FullDuplexInterruptionEffects(
        ducking=ducking,
        callbacks=MagicMock(),
        get_session=lambda: None,
        allow_interruptions=lambda: True,
        get_eot_model=MagicMock(),
        get_timeline=lambda: None,
        get_latest_asr_text=lambda: "嗯",
        get_state_label=lambda: "SPEAKING",
        get_interruption_orchestrator=MagicMock(),
        publish_playback_stop=MagicMock(),
        snapshot_interrupted_context=MagicMock(),
        cancel_residual_commit_suppress_sec=lambda: 0,
        semantic_interrupt_run=MagicMock(),
        correction_topic_stability_window_ms=lambda: 120,
        set_interrupt_cancel_suppression=MagicMock(),
        soft_interrupt_timeout_sec=lambda: 1,
        refresh_output_ui=publisher.refresh_output_state,
    )
    return mixer, local, publisher, effects


async def packets(local):
    await asyncio.sleep(0)
    return [json.loads(call.args[0]) for call in local.publish_data.await_args_list]


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", ["rollback", "pending"])
async def test_resume_same_speech_handle_restores_ui_and_survives_late_user_end(resume):
    mixer, local, publisher, effects = setup_projection()
    publisher.publish_companion_ui_state_for_agent_state("speaking")
    mixer.state = "SUSPENDED"
    publisher.publish_companion_ui_state("listening", "user_state:speaking")
    if resume == "rollback":
        effects.rollback_if_suspended("final_intent_resume")
    else:
        effects.resume_pending_output("pending_evidence")
    publisher.publish_companion_ui_state("listening", "user_state:listening")
    result = await packets(local)
    assert [p["state"] for p in result] == ["speaking", "listening", "speaking", "speaking"]
    assert result[2]["reason"].startswith("output_state:")
    effects._publish_playback_stop.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", ["accepted", "explicit"])
async def test_cancel_masks_stale_sdk_speaking_until_new_output(cancel):
    mixer, local, publisher, effects = setup_projection()
    publisher.publish_companion_ui_state_for_agent_state("speaking")
    mixer.state = "SUSPENDED"
    if cancel == "accepted":
        effects.cancel_and_interrupt(force=True)
        effects._publish_playback_stop.assert_called_once_with("interrupt_cancel")
    else:
        effects.cancel_silent_generation_for_explicit_preempt()
    publisher.publish_companion_ui_state("listening", "user_state:listening")
    assert [p["state"] for p in await packets(local)][-2:] == ["listening", "listening"]
    mixer.state = "SPEAKING"  # Existing output lifecycle resets for the next answer.
    publisher.publish_companion_ui_state_for_agent_state("speaking")
    assert (await packets(local))[-1]["state"] == "speaking"


@pytest.mark.asyncio
async def test_completed_output_is_not_resurrected_by_rollback():
    mixer, local, publisher, effects = setup_projection()
    publisher.publish_companion_ui_state_for_agent_state("speaking")
    mixer.state = "SUSPENDED"
    publisher.publish_companion_ui_state_for_agent_state("listening")
    effects.rollback_if_suspended("user_silent")
    assert (await packets(local))[-1]["state"] == "listening"


@pytest.mark.asyncio
async def test_sdk_evidence_preserved_but_suspended_ui_stays_listening():
    mixer, local, publisher, _ = setup_projection()
    mixer.state = "SUSPENDED"
    publisher.publish_companion_ui_state_for_agent_state("speaking")
    result = await packets(local)
    assert result[0]["state"] == "speaking"
    assert result[0]["reason"] == "agent_state:speaking"
    assert result[-1]["state"] == "listening"
    assert result[-1]["reason"].startswith("output_state:")
