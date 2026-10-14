"""Exercise the installed LiveKit RoomIO queue, not an immediately-writing fake."""

import asyncio

import numpy as np
import pytest
from livekit import rtc
from livekit.agents.voice.room_io._output import _ParticipantAudioOutput

from eidolon.livekit.agent.output.controller import OutputController


class Source:
    def __init__(self, *args, **kwargs):
        self.frames = []
        self.queued_duration = 0.0

    async def capture_frame(self, frame):
        self.frames.append(frame)
        await asyncio.sleep(0)

    def clear_queue(self):
        pass

    async def wait_for_playout(self):
        await asyncio.sleep(0)

    async def aclose(self):
        pass


async def settle():
    for _ in range(30):
        await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_already_synthesized_roomio_tail_obeys_duck(monkeypatch, cancel):
    monkeypatch.setattr(rtc, "AudioSource", Source)
    sink = _ParticipantAudioOutput(
        None, sample_rate=16000, num_channels=1, track_publish_options=rtc.TrackPublishOptions()
    )
    sink._subscribed_fut.set_result(None)
    output = OutputController(sink)
    frame = rtc.AudioFrame(
        np.full(16000, 1234, dtype=np.int16).tobytes(),
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=16000,
    )
    await output.capture_frame(frame)
    output.flush()
    assert output.buffered_frames == 0  # the whole second belongs to RoomIO now
    output.duck()
    sink._forwarding_task = asyncio.create_task(sink._forward_audio())
    try:
        await settle()
        assert sink._audio_source.frames == [], "native queue must pause without new TTS"
        if cancel:
            output.cancel()
            await settle()
            output.reset()
            await settle()
            assert sink._audio_source.frames == [], "cancelled queued speech must not resume"
            next_frame = rtc.AudioFrame(
                np.full(1600, 5678, dtype=np.int16).tobytes(),
                sample_rate=16000,
                num_channels=1,
                samples_per_channel=1600,
            )
            await output.capture_frame(next_frame)
            output.flush()
            await asyncio.wait_for(sink._flush_task, 1)
            played = np.concatenate(
                [np.frombuffer(f.data, dtype=np.int16) for f in sink._audio_source.frames]
            )
            assert len(played) == 1600
            assert (played == 5678).all(), "the next answer must play without the cancelled tail"
        else:
            # Repeated VAD and opt-in passthrough cannot release a native backlog.
            output.duck()
            assert not output.enable_suspended_passthrough(volume=0.25)
            assert sink._audio_source.frames == []
            output.unduck()
            await asyncio.wait_for(sink._flush_task, 1)
            played = np.concatenate(
                [np.frombuffer(f.data, dtype=np.int16) for f in sink._audio_source.frames]
            )
            assert len(played) == 16000
            assert (played == 1234).all(), "false interruption must retain queued content"
    finally:
        await sink.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("intent_name", ["BACKCHANNEL", "HARD_STOP"])
@pytest.mark.parametrize("late_user_end", [False, True])
async def test_semantic_verdict_drives_native_queue_and_companion_ui(
    monkeypatch, intent_name, late_user_end
):
    """Replay queued TTS + semantic verdict; never emit a second SDK speaking edge.

    Only the external classifier, RTC AudioSource and data transport are doubles.
    Policy, decision effects, pipeline wiring, output controller, installed SDK
    queue and UI publisher execute production code.
    """
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from eidolon_sdk.biz.contracts import COMPANION_UI_STATE_TOPIC, CONTROL_TOPIC
    from eidolon.livekit.agent.full_duplex.user_state_handler import FullDuplexUserStateHandler
    from eidolon.livekit.agent.session.decision_effects import DecisionEffectApplier
    from eidolon.livekit.agent.turn_policy import Action, InterruptIntent
    from .test_async_interrupt_intent import setup_handler, result
    from .test_commit_guard import _make_pipeline_with_session

    monkeypatch.setattr(rtc, "AudioSource", Source)
    sink = _ParticipantAudioOutput(
        None,
        sample_rate=16000,
        num_channels=1,
        track_publish_options=rtc.TrackPublishOptions(),
    )
    sink._subscribed_fut.set_result(None)
    output = OutputController(sink)
    pipeline = _make_pipeline_with_session()
    pipeline._ensure_runtime_defaults()
    pipeline._ducking.mixer = output
    pipeline._allow_interruptions = True
    local = SimpleNamespace(publish_data=AsyncMock())
    pipeline._room = SimpleNamespace(local_participant=local)
    publisher = pipeline._ensure_client_control_publisher()
    effects = pipeline._build_interruption_effects()  # Includes actual UI callback wiring.
    classifier = SimpleNamespace(
        classify=AsyncMock(return_value=result(getattr(InterruptIntent, intent_name)))
    )
    handler, state, _, _ = setup_handler(classifier, vad=True)
    state.text = "嗯" if intent_name == "BACKCHANNEL" else "不要说了"
    decisions = []
    applier = DecisionEffectApplier(
        turn_runtime=handler._turn_runtime,
        get_timeline=lambda: None,
        on_cancel=lambda: effects.cancel_and_interrupt(force=True),
        on_rollback=lambda reason, drop: effects.rollback_if_suspended(reason, drop_buffered=drop),
        on_resume=effects.resume_pending_output,
        on_decision=lambda decision, **kwargs: decisions.append(decision),
    )
    handler._apply_decision = applier.apply
    users = FullDuplexUserStateHandler(
        publish_companion_ui_state=publisher.publish_companion_ui_state,
        signal_stt_user_away=MagicMock(),
        signal_stt_user_present=MagicMock(),
        handle_speaking_started=lambda: pipeline._ducking.duck(),
        handle_speaking_stopped=lambda: None,
    )
    # Nonconstant PCM detects duplication, loss and reordering in the queued tail.
    expected = (np.arange(16000) % 2000 + 100).astype(np.int16)
    frame = rtc.AudioFrame(
        expected.tobytes(), sample_rate=16000, num_channels=1, samples_per_channel=16000
    )
    try:
        publisher.publish_companion_ui_state_for_agent_state("speaking")
        await output.capture_frame(frame)
        output.flush()
        assert output.buffered_frames == 0  # TTS already resides in LiveKit's queue.
        users.handle(SimpleNamespace(old_state="listening", new_state="speaking"))
        sink._forwarding_task = asyncio.create_task(sink._forward_audio())
        await settle()
        assert sink._audio_source.frames == []
        handler.run(state.text, is_final=True)
        await handler._intent_task
        if intent_name == "BACKCHANNEL":
            assert decisions[-1].action is Action.HOLD  # Still speaking: don't resume yet.
            assert sink._audio_source.frames == []
        state.vad = False
        if not late_user_end:
            users.handle(SimpleNamespace(old_state="speaking", new_state="listening"))
        handler.run(state.text, is_final=True)
        await settle()
        resolved_ui = [
            json.loads(c.args[0])
            for c in local.publish_data.await_args_list
            if c.kwargs["topic"] == COMPANION_UI_STATE_TOPIC
        ]
        assert resolved_ui[-1]["state"] == (
            "speaking" if intent_name == "BACKCHANNEL" else "listening"
        ), "resolution must refresh UI without waiting for another user or SDK event"
        if late_user_end:
            users.handle(SimpleNamespace(old_state="speaking", new_state="listening"))
        await settle()
        packets = local.publish_data.await_args_list
        ui = [
            json.loads(c.args[0]) for c in packets if c.kwargs["topic"] == COMPANION_UI_STATE_TOPIC
        ]
        controls = [json.loads(c.args[0]) for c in packets if c.kwargs["topic"] == CONTROL_TOPIC]
        classifier.classify.assert_awaited_once()  # Cached final evidence is reused.
        if intent_name == "BACKCHANNEL":
            assert decisions[-1].action is Action.ROLLBACK
            assert controls == [], "rejected interruption must not flush device playback"
            assert ui[-1]["state"] == "speaking"
            assert sum(p["reason"] == "agent_state:speaking" for p in ui) == 1
            await asyncio.wait_for(sink._flush_task, 1)
            played = np.concatenate(
                [np.frombuffer(f.data, dtype=np.int16) for f in sink._audio_source.frames]
            )
            np.testing.assert_array_equal(played, expected)
        else:
            assert decisions[-1].action is Action.CANCEL
            assert len(controls) == 1
            assert ui[-1]["state"] == "listening"
            output.reset()
            await settle()
            assert sink._audio_source.frames == [], "cancel must never release the old tail"
    finally:
        await handler.aclose()
        await sink.aclose()
