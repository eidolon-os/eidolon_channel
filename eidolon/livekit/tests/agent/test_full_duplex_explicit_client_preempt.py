"""Full-duplex fast-path explicit client preempt tests.

``ExplicitClientPreemptHandler`` is the low-latency barge-in path: it fires
on the raw client.audio_state data packet, BEFORE STT produces a transcript, and
hard-cancels the agent's TTS.

PTT is the only full-duplex explicit client preempt signal. The device's
energy-gate ``manual_interrupt`` guess was removed (it falsely tripped on
residual playback echo); open-mic full_duplex barge-in is judged server-side
from the clean transcript/attention path instead. Half-duplex PTT ownership now
lives in ``HalfDuplexPttPipeline``; this test covers the full-duplex fast path
for explicit client preemption.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from eidolon_sdk.biz.contracts import CLIENT_AUDIO_STATE_TOPIC

from eidolon.livekit.agent.integration.client_audio_state import ClientAudioState
from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.output.ducking import OutputDuckingController
from eidolon.livekit.agent.pipeline.types import PipelineState
from eidolon.livekit.agent.full_duplex import StreamingPipeline
from eidolon.livekit.agent.turn_policy import Action, InterruptIntent, TurnPolicyRuntime
from eidolon.livekit.common.config import TurnPolicyConfig


def _pipeline(
    *,
    state: ClientAudioState | None,
    pipeline_state: PipelineState = PipelineState.SPEAKING,
    output_cancelled: bool = False,
    timeline: TurnTimeline | None = None,
) -> StreamingPipeline:
    p = StreamingPipeline.__new__(StreamingPipeline)
    p._turn_policy = TurnPolicyConfig()
    p._state = pipeline_state
    p._timeline = timeline
    p._turn_runtime = TurnPolicyRuntime(p._turn_policy)
    p._ensure_decision_effect_applier = MagicMock()
    p._decision_effects = MagicMock()
    if output_cancelled:
        # is_cancelled requires a mixer in CANCELLED state; stub it directly.
        p._ducking = MagicMock(is_cancelled=True)
    else:
        p._ducking = OutputDuckingController()
    def _agent_output_active_for_interrupts(*, participant_identity=None) -> bool:
        del participant_identity
        return pipeline_state == PipelineState.SPEAKING or (
            state is not None and state.playback_state == "agent_speaking"
        )

    # The state lookup itself is not under test (the packet was received per
    # production logs); the gate logic after it is.
    p._client_audio_state = SimpleNamespace(
        latest_state=MagicMock(return_value=state),
        agent_output_active_for_interrupts=MagicMock(
            side_effect=_agent_output_active_for_interrupts,
        ),
    )
    p._ensure_client_audio_state_view = MagicMock(return_value=p._client_audio_state)
    effects = SimpleNamespace(
        cancel_and_interrupt=MagicMock(),
        cancel_silent_generation_for_explicit_preempt=MagicMock(),
        rollback_if_suspended=MagicMock(),
        handle_hold_decision=MagicMock(),
    )
    p._interruption_effects = effects
    p._ensure_interruption_effects = MagicMock(return_value=effects)
    p._client_preempts = p._build_client_preempt_handler()
    return p


def _packet(identity: str = "dev1") -> SimpleNamespace:
    return SimpleNamespace(
        topic=CLIENT_AUDIO_STATE_TOPIC,
        participant=SimpleNamespace(identity=identity),
    )


def _state(**kwargs) -> ClientAudioState:
    base = {
        "participant_identity": "dev1",
        "playback_state": "agent_speaking",
    }
    base.update(kwargs)
    return ClientAudioState(**base)


def test_ptt_while_speaking_force_cancels() -> None:
    # PTT is a deliberate button press → immediate hard cut, force=True so it
    # cuts through an uninterruptible framework speech handle.
    p = _pipeline(state=_state(ptt=True))
    p._client_preempts.handle_explicit_client_preempt(_packet())
    p._interruption_effects.cancel_and_interrupt.assert_called_once_with(force=True)


def test_ptt_while_generating_preempts_silent_reply() -> None:
    # Real-room regression: after a tap-to-stop, the user can press PTT again
    # while the prior reply is still in LiveKit's GENERATING state. This must
    # cancel that silent speech handle, otherwise commit_user_turn skips the new
    # reply with "current speech generation cannot be interrupted".
    p = _pipeline(
        state=_state(ptt=True, playback_state="idle"),
        pipeline_state=PipelineState.GENERATING,
    )
    p._client_preempts.handle_explicit_client_preempt(_packet())

    p._interruption_effects.cancel_silent_generation_for_explicit_preempt.assert_called_once_with()
    p._interruption_effects.cancel_and_interrupt.assert_not_called()


def test_ptt_fast_path_records_owner_decision() -> None:
    timeline = TurnTimeline("turn-ptt")
    p = _pipeline(state=_state(ptt=True), timeline=timeline)

    p._client_preempts.handle_explicit_client_preempt(_packet())

    p._decision_effects.record_decision_attrs.assert_called_once()
    decision = p._decision_effects.record_decision_attrs.call_args.args[0]
    assert decision.action is Action.CANCEL
    assert decision.intent is InterruptIntent.HARD_STOP
    assert decision.intent_source == "client_ptt"
    assert timeline.attrs["turn_control"]["source"] == "client_ptt"
    assert timeline.attrs["turn_control"]["reason"] == "explicit_client_ptt"
    assert "interrupt_started_at" in timeline.timestamps


def test_ptt_before_turn_timeline_is_attached_to_next_speech_timeline() -> None:
    p = _pipeline(state=_state(ptt=True), timeline=None)

    p._client_preempts.handle_explicit_client_preempt(_packet())

    p._interruption_effects.cancel_and_interrupt.assert_called_once_with(force=True)
    p._decision_effects.record_decision_attrs.assert_not_called()
    pending = p._explicit_preempts.pending
    assert pending is not None
    assert pending["state_attr"]["ptt"] is True
    assert isinstance(pending["received_at"], float)
    assert isinstance(pending["resolved_at"], float)

    timeline = TurnTimeline("turn-ptt-delayed")
    p._timeline = timeline
    timeline.mark("speech_started_at")

    p._apply_pending_explicit_client_preempt(timeline)

    p._decision_effects.record_decision_attrs.assert_called_once()
    decision = p._decision_effects.record_decision_attrs.call_args.args[0]
    assert decision.action is Action.CANCEL
    assert decision.intent is InterruptIntent.HARD_STOP
    assert decision.intent_source == "client_ptt"
    assert p._explicit_preempts.pending is None
    assert timeline.attrs["explicit_client_interrupt"]["ptt"] is True
    assert timeline.attrs["turn_control"]["source"] == "client_ptt"
    assert timeline.attrs["turn_control"]["reason"] == "explicit_client_ptt"
    assert "interrupt_started_at" in timeline.timestamps
    assert "interrupt_resolved_at" in timeline.timestamps
    assert timeline.attrs["cancel_reason"] == "explicit_client_ptt"


def test_no_signal_does_not_cancel() -> None:
    # Plain audio_state (no ptt) must do nothing.
    p = _pipeline(state=_state())
    p._client_preempts.handle_explicit_client_preempt(_packet())
    p._interruption_effects.cancel_and_interrupt.assert_not_called()


def test_manual_interrupt_alone_does_nothing() -> None:
    # The removed energy-gate signal is no longer an interrupt trigger.
    p = _pipeline(state=_state(manual_interrupt=True))
    p._client_preempts.handle_explicit_client_preempt(_packet())
    p._interruption_effects.cancel_and_interrupt.assert_not_called()


def test_wrong_topic_ignored() -> None:
    p = _pipeline(state=_state(ptt=True))
    pkt = SimpleNamespace(
        topic="eidolon.something_else",
        participant=SimpleNamespace(identity="dev1"),
    )
    p._client_preempts.handle_explicit_client_preempt(pkt)
    p._interruption_effects.cancel_and_interrupt.assert_not_called()


def test_no_action_when_output_already_cancelled() -> None:
    # Idempotent: if the agent output is already CANCELLED, the fast path bails.
    p = _pipeline(state=_state(ptt=True), output_cancelled=True)
    p._client_preempts.handle_explicit_client_preempt(_packet())
    p._interruption_effects.cancel_and_interrupt.assert_not_called()


class _FakeRoom:
    def __init__(self) -> None:
        self._handlers: dict[str, list] = {}

    def on(self, event: str):
        def _register(fn):
            self._handlers.setdefault(event, []).append(fn)
            return fn

        return _register

    def emit(self, event: str, packet) -> None:
        for fn in self._handlers.get(event, []):
            fn(packet)


def test_room_data_registration_drives_explicit_preempt() -> None:
    # Regression for the wiring (not the logic): a client.audio_state packet
    # arriving on the registered ``data_received`` callback must reach
    # ``ExplicitClientPreemptHandler`` after ``RoomDataHandler`` stores the
    # latest client state.
    from eidolon.livekit.agent.session.room_data import RoomDataHandler

    p = StreamingPipeline.__new__(StreamingPipeline)
    p._ensure_room_data_handler = MagicMock()
    p._room_data = RoomDataHandler(get_timeline=lambda: None)
    p._ensure_client_preempt_handler = MagicMock()
    p._client_preempts = SimpleNamespace(on_client_room_packet=MagicMock())

    room = _FakeRoom()
    p._ensure_room_data_bridge().install(room)

    pkt = _packet()
    room.emit("data_received", pkt)

    p._client_preempts.on_client_room_packet.assert_called_once_with(pkt)
