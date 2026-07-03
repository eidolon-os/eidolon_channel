"""Fast-path explicit client interrupt tests.

``_handle_explicit_client_interrupt`` is the low-latency barge-in path: it fires
on the raw client.audio_state data packet, BEFORE STT produces a transcript, and
hard-cancels the agent's TTS.

PTT is the ONLY explicit client interrupt. The device's energy-gate
``manual_interrupt`` guess was removed (it falsely tripped on residual playback
echo); open-mic full_duplex barge-in is judged server-side from the clean
transcript/attention path instead. Half-duplex PTT ownership now lives in
``HalfDuplexPttPipeline``; this test covers the shared streaming fast-path for
explicit client preemption.
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
    # The state lookup itself is not under test (the packet was received per
    # production logs); the gate logic after it is.
    p._latest_client_audio_state = MagicMock(return_value=state)
    p._duck_cancel_and_interrupt = MagicMock()
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
    p._handle_explicit_client_interrupt(_packet())
    p._duck_cancel_and_interrupt.assert_called_once_with(force=True)


def test_ptt_while_generating_preempts_silent_reply() -> None:
    # Real-room regression: after a tap-to-stop, the user can press PTT again
    # while the prior reply is still in LiveKit's GENERATING state. This must
    # cancel that silent speech handle, otherwise commit_user_turn skips the new
    # reply with "current speech generation cannot be interrupted".
    p = _pipeline(
        state=_state(ptt=True, playback_state="idle"),
        pipeline_state=PipelineState.GENERATING,
    )
    p._cancel_silent_agent_generation_for_explicit_preempt = MagicMock()

    p._handle_explicit_client_interrupt(_packet())

    p._cancel_silent_agent_generation_for_explicit_preempt.assert_called_once_with()
    p._duck_cancel_and_interrupt.assert_not_called()


def test_ptt_fast_path_records_owner_decision() -> None:
    timeline = TurnTimeline("turn-ptt")
    p = _pipeline(state=_state(ptt=True), timeline=timeline)

    p._handle_explicit_client_interrupt(_packet())

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

    p._handle_explicit_client_interrupt(_packet())

    p._duck_cancel_and_interrupt.assert_called_once_with(force=True)
    p._decision_effects.record_decision_attrs.assert_not_called()
    pending = p._explicit_interrupts.pending
    assert pending is not None
    assert pending["state_attr"]["ptt"] is True
    assert isinstance(pending["received_at"], float)
    assert isinstance(pending["resolved_at"], float)

    timeline = TurnTimeline("turn-ptt-delayed")
    p._timeline = timeline
    timeline.mark("speech_started_at")

    p._apply_pending_explicit_client_interrupt(timeline)

    p._decision_effects.record_decision_attrs.assert_called_once()
    decision = p._decision_effects.record_decision_attrs.call_args.args[0]
    assert decision.action is Action.CANCEL
    assert decision.intent is InterruptIntent.HARD_STOP
    assert decision.intent_source == "client_ptt"
    assert p._explicit_interrupts.pending is None
    assert timeline.attrs["explicit_client_interrupt"]["ptt"] is True
    assert timeline.attrs["turn_control"]["source"] == "client_ptt"
    assert timeline.attrs["turn_control"]["reason"] == "explicit_client_ptt"
    assert "interrupt_started_at" in timeline.timestamps
    assert "interrupt_resolved_at" in timeline.timestamps
    assert timeline.attrs["cancel_reason"] == "explicit_client_ptt"


def test_no_signal_does_not_cancel() -> None:
    # Plain audio_state (no ptt) must do nothing.
    p = _pipeline(state=_state())
    p._handle_explicit_client_interrupt(_packet())
    p._duck_cancel_and_interrupt.assert_not_called()


def test_manual_interrupt_alone_does_nothing() -> None:
    # The removed energy-gate signal is no longer an interrupt trigger.
    p = _pipeline(state=_state(manual_interrupt=True))
    p._handle_explicit_client_interrupt(_packet())
    p._duck_cancel_and_interrupt.assert_not_called()


def test_wrong_topic_ignored() -> None:
    p = _pipeline(state=_state(ptt=True))
    pkt = SimpleNamespace(
        topic="eidolon.something_else",
        participant=SimpleNamespace(identity="dev1"),
    )
    p._handle_explicit_client_interrupt(pkt)
    p._duck_cancel_and_interrupt.assert_not_called()


def test_no_action_when_output_already_cancelled() -> None:
    # Idempotent: if the agent output is already CANCELLED, the fast path bails.
    p = _pipeline(state=_state(ptt=True), output_cancelled=True)
    p._handle_explicit_client_interrupt(_packet())
    p._duck_cancel_and_interrupt.assert_not_called()


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


def test_room_data_registration_drives_explicit_interrupt() -> None:
    # Regression for the wiring (not the logic): a client.audio_state packet
    # arriving on the registered ``data_received`` callback must reach
    # ``_handle_explicit_client_interrupt``. It used to be dead code — only
    # ``_on_room_data_received`` called it and that was never registered — so the
    # logic above was correct but never ran, and explicit client preempt never
    # fired.
    from eidolon.livekit.agent.session.room_data import RoomDataHandler

    p = StreamingPipeline.__new__(StreamingPipeline)
    p._ensure_room_data_handler = MagicMock()
    p._room_data = RoomDataHandler(get_timeline=lambda: None)
    p._handle_explicit_client_interrupt = MagicMock()

    room = _FakeRoom()
    p._install_room_data_observer(room)

    pkt = _packet()
    room.emit("data_received", pkt)

    p._handle_explicit_client_interrupt.assert_called_once_with(pkt)
