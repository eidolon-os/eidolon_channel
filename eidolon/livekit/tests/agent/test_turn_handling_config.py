"""AgentSession turn_handling config — mode-aware interruption options.

Root cause of "PTT tap-to-stop stops the agent but the barge-in utterance gets
no reply": in half_duplex the session is uninterruptible (allow_interruptions=
False), and the framework's ``discard_audio_if_uninterruptible`` then THREW AWAY
the user's deliberate barge-in audio → empty transcript → no reply.

Fix: ``discard_audio_if_uninterruptible`` is mode-aware — False in half_duplex
(the device gates the mic, so any playback-period audio is an intentional
barge-in that must be captured), True in full_duplex (open mic; discarding
echo while a rare uninterruptible speech plays is correct). Full-duplex barge-in
is unaffected because its speeches are interruptible (enabled=True), so the
discard path never triggers there.
"""

from __future__ import annotations

from eidolon.livekit.agent.streaming import StreamingPipeline
from eidolon.livekit.common.config import TurnPolicyConfig


def _pipe(*, is_half_duplex: bool, allow_interruptions: bool) -> StreamingPipeline:
    p = StreamingPipeline.__new__(StreamingPipeline)
    p._is_half_duplex = is_half_duplex
    p._allow_interruptions = allow_interruptions
    p._false_interruption_timeout = 6.0
    p._turn_policy = TurnPolicyConfig()
    return p


def test_half_duplex_keeps_barge_in_audio() -> None:
    th = _pipe(is_half_duplex=True, allow_interruptions=False)._build_turn_handling()
    intr = th["interruption"]
    assert intr["enabled"] is False                       # no VAD auto-interrupt
    assert intr["discard_audio_if_uninterruptible"] is False  # keep PTT barge-in
    assert intr["false_interruption_timeout"] == 6.0


def test_full_duplex_keeps_status_quo() -> None:
    th = _pipe(is_half_duplex=False, allow_interruptions=True)._build_turn_handling()
    intr = th["interruption"]
    assert intr["enabled"] is True                        # VAD/evidence-gate barge-in
    assert intr["discard_audio_if_uninterruptible"] is True   # unchanged


def test_preemptive_passthrough() -> None:
    p = _pipe(is_half_duplex=False, allow_interruptions=True)
    th = p._build_turn_handling()
    assert th["preemptive_generation"]["enabled"] == p._turn_policy.preemptive.enabled
    assert (
        th["preemptive_generation"]["preemptive_tts"]
        == p._turn_policy.preemptive.preemptive_tts
    )
