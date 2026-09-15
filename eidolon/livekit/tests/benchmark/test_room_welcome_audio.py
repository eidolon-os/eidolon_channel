"""Audio-only setup requires received sound and an actual playback lifecycle."""

import asyncio
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
from livekit import rtc

from benchmark import livekit_room_runner as runner
from eidolon.livekit.common.welcome import WelcomeAudio


def _participant(identity="agent", *, is_agent=True):
    return SimpleNamespace(
        identity=identity,
        kind=(rtc.ParticipantKind.PARTICIPANT_KIND_AGENT if is_agent
              else rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD),
    )


async def test_sound_wait_requires_playback_completion_then_receiver_quiet():
    state = runner._RoomCaseState(started=time.monotonic(), events=[])
    state.first_agent_audio.set()
    state.last_agent_audio_monotonic = time.monotonic() - 1
    agent = _participant()
    runner._record_agent_state({"lk.agent.state": "listening"}, agent, state)
    wait = asyncio.create_task(runner._wait_for_agent_quiet(
        state, quiet_ms=50, timeout_sec=1, greeting_expected=True, greeting_audio_only=True,
    ))
    await asyncio.sleep(0.06)
    assert not wait.done()  # Initial listening is not proof of completed playback.
    runner._record_agent_state({"lk.agent.state": "speaking"}, agent, state)
    await asyncio.sleep(0.06)
    assert not wait.done()  # Silence alone is insufficient.
    state.last_agent_audio_monotonic = time.monotonic()
    runner._record_agent_state({"lk.agent.state": "listening"}, agent, state)
    await asyncio.sleep(0.01)
    assert not wait.done()  # The receiver must also drain.
    assert await wait is True
    assert state.agent_transcript_final_timestamps == []


async def test_sound_completion_without_received_audio_fails_setup():
    state = runner._RoomCaseState(started=time.monotonic(), events=[])
    agent = _participant()
    runner._record_agent_state({"lk.agent.state": "speaking"}, agent, state)
    runner._record_agent_state({"lk.agent.state": "listening"}, agent, state)
    assert await runner._wait_for_agent_quiet(
        state, quiet_ms=10, timeout_sec=.08, first_audio_wait_sec=.01,
        greeting_expected=True, greeting_audio_only=True,
    ) is False


async def test_text_still_requires_transcript_even_when_sound_lifecycle_completes():
    state = runner._RoomCaseState(started=time.monotonic(), events=[])
    state.first_agent_audio.set()
    state.agent_playout_finished_timestamps.append(10)
    state.last_agent_audio_monotonic = time.monotonic() - 1
    assert await runner._wait_for_agent_quiet(
        state, quiet_ms=10, timeout_sec=.08, greeting_expected=True,
    ) is False


def test_playback_completion_requires_same_agent_and_ignores_user_attributes():
    state = runner._RoomCaseState(started=time.monotonic(), events=[])
    for value in ("speaking", "listening"):
        runner._record_agent_state({"lk.agent.state": value}, _participant(is_agent=False), state)
    runner._record_agent_state({"lk.agent.state": "speaking"}, _participant("a"), state)
    runner._record_agent_state({"lk.agent.state": "listening"}, _participant("b"), state)
    assert state.agent_playout_finished_timestamps == []


@pytest.mark.parametrize("welcome,audio_only,expected", [
    (WelcomeAudio(), True, True), ("欢迎", False, True), ("", False, False),
])
async def test_room_suite_selects_welcome_completion_contract(
    tmp_path, monkeypatch, welcome, audio_only, expected,
):
    cfg = runner.load_effective_config()
    cfg = replace(cfg, behavior=replace(cfg.behavior, welcome_message=welcome))
    monkeypatch.setattr(runner, "load_effective_config", lambda: cfg)
    observed = []

    async def run_case(case, **kwargs):
        observed.append(kwargs["options"])
        return runner.CaseResult(case_id="test", suite="test", runner="livekit_room", passed=True)

    monkeypatch.setattr(runner, "_run_room_case_with_retries", run_case)
    suite = SimpleNamespace(cases=[object()])
    for intent in ("user_initiated", "presence_initiated", "proactive_initiated"):
        await runner.run_livekit_room_suite([suite], root=tmp_path, options=runner.LiveKitRoomOptions(
            participant_metadata={"session_intent": intent},
        ))
    assert [(o.greeting_expected, o.greeting_audio_only) for o in observed] == [
        (expected, audio_only), (expected, audio_only), (False, False),
    ]


def test_sound_is_not_used_as_spoken_echo_context():
    from benchmark.policy_runner import _assistant_speech_ledger

    case = SimpleNamespace(device_envelope=SimpleNamespace(agent=SimpleNamespace(speaking_text="")))
    assert _assistant_speech_ledger(case, welcome_message=WelcomeAudio()).latest is None
