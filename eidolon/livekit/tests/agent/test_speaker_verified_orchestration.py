"""Observable permission ordering and races, independent of model accuracy."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from eidolon.livekit.agent.speaker_verified.orchestration import (
    SpeakerDecision,
    SpeakerVerifiedTurns,
)
from eidolon.livekit.agent.session.user_turn_coordinator import UserTurnCoordinator
from eidolon.livekit.common.audio_evidence import AudioEvidenceCapture
from eidolon.livekit.common.audio_timeline import AudioSource, AudioSpan
from eidolon.livekit.common.transcript_evidence import TranscriptEvidence


@pytest.fixture
def runtime():
    owner = UserTurnCoordinator(speech_merge_grace_sec=0.8)
    capture = AudioEvidenceCapture(session_id="session", endpoint_id="mic")
    source = AudioSource("input", 1, 16000)
    capture.select_stream("input")
    capture.begin_run(source)
    capture.append(source, start=0, pcm=b"\x01\x00" * 64000)
    owner.start_speech(timeline=None)
    owner.add_transcript(
        "合格的完整句子",
        is_final=True,
        evidence=TranscriptEvidence(
            stream_key="input",
            revision_key="one",
            sequence=1,
            audio_span=AudioSpan("input", 1, 16000, 0, 32000),
        ),
    )
    owner.note_speech_stopped(eot_score=0.9)
    speech = SimpleNamespace(id="old", done=lambda: False)
    session = SimpleNamespace(
        current_speech=speech, interrupt=AsyncMock(), generate_reply=MagicMock()
    )
    verify = AsyncMock(return_value=SpeakerDecision("match", "profile", 3, "calibration"))
    authorize = AsyncMock(return_value=True)
    intent = AsyncMock(return_value="reply")
    complete = AsyncMock(return_value=True)
    record = MagicMock()
    controller = SpeakerVerifiedTurns(
        coordinator=owner,
        capture=lambda: capture,
        session=session,
        verify=verify,
        authorize=authorize,
        classify=intent,
        is_complete=complete,
        committed=record,
        timeout_sec=0.2,
    )
    return SimpleNamespace(
        owner=owner,
        capture=capture,
        source=source,
        session=session,
        verify=verify,
        authorize=authorize,
        intent=intent,
        complete=complete,
        record=record,
        controller=controller,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["non_match", "inconclusive", "unavailable"])
async def test_identity_failure_has_no_playback_or_business_effect(runtime, status):
    r = runtime
    r.verify.return_value = SpeakerDecision(status)
    await r.controller.consider()
    assert r.owner.active.state == "rejected"
    r.session.interrupt.assert_not_called()
    r.session.generate_reply.assert_not_called()
    r.intent.assert_not_called()
    r.authorize.assert_not_called()


@pytest.mark.asyncio
async def test_identity_intent_authorization_precede_cancel_and_commit(runtime):
    r = runtime
    order = []

    async def verify(evidence):
        assert evidence.pcm == b"\x01\x00" * 32000
        order.append("verify")
        return SpeakerDecision("match", "profile", 3, "calibration")

    async def intent(text, output):
        order.append("intent")
        return "reply"

    async def authorize(decision):
        order.append("authorize")
        return True

    async def interrupt(**kwargs):
        order.append("interrupt")
        r.session.current_speech = None

    r.verify.side_effect, r.intent.side_effect = verify, intent
    r.authorize.side_effect, r.session.interrupt.side_effect = authorize, interrupt
    await r.controller.consider()
    assert order == ["verify", "intent", "authorize", "interrupt", "authorize"]
    r.session.generate_reply.assert_called_once_with(
        user_input="合格的完整句子", input_modality="audio", allow_interruptions=False
    )
    assert r.owner.active.state == "committed"
    assert r.controller.consider() is None
    r.record.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["continuation", "source", "stop", "close"])
async def test_late_match_never_controls_changed_candidate(runtime, change):
    r = runtime
    entered, release = asyncio.Event(), asyncio.Event()

    async def verify(evidence):
        entered.set()
        await release.wait()
        return SpeakerDecision("match", "profile", 3, "calibration")

    r.verify.side_effect = verify
    task = r.controller.consider()
    await entered.wait()
    assert r.controller.consider() is None  # no queued extraction
    if change == "continuation":
        r.owner.start_speech(timeline=None, continue_pending=True)
    elif change == "source":
        r.capture.invalidate("replaced")
    elif change == "stop":
        r.controller.invalidate("explicit_stop")
    else:
        await r.controller.aclose()
    release.set()
    await asyncio.gather(task, return_exceptions=True)
    r.session.interrupt.assert_not_called()
    r.session.generate_reply.assert_not_called()
    r.authorize.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["no_action", "stop_output"])
async def test_intent_does_not_always_generate_reply(runtime, action):
    r = runtime
    r.intent.return_value = action
    await r.controller.consider()
    r.session.generate_reply.assert_not_called()
    assert r.session.interrupt.call_count == (1 if action == "stop_output" else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "revoked", "output_changed", "not_complete"])
async def test_failure_keeps_current_playback_and_allows_next_candidate(runtime, failure):
    r = runtime
    if failure == "timeout":

        async def wait(evidence):
            await asyncio.Event().wait()

        r.verify.side_effect = wait
    elif failure == "revoked":
        r.authorize.return_value = False
    elif failure == "output_changed":

        async def authorize(decision):
            r.session.current_speech = SimpleNamespace(id="new", done=lambda: False)
            return True

        r.authorize.side_effect = authorize
    else:
        r.complete.return_value = False
    await r.controller.consider()
    r.session.interrupt.assert_not_called()
    r.session.generate_reply.assert_not_called()
    # A completed/rejected candidate cannot trap the next turn behind a pending task.
    assert r.controller.pending is None


@pytest.mark.asyncio
async def test_revocation_while_cancel_drains_blocks_reply(runtime):
    r = runtime
    r.authorize.side_effect = [True, False]
    await r.controller.consider()
    r.session.interrupt.assert_called_once()
    r.session.generate_reply.assert_not_called()
    assert r.owner.active.state == "rejected"


@pytest.mark.asyncio
async def test_dispatch_failure_after_commit_is_not_reported_as_identity_rejection(runtime):
    r = runtime
    r.session.generate_reply.side_effect = RuntimeError("dispatch unavailable")
    with pytest.raises(RuntimeError, match="dispatch unavailable"):
        await r.controller.consider()
    assert r.owner.active.state == "committed"
    assert r.controller.pending is None
