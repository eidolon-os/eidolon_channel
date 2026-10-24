import asyncio
from types import SimpleNamespace

import pytest
from livekit.agents import stt

from eidolon.livekit.agent.speaker_verified.input import SpeakerVerifiedInput
from eidolon.livekit.agent.speaker_verified.orchestration import SpeakerDecision
from eidolon.livekit.common.audio_timeline import AudioSpan
from eidolon.livekit.common.transcript_evidence import TranscriptEvidence
from .test_speaker_verified_orchestration import runtime as runtime


def transcript(text, revision="two", sequence=2, start=32000, end=64000):
    evidence = TranscriptEvidence(
        stream_key="input",
        revision_key=revision,
        sequence=sequence,
        audio_span=AudioSpan("input", 1, 16000, start, end),
    )
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[stt.SpeechData(language="zh", text=text, metadata=evidence.as_metadata())],
    )


async def eventually(predicate):
    async with asyncio.timeout(1):
        while not predicate():
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_continuation_during_verification_rechecks_complete_candidate_once(runtime):
    r = runtime
    routing = SpeakerVerifiedInput(r.controller, coordinator=r.owner, settle_sec=0)
    entered, release = asyncio.Event(), asyncio.Event()

    async def verify(evidence):
        if r.verify.call_count == 1:
            entered.set()
            await release.wait()
        return SpeakerDecision("match", "profile", 3, "calibration")

    r.verify.side_effect = verify
    try:
        routing.user_state_changed(SimpleNamespace(old_state="speaking", new_state="listening"))
        await entered.wait()
        routing.user_state_changed(SimpleNamespace(old_state="listening", new_state="speaking"))
        routing.observe_speech(transcript("后面还有一整句话"))
        routing.user_state_changed(SimpleNamespace(old_state="speaking", new_state="listening"))
        release.set()
        await eventually(lambda: r.owner.active.state == "committed")
        assert r.verify.call_count == 2
        assert r.verify.call_args_list[1].args[0].end == 64000
        r.session.interrupt.assert_called_once()
        r.session.generate_reply.assert_called_once()
        assert "后面还有一整句话" in r.session.generate_reply.call_args.kwargs["user_input"]
    finally:
        await routing.aclose()


@pytest.mark.asyncio
async def test_incomplete_eot_does_not_poll_or_collect_forever(runtime):
    r = runtime
    r.complete.return_value = False
    routing = SpeakerVerifiedInput(
        r.controller, coordinator=r.owner, settle_sec=0, candidate_timeout_sec=0.03
    )
    try:
        routing.user_state_changed(SimpleNamespace(old_state="listening", new_state="speaking"))
        routing.observe_speech(transcript("尚未说完的内容"))
        routing.user_state_changed(SimpleNamespace(old_state="speaking", new_state="listening"))
        await eventually(lambda: r.owner.active.state == "rejected")
        assert r.owner.active.reject_reason == "candidate_timeout"
        r.complete.assert_awaited_once()
        r.verify.assert_not_called()
        r.session.interrupt.assert_not_called()
    finally:
        await routing.aclose()


@pytest.mark.asyncio
async def test_retired_final_cannot_reopen_or_change_committed_turn(runtime):
    r = runtime
    routing = SpeakerVerifiedInput(r.controller, coordinator=r.owner, settle_sec=0)
    try:
        await r.controller.consider()
        before = r.owner.selected_text
        routing.observe_speech(
            transcript("迟到更正", revision="one", sequence=1, start=0, end=32000)
        )
        assert r.owner.active.state == "committed"
        assert r.owner.selected_text == before
        assert r.controller.pending is None
    finally:
        await routing.aclose()


@pytest.mark.asyncio
async def test_immediate_close_cleans_not_yet_started_verification(runtime):
    r = runtime
    task = r.controller.consider()
    await r.controller.aclose()
    assert task.done() and r.controller.pending is None
    r.verify.assert_not_called()
    r.session.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_rejected_background_does_not_block_next_mapped_turn(runtime):
    r = runtime
    routing = SpeakerVerifiedInput(r.controller, coordinator=r.owner, settle_sec=0)
    try:
        r.verify.return_value = SpeakerDecision("non_match")
        await r.controller.consider()
        r.verify.return_value = SpeakerDecision("match", "profile", 3, "calibration")
        routing.observe_speech(transcript("新的真实用户话语"))
        await eventually(lambda: r.owner.active.state == "committed")
        assert r.owner.selected_text == "新的真实用户话语"
        r.session.generate_reply.assert_called_once()
    finally:
        await routing.aclose()
