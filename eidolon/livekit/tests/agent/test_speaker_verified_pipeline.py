"""Production owner lifecycle, without opening devices or loading another model."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from eidolon_sdk.biz.presentation import OutputSelection

from eidolon.livekit.agent.speaker_verified import pipeline as module
from eidolon.livekit.common.config.schema import EffectiveAgentConfig
from .test_speaker_verified_verifier import fixture as verifier_fixture


@pytest.fixture
def job(monkeypatch):
    monkeypatch.setenv("EIDOLON_CHANNEL_PROVIDER_TOKEN", "test-only")
    verifier, data, _, _, policy, _ = verifier_fixture()
    factory = SimpleNamespace(
        runtime_services=SimpleNamespace(runtime=data),
        runtime_session_id="session",
        output_plan=None,
        outputs=OutputSelection(speech=True),
        aclose=AsyncMock(),
    )
    media = SimpleNamespace(
        session=SimpleNamespace(on=MagicMock(), agent_state="listening"),
        input=SimpleNamespace(invalidate=MagicMock()),
        start=AsyncMock(),
        aclose=AsyncMock(),
    )
    http, eot = SimpleNamespace(aclose=AsyncMock()), SimpleNamespace(end_session=MagicMock())
    monkeypatch.setattr(module, "create_async_client", MagicMock(return_value=http))
    monkeypatch.setattr(module, "create_session_eot_model", MagicMock(return_value=eot))
    assembly = MagicMock(return_value=media)
    monkeypatch.setattr(module, "SpeakerVerifiedSession", assembly)
    monkeypatch.setattr(
        module, "wait_for_runtime_participant_identity", AsyncMock(return_value="pad")
    )
    monkeypatch.setattr(module, "stop_playback", AsyncMock())
    callbacks = dict(
        on_session_started=AsyncMock(),
        on_session_end=AsyncMock(),
        on_session_closed=AsyncMock(),
        on_idle_disconnect=AsyncMock(),
    )
    cfg = EffectiveAgentConfig()
    owner = module.SpeakerVerifiedPipeline(
        factory,
        policy=policy,
        calibration=verifier.calibration,
        cfg=cfg,
        session_intent="user_initiated",
        **callbacks,
    )
    for name in (
        "_warmup_stages",
        "_shutdown_stages",
        "_start_music_runtime",
        "_close_music_runtime",
    ):
        monkeypatch.setattr(owner, name, AsyncMock())
    monkeypatch.setattr(owner, "open_session_trace", MagicMock())
    room = SimpleNamespace(on=MagicMock(), off=MagicMock())
    return SimpleNamespace(
        owner=owner,
        media=media,
        http=http,
        eot=eot,
        room=room,
        factory=factory,
        callbacks=callbacks,
        assembly=assembly,
    )


async def test_ready_and_remote_stop_use_shared_lifecycle(job):
    r = job

    async def start(**kwargs):
        await kwargs["on_started"]()
        r.callbacks["on_session_started"].assert_awaited_once()
        # Ambient speech is not authorized activity.
        r.media.session.user_state = "speaking"
        assert not r.owner._idle._is_busy()
        r.media.session.agent_state = "speaking"
        assert r.owner._idle._is_busy()
        await r.assembly.call_args.kwargs["stop_playback"]()
        r.owner._disconnected()

    r.media.start.side_effect = start
    await r.owner.run(r.room)
    module.stop_playback.assert_awaited_once_with(r.room, "pad", "session", 0)
    r.callbacks["on_session_end"].assert_awaited_once_with("user_left")
    r.callbacks["on_session_closed"].assert_awaited_once()
    r.media.aclose.assert_awaited_once()
    r.http.aclose.assert_awaited_once()
    r.eot.end_session.assert_called_once()
    r.factory.aclose.assert_awaited_once()
    r.room.off.assert_called_once_with("disconnected", r.owner._disconnected)
    await r.owner.shutdown()
    r.factory.aclose.assert_awaited_once()


@pytest.mark.parametrize("failure", [RuntimeError("startup"), asyncio.CancelledError()])
async def test_failed_start_withdraws_job_and_closes_every_resource(job, failure):
    r = job
    r.media.start.side_effect = failure
    with pytest.raises(type(failure)):
        await r.owner.run(r.room)
    r.callbacks["on_session_closed"].assert_awaited_once()
    r.media.aclose.assert_awaited_once()
    r.http.aclose.assert_awaited_once()
    r.eot.end_session.assert_called_once()
    r.factory.aclose.assert_awaited_once()


async def test_shutdown_invalidates_before_awaiting_any_drain(job):
    r = job
    r.owner._verified = r.media

    async def drain():
        r.media.input.invalidate.assert_called_with("pipeline_shutdown")

    r.owner._close_music_runtime.side_effect = drain
    await r.owner.shutdown()


async def test_failed_http_close_cannot_strand_model_and_factory(job):
    r = job
    r.owner._http, r.owner._eot = r.http, r.eot
    r.http.aclose.side_effect = RuntimeError("close")
    with pytest.raises(RuntimeError, match="close"):
        await r.owner.shutdown()
    r.eot.end_session.assert_called_once()
    r.factory.aclose.assert_awaited_once()


async def test_failed_end_notice_still_withdraws_before_cleanup(job):
    r = job
    r.media.start.side_effect = lambda **_: r.owner._closed_event.set()
    r.callbacks["on_session_end"].side_effect = RuntimeError("notice")
    with pytest.raises(RuntimeError, match="notice"):
        await r.owner.run(r.room)
    r.callbacks["on_session_closed"].assert_awaited_once()
    r.factory.aclose.assert_awaited_once()


def test_commit_publishes_exact_text_boundary_and_records_activity(job):
    from eidolon_sdk.biz.dialogue_control import CommittedTurnDecision, TurnCommitBoundary
    from eidolon.livekit.agent.session.user_turn_coordinator import UserTurnDecision

    r = job
    setter = MagicMock()
    r.factory.llm = SimpleNamespace(llm=SimpleNamespace(set_turn_decision_metadata=setter))
    r.owner._idle = SimpleNamespace(mark_activity=MagicMock())
    r.owner._committed(
        UserTurnDecision(action="commit", candidate_id="candidate", transcript="本轮请求")
    )
    parsed = CommittedTurnDecision.from_metadata(setter.call_args.args[0])
    assert parsed.evidence.boundary is TurnCommitBoundary.SPEAKER_VERIFIED
    assert parsed.matches_text("本轮请求")
    assert not parsed.matches_text("背景人声")
    r.owner._idle.mark_activity.assert_called_once()


def test_missing_commit_transport_cannot_silently_dispatch_without_metadata(job):
    from eidolon.livekit.agent.session.user_turn_coordinator import UserTurnDecision

    with pytest.raises(RuntimeError, match="COMMIT_METADATA_UNAVAILABLE"):
        job.owner._committed(UserTurnDecision(action="commit", transcript="本轮请求"))
