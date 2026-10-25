"""Real session boundary: policy authority must precede any audio owner."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from eidolon_sdk.biz.voiceprint import VoicePolicySnapshot

from eidolon.livekit.agent.speaker_verified.selection import resolve_voice_policy


def runtime(mode="off", **snapshot_fields):
    snapshot = VoicePolicySnapshot(
        owner_id="owner",
        companion_id="companion",
        mode=mode,
        revision=3,
        **snapshot_fields,
    )
    services = SimpleNamespace(
        resolve_room=AsyncMock(
            return_value=SimpleNamespace(owner_id="owner", companion_id="companion")
        ),
        runtime=SimpleNamespace(get_voiceprint_snapshot=AsyncMock(return_value=snapshot)),
    )
    return services


@pytest.mark.asyncio
async def test_reads_policy_of_authoritatively_resolved_companion():
    services, room = runtime(), object()
    snapshot = await resolve_voice_policy(services, room)
    services.resolve_room.assert_awaited_once_with(room)
    services.runtime.get_voiceprint_snapshot.assert_awaited_once_with("companion")
    assert snapshot.mode == "off"


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["owner_id", "companion_id"])
async def test_wrong_policy_scope_never_falls_back_to_standard(field):
    services = runtime("allowlist")
    services.runtime.get_voiceprint_snapshot.return_value = (
        services.runtime.get_voiceprint_snapshot.return_value.model_copy(update={field: "other"})
    )
    with pytest.raises(ValueError, match="VOICE_POLICY_SCOPE_MISMATCH"):
        await resolve_voice_policy(services, object())


@pytest.mark.asyncio
async def test_authority_failure_never_becomes_off():
    services = runtime()
    services.runtime.get_voiceprint_snapshot.side_effect = TimeoutError()
    with pytest.raises(TimeoutError):
        await resolve_voice_policy(services, object())


@pytest.mark.asyncio
async def test_direct_llm_has_no_companion_policy_or_extra_services():
    assert await resolve_voice_policy(None, object()) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["full_duplex", "half_duplex", "ptt"])
async def test_server_allowlist_cannot_start_ordinary_pipeline(monkeypatch, mode):
    import json
    from unittest.mock import MagicMock
    from eidolon.livekit.agent import server, full_duplex
    from eidolon.livekit.agent.half_duplex import pipeline as ptt
    from eidolon.livekit.agent.factory import SharedStageFactory
    from eidolon.livekit.common.config.schema import EffectiveAgentConfig
    from .test_server_session_metadata import _FakeContext

    services = runtime("allowlist")
    factory = SimpleNamespace(runtime_services=services, aclose=AsyncMock())
    monkeypatch.setattr(SharedStageFactory, "from_config", lambda *a, **k: factory)
    monkeypatch.setattr(server, "_resolve_session_metadata", AsyncMock(return_value=(mode, False)))
    ordinary, manual = MagicMock(), MagicMock()
    monkeypatch.setattr(full_duplex, "StreamingPipeline", ordinary)
    monkeypatch.setattr(ptt, "HalfDuplexPttPipeline", manual)
    ctx = _FakeContext({})
    ctx.room.name = "protected"
    ctx.proc = SimpleNamespace(userdata={})
    ctx.add_shutdown_callback = MagicMock()
    notices = []

    async def publish(data, **kwargs):
        notices.append(json.loads(data))

    ctx.room.local_participant = SimpleNamespace(publish_data=publish)
    with pytest.raises(ValueError, match="SPEAKER_VERIFIED_"):
        await server.run_agent(ctx, EffectiveAgentConfig())
    ordinary.assert_not_called()
    manual.assert_not_called()
    factory.aclose.assert_awaited_once()
    assert [n["type"] for n in notices] == ["session_end"]
    assert notices[0]["reason"] == "error"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["off", "allowlist"])
async def test_server_selects_exactly_one_owner_from_companion_policy(monkeypatch, tmp_path, mode):
    from dataclasses import replace
    from unittest.mock import MagicMock
    from eidolon.livekit.agent import server, full_duplex
    from eidolon.livekit.agent.speaker_verified import pipeline as verified
    from eidolon_sdk.biz import speaker_calibration as numerical
    from eidolon.livekit.agent.factory import SharedStageFactory
    from eidolon.livekit.common.config.schema import EffectiveAgentConfig
    from .test_server_session_metadata import _FakeContext
    from .test_speaker_verified_verifier import fixture

    numerical_verifier, _, _, _, snapshot, _ = fixture()
    path = tmp_path / "test-only.json"
    path.write_text(numerical_verifier.calibration.model_dump_json())
    services = runtime()
    services.runtime.get_voiceprint_snapshot.return_value = snapshot.model_copy(
        update={"mode": mode}
    )
    factory = SimpleNamespace(runtime_services=services, aclose=AsyncMock())
    monkeypatch.setattr(SharedStageFactory, "from_config", lambda *a, **k: factory)
    monkeypatch.setattr(
        server, "_resolve_session_metadata", AsyncMock(return_value=("full_duplex", False))
    )
    ordinary = MagicMock(return_value=SimpleNamespace(run=AsyncMock()))
    selected = MagicMock(return_value=SimpleNamespace(run=AsyncMock()))
    monkeypatch.setattr(full_duplex, "StreamingPipeline", ordinary)
    monkeypatch.setattr(verified, "SpeakerVerifiedPipeline", selected)
    load = MagicMock(wraps=numerical.load_calibration)
    monkeypatch.setattr(numerical, "load_calibration", load)
    ctx = _FakeContext({})
    ctx.room.name = "same-device-any-companion"
    ctx.proc = SimpleNamespace(userdata={})
    ctx.add_shutdown_callback = MagicMock()
    ctx.room.local_participant = SimpleNamespace(publish_data=AsyncMock())
    cfg = EffectiveAgentConfig()
    cfg = replace(
        cfg,
        voiceprint=replace(cfg.voiceprint, calibration_path=str(path)),
        turn_policy=replace(cfg.turn_policy, eot=replace(cfg.turn_policy.eot, provider="firered")),
    )
    await server.run_agent(ctx, cfg)
    winner, loser = (selected, ordinary) if mode == "allowlist" else (ordinary, selected)
    winner.assert_called_once()
    winner.return_value.run.assert_awaited_once_with(ctx.room)
    loser.assert_not_called()
    assert load.call_count == (1 if mode == "allowlist" else 0)
    # From this point the selected pipeline owns factory cleanup.
    factory.aclose.assert_not_awaited()
