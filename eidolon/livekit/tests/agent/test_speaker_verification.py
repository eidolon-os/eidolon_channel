from __future__ import annotations

import pytest

from eidolon.livekit.agent.speaker_verification import (
    SpeakerVerificationService,
    VoiceprintEmbedding,
    VoiceprintProfile,
    VoiceprintStore,
    default_profile_id,
    default_voiceprint_root,
)
from eidolon.livekit.plugins.speaker_verification import (
    NoopSpeakerVerificationProvider,
)


def test_voiceprint_store_round_trips_profile(tmp_path) -> None:
    store = VoiceprintStore(tmp_path)
    profile = store.load_profile(tenant_id="demo", user_id="alice")
    assert profile is None

    provider = NoopSpeakerVerificationProvider()
    # 100 ms of 16 kHz mono PCM.
    audio = b"\x00\x00" * 1600

    async def _enroll():
        return await provider.enroll(
            tenant_id="demo",
            user_id="alice",
            audio_segments=[audio],
            sample_rate=16000,
        )

    import asyncio

    profile = asyncio.run(_enroll())
    path = store.save_profile(profile)

    assert path.exists()
    assert path.name == "profile.json"
    loaded = store.load_profile(tenant_id="demo", user_id="alice")
    assert loaded is not None
    assert loaded.profile_id == default_profile_id("alice")
    assert loaded.duration_ms == 100
    assert loaded.quality["accepted_segments"] == 1


def test_voiceprint_store_loads_admin_embedding_ref(tmp_path) -> None:
    store = VoiceprintStore(tmp_path)
    profile = VoiceprintProfile(
        profile_id=default_profile_id("alice"),
        tenant_id="demo",
        user_id="alice",
        provider="3d_speaker",
        model="campplus_zh_16k_common",
        embedding_ref="demo/alice/embeddings/vp_alice_default.json",
    )
    store.save_profile(profile)
    embedding_path = tmp_path / profile.embedding_ref
    embedding_path.parent.mkdir(parents=True)
    embedding_path.write_text(
        """
{
  "profile_id": "vp_alice_default",
  "tenant_id": "demo",
  "user_id": "alice",
  "provider": "3d_speaker",
  "model": "campplus_zh_16k_common",
  "vector": [0.1, 0.2, 0.3],
  "dim": 3
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    loaded = store.load_profile(tenant_id="demo", user_id="alice")
    assert loaded is not None
    embedding = store.load_embedding(loaded)
    assert isinstance(embedding, VoiceprintEmbedding)
    assert embedding.vector == (0.1, 0.2, 0.3)
    assert embedding.dim == 3


def test_default_voiceprint_root_uses_eidolon_home(monkeypatch) -> None:
    monkeypatch.delenv("EIDOLON_VOICEPRINT_ROOT", raising=False)
    assert default_voiceprint_root().name == "voiceprints"
    assert default_voiceprint_root().parent.name == "data"


def test_default_voiceprint_root_honors_shared_env(monkeypatch, tmp_path) -> None:
    target = tmp_path / "vp"
    monkeypatch.setenv("EIDOLON_VOICEPRINT_ROOT", str(target))
    assert default_voiceprint_root() == target


@pytest.mark.asyncio
async def test_service_degrades_when_profile_missing(tmp_path) -> None:
    service = SpeakerVerificationService(
        provider=NoopSpeakerVerificationProvider(),
        store=VoiceprintStore(tmp_path),
    )

    signal = await service.verify_turn(
        tenant_id="demo",
        user_id="alice",
        audio=b"\x00\x00" * 1600,
        sample_rate=16000,
        audio_ms=100,
    )

    assert signal.known is False
    assert signal.error == "profile_not_found"
    assert signal.audio_ms == 100
    assert signal.latency_ms is not None
    assert signal.as_timeline_attrs()["provider"] == "noop"


@pytest.mark.asyncio
async def test_service_enrolls_and_noop_verify_reports_disabled(tmp_path) -> None:
    service = SpeakerVerificationService(
        provider=NoopSpeakerVerificationProvider(),
        store=VoiceprintStore(tmp_path),
    )
    profile = await service.enroll_user(
        tenant_id="demo",
        user_id="alice",
        audio_segments=[b"\x01\x00" * 1600],
        sample_rate=16000,
    )

    signal = await service.verify_turn(
        tenant_id="demo",
        user_id="alice",
        audio=b"\x02\x00" * 1600,
        sample_rate=16000,
        audio_ms=100,
    )

    assert profile.profile_id == default_profile_id("alice")
    assert signal.known is False
    assert signal.error == "provider_disabled"
    assert signal.profile_id == profile.profile_id
