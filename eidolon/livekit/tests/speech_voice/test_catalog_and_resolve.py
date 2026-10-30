from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import pytest
from eidolon_sdk.biz.speech_voice import (
    CatalogVoiceSelection,
    CompanionSpeechVoice,
    OwnerVoiceBinding,
    OwnerVoiceRuntime,
    OwnerVoiceSelection,
    VoiceRef,
)
from pydantic import ValidationError

import eidolon.speech_voice as generic
from eidolon.livekit.common.config.validators import TTS_PROVIDERS
from eidolon.livekit.plugins.tts.bailian.voice_catalog import BailianVoiceCatalogAdapter
from eidolon.livekit.plugins.tts.voice_catalogs import PENDING_PROVIDERS, VOICE_CATALOG_ADAPTERS
from eidolon.speech_voice.catalog import HostVoiceCatalogService
from eidolon.speech_voice.catalog_file import load_catalog_file
from eidolon.speech_voice.resolve import resolve, resolve_without_catalog

from .fakes import VOICES, FakeConfig, adapter_class, write_catalog

NOW = datetime(2026, 10, 31, tzinfo=UTC)


# ---------------------------------------------------------------- the catalog file contract

def test_the_bailian_catalog_obeys_the_contract_and_describes_both_defaults() -> None:
    described = {d.voice.voice_id: d.voice for d in BailianVoiceCatalogAdapter.describe()}
    assert "cosyvoice-v3-flash" in described["longanyang"].models
    assert "qwen-audio-3.1-tts-flash" in described["longanyang_v3.1"].models
    # The same speaker on both models, so a choice survives a model switch.
    assert described["longanyang_v3.1"].speaker == described["longanyang"].speaker


@pytest.mark.parametrize("change, message", [
    ({"schema": "speech_voice_catalog/v9"}, "unsupported catalog schema"),
    ({"voices": [{**VOICES[0], "gender": "robot"}]}, "unknown gender"),
    ({"voices": [{**VOICES[0], "styles": ["sparkly"]}]}, "unknown styles"),
    ({"voices": [VOICES[0], VOICES[0]]}, "duplicate voice_id"),
    ({"voices": [{**VOICES[0], "models": ["m9"]}]}, "not in the file"),
])
def test_the_contract_rejects_mistakes_in_our_own_files(tmp_path: Path, change, message) -> None:
    import yaml

    path = write_catalog(tmp_path / "voices.yaml")
    data = {**yaml.safe_load(path.read_text()), **change}
    path.write_text(yaml.safe_dump(data, allow_unicode=True))
    with pytest.raises(ValidationError, match=message):
        load_catalog_file(path, provider="fake")


def test_a_file_for_another_provider_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="describes"):
        load_catalog_file(write_catalog(tmp_path / "v.yaml", provider="other"), provider="fake")


# ---------------------------------------------------------------- registry and V10

def test_every_tts_provider_is_either_registered_or_explicitly_pending() -> None:
    assert set(TTS_PROVIDERS) == set(VOICE_CATALOG_ADAPTERS) | PENDING_PROVIDERS
    assert not set(VOICE_CATALOG_ADAPTERS) & PENDING_PROVIDERS


def test_the_generic_layer_never_names_a_provider() -> None:
    names = {*TTS_PROVIDERS, "cosyvoice", "qwen", "dashscope", "senseaudio"}
    for path in Path(generic.__file__).parent.glob("*.py"):
        strings = [n.value.lower() for n in ast.walk(ast.parse(path.read_text()))
                   if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        assert not [s for s in strings for name in names if name in s], path.name


# ---------------------------------------------------------------- the Host catalog

@pytest.fixture
def service(tmp_path: Path):
    cls = adapter_class(write_catalog(tmp_path / "voices.yaml"))
    svc = HostVoiceCatalogService(cls(FakeConfig()), [cls], tmp_path / "snap.json")
    return svc


async def test_catalog_offers_mandarin_voices_of_the_current_model_with_the_default_first(service) -> None:
    await service.refresh()
    catalog = service.catalog()
    keys = [v.ref.voice_id for v in catalog.voices]
    assert keys[0] == "warm-f" and catalog.voices[0].is_host_default
    assert "yue-f" not in keys and "gentle-f" not in keys  # no Mandarin; other model
    assert catalog.host_selectable and catalog.state == "fresh"
    assert catalog.voices[0].style_labels == ["温暖", "温柔"]


async def test_a_host_with_one_voice_is_not_selectable_and_no_adapter_is_said_so(tmp_path: Path) -> None:
    cls = adapter_class(write_catalog(tmp_path / "v.yaml", voices=[VOICES[0]]))
    single = HostVoiceCatalogService(cls(FakeConfig()), [cls], tmp_path / "s.json")
    await single.refresh()
    assert single.catalog().not_selectable_reason == "single_voice"
    none = HostVoiceCatalogService(None, [], tmp_path / "n.json").catalog()
    assert none.not_selectable_reason == "no_adapter" and none.voices == []


async def test_an_undescribed_default_is_still_listed_but_not_offered(tmp_path: Path) -> None:
    cls = adapter_class(write_catalog(tmp_path / "v.yaml"))
    svc = HostVoiceCatalogService(cls(FakeConfig(voice="mystery")), [cls], tmp_path / "s.json")
    await svc.refresh()
    default = svc.catalog().voices[0]
    assert default.ref.voice_id == "mystery" and not default.curated and not default.selectable


async def test_failures_go_stale_then_back_off_and_a_snapshot_survives_a_restart(tmp_path: Path) -> None:
    clock = [1_000.0]
    failures: list = []
    cls = adapter_class(write_catalog(tmp_path / "v.yaml"), fail=failures)
    svc = HostVoiceCatalogService(cls(FakeConfig()), [cls], tmp_path / "s.json", clock=lambda: clock[0])
    assert svc.catalog().state == "degraded"
    assert await svc.refresh()
    failures += [RuntimeError("down"), RuntimeError("down")]
    assert not await svc.refresh() and svc.catalog().state == "stale"
    assert svc.next_delay() == 60
    assert not await svc.refresh() and svc.next_delay() == 120
    # A restart reuses the last good list while the fingerprint matches...
    again = HostVoiceCatalogService(cls(FakeConfig()), [cls], tmp_path / "s.json", clock=lambda: clock[0])
    assert {v.ref.voice_id for v in again.catalog().voices} >= {"warm-f", "bright-f"}
    # ...and not after the model changed.
    moved = HostVoiceCatalogService(cls(FakeConfig(model="m2", voice="gentle-f")), [cls], tmp_path / "s.json")
    assert moved.catalog().state == "degraded"


async def test_the_schedule_is_daily_with_jitter_and_manual_refresh_is_throttled(tmp_path: Path) -> None:
    clock = [0.0]
    calls = []
    cls = adapter_class(write_catalog(tmp_path / "v.yaml"))
    svc = HostVoiceCatalogService(cls(FakeConfig()), [cls], tmp_path / "s.json",
                                  clock=lambda: clock[0], jitter=lambda: 1.0)
    original = svc.refresh

    async def counted():
        calls.append(clock[0])
        return await original()

    svc.refresh = counted
    await svc.refresh_on_request()
    clock[0] = 300
    await svc.refresh_on_request()
    clock[0] = 700
    await svc.refresh_on_request()
    assert calls == [0.0, 700]
    assert svc.next_delay() == pytest.approx(24 * 3600 * 1.1)


async def test_the_revision_follows_content(service) -> None:
    await service.refresh()
    first = service.catalog().catalog_revision
    assert service.catalog().catalog_revision == first
    service.adapter = type(service.adapter)(FakeConfig(voice="bright-f"))
    assert service.catalog().catalog_revision != first


# ---------------------------------------------------------------- resolution

def _record(selection=None, *, owner=None, state="valid") -> CompanionSpeechVoice:
    return CompanionSpeechVoice(companion_id="c1", revision=1, selection=selection,
                                record_state=state, owner_voice=owner)


def _catalog(selection=None, *, model="m1", default="warm-f"):
    return CatalogVoiceSelection(provider="fake", model=model, voice_id=selection or default, display_name="x")


@pytest.fixture
async def ready(service):
    await service.refresh()
    return service


@pytest.mark.parametrize("record, reason, voice", [
    (None, "unset", "warm-f"),
    (_record(), "unset", "warm-f"),
    (_record(state="invalid"), "invalid_record", "warm-f"),
    (_record(_catalog("bright-f")), "selected", "bright-f"),
    # m2's speaker warm-f-v2 is the same person as m1's warm-f.
    (_record(_catalog("warm-f-v2", model="m2")), "same_voice_other_model", "warm-f"),
    # gentle-f (m2) is not on m1: nearest female youth by styles is warm-f.
    (_record(_catalog("gentle-f", model="m2")), "mapped", "warm-f"),
    (_record(CatalogVoiceSelection(provider="gone", model="x", voice_id="y", display_name="y")),
     "no_equivalent", "warm-f"),
])
def test_resolution_order(ready, record, reason, voice) -> None:
    result = resolve(record, ready.catalog(), ready.overlay)
    assert (result.reason, result.ref.voice_id) == (reason, voice)


async def test_a_voice_of_unknown_gender_is_never_mapped(tmp_path: Path) -> None:
    ghost = {"voice_id": "ghost", "display_name": "影影", "gender": "unspecified", "age_band": "youth",
             "languages": ["zh-CN"], "styles": ["warm"], "rank": 200, "models": ["m2"]}
    cls = adapter_class(write_catalog(tmp_path / "v.yaml", voices=[*VOICES, ghost]))
    svc = HostVoiceCatalogService(cls(FakeConfig()), [cls], tmp_path / "s.json")
    await svc.refresh()
    result = resolve(_record(_catalog("ghost", model="m2")), svc.catalog(), svc.overlay)
    assert (result.reason, result.ref.voice_id) == ("no_equivalent", "warm-f")


def test_mapping_is_deterministic(ready) -> None:
    results = {resolve(_record(_catalog("gentle-f", model="m2")), ready.catalog(), ready.overlay).voice_key
               for _ in range(5)}
    assert len(results) == 1


def _owner(status: str | None, *, model: str = "m1") -> OwnerVoiceRuntime:
    bindings = [] if status is None else [
        OwnerVoiceBinding(provider="fake", model=model, provider_voice_id="pv1", status=status)]
    return OwnerVoiceRuntime(voice_asset_id="ov1", name="小禾", state="active", bindings=bindings)


@pytest.mark.parametrize("owner, reason", [
    (_owner("ready"), "owner_voice"),
    (_owner("deploying"), "owner_voice_pending"),
    (_owner("rejected"), "owner_voice_rejected"),
    (_owner("ready", model="m2"), "owner_voice_unavailable"),
    (None, "owner_voice_unavailable"),
])
def test_owner_voices_use_their_binding_on_this_model_or_the_default(ready, owner, reason) -> None:
    selection = OwnerVoiceSelection(voice_asset_id="ov1", display_name="小禾")
    result = resolve(_record(selection, owner=owner), ready.catalog(), ready.overlay)
    assert result.reason == reason
    assert result.ref.voice_id == ("pv1" if reason == "owner_voice" else "warm-f")


async def test_an_owner_voice_works_on_a_host_with_a_single_system_voice(tmp_path: Path) -> None:
    cls = adapter_class(write_catalog(tmp_path / "v.yaml", voices=[VOICES[0]]))
    single = HostVoiceCatalogService(cls(FakeConfig()), [cls], tmp_path / "s.json")
    await single.refresh()
    selection = OwnerVoiceSelection(voice_asset_id="ov1", display_name="小禾")
    assert resolve(_record(selection, owner=_owner("ready")), single.catalog(), single.overlay).reason == "owner_voice"
    assert resolve(_record(_catalog("bright-f")), single.catalog(), single.overlay).reason == "host_fixed_voice"


def test_without_the_catalog_a_choice_that_fits_this_config_is_kept() -> None:
    current = VoiceRef(provider="fake", model="m1", voice_id="warm-f")
    kept = resolve_without_catalog(_record(_catalog("bright-f")), current, reason="catalog_unavailable")
    assert (kept.reason, kept.ref.voice_id) == ("selected_unverified", "bright-f")
    owner = resolve_without_catalog(
        _record(OwnerVoiceSelection(voice_asset_id="ov1", display_name="小禾"), owner=_owner("ready")),
        current, reason="catalog_unavailable")
    assert (owner.reason, owner.ref.voice_id) == ("selected_unverified", "pv1")
    other_model = resolve_without_catalog(_record(_catalog("gentle-f", model="m2")), current,
                                          reason="catalog_unavailable")
    assert (other_model.reason, other_model.ref) == ("catalog_unavailable", current)
