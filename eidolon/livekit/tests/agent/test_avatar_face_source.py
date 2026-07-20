"""Hermetic tests for companion cond_image resolution (no digital-human service).

Exercises the full new path: a runtime participant → bound companion → active
face asset → JPEG bytes, plus the best-effort ``None`` fallbacks that make the
capability zero-regression for audio-only / unconfigured sessions.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from eidolon.livekit.avatar.face_source import resolve_session_face_image
from eidolon_data import DataStore, load_settings

pytestmark = pytest.mark.asyncio

# JPEG SOI/EOI markers wrap an opaque body — enough for byte-equality assertions.
JPEG = b"\xff\xd8\xff" + b"configured-face-bytes" + b"\xff\xd9"


def _participant(identity: str, metadata: str) -> SimpleNamespace:
    return SimpleNamespace(identity=identity, metadata=metadata)


def _room_with(*participants) -> SimpleNamespace:
    return SimpleNamespace(remote_participants={p.identity: p for p in participants})


def _runtime_admin(*, data_resolve_enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(data_resolve_enabled=data_resolve_enabled)


async def _seed_store(tmp_path, monkeypatch, *, with_face: bool) -> None:
    """Provision a fully resolvable companion + device, optionally with a face."""
    monkeypatch.setenv("EIDOLON_DATA_SQLITE_PATH", str(tmp_path / "eidolon.sqlite3"))
    monkeypatch.setenv("EIDOLON_DATA_OBJECT_STORE_PATH", str(tmp_path / "objects"))
    store = DataStore.open(load_settings())
    try:
        await store.init_schema()
        await store.owner_service.create_owner(owner_id="owner-a", display_name="Owner A")
        workspace = await store.workspace_provisioning.provision_workspace(
            owner_id="owner-a",
            companion_id="companion-a",
            genome_id="genome-a",
            realm_id="realm-a",
        )
        await store.devices.create_device(
            device_id="esp32-a",
            owner_id="owner-a",
            status="approved",
            bound_companion_id=workspace.companion.companion_id,
        )
        if with_face:
            digest = hashlib.sha256(JPEG).hexdigest()
            key = f"owner-a/companion-avatar/companion-a/{digest[:12]}.jpg"
            store.object_storage.put(key, JPEG, expected_sha256=digest)
            await store.companion_face_assets.set_face(
                companion_id="companion-a",
                cond_storage_key=key,
                cond_content_type="image/jpeg",
                cond_size_bytes=len(JPEG),
                cond_sha256=digest,
            )
    finally:
        await store.close()


async def test_resolves_configured_face_for_device(tmp_path, monkeypatch) -> None:
    await _seed_store(tmp_path, monkeypatch, with_face=True)
    room = _room_with(_participant("esp32-a", '{"kind": "device", "device_id": "esp32-a"}'))
    assert await resolve_session_face_image(room, runtime_admin=_runtime_admin()) == JPEG


async def test_resolves_configured_face_for_owner(tmp_path, monkeypatch) -> None:
    await _seed_store(tmp_path, monkeypatch, with_face=True)
    room = _room_with(_participant("owner-a", '{"kind": "owner", "owner_id": "owner-a"}'))
    assert await resolve_session_face_image(room, runtime_admin=_runtime_admin()) == JPEG


async def test_none_when_no_face_configured(tmp_path, monkeypatch) -> None:
    await _seed_store(tmp_path, monkeypatch, with_face=False)
    room = _room_with(_participant("esp32-a", '{"kind": "device", "device_id": "esp32-a"}'))
    assert await resolve_session_face_image(room, runtime_admin=_runtime_admin()) is None


async def test_none_when_no_participant(tmp_path, monkeypatch) -> None:
    await _seed_store(tmp_path, monkeypatch, with_face=True)
    assert await resolve_session_face_image(_room_with(), runtime_admin=_runtime_admin()) is None


async def test_none_when_data_resolve_disabled(tmp_path, monkeypatch) -> None:
    await _seed_store(tmp_path, monkeypatch, with_face=True)
    room = _room_with(_participant("esp32-a", '{"kind": "device", "device_id": "esp32-a"}'))
    admin = _runtime_admin(data_resolve_enabled=False)
    assert await resolve_session_face_image(room, runtime_admin=admin) is None


async def test_none_when_store_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EIDOLON_DATA_SQLITE_PATH", str(tmp_path / "missing.sqlite3"))
    monkeypatch.setenv("EIDOLON_DATA_OBJECT_STORE_PATH", str(tmp_path / "objects"))
    room = _room_with(_participant("esp32-a", '{"kind": "device", "device_id": "esp32-a"}'))
    assert await resolve_session_face_image(room, runtime_admin=_runtime_admin()) is None
