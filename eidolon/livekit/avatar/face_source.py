"""Resolve a companion's configured display face (digital-human ``cond_image``).

The avatar worker seeds the digital-human service with a per-companion still
image.  This module turns the session's runtime participant into that image's
JPEG bytes by resolving the bound companion — via the *same* Eidolon Data
resolution the runtime token uses, so the face always matches the companion
that will actually speak — and reading its active face asset from the object
store.

It is deliberately best-effort: any failure (no local data store, an
unresolvable participant, no configured face, a missing/corrupt blob) returns
``None`` so the worker falls back to the service's own default avatar.  The
capability is purely additive — audio-only and unconfigured sessions are
unaffected and there is no regression path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("agent.avatar.face")


async def resolve_session_face_image(room: Any, *, runtime_admin: Any) -> bytes | None:
    """Return the JPEG bytes of the session companion's configured face, or ``None``.

    ``None`` means "use the digital-human service's default avatar" — it is the
    correct, non-error result for audio-only-configured companions and for any
    environment where local Eidolon Data is unavailable.
    """
    if runtime_admin is not None and not getattr(runtime_admin, "data_resolve_enabled", True):
        return None
    store = _open_local_store()
    if store is None:
        return None
    try:
        companion_id = await _resolve_companion_id(room, store)
        if companion_id is None:
            return None
        asset = await store.companion_face_assets.get_active(companion_id)
        if asset is None:
            logger.info(
                "[avatar.face] companion=%s has no configured face; using default avatar",
                companion_id,
            )
            return None
        data = _read_cond_image(store, asset)
        if data is None:
            return None
        logger.info(
            "[avatar.face] resolved cond_image companion=%s version=%s bytes=%d",
            companion_id,
            asset.version,
            len(data),
        )
        return data
    except Exception:  # noqa: BLE001 — face resolution must never break a session
        logger.warning(
            "[avatar.face] face resolution failed; using default avatar", exc_info=True
        )
        return None
    finally:
        try:
            await store.close()
        except Exception:  # noqa: BLE001 — defensive cleanup
            logger.debug("[avatar.face] data store close failed", exc_info=True)


def _open_local_store() -> Any | None:
    """Open the local Eidolon Data store, or ``None`` when it isn't present."""
    try:
        from eidolon_data import DataStore
        from eidolon_data import load_settings as load_data_settings

        data_settings = load_data_settings()
        sqlite_path = Path(data_settings.sqlite_path).expanduser()
        if not sqlite_path.exists():
            logger.info(
                "[avatar.face] Eidolon Data SQLite not found at %s; using default avatar",
                sqlite_path,
            )
            return None
        return DataStore.open(data_settings)
    except Exception:  # noqa: BLE001 — absence of local data is not an error here
        logger.warning(
            "[avatar.face] Eidolon Data unavailable; using default avatar", exc_info=True
        )
        return None


async def _resolve_companion_id(room: Any, store: Any) -> str | None:
    """Resolve the room's runtime participant to its bound companion id.

    Reuses the runtime resolver's participant selection + kind dispatch and the
    local Eidolon Data resolve client so the face matches the companion the
    session itself resolves to.
    """
    from eidolon.livekit.agent.factory import _DataStoreRuntimeResolveClient
    from eidolon.livekit.agent.runtime.resolver import (
        _participant_identity_and_metadata,
        _resolve_context,
    )

    peek = _participant_identity_and_metadata(room)
    if peek is None:
        return None
    identity, metadata = peek
    client = _DataStoreRuntimeResolveClient(store)
    _, _, ctx = await _resolve_context(admin=client, identity=identity, metadata=metadata)
    return ctx.companion_id


def _read_cond_image(store: Any, asset: Any) -> bytes | None:
    try:
        data = store.object_storage.get(asset.cond_storage_key)
    except Exception:  # noqa: BLE001 — a missing blob degrades to the default avatar
        logger.warning(
            "[avatar.face] cond_image blob unreadable key=%s; using default avatar",
            asset.cond_storage_key,
            exc_info=True,
        )
        return None
    return data or None
