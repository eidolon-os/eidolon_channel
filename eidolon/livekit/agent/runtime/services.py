"""Session-scoped composition for Kernel Mount and System Data runtime facts."""

from __future__ import annotations

import asyncio
from typing import Any

from eidolon_sdk.biz.persona import ResolvedRuntimeIdentity
from eidolon.interaction_context import PersonalMemorySelection

from .resolver import (
    DeviceTokenResolverError,
    _participant_identity_and_metadata,
    _resolve_context,
)


class ChannelRuntimeServices:
    """Own and cache the authoritative runtime context for one LiveKit session.

    The Channel remains an orchestrator: Kernel owns Device Mount, System Data
    owns Companion runtime facts, and this object only composes their narrow
    consumer ports. A factory is created per LiveKit job, so one cached context
    cannot leak across rooms.
    """

    def __init__(self, *, runtime: Any, mounts: Any, target_companion_id: str | None = None,
                 personal_memory: PersonalMemorySelection | None = None) -> None:
        self.personal_memory = (PersonalMemorySelection.model_validate(personal_memory)
                                if personal_memory is not None else None)
        self.target_companion_id = target_companion_id
        self.runtime = runtime
        self.mounts = mounts
        self._context: ResolvedRuntimeIdentity | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    async def resolve_room(self, room: Any) -> ResolvedRuntimeIdentity:
        if self._closed:
            raise DeviceTokenResolverError("runtime session is closed")
        if self._context is not None:
            return self._context
        async with self._lock:
            if self._context is not None:
                return self._context
            peek = _participant_identity_and_metadata(room)
            if peek is None:
                raise DeviceTokenResolverError("remote runtime participant missing")
            identity, metadata = peek
            context = await _resolve_context(
                runtime=self.runtime,
                mounts=self.mounts,
                identity=identity,
                metadata=metadata,
                target_companion_id=self.target_companion_id,
            )
            if self.personal_memory is not None:
                realm = await self.runtime.resolve_person_memory(
                    owner_id=context.owner_id, person_id=self.personal_memory.person_id,
                    person_revision=self.personal_memory.person_revision)
                if not isinstance(realm, str) or not realm.strip():
                    raise DeviceTokenResolverError("personal memory realm is unavailable")
                context = context.model_copy(update={"memory_realm_id": realm})
            if self._closed:
                raise DeviceTokenResolverError("runtime session is closed")
            self._context = context
            return context

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self.runtime, "aclose", None)
        if callable(close):
            await close()


__all__ = ["ChannelRuntimeServices"]
