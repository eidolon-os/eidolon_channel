"""Session-scoped composition for Kernel Mount and System Data runtime facts."""

from __future__ import annotations

import asyncio
from typing import Any

from eidolon_sdk.biz.persona import ResolvedRuntimeIdentity

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

    def __init__(self, *, runtime: Any, mounts: Any) -> None:
        self.runtime = runtime
        self.mounts = mounts
        self._context: ResolvedRuntimeIdentity | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    async def resolve_room(self, room: Any) -> ResolvedRuntimeIdentity:
        if self._context is not None:
            return self._context
        async with self._lock:
            if self._context is not None:
                return self._context
            peek = _participant_identity_and_metadata(room)
            if peek is None:
                raise DeviceTokenResolverError("remote runtime participant missing")
            identity, metadata = peek
            self._context = await _resolve_context(
                runtime=self.runtime,
                mounts=self.mounts,
                identity=identity,
                metadata=metadata,
            )
            return self._context

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self.runtime, "aclose", None)
        if callable(close):
            await close()


__all__ = ["ChannelRuntimeServices"]
