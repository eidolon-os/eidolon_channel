"""Choosing which adapter realises a device's channel.

The device declares what it needs and never chooses: it cannot know whether
this deployment runs a broker, whether LiveKit is reachable, or what the
operator prefers. The Hub does not choose either — it relays an opaque binding
precisely so it need not know. The choice belongs here, in the channel
authority, and it is a pure function so it can be reasoned about and tested
without provisioning anything.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .ports import ChannelAdapter
from .spec import ChannelSpec


class NoAdapterAvailable(RuntimeError):
    """No configured adapter can satisfy what the device declared."""


class AdapterRegistry:
    """The adapters this deployment offers, in preference order."""

    def __init__(self, adapters: Iterable[ChannelAdapter], *, preference: Iterable[str]) -> None:
        self._by_name: Mapping[str, ChannelAdapter] = {a.name: a for a in adapters}
        unknown = [name for name in preference if name not in self._by_name]
        if unknown:
            raise ValueError(f"preference names unconfigured adapters: {','.join(unknown)}")
        self._preference = tuple(preference) or tuple(self._by_name)
        if not self._by_name:
            raise ValueError("a Channel Provider needs at least one adapter")

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def get(self, name: str) -> ChannelAdapter:
        """Resolve a persisted adapter name back to its adapter.

        Revocation must reach the adapter that opened the channel even after
        preference order changes, so this lookup ignores preference entirely.
        """
        try:
            return self._by_name[name]
        except KeyError:
            raise NoAdapterAvailable(f"channel was opened by unconfigured adapter: {name}") from None

    def select(self, spec: ChannelSpec) -> ChannelAdapter:
        """Pick the first preferred adapter that can carry this spec."""
        for name in self._preference:
            adapter = self._by_name[name]
            if supports(adapter, spec):
                return adapter
        raise NoAdapterAvailable(
            f"no adapter carries audio={spec.audio.value} video={spec.video.value} "
            f"for device kind {spec.device_kind!r}"
        )

    async def healthcheck(self) -> None:
        for name in self._preference:
            await self._by_name[name].healthcheck()

    async def shutdown(self) -> None:
        for adapter in self._by_name.values():
            await adapter.shutdown()


def supports(adapter: ChannelAdapter, spec: ChannelSpec) -> bool:
    """Whether an adapter can carry everything the spec asks for.

    Media capability is the only hard constraint today: every adapter carries
    data, so a data-only device can run on any of them, while a device that
    needs audio or video rules out transports that carry no media at all.
    """
    if not spec.needs_media:
        return True
    return adapter.carries_media
