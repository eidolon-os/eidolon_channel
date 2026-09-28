"""Deliver panel messages on the device's already provisioned channel."""

from __future__ import annotations

import time
import json
from typing import Any

from eidolon.capability_runtime.smarthome import panel_command

from .contracts import UnknownChannel
from .selection import AdapterRegistry
from .store import ChannelProviderStore


class ChannelPanelSink:
    def __init__(self, store: ChannelProviderStore, adapters: AdapterRegistry) -> None:
        self._store = store
        self._adapters = adapters

    async def send(
        self, owner_id: str, device_ref: str, op: str, payload: dict[str, Any]
    ) -> None:
        # The runtime has only a panel's stable device id. Resolve the current
        # grant every time so revocation and credential refresh take effect.
        now_ms = time.time_ns() // 1_000_000
        for provision in self._store.active_provisions():
            if (
                provision.owner_id == owner_id
                and provision.device_id == device_ref
                and provision.expires_at_ms > now_ms
            ):
                adapter = self._adapters.get(provision.adapter_name)
                await adapter.send_panel_control(
                    json.loads(provision.handle_json),
                    panel_command(device_ref=device_ref, op=op, payload=payload),
                )
                return
        raise UnknownChannel("panel has no active channel")
