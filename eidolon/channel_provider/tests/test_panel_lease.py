"""A connected listener does not outlive its authority; refresh restores delivery."""
from eidolon.capability_runtime.smarthome.runtime import SmartHomeRuntime
from eidolon.capability_runtime.smarthome.tests.test_runtime import Backend
from eidolon.channel_provider.contracts import ProvisionRequest, RevokeRequest
from eidolon.channel_provider.smarthome_panel import ChannelPanelSink

from .helpers import FakeAdapter, encoded, provision_payload, revoke_payload
from .test_service import _service


class PanelAdapter(FakeAdapter):
    def __init__(self):
        super().__init__(name="livekit")
        self.deliveries = []
        self.panel = None

    async def accept_requests(self, handle, *, sink, panel_sink=None):
        await super().accept_requests(handle, sink=sink)
        self.panel = panel_sink

    async def send_panel_control(self, handle, command):
        self.deliveries.append(command)


async def test_live_listener_expiry_refresh_and_revocation(tmp_path, monkeypatch):
    clock = [1_700_000_000_000]
    adapter = PanelAdapter()
    service, store, _ = _service(tmp_path, clock, adapter)
    service._smarthome = SmartHomeRuntime(
        backend=Backend(), panels=ChannelPanelSink(store, service._registry),
        now_ms=lambda: clock[0],
    )
    monkeypatch.setattr("eidolon.channel_provider.smarthome_panel.time.time_ns",
                        lambda: clock[0] * 1_000_000)
    request = ProvisionRequest.parse(encoded(provision_payload()))
    await service.provision(request)
    expiry = store.active_device(request.device_ref).expires_at_ms
    sync = {"schema_v": 1, "type": "smarthome.sync", "payload": {
        "schema_version": 1, "known_revision": None, "known_seq": None}}

    for now, count in [(expiry - 1, 1), (expiry, 1), (expiry + 2000, 1)]:
        clock[0] = now
        await adapter.panel(sync)
        assert len(adapter.deliveries) == count
        assert adapter.watched  # A transport connection alone is insufficient.

    refresh = provision_payload()
    refresh.update(operation="channel.refresh-device", operation_id="renew-1")
    await service.provision(ProvisionRequest.parse(encoded(refresh)))
    await adapter.panel(sync)
    assert len(adapter.deliveries) == 2
    assert adapter.deliveries[-1]["op"] == "smarthome.snapshot"
    assert adapter.closed == []  # Existing listener/resource reused.

    await service.revoke(RevokeRequest.parse(encoded(revoke_payload())))
    await adapter.panel(sync)  # A late request cannot revive a revoked grant.
    assert len(adapter.deliveries) == 2
