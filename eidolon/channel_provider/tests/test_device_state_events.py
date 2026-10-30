"""A device's own report that one of its settable values moved.

This is the only inbound path where the device speaks without being asked, so
its tests are mostly about what must *not* be believed: a wrong sender, a wrong
topic, a value outside the bounded range, or a payload with no usable key.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from eidolon_sdk.biz.body import BODY_STATE_CHANGED_TYPE
from eidolon_sdk.biz.contracts import CONTROL_TOPIC, EVENT_TOPIC

from eidolon.channel_provider.contracts import ProvisionRequest, UnknownChannel
from eidolon.channel_provider.selection import AdapterRegistry

from .helpers import FakeAdapter, encoded, provision_payload
from .test_livekit_adapter import _DEVICE_1, _adapter, _listening, _spec
from .test_service import _UNKNOWN, _service


def _event(
    *,
    key: str = "volume",
    value: object = 42,
    device: str = _DEVICE_1,
    sender: str | None = None,
    topic: str = EVENT_TOPIC,
    event_type: str = BODY_STATE_CHANGED_TYPE,
    raw: bytes | None = None,
) -> SimpleNamespace:
    body = {
        "schema_v": 1,
        "kind": "event",
        "event_id": "evt-state-1",
        "flow_id": "flow-1",
        "causation_id": "",
        "type": event_type,
        "source": {"device_id": device, "component": "actuator"},
        "occurred_at_ms": 1_700_000_000_000,
        "expires_at_ms": 1_700_000_060_000,
        "payload": {"key": key, "value": value},
    }
    return SimpleNamespace(
        topic=topic,
        participant=SimpleNamespace(identity=sender or device),
        data=raw if raw is not None else json.dumps(body).encode(),
    )


async def _watch(monkeypatch):
    adapter, _ = _adapter()
    grant = await adapter.open(_spec(), issued_at_ms=1_000)
    connection = await _listening(monkeypatch, adapter, grant)
    return adapter, grant.handle["device"], connection


def _deliver(connection, packet) -> None:
    connection.handlers["data_received"](packet)


async def test_a_reported_change_becomes_the_latest_known_value(monkeypatch) -> None:
    adapter, device, connection = await _watch(monkeypatch)

    _deliver(connection, _event(key="volume", value=42))
    _deliver(connection, _event(key="brightness", value=90))

    assert adapter.latest_device_state(device) == {"volume": 42, "brightness": 90}
    await adapter.shutdown()


async def test_a_later_report_supersedes_the_earlier_one(monkeypatch) -> None:
    adapter, device, connection = await _watch(monkeypatch)

    _deliver(connection, _event(key="volume", value=42))
    _deliver(connection, _event(key="volume", value=7))

    assert adapter.latest_device_state(device) == {"volume": 7}
    await adapter.shutdown()


async def test_a_device_that_never_reported_has_no_state(monkeypatch) -> None:
    adapter, device, _ = await _watch(monkeypatch)

    assert adapter.latest_device_state(device) == {}
    await adapter.shutdown()


@pytest.mark.parametrize(
    "packet",
    [
        _event(topic=CONTROL_TOPIC),
        _event(sender="some-other-device"),
        _event(device="device-instance-other"),
        _event(event_type="body.other.thing"),
        _event(key="unknown"),
        _event(value=-1),
        _event(value=101),
        _event(value="42"),
        _event(value=True),
        _event(raw=b"{bad"),
        _event(raw=b"[]"),
    ],
)
async def test_an_event_that_cannot_be_trusted_changes_nothing(
    monkeypatch, packet
) -> None:
    adapter, device, connection = await _watch(monkeypatch)

    _deliver(connection, packet)

    assert adapter.latest_device_state(device) == {}
    await adapter.shutdown()


async def test_a_control_receipt_is_not_read_as_a_state_event(monkeypatch) -> None:
    adapter, device, connection = await _watch(monkeypatch)

    _deliver(
        connection,
        SimpleNamespace(
            topic=CONTROL_TOPIC,
            participant=SimpleNamespace(identity=device),
            data=json.dumps(
                {
                    "v": 1,
                    "kind": "result",
                    "ref": "c1",
                    "device_id": device,
                    "op": "volume.get",
                    "status": "completed",
                    "result": {"volume": 42},
                }
            ).encode(),
        ),
    )

    assert adapter.latest_device_state(device) == {}
    await adapter.shutdown()


async def test_the_snapshot_reports_only_what_moved(monkeypatch) -> None:
    adapter, device, connection = await _watch(monkeypatch)
    _deliver(connection, _event(key="brightness", value=30))

    assert adapter.latest_device_state(device) == {"brightness": 30}
    assert "volume" not in adapter.latest_device_state(device)
    await adapter.shutdown()


class _SubscribingAdapter(FakeAdapter):
    def __init__(self) -> None:
        super().__init__(name="livekit")
        self.observers: dict[str, list] = {}

    def watch_device_state(self, device, observer):
        self.observers.setdefault(device, []).append(observer)

        def _stop() -> None:
            watchers = self.observers.get(device, [])
            if observer in watchers:
                watchers.remove(observer)

        return _stop

    def emit(self, device: str, key: str, value: int) -> None:
        for observer in list(self.observers.get(device, ())):
            observer(key, value)


async def test_a_watcher_is_told_only_about_its_own_device(tmp_path) -> None:
    service, store, _ = _service(tmp_path, [1_700_000_000_000])
    await service.provision(ProvisionRequest.parse(encoded(provision_payload())))
    device_id = store.active_provisions()[0].device_id
    adapter = _SubscribingAdapter()
    service._registry = AdapterRegistry([adapter], preference=("livekit",))

    seen: list[tuple[str, str, int]] = []
    stop = await service.watch_device_state(device_id, lambda k, v: seen.append((k, v)))
    adapter.emit(device_id, "volume", 40)
    adapter.emit("device-instance-other", "volume", 99)
    stop()
    adapter.emit(device_id, "volume", 10)

    assert seen == [("volume", 40)]


async def test_watching_an_unknown_device_is_refused(tmp_path) -> None:
    service, _store, _ = _service(tmp_path, [1_700_000_000_000])

    with pytest.raises(UnknownChannel):
        await service.watch_device_state(_UNKNOWN, lambda k, v: None)


async def test_an_adapter_without_subscription_still_returns_a_stopper(tmp_path) -> None:
    service, store, _ = _service(tmp_path, [1_700_000_000_000])
    await service.provision(ProvisionRequest.parse(encoded(provision_payload())))
    device_id = store.active_provisions()[0].device_id

    stop = await service.watch_device_state(device_id, lambda k, v: None)

    stop()
