"""An Owner reads, starts and ends the one conversation a device holds."""
import json
import sqlite3
from unittest.mock import AsyncMock

import pytest
from eidolon_sdk.biz.control.device_session import DeviceSessionRequest, DeviceSessionStatus

from eidolon.channel_provider.contracts import Forbidden
from eidolon.channel_provider.ports import ServingAction, ServingRequest
from eidolon.channel_provider.store import ChannelProviderStore
from .test_personal_sessions import _device_session, ready, request


async def owner(service, action, **fields):
    result = await service.device_session(DeviceSessionRequest(device=request().device_ref, **fields),
        authenticated_owner_id="owner_1", action=action)
    return DeviceSessionStatus.model_validate(result)


def joins(adapter, receipt="accepted", *, starts=None):
    """The device answers room.join, and asks for its conversation when it starts one."""
    sent = []

    async def deliver(handle, command, *, wait_for_terminal):
        assert command["op"] == "room.join" and command["payload"] == {} and not wait_for_terminal
        sent.append(command)
        if starts:
            await adapter.watched[handle["resource"]](ServingRequest(ServingAction.START, starts))
        return receipt

    adapter.deliver_control = deliver
    return sent


async def test_a_conversation_the_device_started_is_seen_and_ended_by_its_owner(tmp_path):
    service, store, adapter = await ready(tmp_path)
    assert (await owner(service, "status")).state == "standby"
    await _device_session(service, adapter, "lifted")
    seen = await owner(service, "status")
    assert (seen.state, seen.conversation_id) == ("in_conversation", "lifted")
    with pytest.raises(Forbidden):
        await service.device_session(DeviceSessionRequest(device=request().device_ref),
                                     authenticated_owner_id="other")
    # A busy device answers a start with what it is doing; nothing is sent.
    sent = joins(adapter)
    assert await owner(service, "open") == seen and not sent
    # A late end for an earlier conversation leaves this one alone.
    assert await owner(service, "close", conversation_id="earlier") == seen
    assert (await owner(service, "close", conversation_id="lifted")).state == "standby"
    assert adapter.events[-2:] == [("stop", "lifted"), ("clear", "lifted")]
    assert store.personal_session("lifted")["state"] == "closed"
    assert (await owner(service, "close")).state == "standby"


async def test_start_asks_the_device_and_registers_what_it_then_requests(tmp_path):
    service, store, adapter = await ready(tmp_path)
    service._companion_authority = AsyncMock(return_value="partner")
    sent = joins(adapter)
    assert (await owner(service, "open")).state == "starting"
    assert len(sent) == 1 and not store.pending_personal_sessions()
    joins(adapter, starts="asked")
    started = await owner(service, "open")
    assert (started.state, started.conversation_id) == ("in_conversation", "asked")
    assert store.personal_session("asked")["binding"]["device_requested"] is True
    joins(adapter, "rejected")
    await owner(service, "close")
    refused = await owner(service, "open")
    assert (refused.state, refused.error) == ("standby", "DEVICE_REFUSED")


async def test_unconfirmed_end_keeps_the_conversation_closing(tmp_path):
    service, store, adapter = await ready(tmp_path)
    await _device_session(service, adapter, "lifted")
    adapter.cleanup_error = True
    assert (await owner(service, "close")).state == "closing"
    adapter.cleanup_error = False
    assert (await owner(service, "close")).state == "standby"


async def test_a_scene_holding_the_device_is_only_reported(tmp_path):
    service, store, adapter = await ready(tmp_path)
    service._transport_scopes[request().device_ref.device_instance_id] = object()
    sent = joins(adapter)
    for action in ("status", "open", "close"):
        assert (await owner(service, action)).state == "occupied"
    assert not sent and not adapter.events


def test_finished_scene_rows_of_the_old_entry_are_dropped(tmp_path):
    store = ChannelProviderStore(tmp_path / "provider.sqlite3")
    store.initialize()
    store.save_prepared_scene("owner_1", "old", "device_session", {"cleanup_complete": True})
    store.save_prepared_scene("owner_1", "team", "ip_role_group", {"cleanup_complete": True})
    store.cancel_scene("owner_1", "never-opened", "device_session")
    store.initialize()
    assert [row[2] for row in store.prepared_scenes()] == ["ip_role_group"]
    assert not store.scene_cancelled("owner_1", "never-opened", "device_session")
    with sqlite3.connect(store.path) as connection:
        connection.execute("INSERT INTO prepared_scenes VALUES ('owner_1','open','device_session',?)",
                           (json.dumps({"cleanup_complete": False}),))
    with pytest.raises(RuntimeError):
        store.initialize()
