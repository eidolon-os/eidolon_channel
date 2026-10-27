from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from eidolon_sdk.biz.persona import ResolvedRuntimeIdentity
from eidolon_sdk.biz.runtime import RuntimeTokenVerifier
from eidolon.interaction_context import PersonalMemorySelection
from eidolon.livekit.agent.runtime.services import ChannelRuntimeServices
from eidolon.livekit.agent.runtime.resolver import make_device_token_resolver, DeviceTokenResolverError
from eidolon.livekit.agent.server import _resolve_personal_memory


def runtime():
    source = AsyncMock()
    source.resolve_owner.return_value = ResolvedRuntimeIdentity(
        owner_id="owner-a", companion_id="companion-a", device_id=None,
        memory_realm_id="default", genome_id="genome", genome_hash="hash",
        schema_version="schema", realizer_version="realizer")
    source.resolve_person_memory.return_value = "private-a"
    return source


def room():
    return SimpleNamespace(remote_participants={"owner-a": SimpleNamespace(
        identity="owner-a", metadata='{"kind":"owner","owner_id":"owner-a","personal_memory":{"person_id":"forged","identity_epoch":99}}')})


def selection():
    return PersonalMemorySelection(person_id="person-a", identity_epoch=1)


@pytest.mark.parametrize("value", [None, {}, {"person_id":"p"},
    {"person_id":"p", "identity_epoch":True}, {"person_id":"p", "identity_epoch":-1},
    {"person_id":"p", "identity_epoch":"1"}, {"person_id":" p", "identity_epoch":1},
    {"person_id":"p", "identity_epoch":1, "memory_realm_id":"forged"}])
def test_invalid_dispatch_selection_is_rejected(value):
    with pytest.raises(ValidationError):
        PersonalMemorySelection.model_validate(value)


@pytest.mark.asyncio
async def test_authoritative_personal_context_and_signed_claims():
    source = runtime()
    services = ChannelRuntimeServices(runtime=source, mounts=None, personal_memory=selection())
    channel = room()
    token = make_device_token_resolver(room=channel, runtime=source,
        context_resolver=services.resolve_room, personal_memory=selection(),
        session_id="dispatch-a", jwt_secret="test-secret-with-enough-entropy-32b")
    first = await token()
    identity = await RuntimeTokenVerifier(secret="test-secret-with-enough-entropy-32b").verify(first)
    assert (identity.person_id, identity.identity_epoch, identity.session_id) == ("person-a", 1, "dispatch-a")
    assert (await services.resolve_room(channel)).memory_realm_id == "private-a"
    assert source.resolve_owner.return_value.memory_realm_id == "default"
    source.resolve_person_memory.assert_awaited_once_with(owner_id="owner-a", person_id="person-a")
    assert await token() == first
    await services.aclose()
    with pytest.raises(DeviceTokenResolverError, match="closed"):
        await token()


@pytest.mark.asyncio
async def test_personal_failure_does_not_cache_default_context():
    source = runtime()
    source.resolve_person_memory.side_effect = LookupError("inactive")
    services = ChannelRuntimeServices(runtime=source, mounts=None, personal_memory=selection())
    for _ in range(2):
        with pytest.raises(LookupError, match="inactive"):
            await services.resolve_room(room())
    assert services._context is None
    assert source.resolve_person_memory.await_count == 2


@pytest.mark.asyncio
async def test_participant_person_claim_does_not_select_private_memory():
    source = runtime()
    services = ChannelRuntimeServices(runtime=source, mounts=None)
    assert (await services.resolve_room(room())).memory_realm_id == "default"
    source.resolve_person_memory.assert_not_awaited()
    ctx = SimpleNamespace(job=SimpleNamespace(metadata="{}"), room=room())
    assert _resolve_personal_memory(ctx) is None


@pytest.mark.parametrize("extra", ['"team_dispatch":{}', '"presentation_endpoint":{}',
                                   '"session_application":"home_command.v1"'])
def test_personal_dispatch_refuses_other_applications(extra):
    ctx = SimpleNamespace(job=SimpleNamespace(metadata=
        '{"personal_memory":{"person_id":"p","identity_epoch":0},'+extra+'}'))
    with pytest.raises(ValueError):
        _resolve_personal_memory(ctx)


def test_personal_dispatch_uses_companion_application_constant():
    ctx = SimpleNamespace(job=SimpleNamespace(metadata=
        '{"session_application":"companion.v1","personal_memory":{"person_id":"p","identity_epoch":0}}'))
    assert _resolve_personal_memory(ctx).person_id == "p"


def test_personal_session_does_not_start_companion_proactive_consumer():
    from eidolon.livekit.agent.full_duplex.lifecycle import FullDuplexSessionLifecycle
    lifecycle = object.__new__(FullDuplexSessionLifecycle)
    lifecycle._pipeline = SimpleNamespace(_factory=SimpleNamespace(
        runtime_services=SimpleNamespace(personal_memory=selection())))
    # No event loop or task fields: a subscription attempt would fail here.
    lifecycle._start_proactive_consumer()
