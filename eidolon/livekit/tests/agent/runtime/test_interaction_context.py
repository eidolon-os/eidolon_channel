"""Source/target resolution is independent of room metadata and token signing."""

from unittest.mock import AsyncMock

import pytest

from eidolon.interaction_context import (
    InteractionSource,
    InteractionContextError,
    resolve_interaction_context,
)
from .test_channel_contexts import device_ref, runtime_context, _DEVICE_1
from eidolon.livekit.agent.runtime.resolver import DeviceConnectionContext


def mounted(target="resident"):
    mounts = AsyncMock()
    mounts.resolve.return_value = DeviceConnectionContext(
        owner_id="owner-1",
        device_id=_DEVICE_1,
        device_ref=device_ref(),
        mount_revision=7,
        answering_companion_id=target,
    )
    return mounts


@pytest.mark.parametrize("assigned", [None, "resident"])
async def test_temporary_target_preserves_source_and_assignment(assigned):
    mounts, runtime = mounted(assigned), AsyncMock()
    original = mounts.resolve.return_value
    runtime.resolve_companion.return_value = runtime_context(
        device_id=_DEVICE_1, companion_id="visitor"
    )
    result = await resolve_interaction_context(
        source=InteractionSource(owner_id="owner-1", device_id=_DEVICE_1),
        companion_id="visitor",
        runtime=runtime,
        mounts=mounts,
    )
    assert result.runtime.device_id == _DEVICE_1
    assert result.runtime.companion_id == "visitor"
    assert result.mount_revision == 7
    assert mounts.resolve.return_value == original
    assert [call[0] for call in mounts.mock_calls] == ["resolve"]
    runtime.resolve_owner.assert_not_called()


async def test_virtual_owner_can_explicitly_address_a_companion_without_mounts():
    runtime = AsyncMock()
    runtime.resolve_companion.return_value = runtime_context(companion_id="visitor")
    result = await resolve_interaction_context(
        source=InteractionSource(owner_id="owner-1"),
        companion_id="visitor",
        runtime=runtime,
        mounts=None,
    )
    assert result.runtime.companion_id == "visitor"
    runtime.resolve_owner.assert_not_called()


@pytest.mark.parametrize(
    "changes",
    [
        {"owner_id": "other-owner"},
        {"device_id": _DEVICE_1},
    ],
)
async def test_default_runtime_must_match_virtual_source(changes):
    runtime = AsyncMock()
    runtime.resolve_owner.return_value = runtime_context().model_copy(update=changes)
    with pytest.raises(InteractionContextError):
        await resolve_interaction_context(
            source=InteractionSource(owner_id="owner-1"),
            runtime=runtime,
            mounts=None,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"owner_id": "other-owner"},
        {"companion_id": "other-companion"},
        {"device_id": None},
    ],
)
async def test_explicit_target_is_checked_against_source_and_authority(changes):
    runtime = AsyncMock()
    runtime.resolve_companion.return_value = runtime_context(
        device_id=_DEVICE_1, companion_id="visitor"
    ).model_copy(update=changes)
    with pytest.raises(InteractionContextError):
        await resolve_interaction_context(
            source=InteractionSource(owner_id="owner-1", device_id=_DEVICE_1),
            companion_id="visitor",
            runtime=runtime,
            mounts=mounted(),
        )


async def test_explicit_target_never_bypasses_mount_authority():
    runtime, mounts = AsyncMock(), mounted()
    mounts.resolve.side_effect = InteractionContextError("not mounted")
    with pytest.raises(InteractionContextError, match="not mounted"):
        await resolve_interaction_context(
            source=InteractionSource(owner_id="owner-1", device_id=_DEVICE_1),
            companion_id="visitor",
            runtime=runtime,
            mounts=mounts,
        )
    runtime.resolve_companion.assert_not_called()


@pytest.mark.parametrize("kwargs", [{"owner_id": " "}, {"owner_id": "owner-1", "device_id": ""}])
def test_empty_source_identity_is_rejected(kwargs):
    with pytest.raises(InteractionContextError):
        InteractionSource(**kwargs)
