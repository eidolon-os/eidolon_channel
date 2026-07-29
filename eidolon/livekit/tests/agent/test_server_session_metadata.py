from __future__ import annotations

from types import SimpleNamespace

import pytest

from eidolon.livekit.agent.server import _resolve_session_metadata


def _participant(
    identity: str,
    metadata: str = "",
    *,
    can_publish: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        identity=identity,
        metadata=metadata,
        permissions=SimpleNamespace(can_publish=can_publish),
    )


@pytest.mark.asyncio
async def test_session_metadata_ignores_hub_control_participant():
    hub = _participant(
        "eidolon-hub-control-test",
        can_publish=False,
    )
    box = _participant(
        "box-3",
        (
            '{"kind":"device","device_id":"box-3",'
            '"interaction_mode":"full_duplex",'
            '"session_intent":"presence_initiated",'
            '"avatar_requested":false}'
        ),
        can_publish=True,
    )
    room = SimpleNamespace(
        remote_participants={
            hub.identity: hub,
            box.identity: box,
        }
    )
    ctx = SimpleNamespace(room=room)

    mode, intent, avatar = await _resolve_session_metadata(ctx)

    assert mode == "full_duplex"
    assert intent == "presence_initiated"
    assert avatar is False
