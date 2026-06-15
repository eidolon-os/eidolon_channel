from __future__ import annotations

import jwt

from eidolon.livekit.common import token as token_mod
from eidolon.livekit.common.config import AgentConfig, CoreConfig


SECRET = "test-secret-with-enough-entropy-32b"


def test_generate_token_uses_sdk_livekit_dispatch(monkeypatch) -> None:
    monkeypatch.setattr(
        token_mod,
        "_cached_cfg",
        AgentConfig(core=CoreConfig(api_key="devkey", api_secret=SECRET)),
    )

    identity, token = token_mod.generate_token("room-a", "alice")

    payload = jwt.decode(token, SECRET, algorithms=["HS256"])
    assert identity == "alice"
    assert payload["sub"] == "alice"
    assert payload["name"] == "alice"
    assert payload["video"]["room"] == "room-a"
    assert payload["video"]["roomJoin"] is True
    assert payload["roomConfig"]["agents"][0]["agentName"] == "eidolon"
