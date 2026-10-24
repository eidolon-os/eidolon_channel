import json

import httpx
import pytest
from eidolon_sdk.biz.system_data import SystemDataRuntimeClient

from eidolon.livekit.agent.speaker_verified.authority import SpeakerPolicyAuthority
from eidolon.livekit.agent.speaker_verified.orchestration import SpeakerDecision


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", [None, "off", "owner", "companion", "removed", "rerecorded", "revoked"]
)
async def test_current_authority_cannot_turn_off_policy_or_changed_template_into_voice_grant(
    change,
):
    requests = []
    sample = dict(
        model_id="test",
        weights_hash="a" * 64,
        preprocessing_version="test",
        vector=[1, 0],
        speech_ms=2000,
    )
    snapshot = dict(
        companion_id="companion",
        owner_id="owner",
        mode="allowlist",
        revision=7,
        templates=[dict(profile_id="profile", revision=3, sample=sample)],
    )
    if change == "off":
        snapshot["mode"] = "off"
    elif change == "owner":
        snapshot["owner_id"] = "different-owner"
    elif change == "companion":
        snapshot["companion_id"] = "different-companion"
    elif change == "removed":
        snapshot["templates"] = []
    elif change == "rerecorded":
        snapshot["templates"][0]["revision"] = 4

    async def respond(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=snapshot)
        assert json.loads(request.content) == {"revision": 7, "profile_id": "profile"}
        return httpx.Response(200, json={"allowed": change != "revoked"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = SystemDataRuntimeClient(http, "http://data.test", service_token="test")
        authority = SpeakerPolicyAuthority(client, owner_id="owner", companion_id="companion")
        allowed = await authority.authorize(SpeakerDecision("match", "profile", 3, "calibration"))
    assert allowed == (change is None)
    assert sum(r.method == "POST" for r in requests) == (1 if change in {None, "revoked"} else 0)
