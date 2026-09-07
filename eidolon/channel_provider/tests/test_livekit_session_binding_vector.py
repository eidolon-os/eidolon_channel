"""This Provider against the golden the Bodies parse.

`opaque_binding` is opaque to the Authority and must stay that way — relaying a
blob is the whole of what an Authority should know about a room. It is not
opaque to the two ends that use it. This module writes the document; the
firmware and the phone each parse it, each from their own hand-written reader,
and `schema_version: 2` is what one undetected disagreement already cost. The
vector is the third thing all of them can be held to.

Read from the installed SDK, never restated here. What this replaces is
`test_livekit_adapter.py::test_binding_describes_one_session`, which held the
member set in a set literal one screen from the producer — a copy of the
contract, kept honest by whoever remembered to edit both.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import eidolon_sdk

from eidolon.channel_provider.adapters.livekit import BINDING_FORMAT
from eidolon.channel_provider.contracts import ProvisionRequest
from eidolon.channel_provider.selection import AdapterRegistry
from eidolon.channel_provider.service import ChannelProviderService
from eidolon.channel_provider.store import ChannelProviderStore

from .helpers import FakeAdapter, encoded, provision_payload


def golden() -> dict:
    path = (
        Path(eidolon_sdk.__file__).resolve().parents[1]
        / "contracts/device_foundation/v1/golden/livekit-session-binding.json"
    )
    assert path.exists(), f"the canonical session binding vector is not at {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _leaf_paths(value: object, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        found: list[str] = []
        for key, item in value.items():
            found.extend(_leaf_paths(item, f"{prefix}.{key}" if prefix else str(key)))
        return sorted(found)
    return [prefix]


async def test_the_provider_writes_the_member_set_the_vector_pins() -> None:
    """Not "it parses" — the exact positions, because that is what drifted.

    A member this Provider adds is a member no shipped Body reads; a member it
    drops is a Body that cannot join. Neither shows up as an error here: the
    document still parses, the channel is still provisioned, and the device
    connects to a room where something is missing.
    """
    from .test_livekit_adapter import _adapter, _spec

    vector = golden()
    adapter, _client = _adapter()

    grant = await adapter.open(_spec(), issued_at_ms=1_000)

    binding = json.loads(grant.payload)
    assert _leaf_paths(binding) == sorted(vector["member_paths"])
    assert binding["schema_version"] == vector["binding"]["schema_version"]
    assert grant.binding_format == vector["binding_format"]
    # The room the token was minted for is the one the binding names, which is
    # the adapter's own business and stays here rather than in the vector.
    assert binding["session"]["room_name"] == grant.handle["room"]


async def test_the_binding_format_is_the_string_the_vector_names() -> None:
    """The constant, not just whatever a grant happened to carry.

    It was a literal in this module and a literal in the firmware, with the
    version number inside a media type string that nothing compared. Bumping it
    on one side is exactly the change that must not be silent.
    """
    vector = golden()
    assert BINDING_FORMAT == vector["binding_format"]


async def test_the_wire_carries_the_padded_standard_alphabet(tmp_path) -> None:
    """The encoding the service applies, held to the vector's own two strings.

    Every other base64 in the Device Foundation contract — keys, signatures,
    proofs — is URL-safe and unpadded. Reaching for the house idiom here emits
    something no Body can decode, and nothing about that mistake looks wrong
    from this side: the Provider succeeds, the Authority relays, and the device
    fails to parse a blob it was handed correctly-shaped.
    """
    vector = golden()
    canonical = vector["canonical_utf8"].encode("utf-8")
    backend = FakeAdapter(
        name="livekit",
        binding_format=vector["binding_format"],
        payload=canonical,
    )
    store = ChannelProviderStore(tmp_path / "provider.sqlite3")
    service = ChannelProviderService(
        store=store,
        registry=AdapterRegistry([backend], preference=("livekit",)),
        agent_name="eidolon",
        now_ms=lambda: 1_700_000_000_000,
    )
    service.initialize()

    response = await service.provision(ProvisionRequest.parse(encoded(provision_payload())))

    channel = json.loads(response)["channels"][0]
    assert channel["binding_format"] == vector["binding_format"]
    assert channel["opaque_binding"] == vector["opaque_binding"]
    assert channel["opaque_binding"] != vector["not_the_opaque_binding"]["base64url_no_padding"]
    assert base64.b64decode(channel["opaque_binding"], validate=True) == canonical
