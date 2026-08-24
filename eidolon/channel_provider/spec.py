"""What a device needs from a channel, derived from what the device declared.

The Hub forwards the device manifest verbatim and treats the resulting binding
as opaque: it deliberately does not know how a channel is realised. This module
is the one place that reads the manifest and turns it into a transport-neutral
statement of need, so that no adapter has to parse a manifest and no adapter's
vocabulary leaks back into the service.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .contracts import ProvisionDevice

# Turn-taking is a device constraint, not a deployment preference: whether a
# device may listen while it speaks depends on its echo cancellation, not on
# what the operator would like. Devices declare it; the provider never guesses.
INTERACTION_MODES = frozenset({"full_duplex", "half_duplex", "ptt"})
DEFAULT_INTERACTION_MODE = "half_duplex"

_MANIFEST_INTERACTION_MODE = "interaction_mode"


class MediaFlow(Enum):
    """Which way a media kind travels, from the device's point of view."""

    NONE = "none"
    PUBLISH = "publish"
    SUBSCRIBE = "subscribe"
    DUPLEX = "duplex"

    @property
    def publishes(self) -> bool:
        return self in (MediaFlow.PUBLISH, MediaFlow.DUPLEX)

    @property
    def subscribes(self) -> bool:
        return self in (MediaFlow.SUBSCRIBE, MediaFlow.DUPLEX)


_DIRECTION_TO_FLOW = {
    "publish": MediaFlow.PUBLISH,
    "subscribe": MediaFlow.SUBSCRIBE,
    "bidirectional": MediaFlow.DUPLEX,
}


@dataclass(frozen=True, slots=True)
class ServingSpec:
    """The conversational agent this channel must be served by, if any.

    A device that cannot publish audio has nothing for a voice agent to listen
    to, so it gets no serving spec and no agent is ever dispatched for it.
    """

    agent_name: str
    interaction_mode: str


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    """A transport-neutral statement of what one device's channel must carry."""

    device_id: str
    owner_id: str
    device_kind: str
    audio: MediaFlow
    video: MediaFlow
    serving: ServingSpec | None

    @property
    def needs_media(self) -> bool:
        return self.audio is not MediaFlow.NONE or self.video is not MediaFlow.NONE


def _media_flow(manifest: dict[str, Any], kind: str) -> MediaFlow:
    """Collapse every declaration of one media kind into a single flow.

    A manifest may declare a kind more than once (say, publish and subscribe as
    separate entries); the channel has to satisfy all of them at once, so the
    declarations union rather than override.
    """
    publishes = False
    subscribes = False
    for entry in manifest.get("media", ()):
        if not isinstance(entry, dict) or entry.get("kind") != kind:
            continue
        flow = _DIRECTION_TO_FLOW.get(entry.get("direction", ""))
        if flow is None:
            continue
        publishes = publishes or flow.publishes
        subscribes = subscribes or flow.subscribes
    if publishes and subscribes:
        return MediaFlow.DUPLEX
    if publishes:
        return MediaFlow.PUBLISH
    if subscribes:
        return MediaFlow.SUBSCRIBE
    return MediaFlow.NONE


def _declared_interaction_mode(manifest: dict[str, Any]) -> str:
    """Read the device's declared turn-taking mode.

    The manifest schema carries immutable device attributes as properties whose
    schema pins a single value, so an `interaction_mode` property with a `const`
    (or a single-entry `enum`) is the device stating a fact about itself. A
    device that stays silent gets the conservative mode: half duplex never
    assumes echo cancellation that may not exist.
    """
    for prop in manifest.get("properties", ()):
        if not isinstance(prop, dict) or prop.get("name") != _MANIFEST_INTERACTION_MODE:
            continue
        schema = prop.get("schema")
        if not isinstance(schema, dict):
            continue
        declared = schema.get("const")
        if declared is None:
            choices = schema.get("enum")
            if isinstance(choices, list) and len(choices) == 1:
                declared = choices[0]
        if isinstance(declared, str) and declared in INTERACTION_MODES:
            return declared
    return DEFAULT_INTERACTION_MODE


def derive_spec(
    device: ProvisionDevice, *, device_instance_id: str, agent_name: str
) -> ChannelSpec:
    """Turn one device's declaration into what its channel must provide."""
    manifest = device.manifest
    audio = _media_flow(manifest, "audio")
    video = _media_flow(manifest, "video")
    serving = (
        ServingSpec(
            agent_name=agent_name,
            interaction_mode=_declared_interaction_mode(manifest),
        )
        if audio.publishes
        else None
    )
    return ChannelSpec(
        device_id=device_instance_id,
        owner_id=str(device.owner_id),
        device_kind=device.device_kind,
        audio=audio,
        video=video,
        serving=serving,
    )
