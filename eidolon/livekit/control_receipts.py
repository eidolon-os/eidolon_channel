"""Shared correlation checks for authenticated LiveKit device receipts."""

import json
from eidolon_sdk.biz.contracts import CONTROL_TOPIC


def read_control_receipt(packet, *, device: str, max_bytes: int | None = None):
    if (
        getattr(packet, "topic", None) != CONTROL_TOPIC
        or getattr(getattr(packet, "participant", None), "identity", None) != device
    ):
        return None
    try:
        data = bytes(packet.data)
        if max_bytes is not None and len(data) > max_bytes:
            return None
        body = json.loads(data)
    except (AttributeError, TypeError, ValueError, UnicodeDecodeError):
        return None
    if (
        not isinstance(body, dict)
        or type(body.get("v")) is not int
        or body["v"] != 1
        or body.get("kind") not in ("ack", "result")
        or body.get("device_id") != device
        or not isinstance(body.get("ref"), str)
        or not isinstance(body.get("status"), str)
    ):
        return None
    return body
