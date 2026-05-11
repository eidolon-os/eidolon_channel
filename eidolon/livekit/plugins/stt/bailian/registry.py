"""Plugin registry for Bailian FunASR STT.

Follows the same dict + register pattern used by other eidolon component registries
(e.g. pipeline/src/component/{tts,stt,agent}/registry.py).

The registry allows third parties to substitute a custom implementation of
BailianFunASRSTT. Auto-registration is intentionally omitted — call
``register_bailian_stt(BailianFunASRSTT)`` explicitly if needed.
"""

from __future__ import annotations

_PROVIDERS: dict[str, type] = {}


def register_bailian_stt(impl: type) -> None:
    """Register a BailianFunASRSTT implementation under the 'bailian-funasr' key."""
    _PROVIDERS["bailian-funasr"] = impl


def get_bailian_stt() -> type:
    """Return the registered BailianFunASRSTT implementation."""
    impl = _PROVIDERS.get("bailian-funasr")
    if impl is None:
        raise KeyError(
            "No BailianFunASRSTT implementation registered. "
            "Call register_bailian_stt(BailianFunASRSTT) first, "
            "or use create_bailian_stt() which creates the default class directly."
        )
    return impl


def create_bailian_stt(**kwargs) -> object:
    """Factory: create a BailianFunASRSTT instance.

    Uses the registered implementation if one has been registered;
    otherwise instantiates the default ``BailianFunASRSTT`` directly.
    """
    try:
        cls = get_bailian_stt()
    except KeyError:
        # No custom implementation registered — use the default class directly
        from .stt import BailianFunASRSTT

        cls = BailianFunASRSTT
    return cls(**kwargs)
