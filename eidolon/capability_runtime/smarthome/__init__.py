"""Smart home v1 in the Capability Runtime: execute through Providers, keep panels in sync.

The contract (vocabulary, registry, execute and panel wire) is
``eidolon_sdk.biz.smarthome``; nothing here redefines it.
"""

from .ports import PanelSink, RegistrySource, SmartHomeProvider
from .runtime import IdempotencyConflict, SmartHomeRuntime
from .http_registry import HttpRegistrySource
from .virtual import VirtualProvider, apply_command
from .wire import panel_command

__all__ = [
    "IdempotencyConflict",
    "HttpRegistrySource",
    "PanelSink",
    "RegistrySource",
    "SmartHomeProvider",
    "SmartHomeRuntime",
    "VirtualProvider",
    "apply_command",
    "panel_command",
]
