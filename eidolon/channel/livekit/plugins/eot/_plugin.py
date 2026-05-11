# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plugin registration for Eidolon EOT."""

from __future__ import annotations

from .log import logger
from .version import __version__

__all__ = ["EidolonEOTPlugin", "register_plugin"]


def _register_plugin() -> None:
    """Register the plugin with LiveKit. Called lazily on first access."""
    from livekit.agents import Plugin

    class _EidolonEOTPlugin(Plugin):
        def __init__(self) -> None:
            super().__init__(
                title="eidolon-eot",
                version=__version__,
                package="eidolon.channel.livekit.plugins.eot",
                logger=logger,
            )

    Plugin.register_plugin(_EidolonEOTPlugin())


class EidolonEOTPlugin:
    """EOT plugin wrapper with lazy registration."""

    _registered: bool = False

    def __new__(cls) -> "EidolonEOTPlugin":
        if not cls._registered:
            _register_plugin()
            cls._registered = True
        return object.__new__(cls)


def register_plugin() -> None:
    """Explicitly register the plugin. Call this from the main thread."""
    _register_plugin()
