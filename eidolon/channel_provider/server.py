"""Executable composition root for ``eidolon-channel-provider``.

This is the one place that knows which adapters exist. Everything below it
works against the port, so adding a transport means adding a module here and a
name to the deployment's preference list — no other layer changes.
"""

from __future__ import annotations

import logging

from aiohttp import web

from eidolon.locked_environment import require_locked_environment

from .adapters.livekit import LiveKitChannelAdapter
from .config import load_provider_config
from .http import create_app
from .selection import AdapterRegistry
from .service import ChannelProviderService
from .store import ChannelProviderStore

logger = logging.getLogger("eidolon.channel_provider.server")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # Before anything else: the Provider answering 200 while the worker cannot
    # run is the failure this repository has already paid for once. Refuse here
    # rather than hand out room contracts nothing can serve.
    require_locked_environment(logger=logger)
    config = load_provider_config()
    registry = AdapterRegistry(
        [LiveKitChannelAdapter(config.livekit)],
        preference=config.adapter_preference,
    )
    service = ChannelProviderService(
        store=ChannelProviderStore(config.storage.path),
        registry=registry,
        agent_name=config.livekit.agent_name,
    )
    service.initialize()
    app = create_app(service=service, bearer_token=config.bearer_token)
    web.run_app(
        app,
        host=config.http.host,
        port=config.http.port,
        access_log_format='%a "%r" %s %Tf',
    )


if __name__ == "__main__":
    main()
