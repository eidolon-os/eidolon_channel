"""Executable composition root for ``eidolon-channel-provider``.

This is the one place that knows which adapters exist. Everything below it
works against the port, so adding a transport means adding a module here and a
name to the deployment's preference list — no other layer changes.
"""

from __future__ import annotations

import logging

from aiohttp import web
from eidolon_sdk.system import declared_management_networks

from eidolon.locked_environment import require_locked_environment

from .adapters.livekit import LiveKitChannelAdapter
from .config import load_provider_config
from .http import create_app
from .selection import AdapterRegistry
from .service import ChannelProviderService
from .session_traces import SessionTraceReader
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
        # Which links this Host's devices can actually reach it on. Read here,
        # at the one place that composes the process, rather than inside the
        # adapter: it is a declaration Ops delivered in the sealed Host profile
        # this unit already reads, and an adapter that reached for it would be
        # answering from ambient state its caller cannot see.
        [LiveKitChannelAdapter(config.livekit, declared_management_networks())],
        preference=config.adapter_preference,
    )
    service = ChannelProviderService(
        store=ChannelProviderStore(config.storage.path),
        registry=registry,
        agent_name=config.livekit.agent_name,
    )
    service.initialize()
    app = create_app(
        service=service,
        bearer_token=config.bearer_token,
        traces=SessionTraceReader(config.traces.root),
    )
    web.run_app(
        app,
        host=config.http.host,
        port=config.http.port,
        access_log_format='%a "%r" %s %Tf',
    )


if __name__ == "__main__":
    main()
