"""Executable composition root for ``eidolon-channel-provider``."""

from __future__ import annotations

import logging

from aiohttp import web

from .config import load_provider_config
from .http import create_app
from .livekit_backend import LiveKitChannelBackend
from .service import ChannelProviderService
from .store import ChannelProviderStore


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = load_provider_config()
    service = ChannelProviderService(
        store=ChannelProviderStore(config.storage.path),
        backend=LiveKitChannelBackend(config.livekit),
        livekit=config.livekit,
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
