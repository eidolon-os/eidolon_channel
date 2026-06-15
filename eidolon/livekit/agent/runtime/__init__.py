"""Phase 32.B runtime identity resolution for the channel-worker.

When a participant joins the LiveKit room, channel needs to know:

  - **Who** is talking (admin user_id)
  - **Which agent** they're configured to talk to (template_id, ...)
  - **Which memory palace** to dial (memory_mcp_url)

Phase 29 already built ``/api/resolve/{user,device}/{id}`` in admin that
composes all of that in one call. The pieces here:

  - :class:`AdminResolveClient` — thin HTTP wrapper over those two endpoints.
  - :func:`make_device_token_resolver` — given a LiveKit ``Room`` ref +
    a config, returns an async zero-arg callable the gRPC LLM invokes
    once per session to mint the bearer token used for agent gRPC.

Composition lives in ``runtime/resolver.py``; the other two files are
pure helpers tested in isolation.
"""

from eidolon.livekit.agent.runtime.admin_client import (
    AdminResolveClient,
    AdminResolveError,
    AdminResolveNotFound,
    AdminResolvePrecondition,
    AdminResolveUnreachable,
    AdminResolveUpstream,
    ResolvedContext,
)
from eidolon.livekit.agent.runtime.resolver import (
    DeviceTokenResolverError,
    make_device_token_resolver,
)

__all__ = [
    "AdminResolveClient",
    "AdminResolveError",
    "AdminResolveNotFound",
    "AdminResolvePrecondition",
    "AdminResolveUnreachable",
    "AdminResolveUpstream",
    "ResolvedContext",
    "DeviceTokenResolverError",
    "make_device_token_resolver",
]
