"""LiveKit token generation for client connections.

This module lives in the channel layer so the token-generation logic is decoupled
from the HTTP API. The FastAPI router simply calls generate_token() and translates
any ValueError into an HTTPException.
"""

from __future__ import annotations

from typing import Optional

from eidolon_sdk.integrations.livekit import build_livekit_token

from eidolon.livekit.common.config import AgentConfig, load_agent_config

# Cached config: token generation can fire on every API request, and re-reading
# the .env file each time is wasteful (and triggers AgentConfig validation
# logs). The cache is process-local; restart the agent to pick up .env changes.
_cached_cfg: Optional[AgentConfig] = None


def _get_config() -> AgentConfig:
    global _cached_cfg
    if _cached_cfg is None:
        _cached_cfg = load_agent_config()
    return _cached_cfg


def generate_token(room_name: str, participant_name: str) -> tuple[str, str]:
    """Generate a LiveKit access token for a participant joining a room.

    Args:
        room_name: LiveKit room name.
        participant_name: Identity/name of the participant.

    Returns:
        A tuple of (identity, access_token).

    Raises:
        ValueError: If LIVEKIT_API_KEY or LIVEKIT_API_SECRET is not configured.
    """
    cfg = _get_config()

    if not cfg.core.api_key or not cfg.core.api_secret:
        raise ValueError("LIVEKIT_API_KEY or LIVEKIT_API_SECRET not configured")

    # Tell LiveKit server to dispatch the "eidolon" agent worker when this
    # token's holder joins the room. The worker (server.py) registers with the
    # same agent_name to receive the dispatch.
    token = build_livekit_token(
        api_key=cfg.core.api_key,
        api_secret=cfg.core.api_secret,
        room_name=room_name,
        identity=participant_name,
        name=participant_name,
        dispatch_agent=True,
        agent_name="eidolon",
    )

    return participant_name, token
