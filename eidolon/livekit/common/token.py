"""LiveKit token generation for client connections.

This module lives in the channel layer so the token-generation logic is decoupled
from the HTTP API. The FastAPI router simply calls generate_token() and translates
any ValueError into an HTTPException.
"""

from __future__ import annotations

from typing import Optional

from livekit import api

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

    token = (
        api.AccessToken(cfg.core.api_key, cfg.core.api_secret)
        .with_identity(participant_name)
        .with_name(participant_name)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        # Tell LiveKit server to dispatch the "eidolon" agent worker when
        # this token's holder joins the room. The worker (server.py) registers
        # with the same agent_name to receive the dispatch.
        .with_room_config(
            api.RoomConfiguration(
                agents=[api.RoomAgentDispatch(agent_name="eidolon")],
            )
        )
        .to_jwt()
    )

    return participant_name, token
