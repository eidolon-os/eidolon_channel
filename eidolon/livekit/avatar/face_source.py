"""Resolve the selected Companion face through the System Data authority."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("agent.avatar.face")


async def resolve_session_face_image(
    room: Any,
    *,
    context_resolver: Any | None,
    runtime_client: Any | None,
) -> bytes | None:
    """Return the session Companion JPEG, or ``None`` for the default avatar.

    Runtime selection uses the same Kernel Mount + System Data boundary as the
    Agent token path. Face absence or authority failure remains best-effort and
    never interrupts an audio session.
    """

    if context_resolver is None or runtime_client is None:
        return None
    try:
        context = await context_resolver(room)
        companion_id = context.companion_id
        if companion_id is None:
            return None
        data = await runtime_client.get_companion_face(companion_id)
        if data is None:
            logger.info(
                "[avatar.face] companion=%s has no configured face; using default avatar",
                companion_id,
            )
            return None
        logger.info(
            "[avatar.face] resolved cond_image companion=%s bytes=%d",
            companion_id,
            len(data),
        )
        return data
    except Exception:  # noqa: BLE001 — face resolution must never break a session
        logger.warning(
            "[avatar.face] face resolution failed; using default avatar",
            exc_info=True,
        )
        return None


__all__ = ["resolve_session_face_image"]
