"""Bind existing AgentSession media to explicit input and presentation rooms.

The caller must authorize and connect the rooms before calling this adapter.
It never selects a Companion, mints grants, or starts another generation.
"""

import asyncio
from dataclasses import replace
from typing import Any

from livekit.agents.voice.room_io import RoomIO, RoomOptions

from .presentation_input_gate import PresentationInputGate


async def start_room_session(
    *, session: Any, agent: Any, input_room: Any, options: RoomOptions,
    output_room: Any | None = None, output_participant_identity: str | None = None,
) -> RoomIO | None:
    """Return the separately owned output IO, or None for ordinary single-room IO.

    The caller closes returned IO after closing the AgentSession. On startup
    failure this function closes it itself. No output fallback is allowed: if
    the selected speaker cannot start, the source device must not speak instead.
    """
    if output_room is None or output_room is input_room:
        await session.start(agent=agent, room=input_room, room_options=options)
        return None
    if not output_participant_identity:
        raise ValueError("OUTPUT_PARTICIPANT_REQUIRED")
    if not output_room.isconnected():
        raise ValueError("OUTPUT_ROOM_NOT_CONNECTED")
    if output_participant_identity not in {
        p.identity for p in output_room.remote_participants.values()
    }:
        raise ValueError("OUTPUT_PARTICIPANT_NOT_PRESENT")
    # Preinstalled avatar/custom sinks cannot silently bypass the chosen room.
    if session.output.audio is not None or session.output.transcription is not None:
        raise ValueError("OUTPUT_ALREADY_ATTACHED")

    output_io = RoomIO(session, output_room, options=RoomOptions(
        participant_identity=output_participant_identity,
        audio_input=False, video_input=False, text_input=False,
        audio_output=options.audio_output, text_output=options.text_output,
        close_on_disconnect=True,
    ))
    input_gate = PresentationInputGate(session)
    try:
        await output_io.start()
        async with asyncio.timeout(10.0):
            await output_io.wait_for_ready()
        await session.start(agent=agent, room=input_room, room_options=replace(
            options, audio_output=False, text_output=False,
        ))
    except BaseException:
        input_gate.close()
        await output_io.aclose()
        raise
    return output_io
