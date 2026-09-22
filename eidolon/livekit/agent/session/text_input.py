"""Explicit text input has interruption authority, independent of ambient audio."""

from livekit.agents.voice import AgentSession
from livekit.agents.voice.room_io import TextInputEvent


async def accept_text_input(session: AgentSession, event: TextInputEvent) -> None:
    # The default RoomIO callback uses a non-forced interrupt. A manual-turn
    # session disables ambient interruptions, so that callback can discard a
    # typed message arriving during the welcome cue or another response.
    async with session._claim_user_turn():
        await session.interrupt(force=True)
        session.generate_reply(user_input=event.text)
