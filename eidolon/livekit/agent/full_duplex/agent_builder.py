"""LiveKit Agent construction for the full-duplex runtime."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from livekit.agents.voice import Agent as lk_Agent

    from .pipeline import StreamingPipeline

logger = logging.getLogger("agent")


def welcome_on_enter_text(
    *,
    is_proactive: bool,
    welcome_message: str,
) -> str | None:
    """Welcome line to speak on session start, or None to stay silent."""

    if is_proactive:
        return None
    return welcome_message or None


def build_full_duplex_agent(pipeline: StreamingPipeline) -> lk_Agent:
    """Build the LiveKit Agent used by the full-duplex pipeline."""

    from livekit.agents import StopResponse
    from livekit.agents.voice import Agent

    class VoiceAgent(Agent):
        async def on_enter(self) -> None:
            # [lifecycle] welcome timestamp — anchors "welcome played" so Phase
            # 0 can measure the gap to a later idle room-delete and confirm
            # whether "回 JOIN after welcome" is the idle watchdog firing.
            room_name = getattr(getattr(pipeline, "_room", None), "name", None)
            welcome = pipeline._welcome_on_enter_text()
            if welcome is None:
                # Suppressed: proactive session (report is the opening, §4.3.1)
                # or no configured welcome (wait for the user to speak first).
                logger.info(
                    "[lifecycle] welcome on_enter room=%s suppressed (proactive=%s)",
                    room_name,
                    pipeline._is_proactive,
                )
                return
            logger.info(
                "[lifecycle] welcome on_enter room=%s welcome=%r",
                room_name,
                welcome[:30],
            )
            # Round 8 R8.9: use ``session.say(welcome)`` instead of
            # ``session.generate_reply()`` for the initial greeting.
            # generate_reply with no user message hands an empty context to the
            # LLM, which then frequently echoes the system prompt template back.
            self.session.say(welcome, allow_interruptions=True)

        async def on_user_turn_completed(
            self,
            turn_ctx: Any,
            new_message: Any,
        ) -> None:
            del turn_ctx
            allowed = (
                await pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
                    new_message=new_message
                )
            )
            if not allowed:
                raise StopResponse()

    # Round 8 R8.9 (re-fix): turn_handling config (including
    # false_interruption_timeout, preemptive_generation) lives on AGENTSESSION,
    # not Agent. Agent only carries per-agent turn_detection.
    return VoiceAgent(
        instructions=pipeline._instructions,
        stt=pipeline._factory.stt.stt,
        llm=pipeline._factory.llm.llm,
        tts=pipeline._factory.tts.tts,
        vad=pipeline._factory.vad.vad if pipeline._factory.vad else None,
        turn_detection=pipeline._turn_detection(),
    )
