"""LiveKit Agent construction for the full-duplex runtime."""

from __future__ import annotations

import logging
import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from livekit.agents.voice import Agent as lk_Agent

    from .pipeline import StreamingPipeline

logger = logging.getLogger("agent")


def build_full_duplex_agent(pipeline: StreamingPipeline) -> lk_Agent:
    """Build the LiveKit Agent used by the full-duplex pipeline."""

    from livekit.agents import StopResponse
    from livekit.agents.voice import Agent

    class VoiceAgent(Agent):
        async def stt_node(self, audio: Any, model_settings: Any):
            """Admit provider revisions and retain evidence before SDK consumption."""

            events: Any = super().stt_node(audio, model_settings)
            if inspect.isawaitable(events):
                events = await events
            if events is None:
                return
            async for event in events:
                if pipeline._observe_stt_speech_event(event) is False:
                    continue
                yield event

        async def tts_node(self, text: Any, model_settings: Any):
            """Tap the public provider-neutral text stream before TTS synthesis."""

            stream_id = pipeline._begin_streamed_assistant_speech()

            async def observed_text():
                async for chunk in text:
                    pipeline._append_streamed_assistant_speech(stream_id, chunk)
                    yield chunk

            try:
                audio_frames: Any = super().tts_node(observed_text(), model_settings)
                if inspect.isawaitable(audio_frames):
                    audio_frames = await audio_frames
                if audio_frames is None:
                    return
                async for frame in audio_frames:
                    yield frame
            except BaseException:
                pipeline._abort_streamed_assistant_speech(stream_id)
                raise

        async def on_enter(self) -> None:
            # [lifecycle] welcome timestamp — anchors "welcome played" so Phase
            # 0 can measure the gap to a later idle teardown and confirm
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
            pipeline._queue_fixed_assistant_speech(welcome, source="welcome")
            # Round 8 R8.9: use ``session.say(welcome)`` instead of
            # ``session.generate_reply()`` for the initial greeting.
            # generate_reply with no user message hands an empty context to the
            # LLM, which then frequently echoes the system prompt template back.
            self.session.say(welcome)

        async def on_user_turn_completed(
            self,
            turn_ctx: Any,
            new_message: Any,
        ) -> None:
            allowed = await pipeline._ensure_turn_completion().voiceprint_allows_completed_turn(
                turn_ctx=turn_ctx,
                new_message=new_message,
            )
            if not allowed:
                raise StopResponse()

    # Session-wide interruption/preemptive settings remain on AgentSession.
    # LiveKit 1.7 prefers the structured ``turn_handling`` option over the
    # deprecated individual ``turn_detection`` argument; the per-agent turn
    # detector still overrides the session default through that structure.
    from eidolon.livekit.plugins.eot.models.base import EidolonEOTModel
    from ..session.eot_model import InterruptAwareTurnDetector

    detector = pipeline._turn_detection()
    turn_handling: dict[str, Any] = {"turn_detection": detector}
    eot_model = detector.eot_model if isinstance(detector, InterruptAwareTurnDetector) else detector
    if isinstance(eot_model, EidolonEOTModel):
        # Bind the complete-turn fast path through the public SDK contract.
        # Incomplete turns retain the session's maximum (SDK default: 3 s);
        # the model helper's 2 s bound can split a slowly completed question.
        turn_handling["endpointing"] = {
            "min_delay": eot_model.get_dynamic_silence_threshold(
                "", p_complete=1.0, is_final=False,
            ),
        }
    return VoiceAgent(
        instructions=pipeline._instructions,
        stt=pipeline._factory.stt.stt,
        llm=pipeline._factory.llm.llm,
        tts=pipeline._factory.tts.tts,
        vad=pipeline._factory.vad.vad if pipeline._factory.vad else None,
        turn_handling=turn_handling,
    )
