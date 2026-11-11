"""LiveKit Agent construction for the full-duplex runtime."""

from __future__ import annotations

import asyncio
import logging
import inspect
from typing import TYPE_CHECKING, Any

from ..session.welcome import play_welcome

if TYPE_CHECKING:
    from ..session.policy_bound_agent import PolicyBoundAgent as lk_Agent

    from .pipeline import StreamingPipeline

logger = logging.getLogger("agent")


def _incomplete_turn_max_delay(pipeline: Any) -> float:
    """Seconds an unlikely-complete turn waits after the user's last speech."""
    from eidolon.livekit.common.config.schema import EotPolicyConfig

    policy = getattr(pipeline, "_turn_policy", None)
    eot = policy.eot if policy is not None else EotPolicyConfig()
    return eot.max_endpointing_delay_ms / 1000.0


def build_full_duplex_agent(
    pipeline: StreamingPipeline, *, ready: asyncio.Event | None = None,
) -> lk_Agent:
    """Build the LiveKit Agent used by the full-duplex pipeline."""

    from livekit.agents import StopResponse
    from ..session.policy_bound_agent import PolicyBoundAgent

    class VoiceAgent(PolicyBoundAgent):
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
            # Media starts alongside prewarm; do not speak before the device
            # receives the session confirmation that authorizes its output.
            if ready is not None:
                await ready.wait()
            # [lifecycle] welcome timestamp — anchors "welcome played" so Phase
            # 0 can measure the gap to a later idle teardown and confirm
            # whether "回 JOIN after welcome" is the idle watchdog firing.
            room_name = getattr(getattr(pipeline, "_room", None), "name", None)
            welcome = pipeline._welcome_on_enter()
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
                welcome,
            )
            play_welcome(
                self.session, welcome,
                pcm=pipeline._welcome_pcm,
                outputs=pipeline._factory.outputs,
                sample_rate=pipeline._audio_sample_rate,
                queue_text=pipeline._queue_fixed_assistant_speech,
            )

        async def on_exit(self) -> None:
            # SDK shutdown drains endpointing after on_exit. Wake our policy
            # barrier now, including when callers close AgentSession directly.
            pipeline._interruption_orchestrator.close()
            # Transcript settlement is also a speech task drained by the SDK.
            # Settle its product owner before drain, not only in the later
            # session-close notification. Reuse the existing terminal effects.
            pipeline._ensure_lifecycle()._reject_open_user_turn_on_close(
                error=getattr(pipeline, "_close_error", None),
            )
            await pipeline._semantic_interrupts.aclose()

        async def on_user_turn_completed(
            self,
            turn_ctx: Any,
            new_message: Any,
        ) -> None:
            allowed = await pipeline._ensure_turn_completion().allows_completed_turn(
                turn_ctx=turn_ctx,
                new_message=new_message,
            )
            if not allowed:
                raise StopResponse()
            observer = getattr(pipeline, "_speaker_observation", None)
            finish = getattr(observer, "finish", None)
            timeline = pipeline._timeline
            if callable(finish) and timeline is not None:
                from ..session.messages import message_text
                await finish(pipeline._user_turns, turn_id=timeline.turn_id,
                    text=message_text(new_message), llm=pipeline._factory.llm.llm, timeline=timeline)
            prepare = getattr(pipeline, "_prepare_memory_turn", None)
            if callable(prepare):
                from ..session.messages import message_text
                await prepare(text=message_text(new_message))


    # Session-wide interruption/preemptive settings remain on AgentSession.
    # LiveKit 1.7 prefers the structured ``turn_handling`` option over the
    # deprecated individual ``turn_detection`` argument; the per-agent turn
    # detector still overrides the session default through that structure.
    from eidolon.livekit.plugins.eot.models.base import EidolonEOTModel
    from ..session.eot_model import InterruptAwareTurnDetector
    from ..session.streaming_eot import NativeAudioTurnDetector

    detector = pipeline._turn_detection()
    turn_handling: dict[str, Any] = {"turn_detection": detector}
    incomplete_max = _incomplete_turn_max_delay(pipeline)
    if isinstance(detector, NativeAudioTurnDetector):
        # Match the deployed FireRed endpoint delays. Streaming SDK defaults
        # differ, which would otherwise confound a provider comparison.
        turn_handling["endpointing"] = {"min_delay": 0.25, "max_delay": incomplete_max}
    eot_model = detector.eot_model if isinstance(detector, InterruptAwareTurnDetector) else detector
    if isinstance(eot_model, EidolonEOTModel):
        # Bind both bounds through the public SDK contract: a turn EOT scores
        # complete waits the policy's fast bound; one it scores unlikely
        # complete waits ``turn_policy.eot.max_endpointing_delay_ms`` instead
        # of the SDK's 3 s, which also fell on complete questions EOT misread.
        turn_handling["endpointing"] = {
            "min_delay": eot_model.get_dynamic_silence_threshold(
                "", p_complete=1.0, is_final=False,
            ),
            "max_delay": incomplete_max,
        }
    return VoiceAgent(
        outputs=pipeline._factory.outputs,
        instructions=pipeline._instructions,
        stt=pipeline._factory.stt.stt,
        llm=pipeline._factory.llm.llm,
        tts=pipeline._factory.tts.tts if pipeline._factory.tts is not None else None,
        vad=pipeline._factory.vad.vad if pipeline._factory.vad else None,
        turn_handling=turn_handling,
    )
