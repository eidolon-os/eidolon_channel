"""Public SDK boundary for explicitly admitted conversation turns.

Untrusted STT results go to the candidate owner, never the SDK transcript buffer.
This avoids resetting the shared streaming recognizer after every rejected turn.
The orchestrator alone releases approved text through generate_reply().
"""

from collections.abc import Callable
from typing import Any

from livekit.agents import StopResponse, stt

from ..session.policy_bound_agent import PolicyBoundAgent


def speaker_verified_turn_handling() -> dict[str, Any]:
    """Install on AgentSession: audio suppression is a session option in SDK 1.7."""
    return {
        "turn_detection": "manual",
        "interruption": {
            "enabled": False,
            "discard_audio_if_uninterruptible": False,
            "resume_false_interruption": False,
        },
        "preemptive_generation": {"enabled": False, "preemptive_tts": False},
    }


class SpeakerVerifiedAgent(PolicyBoundAgent):
    def __init__(self, *, observe_speech: Callable[[stt.SpeechEvent], None], **kwargs: Any):
        super().__init__(**kwargs, turn_handling={"turn_detection": "manual"})
        self._observe_speech = observe_speech

    async def stt_node(self, audio, model_settings):
        async for event in super().stt_node(audio, model_settings):
            # Consumption must not await inference: renewed speech/revisions
            # invalidate pending evidence while the same input keeps flowing.
            self._observe_speech(event)
            if event.type == stt.SpeechEventType.RECOGNITION_USAGE:
                yield event

    async def on_user_turn_completed(self, turn_ctx, new_message):
        # Automatic SDK completion is never an authorization boundary here.
        raise StopResponse()
