"""Apply output selection at the public LiveKit Agent node boundary."""

from typing import Any
from livekit.agents.voice import Agent
from eidolon_sdk.biz.presentation import OutputSelection


class PolicyBoundAgent(Agent):
    def __init__(self, *, outputs: OutputSelection, **kwargs: Any):
        super().__init__(**kwargs)
        self._selected_outputs = outputs

    async def transcription_node(self, text: Any, model_settings: Any):
        # RoomOptions also disables both user/assistant transcript publication.
        # Suppressing this node prevents invisible text from triggering the
        # framework's text-only "speaking" state in strict silent sessions.
        if not self._selected_outputs.dialogue_text:
            return None
        return Agent.default.transcription_node(self, text, model_settings)
