"""A Provider-authorized presentation endpoint in its original LiveKit room."""

from datetime import timedelta
from typing import Annotated, Self

from livekit import api, rtc
from pydantic import BaseModel, ConfigDict, Field, model_validator
from eidolon_sdk.biz.presentation import SessionOutputPlan


class PresentationEndpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    room: Annotated[str, Field(strict=True, min_length=1, max_length=255)]
    participant_identity: Annotated[str, Field(strict=True, min_length=1, max_length=255)]
    plan: SessionOutputPlan

    @model_validator(mode="after")
    def output_only(self) -> Self:
        if self.plan.inputs.microphone:
            raise ValueError("PRESENTATION_ENDPOINT_CANNOT_CAPTURE")
        if self.plan.outputs.expression or self.plan.outputs.motion or self.plan.outputs.audio_cue:
            raise ValueError("REMOTE_ENDPOINT_CURRENTLY_SUPPORTS_SPEECH_AND_TEXT")
        if not (self.plan.outputs.speech or self.plan.outputs.dialogue_text):
            raise ValueError("PRESENTATION_ENDPOINT_HAS_NO_OUTPUT")
        return self

    def join_token(self, *, api_key: str, api_secret: str, worker_identity: str) -> str:
        # This grant has no subscription permission. B can never become a
        # second input to the same inference, even if it publishes audio.
        return (api.AccessToken(api_key, api_secret)
            .with_identity(worker_identity).with_kind("agent")
            .with_ttl(timedelta(minutes=5))
            .with_grants(api.VideoGrants(room_join=True, room=self.room,
                can_publish=True, can_publish_data=True, can_subscribe=False))
            .to_jwt())

    async def connect(self, *, core, worker_identity: str) -> rtc.Room:
        room = rtc.Room()
        try:
            await room.connect(core.livekit_url, self.join_token(api_key=core.api_key,
                api_secret=core.api_secret, worker_identity=worker_identity))
        except BaseException:
            await room.disconnect()
            raise
        return room
