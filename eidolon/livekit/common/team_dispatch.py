"""Trusted Provider dispatch for an explicit team scene, never device metadata."""
from pydantic import BaseModel, ConfigDict, model_validator
from eidolon_sdk.biz.control.coordination_stream import OpenScene
from eidolon_sdk.biz.presentation import SessionOutputPlan
from .presentation_endpoint import PresentationEndpoint


class TeamDispatch(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    opened: OpenScene
    input_plan: SessionOutputPlan
    endpoints: tuple[PresentationEndpoint, ...]

    @model_validator(mode='after')
    def validate_scene(self):
        if (not self.input_plan.inputs.microphone or self.input_plan.outputs.can_respond
                or self.input_plan.outputs.audio_cue):
            raise ValueError('team requires input-only source')
        selected = [m.output_device.device_instance_id for m in self.opened.selection.members]
        if [e.participant_identity for e in self.endpoints] != selected:
            raise ValueError('team endpoint order must match selected members')
        if len({e.room for e in self.endpoints}) != len(self.endpoints):
            raise ValueError('team endpoints require distinct standing rooms')
        if any(not e.plan.outputs.speech for e in self.endpoints):
            raise ValueError('team endpoints require speech output')
        return self
