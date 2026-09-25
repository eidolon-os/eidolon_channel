"""Internal authority commands; business Owner authentication precedes this API."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field
from eidolon_sdk.biz.control.shared_session import SharedSessionSelection
from eidolon_sdk.device_foundation.v1 import DeviceRef


class DeviceSpecification(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    device_ref: DeviceRef
    device: dict


class OpenSharedSession(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    owner_id: Annotated[str, Field(min_length=1, max_length=255)]
    selection: SharedSessionSelection
    specifications: Annotated[tuple[DeviceSpecification, ...], Field(min_length=2, max_length=16)]


class CloseSharedSession(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    owner_id: Annotated[str, Field(min_length=1, max_length=255)]
    session_id: Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")]
