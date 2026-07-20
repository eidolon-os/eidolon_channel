"""Digital-human (talking-head) video avatar for the LiveKit channel.

Bridges an external audio-driven talking-head service into LiveKit's standard
avatar-worker mechanism: the channel forwards TTS audio to a worker participant,
which POSTs each segment to the service, decodes the returned video, and
publishes a synchronized audio+video track that display-capable clients (web/app)
subscribe to. See ``README.md`` and the plan for the measured service contract.

Public surface:
  * :class:`AvatarWorker` — the worker participant (channel spawns it per session).
  * :func:`avatar_identity_for` — deterministic worker identity for a room.
  * :class:`EidolonDHVideoGenerator` — the ``VideoGenerator`` implementation.
  * :class:`DigitalHumanServiceClient` — the async HTTP client for the service.
  * :func:`resolve_session_face_image` — the session companion's ``cond_image`` bytes.
"""

from __future__ import annotations

from .face_source import resolve_session_face_image
from .service_client import DigitalHumanServiceClient, StreamVideoParams, pcm16_to_wav
from .video_generator import EidolonDHVideoGenerator
from .worker import AvatarWorker, avatar_identity_for

__all__ = [
    "AvatarWorker",
    "avatar_identity_for",
    "EidolonDHVideoGenerator",
    "DigitalHumanServiceClient",
    "StreamVideoParams",
    "pcm16_to_wav",
    "resolve_session_face_image",
]
