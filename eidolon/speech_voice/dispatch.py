"""A session's voice, resolved by channel-provider as it dispatches the job.

channel-provider knows the Companion when it opens a session, and it owns the
resolution (``resolve``). Doing it there, once, spares the job three calls it
made one after another in front of its pipeline -- identity, the Companion's
choice, resolution: ~90 ms of every start (11-01). The job still resolves the
voice itself when a dispatch carries none, so nothing depends on this.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from eidolon_sdk.biz.speech_voice import ResolvedSpeechVoice
from pydantic import ValidationError

#: Dispatch metadata key; travels Provider to agent, never to a participant.
DISPATCH_SPEECH_VOICE_FIELD = "speech_voice"


def dispatched_voice(metadata: Mapping[str, Any]) -> ResolvedSpeechVoice | None:
    """The voice the dispatch carries, or None to resolve it as before."""

    raw = metadata.get(DISPATCH_SPEECH_VOICE_FIELD)
    if not isinstance(raw, Mapping):
        return None
    try:
        return ResolvedSpeechVoice.model_validate(raw)
    except ValidationError:
        return None
