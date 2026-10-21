"""Invalidate optional evidence at the selected input's public SDK lifecycle.

This adapter forwards the same frames and attach/detach calls. It neither reads
another track nor captures PCM. Room events can prove discontinuity, not where
queued frames change source; after a discontinuity this capture stays invalid.
The owner must establish a fresh input boundary before creating another capture.
Not installed by the production session factory.
"""

from collections.abc import Callable
from typing import Any

from livekit import rtc
from livekit.agents.voice.io import AudioInput

from eidolon.livekit.common.audio_evidence import AudioEvidenceCapture


class EvidenceAudioInput(AudioInput):
    def __init__(self, *, source: AudioInput, capture: AudioEvidenceCapture):
        super().__init__(label="evidence-lifecycle", source=source)
        self._capture = capture
        self._room: rtc.Room | None = None
        self._listeners: list[tuple[str, Callable[..., None]]] = []
        self._closed = False

    async def __anext__(self) -> rtc.AudioFrame:
        try:
            return await super().__anext__()
        except BaseException:
            self.close("input_ended")
            raise

    def on_detached(self) -> None:
        self.close("input_detached")
        super().on_detached()

    def watch_room(self, room: rtc.Room, *, participant_identity: str) -> None:
        """Observe only the participant already selected by the session owner.

        Install before exposing this input to the session. A late initial
        subscription also invalidates: it is not evidence of an empty queue.
        This does not choose a track or alter RoomIO's participant selection.
        """
        if not participant_identity or self._room is not None or self._closed:
            raise ValueError("one live room and explicit input participant required")
        self._room = room

        def track_changed(*args: Any) -> None:
            publication, participant = args[-2:]
            if (
                participant.identity == participant_identity
                and publication.source == rtc.TrackSource.SOURCE_MICROPHONE
            ):
                self.close("input_track_changed")

        def participant_left(participant: Any) -> None:
            if participant.identity == participant_identity:
                self.close("input_participant_left")

        def connection_changed(*_: Any) -> None:
            self.close("input_connection_changed")

        for event in (
            "track_subscribed",
            "track_unsubscribed",
            "track_unpublished",
            "track_muted",
            "track_unmuted",
        ):
            self._listeners.append((event, track_changed))
        self._listeners.extend(
            [
                ("participant_disconnected", participant_left),
                ("reconnecting", connection_changed),
                ("disconnected", connection_changed),
            ]
        )
        for event, callback in self._listeners:
            room.on(event, callback)

    def close(self, reason: str = "input_closed") -> None:
        """Release evidence/listeners only; SDK retains ownership of audio IO."""
        if self._closed:
            return
        self._closed = True
        self._capture.invalidate(reason)
        if self._room is not None:
            for event, callback in self._listeners:
                self._room.off(event, callback)
        self._listeners.clear()
        self._room = None
