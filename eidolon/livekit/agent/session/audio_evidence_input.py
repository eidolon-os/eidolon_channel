"""Invalidate optional evidence at the selected input's public SDK lifecycle.

This adapter forwards the same frames and attach/detach calls. It neither reads
another track nor captures PCM. This capture stays invalid after detachment;
EvidenceRoomInput is the sole owner of room events and fresh input recovery.
Not installed by the production session factory.
"""

from collections.abc import Callable

from livekit import rtc
from livekit.agents.voice.io import AudioInput

from eidolon.livekit.common.audio_evidence import AudioEvidenceCapture


class EvidenceAudioInput(AudioInput):
    def __init__(
        self,
        *,
        source: AudioInput,
        capture: AudioEvidenceCapture,
        on_invalidated: Callable[[str], None] | None = None,
        on_resumed: Callable[[], None] | None = None,
    ):
        super().__init__(label="evidence-lifecycle", source=source)
        self._capture = capture
        self._closed = False
        self._on_invalidated = on_invalidated
        self._on_resumed = on_resumed

    def on_attached(self) -> None:
        super().on_attached()
        if self._closed and self._on_resumed is not None:
            self._on_resumed()

    async def __anext__(self) -> rtc.AudioFrame:
        try:
            return await super().__anext__()
        except BaseException:
            self.close("input_ended")
            raise

    def on_detached(self) -> None:
        self.close("input_detached")
        super().on_detached()

    def close(self, reason: str = "input_closed") -> None:
        """Invalidate evidence; the owning RoomIO controls audio IO cleanup."""
        if self._closed:
            return
        self._closed = True
        self._capture.invalidate(reason)
        if self._on_invalidated is not None:
            self._on_invalidated(reason)
