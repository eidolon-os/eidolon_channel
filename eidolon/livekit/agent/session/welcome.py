"""Play welcome text or cached audio through the session's output lifecycle."""

from collections.abc import AsyncIterator, Callable
from typing import Any

from livekit import rtc

from eidolon.livekit.common.welcome import WelcomeAudio, WelcomeMessage, welcome_allowed
from eidolon_sdk.biz.presentation import OutputSelection


async def _audio_frames(pcm: bytes, sample_rate: int) -> AsyncIterator[rtc.AudioFrame]:
    # Immutable cached PCM, fresh frames and cursor for every concurrent session.
    frame_bytes = (sample_rate // 50) * 2
    for offset in range(0, len(pcm), frame_bytes):
        data = bytearray(pcm[offset:offset + frame_bytes])
        yield rtc.AudioFrame(data, sample_rate, 1, len(data) // 2)


def play_welcome(
    session: Any,
    welcome: WelcomeMessage,
    *,
    outputs: OutputSelection,
    pcm: bytes | None,
    sample_rate: int,
    queue_text: Callable[..., None] | None = None,
) -> Any:
    """Keep the session's interruption policy and speech completion events."""
    if not welcome_allowed(welcome, outputs):
        return None
    if isinstance(welcome, WelcomeAudio):
        if pcm is None:
            raise ValueError("welcome audio must be prepared before session entry")
        # SDK 1.7 requires the positional text argument even for audio-only say.
        return session.say("", audio=_audio_frames(pcm, sample_rate), add_to_chat_ctx=False)
    if queue_text is not None:
        queue_text(welcome, source="welcome")
    return session.say(welcome)
