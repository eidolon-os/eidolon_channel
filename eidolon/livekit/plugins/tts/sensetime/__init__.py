"""SenseTime SenseAudio TTS plugin for LiveKit Agents."""
from eidolon.livekit.plugins.tts.sensetime.config import SenseTimeTTSConfig
from eidolon.livekit.plugins.tts.sensetime.tts import SenseTimeTTS
from eidolon.livekit.plugins.tts.sensetime.tts_client import (
    SenseTimeTTSClient,
    SenseTimeTTSError,
    TTSConnection,
)
from eidolon.livekit.plugins.tts.sensetime.protocol import (
    EVENT_CONNECTED_SUCCESS,
    EVENT_TASK_CONTINUE,
    EVENT_TASK_CONTINUED,
    EVENT_TASK_FAILED,
    EVENT_TASK_FINISH,
    EVENT_TASK_START,
    EVENT_TASK_STARTED,
    KEY_AUDIO,
    KEY_CHANNEL,
    KEY_DATA,
    KEY_EVENT,
    KEY_FORMAT,
    KEY_MODEL,
    KEY_SAMPLE_RATE,
    KEY_SPEED,
    KEY_STATUS,
    KEY_TEXT,
    KEY_VOICE_ID,
    KEY_VOL,
)

__all__ = [
    "SenseTimeTTSConfig",
    "SenseTimeTTS",
    "TTSConnection",
    "SenseTimeTTSClient",
    "SenseTimeTTSError",
]
