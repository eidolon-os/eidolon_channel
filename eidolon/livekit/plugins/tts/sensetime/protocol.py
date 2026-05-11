"""SenseAudio TTS WebSocket protocol constants.

Based on https://senseaudio.cn/docs/tts_api_internal
"""

# Server-side events
EVENT_CONNECTED_SUCCESS = "connected_success"
EVENT_TASK_STARTED = "task_started"
EVENT_TASK_CONTINUE = "task_continue"
EVENT_TASK_CONTINUED = "task_continued"
EVENT_TASK_FINISHED = "task_finished"
EVENT_TASK_FAILED = "task_failed"

# Client-side events
EVENT_TASK_START = "task_start"
EVENT_TASK_FINISH = "task_finish"

# Message body keys
KEY_EVENT = "event"
KEY_DATA = "data"
KEY_TEXT = "text"
KEY_AUDIO = "audio"
KEY_STATUS = "status"
KEY_BASE_RESP = "base_resp"
KEY_STATUS_MSG = "status_msg"
KEY_MODEL = "model"
KEY_VOICE_SETTING = "voice_setting"
KEY_AUDIO_SETTING = "audio_setting"
KEY_VOICE_ID = "voice_id"
KEY_SPEED = "speed"
KEY_VOL = "vol"
KEY_PITCH = "pitch"
KEY_SAMPLE_RATE = "sample_rate"
KEY_BITRATE = "bitrate"
KEY_FORMAT = "format"
KEY_CHANNEL = "channel"

# NOTE: SenseAudio docs mention `status=2` (segment complete) and `is_final`,
# but production logs (2026-05-02) show the server only ever sends `status=0`
# and never sends `is_final` in persistent-connection mode. We instead detect
# end-of-segment via the "batch-end" marker: a `task_continued` message whose
# `data.audio` is empty/missing while `data.status` field is present. Counting
# these markers vs `task_continue` segments sent gives a reliable, time-free
# exit condition. See SenseTimeSynthesizeStream._check_exit() in tts.py.
