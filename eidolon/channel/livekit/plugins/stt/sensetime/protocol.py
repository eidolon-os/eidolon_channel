"""SenseAudio STT WebSocket protocol constants.

Source: https://senseaudio.cn/docs/speech_recognition/websocket

Notable differences from SenseAudio TTS:
- Audio is transmitted as **raw binary WebSocket frames**, not as JSON-wrapped
  hex. There is therefore NO ``task_continue`` event for STT — the doc only
  defines ``task_start`` / ``task_finish`` JSON events plus binary audio.
- Server emits ``result_final`` (not ``task_continued``) carrying transcripts.
- ``data.is_final`` in ``result_final`` is always ``true`` (each event = one
  complete sentence), unlike TTS where ``is_final`` is documented but never
  observed.
"""

# Server-side events
EVENT_CONNECTED_SUCCESS = "connected_success"
EVENT_TASK_STARTED = "task_started"
EVENT_RESULT_FINAL = "result_final"
EVENT_TASK_FINISHED = "task_finished"
EVENT_TASK_FAILED = "task_failed"

# Client-side events (only JSON ones; audio frames have no "event")
EVENT_TASK_START = "task_start"
EVENT_TASK_FINISH = "task_finish"

# Top-level message keys
KEY_EVENT = "event"
KEY_DATA = "data"
KEY_BASE_RESP = "base_resp"
KEY_STATUS_MSG = "status_msg"
KEY_STATUS_CODE = "status_code"

# task_start payload keys
KEY_MODEL = "model"
KEY_AUDIO_SETTING = "audio_setting"
KEY_SAMPLE_RATE = "sample_rate"
KEY_FORMAT = "format"
KEY_CHANNEL = "channel"
KEY_VAD_SETTING = "vad_setting"
KEY_TRANSCRIPTION_SETTING = "transcription_setting"

# result_final.data keys
KEY_TEXT = "text"
KEY_IS_FINAL = "is_final"
KEY_SEGMENT_ID = "segment_id"
KEY_TIMESTAMP_END = "timestamp_end"
