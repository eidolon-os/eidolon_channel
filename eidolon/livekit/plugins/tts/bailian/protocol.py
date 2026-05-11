"""DashScope CosyVoice WebSocket protocol helpers."""

from __future__ import annotations

from typing import Any

ACTION_RUN_TASK = "run-task"
ACTION_CONTINUE_TASK = "continue-task"
ACTION_FINISH_TASK = "finish-task"

EVENT_TASK_STARTED = "task-started"
EVENT_TASK_FINISHED = "task-finished"
EVENT_TASK_FAILED = "task-failed"
EVENT_RESULT_GENERATED = "result-generated"


def build_run_task_payload(
    *,
    task_id: str,
    model: str,
    voice: str,
    audio_format: str,
    sample_rate: int,
    rate: float = 1.0,
    volume: int = 50,
    pitch: float = 1.0,
) -> dict[str, Any]:
    """Build the ``run-task`` payload per CosyVoice WebSocket API spec.

    Parameter names follow the official API documentation exactly:
      * ``rate``   — speech speed multiplier, [0.5, 2.0], default 1.0
      * ``volume`` — loudness, integer [0, 100], default 50
      * ``pitch``  — pitch multiplier, [0.5, 2.0], default 1.0
    """
    return {
        "header": {
            "action": ACTION_RUN_TASK,
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {
            "task_group": "audio",
            "task": "tts",
            "function": "SpeechSynthesizer",
            "model": model,
            "input": {},
            "parameters": {
                "text_type": "PlainText",
                "voice": voice,
                "format": audio_format,
                "sample_rate": sample_rate,
                "rate": rate,
                "volume": volume,
                "pitch": pitch,
            },
        },
    }


def build_continue_task_payload(*, task_id: str, text: str) -> dict[str, Any]:
    return {
        "header": {
            "action": ACTION_CONTINUE_TASK,
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {
            "input": {
                "text": text,
            }
        },
    }


def build_finish_task_payload(*, task_id: str) -> dict[str, Any]:
    return {
        "header": {
            "action": ACTION_FINISH_TASK,
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {"input": {}},
    }


def parse_event(message: dict[str, Any]) -> str:
    header = message.get("header")
    if not isinstance(header, dict):
        return ""
    event = header.get("event")
    return event if isinstance(event, str) else ""
