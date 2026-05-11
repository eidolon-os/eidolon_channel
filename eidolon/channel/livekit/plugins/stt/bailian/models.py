"""Typed models for Bailian FunASR protocol (DashScope WebSocket API).

The DashScope protocol uses a ``{"header": {...}, "payload": {...}}`` structure:
- ``header.event`` contains the event type (e.g. "task-started", "result-generated")
- ``header.task_id`` contains the task identifier
- ``header.error_code`` / ``header.error_message`` are set on failure
- ``payload`` contains event-specific data
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class FunASREventType(str, Enum):
    TASK_STARTED = "task-started"
    TASK_FAILED = "task-failed"
    TASK_FINISHED = "task-finished"
    RESULT_GENERATED = "result-generated"
    HEARTBEAT = "heartbeat"


@dataclass
class FunASRWord:
    """A single word with word-level timing from FunASR."""

    text: str
    begin_time: int = 0  # milliseconds
    end_time: int = 0  # milliseconds

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FunASRWord:
        return cls(
            text=d.get("text", "") or "",
            begin_time=int(d.get("begin_time") or 0),
            end_time=int(d.get("end_time") or 0),
        )


@dataclass
class FunASRSentence:
    """A sentence-level result from FunASR."""

    text: str = ""
    begin_time: int = 0  # milliseconds
    end_time: int = 0  # milliseconds
    sentence_end: bool = False
    words: list[FunASRWord] = field(default_factory=list)
    punctuation: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FunASRSentence:
        words_raw = d.get("words", [])
        words = [FunASRWord.from_dict(w) for w in words_raw] if isinstance(words_raw, list) else []
        return cls(
            text=d.get("text", "") or "",
            begin_time=int(d.get("begin_time") or 0),
            end_time=int(d.get("end_time") or 0),
            sentence_end=bool(d.get("sentence_end", False)),
            words=words,
            punctuation=d.get("text_with_punct", "") or "",
        )


@dataclass
class FunASRResultGenerated:
    """A 'result-generated' event payload from the FunASR server."""

    task_id: str
    request_id: str
    sentences: list[FunASRSentence] = field(default_factory=list)
    # The raw JSON dict for any unknown fields
    _extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, msg: dict[str, Any]) -> FunASRResultGenerated:
        header = msg.get("header", {})
        payload = msg.get("payload", {})
        output = payload.get("output", {})
        sentences_raw = output.get("sentence", [])
        if isinstance(sentences_raw, dict):
            sentences_raw = [sentences_raw]
        sentences = [FunASRSentence.from_dict(s) for s in sentences_raw]
        return cls(
            task_id=header.get("task_id", ""),
            request_id=header.get("request_id", ""),
            sentences=sentences,
            _extra={k: v for k, v in msg.items() if k not in ("header", "payload")},
        )

    def merged_text(self) -> str:
        """Concatenate all sentence texts."""
        return "".join(s.text for s in self.sentences)

    def is_final(self) -> bool:
        """True if at least one sentence has sentence_end=True."""
        return any(s.sentence_end for s in self.sentences)

    def latest_sentence(self) -> FunASRSentence | None:
        """Return the last sentence in this result."""
        return self.sentences[-1] if self.sentences else None


@dataclass
class FunASRTaskStarted:
    task_id: str
    task_status: str = ""

    @classmethod
    def from_dict(cls, msg: dict[str, Any]) -> FunASRTaskStarted:
        header = msg.get("header", {})
        return cls(
            task_id=header.get("task_id", ""),
            task_status=header.get("task_status", ""),
        )


@dataclass
class FunASRTaskFailed:
    task_id: str
    error_message: str = ""
    error_code: str = ""

    @classmethod
    def from_dict(cls, msg: dict[str, Any]) -> FunASRTaskFailed:
        header = msg.get("header", {})
        return cls(
            task_id=header.get("task_id", ""),
            error_message=header.get("error_message", ""),
            error_code=header.get("error_code", ""),
        )


@dataclass
class FunASRTaskFinished:
    task_id: str
    request_id: str = ""

    @classmethod
    def from_dict(cls, msg: dict[str, Any]) -> FunASRTaskFinished:
        header = msg.get("header", {})
        return cls(
            task_id=header.get("task_id", ""),
            request_id=header.get("request_id", ""),
        )


@dataclass
class FunASRHeartbeat:
    """Server heartbeat / keep-alive."""

    task_id: str

    @classmethod
    def from_dict(cls, msg: dict[str, Any]) -> FunASRHeartbeat:
        header = msg.get("header", {})
        return cls(task_id=header.get("task_id", ""))


def parse_funasr_message(data: dict[str, Any]) -> tuple[str, Any]:
    """Parse a FunASR server message and return (event_type, parsed_payload).

    The DashScope protocol uses ``{"header": {...}, "payload": {...}}``.
    The event type is found at ``header.event``.

    Returns (event_name, payload_obj). Raises ValueError for unknown event types.
    """
    header = data.get("header", {})
    event: str = header.get("event", "")
    payload_map: dict[str, type] = {
        FunASREventType.TASK_STARTED.value: FunASRTaskStarted,
        FunASREventType.TASK_FAILED.value: FunASRTaskFailed,
        FunASREventType.TASK_FINISHED.value: FunASRTaskFinished,
        FunASREventType.RESULT_GENERATED.value: FunASRResultGenerated,
        FunASREventType.HEARTBEAT.value: FunASRHeartbeat,
    }

    cls = payload_map.get(event)
    if cls is None:
        raise ValueError(f"Unknown FunASR event type: {event!r}")
    return event, cls.from_dict(data)
