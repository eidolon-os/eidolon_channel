"""Helpers for LiveKit conversation message shapes."""

from __future__ import annotations

from typing import Any


def message_text(message: Any) -> str:
    text_content = getattr(message, "text_content", None)
    if isinstance(text_content, str):
        return text_content
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(item) for item in content)
    return str(content or "")
