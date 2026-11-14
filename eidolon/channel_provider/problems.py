"""The RFC 9457 problem body every refusal on this HTTP surface is written as.

Its own module because two layers refuse: the handlers, for what a request
asks, and access control (``access.py``), for who is asking. Both have to say
it in one shape.
"""

from __future__ import annotations

import json

from aiohttp import web

from .contracts import DomainError


def problem(error: DomainError) -> web.Response:
    return problem_body(
        code=error.code,
        category=error.category,
        retryable=error.retryable,
        status=error.http_status,
        detail=str(error),
    )


def problem_body(
    *, code: str, category: str, retryable: bool, status: int, detail: str,
    audio_quality: dict | None = None,
) -> web.Response:
    value = {
        "type": f"https://problems.eidolon.live/channel/{code.lower()}",
        "title": code.replace("_", " ").title(),
        "status": status,
        "detail": detail,
        "code": code,
        "category": category,
        "retryable": retryable,
        "authority": "eidolon-channel-provider",
    }
    if audio_quality is not None:
        value["audio_quality"] = audio_quality
    return web.Response(
        text=json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        status=status,
        content_type="application/problem+json",
    )
