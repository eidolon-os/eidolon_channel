"""Which voice this session's TTS speaks with — decided once, at session start.

Reads the Companion's choice from Data and asks channel-provider to resolve it
against the Host's catalog, all within ``DEADLINE_S``. A voice is experience,
the conversation is function (V2): any failure or timeout falls back to a voice
this Host can certainly produce, and nothing here raises.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from eidolon_sdk.biz.speech_voice import (
    CompanionSpeechVoice,
    ResolvedSpeechVoice,
    ResolveRequest,
)
from eidolon_sdk.biz.system_data import SystemDataNotFound
from eidolon_sdk.core.http import create_async_client

from eidolon.livekit.plugins.tts.voice_catalogs import current_adapter
from eidolon.speech_voice.resolve import resolve_without_catalog

logger = logging.getLogger(__name__)

DEADLINE_S = 0.5


@dataclass(frozen=True)
class SessionVoice:
    resolved: ResolvedSpeechVoice
    resolve_ms: int

    def trace_fields(self) -> dict[str, object]:
        r = self.resolved
        return {"voice_key": r.voice_key or "", "reason": r.reason,
                "catalog_revision": r.catalog_revision or "", "resolve_ms": self.resolve_ms}


def _provider_endpoint() -> tuple[str, str]:
    from eidolon.channel_provider.config import load_provider_config

    provider = load_provider_config()
    return f"http://127.0.0.1:{provider.http.port}", provider.bearer_token


async def resolve_session_voice(
    cfg, *, companion_id: Callable[[], Awaitable[str | None]], authority,
    dispatched: ResolvedSpeechVoice | None = None,
) -> SessionVoice:
    """``companion_id()`` may be slow (it waits for identity); the deadline covers it too.

    ``dispatched`` is what channel-provider resolved as it dispatched this job
    (speech_voice.dispatch); with it, none of the calls below are made.
    """
    started = time.monotonic()
    adapter = current_adapter(cfg)
    current = adapter.host_default() if adapter is not None else None

    def done(resolved: ResolvedSpeechVoice) -> SessionVoice:
        return SessionVoice(resolved, int((time.monotonic() - started) * 1000))

    if current is None:
        return done(ResolvedSpeechVoice(ref=None, voice_key=None, display_name="主机默认",
                                        reason="host_fixed_voice"))
    if dispatched is not None:
        return done(dispatched)
    record: CompanionSpeechVoice | None = None
    stage = "authority_unavailable"
    try:
        async with asyncio.timeout(DEADLINE_S):
            target = await companion_id()
            if target is None:
                return done(resolve_without_catalog(None, current, reason="unset"))
            try:
                record = await authority.get_companion_speech_voice(target)
            except SystemDataNotFound:
                record = None  # an authority that predates voice choices
            stage = "catalog_unavailable"
            url, token = _provider_endpoint()
            async with create_async_client(timeout=DEADLINE_S, trust_env=False) as http:
                response = await http.post(
                    f"{url}/v1/speech-voices/resolve",
                    json=ResolveRequest(speech_voice=record).model_dump(mode="json"),
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return done(ResolvedSpeechVoice.model_validate(response.json()))
    except Exception:  # includes the deadline
        logger.warning("speech voice not resolved (%s); using a voice this config can produce", stage,
                       exc_info=True)
    return done(resolve_without_catalog(record, current, reason=stage))


async def shielded_companion_id(identity: Awaitable) -> str | None:
    """The session's Companion, without letting a voice deadline cancel identity resolution."""
    task = asyncio.ensure_future(identity)
    # Identity keeps resolving for the session after a timeout here; never leave its error unread.
    task.add_done_callback(lambda t: t.cancelled() or t.exception())
    context = await asyncio.shield(task)
    return getattr(context, "companion_id", None)

