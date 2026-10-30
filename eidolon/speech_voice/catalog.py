"""This Host's voice catalog: built from the current adapter, refreshed on a fixed schedule.

Only this object calls ``list_voices`` (V7), and it treats every adapter alike:
at start-up, then every 24 hours (±10%), backing off 1, 2, 4… up to 60 minutes
after a failure, and at most once every 10 minutes on the Owner's request.
What a call actually does — read a file, ask a provider — is the adapter's own
business. The last good result is kept on disk and reused while the adapter's
fingerprint still matches.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import tempfile
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path

from eidolon_sdk.biz.speech_voice import (
    CONVERSATION_LANGUAGE,
    EnrollmentCapabilities,
    HostVoiceCatalog,
    VoiceOption,
    VoiceRef,
    catalog_voice_key,
    style_labels,
)

from .adapter import ProviderVoices, VoiceCatalogAdapter, VoiceEnrollmentAdapter
from .catalog_file import CatalogVoice
from .resolve import Overlay

logger = logging.getLogger(__name__)

REFRESH_INTERVAL_S = 24 * 3600
REFRESH_JITTER = 0.1
BACKOFF_FIRST_S, BACKOFF_MAX_S = 60, 3600
MANUAL_MIN_INTERVAL_S = 600


class HostVoiceCatalogService:
    def __init__(
        self,
        adapter: VoiceCatalogAdapter | None,
        adapter_classes: Iterable[type[VoiceCatalogAdapter]],
        snapshot_path: Path,
        *,
        clock: Callable[[], float] = time.time,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self.adapter = adapter
        self.overlay: Overlay = {
            (d.provider, d.voice.voice_id): d.voice for cls in adapter_classes for d in cls.describe()
        }
        self._snapshot_path = snapshot_path
        self._clock, self._jitter = clock, jitter
        self._lock = asyncio.Lock()
        self._voices: ProviderVoices | None = None
        self._refreshed_at: float | None = None
        self._failures = 0
        self._restore()

    # ------------------------------------------------------------------ state

    @property
    def state(self) -> str:
        if self._voices is None:
            return "degraded"
        fresh = self._failures == 0 and self._clock() - (self._refreshed_at or 0) < REFRESH_INTERVAL_S * (1 + REFRESH_JITTER)
        return "fresh" if fresh else "stale"

    def catalog(self) -> HostVoiceCatalog:
        adapter = self.adapter
        if adapter is None:
            return _revised(HostVoiceCatalog(
                catalog_revision="", state="fresh", host_selectable=False,
                not_selectable_reason="no_adapter", voices=[],
            ))
        default = adapter.host_default()
        found = self._voices.voices if self._voices else ()
        options = [_option(default, voice) for voice in found]
        options = [o for o in options if o.selectable or o.ref == default]
        if not any(o.ref == default for o in options):
            described = self.overlay.get((default.provider, default.voice_id))
            options.append(_option(default, described, selectable=False) if described
                           else _uncurated(default))
        options = [o.model_copy(update={"is_host_default": o.ref == default}) for o in options]
        options.sort(key=lambda o: (not o.is_host_default, o.rank, o.voice_key))
        selectable = sum(o.selectable for o in options)
        enrollment = (adapter.enrollment_capabilities()
                      if isinstance(adapter, VoiceEnrollmentAdapter) else EnrollmentCapabilities())
        return _revised(HostVoiceCatalog(
            catalog_revision="", state=self.state, host_selectable=selectable >= 2,
            not_selectable_reason=None if selectable >= 2 else "single_voice",
            default_voice_key=catalog_voice_key(default),
            refreshed_at=datetime.fromtimestamp(self._refreshed_at, UTC) if self._refreshed_at else None,
            voices=options, uncurated_available=self._voices.uncurated if self._voices else 0,
            enrollment=enrollment,
        ))

    # ------------------------------------------------------------------ the schedule

    async def refresh(self) -> bool:
        """One ``list_voices`` call. Returns whether it succeeded."""
        if self.adapter is None:
            return True
        async with self._lock:
            try:
                voices = await self.adapter.list_voices()
            except Exception:
                self._failures += 1
                logger.warning("voice catalog refresh failed (%d in a row)", self._failures, exc_info=True)
                return False
            self._voices, self._refreshed_at, self._failures = voices, self._clock(), 0
            self._persist()
            return True

    async def refresh_on_request(self) -> HostVoiceCatalog:
        """The Owner pulled to refresh; honoured at most once every 10 minutes."""
        if self._refreshed_at is None or self._clock() - self._refreshed_at >= MANUAL_MIN_INTERVAL_S:
            await self.refresh()
        return self.catalog()

    def next_delay(self) -> float:
        if self._failures:
            return min(BACKOFF_FIRST_S * 2 ** (self._failures - 1), BACKOFF_MAX_S)
        return REFRESH_INTERVAL_S * (1 + REFRESH_JITTER * (2 * self._jitter() - 1))

    async def run(self) -> None:
        """Refresh now, then on the schedule, until cancelled."""
        await self.refresh()
        while True:
            await asyncio.sleep(self.next_delay())
            await self.refresh()

    # ------------------------------------------------------------------ the snapshot

    def _persist(self) -> None:
        assert self.adapter is not None and self._voices is not None
        payload = {
            "fingerprint": self.adapter.fingerprint(), "refreshed_at": self._refreshed_at,
            "voice_ids": [v.voice_id for v in self._voices.voices], "uncurated": self._voices.uncurated,
        }
        try:
            self._snapshot_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".catalog-", dir=self._snapshot_path.parent)
            with os.fdopen(fd, "w") as out:
                json.dump(payload, out)
            os.replace(temporary, self._snapshot_path)
        except OSError:
            logger.warning("voice catalog snapshot not written", exc_info=True)

    def _restore(self) -> None:
        if self.adapter is None:
            return
        try:
            payload = json.loads(self._snapshot_path.read_text())
        except (OSError, ValueError):
            return
        if payload.get("fingerprint") != self.adapter.fingerprint():
            return
        described = {d.voice.voice_id: d.voice for d in type(self.adapter).describe()}
        voices = tuple(described[i] for i in payload.get("voice_ids", []) if i in described)
        self._voices = ProviderVoices(voices=voices, uncurated=int(payload.get("uncurated", 0)))
        self._refreshed_at = float(payload.get("refreshed_at") or 0) or None


def _option(default: VoiceRef, voice: CatalogVoice, *, selectable: bool | None = None) -> VoiceOption:
    ref = VoiceRef(provider=default.provider, model=default.model, voice_id=voice.voice_id)
    return VoiceOption(
        voice_key=catalog_voice_key(ref), ref=ref, display_name=voice.display_name, summary=voice.summary,
        gender=voice.gender, age_band=voice.age_band, languages=voice.languages, styles=voice.styles,
        style_labels=style_labels(voice.styles), rank=voice.rank,
        selectable=CONVERSATION_LANGUAGE in voice.languages if selectable is None else selectable,
    )


def _uncurated(default: VoiceRef) -> VoiceOption:
    return VoiceOption(
        voice_key=catalog_voice_key(default), ref=default, display_name=default.voice_id,
        gender="unspecified", age_band="unspecified", languages=[], styles=[], style_labels=[],
        rank=0, selectable=False, curated=False,
    )


def _revised(catalog: HostVoiceCatalog) -> HostVoiceCatalog:
    body = catalog.model_dump(mode="json", exclude={"catalog_revision", "refreshed_at"})
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return catalog.model_copy(update={"catalog_revision": f"sha256:{digest[:32]}"})
