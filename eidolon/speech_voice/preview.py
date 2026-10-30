"""Hearing a voice before choosing it: fixed sentences, synthesized once, kept as WAV files.

Every file handed out carries the AIGC implicit identifier (see ``wav.py``).
One synthesis runs at a time per process and at most four wait behind it; a
fifth is told to come back later rather than queued without bound.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path

from eidolon_sdk.biz.speech_voice import VoiceRef

from .wav import wav_bytes

#: Two fixed sentences, each within the local engine's 60-character limit, no Companion name.
PREVIEW_TEXTS = (
    "你好，很高兴认识你。接下来，让我们一起聊聊今天的生活吧。",
    "今天过得怎么样？有什么开心的事，都可以慢慢说给我听。",
)
#: Read by a newly designed voice so it can be cloned again on another model.
RENDITION_TEXT = (
    "今天下班的路上，我顺手买了一束向日葵，放在窗台上，整个房间都亮了起来。"
    "晚饭想做个番茄炒蛋，再煮一锅玉米排骨汤。你要是饿了，就早点回来吧。"
)
TEXT_REVISION = 1
SYNTHESIS_TIMEOUT_S = 60

Synthesize = Callable[[VoiceRef, str], Awaitable[tuple[bytes, int]]]


class PreviewBusy(Exception):
    pass


class VoicePreviews:
    def __init__(self, root: Path, *, synthesize: Synthesize, propagator: str,
                 max_bytes: int = 64 * 1024 * 1024, queue_limit: int = 4) -> None:
        if max_bytes <= 0:
            raise ValueError("preview cache capacity must be positive")
        self._root, self._synthesize, self._propagator = Path(root), synthesize, propagator
        self._max_bytes, self._queue_limit = max_bytes, queue_limit
        self._slot = asyncio.Semaphore(1)
        self._waiting = 0

    async def wav(self, ref: VoiceRef, text: str, *, keep: bool = True) -> bytes:
        key = hashlib.sha256(json.dumps(
            [ref.provider, ref.model, ref.voice_id, text, TEXT_REVISION], ensure_ascii=False
        ).encode()).hexdigest()
        path = self._root / f"{key}.wav"
        if keep and (cached := self._read(path)) is not None:
            return cached
        if self._waiting >= self._queue_limit:
            raise PreviewBusy("too many previews are being made")
        self._waiting += 1
        try:
            async with self._slot:
                if keep and (cached := self._read(path)) is not None:
                    return cached
                pcm, rate = await asyncio.wait_for(self._synthesize(ref, text), SYNTHESIS_TIMEOUT_S)
        finally:
            self._waiting -= 1
        if not pcm:
            raise ValueError("synthesis produced no audio")
        data = wav_bytes(pcm, rate, aigc={
            "ContentProducer": f"eidolon/{ref.provider}", "ProduceID": key[:32],
            "ContentPropagator": self._propagator, "PropagateID": key[:32],
        })
        if keep:
            self._store(path, data)
        return data

    @staticmethod
    def _read(path: Path) -> bytes | None:
        try:
            data = path.read_bytes()
            os.utime(path, None)
            return data
        except OSError:
            return None

    def _store(self, path: Path, data: bytes) -> None:
        try:
            self._root.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".preview-", dir=self._root)
            with os.fdopen(fd, "wb") as out:
                out.write(data)
            os.replace(temporary, path)
            files = sorted(self._root.glob("*.wav"), key=lambda p: p.stat().st_mtime)
            total = sum(p.stat().st_size for p in files)
            for old in files:
                if total <= self._max_bytes:
                    break
                if old != path:
                    total -= old.stat().st_size
                    old.unlink(missing_ok=True)
        except OSError:
            pass  # a preview that could not be kept is still returned
