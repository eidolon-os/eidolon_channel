"""A conversation's voice tracks (design §3.2).

A segment continues the track of its own voice on this device: among the
active tracks, the one it matches best, when its same-device LLR against that
track reaches the policy's continuation threshold. Otherwise it opens a new
track. Continuing needs the segment's own acoustic evidence; it never follows
from who spoke last.

A track is scored as one probe: the mean of its latest segments' vectors with
their voiced speech summed (design §3.3 L5), over no more segments than the
condition was calibrated on.

Bounded and in memory: at most six tracks of their latest eight segments; a
track idle for ten minutes expires; all of it ends with the conversation.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

MAX_TRACKS = 6
MAX_SEGMENTS = 8
IDLE_S = 600


@dataclass
class Track:
    track_id: str
    last_active: float
    segments: list[tuple[np.ndarray, int]] = field(default_factory=list)

    def probe(self, latest: int) -> tuple[np.ndarray, int, int]:
        """The mean vector, voiced speech and segment count of the latest ``latest`` segments."""
        recent = self.segments[-latest:]
        return np.mean([v for v, _ in recent], axis=0), sum(ms for _, ms in recent), len(recent)


class Tracks:
    def __init__(self, *, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._tracks: list[Track] = []

    def assign(self, vector, speech_ms: int, same_voice_llr: Callable[[Track], float],
               threshold: float) -> Track:
        """The track this segment continues, or a new one; the segment is added to it."""
        now = self._clock()
        self._tracks = [t for t in self._tracks if now - t.last_active < IDLE_S]
        scored = [(same_voice_llr(t), t) for t in self._tracks]
        matching = [(llr, t) for llr, t in scored if llr >= threshold]
        if matching:
            track = max(matching, key=lambda x: x[0])[1]
        else:
            track = Track(uuid.uuid4().hex[:16], now)
            self._tracks.append(track)
            if len(self._tracks) > MAX_TRACKS:
                self._tracks.remove(min(self._tracks, key=lambda t: t.last_active))
        track.segments = (track.segments + [(np.asarray(vector, dtype=np.float64), int(speech_ms))])[-MAX_SEGMENTS:]
        track.last_active = now
        return track
