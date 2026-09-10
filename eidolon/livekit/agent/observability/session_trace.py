"""Session-scoped trace writer: one NDJSON file per voice session.

This is an out-edge, not an instrument. Nothing here decides anything about a
turn; it takes records the pipeline already produces — session marks, the
existing per-turn timeline snapshot, and the semantic events
``ChannelTurnEventSink`` already emits and currently drops — and appends them
where they can be read afterwards.

Three properties are load-bearing, and each one exists because the alternative
would put observation on the voice hot path:

* **Asynchronous.** The caller enqueues and returns. Writing is a daemon
  thread's job. ``TurnTimeline.append_debug_jsonl`` does a synchronous
  open/write/close and is called several times per turn; a session trace runs
  an order of magnitude more often and could not do the same.
* **Bounded.** A full queue drops the record and counts it. A file past its
  byte ceiling stops accepting payloads and says so once. An observer that
  can grow without limit is an observer that eventually takes the Host down.
* **Silent on failure.** Every public method swallows its own errors. A trace
  that cannot be written is a trace that is missing, never a turn that failed.

Wall clock is recorded once per session, as a pair with the monotonic reading
taken at the same instant (:class:`ClockAnchor`). Every other timestamp here
stays monotonic — that is what the pipeline measures with, and converting each
one at write time would bake this process's clock into every row. One anchor
per file lets a reader convert all of them, and lets rows from this process be
placed on the same axis as the Agent's, the provider's and the device's logs.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .timeline import ClockAnchor

logger = logging.getLogger("agent.observability.session_trace")

#: One line's ``record_kind``. Kept small and explicit: a reader selects rows by
#: this field, and the per-turn distinction below is the whole reason it exists.
RECORD_SESSION_OPEN = "session_open"
RECORD_SESSION_MARK = "session_mark"
RECORD_SESSION_CLOSE = "session_close"
RECORD_TURN_PROGRESS = "turn_progress"
RECORD_TURN_FINAL = "turn_final"
RECORD_EVENT = "event"

#: Session-scope marks, in the order a healthy session reaches them. Declared
#: rather than free-form so a missing one is visible as an absence instead of a
#: typo that silently never matched.
SESSION_MARKS: tuple[str, ...] = (
    "room_joined",
    "runtime_participant_resolved",
    "warmup_done",
    "avatar_ready",
    "session_started",
    "first_turn",
)

#: Dots are deliberately not in the safe set. An id here is hex, a uuid or a
#: kebab slug, so nothing needs one — and keeping them out means a hostile
#: ``owner_id`` of ``../..`` cannot even produce a *filename* that reads like a
#: traversal, let alone perform one.
_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass(frozen=True)
class _CloseRequest:
    """The last record, finished by the thread that knows the real counters.

    Everything else is serialized by the caller — that snapshots payloads that
    share nested dicts with a live ``TurnTimeline``, so a later mutation cannot
    rewrite a row already handed over. The closing summary is the exception on
    purpose: ``written``, ``dropped`` and ``truncated`` are only true once the
    queue has drained, and the caller reads them before that has happened.
    """

    payload: dict[str, Any]


@dataclass(frozen=True)
class SessionTraceSettings:
    """What the writer is allowed to do, resolved once at session open."""

    root: str
    max_queue: int = 4096
    max_file_bytes: int = 8_000_000
    retention_days: int = 7

    @property
    def enabled(self) -> bool:
        return bool(self.root.strip())


class SessionTraceWriter:
    """Append trace records for one session. Disabled writers are no-ops.

    Construct through :meth:`open`, which returns a disabled writer rather than
    raising when tracing is off or the path cannot be prepared — so callers
    never branch on whether observation is available.
    """

    def __init__(
        self,
        *,
        path: Path | None,
        settings: SessionTraceSettings,
        session_id: str,
        anchor: ClockAnchor | None = None,
    ) -> None:
        self._path = path
        self._settings = settings
        self._session_id = session_id
        self._anchor = anchor or ClockAnchor.now()
        self._seq = 0
        self._dropped = 0
        self._written = 0
        self._bytes = 0
        self._truncated = False
        self._session_marks: dict[str, float] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[str | _CloseRequest] | None = None
        self._thread: threading.Thread | None = None
        if self._path is not None:
            self._queue = queue.Queue(maxsize=max(16, settings.max_queue))
            self._thread = threading.Thread(
                target=self._drain,
                name=f"session-trace-{session_id[:12]}",
                daemon=True,
            )
            self._thread.start()

    # -- construction ----------------------------------------------------- #

    @classmethod
    def open(
        cls,
        *,
        settings: SessionTraceSettings,
        session_id: str,
        owner_id: str = "",
        companion_id: str = "",
        room_name: str = "",
        interaction_mode: str = "",
    ) -> "SessionTraceWriter":
        """Return a writer for this session, or a disabled one.

        A disabled writer is the normal outcome of tracing being off. It is also
        the outcome of an unusable path: the reason is logged once and the voice
        session proceeds, because a session that refuses to run without its
        trace file would be a worse product than one that runs untraced.
        """

        anchor = ClockAnchor.now()
        path: Path | None = None
        if settings.enabled and session_id and _usable_root(settings.root):
            try:
                path = _resolve_path(
                    root=settings.root,
                    session_id=session_id,
                    owner_id=owner_id,
                    companion_id=companion_id,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                _prune_expired(Path(settings.root).expanduser(), settings.retention_days)
            except Exception as exc:  # noqa: BLE001 - observation must not break voice
                logger.warning("session trace disabled for %s: %s", session_id, exc)
                path = None
        writer = cls(path=path, settings=settings, session_id=session_id, anchor=anchor)
        writer._emit(
            RECORD_SESSION_OPEN,
            {
                "clock_anchor": anchor.as_dict(),
                "owner_id": owner_id or None,
                "companion_id": companion_id or None,
                "room_name": room_name or None,
                "interaction_mode": interaction_mode or None,
                "pid": os.getpid(),
            },
        )
        return writer

    # -- properties ------------------------------------------------------- #

    @property
    def enabled(self) -> bool:
        return self._path is not None

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def clock_anchor(self) -> ClockAnchor:
        return self._anchor

    @property
    def dropped_count(self) -> int:
        return self._dropped

    @property
    def written_count(self) -> int:
        return self._written

    # -- recording -------------------------------------------------------- #

    def session_mark(self, name: str, *, at: float | None = None, **fields: Any) -> None:
        """Record reaching one session-scope milestone. First occurrence wins.

        First-occurrence, like the per-turn marks: a session joins its room
        once, and a second ``room_joined`` would mean a reconnect, which is a
        different fact and deserves its own name rather than overwriting this
        one.

        ``at`` replays a monotonic reading taken earlier. The pipeline reaches
        ``room_joined`` before it knows which Owner and Companion the file
        should be named for, so those marks are buffered and handed over once
        the writer exists — with the time they actually happened, not the time
        they were flushed.
        """

        if name not in SESSION_MARKS:
            # A typo here would otherwise become a mark nobody ever queries.
            logger.debug("ignoring unknown session mark %r", name)
            return
        with self._lock:
            if name in self._session_marks:
                return
            at = time.monotonic() if at is None else at
            self._session_marks[name] = at
        elapsed = at - self._anchor.monotonic
        self._emit(
            RECORD_SESSION_MARK,
            {
                "mark": name,
                "monotonic": at,
                "since_open_ms": round(max(0.0, elapsed) * 1000, 1),
                **_scalars(fields),
            },
        )

    def turn_record(self, snapshot: dict[str, Any], *, final: bool, reason: str = "") -> None:
        """Record one per-turn timeline snapshot.

        ``final`` is the whole point of this method. The pipeline writes a
        turn's snapshot more than once — a superseded candidate is appended
        where it was replaced, and the terminal flush appends it again — so a
        reader that treats every row as a turn counts the abnormal turns two or
        three times. Saying which row is the settled one lets the reader pick.
        """

        payload = dict(snapshot)
        payload["final"] = bool(final)
        if reason:
            payload["record_reason"] = reason
        self._emit(RECORD_TURN_FINAL if final else RECORD_TURN_PROGRESS, payload)

    def event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Record one semantic channel event (phase change, milestone, terminal)."""

        self._emit(RECORD_EVENT, {"event_type": event_type, **payload})

    def close(self, *, reason: str = "session_ended", timeout: float = 2.0) -> None:
        """Write the closing record and stop the writer thread.

        Bounded join: a stuck filesystem must not hold up teardown, and the
        thread is a daemon precisely so that abandoning it is safe.
        """

        with self._lock:
            marks = dict(self._session_marks)
            self._seq += 1
            seq = self._seq
        closed_at = time.monotonic()
        payload = {
            "record_kind": RECORD_SESSION_CLOSE,
            "session_id": self._session_id,
            "seq": seq,
            "t_mono": closed_at,
            "reason": reason,
            "duration_ms": round(max(0.0, closed_at - self._anchor.monotonic) * 1000, 1),
            "session_marks_ms": {
                name: round((at - self._anchor.monotonic) * 1000, 1)
                for name, at in sorted(marks.items(), key=lambda kv: kv[1])
            },
            "missing_session_marks": [m for m in SESSION_MARKS if m not in marks],
        }
        thread, q = self._thread, self._queue
        self._thread = None
        if q is not None:
            request = _CloseRequest(payload=payload)
            try:
                q.put_nowait(request)
            except queue.Full:
                # The queue is full precisely when the summary matters most, so
                # make room for it: the oldest pending record is worth less than
                # knowing how many were lost.
                try:
                    q.get_nowait()
                    self._dropped += 1
                    q.put_nowait(request)
                except (queue.Empty, queue.Full):  # pragma: no cover - best effort
                    pass
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("session trace writer did not finish within %.1fs", timeout)
        self._path = None

    # -- internals -------------------------------------------------------- #

    def _emit(self, record_kind: str, payload: dict[str, Any]) -> None:
        q = self._queue
        if q is None or self._path is None:
            return
        with self._lock:
            self._seq += 1
            seq = self._seq
        record = {
            "record_kind": record_kind,
            "session_id": self._session_id,
            "seq": seq,
            "t_mono": time.monotonic(),
            **payload,
        }
        try:
            line = json.dumps(record, ensure_ascii=False, default=_fallback)
        except Exception:  # noqa: BLE001 - an unserializable payload is not a voice failure
            self._dropped += 1
            logger.debug("session trace record not serializable kind=%s", record_kind)
            return
        try:
            q.put_nowait(line)
        except queue.Full:
            self._dropped += 1

    def _drain(self) -> None:
        path = self._path
        if path is None:  # pragma: no cover - constructor guarantees otherwise
            return
        try:
            handle = path.open("a", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("session trace file unavailable %s: %s", path, exc)
            return
        q = self._queue
        assert q is not None
        with handle:
            while True:
                item = q.get()
                if isinstance(item, _CloseRequest):
                    self._write_closing(handle, item)
                    return
                line = item
                if self._bytes >= self._settings.max_file_bytes:
                    if not self._truncated:
                        self._truncated = True
                        handle.write(
                            json.dumps(
                                {
                                    "record_kind": "truncated",
                                    "session_id": self._session_id,
                                    "max_file_bytes": self._settings.max_file_bytes,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        handle.flush()
                    self._dropped += 1
                    continue
                try:
                    handle.write(line + "\n")
                    handle.flush()
                except Exception:  # noqa: BLE001
                    self._dropped += 1
                    continue
                self._bytes += len(line) + 1
                self._written += 1

    def _write_closing(self, handle: Any, request: _CloseRequest) -> None:
        """Write the summary last, with the counters this thread actually owns.

        Deliberately exempt from the byte ceiling: the row that says how much
        was lost is the one row a truncated file must still carry.
        """

        payload = {
            **request.payload,
            "dropped_record_count": self._dropped,
            "written_record_count": self._written,
            "truncated": self._truncated,
        }
        try:
            handle.write(json.dumps(payload, ensure_ascii=False, default=_fallback) + "\n")
            handle.flush()
        except Exception:  # noqa: BLE001 - teardown must not raise into the session
            logger.debug("session trace closing record not written", exc_info=True)


def _usable_root(root: str) -> bool:
    """Refuse a root that still carries an unexpanded ``$VAR``.

    The loader expands ``$VARS`` through ``os.path.expandvars``, which leaves an
    *undefined* variable exactly as it found it. This repo already has one
    directory literally named ``$EIDOLON_CACHE_ROOT`` in its root, created by
    ``bailian_stt.dump_dir`` doing ``mkdir`` on such a string — 11 MB of
    recorded speech in a path nobody meant to write to, and not covered by
    ``.gitignore``.

    So a root that still contains ``$`` is a misconfiguration, not a path.
    Saying so and writing nothing is strictly better than making a second one.
    """

    if "$" not in root:
        return True
    logger.warning(
        "session trace disabled: session_trace_path still contains an unexpanded "
        "variable (%s) — set it in the worker environment or use a literal path",
        root,
    )
    return False


def _resolve_path(*, root: str, session_id: str, owner_id: str, companion_id: str) -> Path:
    """``<root>/<date>/<owner>__<companion>__<session>.ndjson``.

    The layout *is* the index: a reader lists a day by reading one directory and
    finds one session by its name, so nothing here needs a database until a
    query arrives that a filename cannot answer.
    """

    base = Path(root).expanduser()
    name = "__".join(
        (
            _segment(owner_id or "unknown-owner"),
            _segment(companion_id or "unknown-companion"),
            _segment(session_id),
        )
    )
    return base / date.today().isoformat() / f"{name}.ndjson"


def _segment(value: str) -> str:
    """Make one path segment out of an id without letting it escape the root."""

    cleaned = _SAFE_SEGMENT.sub("-", (value or "").strip()).strip("-.")
    return (cleaned or "unknown")[:64]


def _prune_expired(root: Path, retention_days: int) -> None:
    """Drop whole day-directories past the retention window.

    Per-day granularity keeps this to one readdir and makes what was deleted
    obvious from the outside. A malformed directory name is left alone rather
    than guessed at.
    """

    if retention_days <= 0 or not root.is_dir():
        return
    cutoff = date.today() - timedelta(days=retention_days)
    for child in root.iterdir():
        if not child.is_dir():
            continue
        try:
            when = date.fromisoformat(child.name)
        except ValueError:
            continue
        if when < cutoff:
            shutil.rmtree(child, ignore_errors=True)


def _scalars(fields: dict[str, Any]) -> dict[str, Any]:
    """Keep bounded scalar context only; a mark is not a place for payloads."""

    return {
        key: value
        for key, value in fields.items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }


def _fallback(value: Any) -> str:
    return f"<{type(value).__name__}>"


__all__ = [
    "RECORD_EVENT",
    "RECORD_SESSION_CLOSE",
    "RECORD_SESSION_MARK",
    "RECORD_SESSION_OPEN",
    "RECORD_TURN_FINAL",
    "RECORD_TURN_PROGRESS",
    "SESSION_MARKS",
    "SessionTraceSettings",
    "SessionTraceWriter",
]
