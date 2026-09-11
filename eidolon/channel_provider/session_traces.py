"""Read the per-session traces the Agent worker wrote.

The worker and this process are both the Channel authority on the same Host, so
the traces are this process's to serve — and serving them is the whole point of
writing them. Without a read surface a trace is a file somebody has to ssh for.

This module is a directory reader, deliberately not a database:

* **The layout is the index.** One file per session under a day directory, so
  listing a day is one ``readdir`` and reading a session is one file. A SQLite
  index earns its place when a query arrives that a filename and a head/tail
  read cannot answer — cross-session aggregation by phase, say — and not before.
* **A summary costs two lines, not a file.** The writer's ``session_open`` and
  ``session_close`` records already carry identity, duration, session marks,
  missing marks and the dropped/written counters. Listing reads the first line
  and the last line and never the middle.
* **The record is the truth, the filename is a label.** Filenames are sanitized
  (``[^A-Za-z0-9_-]`` collapses), while a ``conversation_id`` may legitimately
  contain ``.`` or ``:``. So lookups match the ``session_id`` *inside* the file.
  Matching the label instead would silently miss those sessions.

Nothing here imports the agent package. This process is the lightweight control
plane; pulling in the voice pipeline's dependency tree to parse JSON lines would
be a poor trade.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger("eidolon.channel_provider.session_traces")

#: How far back a listing will look, and how many sessions it will return at
#: once. Both are bounds on the answer, not on what was recorded.
_MAX_LIMIT = 200
_DEFAULT_LIMIT = 50
#: Enough to hold the last line of a trace. A ``session_close`` record carries
#: the session marks and counters, not payloads, so it stays small.
_TAIL_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class TraceQuery:
    """A bounded question about which sessions to describe."""

    owner_id: str = ""
    companion_id: str = ""
    since: datetime | None = None
    limit: int = _DEFAULT_LIMIT

    def bounded(self) -> "TraceQuery":
        limit = self.limit if 1 <= self.limit <= _MAX_LIMIT else _DEFAULT_LIMIT
        return TraceQuery(
            owner_id=self.owner_id.strip(),
            companion_id=self.companion_id.strip(),
            since=self.since,
            limit=limit,
        )


class SessionTraceReader:
    """Serve the session traces under one root directory."""

    def __init__(self, root: Path | None) -> None:
        self._root = root

    @property
    def root(self) -> Path | None:
        return self._root

    @property
    def available(self) -> bool:
        """Whether tracing has produced anything to read.

        A missing directory is the normal state when tracing is off, so it is
        an empty answer rather than an error: a caller asking "what sessions
        are there" is answered truthfully with "none recorded".
        """

        return self._root is not None and self._root.is_dir()

    def list(self, query: TraceQuery) -> list[dict[str, Any]]:
        """Describe the most recent sessions, newest first."""

        bounded = query.bounded()
        summaries: list[dict[str, Any]] = []
        for path in self._iter_paths():
            summary = self._summarize(path)
            if summary is None or not _matches(summary, bounded):
                continue
            summaries.append(summary)
            if len(summaries) >= bounded.limit:
                break
        return summaries

    def read(
        self,
        session_id: str,
        *,
        kinds: frozenset[str] | None = None,
    ) -> dict[str, Any] | None:
        """Return one session's summary and its records, or None if unknown."""

        wanted = (session_id or "").strip()
        if not wanted:
            return None
        for path in self._iter_paths():
            head = _read_head(path)
            if head is None or str(head.get("session_id") or "") != wanted:
                continue
            records = [
                record
                for record in _read_all(path)
                if kinds is None or str(record.get("record_kind") or "") in kinds
            ]
            summary = self._summarize(path) or {}
            return {
                "session": summary,
                "records": records,
                "record_count": len(records),
            }
        return None

    # -- internals -------------------------------------------------------- #

    def _iter_paths(self) -> Iterator[Path]:
        """Trace files, newest first.

        Ordered by modification time rather than by name: a filename begins
        with the Owner, so sorting by it would group by household and hide
        recency, which is the one thing every caller here wants.
        """

        if not self.available:
            return
        assert self._root is not None
        for day in _day_dirs(self._root):
            try:
                files = [p for p in day.iterdir() if p.suffix == ".ndjson" and p.is_file()]
            except OSError:  # pragma: no cover - a day removed mid-listing
                continue
            for path in sorted(files, key=_mtime, reverse=True):
                yield path

    def _summarize(self, path: Path) -> dict[str, Any] | None:
        head = _read_head(path)
        if head is None or str(head.get("record_kind") or "") != "session_open":
            # A file whose first line is not the open record is not a trace this
            # reader wrote. Skipping beats guessing at its shape.
            return None
        raw_anchor = head.get("clock_anchor")
        anchor: dict[str, Any] = raw_anchor if isinstance(raw_anchor, dict) else {}
        summary: dict[str, Any] = {
            "session_id": head.get("session_id"),
            "owner_id": head.get("owner_id"),
            "companion_id": head.get("companion_id"),
            "room_name": head.get("room_name"),
            "interaction_mode": head.get("interaction_mode"),
            "started_at": anchor.get("iso"),
            "started_at_unix_ns": anchor.get("unix_ns"),
            "trace_file": path.name,
            "trace_bytes": _size(path),
            # Absent until the closing record lands. A session with no close is
            # either still being served or was ended by something that took the
            # worker with it — "running" would claim the first without evidence.
            "status": "open",
            "reason": None,
            "duration_ms": None,
            "session_marks_ms": {},
            "missing_session_marks": None,
            "written_record_count": None,
            "dropped_record_count": None,
            "truncated": None,
        }
        tail = _read_tail(path)
        if tail is not None and str(tail.get("record_kind") or "") == "session_close":
            summary.update(
                {
                    "status": "closed",
                    "reason": tail.get("reason"),
                    "duration_ms": tail.get("duration_ms"),
                    "session_marks_ms": tail.get("session_marks_ms") or {},
                    "missing_session_marks": tail.get("missing_session_marks"),
                    "written_record_count": tail.get("written_record_count"),
                    "dropped_record_count": tail.get("dropped_record_count"),
                    "truncated": tail.get("truncated"),
                }
            )
        return summary


def _matches(summary: dict[str, Any], query: TraceQuery) -> bool:
    if query.owner_id and str(summary.get("owner_id") or "") != query.owner_id:
        return False
    if query.companion_id and str(summary.get("companion_id") or "") != query.companion_id:
        return False
    if query.since is not None:
        started = summary.get("started_at_unix_ns")
        if not isinstance(started, int):
            return False
        if started < int(query.since.timestamp() * 1e9):
            return False
    return True


def _day_dirs(root: Path) -> list[Path]:
    """Day directories, newest first. A name that is not a date is not ours.

    Sorted by name, which for ISO dates is chronological — and only the day
    granularity is ordered here; recency within a day is settled by mtime.
    """

    out: list[Path] = []
    try:
        children = list(root.iterdir())
    except OSError:  # pragma: no cover - unreadable root reads as empty
        return out
    for child in children:
        if not child.is_dir():
            continue
        try:
            date.fromisoformat(child.name)
        except ValueError:
            continue
        out.append(child)
    return sorted(out, key=lambda path: path.name, reverse=True)


def _read_head(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    return _loads(stripped, path)
    except OSError:
        logger.debug("session trace unreadable: %s", path, exc_info=True)
    return None


def _read_tail(path: Path) -> dict[str, Any] | None:
    """The last complete record, read from the end rather than by scanning."""

    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - _TAIL_BYTES))
            chunk = handle.read()
    except OSError:
        logger.debug("session trace tail unreadable: %s", path, exc_info=True)
        return None
    for raw in reversed(chunk.splitlines()):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            text = stripped.decode("utf-8")
        except UnicodeDecodeError:
            # The seek may have landed mid-character; that line is not the last
            # one anyway once a later one decodes.
            continue
        record = _loads(text, path)
        if record is not None:
            return record
    return None


def _read_all(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                record = _loads(stripped, path)
                if record is not None:
                    records.append(record)
    except OSError:
        logger.debug("session trace unreadable: %s", path, exc_info=True)
    return records


def _loads(text: str, path: Path) -> dict[str, Any] | None:
    """Parse one line, or skip it.

    A partial last line is expected: the writer appends while this reads. One
    unparsable line must not make a whole session unreadable.
    """

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        logger.debug("session trace line is not JSON: %s", path)
        return None
    return value if isinstance(value, dict) else None


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:  # pragma: no cover - removed mid-listing
        return 0.0


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:  # pragma: no cover - removed mid-listing
        return 0


__all__ = ["SessionTraceReader", "TraceQuery"]
