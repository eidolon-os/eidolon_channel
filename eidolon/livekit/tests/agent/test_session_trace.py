from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from eidolon.livekit.agent.observability import ChannelTurnEventSink

from eidolon.livekit.agent.observability.session_trace import (
    RECORD_SESSION_CLOSE,
    RECORD_SESSION_MARK,
    RECORD_SESSION_OPEN,
    RECORD_TURN_FINAL,
    RECORD_TURN_PROGRESS,
    SESSION_MARKS,
    SessionTraceSettings,
    SessionTraceWriter,
    _resolve_path,
    _segment,
)
from eidolon.livekit.agent.observability.timeline import ClockAnchor


def _open(root: Path, **kwargs) -> SessionTraceWriter:
    settings = SessionTraceSettings(root=str(root), **kwargs.pop("settings", {}))
    return SessionTraceWriter.open(
        settings=settings,
        session_id=kwargs.pop("session_id", "sess-1"),
        owner_id=kwargs.pop("owner_id", "owner-1"),
        companion_id=kwargs.pop("companion_id", "companion-1"),
        **kwargs,
    )


def _rows(root: Path) -> list[dict]:
    files = sorted(root.rglob("*.ndjson"))
    assert len(files) == 1, files
    return [json.loads(line) for line in files[0].open(encoding="utf-8") if line.strip()]


def test_tracing_off_writes_nothing_and_never_raises(tmp_path: Path) -> None:
    """An unset path is the default, so a disabled writer must be a full no-op."""

    writer = SessionTraceWriter.open(
        settings=SessionTraceSettings(root=""), session_id="sess-1"
    )
    assert writer.enabled is False
    writer.session_mark("room_joined")
    writer.turn_record({"turn_id": "t1"}, final=True)
    writer.event("channel.turn.milestone", {"milestone": "first_audio"})
    writer.close()
    assert list(tmp_path.rglob("*.ndjson")) == []


def test_session_open_carries_a_convertible_clock_anchor(tmp_path: Path) -> None:
    """The anchor pair is the only thing that makes monotonic marks absolute."""

    writer = _open(tmp_path)
    anchor = writer.clock_anchor
    writer.close()

    opened = _rows(tmp_path)[0]
    assert opened["record_kind"] == RECORD_SESSION_OPEN
    recorded = opened["clock_anchor"]
    assert recorded["monotonic"] == anchor.monotonic
    assert recorded["unix_ns"] == anchor.unix_ns

    # One second of monotonic time is one second of wall clock away from the
    # anchor: that arithmetic is the whole contract a reader depends on.
    assert anchor.to_unix_ns(anchor.monotonic + 1.0) == anchor.unix_ns + 1_000_000_000


def test_session_mark_keeps_the_first_occurrence(tmp_path: Path) -> None:
    writer = _open(tmp_path)
    writer.session_mark("room_joined")
    writer.session_mark("room_joined")
    writer.close()

    marks = [r for r in _rows(tmp_path) if r["record_kind"] == RECORD_SESSION_MARK]
    assert [m["mark"] for m in marks] == ["room_joined"]


def test_unknown_session_mark_is_dropped_not_recorded(tmp_path: Path) -> None:
    """A typo must not become a mark nobody can ever query."""

    writer = _open(tmp_path)
    writer.session_mark("rooom_joined")
    writer.close()

    assert [r for r in _rows(tmp_path) if r["record_kind"] == RECORD_SESSION_MARK] == []


def test_turn_rows_separate_progress_from_the_settled_one(tmp_path: Path) -> None:
    """The dedup contract: a reader must be able to pick one row per turn.

    The pipeline appends a turn's snapshot more than once — once where a
    candidate was superseded, once at the terminal flush — so without this
    distinction an abnormal turn is counted two or three times.
    """

    writer = _open(tmp_path)
    writer.turn_record({"turn_id": "t1", "timestamps": {"a": 1.0}}, final=False, reason="superseded")
    writer.turn_record(
        {"turn_id": "t1", "timestamps": {"a": 1.0, "b": 2.0}},
        final=True,
        reason="playback_done",
    )
    writer.close()

    rows = _rows(tmp_path)
    kinds = [r["record_kind"] for r in rows if r["record_kind"].startswith("turn_")]
    assert kinds == [RECORD_TURN_PROGRESS, RECORD_TURN_FINAL]

    finals = [r for r in rows if r["record_kind"] == RECORD_TURN_FINAL]
    assert len(finals) == 1
    assert finals[0]["turn_id"] == "t1"
    assert finals[0]["record_reason"] == "playback_done"
    assert finals[0]["final"] is True


def test_close_reports_missing_marks_and_counters(tmp_path: Path) -> None:
    writer = _open(tmp_path)
    writer.session_mark("room_joined")
    writer.session_mark("session_started")
    writer.close(reason="session_error")

    closing = _rows(tmp_path)[-1]
    assert closing["record_kind"] == RECORD_SESSION_CLOSE
    assert closing["reason"] == "session_error"
    assert set(closing["session_marks_ms"]) == {"room_joined", "session_started"}
    assert closing["missing_session_marks"] == [
        m for m in SESSION_MARKS if m not in {"room_joined", "session_started"}
    ]
    assert closing["dropped_record_count"] == 0
    assert closing["written_record_count"] >= 3


def test_a_full_queue_drops_and_counts_instead_of_blocking(tmp_path: Path) -> None:
    """Back-pressure onto the voice loop is the one outcome not allowed here."""

    writer = SessionTraceWriter(
        path=None,
        settings=SessionTraceSettings(root=str(tmp_path), max_queue=16),
        session_id="sess-full",
    )
    # A writer with no thread draining it: every emit must still return.
    writer._path = tmp_path / "sink.ndjson"  # type: ignore[attr-defined]
    import queue as _queue

    writer._queue = _queue.Queue(maxsize=2)  # type: ignore[attr-defined]
    for index in range(50):
        writer.event("channel.turn.milestone", {"n": index})
    assert writer.dropped_count == 48


def test_an_unserializable_payload_is_dropped_not_raised(tmp_path: Path) -> None:
    writer = _open(tmp_path)
    writer.event("channel.turn.milestone", {"bad": {1, 2, 3}})
    writer.close()
    rows = _rows(tmp_path)
    # json.dumps falls back to a type name rather than failing the record.
    assert any(r["record_kind"] == "event" for r in rows)


def test_file_ceiling_writes_one_marker_then_stops(tmp_path: Path) -> None:
    writer = _open(tmp_path, settings={"max_file_bytes": 400})
    for index in range(200):
        writer.event("channel.turn.milestone", {"filler": "x" * 200, "n": index})
    writer.close()

    rows = _rows(tmp_path)
    assert [r["record_kind"] for r in rows].count("truncated") == 1
    assert writer.dropped_count > 0


def test_ids_cannot_shape_a_path_outside_the_root(tmp_path: Path) -> None:
    assert _segment("own/../1") == "own-1"
    assert _segment("../../etc/passwd") == "etc-passwd"
    assert _segment("") == "unknown"
    assert len(_segment("a" * 200)) == 64

    path = _resolve_path(
        root=str(tmp_path), session_id="../../x", owner_id="../..", companion_id="c"
    )
    assert path.resolve().is_relative_to(tmp_path.resolve())
    assert len(path.relative_to(tmp_path).parts) == 2


def test_an_unexpanded_variable_disables_tracing_instead_of_making_a_directory() -> None:
    """The repo already carries one directory literally named ``$EIDOLON_CACHE_ROOT``.

    ``os.path.expandvars`` leaves an undefined variable untouched, so a config
    value can reach this code still spelled ``$VAR``. Writing nothing is the
    only acceptable outcome; creating a second such directory is not.
    """

    writer = SessionTraceWriter.open(
        settings=SessionTraceSettings(root="$EIDOLON_LOG_ROOT/channel/traces"),
        session_id="sess-1",
    )
    assert writer.enabled is False
    assert not Path("$EIDOLON_LOG_ROOT").exists()
    writer.session_mark("room_joined")
    writer.close()


def test_retention_drops_whole_expired_day_directories(tmp_path: Path) -> None:
    stale = tmp_path / (date.today() - timedelta(days=30)).isoformat()
    fresh = tmp_path / date.today().isoformat()
    unrelated = tmp_path / "not-a-date"
    for folder in (stale, fresh, unrelated):
        folder.mkdir(parents=True)
        (folder / "keep.ndjson").write_text("{}\n", encoding="utf-8")

    writer = _open(tmp_path, settings={"retention_days": 7})
    writer.close()

    assert not stale.exists()
    assert fresh.exists()
    # A directory whose name is not a date is left alone rather than guessed at.
    assert unrelated.exists()


def test_clock_anchor_now_pairs_both_clocks() -> None:
    anchor = ClockAnchor.now()
    assert anchor.unix_ns > 0
    assert "iso" in anchor.as_dict()


# --------------------------------------------------------------------------- #
# Pipeline wiring: the writer is opened, marked, fed and closed by the base
# pipeline, so both duplex implementations get a session envelope.
# --------------------------------------------------------------------------- #


class _Factory:
    def __init__(self, session_id: str = "sess-wired") -> None:
        self.runtime_session_id = session_id


class _Pipeline:
    """A bare BasePipeline subclass — the shared session-trace surface only."""

    def __init__(self, observability) -> None:
        from eidolon.livekit.agent.shared.pipeline import BasePipeline

        self._factory = _Factory()
        self._observability = observability
        self._session_trace = None
        self._pending_session_marks = []
        self.session_mark = BasePipeline.session_mark.__get__(self)
        self.open_session_trace = BasePipeline.open_session_trace.__get__(self)
        self.close_session_trace = BasePipeline.close_session_trace.__get__(self)


def _observability(tmp_path: Path, **overrides):
    from eidolon.livekit.common.config import ObservabilityConfig

    return ObservabilityConfig(session_trace_path=str(tmp_path), **overrides)


def test_marks_taken_before_the_writer_exists_are_replayed_with_their_own_time(
    tmp_path: Path,
) -> None:
    """``room_joined`` happens before the Owner is known, so it must be buffered.

    Naming the file needs the Owner/Companion pair, which the pipeline only has
    after the runtime participant resolves — but the join it wants to measure
    already happened. Buffering keeps the measurement instead of the ordering.
    """

    pipeline = _Pipeline(_observability(tmp_path))
    pipeline.session_mark("room_joined")
    pipeline.session_mark("warmup_done")
    assert len(pipeline._pending_session_marks) == 2

    pipeline.open_session_trace(None, owner_id="owner-9", companion_id="comp-9")
    pipeline.session_mark("session_started")
    pipeline.close_session_trace("session_ended")

    rows = _rows(tmp_path)
    marks = [r["mark"] for r in rows if r["record_kind"] == RECORD_SESSION_MARK]
    assert marks == ["room_joined", "warmup_done", "session_started"]

    # Replayed marks keep their original ordering by time, not by flush order.
    closing = rows[-1]
    ordered = list(closing["session_marks_ms"])
    assert ordered == ["room_joined", "warmup_done", "session_started"]
    assert closing["missing_session_marks"] == [
        "runtime_participant_resolved",
        "avatar_ready",
        "first_turn",
    ]


def test_the_file_is_named_for_the_owner_companion_and_session(tmp_path: Path) -> None:
    pipeline = _Pipeline(_observability(tmp_path))
    pipeline.open_session_trace(None, owner_id="owner-9", companion_id="comp-9")
    path = pipeline._session_trace.path
    pipeline.close_session_trace("session_ended")

    assert path is not None
    assert path.name == "owner-9__comp-9__sess-wired.ndjson"


def test_tracing_stays_off_when_no_path_is_configured(tmp_path: Path) -> None:
    """The default. A pipeline must behave identically with tracing off."""

    from eidolon.livekit.common.config import ObservabilityConfig

    pipeline = _Pipeline(ObservabilityConfig())
    pipeline.session_mark("room_joined")
    pipeline.open_session_trace(None, owner_id="o", companion_id="c")
    assert pipeline._session_trace is not None
    assert pipeline._session_trace.enabled is False
    pipeline.close_session_trace("session_ended")
    assert list(tmp_path.rglob("*.ndjson")) == []


def test_open_is_idempotent(tmp_path: Path) -> None:
    pipeline = _Pipeline(_observability(tmp_path))
    pipeline.open_session_trace(None, owner_id="owner-1", companion_id="comp-1")
    first = pipeline._session_trace
    pipeline.open_session_trace(None, owner_id="owner-2", companion_id="comp-2")
    assert pipeline._session_trace is first
    pipeline.close_session_trace("session_ended")
    assert len(list(tmp_path.rglob("*.ndjson"))) == 1


def test_close_twice_writes_one_closing_record(tmp_path: Path) -> None:
    """``BasePipeline.shutdown`` closes as a backstop after the lifecycle did."""

    pipeline = _Pipeline(_observability(tmp_path))
    pipeline.open_session_trace(None, owner_id="owner-1", companion_id="comp-1")
    pipeline.close_session_trace("session_ended")
    pipeline.close_session_trace("session_ended")

    rows = _rows(tmp_path)
    assert [r["record_kind"] for r in rows].count(RECORD_SESSION_CLOSE) == 1


# --- the wiring: a device session must reach the writer already attributable ---
#
# `test_channel_turn_events.py` proves the resolution. These prove that each
# pipeline actually hands the resolver over — which is the half that was
# missing, and the half no amount of testing the resolver would have caught:
# every device trace on disk was named `unknown-owner__unknown-companion`
# while the resolver's own tests passed.


def _mounted_factory(companion_id: str, session_id: str = "sess-wired"):
    """A factory whose runtime resolver answers as a mounted body's does."""

    async def resolve_room(room):  # noqa: ANN001 - mirrors ChannelRuntimeServices
        return SimpleNamespace(
            runtime=SimpleNamespace(
                owner_id="owner-mounted",
                companion_id=companion_id,
                device_id="device-1",
            ),
            mount_revision=3,
        )

    return SimpleNamespace(
        runtime_session_id=session_id, runtime_context_resolver=resolve_room
    )


def _device_room():
    """A device token's real metadata shape: no `companion_id` in it."""

    participant = SimpleNamespace(
        identity="device-1",
        metadata=json.dumps(
            {"kind": "device", "device_id": "device-1", "owner_id": "owner-mounted"}
        ),
    )
    return SimpleNamespace(
        name="room-device", remote_participants={"device-1": participant}
    )


@pytest.mark.asyncio
async def test_full_duplex_names_a_device_trace_for_the_mounted_companion(
    tmp_path: Path,
) -> None:
    from eidolon.livekit.agent.full_duplex.lifecycle import FullDuplexSessionLifecycle

    pipeline = _Pipeline(_observability(tmp_path))
    pipeline._factory = _mounted_factory("companion-mounted")
    sink = ChannelTurnEventSink(observer=lambda event: None)
    pipeline._ensure_turn_event_sink = lambda: sink

    lifecycle = FullDuplexSessionLifecycle.__new__(FullDuplexSessionLifecycle)
    lifecycle._pipeline = pipeline

    await lifecycle.begin_session_observation(_device_room())
    pipeline.close_session_trace("session_ended")

    name = sorted(tmp_path.rglob("*.ndjson"))[0].name
    assert name.startswith("owner-mounted__companion-mounted__")
    assert "unknown-owner" not in name


@pytest.mark.asyncio
async def test_ptt_names_a_device_trace_for_the_mounted_companion(
    tmp_path: Path,
) -> None:
    from eidolon.livekit.agent.half_duplex.pipeline import HalfDuplexPttPipeline

    pipeline = _Pipeline(_observability(tmp_path))
    pipeline._factory = _mounted_factory("companion-mounted")
    pipeline._interaction_mode = "ptt"
    pipeline._open_ptt_session_trace = (
        HalfDuplexPttPipeline._open_ptt_session_trace.__get__(pipeline)
    )

    await pipeline._open_ptt_session_trace(_device_room())
    pipeline.close_session_trace("session_ended")

    name = sorted(tmp_path.rglob("*.ndjson"))[0].name
    assert name.startswith("owner-mounted__companion-mounted__")
    assert "unknown-owner" not in name
