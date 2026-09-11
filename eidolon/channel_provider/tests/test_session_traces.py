"""The session-trace read surface: a directory reader and two GETs.

Traces are written by the Agent worker and served here because both are the
Channel authority on one Host. These tests build the files the worker would
have written rather than running it, so the reader's contract is pinned
independently of the pipeline that fills the directory.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from eidolon.channel_provider.http import create_app
from eidolon.channel_provider.selection import AdapterRegistry
from eidolon.channel_provider.service import ChannelProviderService
from eidolon.channel_provider.session_traces import SessionTraceReader, TraceQuery
from eidolon.channel_provider.store import ChannelProviderStore

from .helpers import FakeAdapter

_TOKEN = "s" * 32
_HEADERS = {"Authorization": f"Bearer {_TOKEN}"}


def _write_trace(
    root: Path,
    *,
    session_id: str,
    owner_id: str = "owner-1",
    companion_id: str = "companion-1",
    day: str | None = None,
    unix_ns: int = 1_789_000_000_000_000_000,
    closed: bool = True,
    turns: int = 1,
    reason: str = "session_ended",
) -> Path:
    """Write one trace the way ``SessionTraceWriter`` does."""

    folder = root / (day or date.today().isoformat())
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{owner_id}__{companion_id}__{session_id}.ndjson"
    rows: list[dict] = [
        {
            "record_kind": "session_open",
            "session_id": session_id,
            "seq": 1,
            "t_mono": 100.0,
            "clock_anchor": {
                "monotonic": 100.0,
                "unix_ns": unix_ns,
                "iso": "2026-09-11T01:00:00+08:00",
            },
            "owner_id": owner_id,
            "companion_id": companion_id,
            "room_name": "room-1",
            "interaction_mode": "full_duplex",
            "pid": 4242,
        },
        {
            "record_kind": "session_mark",
            "session_id": session_id,
            "seq": 2,
            "mark": "room_joined",
            "since_open_ms": 3.1,
        },
    ]
    seq = 3
    for index in range(turns):
        rows.append(
            {
                "record_kind": "event",
                "session_id": session_id,
                "seq": seq,
                "event_type": "channel.turn.milestone",
                "channel_turn_id": f"turn-{index}",
                "milestone": "first_audio",
                "elapsed_ms": 860.0 + index,
            }
        )
        seq += 1
        rows.append(
            {
                "record_kind": "turn_final",
                "session_id": session_id,
                "seq": seq,
                "turn_id": f"turn-{index}",
                "final": True,
                "record_reason": "agent_audio_playback_done",
                "timestamps": {"speech_started_at": 1.0, "tts_first_audio_at": 1.86},
                "clock_anchor": {"monotonic": 100.0, "unix_ns": unix_ns},
            }
        )
        seq += 1
    if closed:
        rows.append(
            {
                "record_kind": "session_close",
                "session_id": session_id,
                "seq": seq,
                "reason": reason,
                "duration_ms": 3173.1,
                "session_marks_ms": {"room_joined": 3.1},
                "missing_session_marks": ["avatar_ready"],
                "written_record_count": seq - 1,
                "dropped_record_count": 0,
                "truncated": False,
            }
        )
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


# --------------------------------------------------------------------------- #
# Reader
# --------------------------------------------------------------------------- #


def test_no_root_is_an_empty_answer_not_an_error() -> None:
    """Tracing off is the default, so "none recorded" must be sayable."""

    reader = SessionTraceReader(None)
    assert reader.available is False
    assert reader.list(TraceQuery()) == []
    assert reader.read("sess-1") is None


def test_a_missing_directory_reads_as_empty(tmp_path: Path) -> None:
    reader = SessionTraceReader(tmp_path / "never-created")
    assert reader.available is False
    assert reader.list(TraceQuery()) == []


def test_a_summary_comes_from_the_open_and_close_records(tmp_path: Path) -> None:
    """Two lines, not the file: that is what keeps listing cheap."""

    _write_trace(tmp_path, session_id="sess-a", turns=3)
    summaries = SessionTraceReader(tmp_path).list(TraceQuery())

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["session_id"] == "sess-a"
    assert summary["owner_id"] == "owner-1"
    assert summary["companion_id"] == "companion-1"
    assert summary["interaction_mode"] == "full_duplex"
    assert summary["status"] == "closed"
    assert summary["reason"] == "session_ended"
    assert summary["duration_ms"] == 3173.1
    assert summary["session_marks_ms"] == {"room_joined": 3.1}
    assert summary["missing_session_marks"] == ["avatar_ready"]
    assert summary["dropped_record_count"] == 0
    assert summary["trace_bytes"] > 0


def test_a_session_with_no_close_record_reads_as_open(tmp_path: Path) -> None:
    """A worker killed mid-session leaves no close line. That is not "closed"."""

    _write_trace(tmp_path, session_id="sess-open", closed=False)
    summary = SessionTraceReader(tmp_path).list(TraceQuery())[0]

    assert summary["status"] == "open"
    assert summary["reason"] is None
    assert summary["duration_ms"] is None


def test_listing_filters_on_the_record_not_the_filename(tmp_path: Path) -> None:
    """Filenames are sanitized; a conversation id may contain ``.`` or ``:``.

    Matching the label would silently miss exactly those sessions, so the
    filter reads the identity written inside the file.
    """

    root = tmp_path
    folder = root / date.today().isoformat()
    folder.mkdir(parents=True)
    # The writer would have collapsed ``.``/``:`` out of the filename while the
    # record keeps the real id.
    path = folder / "owner-1__companion-1__sess-a-b-c.ndjson"
    path.write_text(
        json.dumps(
            {
                "record_kind": "session_open",
                "session_id": "sess.a:b-c",
                "seq": 1,
                "clock_anchor": {"monotonic": 1.0, "unix_ns": 1, "iso": "x"},
                "owner_id": "owner/1",
                "companion_id": "companion-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    reader = SessionTraceReader(root)

    assert reader.list(TraceQuery(owner_id="owner/1"))[0]["session_id"] == "sess.a:b-c"
    assert reader.list(TraceQuery(owner_id="owner-1")) == []
    assert reader.read("sess.a:b-c") is not None
    assert reader.read("sess-a-b-c") is None


def test_listing_filters_by_owner_companion_and_start_time(tmp_path: Path) -> None:
    old = int(datetime(2026, 9, 1, tzinfo=None).astimezone().timestamp() * 1e9)
    new = int(datetime(2026, 9, 10, tzinfo=None).astimezone().timestamp() * 1e9)
    _write_trace(tmp_path, session_id="s-old", day="2026-09-01", unix_ns=old)
    _write_trace(tmp_path, session_id="s-new", day="2026-09-10", unix_ns=new)
    _write_trace(tmp_path, session_id="s-other", owner_id="owner-2", unix_ns=new)
    reader = SessionTraceReader(tmp_path)

    assert {s["session_id"] for s in reader.list(TraceQuery())} == {"s-old", "s-new", "s-other"}
    assert {s["session_id"] for s in reader.list(TraceQuery(owner_id="owner-2"))} == {"s-other"}
    assert {s["session_id"] for s in reader.list(TraceQuery(companion_id="nope"))} == set()

    since = datetime(2026, 9, 5).astimezone()
    assert "s-old" not in {s["session_id"] for s in reader.list(TraceQuery(since=since))}


def test_listing_is_newest_day_first_and_respects_limit(tmp_path: Path) -> None:
    for day in ("2026-09-01", "2026-09-05", "2026-09-09"):
        _write_trace(tmp_path, session_id=f"s-{day}", day=day)
    reader = SessionTraceReader(tmp_path)

    ordered = [s["session_id"] for s in reader.list(TraceQuery())]
    assert ordered == ["s-2026-09-09", "s-2026-09-05", "s-2026-09-01"]
    assert len(reader.list(TraceQuery(limit=2))) == 2
    # An absurd limit falls back to the default rather than being honoured.
    assert len(reader.list(TraceQuery(limit=10_000))) == 3


def test_a_directory_that_is_not_a_date_is_left_alone(tmp_path: Path) -> None:
    stray = tmp_path / "not-a-date"
    stray.mkdir()
    (stray / "x.ndjson").write_text('{"record_kind":"session_open"}\n', encoding="utf-8")
    _write_trace(tmp_path, session_id="s-real")

    assert [s["session_id"] for s in SessionTraceReader(tmp_path).list(TraceQuery())] == ["s-real"]


def test_reading_one_session_returns_its_records_in_order(tmp_path: Path) -> None:
    _write_trace(tmp_path, session_id="sess-a", turns=2)
    found = SessionTraceReader(tmp_path).read("sess-a")

    assert found is not None
    assert found["session"]["session_id"] == "sess-a"
    kinds = [record["record_kind"] for record in found["records"]]
    assert kinds[0] == "session_open"
    assert kinds[-1] == "session_close"
    assert kinds.count("turn_final") == 2
    assert found["record_count"] == len(found["records"])
    assert [record["seq"] for record in found["records"]] == sorted(
        record["seq"] for record in found["records"]
    )


def test_reading_can_narrow_to_the_kinds_a_caller_needs(tmp_path: Path) -> None:
    _write_trace(tmp_path, session_id="sess-a", turns=2)
    found = SessionTraceReader(tmp_path).read("sess-a", kinds=frozenset({"turn_final"}))

    assert found is not None
    assert {record["record_kind"] for record in found["records"]} == {"turn_final"}
    assert found["record_count"] == 2
    # The summary still comes from open/close even when they are filtered out.
    assert found["session"]["status"] == "closed"


def test_an_unparsable_line_does_not_lose_the_session(tmp_path: Path) -> None:
    """The writer appends while this reads, so a torn last line is expected."""

    path = _write_trace(tmp_path, session_id="sess-a")
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"record_kind": "turn_fin')

    found = SessionTraceReader(tmp_path).read("sess-a")
    assert found is not None
    assert found["session"]["status"] == "closed"
    assert all(isinstance(record, dict) for record in found["records"])


def test_a_file_whose_first_line_is_not_an_open_record_is_skipped(tmp_path: Path) -> None:
    folder = tmp_path / date.today().isoformat()
    folder.mkdir(parents=True)
    (folder / "o__c__s.ndjson").write_text('{"record_kind":"turn_final"}\n', encoding="utf-8")

    assert SessionTraceReader(tmp_path).list(TraceQuery()) == []


# --------------------------------------------------------------------------- #
# HTTP surface
# --------------------------------------------------------------------------- #


@pytest.fixture
def service(tmp_path: Path) -> ChannelProviderService:
    service = ChannelProviderService(
        store=ChannelProviderStore(tmp_path / "provider.sqlite3"),
        registry=AdapterRegistry([FakeAdapter(name="livekit")], preference=("livekit",)),
        agent_name="eidolon",
        now_ms=lambda: 1_700_000_000_000,
    )
    # ``create_app``'s startup hook resumes the channels this Provider granted,
    # which needs its schema. The trace reads do not touch it — that is what
    # ``test_the_reads_do_not_disturb_the_provisioning_surface`` is for — but
    # the app they are mounted on still comes up the production way.
    service.initialize()
    return service


async def _client(service: ChannelProviderService, root: Path | None) -> TestClient:
    app = create_app(
        service=service,
        bearer_token=_TOKEN,
        traces=SessionTraceReader(root),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


@pytest.mark.asyncio
async def test_both_reads_require_the_bearer(service, tmp_path: Path) -> None:
    client = await _client(service, tmp_path)
    try:
        for path in ("/v1/session-traces", "/v1/session-traces/sess-a"):
            response = await client.get(path)
            assert response.status == 401
            assert response.content_type == "application/problem+json"
            assert (await response.json())["code"] == "UNAUTHENTICATED"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_listing_reports_whether_anything_is_recorded(service, tmp_path: Path) -> None:
    client = await _client(service, tmp_path / "absent")
    try:
        response = await client.get("/v1/session-traces", headers=_HEADERS)
        assert response.status == 200
        body = await response.json()
        assert body["operation"] == "channel.session-traces"
        assert body["recording"] is False
        assert body["sessions"] == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_listing_serves_summaries_and_filters(service, tmp_path: Path) -> None:
    _write_trace(tmp_path, session_id="sess-a")
    _write_trace(tmp_path, session_id="sess-b", owner_id="owner-2")
    client = await _client(service, tmp_path)
    try:
        response = await client.get("/v1/session-traces", headers=_HEADERS)
        body = await response.json()
        assert body["recording"] is True
        assert {s["session_id"] for s in body["sessions"]} == {"sess-a", "sess-b"}

        filtered = await client.get(
            "/v1/session-traces", params={"owner_id": "owner-2"}, headers=_HEADERS
        )
        assert {s["session_id"] for s in (await filtered.json())["sessions"]} == {"sess-b"}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_a_malformed_filter_is_refused_not_ignored(service, tmp_path: Path) -> None:
    """A dropped filter would let a caller read the unfiltered answer as filtered."""

    client = await _client(service, tmp_path)
    try:
        for params in ({"limit": "many"}, {"since": "last-tuesday"}):
            response = await client.get(
                "/v1/session-traces", params=params, headers=_HEADERS
            )
            assert response.status == 422
            assert (await response.json())["code"] == "INVALID_ARGUMENT"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_listing_without_a_limit_uses_the_default(service, tmp_path: Path) -> None:
    _write_trace(tmp_path, session_id="sess-a")
    client = await _client(service, tmp_path)
    try:
        response = await client.get("/v1/session-traces", headers=_HEADERS)
        assert response.status == 200
        assert len((await response.json())["sessions"]) == 1
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_one_session_is_served_whole_and_can_be_narrowed(service, tmp_path: Path) -> None:
    _write_trace(tmp_path, session_id="sess-a", turns=2)
    client = await _client(service, tmp_path)
    try:
        response = await client.get("/v1/session-traces/sess-a", headers=_HEADERS)
        assert response.status == 200
        body = await response.json()
        assert body["operation"] == "channel.session-trace"
        assert body["session"]["session_id"] == "sess-a"
        assert body["record_count"] == len(body["records"])

        narrowed = await client.get(
            "/v1/session-traces/sess-a", params={"kinds": "turn_final,event"}, headers=_HEADERS
        )
        kinds = {r["record_kind"] for r in (await narrowed.json())["records"]}
        assert kinds == {"turn_final", "event"}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_an_unknown_session_is_a_problem_document(service, tmp_path: Path) -> None:
    client = await _client(service, tmp_path)
    try:
        response = await client.get("/v1/session-traces/nope", headers=_HEADERS)
        assert response.status == 404
        assert response.content_type == "application/problem+json"
        body = await response.json()
        assert body["code"] == "NOT_FOUND"
        assert body["authority"] == "eidolon-channel-provider"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_the_reads_do_not_disturb_the_provisioning_surface(service, tmp_path: Path) -> None:
    """Serving traces must not need or touch the provider's own state."""

    client = await _client(service, tmp_path)
    try:
        health = await client.get("/health")
        assert health.status == 200
        listed = await client.get("/v1/session-traces", headers=_HEADERS)
        assert listed.status == 200
        assert (await (await client.get("/health")).json())["status"] == "ok"
    finally:
        await client.close()


def test_retention_window_is_the_writer_s_job_not_the_reader_s(tmp_path: Path) -> None:
    """The reader shows what is there; it never deletes.

    Pruning belongs to the writer, which does it when a session opens. A reader
    that also pruned would make reading a mutation.
    """

    old_day = (date.today() - timedelta(days=400)).isoformat()
    _write_trace(tmp_path, session_id="s-ancient", day=old_day)
    reader = SessionTraceReader(tmp_path)

    assert [s["session_id"] for s in reader.list(TraceQuery())] == ["s-ancient"]
    assert (tmp_path / old_day).exists()
