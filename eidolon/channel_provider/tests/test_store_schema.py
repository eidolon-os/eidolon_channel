"""Coming up on a database written before a channel was one thing.

A Provider that refuses to start is not a cautious Provider — it is a Host with
no device channels at all, and this one refused on a real deployment: the store
accepted schema 0 or 2 and the running Host was on 1.

The schema it was on is reproduced here exactly as the deployment had it, so
the case that actually happened is the case under test.
"""

from __future__ import annotations

import sqlite3

import pytest

from eidolon.channel_provider.store import ChannelProviderStore

# Verbatim from the Host that refused to start.
_TWO_ROOM_SCHEMA = """
CREATE TABLE provider_provisions (
    operation_id TEXT PRIMARY KEY,
    request_fingerprint TEXT NOT NULL,
    owner_domain_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    manifest_revision TEXT NOT NULL,
    active_room TEXT NOT NULL,
    control_room TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    response_json TEXT NOT NULL,
    expires_at_ms INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'revoked'))
);
CREATE UNIQUE INDEX uq_provider_active_device
ON provider_provisions(owner_domain_id, device_id)
WHERE status = 'active';
CREATE TABLE provider_revocations (
    operation_id TEXT PRIMARY KEY,
    request_fingerprint TEXT NOT NULL,
    device_id TEXT NOT NULL,
    response_json TEXT NOT NULL
);
PRAGMA user_version = 1;
"""


def _two_room_database(path, *, provisions: int = 3, revocations: int = 4) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(_TWO_ROOM_SCHEMA)
    for index in range(provisions):
        connection.execute(
            "INSERT INTO provider_provisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                f"op-{index}", "sha256:x", "hub-1", f"device-{index}", "owner-1",
                "sha256:m", f"voice-{index}", f"control-{index}", f"chan-{index}",
                '{"channels":[{"binding_format":"...livekit-device+json;v=1"}]}',
                1_700_000_000_000, "active",
            ),
        )
    for index in range(revocations):
        connection.execute(
            "INSERT INTO provider_revocations VALUES (?,?,?,?)",
            (f"rev-{index}", "sha256:y", f"device-{index}", '{"operation":"..."}'),
        )
    connection.commit()
    connection.close()


def test_a_two_room_database_does_not_stop_the_provider_starting(tmp_path) -> None:
    path = tmp_path / "provider.sqlite3"
    _two_room_database(path)

    ChannelProviderStore(path).initialize()

    connection = sqlite3.connect(path)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(provider_provisions)")
    }
    assert "handle_json" in columns and "adapter_name" in columns
    assert "active_room" not in columns and "control_room" not in columns
    connection.close()


def test_provisions_that_cannot_be_honoured_are_let_go(tmp_path) -> None:
    """Replaying one would hand a device a channel shaped like nothing served."""
    path = tmp_path / "provider.sqlite3"
    _two_room_database(path)

    ChannelProviderStore(path).initialize()

    connection = sqlite3.connect(path)
    assert connection.execute("SELECT COUNT(*) FROM provider_provisions").fetchone()[0] == 0
    connection.close()


def test_a_device_that_was_cut_off_stays_cut_off(tmp_path) -> None:
    """Revocations did not change shape, so a schema change must not undo one."""
    path = tmp_path / "provider.sqlite3"
    _two_room_database(path)

    ChannelProviderStore(path).initialize()

    connection = sqlite3.connect(path)
    assert connection.execute("SELECT COUNT(*) FROM provider_revocations").fetchone()[0] == 4
    connection.close()


def test_starting_twice_on_a_carried_over_database_is_stable(tmp_path) -> None:
    path = tmp_path / "provider.sqlite3"
    _two_room_database(path)
    store = ChannelProviderStore(path)

    store.initialize()
    store.initialize()

    assert store.active_provisions() == []


def test_a_schema_from_the_future_still_stops_the_provider(tmp_path) -> None:
    """Only the version this store knows how to leave behind is left behind."""
    path = tmp_path / "provider.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript("PRAGMA user_version = 99;")
    connection.close()

    with pytest.raises(RuntimeError, match="unsupported"):
        ChannelProviderStore(path).initialize()
