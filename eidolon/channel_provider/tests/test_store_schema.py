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

_V3_SCHEMA = """
CREATE TABLE provider_provisions (
    operation_id TEXT PRIMARY KEY,
    request_fingerprint TEXT NOT NULL,
    owner_domain_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    manifest_revision TEXT NOT NULL,
    adapter_name TEXT NOT NULL,
    handle_json TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    response_json TEXT NOT NULL,
    expires_at_ms INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'revoked'))
);
CREATE UNIQUE INDEX uq_provider_active_device
ON provider_provisions(owner_domain_id, device_id) WHERE status = 'active';
CREATE TABLE provider_revocations (
    operation_id TEXT PRIMARY KEY,
    request_fingerprint TEXT NOT NULL,
    device_id TEXT NOT NULL,
    response_json TEXT NOT NULL
);
PRAGMA user_version = 3;
"""


def _two_room_database(path, *, provisions: int = 3, revocations: int = 4) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(_TWO_ROOM_SCHEMA)
    for index in range(provisions):
        connection.execute(
            "INSERT INTO provider_provisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                f"op-{index}",
                "sha256:x",
                "hub-1",
                f"device-{index}",
                "owner-1",
                "sha256:m",
                f"voice-{index}",
                f"control-{index}",
                f"chan-{index}",
                '{"channels":[{"binding_format":"...livekit-device+json;v=1"}]}',
                1_700_000_000_000,
                "active",
            ),
        )
    for index in range(revocations):
        connection.execute(
            "INSERT INTO provider_revocations VALUES (?,?,?,?)",
            (f"rev-{index}", "sha256:y", f"device-{index}", '{"operation":"..."}'),
        )
    connection.commit()
    connection.close()


def _v3_database(path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(_V3_SCHEMA)
    connection.execute(
        "INSERT INTO provider_provisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "legacy-op",
            "sha256:legacy",
            "owner-domain-1",
            "device-1",
            "owner_1",
            "manifest-1",
            "livekit",
            '{"room":"legacy","token":"secret"}',
            "chan-legacy",
            '{"opaque_binding":"secret"}',
            1_700_000_000_000,
            "active",
        ),
    )
    connection.execute(
        "INSERT INTO provider_revocations VALUES (?,?,?,?)",
        ("legacy-revoke", "sha256:revoke", "device-2", '{"token":"secret"}'),
    )
    connection.commit()
    connection.close()


def test_a_two_room_database_does_not_stop_the_provider_starting(tmp_path) -> None:
    path = tmp_path / "provider.sqlite3"
    _two_room_database(path)

    ChannelProviderStore(path).initialize()

    connection = sqlite3.connect(path)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
    columns = {row[1] for row in connection.execute("PRAGMA table_info(provider_operations)")}
    assert {
        "device_instance_id",
        "owner_domain_id",
        "owner_domain_generation",
        "claim_generation",
        "trust_epoch",
    } <= columns
    assert "device_id" not in columns
    connection.close()


def test_generation_blind_provisions_become_terminal_migration_audit(tmp_path) -> None:
    path = tmp_path / "provider.sqlite3"
    _two_room_database(path)

    ChannelProviderStore(path).initialize()

    connection = sqlite3.connect(path)
    assert connection.execute("SELECT COUNT(*) FROM provider_operations").fetchone()[0] == 0
    rows = connection.execute(
        "SELECT record_json, terminal_reason FROM provider_migration_audit "
        "WHERE source_table='provider_provisions'"
    ).fetchall()
    assert len(rows) == 3
    assert {row[1] for row in rows} == {"legacy_generation_unknown_fenced"}
    assert all("<redacted>" in row[0] for row in rows)
    connection.close()


def test_legacy_revocations_remain_terminal_audit_without_runtime_fallback(tmp_path) -> None:
    path = tmp_path / "provider.sqlite3"
    _two_room_database(path)

    ChannelProviderStore(path).initialize()

    connection = sqlite3.connect(path)
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM provider_migration_audit "
            "WHERE source_table='provider_revocations'"
        ).fetchone()[0]
        == 4
    )
    assert connection.execute("SELECT COUNT(*) FROM provider_operations").fetchone()[0] == 0
    connection.close()


def test_starting_twice_on_a_carried_over_database_is_stable(tmp_path) -> None:
    path = tmp_path / "provider.sqlite3"
    _two_room_database(path)
    store = ChannelProviderStore(path)

    store.initialize()
    store.initialize()

    assert store.active_provisions() == []


def test_v3_migration_preserves_audit_but_activates_no_generation_blind_row(tmp_path) -> None:
    path = tmp_path / "provider.sqlite3"
    _v3_database(path)

    store = ChannelProviderStore(path)
    store.initialize()

    connection = sqlite3.connect(path)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
    assert connection.execute("SELECT COUNT(*) FROM provider_operations").fetchone()[0] == 0
    audit = connection.execute(
        "SELECT record_json, terminal_reason FROM provider_migration_audit ORDER BY source_table"
    ).fetchall()
    assert len(audit) == 2
    assert all(row[1] == "legacy_generation_unknown_fenced" for row in audit)
    assert all("secret" not in row[0] for row in audit)
    connection.close()
    assert store.active_provisions() == []


def test_broken_v3_migration_fails_closed_and_does_not_advance_version(tmp_path) -> None:
    path = tmp_path / "provider.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA user_version = 3")
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="failed closed"):
        ChannelProviderStore(path).initialize()

    connection = sqlite3.connect(path)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name='provider_operations'"
        ).fetchone()[0]
        == 0
    )
    connection.close()


def test_corrupt_current_schema_fails_closed_instead_of_running_partially(tmp_path) -> None:
    path = tmp_path / "provider.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE provider_operations (operation_id TEXT)")
    connection.execute("PRAGMA user_version = 4")
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="failed closed"):
        ChannelProviderStore(path).initialize()

    connection = sqlite3.connect(path)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
    columns = {row[1] for row in connection.execute("PRAGMA table_info(provider_operations)")}
    assert columns == {"operation_id"}
    connection.close()


def test_a_schema_from_the_future_still_stops_the_provider(tmp_path) -> None:
    """Only the version this store knows how to leave behind is left behind."""
    path = tmp_path / "provider.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript("PRAGMA user_version = 99;")
    connection.close()

    with pytest.raises(RuntimeError, match="unsupported"):
        ChannelProviderStore(path).initialize()
