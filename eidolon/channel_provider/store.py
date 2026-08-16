"""Provider-owned idempotency and credential state.

Hub deliberately never persists opaque bindings. This SQLite file is therefore
owned exclusively by the Channel Provider, created with restrictive filesystem
permissions, and securely clears cached token responses on revocation.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class StoredProvision:
    operation_id: str
    request_fingerprint: str
    hub_id: str
    device_id: str
    owner_id: str
    manifest_revision: str
    # Which adapter opened this channel, and its own opaque handle for it.
    # No layer above the adapter may interpret handle_json.
    adapter_name: str
    handle_json: str
    channel_id: str
    response_json: str
    expires_at_ms: int
    status: str


@dataclass(frozen=True, slots=True)
class StoredRevocation:
    operation_id: str
    request_fingerprint: str
    device_id: str
    response_json: str


class ChannelProviderStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self._path.parent, 0o700)
        except PermissionError:
            pass
        with self._connect() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in {0, 2}:
                raise RuntimeError(
                    f"unsupported Channel Provider database schema version: {version}"
                )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS provider_provisions (
                    operation_id TEXT PRIMARY KEY,
                    request_fingerprint TEXT NOT NULL,
                    hub_id TEXT NOT NULL,
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
                CREATE UNIQUE INDEX IF NOT EXISTS uq_provider_active_device
                ON provider_provisions(hub_id, device_id)
                WHERE status = 'active';
                CREATE TABLE IF NOT EXISTS provider_revocations (
                    operation_id TEXT PRIMARY KEY,
                    request_fingerprint TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    response_json TEXT NOT NULL
                );
                PRAGMA user_version = 2;
                """
            )
        os.chmod(self._path, 0o600)

    def healthcheck(self) -> None:
        with self._connect() as connection:
            value = connection.execute("SELECT 1").fetchone()[0]
        if value != 1:
            raise RuntimeError("Channel Provider database health check failed")

    def provision(self, operation_id: str) -> StoredProvision | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM provider_provisions WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
        return self._provision(row)

    def active_device(self, hub_id: str, device_id: str) -> StoredProvision | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM provider_provisions
                WHERE hub_id = ? AND device_id = ? AND status = 'active'
                """,
                (hub_id, device_id),
            ).fetchone()
        return self._provision(row)

    def create_provision(self, value: StoredProvision) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO provider_provisions (
                    operation_id, request_fingerprint, hub_id, device_id, owner_id,
                    manifest_revision, adapter_name, handle_json, channel_id,
                    response_json, expires_at_ms, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    value.operation_id,
                    value.request_fingerprint,
                    value.hub_id,
                    value.device_id,
                    value.owner_id,
                    value.manifest_revision,
                    value.adapter_name,
                    value.handle_json,
                    value.channel_id,
                    value.response_json,
                    value.expires_at_ms,
                    value.status,
                ),
            )
            connection.commit()

    def refresh_provision(
        self,
        *,
        operation_id: str,
        request_fingerprint: str,
        handle_json: str,
        response_json: str,
        expires_at_ms: int,
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE provider_provisions
                SET handle_json = ?, response_json = ?, expires_at_ms = ?
                WHERE operation_id = ? AND request_fingerprint = ? AND status = 'active'
                """,
                (handle_json, response_json, expires_at_ms, operation_id, request_fingerprint),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise RuntimeError("provision authority changed during refresh")
            connection.commit()

    def revocation(self, operation_id: str) -> StoredRevocation | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM provider_revocations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
        if row is None:
            return None
        return StoredRevocation(
            operation_id=row["operation_id"],
            request_fingerprint=row["request_fingerprint"],
            device_id=row["device_id"],
            response_json=row["response_json"],
        )

    def complete_revocation(
        self,
        *,
        operation_id: str,
        request_fingerprint: str,
        hub_id: str,
        device_id: str,
        response_json: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE provider_provisions
                SET status = 'revoked', response_json = '', expires_at_ms = 0
                WHERE hub_id = ? AND device_id = ? AND status = 'active'
                """,
                (hub_id, device_id),
            )
            connection.execute(
                """
                INSERT INTO provider_revocations (
                    operation_id, request_fingerprint, device_id, response_json
                ) VALUES (?, ?, ?, ?)
                """,
                (operation_id, request_fingerprint, device_id, response_json),
            )
            connection.commit()
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA secure_delete = ON")
        return connection

    @staticmethod
    def _provision(row: sqlite3.Row | None) -> StoredProvision | None:
        if row is None:
            return None
        return StoredProvision(
            operation_id=row["operation_id"],
            request_fingerprint=row["request_fingerprint"],
            hub_id=row["hub_id"],
            device_id=row["device_id"],
            owner_id=row["owner_id"],
            manifest_revision=row["manifest_revision"],
            adapter_name=row["adapter_name"],
            handle_json=row["handle_json"],
            channel_id=row["channel_id"],
            response_json=row["response_json"],
            expires_at_ms=row["expires_at_ms"],
            status=row["status"],
        )
