"""Generation-fenced Channel operation ledger and credential state."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from eidolon_sdk.device_foundation.v1 import DeviceRef

from .contracts import IdempotencyConflict, InvalidTransition, StaleGeneration

SCHEMA_VERSION = 4
PROVISION = "channel.provision-device"
REFRESH = "channel.refresh-device"
REVOKE = "channel.revoke-device"


@dataclass(frozen=True, slots=True)
class StoredProvision:
    operation_id: str
    operation_kind: str
    request_fingerprint: str
    device_ref: DeviceRef
    owner_id: str
    manifest_revision: str
    adapter_name: str
    handle_json: str
    channel_id: str
    response_json: str
    expires_at_ms: int
    status: str
    terminal_reason: str | None = None
    created_at_ms: int = 0
    updated_at_ms: int = 0

    @property
    def owner_domain_id(self) -> str:
        return str(self.device_ref.owner_domain_id)

    @property
    def device_id(self) -> str:
        return self.device_ref.device_instance_id


@dataclass(frozen=True, slots=True)
class StoredRevocation:
    operation_id: str
    request_fingerprint: str
    device_ref: DeviceRef
    response_json: str

    @property
    def device_id(self) -> str:
        return self.device_ref.device_instance_id


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
            if version not in {0, 1, 2, 3, SCHEMA_VERSION}:
                raise RuntimeError(
                    f"unsupported Channel Provider database schema version: {version}"
                )
            try:
                connection.execute("BEGIN IMMEDIATE")
                if version in {1, 2, 3}:
                    self._migrate_legacy(connection, version)
                self._create_schema(connection)
                self._validate_schema(connection)
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                connection.commit()
            except Exception as exc:
                connection.rollback()
                raise RuntimeError("Channel Provider schema migration failed closed") from exc
        os.chmod(self._path, 0o600)

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS provider_operations (
                operation_id TEXT NOT NULL,
                operation_kind TEXT NOT NULL CHECK (operation_kind IN (
                    'channel.provision-device', 'channel.refresh-device',
                    'channel.revoke-device'
                )),
                request_fingerprint TEXT NOT NULL,
                device_instance_id TEXT NOT NULL,
                owner_domain_id TEXT NOT NULL,
                owner_domain_generation INTEGER NOT NULL CHECK (owner_domain_generation >= 1),
                claim_generation INTEGER NOT NULL CHECK (claim_generation >= 1),
                trust_epoch INTEGER NOT NULL CHECK (trust_epoch >= 1),
                owner_id TEXT NOT NULL DEFAULT '', manifest_revision TEXT NOT NULL DEFAULT '',
                adapter_name TEXT NOT NULL DEFAULT '', handle_json TEXT NOT NULL DEFAULT '',
                channel_id TEXT NOT NULL DEFAULT '', response_json TEXT NOT NULL,
                expires_at_ms INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL CHECK (status IN (
                    'active', 'expired', 'fenced', 'revoked', 'completed'
                )),
                terminal_reason TEXT, created_at_ms INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL,
                PRIMARY KEY (
                    device_instance_id, owner_domain_id, owner_domain_generation,
                    claim_generation, trust_epoch, operation_kind, operation_id
                )
            )
            """
        )

    @staticmethod
    def _validate_schema(connection: sqlite3.Connection) -> None:
        required = {
            "operation_id",
            "operation_kind",
            "request_fingerprint",
            "device_instance_id",
            "owner_domain_id",
            "owner_domain_generation",
            "claim_generation",
            "trust_epoch",
            "response_json",
            "status",
            "terminal_reason",
        }
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(provider_operations)")
        }
        if not required <= columns:
            missing = ",".join(sorted(required - columns))
            raise RuntimeError(f"provider_operations is missing canonical columns: {missing}")
        connection.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS uq_provider_active_device
            ON provider_operations(owner_domain_id, device_instance_id)
            WHERE status = 'active'"""
        )
        connection.execute(
            """CREATE INDEX IF NOT EXISTS ix_provider_device_generation
            ON provider_operations(owner_domain_id, device_instance_id,
            owner_domain_generation, claim_generation, trust_epoch)"""
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS provider_migration_audit (
                source_schema_version INTEGER NOT NULL, source_table TEXT NOT NULL,
                source_key TEXT NOT NULL, record_json TEXT NOT NULL,
                terminal_reason TEXT NOT NULL, migrated_at_ms INTEGER NOT NULL,
                PRIMARY KEY (source_schema_version, source_table, source_key)
            )
            """
        )

    @staticmethod
    def _migrate_legacy(connection: sqlite3.Connection, version: int) -> None:
        """Fence generation-blind v1-v3 rows into audit-only storage."""
        ChannelProviderStore._create_schema(connection)
        ChannelProviderStore._validate_schema(connection)
        migrated_at_ms = time.time_ns() // 1_000_000
        for table in ("provider_provisions", "provider_revocations"):
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if exists is None:
                raise RuntimeError(f"legacy schema {version} is missing {table}")
            for index, row in enumerate(connection.execute(f"SELECT * FROM {table}")):
                record = dict(row)
                for secret in ("handle_json", "response_json"):
                    if record.get(secret):
                        record[secret] = "<redacted>"
                source_key = str(record.get("operation_id") or index)
                connection.execute(
                    """INSERT INTO provider_migration_audit VALUES
                    (?, ?, ?, ?, 'legacy_generation_unknown_fenced', ?)""",
                    (
                        version,
                        table,
                        source_key,
                        json.dumps(record, sort_keys=True, separators=(",", ":")),
                        migrated_at_ms,
                    ),
                )
        connection.execute("DROP INDEX IF EXISTS uq_provider_active_device")
        connection.execute("DROP TABLE provider_provisions")
        connection.execute("DROP TABLE provider_revocations")

    def healthcheck(self) -> None:
        with self._connect() as connection:
            if connection.execute("SELECT 1").fetchone()[0] != 1:
                raise RuntimeError("Channel Provider database health check failed")

    def expire_credentials(self, now_ms: int) -> int:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = self._expire(connection, now_ms)
            connection.commit()
            return cursor.rowcount

    def operation(
        self, device_ref: DeviceRef, operation_kind: str, operation_id: str
    ) -> StoredProvision | None:
        with self._connect() as connection:
            row = self._select_operation(connection, device_ref, operation_kind, operation_id)
        return self._stored(row)

    def assert_not_stale(self, device_ref: DeviceRef) -> None:
        with self._connect() as connection:
            latest = self._latest_generation(connection, device_ref)
        if latest is not None and self._generation(device_ref) < latest:
            raise StaleGeneration("operation targets an older DeviceRef generation")

    def active_device(self, device_ref: DeviceRef) -> StoredProvision | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM provider_operations
                WHERE device_instance_id=? AND owner_domain_id=?
                AND owner_domain_generation=? AND claim_generation=? AND trust_epoch=?
                AND status='active'""",
                self._ref_values(device_ref),
            ).fetchone()
        return self._stored(row)

    def current_channel(self, device_ref: DeviceRef) -> StoredProvision | None:
        """Latest credential-bearing row for this stable device identity."""
        with self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM provider_operations WHERE owner_domain_id=?
                AND device_instance_id=? AND operation_kind IN (?,?)
                AND status IN ('active','expired')
                ORDER BY owner_domain_generation DESC, claim_generation DESC,
                trust_epoch DESC, created_at_ms DESC LIMIT 1""",
                (
                    str(device_ref.owner_domain_id),
                    device_ref.device_instance_id,
                    PROVISION,
                    REFRESH,
                ),
            ).fetchone()
        return self._stored(row)

    def active_provisions(self) -> list[StoredProvision]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM provider_operations WHERE status='active'"
            ).fetchall()
        return [item for item in map(self._stored, rows) if item is not None]

    def create_provision(
        self, value: StoredProvision, *, now_ms: int
    ) -> tuple[StoredProvision, bool]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._expire(connection, now_ms)
            latest = self._latest_generation(connection, value.device_ref)
            incoming = self._generation(value.device_ref)
            if latest is not None and incoming < latest:
                connection.rollback()
                raise StaleGeneration("operation targets an older DeviceRef generation")
            prior = self._stored(
                self._select_operation(
                    connection, value.device_ref, value.operation_kind, value.operation_id
                )
            )
            if prior is not None:
                if prior.request_fingerprint != value.request_fingerprint:
                    connection.rollback()
                    raise IdempotencyConflict(
                        "operation id was reused with a different canonical payload"
                    )
                connection.commit()
                return prior, True
            same_generation = self._generation_rows(connection, value.device_ref)
            if latest == incoming:
                if value.operation_kind == PROVISION and same_generation:
                    connection.rollback()
                    raise InvalidTransition(
                        "the DeviceRef generation already has a provision lifecycle"
                    )
                if value.operation_kind == REFRESH and not any(
                    row["operation_kind"] in {PROVISION, REFRESH}
                    and (row["status"] == "expired" or (
                        row["status"] == "active"
                        and row["manifest_revision"] != value.manifest_revision
                    ))
                    for row in same_generation
                ):
                    connection.rollback()
                    raise InvalidTransition(
                        "credential refresh requires expiry or a changed Manifest"
                    )
            elif value.operation_kind != PROVISION:
                connection.rollback()
                raise InvalidTransition("a new generation must begin with provision")
            if latest is not None and incoming > latest:
                self._fence_older(connection, value.device_ref, now_ms, "generation_advanced")
            elif value.operation_kind == REFRESH:
                connection.execute(
                    """UPDATE provider_operations SET status='fenced',
                    terminal_reason='credential_refreshed', handle_json='',
                    expires_at_ms=0, updated_at_ms=?
                    WHERE device_instance_id=? AND owner_domain_id=?
                    AND owner_domain_generation=? AND claim_generation=? AND trust_epoch=?
                    AND status IN ('active','expired')""",
                    (now_ms, *self._ref_values(value.device_ref)),
                )
            self._insert_operation(connection, value)
            connection.commit()
            return value, False

    def revocation(self, device_ref: DeviceRef, operation_id: str) -> StoredRevocation | None:
        value = self.operation(device_ref, REVOKE, operation_id)
        if value is None:
            return None
        return StoredRevocation(
            value.operation_id, value.request_fingerprint, value.device_ref, value.response_json
        )

    def complete_revocation(
        self,
        *,
        operation_id: str,
        request_fingerprint: str,
        device_ref: DeviceRef,
        response_json: str,
        now_ms: int,
    ) -> tuple[StoredRevocation, bool]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._expire(connection, now_ms)
            latest = self._latest_generation(connection, device_ref)
            incoming = self._generation(device_ref)
            if latest is not None and incoming < latest:
                connection.rollback()
                raise StaleGeneration("revocation targets an older DeviceRef generation")
            prior = self._stored(
                self._select_operation(connection, device_ref, REVOKE, operation_id)
            )
            if prior is not None:
                if prior.request_fingerprint != request_fingerprint:
                    connection.rollback()
                    raise IdempotencyConflict(
                        "revocation id was reused with a different canonical payload"
                    )
                connection.commit()
                return StoredRevocation(
                    prior.operation_id,
                    prior.request_fingerprint,
                    prior.device_ref,
                    prior.response_json,
                ), True
            if latest is not None and incoming > latest:
                self._fence_older(connection, device_ref, now_ms, "generation_advanced")
            connection.execute(
                """UPDATE provider_operations SET status='revoked',
                terminal_reason='claim_revoked', handle_json='', expires_at_ms=0,
                updated_at_ms=? WHERE device_instance_id=? AND owner_domain_id=?
                AND owner_domain_generation=? AND claim_generation=? AND trust_epoch=?
                AND status IN ('active','expired')""",
                (now_ms, *self._ref_values(device_ref)),
            )
            stored = StoredProvision(
                operation_id,
                REVOKE,
                request_fingerprint,
                device_ref,
                "",
                "",
                "",
                "",
                "",
                response_json,
                0,
                "completed",
                "claim_revoked",
                now_ms,
                now_ms,
            )
            self._insert_operation(connection, stored)
            connection.commit()
            return StoredRevocation(
                operation_id, request_fingerprint, device_ref, response_json
            ), False

    @staticmethod
    def _expire(connection: sqlite3.Connection, now_ms: int) -> sqlite3.Cursor:
        return connection.execute(
            """UPDATE provider_operations SET status='expired',
            terminal_reason='credential_expired', updated_at_ms=?
            WHERE status='active' AND expires_at_ms<=?""",
            (now_ms, now_ms),
        )

    @staticmethod
    def _fence_older(
        connection: sqlite3.Connection, device_ref: DeviceRef, now_ms: int, reason: str
    ) -> None:
        incoming = ChannelProviderStore._generation(device_ref)
        rows = connection.execute(
            """SELECT rowid,* FROM provider_operations WHERE owner_domain_id=?
            AND device_instance_id=? AND status IN ('active','expired')""",
            (str(device_ref.owner_domain_id), device_ref.device_instance_id),
        ).fetchall()
        for row in rows:
            generation = (
                row["owner_domain_generation"],
                row["claim_generation"],
                row["trust_epoch"],
            )
            if generation < incoming:
                connection.execute(
                    """UPDATE provider_operations SET status='fenced', terminal_reason=?,
                    handle_json='', expires_at_ms=0, updated_at_ms=? WHERE rowid=?""",
                    (reason, now_ms, row["rowid"]),
                )

    @staticmethod
    def _latest_generation(
        connection: sqlite3.Connection, device_ref: DeviceRef
    ) -> tuple[int, int, int] | None:
        row = connection.execute(
            """SELECT owner_domain_generation,claim_generation,trust_epoch
            FROM provider_operations WHERE owner_domain_id=? AND device_instance_id=?
            ORDER BY owner_domain_generation DESC,claim_generation DESC,trust_epoch DESC
            LIMIT 1""",
            (str(device_ref.owner_domain_id), device_ref.device_instance_id),
        ).fetchone()
        return None if row is None else (row[0], row[1], row[2])

    @staticmethod
    def _generation_rows(
        connection: sqlite3.Connection, device_ref: DeviceRef
    ) -> list[sqlite3.Row]:
        return connection.execute(
            """SELECT * FROM provider_operations WHERE device_instance_id=?
            AND owner_domain_id=? AND owner_domain_generation=?
            AND claim_generation=? AND trust_epoch=?""",
            ChannelProviderStore._ref_values(device_ref),
        ).fetchall()

    @staticmethod
    def _select_operation(
        connection: sqlite3.Connection,
        device_ref: DeviceRef,
        operation_kind: str,
        operation_id: str,
    ) -> sqlite3.Row | None:
        return connection.execute(
            """SELECT * FROM provider_operations WHERE device_instance_id=?
            AND owner_domain_id=? AND owner_domain_generation=? AND claim_generation=?
            AND trust_epoch=? AND operation_kind=? AND operation_id=?""",
            (*ChannelProviderStore._ref_values(device_ref), operation_kind, operation_id),
        ).fetchone()

    @staticmethod
    def _insert_operation(connection: sqlite3.Connection, value: StoredProvision) -> None:
        connection.execute(
            """INSERT INTO provider_operations (
            operation_id,operation_kind,request_fingerprint,device_instance_id,
            owner_domain_id,owner_domain_generation,claim_generation,trust_epoch,
            owner_id,manifest_revision,adapter_name,handle_json,channel_id,response_json,
            expires_at_ms,status,terminal_reason,created_at_ms,updated_at_ms)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                value.operation_id,
                value.operation_kind,
                value.request_fingerprint,
                *ChannelProviderStore._ref_values(value.device_ref),
                value.owner_id,
                value.manifest_revision,
                value.adapter_name,
                value.handle_json,
                value.channel_id,
                value.response_json,
                value.expires_at_ms,
                value.status,
                value.terminal_reason,
                value.created_at_ms,
                value.updated_at_ms,
            ),
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA secure_delete=ON")
        return connection

    @staticmethod
    def _ref_values(device_ref: DeviceRef) -> tuple[object, ...]:
        return (
            device_ref.device_instance_id,
            str(device_ref.owner_domain_id),
            device_ref.owner_domain_generation,
            device_ref.claim_generation,
            device_ref.trust_epoch,
        )

    @staticmethod
    def _generation(device_ref: DeviceRef) -> tuple[int, int, int]:
        return (
            device_ref.owner_domain_generation,
            device_ref.claim_generation,
            device_ref.trust_epoch,
        )

    @staticmethod
    def _stored(row: sqlite3.Row | None) -> StoredProvision | None:
        if row is None:
            return None
        device_ref = DeviceRef.model_validate(
            {
                "device_instance_id": row["device_instance_id"],
                "owner_domain_id": row["owner_domain_id"],
                "owner_domain_generation": row["owner_domain_generation"],
                "claim_generation": row["claim_generation"],
                "trust_epoch": row["trust_epoch"],
            }
        )
        return StoredProvision(
            row["operation_id"],
            row["operation_kind"],
            row["request_fingerprint"],
            device_ref,
            row["owner_id"],
            row["manifest_revision"],
            row["adapter_name"],
            row["handle_json"],
            row["channel_id"],
            row["response_json"],
            row["expires_at_ms"],
            row["status"],
            row["terminal_reason"],
            row["created_at_ms"],
            row["updated_at_ms"],
        )
