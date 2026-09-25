"""Admission barrier for one explicitly selected shared session.

The orchestrator owns this object alongside the temporary room. Callers must
verify ACK transport identity and provide participant observations from that
room. This object neither authenticates devices nor grants execution authority.
It stores invitation correlation only, never Companion bindings or credentials.
"""

from collections.abc import Mapping, Set

from eidolon_sdk.biz.control import command_status_from_ack


class SharedAdmission:
    def __init__(self, invitations: Mapping[str, str], *, deadline_ms: int) -> None:
        if len(invitations) < 2 or not all(invitations) or not all(invitations.values()):
            raise ValueError("at least two devices with invitation IDs are required")
        if len(set(invitations.values())) != len(invitations):
            raise ValueError("invitation IDs must be unique")
        if type(deadline_ms) is not int or deadline_ms <= 0:
            raise ValueError("a positive deadline is required")
        self._invitations = dict(invitations)
        self._accepted: set[str] = set()
        self._present: set[str] = set()
        self._failed: set[str] = set()
        self._deadline_ms = deadline_ms
        self._closed = False

    def acknowledge(self, device_id: str, command_id: str, status: str) -> None:
        if self._closed or self._invitations.get(device_id) != command_id:
            return
        normalized = command_status_from_ack(status)
        if normalized in {"accepted", "running", "succeeded"}:
            if device_id not in self._failed:
                self._accepted.add(device_id)
        else:
            self._failed.add(device_id)
            self._accepted.discard(device_id)

    def observe_members(self, device_ids: Set[str]) -> None:
        # A complete snapshot replaces the last observation; extra participants
        # (including the worker) never stand in for a selected device.
        if not self._closed:
            self._present = set(device_ids).intersection(self._invitations)

    @property
    def failed_devices(self) -> frozenset[str]:
        return frozenset(self._failed)

    def ready(self, *, now_ms: int) -> bool:
        if now_ms >= self._deadline_ms:
            self.close()
        return (
            not self._closed
            and not self._failed
            and self._accepted == self._present == self._invitations.keys()
        )

    def close(self) -> None:
        self._closed = True
        self._accepted.clear()
        self._present.clear()
