"""Input-only PTT delivery port, independent of any conversation scenario.

The capture pipeline owns press/release timing and final ASR. A destination owns
where committed input goes and its output lifecycle. Scenario composition supplies
this port; no solo/PTT pipeline imports a role-group executor or transport.
"""
from typing import Protocol
import asyncio


class PttDestination(Protocol):
    ready: asyncio.Event
    closed: asyncio.Event
    cleanup_ok: bool

    def press(self, capture_id: str) -> None: ...
    def release(self, capture_id: str) -> None: ...
    def transcript(self, capture_id: str, text: str, commitment: dict | None = None) -> None: ...
    def close(self) -> None: ...
