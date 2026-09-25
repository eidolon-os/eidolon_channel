"""Connection-owned shared transport for an authenticated internal caller.

The existing Provider service credential authenticates the calling authority.
That authority must authenticate the business Owner before supplying owner_id;
this is not a Mobile-facing API or a separate business session registry.
"""

from __future__ import annotations

import asyncio
import json

from aiohttp import WSMsgType, web
from pydantic import ValidationError
from eidolon_sdk.biz.control.shared_session import SharedSessionSelection

from .contracts import ContractError, DomainError, ProvisionRequest


def parse_start(value):
    if not isinstance(value, dict) or set(value) != {"selection", "owner_id", "provisions"}:
        raise ValueError("invalid start fields")
    if not isinstance(value["owner_id"], str) or not value["owner_id"].strip():
        raise ValueError("Owner required")
    if not isinstance(value["provisions"], list) or not 2 <= len(value["provisions"]) <= 16:
        raise ValueError("complete provision set required")
    selection = SharedSessionSelection.model_validate(value["selection"])
    provisions = tuple(ProvisionRequest.parse(json.dumps(p).encode()) for p in value["provisions"])
    return selection, provisions, value["owner_id"]


async def shared_transport_socket(request, service):
    """Caller is authenticated by the existing HTTP router before upgrading."""
    ws = web.WebSocketResponse(heartbeat=15, max_msg_size=256 * 1024)
    await ws.prepare(request)
    tasks = []
    try:
        async with asyncio.timeout(10):
            first = await ws.receive()
        if first.type != WSMsgType.TEXT:
            return ws
        selection, provisions, owner = parse_start(json.loads(first.data))

        async def visit():
            async with service.shared_transport(
                selection,
                provisions,
                authenticated_owner_id=owner,
            ) as ready:
                await ws.send_json(ready)
                await asyncio.Event().wait()

        async def until_close():
            message = await ws.receive()
            if message.type == WSMsgType.TEXT and json.loads(message.data) == {"action": "close"}:
                return
            if message.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                return
            raise ValueError("only close is accepted after start")

        visit_task = asyncio.create_task(visit())
        close_task = asyncio.create_task(until_close())
        tasks = [visit_task, close_task]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        if visit_task in done:
            # Revocation/refresh cancels this task; do not advertise clean close
            # before observing its cleanup result.
            if visit_task.cancelled():
                await ws.send_json({"state": "ended", "reason": "lifecycle_changed"})
            else:
                await visit_task
        else:
            await close_task
            visit_task.cancel()
            try:
                await visit_task
            except asyncio.CancelledError:
                pass
            if not ws.closed:
                await ws.send_json({"state": "closed", "session_id": selection.session_id})
    except (ValueError, ValidationError, ContractError):
        if not ws.closed:
            await ws.send_json({"state": "error", "code": "INVALID_CONTRACT", "retryable": False})
    except DomainError as exc:
        if not ws.closed:
            await ws.send_json({"state": "error", "code": exc.code, "retryable": exc.retryable})
    except TimeoutError:
        if not ws.closed:
            await ws.send_json({"state": "ended", "reason": "deadline"})
    finally:
        # On disconnect or handler cancellation, await resource cleanup. Never
        # detach the visit and leave it running after its connection has gone.
        for task in tasks:
            if not task.done() and not task.cancelling():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await ws.close()
    return ws
