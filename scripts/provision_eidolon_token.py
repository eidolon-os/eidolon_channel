"""Provision a long-lived device_token for talking to ``EidolonAgent.Chat``.

Steps:

    1. POST /api/admin/pairing/codes (admin HTTP) to issue a pairing code
    2. gRPC ExchangePairingCode (no auth needed) -> device_token JWT
    3. Print the token; operator pastes into ``REMOTE_AGENT_RPC_DEVICE_TOKEN``

Run from the eidolon_channel repository root::

    python scripts/provision_eidolon_token.py \\
        --admin-base-url http://127.0.0.1:8081 \\
        --grpc-target 127.0.0.1:50052 \\
        --tenant-id demo --user-id alice \\
        --device-name livekit-dev

Defaults assume eidolon_agent's stock dev config (admin HTTP on :8081,
gRPC on :50052 — note 50052, not 50051, per config/config.yaml).

Requires ``grpcio`` and ``httpx`` (both already in pyproject deps).
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import grpc.aio
import httpx

from eidolon.livekit.agent.eidolon_agent_rpc.v1.grpc_gen import (
    eidolon_pb2 as pb,
)
from eidolon.livekit.agent.eidolon_agent_rpc.v1.grpc_gen import (
    eidolon_pb2_grpc as pbg,
)


async def provision(
    *,
    admin_base_url: str,
    grpc_target: str,
    tenant_id: str,
    user_id: str,
    device_id: str,
    device_name: str,
    default_template_id: str | None,
) -> str:
    async with httpx.AsyncClient(timeout=10.0) as http:
        body = {"tenant_id": tenant_id, "user_id": user_id}
        if default_template_id:
            body["default_template_id"] = default_template_id
        resp = await http.post(f"{admin_base_url}/api/admin/pairing/codes", json=body)
        resp.raise_for_status()
        code = resp.json()["code"]
        print(f"# issued pairing code: {code}", file=sys.stderr)

    async with grpc.aio.insecure_channel(grpc_target) as channel:
        stub = pbg.EidolonAgentStub(channel)
        exch = await stub.ExchangePairingCode(
            pb.ExchangeRequest(
                pairing_code=code,
                device_id=device_id,
                device_name=device_name,
            )
        )
        print(
            f"# device_id={exch.device_id} tenant={exch.tenant_id} user={exch.user_id}",
            file=sys.stderr,
        )
        return exch.device_token


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--admin-base-url", default="http://127.0.0.1:8081")
    p.add_argument("--grpc-target", default="127.0.0.1:50052")
    p.add_argument("--tenant-id", required=True)
    p.add_argument("--user-id", required=True)
    p.add_argument("--device-id", default="livekit-channel")
    p.add_argument("--device-name", default="livekit-channel")
    p.add_argument("--default-template-id", default=None)
    args = p.parse_args()

    token = asyncio.run(
        provision(
            admin_base_url=args.admin_base_url,
            grpc_target=args.grpc_target,
            tenant_id=args.tenant_id,
            user_id=args.user_id,
            device_id=args.device_id,
            device_name=args.device_name,
            default_template_id=args.default_template_id,
        )
    )
    # stdout is the token alone — convenient for `export ...=$(... | tail -1)`.
    print(token)


if __name__ == "__main__":
    main()
