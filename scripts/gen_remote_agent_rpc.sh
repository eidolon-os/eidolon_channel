#!/usr/bin/env bash
# Regenerate Python gRPC stubs for remote_agent_rpc (thin wrapper).
# Requires: grpcio-tools (see pyproject.toml). Run from repository root:
#   ./scripts/gen_remote_agent_rpc.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$ROOT/scripts/gen_grpc_stubs.sh" \
  "eidolon/livekit/agent/remote_agent_rpc/v1/grpc_gen/remote_agent_rpc.proto"
