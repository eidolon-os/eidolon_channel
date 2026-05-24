#!/usr/bin/env bash
# Regenerate Python gRPC stubs for the EidolonAgent (eidolon.agent.v1) brain
# service. Proto authoritative copy lives in eidolon_agent; mirror it here when
# the contract changes. Run from repository root:
#   ./scripts/gen_eidolon_agent_rpc.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$ROOT/scripts/gen_grpc_stubs.sh" \
  "eidolon/livekit/agent/eidolon_agent_rpc/v1/grpc_gen/eidolon.proto"
