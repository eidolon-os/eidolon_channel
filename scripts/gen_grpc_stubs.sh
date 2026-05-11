#!/usr/bin/env bash
# Run grpc_tools.protoc for one or more .proto files under proto/.
# Each argument is a path relative to proto/ (the -I root), mirroring the
# output tree under the repository root (same layout as remote_agent_rpc /
# scout_api wrappers).
#
# Requires: grpcio-tools (see pyproject.toml). From repository root:
#   ./scripts/gen_grpc_stubs.sh eidolon/agent/api/v1/grpc_gen/scout_api.proto
#   ./scripts/gen_grpc_stubs.sh path/a.proto path/b.proto

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $(basename "$0") <proto_rel> [proto_rel ...]" >&2
  echo "  Each proto_rel is relative to proto/, e.g. eidolon/agent/api/v1/grpc_gen/scout_api.proto" >&2
  exit 1
fi

for rel in "$@"; do
  if [[ ! -f "$ROOT/proto/$rel" ]]; then
    echo "Missing proto/$rel" >&2
    exit 1
  fi
done

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi

(
  cd "$ROOT/proto"
  "$PY" -m grpc_tools.protoc \
    -I. \
    --python_out="$ROOT" \
    --grpc_python_out="$ROOT" \
    "$@"
)

for rel in "$@"; do
  out_dir="${rel%/*}"
  echo "Generated gRPC stubs under ${out_dir}/"
done
