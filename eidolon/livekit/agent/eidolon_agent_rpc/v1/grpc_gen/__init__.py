"""The Agent gRPC bindings, re-exported from the Contract Plane.

This package used to hold a mirrored copy of eidolon.proto and its own
generated stubs. Two copies of one wire contract drift, and the mirror was
maintained by hand — the old generation script said so outright.

The contract now has a single copy in eidolon_sdk. This module stays as the
import surface so call sites keep working; edit the contract at
``eidolon_sdk/contracts/grpc/eidolon_sdk/grpc/eidolon_agent/v1/eidolon.proto``
and regenerate with ``eidolon_sdk/scripts/gen_grpc_stubs.sh``.
"""

from eidolon_sdk.grpc.eidolon_agent.v1 import eidolon_pb2, eidolon_pb2_grpc

__all__ = ["eidolon_pb2", "eidolon_pb2_grpc"]
