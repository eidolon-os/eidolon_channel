"""EidolonAgent (eidolon.agent.v1) gRPC bridge.

The channel uses :class:`EidolonAgentGrpcLlm` as a LiveKit ``LLM`` implementation
that delegates token generation to the companion-brain service running
eidolon_agent. Generated stubs live under ``v1/grpc_gen/``.
"""

from eidolon.livekit.agent.eidolon_agent_rpc.grpc_llm import EidolonAgentGrpcLlm

__all__ = ["EidolonAgentGrpcLlm"]
