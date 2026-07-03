"""Compatibility import for the full-duplex realtime pipeline.

New code should import ``StreamingPipeline`` from
``eidolon.livekit.agent.full_duplex``.
"""

from .full_duplex import StreamingPipeline

__all__ = ["StreamingPipeline"]
