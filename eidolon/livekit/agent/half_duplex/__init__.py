"""Half-duplex voice interaction components.

The half-duplex layer owns push-to-talk turn boundaries.  It deliberately does
not depend on full-duplex streaming STT events, EOT scoring, or natural
barge-in policy.  Provider stages such as STT/LLM/TTS still come from the shared
pipeline factory.
"""

from .ptt_segment import PttAudioSegment, PttAudioSegmentConfig, PttAudioSegmentRecorder
from .ptt_transcriber import (
    PttSegmentTranscriber,
    PttSegmentTranscriberConfig,
    PttSegmentTranscriptionResult,
)
from .ptt_turn_controller import (
    HalfDuplexPttTurnController,
    PttSegmentTurnResult,
)
from .pipeline import HalfDuplexPttPipeline
from .control import (
    PTT_OUTCOME_COMMITTED,
    PTT_OUTCOME_FINALIZING,
    PTT_OUTCOME_RECORDING,
    build_ptt_turn_status_payload,
    ptt_rejected_outcome,
    should_drop_pending_ptt_control_event,
)

__all__ = [
    "HalfDuplexPttPipeline",
    "HalfDuplexPttTurnController",
    "PttAudioSegment",
    "PttAudioSegmentConfig",
    "PttAudioSegmentRecorder",
    "PttSegmentTranscriber",
    "PttSegmentTranscriberConfig",
    "PttSegmentTranscriptionResult",
    "PttSegmentTurnResult",
    "PTT_OUTCOME_COMMITTED",
    "PTT_OUTCOME_FINALIZING",
    "PTT_OUTCOME_RECORDING",
    "build_ptt_turn_status_payload",
    "ptt_rejected_outcome",
    "should_drop_pending_ptt_control_event",
]
