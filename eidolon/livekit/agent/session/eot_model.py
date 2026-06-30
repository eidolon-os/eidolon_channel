"""Shared EOT model loading for streaming sessions."""

from __future__ import annotations

import logging
from typing import Any

from eidolon.livekit.agent.turn_policy import eot_kwargs_from_turn_policy
from eidolon.livekit.common.config import TurnPolicyConfig

logger = logging.getLogger("agent.session.eot_model")

_eot_model_cache: Any = None
_eot_model_cache_key: tuple | None = None


def get_shared_eot_model(turn_policy: TurnPolicyConfig | None = None) -> Any:
    """Lazily create and cache the shared ChineseModel instance.

    The EotManager inside ChineseModel is a thread-safe singleton that holds
    the ONNX session, so all callers share the same model weights in memory.
    """
    global _eot_model_cache, _eot_model_cache_key
    kwargs = eot_kwargs_from_turn_policy(turn_policy)
    key = tuple(sorted(kwargs.items()))
    if _eot_model_cache is None or _eot_model_cache_key != key:
        from eidolon.livekit.plugins.eot import ChineseModel

        logger.info("[StreamingPipeline] loading EOT model...")
        _eot_model_cache = ChineseModel(**kwargs)
        _eot_model_cache_key = key
        logger.info("[StreamingPipeline] EOT model loaded")
    return _eot_model_cache
