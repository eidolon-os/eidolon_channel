"""Session/plugin signal bridge helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("agent.session.signals")


class SessionSignalBridge:
    """Bridge LiveKit session/VAD events to provider-specific hooks."""

    def __init__(
        self,
        *,
        factory: Any,
        get_eot_model: Callable[[], Any],
    ) -> None:
        self._factory = factory
        self._get_eot_model = get_eot_model

    def register_vad_inference_callback(self) -> bool:
        """Forward VAD probability samples to EOT state and optional STT gate."""

        try:
            raw_vad = self._raw_vad()
            if raw_vad is None:
                return False
            register = getattr(raw_vad, "register_inference_callback", None)
            if not callable(register):
                return False

            eot_model = self._get_eot_model()
            stt_notify = self._stt_notify_vad_state()

            def _on_inference(probability: float, speaking: bool) -> None:
                try:
                    eot_model.update_vad_probability(probability)
                except Exception:
                    pass
                if stt_notify is not None:
                    try:
                        stt_notify(probability, 0.0)
                    except Exception:
                        pass

            register(_on_inference)
            logger.info(
                "[SessionSignalBridge] VAD inference callback registered "
                "(per-frame probability -> EOT state%s)",
                " + STT gate" if stt_notify else "",
            )
            return True
        except Exception:
            logger.exception("[SessionSignalBridge] failed to register VAD callback")
            return False

    def signal_stt_user_away(self) -> None:
        self._call_stt_hook("signal_user_away", "user_away")

    def signal_stt_user_present(self) -> None:
        self._call_stt_hook("signal_user_present", "user_present")

    def _raw_vad(self) -> Any:
        vad_stage = getattr(self._factory, "vad", None)
        return getattr(vad_stage, "vad", None) if vad_stage is not None else None

    def _raw_stt(self) -> Any:
        stt_stage = getattr(self._factory, "stt", None)
        if stt_stage is None:
            return None
        return getattr(stt_stage, "_stt", None) or getattr(stt_stage, "stt", None)

    def _stt_notify_vad_state(self) -> Callable[[float, float], None] | None:
        stt = self._raw_stt()
        notify = getattr(stt, "notify_vad_state", None)
        return notify if callable(notify) else None

    def _call_stt_hook(self, hook_name: str, label: str) -> None:
        try:
            stt = self._raw_stt()
            hook = getattr(stt, hook_name, None)
            if callable(hook):
                hook()
        except Exception:
            logger.exception("[SessionSignalBridge] error signaling %s to STT", label)
