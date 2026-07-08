"""Full-duplex output ducking flow."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from .state_machine import FullDuplexPhase

if TYPE_CHECKING:
    from livekit.agents.voice import AgentSession

    from .pipeline import StreamingPipeline

logger = logging.getLogger("agent")


class FullDuplexOutputFlow:
    """Own output-middleware install and VAD-start duck arming."""

    def __init__(self, pipeline: StreamingPipeline) -> None:
        self._pipeline = pipeline

    def install_duck_mixer(self, session: AgentSession) -> None:
        """Wrap the session's audio output sink with the ducking controller."""
        pipeline = self._pipeline
        cfg = pipeline._get_eot_model()._config
        mixer = pipeline._ducking.install(session, cfg)
        if mixer is None:
            return
        logger.info(
            "[StreamingPipeline] DuckingMixer installed "
            "(fade_out=%dms fade_in=%dms suspend_vol=%.2f "
            "buffer_max=%.1fs cancel_thr=%.2f resume_thr=%.2f "
            "timeout=%.2fs cooldown=%.2fs)",
            cfg.duck_fade_ms,
            cfg.duck_fade_in_ms,
            cfg.duck_suspend_volume,
            cfg.duck_buffer_max_sec,
            cfg.duck_early_cancel_score_threshold,
            cfg.duck_early_resume_score_threshold,
            cfg.duck_suspend_timeout_sec,
            cfg.duck_cooldown_sec,
        )

    def duck_and_arm_timeout(self) -> bool:
        """Fade output to silence and arm the suspend-window fallback."""
        pipeline = self._pipeline
        if not pipeline._ducking.installed:
            return False
        cfg = pipeline._get_eot_model()._config
        if pipeline._filler is not None and pipeline._filler.is_playing:
            logger.info("[StreamingPipeline] duck skipped - filler playing")
            return False
        now = time.monotonic()
        if now - pipeline._ducking.last_unduck_time < cfg.duck_cooldown_sec:
            logger.info(
                "[StreamingPipeline] duck skipped - within cooldown "
                "(%.2fs since last unduck)",
                now - pipeline._ducking.last_unduck_time,
            )
            return False
        pipeline._ducking.cancel_timeout()
        if not pipeline._ducking.duck(now=now):
            self.record_duck_event(
                "duck_skipped",
                reason="output_not_suspendable",
                output_state=(
                    pipeline._ducking.mixer.state
                    if pipeline._ducking.mixer is not None
                    else "not_installed"
                ),
            )
            logger.info(
                "[StreamingPipeline] duck skipped - output not suspendable state=%s",
                (
                    pipeline._ducking.mixer.state
                    if pipeline._ducking.mixer is not None
                    else "not_installed"
                ),
            )
            return False
        if pipeline._timeline is not None:
            pipeline._timeline.mark("interrupt_started_at")
            self.record_duck_event(
                "duck_started",
                vad_to_duck_ms=(
                    (now - pipeline._user_speaking_start_time) * 1000
                    if pipeline._user_speaking_start_time is not None
                    else None
                ),
                timeout_sec=cfg.duck_suspend_timeout_sec,
                cooldown_sec=cfg.duck_cooldown_sec,
            )
            pipeline._record_full_duplex_transition(
                FullDuplexPhase.PROVISIONAL_DUCK,
                event="duck_started",
                reason="vad_started",
                side_effect="reversible",
                details={
                    "timeout_sec": cfg.duck_suspend_timeout_sec,
                    "cooldown_sec": cfg.duck_cooldown_sec,
                },
            )
        pipeline._callbacks.on_duck_started()
        vad_to_duck_ms = 0.0
        if pipeline._user_speaking_start_time is not None:
            vad_to_duck_ms = (now - pipeline._user_speaking_start_time) * 1000
        logger.info(
            "[StreamingPipeline] duck armed  vad->duck=%.1fms  timeout=%.2fs  cooldown=%.2fs",
            vad_to_duck_ms,
            cfg.duck_suspend_timeout_sec,
            cfg.duck_cooldown_sec,
        )
        pipeline._ducking.timeout_task = asyncio.create_task(
            pipeline._duck_deadline.run(cfg.duck_suspend_timeout_sec)
        )
        return True

    def record_duck_event(self, event: str, **fields: object) -> None:
        timeline = self._pipeline._timeline
        if timeline is None:
            return
        payload = {"event": event, **fields}
        events = list(timeline.attrs.get("duck_events") or ())
        events.append(payload)
        timeline.set_attr("duck_events", events)
        timeline.set_attr("duck_last_event", payload)
