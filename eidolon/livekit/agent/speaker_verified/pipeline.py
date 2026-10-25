"""Job lifecycle for verified voice turns; no ordinary interruption machinery."""

import asyncio
import os
from contextlib import AsyncExitStack

from livekit.agents.voice.room_io import AudioInputOptions, AudioOutputOptions, RoomOptions
from eidolon_sdk.biz.contracts import SESSION_END_ERROR, SESSION_END_USER_LEFT
from eidolon_sdk.biz.dialogue_control import CommittedTurnDecision, TurnCommitBoundary
from eidolon_sdk.core.http import create_async_client

from ..runtime.resolver import wait_for_runtime_participant_identity
from ..runtime.interaction_mode import resolve_idle_policy, resolve_welcome
from ..shared.pipeline import BasePipeline
from ..session.committed_turn import publish_committed_turn_decision
from ..session.eot_model import create_session_eot_model
from ..session.idle import IdleWatchdog
from ..session.playback_stop import stop_playback
from ..session.speaker_extraction_client import SpeakerExtractionClient
from .session import SpeakerVerifiedSession
from .verifier import CalibratedSpeakerVerifier
from .feedback import VoiceAdmissionPublisher


class SpeakerVerifiedPipeline(BasePipeline):
    def __init__(
        self,
        factory,
        *,
        policy,
        calibration,
        cfg,
        session_intent,
        on_session_started,
        on_session_end,
        on_session_closed,
        on_idle_disconnect,
    ):
        super().__init__(factory)
        if not calibration.supports(policy):
            raise ValueError("SPEAKER_VERIFIED_CALIBRATION_SCOPE_MISMATCH")
        self._policy, self._calibration, self._cfg = policy, calibration, cfg
        self._intent = session_intent
        self._on_started, self._on_end = on_session_started, on_session_end
        self._on_closed, self._on_idle = on_session_closed, on_idle_disconnect
        self._closed_event = asyncio.Event()
        self._close_error = False
        self._verified = self._http = self._idle = self._eot = None
        self._closing = False
        self._feedback = None

    def _closed(self, event):
        self._close_error = bool(getattr(event, "error", None))
        self._closed_event.set()

    def _disconnected(self, *_):
        if self._verified is not None:
            self._verified.input.invalidate("room_disconnected")
        self._closed_event.set()

    def _committed(self, decision):
        # This callback is reachable only after current Data authorization and
        # the coordinator's once-only commit. Raw ASR never publishes metadata.
        record = CommittedTurnDecision.create(
            text=decision.transcript, boundary=TurnCommitBoundary.SPEAKER_VERIFIED
        )
        if not publish_committed_turn_decision(
            factory=self._factory, decision=record, timeline=None
        ):
            raise RuntimeError("SPEAKER_VERIFIED_COMMIT_METADATA_UNAVAILABLE")
        self.session_mark(
            "speaker_verified_committed",
            candidate_id=decision.candidate_id,
            calibration_version=self._calibration.digest,
        )
        if self._idle is not None:
            self._idle.mark_activity()

    async def run(self, room, *, output_room=None, output_participant_identity=None):
        self._room, self._started = room, True
        try:
            room.on("disconnected", self._disconnected)
            participant = await wait_for_runtime_participant_identity(room)
            self._feedback = VoiceAdmissionPublisher(
                room, participant, self._factory.runtime_session_id
            )
            self.open_session_trace(
                room,
                observability=self._cfg.observability,
                owner_id=self._policy.owner_id,
                companion_id=self._policy.companion_id,
                interaction_mode="speaker_verified",
            )
            self._http = create_async_client(trust_env=False, timeout=8)
            extractor = SpeakerExtractionClient(
                self._http,
                os.environ.get("EIDOLON_CHANNEL_PROVIDER_URL", "http://127.0.0.1:8767"),
                token=os.environ.get("EIDOLON_CHANNEL_PROVIDER_TOKEN", ""),
            )
            verifier = CalibratedSpeakerVerifier(
                data=self._factory.runtime_services.runtime,
                extractor=extractor,
                calibration=self._calibration,
                owner_id=self._policy.owner_id,
                companion_id=self._policy.companion_id,
            )
            self._eot = create_session_eot_model(self._cfg.turn_policy)

            async def stop_device_output():
                selected_room = output_room if output_room is not None else room
                selected_peer = (
                    output_participant_identity if output_room is not None else participant
                )
                revision = (
                    self._factory.output_plan.policy_revision if self._factory.output_plan else 0
                )
                await stop_playback(
                    selected_room, selected_peer, self._factory.runtime_session_id, revision
                )

            self._verified = SpeakerVerifiedSession(
                factory=self._factory,
                policy=self._policy,
                verify=verifier,
                eot=self._eot,
                committed=self._committed,
                instructions=self._cfg.behavior.instructions,
                settle_sec=self._cfg.turn_policy.eot.speech_merge_grace_ms / 1000,
                audio_sample_rate=self._cfg.behavior.audio_sample_rate,
                welcome=resolve_welcome(
                    session_intent=self._intent,
                    welcome_message=self._cfg.behavior.welcome_message,
                    outputs=self._factory.outputs,
                ),
                aec_warmup_duration=(
                    None
                    if self._cfg.turn_policy.interrupt.aec_warmup_ms is None
                    else self._cfg.turn_policy.interrupt.aec_warmup_ms / 1000
                ),
                stop_playback=stop_device_output,
                feedback=self._feedback.update,
                min_audio_ms=self._calibration.min_audio_ms,
                max_audio_ms=self._calibration.max_audio_ms,
            )
            session = self._verified.session
            session.on("close", self._closed)
            session.on("agent_state_changed", self._on_agent_state_changed)
            idle = resolve_idle_policy(
                session_intent=self._intent, idle_config=self._cfg.turn_policy.idle
            )
            self._idle = IdleWatchdog(
                timeout_sec=idle.timeout_sec,
                get_session=lambda: session,
                get_room=lambda: room,
                get_timeline=lambda: None,
                session_closed_event=self._closed_event,
                on_idle_disconnect=self._on_idle,
                on_session_end=self._on_end,
                disconnect_grace_sec=self._cfg.turn_policy.idle.disconnect_grace_ms / 1000,
                idle_end_reason=idle.end_reason,
                # Ambient voices must not keep a verified conversation alive.
                is_busy=lambda: session.agent_state in ("thinking", "speaking"),
            )
            await self._warmup_stages()

            async def started():
                await self._start_music_runtime(session)
                await self._on_started()
                self._feedback.start()
                self._idle.start()

            await self._verified.start(
                room=room,
                participant_identity=participant,
                options=RoomOptions(
                    audio_input=AudioInputOptions(sample_rate=16000),
                    audio_output=(
                        AudioOutputOptions(sample_rate=self._cfg.behavior.audio_sample_rate)
                        if self._factory.outputs.speech or self._factory.outputs.audio_cue
                        else False
                    ),
                    text_output=self._factory.outputs.dialogue_text,
                ),
                on_started=started,
                output_room=output_room,
                output_participant_identity=output_participant_identity,
            )
            await self._closed_event.wait()
        except BaseException:
            self._close_error = True
            raise
        finally:
            if self._verified is not None:
                self._verified.input.invalidate("pipeline_ended")
            # Notify while the client is still connected, then withdraw the job
            # before slow provider drains. Every cleanup runs even if one fails.
            async with AsyncExitStack() as cleanup:
                cleanup.push_async_callback(self.shutdown)
                cleanup.callback(room.off, "disconnected", self._disconnected)
                cleanup.push_async_callback(self._on_closed)
                await self._on_end(
                    SESSION_END_ERROR if self._close_error else SESSION_END_USER_LEFT
                )

    async def shutdown(self):
        if self._closing:
            return
        self._closing = True
        if self._verified is not None:
            self._verified.input.invalidate("pipeline_shutdown")
        if self._idle is not None:
            self._idle.stop()
        async with AsyncExitStack() as cleanup:
            cleanup.push_async_callback(super().shutdown)
            cleanup.push_async_callback(self._factory.aclose)
            cleanup.push_async_callback(self._shutdown_stages)
            if self._eot is not None:
                cleanup.callback(self._eot.end_session)
            if self._http is not None:
                cleanup.push_async_callback(self._http.aclose)
            if self._verified is not None:
                cleanup.push_async_callback(self._verified.aclose)
            if self._feedback is not None:
                cleanup.push_async_callback(self._feedback.aclose)
            cleanup.push_async_callback(self._close_music_runtime)
