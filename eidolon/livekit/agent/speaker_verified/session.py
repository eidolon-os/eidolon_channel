"""Concrete SDK assembly for speaker admission, sharing providers and RoomIO.

The factory remains owned by the job. This object owns only its AgentSession,
input reader and candidate tasks; no ordinary interruption owner is installed.
A calibrated verifier is a required dependency, never a guessed default.
"""

import asyncio
from dataclasses import replace

from livekit.agents import AgentSession
from livekit.agents.voice.room_io import RoomOptions, TextInputOptions

from ..session.assistant_speech import AssistantSpeechLedger
from ..session.evidence_room_input import EvidenceRoomInput
from ..session.room_io import start_room_session
from ..session.text_input import accept_text_input
from ..session.user_turn_coordinator import UserTurnCoordinator
from ..session.welcome import play_welcome
from ...common.welcome import prepare_welcome_audio, welcome_allowed
from .agent import SpeakerVerifiedAgent, speaker_verified_turn_handling
from .authority import SpeakerPolicyAuthority
from .input import SpeakerVerifiedInput
from .orchestration import SpeakerVerifiedTurns, VERIFICATION_TIMEOUT_SEC
from .semantics import CandidateSemantics


class SpeakerVerifiedSession:
    def __init__(
        self,
        *,
        factory,
        policy,
        verify,
        eot,
        committed,
        instructions="",
        settle_sec=0.8,
        audio_sample_rate=16000,
        welcome=None,
        aec_warmup_duration=None,
        stop_playback=None,
        feedback=None,
        min_audio_ms=1500,
        max_audio_ms=30000,
    ):
        stt = getattr(factory.stt, "stt", None)
        if (
            policy.mode != "allowlist"
            or not policy.owner_id
            or not policy.companion_id
            or not factory.runtime_session_id
            or factory.runtime_services is None
        ):
            raise ValueError("SPEAKER_VERIFIED_SCOPE_REQUIRED")
        if not hasattr(stt, "audio_evidence_capture") or not hasattr(eot, "score_candidate"):
            raise ValueError("SPEAKER_VERIFIED_EVIDENCE_UNSUPPORTED")
        if factory.vad is None or factory.interrupt_classifier is None:
            raise ValueError("SPEAKER_VERIFIED_SEMANTICS_REQUIRED")
        if factory.voiceprint_service is not None:
            raise ValueError("SPEAKER_VERIFIED_LEGACY_OWNER_CONFLICT")
        self._factory, self._stt = factory, stt
        # One complete candidate, plus settlement/provider-result latency.
        # Never expire/evict a maximum-length utterance before it can be checked.
        self._candidate_timeout = max_audio_ms / 1000 + settle_sec + VERIFICATION_TIMEOUT_SEC
        self._stop_playback = stop_playback
        self._capture = None
        self._input = self._presentation_io = None
        self._started = self._closed = False
        self._ready = asyncio.Event()
        self._speech = AssistantSpeechLedger()
        self.session = AgentSession(
            session_close_transcript_timeout=0,
            aec_warmup_duration=aec_warmup_duration,
            turn_handling=speaker_verified_turn_handling(),
        )
        self.coordinator = UserTurnCoordinator(speech_merge_grace_sec=settle_sec)
        authority = SpeakerPolicyAuthority(
            factory.runtime_services.runtime,
            owner_id=policy.owner_id,
            companion_id=policy.companion_id,
        )
        semantics = CandidateSemantics(
            eot, classifier=factory.interrupt_classifier, assistant_text=self._assistant_text
        )
        self.turns = SpeakerVerifiedTurns(
            coordinator=self.coordinator,
            capture=lambda: self._capture,
            session=self.session,
            verify=verify,
            is_complete=semantics.is_complete,
            classify=semantics.classify,
            authorize=authority.authorize,
            committed=committed,
            cancel_output=self._cancel_output,
            feedback=feedback,
            min_audio_ms=min_audio_ms,
            max_audio_ms=max_audio_ms,
        )
        self.input = SpeakerVerifiedInput(
            self.turns, coordinator=self.coordinator, settle_sec=settle_sec,
            candidate_timeout_sec=self._candidate_timeout,
        )
        self.session.on("user_state_changed", self._user_state_changed)
        self.session.on("agent_state_changed", self._agent_state_changed)
        self.session.on("close", self._session_closed)
        selected_welcome = welcome if welcome_allowed(welcome, factory.outputs) else None
        pcm = prepare_welcome_audio(selected_welcome, sample_rate=audio_sample_rate)
        owner = self

        class VoiceAgent(SpeakerVerifiedAgent):
            async def on_enter(self):
                await owner._ready.wait()
                if not owner._closed and selected_welcome is not None:
                    play_welcome(
                        self.session,
                        selected_welcome,
                        outputs=factory.outputs,
                        pcm=pcm,
                        sample_rate=audio_sample_rate,
                        queue_text=owner._speech.queue_fixed_speech,
                    )

            async def on_exit(self):
                # SDK drains after on_exit: pending verification must lose its
                # scope before that drain, even if session.aclose is called directly.
                await owner.input.aclose()

            async def tts_node(self, text, model_settings):
                stream_id = owner._speech.begin_streamed_speech()

                async def observed():
                    async for chunk in text:
                        owner._speech.append_streamed_speech(stream_id, chunk)
                        yield chunk

                try:
                    async for frame in super().tts_node(observed(), model_settings):
                        yield frame
                except BaseException:
                    owner._speech.abort_streamed_speech(stream_id)
                    raise

        self.agent = VoiceAgent(
            outputs=factory.outputs,
            instructions=instructions,
            stt=stt,
            vad=factory.vad.vad,
            llm=factory.llm.llm,
            tts=factory.tts.tts if factory.tts is not None else None,
            observe_speech=self._observe_speech,
        )

    def _observe_speech(self, event):
        if self._ready.is_set() and not self._closed:
            self.input.observe_speech(event)

    def _user_state_changed(self, event):
        if self._ready.is_set() and not self._closed:
            self.input.user_state_changed(event)

    def _bind_capture(self, capture):
        self._capture = capture
        self._stt.audio_evidence_capture = capture

    def _assistant_text(self, output):
        if self.session.current_speech is not output:
            return ""
        return self._speech.current_or_recent_text(factory=self._factory)

    def _agent_state_changed(self, event):
        if event.new_state == "speaking":
            self._speech.on_playback_started()
        elif event.old_state == "speaking":
            self._speech.on_playback_finished()

    def _session_closed(self, event):
        self.input.invalidate("session_closed")

    async def start(
        self,
        *,
        room,
        participant_identity,
        options: RoomOptions,
        on_started,
        output_room=None,
        output_participant_identity=None,
    ):
        if self._started or self._closed:
            raise RuntimeError("session already started or closed")
        if not participant_identity:
            raise ValueError("resolved participant required")
        audio_options = options.get_audio_input_options()
        if audio_options is None:
            raise ValueError("speaker verification requires microphone input")
        self._started = True
        try:
            self._presentation_io = await start_room_session(
                session=self.session,
                agent=self.agent,
                input_room=room,
                options=replace(
                    options,
                    participant_identity=participant_identity,
                    audio_input=False,
                    text_input=TextInputOptions(text_input_cb=self.accept_text),
                ),
                output_room=output_room,
                output_participant_identity=output_participant_identity,
            )
            self._input = EvidenceRoomInput(
                session=self.session,
                room=room,
                participant_identity=participant_identity,
                session_id=self._factory.runtime_session_id,
                endpoint_id=participant_identity,
                bind_capture=self._bind_capture,
                retire_candidate=self.input.invalidate,
                audio_options=audio_options,
                capture_max_seconds=self._candidate_timeout,
            )
            await self._input.start()
            await on_started()
            self._ready.set()
        except BaseException:
            await self.aclose()
            raise

    async def stop_output(self):
        """Caller must first authenticate the explicit button/control event."""
        self.input.invalidate("explicit_stop")
        await self._cancel_output()

    async def _cancel_output(self):
        output = self.session.current_speech
        await self.session.interrupt(force=True)
        # Do not flush a newer output scheduled while the SDK drained the old one.
        if self.session.current_speech is not None and self.session.current_speech is not output:
            return
        if self._stop_playback is not None:
            await self._stop_playback()

    async def accept_text(self, session, event):
        if not self._ready.is_set() or self._closed:
            return
        # A typed request is explicit control. Retire ambient candidates before
        # awaiting playback cancellation, so they cannot overtake typed input.
        self.input.invalidate("explicit_text")
        await accept_text_input(session, event)

    async def aclose(self):
        if self._closed:
            return
        self._closed = True
        self._ready.set()
        await self.input.aclose()
        try:
            if self._input is not None:
                await self._input.aclose()
        finally:
            try:
                await self.session.aclose()
            finally:
                if self._presentation_io is not None:
                    await self._presentation_io.aclose()
                self._bind_capture(None)
