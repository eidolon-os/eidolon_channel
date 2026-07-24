"""Avatar worker — a second room participant that renders the talking-head video.

For M1 the channel job spawns this in-process (the plan's simplified deployment):
it connects a *second* ``rtc.Room`` under a deterministic identity
(``avatar-<room>``), receives the agent's TTS audio over LiveKit DataStream
(:class:`DataStreamAudioReceiver`), bridges it to the digital-human service via
:class:`EidolonDHVideoGenerator`, and publishes a synchronized audio+video track
through :class:`AvatarRunner`. Web clients subscribe to that video track.

Identity coordination is trivial here because the channel mints the worker's
token with a known identity and passes the channel agent's own identity in, so
there is no cross-process discovery handshake. Decode/transcode runs off the
event loop (in a thread inside the decoder), so in-process is acceptable for M1;
moving this to its own process/container is a later isolation step.
"""

from __future__ import annotations

import logging

from livekit import api, rtc
from livekit.agents.voice.avatar import (
    AvatarOptions,
    AvatarRunner,
    DataStreamAudioReceiver,
)

from eidolon.livekit.common.config import AvatarConfig

from ._video_gen_base import DHVideoGeneratorBase
from .ditto_streaming_client import DittoStreamClient
from .service_client import DigitalHumanServiceClient, StreamVideoParams
from .streaming_video_generator import StreamingDHVideoGenerator
from .video_generator import EidolonDHVideoGenerator

logger = logging.getLogger("agent.avatar.worker")

# Attribute LiveKit clients use to attribute an avatar worker's tracks to the
# agent it speaks for. Set at runtime after connect (best-effort; media flows
# regardless — the channel↔worker RPCs are addressed by identity directly).
_PUBLISH_ON_BEHALF = "lk.publish_on_behalf"


def avatar_identity_for(room_name: str, *, prefix: str = "avatar") -> str:
    return f"{prefix}-{room_name}"


class AvatarWorker:
    """Owns the avatar participant's room connection + AvatarRunner lifecycle."""

    def __init__(
        self,
        cfg: AvatarConfig,
        *,
        livekit_url: str,
        api_key: str,
        api_secret: str,
        room_name: str,
        agent_identity: str,
        face_image: bytes | None = None,
    ) -> None:
        self._cfg = cfg
        self._url = livekit_url
        self._api_key = api_key
        self._api_secret = api_secret
        self._room_name = room_name
        self._agent_identity = agent_identity
        self._face_image = face_image

        self._identity = avatar_identity_for(room_name, prefix=cfg.worker_identity_prefix)
        self._room: rtc.Room | None = None
        self._runner: AvatarRunner | None = None
        self._service: DigitalHumanServiceClient | None = None
        self._video_gen: DHVideoGeneratorBase | None = None

    @property
    def avatar_identity(self) -> str:
        return self._identity

    async def start(self) -> str:
        """Connect the avatar participant and start rendering. Returns its identity."""
        # Join as an AGENT-kind participant with ``lk.publish_on_behalf`` set at
        # mint time. This is the LiveKit-idiomatic avatar-worker identity: it (a)
        # stops a WorkerType.PUBLISHER channel worker from dispatching a *second*
        # agent to "serve" this participant when it publishes (which would spawn a
        # spurious session observing the avatar's own track), and (b) lets clients
        # attribute the video/audio to the agent it speaks for. Set in the token
        # (not post-connect) so it is authoritative at join, before dispatch.
        token = (
            api.AccessToken(self._api_key, self._api_secret)
            .with_identity(self._identity)
            .with_name(self._identity)
            .with_kind("agent")
            .with_attributes({_PUBLISH_ON_BEHALF: self._agent_identity})
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=self._room_name,
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_data=True,
                    agent=True,
                )
            )
            .to_jwt()
        )

        room = rtc.Room()
        await room.connect(self._url, token)
        self._room = room

        if self._cfg.streaming:
            # Streaming ingestion: feed TTS live over /ws/audio_stream and
            # progressively decode. No persistent service client — the generator
            # opens a session per turn.
            self._video_gen = StreamingDHVideoGenerator(
                DittoStreamClient(
                    self._cfg.service_url,
                    connect_timeout_sec=self._cfg.connect_timeout_sec,
                ),
                width=self._cfg.width,
                height=self._cfg.height,
                target_fps=self._cfg.fps,
                output_sample_rate=self._cfg.output_sample_rate,
                face_image=self._face_image,
                opus_frame_ms=self._cfg.stream_opus_frame_ms,
                fast_start_samples=self._cfg.stream_fast_start_samples,
            )
        else:
            params = StreamVideoParams(
                width=self._cfg.width,
                height=self._cfg.height,
                fps=self._cfg.fps,
                fmt=self._cfg.format,
            )
            self._service = DigitalHumanServiceClient(
                self._cfg.service_url,
                params=params,
                request_timeout_sec=self._cfg.request_timeout_sec,
                connect_timeout_sec=self._cfg.connect_timeout_sec,
            )
            self._video_gen = EidolonDHVideoGenerator(
                self._service,
                width=self._cfg.width,
                height=self._cfg.height,
                target_fps=self._cfg.fps,
                output_sample_rate=self._cfg.output_sample_rate,
                face_image=self._face_image,
            )
        await self._video_gen.warmup()

        audio_recv = DataStreamAudioReceiver(room, sender_identity=self._agent_identity)
        options = AvatarOptions(
            video_width=self._cfg.width,
            video_height=self._cfg.height,
            video_fps=self._cfg.fps,
            audio_sample_rate=self._cfg.output_sample_rate,
            audio_channels=1,
        )
        self._runner = AvatarRunner(
            room, audio_recv=audio_recv, video_gen=self._video_gen, options=options
        )
        await self._runner.start()
        logger.info(
            "[avatar.worker] started identity=%s room=%s sender=%s %dx%d@%.1f",
            self._identity,
            self._room_name,
            self._agent_identity,
            self._cfg.width,
            self._cfg.height,
            self._cfg.fps,
        )
        return self._identity

    async def aclose(self) -> None:
        if self._runner is not None:
            try:
                await self._runner.aclose()
            except Exception:
                logger.debug("[avatar.worker] runner aclose failed", exc_info=True)
        if self._video_gen is not None:
            try:
                await self._video_gen.aclose()
            except Exception:
                logger.debug("[avatar.worker] video_gen aclose failed", exc_info=True)
        if self._room is not None:
            try:
                await self._room.disconnect()
            except Exception:
                logger.debug("[avatar.worker] room disconnect failed", exc_info=True)
        self._runner = self._video_gen = self._room = self._service = None
