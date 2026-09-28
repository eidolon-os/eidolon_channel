"""Deliver one committed Korvo-1 transcript to the smart-home use case."""

from __future__ import annotations

import logging
import os
import asyncio
from collections.abc import Awaitable, Callable

import httpx
from eidolon_sdk.biz.smarthome import VoiceResult

from eidolon.livekit.agent.shared.types import generate_turn_id

logger = logging.getLogger(__name__)


async def run_smarthome_session(
    *,
    room,
    cfg,
    prebuilt_vad,
    owner_id: str,
    device_ref: str,
    on_started: Callable[[], Awaitable[None]],
    on_end: Callable[[str], Awaitable[None]],
    on_closed: Callable[[], Awaitable[None]],
) -> None:
    """Listen on the existing device room; this profile has no output track."""
    from livekit.agents import StopResponse
    from livekit.agents.voice import Agent, AgentSession
    from livekit.agents.voice.room_io import RoomOptions

    from eidolon.livekit.agent.factory import SharedStageFactory
    from eidolon.livekit.agent.runtime.resolver import wait_for_runtime_participant_identity

    participant = await wait_for_runtime_participant_identity(room)
    if participant != device_ref:
        raise ValueError("smart-home dispatch does not match room participant")

    stt = SharedStageFactory._build_stt(cfg).stt
    vad = prebuilt_vad if prebuilt_vad is not None else SharedStageFactory._build_vad(cfg)
    closed = asyncio.Event()
    disconnected = asyncio.Event()

    class HomeAgent(Agent):
        async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
            transcript = (new_message.text_content or "").strip()
            if transcript:
                try:
                    await handle_transcript(owner_id, device_ref, transcript)
                except Exception:
                    logger.exception("smart-home panel result delivery failed room=%s", room.name)
            raise StopResponse()

    session = AgentSession(
        stt=stt,
        vad=vad,
        turn_handling={
            "turn_detection": "vad" if vad is not None else "stt",
            "interruption": {"enabled": False},
        },
    )
    session.on("close", lambda *_: closed.set())
    room.on("disconnected", lambda *_: disconnected.set())
    try:
        await session.start(
            agent=HomeAgent(instructions="仅转写本次家居指令。", llm=None, tts=None),
            room=room,
            room_options=RoomOptions(
                participant_identity=device_ref,
                audio_output=False,
                text_output=False,
                text_input=False,
            ),
        )
        await on_started()
        waits = {asyncio.create_task(closed.wait()), asyncio.create_task(disconnected.wait())}
        try:
            done, pending = await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*waits, return_exceptions=True)
            for task in done:
                task.result()
        finally:
            await on_end("user_left")
            await on_closed()
    finally:
        await session.aclose()
        await stt.aclose()


async def handle_transcript(owner_id: str, device_ref: str, transcript: str) -> None:
    turn_id = generate_turn_id()
    utterance = " ".join(transcript.split())[:200]
    agent_token = os.environ.get("EIDOLON_AGENT_ADMIN_API_TOKEN", "")
    panel_token = os.environ.get("EIDOLON_CHANNEL_PROVIDER_TOKEN", "")
    if not agent_token or not panel_token:
        raise RuntimeError("smart-home service credentials are missing")

    async with httpx.AsyncClient(trust_env=False, timeout=12) as client:
        try:
            response = await client.post(
                os.environ.get("EIDOLON_TEAM_AGENT_URL", "http://127.0.0.1:8081").rstrip("/")
                + "/api/admin/smarthome/command",
                headers={"Authorization": f"Bearer {agent_token}"},
                json={
                    "owner_id": owner_id,
                    "device_ref": device_ref,
                    "turn_id": turn_id,
                    "utterance": transcript[:512],
                },
            )
            response.raise_for_status()
            result = VoiceResult.model_validate(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("smart-home command unavailable for turn=%s: %s", turn_id, type(exc).__name__)
            result = VoiceResult(
                turn_id=turn_id,
                utterance=utterance,
                outcome="unavailable",
                message="智能家居服务暂不可用，请稍后重试",
            )

        response = await client.post(
            "http://127.0.0.1:8767/v1/smarthome/result",
            headers={"Authorization": f"Bearer {panel_token}"},
            json={
                "owner_id": owner_id,
                "device_ref": device_ref,
                "result": result.model_dump(mode="json"),
            },
        )
        response.raise_for_status()
        logger.info("smart-home turn=%s outcome=%s delivered", turn_id, result.outcome)
