"""Deliver one committed device transcript to the smart-home use case."""

from __future__ import annotations

import logging
import os
import asyncio
from collections.abc import Awaitable, Callable

import httpx
from eidolon_sdk.biz.contracts import SESSION_INTENT_USER_INITIATED
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
    session_id: str,
    on_started: Callable[[], Awaitable[None]],
    on_end: Callable[[str], Awaitable[None]],
    on_closed: Callable[[], Awaitable[None]],
    on_idle: Callable[[], Awaitable[None]] | None = None,
    session_intent: str = SESSION_INTENT_USER_INITIATED,
) -> None:
    """Listen on the existing device room; this profile has no output track.

    Bounded by the same idle window as a Companion session. Nothing else ends
    this session when its device does not: the room outlives the device (the
    Provider's listener keeps it occupied), and the participant disconnect a
    reset or power loss produces is not one AgentSession closes on — the next
    connection with the same identity is linked instead. Without the window
    a device that restarted mid-session left this session attached to its
    next run until the host was restarted, and every conversation the device
    asked for meanwhile was refused (2026-09-29). `on_idle` withdraws the
    dispatch; `on_closed` is not called after it.
    """
    from livekit.agents import StopResponse
    from livekit.agents.voice import Agent, AgentSession
    from livekit.agents.voice.room_io import RoomOptions

    from eidolon.livekit.agent.factory import SharedStageFactory
    from eidolon.livekit.agent.runtime.interaction_mode import resolve_idle_policy
    from eidolon.livekit.agent.runtime.resolver import wait_for_runtime_participant_identity
    from eidolon.livekit.agent.session.idle import IdleWatchdog

    participant = await wait_for_runtime_participant_identity(room)
    if participant != device_ref:
        raise ValueError("smart-home dispatch does not match room participant")

    stt = SharedStageFactory._build_stt(cfg).stt
    vad = prebuilt_vad if prebuilt_vad is not None else SharedStageFactory._build_vad(cfg)
    closed = asyncio.Event()
    disconnected = asyncio.Event()
    delivering = 0
    idle_ended = False

    class HomeAgent(Agent):
        async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
            nonlocal delivering
            transcript = (new_message.text_content or "").strip()
            if transcript:
                delivering += 1
                watchdog.mark_activity()
                try:
                    await handle_transcript(owner_id, device_ref, transcript, session_id=session_id)
                except Exception:
                    logger.exception("smart-home panel result delivery failed room=%s", room.name)
                finally:
                    delivering -= 1
                    watchdog.mark_activity()
            raise StopResponse()

    session = AgentSession(
        stt=stt,
        vad=vad,
        turn_handling={
            "turn_detection": "vad" if vad is not None else "stt",
            "interruption": {"enabled": False},
        },
    )

    async def _idle_disconnect() -> None:
        nonlocal idle_ended
        idle_ended = True
        if on_idle is not None:
            await on_idle()

    idle_policy = resolve_idle_policy(
        session_intent=session_intent, idle_config=cfg.turn_policy.idle
    )
    watchdog = IdleWatchdog(
        timeout_sec=idle_policy.timeout_sec,
        get_session=lambda: session,
        get_room=lambda: room,
        get_timeline=lambda: None,
        session_closed_event=closed,
        on_idle_disconnect=_idle_disconnect,
        on_session_end=on_end,
        disconnect_grace_sec=cfg.turn_policy.idle.disconnect_grace_ms / 1000.0,
        idle_end_reason=idle_policy.end_reason,
        # A command being delivered is work even though nobody is speaking.
        is_busy=lambda: delivering > 0 or getattr(session, "user_state", None) == "speaking",
    )
    # Recognised text is activity; raw VAD and noise are not, the same rule as
    # a Companion session, so a noisy but silent panel still goes idle.
    session.on(
        "user_input_transcribed",
        lambda event: watchdog.mark_activity()
        if (getattr(event, "transcript", "") or "").strip() else None,
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
        watchdog.start()
        waits = {asyncio.create_task(closed.wait()), asyncio.create_task(disconnected.wait())}
        try:
            done, pending = await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*waits, return_exceptions=True)
            for task in done:
                task.result()
        finally:
            if idle_ended and watchdog.task is not None:
                # The idle end is withdrawing the dispatch, and withdrawing it
                # is what drops the room; let it finish rather than cancel it
                # half way and leave the dispatch behind.
                await asyncio.gather(watchdog.task, return_exceptions=True)
            watchdog.stop()
            await on_end("user_left")
            if not idle_ended:
                await on_closed()
    finally:
        try:
            await end_home_session(owner_id, device_ref, session_id)
        finally:
            await session.aclose()
            await stt.aclose()


async def handle_transcript(owner_id: str, device_ref: str, transcript: str, *, session_id: str) -> None:
    turn_id = generate_turn_id()
    utterance = " ".join(transcript.split())[:200]
    agent_token = os.environ.get("EIDOLON_AGENT_ADMIN_API_TOKEN", "")
    panel_token = os.environ.get("EIDOLON_CHANNEL_PROVIDER_TOKEN", "")
    if not agent_token or not panel_token:
        raise RuntimeError("smart-home service credentials are missing")

    async with httpx.AsyncClient(trust_env=False, timeout=12) as client:
        try:
            response = await client.post(
                os.environ.get("EIDOLON_AGENT_ADMIN_URL", "http://127.0.0.1:8081").rstrip("/")
                + "/api/admin/smarthome/command",
                headers={"Authorization": f"Bearer {agent_token}"},
                json={
                    "owner_id": owner_id,
                    "device_ref": device_ref,
                    "turn_id": turn_id,
                    "utterance": transcript[:512],
                    "session_id": session_id,
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
            os.environ.get("EIDOLON_CHANNEL_PROVIDER_URL", "http://127.0.0.1:8767").rstrip("/")
            + "/v1/smarthome/result",
            headers={"Authorization": f"Bearer {panel_token}"},
            json={
                "owner_id": owner_id,
                "device_ref": device_ref,
                "result": result.model_dump(mode="json"),
            },
        )
        response.raise_for_status()
        logger.info("smart-home turn=%s outcome=%s delivered", turn_id, result.outcome)


async def end_home_session(owner_id: str, device_ref: str, session_id: str) -> None:
    """Forward lifecycle only; conversational state belongs to Agent."""
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=3) as client:
            response = await client.post(
                os.environ.get("EIDOLON_AGENT_ADMIN_URL", "http://127.0.0.1:8081").rstrip("/")
                + "/api/admin/smarthome/session/end",
                headers={"Authorization": f"Bearer {os.environ.get('EIDOLON_AGENT_ADMIN_API_TOKEN', '')}"},
                json={"owner_id": owner_id, "device_ref": device_ref, "session_id": session_id},
            )
            response.raise_for_status()
    except httpx.HTTPError:
        # Agent also expires context, including after transport/process failures.
        logger.warning("home session close notification failed session=%s", session_id)
