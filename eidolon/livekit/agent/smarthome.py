"""Deliver one committed Korvo-1 transcript to the smart-home use case."""

from __future__ import annotations

import logging
import os

import httpx
from eidolon_sdk.biz.smarthome import VoiceResult

from eidolon.livekit.agent.shared.types import generate_turn_id

logger = logging.getLogger(__name__)


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
