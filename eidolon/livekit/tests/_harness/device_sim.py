# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""In-process device simulator for contract / E2E tests (plan Track C2).

Builds wire-contract-correct packets exactly as a real device (ESP32 / web)
would — using ONLY the single-source constants in ``eidolon_sdk.biz.contracts`` — so
a test never hand-rolls a JSON body that could drift from the real client. It
also documents the device⇄server handshake in one place and doubles as the
reference for third-party device integrators (Track D / D4).

Two halves of the handshake:
  * ``voice_token_metadata`` — what hub stamps into the VOICE token's
    ``participant_metadata`` (mirrors ``eidolon_hub`` ``.../system/config.py``);
    channel resolves the session from exactly this dict.
  * ``SimulatedDevice.audio_state*`` — the ``client.audio_state`` packets the
    device publishes on ``CLIENT_AUDIO_STATE_TOPIC``; channel parses these.

Production code MUST NOT import from this package.
"""

from __future__ import annotations

import json
import time

from eidolon_sdk.biz.contracts import (
    CLIENT_AUDIO_STATE_TYPE,
    INPUT_MODE_AUTO,
    INPUT_MODE_PTT,
    INTERACTION_MODE_HALF_DUPLEX,
    PLAYBACK_STATE_IDLE,
    SESSION_INTENT_USER_INITIATED,
    WIRE_SCHEMA_VERSION,
)


def voice_token_metadata(
    *, device_id: str, interaction_mode: str, session_intent: str
) -> dict:
    """The ``participant_metadata`` hub stamps into the device VOICE token.

    Mirrors the dict built in ``eidolon_hub`` ``.../system/config.py`` (the
    ``kind="device"`` token). Channel's ``resolve_interaction_mode`` /
    ``resolve_session_intent`` read the session from exactly this shape — so this
    helper is the canonical record of the cross-service metadata bus.
    """
    return {
        "kind": "device",
        "device_id": device_id,
        "interaction_mode": interaction_mode,
        "session_intent": session_intent,
    }


class SimulatedDevice:
    """A minimal contract-correct device for tests.

    ``interaction_mode`` selects the ``input_mode`` the device reports
    (``ptt`` for half-duplex, ``auto`` for full-duplex), matching real firmware.
    """

    def __init__(
        self,
        *,
        device_id: str = "device-sim",
        interaction_mode: str = INTERACTION_MODE_HALF_DUPLEX,
        session_intent: str = SESSION_INTENT_USER_INITIATED,
    ) -> None:
        self.device_id = device_id
        self.interaction_mode = interaction_mode
        self.session_intent = session_intent
        self._input_mode = (
            INPUT_MODE_PTT
            if interaction_mode == INTERACTION_MODE_HALF_DUPLEX
            else INPUT_MODE_AUTO
        )
        self._seq = 0

    # ── session metadata bus (what hub stamps; what channel resolves) ──
    def token_metadata(self) -> dict:
        return voice_token_metadata(
            device_id=self.device_id,
            interaction_mode=self.interaction_mode,
            session_intent=self.session_intent,
        )

    def token_metadata_json(self) -> str:
        return json.dumps(self.token_metadata())

    # ── client.audio_state packets (what the device publishes) ──
    def audio_state(
        self,
        *,
        ptt: bool = False,
        playback_state: str = PLAYBACK_STATE_IDLE,
        mic_muted: bool = False,
        manual_interrupt: bool = False,
    ) -> dict:
        self._seq += 1
        return {
            "schema_v": WIRE_SCHEMA_VERSION,
            "type": CLIENT_AUDIO_STATE_TYPE,
            "seq": self._seq,
            "input_mode": self._input_mode,
            "ptt": ptt,
            "manual_interrupt": manual_interrupt,
            "playback_state": playback_state,
            "mic_muted": mic_muted,
            "client_ts_ms": int(time.time() * 1000),
        }

    def audio_state_bytes(self, **kwargs) -> bytes:
        return json.dumps(self.audio_state(**kwargs)).encode("utf-8")

    # PTT convenience: a half-duplex device toggles ptt on press/release.
    def ptt_press(self, **kwargs) -> bytes:
        return self.audio_state_bytes(ptt=True, **kwargs)

    def ptt_release(self, **kwargs) -> bytes:
        return self.audio_state_bytes(ptt=False, **kwargs)
