#!/usr/bin/env python3
"""Prepare synthetic-default voiceprint assets and benchmark cases.

This script creates a dedicated synthetic voiceprint for the ``default`` user
from the configured TTS voice. It then writes a small LiveKit room benchmark
suite whose owner-positive clips use the same synthetic voice, while the
negative case can use a real non-default sample if available.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import yaml

from eidolon.livekit.agent.factory import SharedStageFactory
from benchmark.audio_assets import (
    synthesize_composite_pcm,
    synthesize_pcm,
    wav_duration_ms,
    write_wav,
)
from eidolon.livekit.common.config import load_effective_config


SYNTHETIC_CLIPS: dict[str, str] = {
    "enroll_calm_rain": "小雨落在屋檐上，声音很轻，房间里只听到均匀而稳定的说话声。",
    "enroll_medical_system": "我正在测试实时语音系统，希望每句话都清楚、稳定、自然。",
    "enroll_project_risk": "这个医疗项目需要先确认用户角色、业务流程和核心风险。",
    "owner_medical_project": "今天我想聊一下一个新的医疗项目。",
    "owner_long_medical_plan": "我大概想做一个系统，给私立医院的医生用，帮他们整理病人的信息和随访计划。",
    "normal_ask_intro": "帮我详细介绍一下这个方案。",
    "hard_stop_stop": "停一下。",
    "topic_switch_pricing": "换个话题，我们聊一下定价。",
    "backchannel_duiya": "对呀。",
}

COMPOSITE_CLIPS: dict[str, list[tuple[str, int]]] = {
    "owner_pause_private_hospital": [
        ("私立医院的。", 900),
        ("主要给医生做的系统。", 900),
        ("不是给患者的。", 0),
    ],
}

ENROLLMENT_CLIPS = (
    "enroll_calm_rain",
    "enroll_medical_system",
    "enroll_project_risk",
)


def _default_model_dir() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "eidolon"
        / "livekit"
        / "plugins"
        / "speaker_verification"
        / "resources"
        / "3dspeaker"
        / "campplus_zh_16k_common"
    )


async def _generate_audio(args: argparse.Namespace) -> dict[str, Path]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {clip_id: out_dir / f"{clip_id}.wav" for clip_id in SYNTHETIC_CLIPS}
    paths.update({clip_id: out_dir / f"{clip_id}.wav" for clip_id in COMPOSITE_CLIPS})

    if args.skip_tts and all(path.is_file() for path in paths.values()):
        return paths

    cfg = load_effective_config()
    stages = SharedStageFactory.components_from_config(cfg)
    await stages.tts.warmup()
    try:
        for clip_id, text in SYNTHETIC_CLIPS.items():
            path = paths[clip_id]
            if args.skip_tts and path.is_file():
                continue
            pcm, sample_rate = await synthesize_pcm(stages.tts.synthesize, text)
            write_wav(path, pcm, sample_rate=sample_rate)
            print(f"generated {clip_id}: {path}")

        for clip_id, parts in COMPOSITE_CLIPS.items():
            path = paths[clip_id]
            if args.skip_tts and path.is_file():
                continue
            pcm, sample_rate = await synthesize_composite_pcm(
                stages.tts.synthesize, parts
            )
            write_wav(path, pcm, sample_rate=sample_rate)
            print(f"generated {clip_id}: {path}")
    finally:
        await stages.tts.shutdown()
    return paths


async def _admin_json(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    **kwargs: Any,
) -> dict[str, Any]:
    response = await client.request(method, path, **kwargs)
    response.raise_for_status()
    if not response.content:
        return {}
    return response.json()


async def _ensure_default_agent(args: argparse.Namespace) -> None:
    async with httpx.AsyncClient(base_url=args.admin_url, timeout=60.0, trust_env=False) as client:
        users = await _admin_json(client, "GET", "/api/users")
        for user in users.get("users", []):
            if user.get("spec", {}).get("user_id") == args.user_id and user.get("active_agent_id"):
                print(f"user {args.user_id!r} already has active agent {user['active_agent_id']}")
                return
        created = await _admin_json(
            client,
            "POST",
            "/api/agents",
            json={
                "user_id": args.user_id,
                "template_id": args.template_id,
                "display_name": f"{args.user_id} synthetic voiceprint E2E agent",
                "set_active": True,
            },
        )
        print(f"created active agent for {args.user_id}: {created.get('agent_id')}")


async def _update_voiceprint(
    args: argparse.Namespace,
    paths: dict[str, Path],
) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=args.admin_url, timeout=240.0, trust_env=False) as client:
        enrollment = await _admin_json(
            client,
            "POST",
            f"/api/users/{args.user_id}/voiceprint/enrollments",
            json={
                "provider": "3d_speaker",
                "model": "campplus_zh_16k_common",
                "sample_rate": 16_000,
            },
        )
        enrollment_id = enrollment["enrollment_id"]
        print(f"created enrollment {enrollment_id}")
        for clip_id in ENROLLMENT_CLIPS:
            wav_bytes = paths[clip_id].read_bytes()
            sample = await _admin_json(
                client,
                "POST",
                f"/api/users/{args.user_id}/voiceprint/enrollments/{enrollment_id}/samples",
                content=wav_bytes,
                headers={"content-type": "audio/wav"},
            )
            print(
                f"uploaded {clip_id}: {sample.get('sample_id')} "
                f"{sample.get('duration_ms')}ms"
            )
        completed = await _admin_json(
            client,
            "POST",
            f"/api/users/{args.user_id}/voiceprint/enrollments/{enrollment_id}/complete",
        )
        profile = completed["profile"]
        print(
            "completed voiceprint "
            f"profile={profile.get('profile_id')} embedding={profile.get('embedding_ref')}"
        )
        return profile


def _negative_clip(args: argparse.Namespace) -> Path | None:
    path = Path(args.negative_sample).expanduser()
    return path if path.is_file() else None


def _existing_profile(args: argparse.Namespace) -> dict[str, Any] | None:
    profile_path = (
        Path("~/eidolon/voiceprints")
        .expanduser()
        / args.tenant_id
        / args.user_id
        / "profile.json"
    )
    if not profile_path.is_file():
        return None
    return json.loads(profile_path.read_text(encoding="utf-8"))


def _clip_entry(clip_id: str, text: str, path: Path, intent: str = "normal") -> dict[str, Any]:
    return {
        "id": clip_id,
        "text": text,
        "path": str(path),
        "intent": intent,
    }


def _write_suite(args: argparse.Namespace, paths: dict[str, Path]) -> Path:
    out_path = Path(args.cases_out)
    negative = _negative_clip(args)
    cases: list[dict[str, Any]] = [
        {
            "case_id": "synthetic_default_owner_normal_001",
            "suite": "synthetic_default_voiceprint_e2e",
            "description": "default synthetic owner voiceprint: normal owner turn should pass voiceprint and reach brain/TTS.",
            "timeout_sec": 90,
            "tags": ["synthetic_default", "voiceprint", "owner_positive"],
            "audio_clips": [
                _clip_entry(
                    "owner_medical_project",
                    SYNTHETIC_CLIPS["owner_medical_project"],
                    paths["owner_medical_project"],
                )
            ],
            "user_steps": [
                {
                    "text": "今天我想聊一下一个新的医疗项目",
                    "audio": "owner_medical_project",
                    "start_ms": 200,
                    "duration_ms": wav_duration_ms(paths["owner_medical_project"]),
                    "final_delay_ms": 160,
                    "agent_speaking": False,
                    "client_playback_state": "idle",
                }
            ],
            "expect": {
                "action": "none",
                "intent": "uncertain",
                "voiceprint": "allowed",
                "brain": "required",
                "agent_audio_response": "after_user_done",
                "canonical_contains": ["医疗项目"],
                "min_user_finals": 1,
                "min_agent_messages": 1,
            },
        },
        {
            "case_id": "synthetic_default_owner_pause_001",
            "suite": "synthetic_default_voiceprint_e2e",
            "description": "default synthetic owner voiceprint: natural pauses inside one utterance should keep the medical-system meaning.",
            "timeout_sec": 90,
            "tags": ["synthetic_default", "voiceprint", "turn_boundary"],
            "audio_clips": [
                _clip_entry(
                    "owner_pause_private_hospital",
                    "私立医院的。主要给医生做的系统。不是给患者的。",
                    paths["owner_pause_private_hospital"],
                )
            ],
            "user_steps": [
                {
                    "text": "私立医院的，主要给医生做的系统，不是给患者的",
                    "audio": "owner_pause_private_hospital",
                    "start_ms": 200,
                    "duration_ms": wav_duration_ms(paths["owner_pause_private_hospital"]),
                    "final_delay_ms": 180,
                    "agent_speaking": False,
                    "client_playback_state": "idle",
                }
            ],
            "expect": {
                "action": "none",
                "intent": "uncertain",
                "voiceprint": "allowed",
                "brain": "required",
                "agent_audio_response": "after_user_done",
                "canonical_contains": ["医生"],
                "min_user_finals": 1,
                "min_agent_messages": 1,
            },
        },
        {
            "case_id": "synthetic_default_hard_stop_001",
            "suite": "synthetic_default_voiceprint_e2e",
            "description": "default synthetic owner voiceprint: hard stop should cancel current output while preserving the first owner turn.",
            "timeout_sec": 90,
            "tags": ["synthetic_default", "voiceprint", "hard_stop"],
            "audio_clips": [
                _clip_entry("normal_ask_intro", SYNTHETIC_CLIPS["normal_ask_intro"], paths["normal_ask_intro"]),
                _clip_entry("hard_stop_stop", SYNTHETIC_CLIPS["hard_stop_stop"], paths["hard_stop_stop"], "hard_stop"),
            ],
            "user_steps": [
                {
                    "text": "帮我详细介绍一下这个方案",
                    "audio": "normal_ask_intro",
                    "start_ms": 200,
                    "duration_ms": wav_duration_ms(paths["normal_ask_intro"]),
                    "final_delay_ms": 120,
                    "agent_speaking": False,
                    "client_playback_state": "idle",
                },
                {
                    "text": "停一下",
                    "audio": "hard_stop_stop",
                    "start_ms": 1800,
                    "duration_ms": wav_duration_ms(paths["hard_stop_stop"]),
                    "interims": ["停一下"],
                    "final_delay_ms": 80,
                    "agent_speaking": True,
                    "client_playback_state": "agent_speaking",
                },
            ],
            "expect": {
                "action": "cancel",
                "intent": "hard_stop",
                "voiceprint": "allowed",
                "brain": "any",
                "agent_audio_response": "first",
                "max_interrupt_decision_ms": 700,
                "min_user_finals": 1,
                "min_agent_messages": 1,
            },
        },
        {
            "case_id": "synthetic_default_topic_switch_001",
            "suite": "synthetic_default_voiceprint_e2e",
            "description": "default synthetic owner voiceprint: topic switch during agent output should cancel and keep the requested new topic.",
            "timeout_sec": 90,
            "tags": ["synthetic_default", "voiceprint", "topic_switch"],
            "audio_clips": [
                _clip_entry("normal_ask_intro", SYNTHETIC_CLIPS["normal_ask_intro"], paths["normal_ask_intro"]),
                _clip_entry(
                    "topic_switch_pricing",
                    SYNTHETIC_CLIPS["topic_switch_pricing"],
                    paths["topic_switch_pricing"],
                    "topic_switch",
                ),
            ],
            "user_steps": [
                {
                    "text": "帮我详细介绍一下这个方案",
                    "audio": "normal_ask_intro",
                    "start_ms": 200,
                    "duration_ms": wav_duration_ms(paths["normal_ask_intro"]),
                    "final_delay_ms": 120,
                    "agent_speaking": False,
                    "client_playback_state": "idle",
                },
                {
                    "text": "换个话题，我们聊一下定价",
                    "audio": "topic_switch_pricing",
                    "start_ms": 1800,
                    "duration_ms": wav_duration_ms(paths["topic_switch_pricing"]),
                    "final_delay_ms": 100,
                    "agent_speaking": True,
                    "client_playback_state": "agent_speaking",
                },
            ],
            "expect": {
                "action": "cancel",
                "voiceprint": "allowed",
                "brain": "required",
                "agent_audio_response": "after_user_done",
                "canonical_contains": ["换个话题"],
                "min_user_finals": 1,
                "min_agent_messages": 1,
            },
        },
        {
            "case_id": "synthetic_default_backchannel_001",
            "suite": "synthetic_default_voiceprint_e2e",
            "description": "default synthetic owner voiceprint: short backchannel during output should not become a semantic user request.",
            "timeout_sec": 90,
            "tags": ["synthetic_default", "voiceprint", "backchannel"],
            "audio_clips": [
                _clip_entry("normal_ask_intro", SYNTHETIC_CLIPS["normal_ask_intro"], paths["normal_ask_intro"]),
                _clip_entry("backchannel_duiya", SYNTHETIC_CLIPS["backchannel_duiya"], paths["backchannel_duiya"], "backchannel"),
            ],
            "user_steps": [
                {
                    "text": "帮我详细介绍一下这个方案",
                    "audio": "normal_ask_intro",
                    "start_ms": 200,
                    "duration_ms": wav_duration_ms(paths["normal_ask_intro"]),
                    "final_delay_ms": 120,
                    "agent_speaking": False,
                    "client_playback_state": "idle",
                },
                {
                    "text": "对呀",
                    "audio": "backchannel_duiya",
                    "start_ms": 1800,
                    "duration_ms": wav_duration_ms(paths["backchannel_duiya"]),
                    "interims": ["对呀"],
                    "final_delay_ms": 80,
                    "agent_speaking": True,
                    "client_playback_state": "agent_speaking",
                },
            ],
            "expect": {
                "action": "any",
                "decision_action": "rollback",
                "decision_intent": "backchannel",
                "voiceprint": "allowed",
                "brain": "any",
                "rejected_turn_brain": "forbidden",
                "agent_audio_response": "first",
                "min_user_finals": 1,
                "min_agent_messages": 1,
            },
        },
    ]

    if negative is not None:
        cases.append(
            {
                "case_id": "synthetic_default_non_owner_rejected_001",
                "suite": "synthetic_default_voiceprint_e2e",
                "description": "default synthetic voiceprint: non-default voice should be transcribed but blocked before brain.",
                "timeout_sec": 70,
                "tags": ["synthetic_default", "voiceprint", "negative"],
                "audio_clips": [
                    _clip_entry(
                        "non_owner_sample",
                        "non-default speaker sample",
                        negative,
                    )
                ],
                "user_steps": [
                    {
                        "text": "non-default speaker sample",
                        "audio": "non_owner_sample",
                        "start_ms": 200,
                        "duration_ms": wav_duration_ms(negative),
                        "final_delay_ms": 180,
                        "agent_speaking": False,
                        "client_playback_state": "idle",
                    }
                ],
                "expect": {
                    "action": "none",
                    "intent": "uncertain",
                    "voiceprint": "blocked",
                    "brain": "forbidden",
                    "agent_audio_response": "none",
                    "min_user_finals": 1,
                    "min_agent_messages": 0,
                },
            }
        )
    else:
        print(f"negative sample not found, skipping negative case: {args.negative_sample}")

    payload = {"suite_id": "synthetic_default_voiceprint_e2e", "cases": cases}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"wrote suite: {out_path}")
    return out_path


def _write_manifest(args: argparse.Namespace, paths: dict[str, Path], profile: dict[str, Any] | None) -> None:
    manifest_path = Path(args.manifest_out)
    manifest = {
        "tenant_id": args.tenant_id,
        "user_id": args.user_id,
        "admin_url": args.admin_url,
        "audio": {
            clip_id: {
                "path": str(path),
                "duration_ms": wav_duration_ms(path),
            }
            for clip_id, path in sorted(paths.items())
        },
        "enrollment_clips": list(ENROLLMENT_CLIPS),
        "profile": profile or {},
        "cases": args.cases_out,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"wrote manifest: {manifest_path}")


async def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant-id", default="default")
    parser.add_argument("--user-id", default="default")
    parser.add_argument("--template-id", default="caretaker_jiezhi")
    parser.add_argument("--admin-url", default="http://127.0.0.1:9000")
    parser.add_argument("--out-dir", default="benchmark/audio/synthetic_default")
    parser.add_argument(
        "--cases-out",
        default="benchmark/cases/synthetic_default_voiceprint_e2e.yaml",
    )
    parser.add_argument(
        "--manifest-out",
        default="benchmark/audio/synthetic_default/manifest.yaml",
    )
    parser.add_argument(
        "--negative-sample",
        default=(
            "/Users/manson/eidolon/voiceprints/default/manson/enrollments/"
            "vpe_4b5aca8dd8d0/samples/sample_001.wav"
        ),
    )
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--skip-admin", action="store_true")
    parser.add_argument("--skip-agent", action="store_true")
    args = parser.parse_args()

    paths = await _generate_audio(args)
    profile = None
    if not args.skip_admin:
        if not args.skip_agent:
            await _ensure_default_agent(args)
        profile = await _update_voiceprint(args, paths)
    else:
        profile = _existing_profile(args)
    suite_path = _write_suite(args, paths)
    _write_manifest(args, paths, profile)
    print(
        "next: ./.venv/bin/python scripts/bench_voice.py --runner livekit_room "
        f"--cases {suite_path} --livekit-participant-identity {args.user_id} "
        "--livekit-participant-kind user"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
