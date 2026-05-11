"""Test BatchPipeline (manual mode) with real audio file vad.m4a.

Simulates what BatchPipeline does: STT -> LLM -> TTS
without needing a LiveKit server.

Usage::

    cd <repository-root> && source .venv/bin/activate
    python test_batch_manual.py
"""

import asyncio
import logging
import os
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("batch_test")


def load_env() -> None:
    """Load .env from agent module directory into os.environ."""
    env_path = Path(__file__).parent / "eidolon" / "channel" / "livekit" / "agent" / ".env"
    if not env_path.exists():
        logger.warning(".env not found at %s", env_path)
        return
    from dotenv import dotenv_values

    env = dotenv_values(env_path)
    for k, v in env.items():
        if k not in os.environ:
            os.environ[k] = v
    logger.info("Loaded .env from %s", env_path)


async def main():
    load_env()

    # ---- Load audio file ----
    audio_path = Path(__file__).parent / "tmp" / "pipeline" / "data" / "vad.m4a"
    if not audio_path.exists():
        logger.error("Audio file not found: %s", audio_path)
        return

    try:
        from pydub import AudioSegment
    except ImportError:
        logger.error("pydub not installed. Install with: pip install pydub")
        return

    audio = AudioSegment.from_file(str(audio_path))
    logger.info(
        "Audio loaded: channels=%d sample_rate=%d duration=%.2fs original_data=%d bytes",
        audio.channels,
        audio.frame_rate,
        len(audio) / 1000.0,
        len(audio.raw_data),
    )

    # Convert to 16kHz mono PCM (required by STT)
    audio_16k = audio.set_frame_rate(16000).set_channels(1)
    pcm_blob = audio_16k.raw_data
    logger.info(
        "Converted: %d bytes, %d samples, %.2fs at 16kHz mono",
        len(pcm_blob),
        len(pcm_blob) // 2,
        len(pcm_blob) // 2 / 16000.0,
    )

    # ---- Build components ----
    llm_base_url = os.environ.get("LLM_BASE_URL", "")
    llm_model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    llm_api_key = os.environ.get("LLM_API_KEY", "")
    stt_api_url = os.environ.get("STT_URI", "")
    stt_api_key = os.environ.get("STT_API_KEY", "")
    tts_api_url = os.environ.get("TTS_BASE_WEBSOCKET_API_URL", "wss://api.senseaudio.cn/ws/v1/t2a_v2")
    tts_api_key = os.environ.get("TTS_API_KEY", "")
    tts_voice = os.environ.get("TTS_VOICE", "female_0033_a")

    logger.info(
        "Config: llm=%s@%s stt=%s tts=%s voice=%s",
        llm_model,
        llm_base_url,
        stt_api_url,
        tts_api_url,
        tts_voice,
    )

    # Build LLM
    from livekit.plugins import openai as lk_openai
    from eidolon.livekit.agent.pipeline.llm import LlmParams, LivekitLlmStage

    llm_kwargs = {"model": llm_model}
    if llm_base_url:
        llm_kwargs["base_url"] = llm_base_url
    if llm_api_key:
        llm_kwargs["api_key"] = llm_api_key
    raw_llm = lk_openai.LLM(**llm_kwargs)
    llm_stage = LivekitLlmStage(raw_llm, params=LlmParams(model=llm_model, temperature=0.6))
    logger.info("LLM built: %s", type(llm_stage).__name__)

    # Build STT (Bailian)
    from eidolon.livekit.agent.pipeline.stt import SttParams, SttStage
    from eidolon.livekit.plugins.stt.bailian import BailianFunASRSTT
    from eidolon.livekit.plugins.stt.bailian.config import BailianSTTConfig

    if stt_api_url:
        bailian_cfg = BailianSTTConfig(api_url=stt_api_url, api_key=stt_api_key)
    else:
        bailian_cfg = BailianSTTConfig()
    stt_stage = SttStage(BailianFunASRSTT(config=bailian_cfg), params=SttParams(language="zh"))
    logger.info("STT built: %s", type(stt_stage).__name__)

    # Build TTS (SenseTime)
    from eidolon.livekit.agent.pipeline.tts import TtsParams, TtsStage
    from eidolon.livekit.plugins.tts.sensetime import SenseTimeTTS
    from eidolon.livekit.plugins.tts.sensetime.config import SenseTimeTTSConfig

    sensetime_cfg = SenseTimeTTSConfig(
        api_url=tts_api_url,
        api_key=tts_api_key,
        voice=tts_voice,
    )
    tts_stage = TtsStage(SenseTimeTTS(config=sensetime_cfg), params=TtsParams(voice=tts_voice))
    logger.info("TTS built: %s", type(tts_stage).__name__)

    # ---- Step 1: STT ----
    logger.info("=" * 60)
    logger.info("STEP 1: STT (FunASR Bailian)")
    logger.info("=" * 60)
    t0 = time.monotonic()
    transcript = await stt_stage.recognize(pcm_blob)
    stt_latency_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "STT result: %r", transcript
    )
    logger.info("STT latency: %.1f ms", stt_latency_ms)

    if not transcript.strip():
        logger.warning("STT returned empty transcript — skipping LLM/TTS")
        return

    # ---- Step 2: LLM ----
    logger.info("=" * 60)
    logger.info("STEP 2: LLM (Kimi-K2.6 via litellm)")
    logger.info("=" * 60)
    from eidolon.livekit.agent.pipeline.llm import LlmInput

    t1 = time.monotonic()
    llm_output = await llm_stage.chat(LlmInput(text=transcript))
    llm_latency_ms = (time.monotonic() - t1) * 1000
    response_text = llm_output.text if hasattr(llm_output, "text") else llm_output
    logger.info(
        "LLM result: %r", response_text
    )
    logger.info("LLM latency: %.1f ms", llm_latency_ms)

    # Strip markdown and emojis before TTS — SenseAudio only handles plain text
    import re

    # Remove markdown bold/italic/etc. markers
    response_clean = re.sub(r"[*_`#>\[\]]+", "", response_text)
    # Remove emojis
    response_clean = re.sub(r"[\U00010000-\U0010ffff]", "", response_clean)
    # Normalize whitespace
    response_clean = " ".join(response_clean.split())
    if response_clean != response_text:
        logger.info("TTS input (cleaned): %r", response_clean)
    else:
        logger.info("TTS input: %r", response_clean)

    if not response_text.strip():
        logger.warning("LLM returned empty response — skipping TTS")
        return

    # ---- Step 3: TTS ----
    logger.info("=" * 60)
    logger.info("STEP 3: TTS (SenseTime SenseAudio)")
    logger.info("=" * 60)
    t2 = time.monotonic()
    tts_frames = []

    # Test with hardcoded Chinese text first (same as integration tests)
    # then try the LLM response
    test_texts = [
        "好的，今天天气晴朗，适合外出散步。",
        response_clean,
    ]

    import logging as tts_log
    tts_log.getLogger("sensetime").setLevel(logging.DEBUG)
    tts_log.getLogger("sensetime.tts").setLevel(logging.DEBUG)
    tts_log.getLogger("sensetime.tts.client").setLevel(logging.DEBUG)

    tts_success = False
    for test_text in test_texts:
        logger.info("--- Testing TTS with: %r ---", test_text)
        # Each test needs a fresh stream instance
        stream = tts_stage.stream()
        stream.push_text(test_text)
        stream.end_input()
        frames_this_run = []
        try:
            async for event in stream:
                frames_this_run.append(event)
            total_samples = sum(ev.frame.samples_per_channel for ev in frames_this_run)
            tts_duration_s = total_samples / 16000.0
            logger.info(
                "TTS SUCCESS for %r: %d frames, %d samples, %.2fs audio",
                test_text[:30], len(frames_this_run), total_samples, tts_duration_s
            )
            tts_frames = frames_this_run
            tts_success = True
            break
        except Exception as e:
            logger.warning("TTS FAILED for %r: %s", test_text[:30], e)

    if tts_success:
        tts_latency_ms = (time.monotonic() - t2) * 1000
    else:
        tts_latency_ms = (time.monotonic() - t2) * 1000
        logger.warning("TTS failed, skipping summary")
        return

    total_samples = sum(ev.frame.samples_per_channel for ev in tts_frames)
    tts_duration_s = total_samples / 16000.0
    tts_latency_ms = (time.monotonic() - t2) * 1000

    # ---- Summary ----
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    total_ms = stt_latency_ms + llm_latency_ms + tts_latency_ms
    logger.info("Audio file: %s (%.2fs)", audio_path, len(audio) / 1000.0)
    logger.info("Audio processed: %.2fs at 16kHz mono", len(pcm_blob) / 2 / 16000.0)
    logger.info("STT:  %.1f ms  -> %r", stt_latency_ms, transcript[:100])
    logger.info("LLM:  %.1f ms  -> %r", llm_latency_ms, response_clean[:100])
    logger.info("TTS:  %.1f ms  -> %d frames (%.2fs audio)", tts_latency_ms, len(tts_frames), tts_duration_s)
    logger.info("Total: %.1f ms", total_ms)


if __name__ == "__main__":
    asyncio.run(main())
