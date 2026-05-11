"""Generate real TTS audio from SenseAudio and save as WAV.

Usage::

    PYTHONPATH=.venv/bin/python scripts/play_tts.py
"""
import asyncio
import json
import struct


def write_wav(filename: str, pcm_data: bytes, sample_rate: int = 16000, num_channels: int = 1, bits_per_sample: int = 16):
    with open(filename, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(pcm_data)))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))   # chunk size
        f.write(struct.pack("<H", 1))    # PCM
        f.write(struct.pack("<H", num_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * num_channels * bits_per_sample // 8))  # byte rate
        f.write(struct.pack("<H", num_channels * bits_per_sample // 8))  # block align
        f.write(struct.pack("<H", bits_per_sample))
        f.write(b"data")
        f.write(struct.pack("<I", len(pcm_data)))
        f.write(pcm_data)


async def main():
    import sys
    from pathlib import Path

    _repo_root = Path(__file__).resolve().parents[6]
    sys.path.insert(0, str(_repo_root))
    from eidolon.livekit.plugins.tts.sensetime import (
        SenseTimeTTSClient,
        SenseTimeTTSConfig,
    )

    config = SenseTimeTTSConfig()
    text = "你好，欢迎使用语音合成系统，今天天气真不错！"

    print(f"Connecting to: {config.api_url}")
    print(f"Text: {text}")

    # Collect all audio chunks
    audio_chunks: list[bytes] = []

    async def on_message(msg: dict) -> None:
        event = msg.get("event", "")
        data = msg.get("data") or {}

        if event == "task_continued":
            audio_hex = data.get("audio", "")
            if audio_hex:
                raw = bytes.fromhex(audio_hex)
                if raw and raw != b"\x00" * len(raw):  # skip pure silence
                    audio_chunks.append(raw)
                    print(f"  audio chunk: {len(raw)} bytes, is_final={msg.get('is_final', False)}")

            if msg.get("is_final"):
                print(f"  task_continued is_final=True, extra_info: {msg.get('extra_info')}")

        elif event == "task_finished":
            print("  task_finished received")

        elif event == "task_failed":
            print(f"  task_failed: {msg}")

    client = SenseTimeTTSClient(
        uri=config.api_url,
        api_key=config.api_key,
        model=config.model,
        voice_id=config.voice,
        sample_rate=config.sample_rate,
        speed=config.speed,
        vol=config.vol,
        pitch=config.pitch,
        on_message_callback=on_message,
    )

    connected = await client.connect()
    if not connected:
        print("Failed to connect!")
        return

    print("Connected, sending task_start...")
    await client.send_task_start()

    # Wait for task_started
    await asyncio.sleep(0.5)

    print("Sending task_continue...")
    await client.send_task_continue(text)

    # Wait for all audio to arrive
    print("Waiting for audio...")
    await asyncio.sleep(5.0)

    await client.disconnect()

    if audio_chunks:
        pcm_data = b"".join(audio_chunks)
        duration = len(pcm_data) / 2 / config.sample_rate
        print(f"\nTotal audio: {len(pcm_data)} bytes, {duration:.2f}s")

        wav_path = "/tmp/sensetime_tts_real.wav"
        write_wav(wav_path, pcm_data, sample_rate=config.sample_rate)
        print(f"Saved to: {wav_path}")
    else:
        print("\nNo audio chunks received!")


if __name__ == "__main__":
    asyncio.run(main())
