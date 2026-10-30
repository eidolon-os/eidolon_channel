"""WAV files this Host hands out, optionally carrying the AIGC implicit identifier.

The identifier is a RIFF chunk named ``AIGC`` holding the GB 45438-2025 Annex E
fields (``Label``, ``ContentProducer``, ``ProduceID``, ``ContentPropagator``,
``PropagateID``) — the same chunk a TTS provider writes when asked. Written here,
once, for every provider, rather than requested from each one; it changes no sample.
"""

from __future__ import annotations

import json
import struct


def wav_bytes(pcm: bytes, sample_rate: int, *, aigc: dict[str, str] | None = None) -> bytes:
    fmt = struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16)
    chunks = [b"fmt " + struct.pack("<I", len(fmt)) + fmt]
    if aigc is not None:
        body = json.dumps({"Label": "1", **aigc}, ensure_ascii=False, separators=(",", ":")).encode()
        chunks.append(b"AIGC" + struct.pack("<I", len(body)) + body + (b"\x00" if len(body) % 2 else b""))
    chunks.append(b"data" + struct.pack("<I", len(pcm)) + pcm)
    body = b"WAVE" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def wav_pcm(data: bytes) -> tuple[bytes, int]:
    """PCM16 mono samples and rate of a WAV this module (or a provider) wrote."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    offset, rate = 12, None
    while offset + 8 <= len(data):
        cid, size = data[offset:offset + 4], struct.unpack("<I", data[offset + 4:offset + 8])[0]
        body = data[offset + 8:offset + 8 + size]
        if cid == b"fmt ":
            audio_format, channels, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            if (audio_format, channels, bits) != (1, 1, 16):
                raise ValueError("expected PCM16 mono")
        elif cid == b"data":
            if rate is None:
                raise ValueError("WAV data before format")
            # Streamed WAV headers carry a placeholder length; the data runs to the end.
            return data[offset + 8:offset + 8 + min(size, len(data) - offset - 8)], rate
        offset += 8 + size + (size & 1)
    raise ValueError("WAV without data")
