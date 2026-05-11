"""LiveKit voice-pipeline plugins (STT / TTS / VAD / EOT).

Subpackages:

- ``stt/bailian/``   — Bailian (Aliyun DashScope) FunASR STT
- ``stt/sensetime/`` — SenseTime SenseAudio STT
- ``tts/sensetime/`` — SenseTime SenseAudio TTS
- ``vad/firered/``   — FireRedChat pVAD (speaker-adaptive)
- ``eot/``           — Eidolon End-of-Turn detection (FireRed Chat Turn Detector)

Import directly from the relevant subpackage, e.g.::

    from eidolon.livekit.plugins.stt.sensetime import SenseTimeSTT
    from eidolon.livekit.plugins.tts.sensetime import SenseTimeTTS
"""

from __future__ import annotations
