# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FireRedChat pVAD plugin for eidolon LiveKit.

This plugin provides a speaker-adaptive Voice Activity Detector based on the
FireRedChat pVAD model. It implements the ``livekit.agents.vad.VAD`` interface
and is wrapped by the pipeline-level :class:`VadStage` for lifecycle / DI.

Usage::

    from eidolon.livekit.plugins.vad.firered import FireredPvadVAD

    vad = FireredPvadVAD.load(
        activation_threshold=0.5,
        min_speech_duration=0.1,
        min_silence_duration=0.4,
    )
    stream = vad.stream()

    # After the first user utterance (~1-5s), adapt to the speaker:
    stream.update_speaker(audio_16k_samples)
"""

from __future__ import annotations

from . import processor
from .config import FireredPvadConfig
from .processor import PvadProcessor, SpeakerEmbExtractor
from .vad import VAD as FireredPvadVAD
from .version import __version__

__all__ = [
    "FireredPvadVAD",
    "FireredPvadConfig",
    "PvadProcessor",
    "SpeakerEmbExtractor",
    "register_plugin",
    "__version__",
]


def __getattr__(name: str):
    # Lazy-load the plugin wrapper class to avoid registration on import.
    # Use register_plugin() from the main thread instead.
    if name == "FireRedPvadPlugin":
        from ._plugin import FireRedPvadPlugin

        return FireRedPvadPlugin

    if name == "register_plugin":
        from ._plugin import register_plugin

        return register_plugin

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
