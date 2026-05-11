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

"""FireRedChat pVAD processor: ONNX inference + ECAPA speaker embedding.

Architecture notes:
- ONNX session is lazily loaded on first use (not in __init__) to avoid blocking
  the event loop during module import.
- SpeakerEmbExtractor degrades gracefully: if speechbrain is missing or the ECAPA
  model is not present, it returns a zero vector instead of crashing.
- Model paths can be overridden via the EIDOLON_FIRERED_PVAD_MODEL_DIR env var.
"""

from __future__ import annotations

import os
import threading
from functools import cached_property
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

from .log import logger

# ------------------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------------------

# Number of audio samples per inference window (10 ms at 16 kHz)
WINDOW_SIZE_SAMPLES = 160

# Number of samples used for speaker embedding extraction (~1 s at 16 kHz)
ECAPA_WINDOW_SAMPLES = 16000

# Number of samples used for speaker embedding in the original plugin (~5 s)
# We use 1 s to reduce latency and memory usage; the original 5 s is available
# as SPKEMB_WINDOW_SAMPLES_ORIGINAL
SPKEMB_WINDOW_SAMPLES_ORIGINAL = 80000

# ------------------------------------------------------------------------------------------
# SpeakerEmbExtractor
# ------------------------------------------------------------------------------------------


class SpeakerEmbExtractor:
    """Extracts speaker embeddings using ECAPA-VoxCeleb (SpeechBrain).

    Gracefully degrades when speechbrain is not installed or the model is missing,
    returning a zero vector so the VAD can still function without speaker adaptation.
    """

    def __init__(
        self,
        ckpt_path: Optional[str | Path] = None,
        device: str = "cpu",
    ) -> None:
        self._classifier: Optional[object] = None
        self._device = device
        self._loaded = False

        if ckpt_path is None:
            logger.debug("SpeakerEmbExtractor: no ckpt_path provided, embedding disabled")
            return

        resolved = Path(ckpt_path).resolve()
        if not resolved.is_dir():
            logger.warning(
                "SpeakerEmbExtractor: path does not exist or is not a directory: %s",
                resolved,
            )
            return

        self._try_load(resolved)

    def _try_load(self, ckpt_path: Path) -> None:
        """Attempt to load the SpeechBrain ECAPA model. Logs warnings on failure."""
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            logger.warning(
                "SpeakerEmbExtractor: speechbrain not installed. "
                "Speaker adaptation will use a zero vector. "
                "Install with: pip install speechbrain"
            )
            return
        except Exception as e:
            logger.warning("SpeakerEmbExtractor: import error: %s", e)
            return

        try:
            device_str = "cpu" if self._device == "cpu" else f"cuda:{self._device}"
            self._classifier = EncoderClassifier.from_hparams(
                source=str(ckpt_path),
                savedir=str(ckpt_path),
                run_opts={"device": device_str},
            )
            self._loaded = True
            logger.info("SpeakerEmbExtractor: loaded from %s", ckpt_path)
        except Exception as e:
            logger.warning(
                "SpeakerEmbExtractor: failed to load ECAPA model from %s: %s",
                ckpt_path,
                e,
            )

    @property
    def is_loaded(self) -> bool:
        """Whether the ECAPA model was successfully loaded."""
        return self._loaded

    def get_embedding(self, input_signal: np.ndarray) -> np.ndarray:
        """Extract a normalized speaker embedding from audio.

        Args:
            input_signal: Audio samples as float32 array. Can be (samples,) or
                (batch, samples). Expected sample rate: 16 kHz.

        Returns:
            A (1, 192) normalized embedding array. Returns a zero vector if
            the model is not loaded or extraction fails.
        """
        if self._classifier is None:
            return np.zeros((1, 192), dtype=np.float32)

        try:
            import torch

            if input_signal.ndim == 1:
                input_signal = input_signal.reshape(1, -1)

            t = torch.from_numpy(input_signal.astype(np.float32))
            emb = self._classifier.encode_batch(t)[0][0].detach()
            emb = emb / emb.norm(p=2, dim=0, keepdim=True)
            return emb.cpu().unsqueeze(0).numpy()
        except Exception as e:
            logger.debug("SpeakerEmbExtractor.get_embedding failed: %s", e)
            return np.zeros((1, 192), dtype=np.float32)


# ------------------------------------------------------------------------------------------
# PvadProcessor
# ------------------------------------------------------------------------------------------


class PvadProcessor:
    """Streaming pVAD processor backed by the FireRedChat pvad.onnx model.

    This class is NOT tied to the livekit.agents.vad interface — it is a standalone
    inference engine that can be used by any caller. It is safe to share a single
    instance across multiple concurrent audio streams as long as each stream
    maintains its own set of state buffers.

    The ONNX session is loaded lazily on first use to avoid blocking imports.
    """

    # --------------------------------------------------------------------------
    # Construction / lazy loading
    # --------------------------------------------------------------------------

    def __init__(
        self,
        model_dir: Optional[str | Path] = None,
        force_cpu: bool = True,
    ) -> None:
        """
        Args:
            model_dir: Path to the directory containing pvad.onnx and
                spkrec-ecapa-voxceleb/. Defaults to the resources/ subdirectory
                of this module's location. Can also be overridden via the
                EIDOLON_FIRERED_PVAD_MODEL_DIR environment variable.
            force_cpu: If True, only use CPUExecutionProvider. If False, prefer
                CUDA when available.
        """
        self._model_dir: Optional[Path] = self._resolve_model_dir(model_dir)
        self._force_cpu = force_cpu
        self._session: Optional[ort.InferenceSession] = None
        self._spk_extractor: Optional[SpeakerEmbExtractor] = None

        self.window_size_samples = WINDOW_SIZE_SAMPLES
        self.sample_rate = 16000

        # Streaming state — these are reset via reset() and are per-stream
        self.mel_buffer: np.ndarray = np.zeros((1, 80, 15), dtype=np.float32)
        self.gru_buffer: np.ndarray = np.zeros((2, 1, 256), dtype=np.float32)
        self.spkemb: np.ndarray = np.zeros((1, 192), dtype=np.float32)
        # Guards concurrent access to spkemb (the singleton processor is
        # shared across all VAD streams; speaker updates from one stream
        # could otherwise tear with reads in another stream's inference).
        self._spkemb_lock = threading.Lock()

    @staticmethod
    def _resolve_model_dir(model_dir: Optional[str | Path]) -> Optional[Path]:
        """Resolve the model directory, checking env var, explicit path, and resources/."""
        # 1. Environment variable override
        env_path = os.environ.get("EIDOLON_FIRERED_PVAD_MODEL_DIR")
        if env_path:
            resolved = Path(env_path).resolve()
            if resolved.is_dir():
                logger.info(
                    "PvadProcessor: using model_dir from EIDOLON_FIRERED_PVAD_MODEL_DIR: %s",
                    resolved,
                )
                return resolved
            logger.warning(
                "EIDOLON_FIRERED_PVAD_MODEL_DIR is set but is not a directory: %s",
                resolved,
            )

        # 2. Explicit model_dir argument
        if model_dir:
            resolved = Path(model_dir).resolve()
            if resolved.is_dir():
                return resolved
            raise FileNotFoundError(
                f"model_dir does not exist or is not a directory: {resolved}"
            )

        # 3. Default to resources/ relative to this file
        default = Path(__file__).parent.resolve() / "resources"
        if default.is_dir():
            return default

        raise FileNotFoundError(
            f"FireRedChat pVAD model directory not found. "
            f"Set EIDOLON_FIRERED_PVAD_MODEL_DIR or pass model_dir. "
            f"Checked: {env_path!r}, {model_dir!r}, {default}"
        )

    @cached_property
    def onnx_session(self) -> ort.InferenceSession:
        """Lazily create and cache the ONNX Runtime session."""
        if self._model_dir is None:
            raise RuntimeError("PvadProcessor model directory not resolved")

        onnx_path = self._model_dir / "pvad.onnx"
        if not onnx_path.is_file():
            raise FileNotFoundError(
                f"pvad.onnx not found at {onnx_path}. "
                f"Please download the model from FireRedTeam/FireRedChat-pvad on HuggingFace."
            )

        opts = ort.SessionOptions()
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if self._force_cpu:
            providers = ["CPUExecutionProvider"]
        else:
            available = ort.get_available_providers()
            providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available]

        logger.info("PvadProcessor: loading ONNX model from %s (providers=%s)", onnx_path, providers)
        session = ort.InferenceSession(
            str(onnx_path),
            providers=providers,
            sess_options=opts,
        )
        logger.info("PvadProcessor: ONNX model loaded successfully")
        return session

    @cached_property
    def spk_extractor(self) -> SpeakerEmbExtractor:
        """Lazily create and cache the SpeakerEmbExtractor."""
        if self._model_dir is None:
            raise RuntimeError("PvadProcessor model directory not resolved")

        spk_path = self._model_dir / "spkrec-ecapa-voxceleb"
        extractor = SpeakerEmbExtractor(
            str(spk_path) if spk_path.is_dir() else None,
            device="cpu" if self._force_cpu else 0,
        )
        logger.info(
            "PvadProcessor: SpeakerEmbExtractor %s (loaded=%s)",
            "loaded" if extractor.is_loaded else "using zero vectors",
            extractor.is_loaded,
        )
        return extractor

    # --------------------------------------------------------------------------
    # Public inference API
    # --------------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all streaming state buffers to zero.

        Call this at the start of a new session or turn.
        """
        self.mel_buffer = np.zeros((1, 80, 15), dtype=np.float32)
        self.gru_buffer = np.zeros((2, 1, 256), dtype=np.float32)
        with self._spkemb_lock:
            self.spkemb = np.zeros((1, 192), dtype=np.float32)

    def __call__(self, wav_np: np.ndarray) -> float:
        """Run inference on one audio window.

        Args:
            wav_np: Audio samples as a float32 array with shape (WINDOW_SIZE_SAMPLES,)
                or (1, WINDOW_SIZE_SAMPLES). Values should be in [-1, 1].

        Returns:
            The speech probability for this window, a float in [0.0, 1.0].
            Returns 0.0 if the input shape is incorrect.
        """
        expected = self.window_size_samples
        if wav_np.ndim > 1:
            wav_np = wav_np.reshape(-1)
        if wav_np.shape[0] != expected:
            logger.debug(
                "PvadProcessor: expected %d samples, got %d — returning 0.0",
                expected,
                wav_np.shape[0],
            )
            return 0.0

        wav_np = wav_np.reshape(1, expected).astype(np.float32)

        # Snapshot spkemb under lock so a concurrent update_speaker_embedding
        # call doesn't tear the array we're feeding into ONNX.
        with self._spkemb_lock:
            spkemb = self.spkemb

        outputs = self.onnx_session.run(
            None,
            {
                "input_audio": wav_np,
                "spkemb": spkemb,
                "mel_buffer": self.mel_buffer,
                "gru_buffer": self.gru_buffer,
            },
        )
        raw_prob = float(outputs[1][0].tolist()[0])
        self.mel_buffer = outputs[2]  # (1, 80, 15)
        self.gru_buffer = outputs[3]  # (2, 1, 256)
        return raw_prob

    def update_speaker_embedding(self, audio_16k_f32: np.ndarray) -> None:
        """Update the speaker embedding used by subsequent inference calls.

        This enables speaker-adaptive VAD — the model becomes more sensitive to
        the target speaker's voice characteristics and more robust to other speakers
        or background noise.

        Args:
            audio_16k_f32: Audio samples at 16 kHz as a float32 array. Values
                should be in [-1, 1]. Recommended length: 1–5 seconds.
                If longer, only the most recent SPKEMB_WINDOW_SAMPLES_ORIGINAL
                samples are used.
        """
        if audio_16k_f32.ndim == 0:
            return

        if audio_16k_f32.ndim > 1:
            audio_16k_f32 = audio_16k_f32.reshape(-1)

        if audio_16k_f32.shape[0] < 1600:  # less than 100 ms — too short
            logger.debug("update_speaker_embedding: audio too short (%d samples)", audio_16k_f32.shape[0])
            return

        # Use the most recent N seconds if audio is very long
        if audio_16k_f32.shape[0] > SPKEMB_WINDOW_SAMPLES_ORIGINAL:
            audio_16k_f32 = audio_16k_f32[-SPKEMB_WINDOW_SAMPLES_ORIGINAL:]

        new_spkemb = self.spk_extractor.get_embedding(audio_16k_f32)
        with self._spkemb_lock:
            self.spkemb = new_spkemb
        logger.info("PvadProcessor: speaker embedding updated (shape=%s)", new_spkemb.shape)
