"""Registration windows: same-person evidence registration gives for free.

Design §3.3. Every registration recording is also cut into windows anchored
to its voiced span; each window is embedded like a conversation probe. A
sample's windows are scored against the mean of the other samples and the
trial's against the full template, so no window meets a template built from
itself. The household adaptation (plan 1c) uses these as in-domain
same-person trials.
"""

from __future__ import annotations

import numpy as np
from eidolon_sdk.biz.voiceprint import (
    CalibrationCrop,
    GenuineScore,
    TemplateCalibration,
)

SAMPLE_RATE = 16000
# Raw window lengths; the band is decided by the voiced speech the extractor
# measures in each window, exactly as for a conversation probe.
WINDOW_SECONDS = (1.0, 2.0, 3.5)
CROP_MIN_SPEECH_MS = 600


def calibration_windows(total_samples: int, speech: tuple[int, int] | None) -> list[tuple[int, int]]:
    """Windows anchored to the start and the end of the voiced span.

    Two per length, so a short registration still yields windows in every
    band it can reach; identical windows collapse.
    """
    if speech is None:
        return []
    first, last = speech
    windows = []
    for seconds in WINDOW_SECONDS:
        size = int(seconds * SAMPLE_RATE)
        if size > total_samples:
            continue
        start = max(0, min(first, total_samples - size))
        end = min(total_samples, max(last, size))
        for window in ((start, start + size), (end - size, end)):
            if window not in windows:
                windows.append(window)
    return windows


def crop(speech_ms: int, vector) -> CalibrationCrop:
    return CalibrationCrop(speech_ms=speech_ms, vector=[float(x) for x in vector])


def _unit(rows) -> np.ndarray:
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0, norms, 1)


def enrollment_calibration(samples, trial, template) -> TemplateCalibration:
    """Genuine scores whose window never contributed to the template it meets.

    A registration sample's windows meet the mean of the other two samples;
    the trial's windows meet the full template (the trial is not part of it).
    """
    vectors = [np.asarray(s.vector, dtype=np.float64) for s in samples]
    genuine = []
    crops = []
    for index, sample in enumerate(samples):
        others = _unit(np.mean([v for i, v in enumerate(vectors) if i != index], axis=0))[0]
        for window in sample.crops:
            crops.append(window)
            genuine.append((window.speech_ms, float(_unit(window.vector)[0] @ others)))
    full = _unit(template.vector)[0]
    for window in trial.crops:
        crops.append(window)
        genuine.append((window.speech_ms, float(_unit(window.vector)[0] @ full)))
    return TemplateCalibration(
        crops=crops[:32],
        genuine=[GenuineScore(speech_ms=ms, similarity=max(-1.0, min(1.0, s)))
                 for ms, s in genuine[:32]],
    )
