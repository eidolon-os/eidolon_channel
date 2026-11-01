"""The EOT tokenizer, and what loading it may cost a job process.

Ids below were produced by transformers' AutoTokenizer (left truncation, 128)
on the same tokenizer files, before the backend stopped importing it.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from eidolon.livekit.plugins.eot.impl.eot_backend import MAX_HISTORY_TOKENS, OnnxEotBackend
from eidolon.livekit.plugins.eot.impl.eot_manager import _resolve_default_model_dir


@pytest.fixture(scope="module")
def backend() -> OnnxEotBackend:
    return OnnxEotBackend(_resolve_default_model_dir())


@pytest.mark.parametrize(
    ("text", "ids"),
    [
        ("今天天气怎么样", [101, 2187, 3198, 3198, 4853, 3814, 2118, 4577, 102]),
        (
            "帮我把明天早上八点的会议改到九点",
            [101, 3620, 3976, 4033, 4368, 3198, 4352, 2078, 2451, 5286, 5718, 2226, 7306,
             4282, 2555, 2133, 5286, 102],
        ),
        ("Hello World ABC 123", [101, 31178, 10315, 16052, 16968, 102]),
    ],
)
def test_ids_are_the_ones_the_model_was_scored_with(backend, text, ids) -> None:
    input_ids, attention_mask = backend._encode(text)
    assert input_ids.tolist() == [ids]
    assert attention_mask.tolist() == [[1] * len(ids)]
    assert input_ids.dtype == np.int64 and attention_mask.dtype == np.int64


def test_a_long_history_keeps_its_latest_tokens(backend) -> None:
    input_ids, attention_mask = backend._encode("一二三四五" * 40)
    ids = input_ids[0].tolist()
    assert len(ids) == MAX_HISTORY_TOKENS == attention_mask.shape[1]
    # [CLS] and [SEP] stay; what is dropped is the oldest speech, not the end.
    assert ids[:3] == [101, 2155, 2072] and ids[-3:] == [2998, 2155, 102]


def test_scoring_still_runs_end_to_end(backend) -> None:
    score = backend.score("帮我把明天早上八点的会议改到九点。")
    assert 0.0 < score <= 1.0


def test_loading_the_turn_models_does_not_import_torch() -> None:
    """A job process is started per conversation; this is on its critical path.

    transformers (for the tokenizer) pulled torch in: 2.5-7 s and ~0.5 GB per
    job process until 11-01, which is why a Host kept a single warm process and
    a second conversation within that window waited for a new one.
    """

    probe = (
        "import sys\n"
        "import eidolon.livekit.agent.server\n"
        "from eidolon.livekit.plugins.eot import ChineseModel\n"
        "from eidolon.livekit.plugins.vad.firered import FireredPvadVAD\n"
        "ChineseModel(); FireredPvadVAD.load()\n"
        "heavy = sorted(m for m in ('torch', 'torchaudio', 'transformers') if m in sys.modules)\n"
        "print('HEAVY', ','.join(heavy))\n"
    )
    result = subprocess.run(
        (sys.executable, "-c", probe), capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr[-2000:]
    line = next(row for row in result.stdout.splitlines() if row.startswith("HEAVY"))
    assert line == "HEAVY ", line
