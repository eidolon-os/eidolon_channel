from __future__ import annotations

import json
import re
from pathlib import Path

_PROVIDER_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST = _PROVIDER_ROOT / "requirements" / "ph4-a.json"


def test_every_ph4_a_requirement_has_distinct_executable_positive_and_negative_tests() -> None:
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    requirements = manifest["requirements"]
    ids = [item["id"] for item in requirements]
    discovered: set[str] = set()
    for path in Path(__file__).parent.glob("test_*.py"):
        discovered.update(
            re.findall(
                r"^(?:async )?def (test_[a-zA-Z0-9_]+)",
                path.read_text(encoding="utf-8"),
                re.MULTILINE,
            )
        )

    assert len(ids) == len(set(ids))
    for requirement in requirements:
        assert requirement["normative_ref"]
        assert requirement["positive_test"] != requirement["negative_test"]
        assert requirement["positive_test"] in discovered
        assert requirement["negative_test"] in discovered
