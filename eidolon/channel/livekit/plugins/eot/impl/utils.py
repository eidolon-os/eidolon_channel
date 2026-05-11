# Copyright 2025 Eidolon Team
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

"""
Turn Detection utility functions.
Copied from eidolon/pipeline/src/processor/turn_detector/utils.py
"""

from __future__ import annotations

import hashlib


def longest_common_prefix_len(a: str, b: str) -> int:
    """Calculate longest common prefix length of two strings."""
    min_len = min(len(a), len(b))
    for i in range(min_len):
        if a[i] != b[i]:
            return i
    return min_len


def extract_incremental_text(full: str, prev: str) -> str:
    """
    Extract incremental text (new part).

    Args:
        full: Full text
        prev: Previous text

    Returns:
        New part text
    """
    if not prev:
        return full

    prefix_len = longest_common_prefix_len(full, prev)
    return full[prefix_len:]


def is_similar_text(text1: str, text2: str, threshold: float = 0.85) -> bool:
    """
    Check if two texts are similar.

    Args:
        text1: First text
        text2: Second text
        threshold: Similarity threshold

    Returns:
        Whether they are similar
    """
    if not text1 or not text2:
        return text1 == text2

    prefix_len = longest_common_prefix_len(text1, text2)
    max_len = max(len(text1), len(text2))

    if max_len == 0:
        return True

    similarity = prefix_len / max_len
    return similarity >= threshold


def compute_text_hash(text: str) -> str:
    """Compute MD5 hash of text."""
    return hashlib.md5(text.encode()).hexdigest()


def format_duration(seconds: float) -> str:
    """Format duration to human-readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    else:
        return f"{seconds:.2f}s"
