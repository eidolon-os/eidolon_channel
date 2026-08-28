"""Pure transcript-revision comparison shared by turn assembly boundaries."""

from __future__ import annotations

import unicodedata


DEFAULT_TRANSCRIPT_REVISION_MIN_NORMALIZED_CHARS = 4


def transcript_revision_matches(
    existing: str,
    revision: str,
    *,
    min_normalized_chars: int = DEFAULT_TRANSCRIPT_REVISION_MIN_NORMALIZED_CHARS,
) -> bool:
    """Return whether two texts can be revisions of one STT hypothesis."""

    existing_norm = normalize_revision_text(existing)
    revision_norm = normalize_revision_text(revision)
    if not existing_norm or not revision_norm:
        return False
    if existing_norm == revision_norm:
        return True
    if existing_norm.startswith(revision_norm) or revision_norm.startswith(existing_norm):
        return True
    if min(len(existing_norm), len(revision_norm)) < max(1, min_normalized_chars):
        return False
    shorter, longer = (
        (existing_norm, revision_norm)
        if len(existing_norm) <= len(revision_norm)
        else (revision_norm, existing_norm)
    )
    return shorter in longer


def normalized_text_equal(left: str, right: str) -> bool:
    left_norm = normalize_revision_text(left)
    right_norm = normalize_revision_text(right)
    return bool(left_norm and right_norm and left_norm == right_norm)


def normalize_revision_text(text: str) -> str:
    return "".join(
        char
        for char in text.strip().lower()
        if not unicodedata.category(char).startswith(("P", "Z"))
    )
