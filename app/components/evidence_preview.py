"""Shared compact preview logic for evidence/source text."""

from __future__ import annotations

import re


_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?])\s*|(?<=[.!?])\s+")


def build_evidence_preview(
    text: str,
    *,
    max_sentences: int = 2,
    max_chars: int = 260,
) -> tuple[str, bool]:
    """Return a 1–2 sentence preview and whether the full text is longer.

    English and Chinese sentence endings are both supported.  Academic source
    sentences can be very long, so ``max_chars`` remains a hard readability cap.
    """

    normalized = " ".join((text or "").split())
    if not normalized:
        return "", False
    if len(normalized) <= max_chars:
        return normalized, False

    sentences = [part.strip() for part in _SENTENCE_SPLIT.split(normalized) if part.strip()]
    if sentences:
        selected: list[str] = []
        for sentence in sentences[:max_sentences]:
            candidate = " ".join((*selected, sentence)).strip()
            if len(candidate) > max_chars:
                break
            selected.append(sentence)
        if selected:
            preview = " ".join(selected).strip()
            if preview != normalized:
                return f"{preview.rstrip()}…", True

    # One very long sentence (common in papers): retain a compact character cap.
    compact = normalized[:max_chars].rstrip()
    return f"{compact}…", True
