"""Shared answer normalization utilities."""

from __future__ import annotations

import re
import string


def normalize_answer(text: str) -> str:
    """Normalize answer text for deterministic evaluation."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    text = text.lower()

    punctuation_to_spaces = str.maketrans(
        {character: " " for character in string.punctuation}
    )
    text = text.translate(punctuation_to_spaces)

    text = re.sub(r"\b(a|an|the)\b", " ", text)

    return " ".join(text.split())