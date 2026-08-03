"""Tests for shared answer normalization."""

import pytest

from src.evaluation.normalization import normalize_answer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The Quick Brown Fox", "quick brown fox"),
        ("An answer.", "answer"),
        ("A   spaced\tanswer", "spaced answer"),
        ("Hello, world!", "hello world"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_normalize_answer(text, expected):
    assert normalize_answer(text) == expected


def test_normalize_answer_rejects_non_string():
    with pytest.raises(TypeError):
        normalize_answer(None)
