import pytest

from src.retrieval.result import RetrievalResult


def test_result_has_score():
    result = RetrievalResult(
        chunk=None,
        score=0.95,
    )

    assert result.score == 0.95
