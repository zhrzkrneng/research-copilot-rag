"""Tests for TokenF1."""

import pytest

from src.evaluation.token_f1 import TokenF1


def test_metric_name():
    assert TokenF1().name == "token_f1"


def test_identical_answers_score_one():
    assert TokenF1().compute(
        "retrieval augmented generation",
        "retrieval augmented generation",
    ) == 1.0


def test_normalization_is_applied():
    assert TokenF1().compute(
        "The Retrieval-Augmented Generation!",
        "retrieval augmented generation",
    ) == 1.0


def test_partial_overlap():
    assert TokenF1().compute(
        "retrieval augmented generation",
        "retrieval generation system",
    ) == pytest.approx(2.0 / 3.0)


def test_no_overlap_scores_zero():
    assert TokenF1().compute("alpha beta", "gamma delta") == 0.0


def test_both_empty_score_one():
    assert TokenF1().compute("", "   ") == 1.0


@pytest.mark.parametrize(
    ("prediction", "reference"),
    [("answer", ""), ("", "answer")],
)
def test_one_empty_answer_scores_zero(prediction, reference):
    assert TokenF1().compute(prediction, reference) == 0.0


def test_repeated_tokens_use_multiplicity():
    assert TokenF1().compute(
        "alpha alpha beta",
        "alpha beta beta",
    ) == pytest.approx(2.0 / 3.0)


def test_prediction_must_be_string():
    with pytest.raises(TypeError):
        TokenF1().compute(1, "answer")


def test_reference_must_be_string():
    with pytest.raises(TypeError):
        TokenF1().compute("answer", None)
