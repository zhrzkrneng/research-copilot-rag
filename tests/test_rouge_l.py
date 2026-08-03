"""Tests for RougeL."""

import pytest

from src.evaluation.rouge_l import RougeL


def test_metric_name():
    assert RougeL().name == "rouge_l"


def test_identical_answers_score_one():
    assert RougeL().compute(
        "retrieval augmented generation",
        "retrieval augmented generation",
    ) == 1.0


def test_normalization_is_applied():
    assert RougeL().compute(
        "The Retrieval-Augmented Generation!",
        "retrieval augmented generation",
    ) == 1.0


def test_partial_overlap_uses_sequence_order():
    assert RougeL().compute(
        "alpha beta gamma",
        "alpha gamma delta",
    ) == pytest.approx(2.0 / 3.0)


def test_reordered_tokens_are_not_full_match():
    assert RougeL().compute(
        "alpha beta gamma",
        "gamma beta alpha",
    ) == pytest.approx(1.0 / 3.0)


def test_no_overlap_scores_zero():
    assert RougeL().compute("alpha beta", "gamma delta") == 0.0


def test_both_empty_score_one():
    assert RougeL().compute("", "   ") == 1.0


@pytest.mark.parametrize(
    ("prediction", "reference"),
    [("answer", ""), ("", "answer")],
)
def test_one_empty_answer_scores_zero(prediction, reference):
    assert RougeL().compute(prediction, reference) == 0.0


def test_repeated_tokens_are_handled():
    assert RougeL().compute(
        "alpha alpha beta",
        "alpha beta beta",
    ) == pytest.approx(2.0 / 3.0)


def test_articles_and_punctuation_are_normalized():
    assert RougeL().compute(
        "The model, works.",
        "model works",
    ) == 1.0


def test_unicode_text():
    assert RougeL().compute(
        "مدل بازیابی خوب است",
        "مدل بازیابی خوب است",
    ) == 1.0


def test_compute_is_deterministic():
    metric = RougeL()
    first = metric.compute("alpha beta gamma", "alpha gamma")
    second = metric.compute("alpha beta gamma", "alpha gamma")
    assert first == second


def test_prediction_must_be_string():
    with pytest.raises(TypeError):
        RougeL().compute(1, "answer")


def test_reference_must_be_string():
    with pytest.raises(TypeError):
        RougeL().compute("answer", None)


@pytest.mark.parametrize(
    ("prediction_tokens", "reference_tokens", "expected"),
    [
        ([], [], 0),
        (["a"], [], 0),
        ([], ["a"], 0),
        (["a"], ["a"], 1),
        (["a", "b", "c"], ["a", "c"], 2),
        (["a", "b", "c"], ["c", "b", "a"], 1),
        (
            ["retrieval", "augmented", "generation"],
            ["dense", "retrieval", "generation"],
            2,
        ),
    ],
)
def test_lcs_length(
    prediction_tokens,
    reference_tokens,
    expected,
):
    assert RougeL._lcs_length(
        prediction_tokens,
        reference_tokens,
    ) == expected
