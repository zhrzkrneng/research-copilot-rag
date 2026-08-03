"""Tests for ExactMatch."""

from __future__ import annotations

import pytest

from src.evaluation.exact_match import ExactMatch


def test_metric_name():
    assert ExactMatch().name == "exact_match"


@pytest.mark.parametrize(
    ("prediction", "reference"),
    [
        ("answer", "answer"),
        (" answer ", "answer"),
        ("answer", " answer "),
    ],
)
def test_exact_match_returns_one(
    prediction,
    reference,
):
    assert (
        ExactMatch().compute(
            prediction,
            reference,
        )
        == 1.0
    )


@pytest.mark.parametrize(
    ("prediction", "reference"),
    [
        ("answer", "different"),
        ("abc", "ABC"),
        ("one two", "one"),
    ],
)
def test_exact_match_returns_zero(
    prediction,
    reference,
):
    assert (
        ExactMatch().compute(
            prediction,
            reference,
        )
        == 0.0
    )


def test_prediction_must_be_string():
    with pytest.raises(TypeError):
        ExactMatch().compute(1, "x")


def test_reference_must_be_string():
    with pytest.raises(TypeError):
        ExactMatch().compute("x", None)
