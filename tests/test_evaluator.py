"""Tests for Evaluator."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from src.evaluation.base_metric import BaseMetric
from src.evaluation.evaluator import Evaluator
from src.evaluation.exact_match import ExactMatch
from src.evaluation.rouge_l import RougeL
from src.evaluation.token_f1 import TokenF1


class ConstantMetric(BaseMetric):
    """Deterministic metric used for evaluator tests."""

    def __init__(
        self,
        metric_name: str,
        score: float,
    ) -> None:
        self._name = metric_name
        self._score = score

    @property
    def name(self) -> str:
        return self._name

    def compute(
        self,
        prediction: str,
        reference: str,
    ) -> float:
        return self._score


def build_default_evaluator() -> Evaluator:
    return Evaluator(
        metrics=[
            ExactMatch(),
            TokenF1(),
            RougeL(),
        ]
    )


def test_constructor_preserves_metric_order():
    metrics: Sequence[BaseMetric] = [
        ExactMatch(),
        TokenF1(),
        RougeL(),
    ]

    evaluator = Evaluator(metrics)

    assert evaluator.metrics == tuple(metrics)


def test_constructor_copies_metric_sequence():
    metrics = [
        ExactMatch(),
        TokenF1(),
    ]

    evaluator = Evaluator(metrics)
    metrics.append(RougeL())

    assert len(evaluator.metrics) == 2


@pytest.mark.parametrize(
    "metrics",
    [
        "not-a-sequence",
        b"not-a-sequence",
        123,
        None,
    ],
)
def test_constructor_rejects_invalid_metric_container(metrics):
    with pytest.raises(TypeError):
        Evaluator(metrics)  # type: ignore[arg-type]


def test_constructor_rejects_empty_metrics():
    with pytest.raises(ValueError):
        Evaluator([])


def test_constructor_rejects_non_metric_items():
    with pytest.raises(TypeError):
        Evaluator(
            [
                ExactMatch(),
                object(),  # type: ignore[list-item]
            ]
        )


def test_constructor_rejects_duplicate_metric_names():
    with pytest.raises(ValueError):
        Evaluator(
            [
                ConstantMetric("duplicate", 0.0),
                ConstantMetric("duplicate", 1.0),
            ]
        )


def test_evaluate_returns_scores_by_metric_name():
    evaluator = build_default_evaluator()

    scores = evaluator.evaluate(
        "retrieval augmented generation",
        "retrieval augmented generation",
    )

    assert scores == {
        "exact_match": 1.0,
        "token_f1": 1.0,
        "rouge_l": 1.0,
    }


def test_evaluate_preserves_metric_order():
    evaluator = Evaluator(
        [
            ConstantMetric("first", 0.1),
            ConstantMetric("second", 0.2),
            ConstantMetric("third", 0.3),
        ]
    )

    scores = evaluator.evaluate("prediction", "reference")

    assert list(scores) == [
        "first",
        "second",
        "third",
    ]


def test_evaluate_returns_partial_scores():
    evaluator = build_default_evaluator()

    scores = evaluator.evaluate(
        "alpha beta gamma",
        "alpha gamma delta",
    )

    assert scores["exact_match"] == 0.0
    assert scores["token_f1"] == pytest.approx(2.0 / 3.0)
    assert scores["rouge_l"] == pytest.approx(2.0 / 3.0)


def test_evaluate_supports_unicode():
    evaluator = build_default_evaluator()

    scores = evaluator.evaluate(
        "مدل بازیابی خوب است",
        "مدل بازیابی خوب است",
    )

    assert scores == {
        "exact_match": 1.0,
        "token_f1": 1.0,
        "rouge_l": 1.0,
    }


@pytest.mark.parametrize(
    ("prediction", "reference"),
    [
        (1, "reference"),
        (None, "reference"),
    ],
)
def test_evaluate_rejects_invalid_prediction(
    prediction,
    reference,
):
    with pytest.raises(TypeError):
        build_default_evaluator().evaluate(
            prediction,  # type: ignore[arg-type]
            reference,
        )


@pytest.mark.parametrize(
    ("prediction", "reference"),
    [
        ("prediction", 1),
        ("prediction", None),
    ],
)
def test_evaluate_rejects_invalid_reference(
    prediction,
    reference,
):
    with pytest.raises(TypeError):
        build_default_evaluator().evaluate(
            prediction,
            reference,  # type: ignore[arg-type]
        )


def test_evaluate_dataset_returns_metric_averages():
    evaluator = Evaluator(
        [
            ConstantMetric("low", 0.25),
            ConstantMetric("high", 0.75),
        ]
    )

    scores = evaluator.evaluate_dataset(
        ["one", "two", "three"],
        ["one", "different", "three"],
    )

    assert scores == {
        "low": 0.25,
        "high": 0.75,
    }


def test_evaluate_dataset_computes_real_metric_averages():
    evaluator = build_default_evaluator()

    scores = evaluator.evaluate_dataset(
        [
            "alpha beta",
            "gamma",
        ],
        [
            "alpha beta",
            "delta",
        ],
    )

    assert scores["exact_match"] == pytest.approx(0.5)
    assert scores["token_f1"] == pytest.approx(0.5)
    assert scores["rouge_l"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "predictions",
    [
        "not-a-sequence",
        b"not-a-sequence",
        123,
        None,
    ],
)
def test_evaluate_dataset_rejects_invalid_predictions(
    predictions,
):
    with pytest.raises(TypeError):
        build_default_evaluator().evaluate_dataset(
            predictions,  # type: ignore[arg-type]
            ["reference"],
        )


@pytest.mark.parametrize(
    "references",
    [
        "not-a-sequence",
        b"not-a-sequence",
        123,
        None,
    ],
)
def test_evaluate_dataset_rejects_invalid_references(
    references,
):
    with pytest.raises(TypeError):
        build_default_evaluator().evaluate_dataset(
            ["prediction"],
            references,  # type: ignore[arg-type]
        )


def test_evaluate_dataset_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        build_default_evaluator().evaluate_dataset(
            ["one", "two"],
            ["one"],
        )


def test_evaluate_dataset_rejects_empty_dataset():
    with pytest.raises(ValueError):
        build_default_evaluator().evaluate_dataset([], [])


def test_evaluate_dataset_rejects_invalid_prediction_item():
    with pytest.raises(TypeError):
        build_default_evaluator().evaluate_dataset(
            [
                "valid",
                1,  # type: ignore[list-item]
            ],
            [
                "valid",
                "reference",
            ],
        )


def test_evaluate_dataset_rejects_invalid_reference_item():
    with pytest.raises(TypeError):
        build_default_evaluator().evaluate_dataset(
            [
                "valid",
                "prediction",
            ],
            [
                "valid",
                None,  # type: ignore[list-item]
            ],
        )


def test_evaluate_is_deterministic():
    evaluator = build_default_evaluator()

    first = evaluator.evaluate(
        "alpha beta gamma",
        "alpha gamma",
    )
    second = evaluator.evaluate(
        "alpha beta gamma",
        "alpha gamma",
    )

    assert first == second


def test_evaluate_dataset_is_deterministic():
    evaluator = build_default_evaluator()

    first = evaluator.evaluate_dataset(
        ["alpha beta", "gamma delta"],
        ["alpha", "gamma"],
    )
    second = evaluator.evaluate_dataset(
        ["alpha beta", "gamma delta"],
        ["alpha", "gamma"],
    )

    assert first == second
