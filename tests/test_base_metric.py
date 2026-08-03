"""Tests for BaseMetric."""

from __future__ import annotations

import pytest

from src.evaluation.base_metric import BaseMetric


class DummyMetric(BaseMetric):
    """Minimal concrete metric used by unit tests."""

    @property
    def name(self) -> str:
        return "dummy"

    def compute(
        self,
        prediction: str,
        reference: str,
    ) -> float:
        return 1.0


def test_base_metric_is_abstract():
    with pytest.raises(TypeError):
        BaseMetric()


def test_dummy_metric():
    metric = DummyMetric()

    assert metric.name == "dummy"
    assert metric.compute(
        "prediction",
        "reference",
    ) == 1.0
