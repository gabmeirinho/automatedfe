from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from geneticengine.evaluation.sequential import SequentialEvaluator
from geneticengine.problems import MultiObjectiveProblem
from geneticengine.solutions.individual import ConcreteIndividual

from automatedfe.features.archive import (
    ActiveSetManager,
    absolute_pearson_correlation,
    correlation_rejection,
)


@dataclass(frozen=True)
class Expression:
    name: str


def make_problem(scores):
    return MultiObjectiveProblem(
        fitness_function=lambda expression: scores[expression.name],
        minimize=[False, False, False, True],
    )


def run(step, problem, expressions, *, generation):
    individuals = []
    for expression in expressions:
        individual = ConcreteIndividual(expression)
        individual.set_fitness(problem, problem.evaluate(expression))
        individuals.append(individual)
    return list(
        step.apply(
            problem,
            SequentialEvaluator(),
            representation=None,
            random=None,
            population=iter(individuals),
            target_size=len(individuals),
            generation=generation,
        )
    )


def test_correlation_helpers_are_absolute_and_thresholds_are_inclusive():
    signal = np.array([0.0, 1.0, 2.0, 3.0])
    assert absolute_pearson_correlation(signal, -signal) == pytest.approx(1.0)
    assert correlation_rejection(signal, [signal], 1.0)["reason"] == "pairwise_threshold"
    assert correlation_rejection(signal, [np.array([0.0, 1.0, 0.0, 1.0])], 0.99)[
        "rejected"
    ] is False


def test_active_set_manager_applies_quality_history_and_peer_filters_in_order():
    scores = {
        "a": (0.8, 0.8, 0.8, 1.0),
        "b": (0.8, 0.9, 0.7, 2.0),
        "c": (0.9, 0.9, 0.9, 3.0),
        "dominated": (0.7, 0.7, 0.7, 4.0),
        "low": (0.0005, 0.8, 0.8, 1.0),
    }
    signals = {
        "a": np.array([0.0, 1.0, 2.0, 3.0]),
        "b": np.array([0.0, 2.0, 4.0, 6.0]),
        "c": np.array([3.0, 0.0, 2.0, 1.0]),
        "dominated": np.array([1.0, 3.0, 0.0, 2.0]),
        "low": np.array([3.0, 1.0, 2.0, 0.0]),
    }
    problem = make_problem(scores)
    step = ActiveSetManager(signal_provider=lambda expression: signals[expression.name])

    run(
        step,
        problem,
        [Expression("a"), Expression("b"), Expression("c"), Expression("dominated")],
        generation=0,
    )

    assert [item.get_phenotype().name for item in step.history] == ["a", "c"]
    assert step.history_objectives == (
        scores["a"],
        scores["c"],
    )
    assert any(
        item["outcome"] == "rejected" and item["reason"] == "same_generation_peer_cluster"
        for item in step.filter_diagnostics
    )

    run(step, problem, [Expression("low")], generation=1)
    assert [item.get_phenotype().name for item in step.history] == ["a", "c"]
    assert any(item["reason"] == "quality_threshold" for item in step.filter_diagnostics)


def test_active_set_manager_rejects_invalid_signals_and_preserves_admission_objectives():
    scores = {
        "a": (0.8, 0.8, 0.8, 1.0),
        "duplicate": (0.8, 0.8, 0.8, 1.0),
        "constant": (0.9, 0.7, 0.7, 2.0),
        "nan": (0.7, 0.9, 0.7, 3.0),
        "empty": (0.7, 0.7, 0.9, 4.0),
    }
    signals = {
        "a": np.array([0.0, 1.0, 2.0, 3.0]),
        "duplicate": np.array([3.0, 2.0, 1.0, 0.0]),
        "constant": np.ones(4),
        "nan": np.array([0.0, np.nan, 1.0, 2.0]),
        "empty": np.array([]),
    }
    problem = make_problem(scores)
    step = ActiveSetManager(signal_provider=lambda expression: signals[expression.name])
    run(step, problem, [Expression("a")], generation=0)
    original = step.history_objectives

    run(
        step,
        problem,
        [
            Expression("a"),
            Expression("constant"),
            Expression("nan"),
            Expression("empty"),
        ],
        generation=1,
    )
    assert [item.get_phenotype().name for item in step.history] == ["a"]
    assert step.history_objectives == original
    reasons = {item["reason"] for item in step.filter_diagnostics}
    assert "duplicate_history_expression" in reasons
    assert "signal_constant" in reasons
    assert "signal_nonfinite" in reasons
    assert "signal_empty" in reasons


@pytest.mark.parametrize("kwargs", [{"archive_correlation_threshold": -0.1}, {"archive_correlation_threshold": 1.1}, {"archive_quality_threshold": -1.0}])
def test_active_set_manager_validates_thresholds(kwargs):
    with pytest.raises(ValueError):
        ActiveSetManager(**kwargs)
