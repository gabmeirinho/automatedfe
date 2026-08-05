import csv
from typing import get_type_hints

import numpy as np
import pytest
from geneticengine.evaluation.budget import EvaluationBudget

from automatedfe.features import (
    AMOUNT_COLUMN,
    WINDOW_CATALOG,
    Aggregation,
    Feature,
    MaterializingGeneticProgramming,
    Mean,
    WindowIndex,
    build_grammar,
    build_search_algorithm,
)
from automatedfe.features.feature_materialization import FeatureMaterializer


def test_grammar_is_a_depth_one_mean_with_feature_and_window_parameters():
    grammar = build_grammar()

    assert grammar.starting_symbol is Mean
    assert grammar.get_min_tree_depth() == 1
    assert list(Mean.__annotations__) == ["feature", "window"]
    annotations = get_type_hints(Mean, include_extras=True)
    assert annotations["feature"] == Feature
    assert annotations["window"] == WindowIndex


@pytest.mark.parametrize("window_i", range(len(WINDOW_CATALOG)))
def test_mean_maps_its_feature_and_window_to_a_feature_spec(window_i):
    expression = Mean(AMOUNT_COLUMN, window_i)
    spec = expression.to_feature_spec()

    assert expression.feature == AMOUNT_COLUMN
    assert expression.selected_window == WINDOW_CATALOG[window_i]
    assert spec.aggregation is Aggregation.MEAN
    assert spec.input_column == AMOUNT_COLUMN
    assert spec.window == WINDOW_CATALOG[window_i]
    assert str(expression) == spec.name


def test_search_uses_the_given_budget_and_max_depth_one():
    grammar = build_grammar()
    budget = EvaluationBudget(10)
    algorithm = build_search_algorithm(
        grammar,
        lambda expression: float(expression.window),
        budget,
        population_size=10,
        seed=123,
    )

    assert algorithm.budget is budget
    assert algorithm.representation.decider.max_depth == 1

    algorithm.search()

    assert algorithm.tracker.get_number_evaluations() == 10


def test_tracker_records_every_evaluation_to_csv(tmp_path):
    csv_path = tmp_path / "gp_search.csv"
    algorithm = build_search_algorithm(
        build_grammar(),
        lambda expression: float(expression.window),
        EvaluationBudget(10),
        population_size=10,
        seed=123,
        csv_path=csv_path,
    )

    algorithm.search()

    with csv_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == algorithm.tracker.get_number_evaluations()
    assert list(rows[0]) == ["Generation", "Expression", "Feature", "Window", "Fitness"]
    assert {row["Feature"] for row in rows} == {AMOUNT_COLUMN}
    assert all(row["Expression"].startswith("mean_amount_") for row in rows)


def test_search_requires_a_positive_population_size():
    with pytest.raises(ValueError, match="population_size must be positive"):
        build_search_algorithm(
            build_grammar(),
            lambda _expression: 0.0,
            EvaluationBudget(1),
            population_size=0,
        )


def test_search_materializes_complete_population_before_fitness(tmp_path):
    columns = {
        "merchant_id": np.array([1, 1, 1, 2]),
        "amount": np.array([10.0, 20.0, 30.0, 100.0]),
        "created_at": np.array([1, 2, 3, 1], dtype=np.int64),
    }
    materializer = FeatureMaterializer(columns, output_dir=tmp_path)
    events = []

    original_materialize = materializer.materialize

    def record_materialization(individual):
        events.append(("materialize", str(individual)))
        return original_materialize(individual)

    materializer.materialize = record_materialization

    algorithm = build_search_algorithm(
        build_grammar(),
        lambda individual: events.append(("fitness", str(individual))) or 0.0,
        EvaluationBudget(10),
        population_size=10,
        seed=123,
        materializer=materializer,
    )

    algorithm.search()

    assert isinstance(algorithm, MaterializingGeneticProgramming)
    assert [event for event, _ in events] == ["materialize"] * 10 + ["fitness"] * 10
    assert list(tmp_path.glob("mean_amount_*.mmap"))


def test_materialization_only_search_defaults_every_fitness_to_zero():
    seen = []

    algorithm = build_search_algorithm(
        build_grammar(),
        EvaluationBudget(4),
        population_size=4,
        materializer=lambda individual: seen.append(individual),
    )
    best = algorithm.search()

    assert len(seen) == 4
    assert all(isinstance(individual, Mean) for individual in seen)
    assert best[0].get_fitness(algorithm.problem).fitness_components == [0.0]
