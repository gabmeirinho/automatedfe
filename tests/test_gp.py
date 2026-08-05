import pytest
import csv
from typing import get_type_hints

from geneticengine.evaluation.budget import EvaluationBudget

from automatedfe.features import (
    AMOUNT_COLUMN,
    Aggregation,
    Feature,
    Mean,
    WINDOW_CATALOG,
    WindowIndex,
    build_grammar,
    build_search_algorithm,
)


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
