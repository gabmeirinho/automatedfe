import pytest

from geneticengine.evaluation.budget import EvaluationBudget

from automatedfe.features import (
    Aggregation,
    Feature,
    TransactionCount,
    WINDOW_CATALOG,
    build_grammar,
    build_search_algorithm,
)


def test_grammar_contains_one_feature_with_a_window_parameter():
    grammar = build_grammar()

    assert grammar.starting_symbol is Feature
    assert grammar.alternatives == {Feature: [TransactionCount]}


@pytest.mark.parametrize("window_i", range(len(WINDOW_CATALOG)))
def test_transaction_count_uses_the_selected_catalog_window(window_i):
    expression = TransactionCount(window_i)
    spec = expression.to_feature_spec()

    assert expression.window == WINDOW_CATALOG[window_i]
    assert spec.aggregation is Aggregation.COUNT
    assert spec.input_column is None
    assert spec.window == WINDOW_CATALOG[window_i]
    assert str(expression) == spec.name


def test_search_uses_the_given_budget_and_max_depth_one():
    grammar = build_grammar()
    budget = EvaluationBudget(10)
    algorithm = build_search_algorithm(
        grammar,
        lambda expression: float(expression.window_i),
        budget,
        population_size=10,
        seed=123,
    )

    assert algorithm.budget is budget
    assert algorithm.representation.decider.max_depth == 1

    algorithm.search()

    assert algorithm.tracker.get_number_evaluations() == 10


def test_search_requires_a_positive_population_size():
    with pytest.raises(ValueError, match="population_size must be positive"):
        build_search_algorithm(
            build_grammar(),
            lambda _expression: 0.0,
            EvaluationBudget(1),
            population_size=0,
        )
