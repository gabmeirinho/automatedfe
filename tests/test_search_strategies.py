from itertools import islice

import pytest

import automatedfe.features.search_strategies as strategies
from automatedfe.features import (
    Add,
    CountCategory,
    MeanAmount,
    build_grammar,
    canonical_expression_key,
    collect_unique_expressions,
    iter_bounded_expressions,
    tree_depth,
)


LABEL_MAPPING = {
    "status": {"approved": 0, "complete": 1},
    "capture_method": {"contactless": 0},
    "payment_method": {"credit": 0},
    "card_brand": {"visa": 0},
    "document_type": {"cpf": 0},
}


def test_bounded_stream_delegates_and_does_not_stop_at_first_over_depth(
    monkeypatch,
):
    shallow_one = MeanAmount(0)
    too_deep = Add(shallow_one, MeanAmount(1))
    shallow_two = MeanAmount(1)
    calls = []

    def fake_iterate_grammar(grammar, starting_symbol):
        calls.append((grammar, starting_symbol))
        yield from (shallow_one, too_deep, shallow_two)

    monkeypatch.setattr(strategies, "iterate_grammar", fake_iterate_grammar)
    grammar = build_grammar(LABEL_MAPPING)

    stream = iter_bounded_expressions(grammar, max_depth=1)
    assert list(stream) == [shallow_one, shallow_two]
    assert stream.exhausted
    assert calls == [(grammar, grammar.starting_symbol)]


def test_complete_grammar_order_is_stable_and_dependent_values_are_ascending():
    grammar = build_grammar(LABEL_MAPPING)
    first = list(islice(iter_bounded_expressions(grammar, max_depth=1), 10))

    assert all(tree_depth(expression) == 1 for expression in first)
    assert all(isinstance(expression, CountCategory) for expression in first)
    assert [(e.category_family_i, e.category_code_i, e.window_i) for e in first] == [
        (0, 0, index) for index in range(10)
    ]

    repeated = list(islice(iter_bounded_expressions(grammar, max_depth=1), 10))
    assert [canonical_expression_key(e) for e in first] == [
        canonical_expression_key(e) for e in repeated
    ]


def test_evaluation_free_collector_reports_exhaustion_and_unique_count(
    monkeypatch,
):
    grammar = build_grammar(LABEL_MAPPING)
    yielded = [MeanAmount(0), MeanAmount(0), MeanAmount(1)]
    calls = {"count": 0}

    def fake_iterate_grammar(grammar, starting_symbol):
        calls["count"] += 1
        yield from yielded

    monkeypatch.setattr(strategies, "iterate_grammar", fake_iterate_grammar)
    result = collect_unique_expressions(grammar, 10, max_depth=1)

    assert result.expressions == (MeanAmount(0), MeanAmount(1))
    assert result.exhausted
    assert calls["count"] == 1


def test_invalid_bounds_are_rejected():
    grammar = build_grammar(LABEL_MAPPING)

    with pytest.raises(ValueError, match="max_depth must be positive"):
        iter_bounded_expressions(grammar, max_depth=0)
    with pytest.raises(ValueError, match="candidate_count must be positive"):
        collect_unique_expressions(grammar, 0)
