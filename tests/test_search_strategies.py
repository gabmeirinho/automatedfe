from itertools import islice

import pytest
from geneticengine.evaluation.budget import EvaluationBudget

import automatedfe.features.search.enumerative_search as enumerative_module
import automatedfe.features.search.random_search as random_module
import automatedfe.features.search.search as shared_module
import automatedfe.features.search.unbound_enumerative_search as unbound_module
from automatedfe.features import (
    Add,
    CountCategory,
    MaterializingArchiveSearch,
    MeanAmount,
    build_enumerative_search,
    build_unbound_enumerative_search,
    build_grammar,
    build_random_search,
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

    monkeypatch.setattr(enumerative_module, "iterate_grammar", fake_iterate_grammar)
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

    monkeypatch.setattr(enumerative_module, "iterate_grammar", fake_iterate_grammar)
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


@pytest.fixture
def evaluated_strategy_dependencies(monkeypatch):
    prepared = []

    class StubMaterializer:
        def __init__(self, *_args, **_kwargs):
            pass

    class StubFitness:
        def __init__(self, *_args, **_kwargs):
            pass

        def prepare_population(self, expressions):
            prepared.extend(expressions)

        def __call__(self, _expression):
            return 0.2

        def objective_vector(self, _expression):
            return [0.1, 0.2, 0.3, 0.01]

    monkeypatch.setattr(shared_module, "FeatureMaterializer", StubMaterializer)
    monkeypatch.setattr(shared_module, "RandomForestFitness", StubFitness)
    return prepared


def test_evaluated_strategies_share_one_lifecycle(
    tmp_path,
    evaluated_strategy_dependencies,
):
    common = {
        "mapping": LABEL_MAPPING,
        "mmap_dir": tmp_path / "mmap",
        "dataset_path": tmp_path / "dataset.parquet",
    }

    enumerative = build_enumerative_search(EvaluationBudget(2), **common)
    random = build_random_search(EvaluationBudget(2), seed=7, **common)

    assert type(enumerative) is MaterializingArchiveSearch
    assert type(random) is MaterializingArchiveSearch
    assert isinstance(
        enumerative.candidate_generator,
        enumerative_module._EnumerativeCandidateGenerator,
    )
    assert isinstance(random.candidate_generator, random_module._RandomCandidateGenerator)

    enumerative.search()
    random.search()

    assert enumerative.tracker.get_number_evaluations() == 2
    assert random.tracker.get_number_evaluations() == 2
    assert enumerative.accepted_count == random.accepted_count == 2
    assert len(evaluated_strategy_dependencies) == 4


def test_unbound_search_generates_one_batch_without_evaluation(monkeypatch):
    def fake_iterate_grammar(grammar, starting_symbol):
        yield from (MeanAmount(index) for index in range(5))

    monkeypatch.setattr(enumerative_module, "iterate_grammar", fake_iterate_grammar)
    unbound = build_unbound_enumerative_search(
        5,
        mapping=LABEL_MAPPING,
    )

    assert type(unbound) is unbound_module.UnboundEnumerativeSearch
    assert not hasattr(unbound, "archive")
    assert not hasattr(unbound, "fitness_evaluator")

    results = unbound.search()

    assert unbound.accepted_count == 5
    assert len(results) == 5
    assert results == [MeanAmount(index) for index in range(5)]


def test_unbound_search_exhausts_the_grammar_stream_before_the_batch_limit(
    monkeypatch,
):
    yielded = [MeanAmount(0), MeanAmount(1)]

    def fake_iterate_grammar(grammar, starting_symbol):
        yield from yielded

    monkeypatch.setattr(enumerative_module, "iterate_grammar", fake_iterate_grammar)
    unbound = build_unbound_enumerative_search(
        10,
        mapping=LABEL_MAPPING,
    )

    results = unbound.search()

    assert unbound.grammar_exhausted
    assert unbound.accepted_count == 2
    assert unbound.generated_count == 2
    assert unbound.duplicate_count == 0
    assert len(results) == 2


def test_unbound_search_generates_exactly_the_requested_batch_size(monkeypatch):
    def fake_iterate_grammar(grammar, starting_symbol):
        yield from (MeanAmount(index) for index in range(7))

    monkeypatch.setattr(enumerative_module, "iterate_grammar", fake_iterate_grammar)
    unbound = build_unbound_enumerative_search(
        3,
        mapping=LABEL_MAPPING,
    )
    results = unbound.search()

    assert unbound.accepted_count == 3
    assert len(results) == 3


def test_unbound_search_preserves_enumeration_order(monkeypatch):
    def fake_iterate_grammar(grammar, starting_symbol):
        yield MeanAmount(0)
        yield MeanAmount(1)

    monkeypatch.setattr(enumerative_module, "iterate_grammar", fake_iterate_grammar)
    unbound = build_unbound_enumerative_search(
        10,
        mapping=LABEL_MAPPING,
    )
    results = unbound.search()

    assert results == [MeanAmount(0), MeanAmount(1)]
