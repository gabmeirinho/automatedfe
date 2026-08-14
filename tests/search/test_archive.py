import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from geneticengine.evaluation.sequential import SequentialEvaluator
from geneticengine.problems import Fitness, MultiObjectiveProblem
from geneticengine.solutions.individual import ConcreteIndividual

import automatedfe.search.archive as archive_module
from automatedfe.features.grammar import (
    Add,
    AvgDailyAmount,
    AvgDailyAmountCategory,
    AvgDailyCount,
    AvgDailyCountCategory,
    AvgDailyTotalAmount,
    CategoryRate,
    CountCategory,
    CountTotal,
    Log,
    MaxAmount,
    MeanAmount,
    Mul,
    SafeDiv,
    StdAmount,
    Sub,
    TotalAmount,
    build_grammar,
)
from automatedfe.search.archive import (
    ActiveSetManager,
    ArchiveStep,
    SNAPSHOT_MAPPING_REFERENCE,
    build_snapshot_document,
    decode_expression,
    encode_expression,
    load_active_set_snapshot,
    load_archive,
    load_snapshot,
    write_snapshot,
)


@dataclass(frozen=True)
class Expression:
    name: str

    def __str__(self) -> str:
        return self.name


LABEL_MAPPING = {
    "status": {"approved": 0, "complete": 1, "denied": 2, "others": 3},
    "capture_method": {"contactless": 0, "emv": 1, "pix": 2},
    "payment_method": {"debit": 0, "credit": 1, "null": -1},
    "card_brand": {"mastercard": 0, "visa": 1, "null": -1},
    "document_type": {"cnpj": 0, "cpf": 1, "null": -1},
}


@pytest.fixture(autouse=True)
def configure_label_mapping():
    build_grammar(LABEL_MAPPING)


def individual(name: str) -> ConcreteIndividual[Expression]:
    return ConcreteIndividual(Expression(name))


def make_problem(scores):
    return MultiObjectiveProblem(
        fitness_function=lambda expression: scores[expression.name],
        minimize=[False, False, False, True],
    )


def evaluated_individuals(problem, scores):
    individuals = []
    for name, objective_values in scores.items():
        candidate = individual(name)
        candidate.set_fitness(problem, Fitness(objective_values))
        individuals.append(candidate)
    return individuals


def run_archive_step(step, problem, individuals, *, target_size=None):
    if target_size is None:
        target_size = len(individuals)
    return list(
        step.apply(
            problem,
            SequentialEvaluator(),
            representation=None,
            random=None,
            population=iter(individuals),
            target_size=target_size,
            generation=0,
        )
    )


def archived_names(step):
    return [str(candidate.get_phenotype()) for candidate in step.archive]


def test_archive_step_requires_four_objectives():
    problem = MultiObjectiveProblem(
        fitness_function=lambda _expression: [0.8, 0.8, 0.8],
        minimize=[False, False, False],
    )

    with pytest.raises(ValueError, match="four objectives"):
        run_archive_step(ArchiveStep(), problem, [])


def test_archive_step_uses_the_complete_population_and_passes_it_through():
    scores = {
        "front": (0.8, 0.8, 0.8, 1.0),
        "tradeoff": (0.9, 0.7, 0.9, 1.5),
        "dominated": (0.7, 0.7, 0.7, 2.0),
    }
    problem = make_problem(scores)
    population = evaluated_individuals(problem, scores)
    step = ArchiveStep()

    output = run_archive_step(step, problem, population, target_size=1)

    assert output == population
    assert archived_names(step) == ["front", "tradeoff"]


def test_archive_step_delegates_front_calculation_to_genetic_engine(monkeypatch):
    scores = {
        "first": (0.8, 0.8, 0.8, 1.0),
        "second": (0.9, 0.8, 0.8, 1.0),
    }
    problem = make_problem(scores)
    population = evaluated_individuals(problem, scores)
    step = ArchiveStep()
    calls = []
    original = archive_module.non_dominated

    def recording_non_dominated(candidates, received_problem):
        candidates = list(candidates)
        calls.append((candidates, received_problem.minimize))
        return original(iter(candidates), received_problem)

    monkeypatch.setattr(archive_module, "non_dominated", recording_non_dominated)
    run_archive_step(step, problem, population)

    assert len(calls) == 1
    assert len(calls[0][0]) == len(population)
    assert calls[0][1] == [False, False, False, True]


def test_archive_step_rejects_dominated_incoming_candidates():
    scores = {
        "old": (0.8, 0.8, 0.8, 1.0),
        "dominated": (0.7, 0.7, 0.7, 2.0),
    }
    problem = make_problem(scores)
    step = ArchiveStep()

    run_archive_step(
        step,
        problem,
        evaluated_individuals(problem, {"old": scores["old"]}),
    )
    run_archive_step(
        step,
        problem,
        evaluated_individuals(problem, {"dominated": scores["dominated"]}),
    )

    assert archived_names(step) == ["old"]


def test_archive_step_admits_a_non_dominated_tradeoff():
    scores = {
        "old": (0.8, 0.8, 0.8, 1.0),
        "tradeoff": (0.9, 0.7, 0.9, 1.5),
    }
    problem = make_problem(scores)
    step = ArchiveStep()

    run_archive_step(
        step,
        problem,
        evaluated_individuals(problem, {"old": scores["old"]}),
    )
    run_archive_step(
        step,
        problem,
        evaluated_individuals(problem, {"tradeoff": scores["tradeoff"]}),
    )

    assert archived_names(step) == ["old", "tradeoff"]


def test_archive_step_admits_later_winner_without_evicting_older_members():
    scores = {
        "old": (0.8, 0.8, 0.8, 1.0),
        "tradeoff": (0.9, 0.7, 0.9, 1.5),
        "winner": (0.9, 0.9, 0.9, 0.5),
    }
    problem = make_problem(scores)
    step = ArchiveStep()

    run_archive_step(step, problem, evaluated_individuals(problem, {"old": scores["old"]}))
    run_archive_step(
        step,
        problem,
        evaluated_individuals(problem, {"tradeoff": scores["tradeoff"]}),
    )
    assert archived_names(step) == ["old", "tradeoff"]

    run_archive_step(
        step,
        problem,
        evaluated_individuals(problem, {"winner": scores["winner"]}),
    )

    assert archived_names(step) == ["old", "tradeoff", "winner"]


def test_archive_step_excludes_invalid_candidates_but_yields_them():
    scores = {
        "valid": (0.8, 0.8, 0.8, 1.0),
        "invalid": (0.9, 0.9, 0.9, 0.1),
    }
    problem = make_problem(scores)
    population = evaluated_individuals(problem, scores)
    population[1].set_fitness(problem, Fitness(list(scores["invalid"]), valid=False))
    step = ArchiveStep()

    output = run_archive_step(step, problem, population)

    assert output == population
    assert archived_names(step) == ["valid"]


def test_archive_step_deduplicates_expressions_and_keeps_first_live_individual():
    scores = {
        "same": (0.8, 0.8, 0.8, 1.0),
    }
    problem = make_problem(scores)
    first = evaluated_individuals(problem, scores)[0]
    duplicate = individual("same")
    duplicate.set_fitness(problem, Fitness((0.9, 0.9, 0.9, 0.5)))
    step = ArchiveStep()

    run_archive_step(step, problem, [first, duplicate])

    assert step.archive == [first]


def test_archive_step_uses_directions_from_the_problem():
    scores = {
        "lower": (0.1, 0.1, 0.1, 0.1),
        "higher": (0.9, 0.9, 0.9, 0.9),
    }
    problem = MultiObjectiveProblem(
        fitness_function=lambda expression: scores[expression.name],
        minimize=[True, True, True, True],
    )
    population = evaluated_individuals(problem, scores)

    step = ArchiveStep()
    run_archive_step(step, problem, population)
    assert archived_names(step) == ["lower"]


def test_archive_step_rejects_non_finite_objectives():
    scores = {
        "nan": (0.8, float("nan"), 0.8, 1.0),
        "infinite": (0.8, 0.8, 0.8, float("inf")),
    }
    problem = make_problem(scores)
    step = ArchiveStep()

    run_archive_step(step, problem, evaluated_individuals(problem, scores))

    assert step.archive == []


def make_grammar_problem(scores):
    return MultiObjectiveProblem(
        fitness_function=lambda expression: scores[str(expression)],
        minimize=[False, False, False, True],
    )


def grammar_evaluated_individuals(problem, expressions):
    individuals = []
    for expression in expressions:
        candidate = ConcreteIndividual(expression)
        candidate.set_fitness(problem, problem.evaluate(expression))
        individuals.append(candidate)
    return individuals


def test_expression_round_trip_covers_every_grammar_node_type():
    terminals = [
        MeanAmount(0),
        MaxAmount(3),
        TotalAmount(8),
        StdAmount(1),
        CountTotal(5),
        CountCategory(0, 1, 2),
        AvgDailyCount(1),
        AvgDailyCountCategory(1, 0, 2),
        AvgDailyAmount(0),
        AvgDailyAmountCategory(2, 1, 1),
        AvgDailyTotalAmount(2),
        CategoryRate(0, 0, 3),
    ]
    composites = [
        Add(terminals[0], terminals[4]),
        Sub(terminals[0], terminals[4]),
        Mul(terminals[0], terminals[4]),
        SafeDiv(terminals[0], terminals[4]),
        Log(terminals[0]),
        Add(Log(Mul(terminals[0], terminals[4])), Sub(terminals[1], terminals[5])),
    ]

    for expression in [*terminals, *composites]:
        decoded = decode_expression(encode_expression(expression))
        assert decoded == expression
        assert str(decoded) == str(expression)


def test_categorical_expressions_reconstruct_to_the_same_feature_names():
    expressions = [
        CountCategory(0, 2, 0),
        AvgDailyCountCategory(1, 0, 2),
        AvgDailyAmountCategory(3, 1, 0),
        CategoryRate(0, 1, 5),
    ]

    for expression in expressions:
        decoded = decode_expression(encode_expression(expression))
        assert decoded.to_feature_spec() == expression.to_feature_spec()


def test_encode_expression_rejects_unknown_node_types():
    with pytest.raises(TypeError, match="Unsupported expression node type"):
        encode_expression(object())
    with pytest.raises(TypeError, match="Unsupported expression node type"):
        encode_expression(MeanAmount)


def test_decode_expression_rejects_malformed_structures():
    with pytest.raises(TypeError, match="JSON object"):
        decode_expression("not an object")
    with pytest.raises(ValueError, match="Unknown expression node type"):
        decode_expression({"type": "MysteryNode", "fields": {}})
    with pytest.raises(ValueError, match="expects fields"):
        decode_expression({"type": "Add", "fields": {"left": 1}})
    with pytest.raises(ValueError, match="Cannot reconstruct"):
        decode_expression({"type": "MeanAmount", "fields": {"window_i": "1"}})


def test_archive_save_and_load_round_trip_grammar_expressions(tmp_path):
    scores = {
        str(Add(MeanAmount(0), CountTotal(0))): (0.8, 0.8, 0.8, 1.0),
        str(Mul(MaxAmount(1), CountTotal(1))): (0.9, 0.7, 0.9, 1.5),
    }
    expressions = [Add(MeanAmount(0), CountTotal(0)), Mul(MaxAmount(1), CountTotal(1))]
    problem = make_grammar_problem(scores)
    step = ArchiveStep(mapping=LABEL_MAPPING)
    run_archive_step(step, problem, grammar_evaluated_individuals(problem, expressions))
    archive_path = tmp_path / "archive.json"
    step.save(archive_path)

    snapshot = load_archive(archive_path)
    assert snapshot.version == archive_module.FORMAT_VERSION
    assert snapshot.minimize == (False, False, False, True)
    assert snapshot.mapping == LABEL_MAPPING
    assert [str(expression) for expression in snapshot.expressions] == [
        str(expression) for expression in expressions
    ]
    assert snapshot.objectives == (
        (0.8, 0.8, 0.8, 1.0),
        (0.9, 0.7, 0.9, 1.5),
    )


def test_archive_snapshot_contains_only_the_current_front(tmp_path):
    front = Add(MeanAmount(0), CountTotal(0))
    tradeoff = Mul(MaxAmount(1), CountTotal(1))
    dominated = Sub(StdAmount(0), AvgDailyCount(0))
    scores = {
        str(front): (0.8, 0.8, 0.8, 1.0),
        str(tradeoff): (0.9, 0.7, 0.9, 1.5),
        str(dominated): (0.7, 0.7, 0.7, 2.0),
    }
    problem = make_grammar_problem(scores)
    step = ArchiveStep(mapping=LABEL_MAPPING)
    run_archive_step(
        step,
        problem,
        grammar_evaluated_individuals(problem, [front, tradeoff, dominated]),
    )
    archive_path = tmp_path / "archive.json"
    step.save(archive_path)

    snapshot = load_archive(archive_path)
    assert [str(expression) for expression in snapshot.expressions] == [
        str(front),
        str(tradeoff),
    ]


def test_archive_json_contains_versioned_metadata(tmp_path):
    expression = Add(MeanAmount(0), CountTotal(0))
    scores = {str(expression): (0.8, 0.8, 0.8, 1.0)}
    problem = make_grammar_problem(scores)
    step = ArchiveStep(mapping=LABEL_MAPPING)
    run_archive_step(step, problem, grammar_evaluated_individuals(problem, [expression]))
    archive_path = tmp_path / "archive.json"
    step.save(archive_path)

    data = json.loads(archive_path.read_text())
    assert data["format"] == "automatedfe-archive"
    assert data["version"] == archive_module.FORMAT_VERSION
    assert data["problem"] == {
        "number_of_objectives": 4,
        "minimize": [False, False, False, True],
    }
    assert data["mapping"] == LABEL_MAPPING


def test_archive_writes_are_atomic(tmp_path, monkeypatch):
    replaced = []
    original_replace = os.replace

    def recording_replace(source, destination):
        replaced.append((Path(source).name, Path(destination).name))
        return original_replace(source, destination)

    monkeypatch.setattr(archive_module.os, "replace", recording_replace)
    expression = Add(MeanAmount(0), CountTotal(0))
    scores = {str(expression): (0.8, 0.8, 0.8, 1.0)}
    problem = make_grammar_problem(scores)
    step = ArchiveStep(mapping=LABEL_MAPPING)
    run_archive_step(step, problem, grammar_evaluated_individuals(problem, [expression]))

    archive_path = tmp_path / "archive.json"
    step.save(archive_path)

    assert len(replaced) == 1
    assert replaced[0][0].startswith(".archive.json.")
    assert replaced[0][0].endswith(".tmp")
    assert replaced[0][1] == "archive.json"
    assert list(tmp_path.glob("*.tmp")) == []
    load_archive(archive_path)


def test_archive_auto_snapshot_retains_permanent_membership(tmp_path):
    archive_path = tmp_path / "archive.json"
    step = ArchiveStep(archive_path=archive_path, mapping=LABEL_MAPPING)
    old = Add(MeanAmount(0), CountTotal(0))
    winner = Mul(MaxAmount(1), CountTotal(1))
    scores = {
        str(old): (0.8, 0.8, 0.8, 1.0),
        str(winner): (0.9, 0.9, 0.9, 0.5),
    }
    problem = make_grammar_problem(scores)

    run_archive_step(step, problem, grammar_evaluated_individuals(problem, [old]))
    assert [str(expression) for expression in load_archive(archive_path).expressions] == [
        str(old)
    ]

    run_archive_step(step, problem, grammar_evaluated_individuals(problem, [winner]))
    assert [str(expression) for expression in load_archive(archive_path).expressions] == [
        str(old),
        str(winner),
    ]


def test_invalid_archive_refresh_retains_finite_evidence_and_round_trips(tmp_path):
    expression = Add(MeanAmount(0), CountTotal(0))
    objectives = (0.8, 0.8, 0.8, 1.0)
    problem = make_grammar_problem({str(expression): objectives})
    step = ArchiveStep(mapping=LABEL_MAPPING)
    run_archive_step(
        step,
        problem,
        grammar_evaluated_individuals(problem, [expression]),
    )

    class InvalidRefreshEvaluator:
        def __init__(self):
            self.calls = 0

        def evaluate(self, received_problem, candidates):
            for candidate in candidates:
                self.calls += 1
                candidate.set_fitness(
                    received_problem,
                    Fitness([float("nan")] * 4, valid=False),
                )
                yield candidate

    evaluator = InvalidRefreshEvaluator()
    step.reevaluate_archive(problem, evaluator)

    assert evaluator.calls == 1
    assert [str(individual.get_phenotype()) for individual in step.archive] == [
        str(expression)
    ]
    assert step.archive[0].get_fitness(problem).valid is True
    assert tuple(step.archive[0].get_fitness(problem).fitness_components) == objectives

    archive_path = tmp_path / "archive.json"
    step.save(archive_path)
    snapshot = load_archive(archive_path)
    assert snapshot.expressions == (expression,)
    assert snapshot.objectives == (objectives,)


def test_save_requires_an_evaluated_population(tmp_path):
    step = ArchiveStep(mapping=LABEL_MAPPING)

    with pytest.raises(ValueError, match="has not evaluated"):
        step.save(tmp_path / "archive.json")


def test_save_requires_a_path_when_none_configured():
    step = ArchiveStep(mapping=LABEL_MAPPING)

    with pytest.raises(ValueError, match="No archive path configured"):
        step.save()


def test_load_archive_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_archive(tmp_path / "missing.json")


def test_load_archive_rejects_malformed_json(tmp_path):
    path = tmp_path / "archive.json"
    path.write_text("{not valid json")

    with pytest.raises(ValueError, match="not valid JSON"):
        load_archive(path)


def test_load_archive_rejects_unknown_format(tmp_path):
    path = tmp_path / "archive.json"
    path.write_text(json.dumps({"format": "other", "version": 1}))

    with pytest.raises(ValueError, match="Unknown archive format"):
        load_archive(path)


def test_load_archive_rejects_unsupported_version(tmp_path):
    path = tmp_path / "archive.json"
    path.write_text(json.dumps({"format": "automatedfe-archive", "version": 999}))

    with pytest.raises(ValueError, match="Unsupported archive version"):
        load_archive(path)


def test_load_archive_rejects_missing_problem_metadata(tmp_path):
    path = tmp_path / "archive.json"
    path.write_text(json.dumps({"format": "automatedfe-archive", "version": 1}))

    with pytest.raises(TypeError, match="'problem' metadata"):
        load_archive(path)


def test_load_archive_rejects_wrong_objective_count(tmp_path):
    path = tmp_path / "archive.json"
    path.write_text(
        json.dumps(
            {
                "format": "automatedfe-archive",
                "version": 1,
                "problem": {"number_of_objectives": 3, "minimize": [False, False, False]},
                "mapping": LABEL_MAPPING,
                "expressions": [],
            }
        )
    )

    with pytest.raises(ValueError, match="4 objectives"):
        load_archive(path)


def test_load_archive_rejects_unknown_node_type(tmp_path):
    path = tmp_path / "archive.json"
    path.write_text(
        json.dumps(
            {
                "format": "automatedfe-archive",
                "version": 1,
                "problem": {
                    "number_of_objectives": 4,
                    "minimize": [False, False, False, True],
                },
                "mapping": LABEL_MAPPING,
                "expressions": [
                    {
                        "expression": {"type": "MysteryNode", "fields": {}},
                        "objectives": [0.0, 0.0, 0.0, 0.0],
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="Unknown expression node type"):
        load_archive(path)


def test_load_archive_rejects_entry_without_expression(tmp_path):
    path = tmp_path / "archive.json"
    path.write_text(
        json.dumps(
            {
                "format": "automatedfe-archive",
                "version": 1,
                "problem": {
                    "number_of_objectives": 4,
                    "minimize": [False, False, False, True],
                },
                "mapping": LABEL_MAPPING,
                "expressions": [{"objectives": [0.0, 0.0, 0.0, 0.0]}],
            }
        )
    )

    with pytest.raises(ValueError, match="'expression' and 'objectives'"):
        load_archive(path)


def test_load_archive_rejects_wrong_entry_objective_count(tmp_path):
    path = tmp_path / "archive.json"
    path.write_text(
        json.dumps(
            {
                "format": "automatedfe-archive",
                "version": 1,
                "problem": {
                    "number_of_objectives": 4,
                    "minimize": [False, False, False, True],
                },
                "mapping": LABEL_MAPPING,
                "expressions": [
                    {
                        "expression": {"type": "MeanAmount", "fields": {"window_i": 0}},
                        "objectives": [0.0, 0.0],
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="exactly 4 objective values"):
        load_archive(path)


def test_load_archive_validates_mapping_match(tmp_path):
    mismatched_mapping = {
        "status": {"approved": 0, "complete": 1, "denied": 2, "others": 3, "extra": 4},
        "capture_method": {"contactless": 0, "emv": 1, "pix": 2},
        "payment_method": {"debit": 0, "credit": 1, "null": -1},
        "card_brand": {"mastercard": 0, "visa": 1, "null": -1},
        "document_type": {"cnpj": 0, "cpf": 1, "null": -1},
    }
    expression = Add(MeanAmount(0), CountTotal(0))
    scores = {str(expression): (0.8, 0.8, 0.8, 1.0)}
    problem = make_grammar_problem(scores)
    step = ArchiveStep(mapping=LABEL_MAPPING)
    run_archive_step(step, problem, grammar_evaluated_individuals(problem, [expression]))
    archive_path = tmp_path / "archive.json"
    step.save(archive_path)

    with pytest.raises(ValueError, match="does not match"):
        load_archive(archive_path, mapping=mismatched_mapping)
    snapshot = load_archive(archive_path, mapping=LABEL_MAPPING)
    assert len(snapshot) == 1


def make_active_manager(scores, signals, *, mapping=LABEL_MAPPING, **kwargs):
    defaults = {
        "promotion_refresh_top_n": 0,
        "first_promotion_top_k": 2,
        "promotion_add_k": 1,
        "promotion_min_gain": 0.0,
        "promotion_mean_gain": 0.0,
        "active_correlation_threshold": 0.99,
    }
    defaults.update(kwargs)
    manager = ActiveSetManager(
        mapping=mapping,
        signal_provider=lambda expression: signals[str(expression)],
        use_active_set=True,
        **defaults,
    )
    return manager


def test_active_manager_history_round_trips_with_immutable_admission_objectives(
    tmp_path,
):
    expressions = [Add(MeanAmount(0), CountTotal(0)), Mul(MaxAmount(1), CountTotal(1))]
    scores = {
        str(expressions[0]): (0.8, 0.8, 0.8, 1.0),
        str(expressions[1]): (0.9, 0.7, 0.9, 1.5),
    }
    signals = {
        str(expressions[0]): np.array([0.0, 1.0, 0.0, 1.0]),
        str(expressions[1]): np.array([0.0, 0.0, 1.0, 1.0]),
    }
    problem = make_grammar_problem(scores)
    manager = make_active_manager(scores, signals)
    individuals = grammar_evaluated_individuals(problem, expressions)
    manager.process_evaluated_population(problem, individuals, generation=0)
    individuals[0].set_fitness(problem, Fitness([0.1, 0.1, 0.1, 9.0], valid=True))

    history_path = tmp_path / "history.json"
    manager.save_history(history_path)

    snapshot = load_archive(history_path)
    assert [str(expression) for expression in snapshot.expressions] == [
        str(expression) for expression in expressions
    ]
    assert snapshot.objectives == (
        (0.8, 0.8, 0.8, 1.0),
        (0.9, 0.7, 0.9, 1.5),
    )
    assert snapshot.mapping == LABEL_MAPPING
    assert snapshot.minimize == (False, False, False, True)


def test_active_manager_snapshot_round_trip_records_promotion_metadata(tmp_path):
    expressions = [Add(MeanAmount(0), CountTotal(0)), Mul(MaxAmount(1), CountTotal(1))]
    scores = {
        str(expressions[0]): (0.8, 0.8, 0.8, 1.0),
        str(expressions[1]): (0.9, 0.9, 0.9, 2.0),
    }
    signals = {
        str(expressions[0]): np.array([0.0, 1.0, 0.0, 1.0]),
        str(expressions[1]): np.array([0.0, 0.0, 1.0, 1.0]),
    }
    problem = make_grammar_problem(scores)
    manager = make_active_manager(scores, signals, first_promotion_top_k=1)
    manager.process_evaluated_population(
        problem,
        grammar_evaluated_individuals(problem, expressions),
        generation=0,
    )

    class ScoringEvaluator:
        def evaluate(self, evaluation_problem, candidates):
            for candidate in candidates:
                candidate.set_fitness(
                    evaluation_problem,
                    Fitness(scores[str(candidate.get_phenotype())], valid=True),
                )
                yield candidate

    assert manager.maybe_promote(problem, 5, evaluator=ScoringEvaluator())
    promoted = manager.active_individuals[0].get_phenotype()

    active_path = tmp_path / "active.json"
    manager.save_active_snapshot(active_path)

    snapshot = load_active_set_snapshot(active_path)
    assert len(snapshot) == 1
    assert snapshot.baseline_version == 1
    assert [str(expression) for expression in snapshot.expressions] == [str(promoted)]
    assert snapshot.gains == ((0.9, 0.9, 0.9),)
    assert snapshot.promotion_generations == (5,)
    assert snapshot.promoted_baseline_versions == (1,)
    assert snapshot.promotion_events
    assert snapshot.promotion_checks
    assert snapshot.mapping == LABEL_MAPPING
    assert snapshot.minimize == (False, False, False, True)


def test_active_manager_saves_empty_snapshot_with_promotion_state(tmp_path):
    expression = Add(MeanAmount(0), CountTotal(0))
    scores = {str(expression): (0.8, 0.8, 0.8, 1.0)}
    signals = {str(expression): np.array([0.0, 1.0, 0.0, 1.0])}
    problem = make_grammar_problem(scores)
    manager = make_active_manager(scores, signals)
    manager.process_evaluated_population(
        problem,
        grammar_evaluated_individuals(problem, [expression]),
        generation=0,
    )

    active_path = tmp_path / "active.json"
    manager.save_active_snapshot(active_path)

    snapshot = load_active_set_snapshot(active_path)
    assert len(snapshot) == 0
    assert snapshot.baseline_version == 0
    assert snapshot.promotion_events == ()
    assert snapshot.mapping == LABEL_MAPPING


def test_load_active_set_snapshot_validates_mapping_match(tmp_path):
    expression = Add(MeanAmount(0), CountTotal(0))
    scores = {str(expression): (0.8, 0.8, 0.8, 1.0)}
    signals = {str(expression): np.array([0.0, 1.0, 0.0, 1.0])}
    problem = make_grammar_problem(scores)
    manager = make_active_manager(scores, signals)
    manager.process_evaluated_population(
        problem,
        grammar_evaluated_individuals(problem, [expression]),
        generation=0,
    )
    active_path = tmp_path / "active.json"
    manager.save_active_snapshot(active_path)

    mismatched_mapping = {
        "status": {"approved": 0, "complete": 1, "denied": 2, "others": 3, "extra": 4},
        "capture_method": {"contactless": 0, "emv": 1, "pix": 2},
        "payment_method": {"debit": 0, "credit": 1, "null": -1},
        "card_brand": {"mastercard": 0, "visa": 1, "null": -1},
        "document_type": {"cnpj": 0, "cpf": 1, "null": -1},
    }
    with pytest.raises(ValueError, match="does not match"):
        load_active_set_snapshot(active_path, mapping=mismatched_mapping)
    snapshot = load_active_set_snapshot(active_path, mapping=LABEL_MAPPING)
    assert len(snapshot) == 0


def test_save_history_and_active_snapshot_require_a_processed_population(tmp_path):
    manager = ActiveSetManager(use_active_set=True)

    with pytest.raises(ValueError, match="has not processed"):
        manager.save_history(tmp_path / "history.json")
    with pytest.raises(ValueError, match="has not processed"):
        manager.save_active_snapshot(tmp_path / "active.json")


def test_snapshot_round_trip_resolves_the_run_level_mapping(tmp_path):
    expressions = [Add(MeanAmount(0), CountTotal(0)), Mul(MaxAmount(1), CountTotal(1))]
    objectives = ((0.8, 0.8, 0.8, 1.0), (0.9, 0.7, 0.9, 1.5))
    snapshot_path = tmp_path / "snapshots" / "generation_000000.json"

    write_snapshot(
        snapshot_path,
        expressions,
        objectives,
        minimize=(False, False, False, True),
        mapping_ref=SNAPSHOT_MAPPING_REFERENCE,
    )

    snapshot = load_snapshot(snapshot_path, LABEL_MAPPING)
    assert snapshot.version == archive_module.SNAPSHOT_FORMAT_VERSION
    assert snapshot.minimize == (False, False, False, True)
    assert snapshot.mapping == LABEL_MAPPING
    assert [str(expression) for expression in snapshot.expressions] == [
        str(expression) for expression in expressions
    ]
    assert snapshot.objectives == objectives


def test_snapshot_rejects_an_embedded_mapping_copy(tmp_path):
    snapshot_path = tmp_path / "snapshots" / "generation_000000.json"
    snapshot_path.parent.mkdir(parents=True)
    document = build_snapshot_document(
        [Add(MeanAmount(0), CountTotal(0))],
        [(0.8, 0.8, 0.8, 1.0)],
        minimize=(False, False, False, True),
        mapping_ref=SNAPSHOT_MAPPING_REFERENCE,
    )
    document["mapping"] = LABEL_MAPPING
    snapshot_path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="must not embed a mapping copy"):
        load_snapshot(snapshot_path, LABEL_MAPPING)


def test_snapshot_requires_the_run_level_mapping(tmp_path):
    snapshot_path = tmp_path / "snapshots" / "generation_000000.json"
    write_snapshot(
        snapshot_path,
        [Add(MeanAmount(0), CountTotal(0))],
        [(0.8, 0.8, 0.8, 1.0)],
        minimize=(False, False, False, True),
        mapping_ref=SNAPSHOT_MAPPING_REFERENCE,
    )

    with pytest.raises(ValueError, match="run-level mapping"):
        load_snapshot(snapshot_path, None)


def test_snapshot_rejects_missing_or_malformed_mapping_ref(tmp_path):
    snapshot_path = tmp_path / "snapshots" / "generation_000000.json"
    snapshot_path.parent.mkdir(parents=True)
    document = build_snapshot_document(
        [Add(MeanAmount(0), CountTotal(0))],
        [(0.8, 0.8, 0.8, 1.0)],
        minimize=(False, False, False, True),
        mapping_ref=SNAPSHOT_MAPPING_REFERENCE,
    )
    del document["mapping_ref"]
    snapshot_path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="mapping_ref"):
        load_snapshot(snapshot_path, LABEL_MAPPING)

    document["mapping_ref"] = {"file": "manifest.json"}
    snapshot_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="'file' and 'source' keys"):
        load_snapshot(snapshot_path, LABEL_MAPPING)

    document["mapping_ref"] = {"file": "", "source": "run_manifest"}
    snapshot_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="non-empty string"):
        load_snapshot(snapshot_path, LABEL_MAPPING)


def test_snapshot_rejects_unsupported_version_and_unknown_format(tmp_path):
    snapshot_path = tmp_path / "snapshots" / "generation_000000.json"
    snapshot_path.parent.mkdir(parents=True)
    document = build_snapshot_document(
        [Add(MeanAmount(0), CountTotal(0))],
        [(0.8, 0.8, 0.8, 1.0)],
        minimize=(False, False, False, True),
        mapping_ref=SNAPSHOT_MAPPING_REFERENCE,
    )
    document["version"] = 999
    snapshot_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Unsupported snapshot version"):
        load_snapshot(snapshot_path, LABEL_MAPPING)

    document["version"] = archive_module.SNAPSHOT_FORMAT_VERSION
    document["format"] = "other-format"
    snapshot_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Unknown snapshot format"):
        load_snapshot(snapshot_path, LABEL_MAPPING)


def test_snapshot_rejects_an_incomplete_run_level_mapping(tmp_path):
    snapshot_path = tmp_path / "snapshots" / "generation_000000.json"
    write_snapshot(
        snapshot_path,
        [Add(MeanAmount(0), CountTotal(0))],
        [(0.8, 0.8, 0.8, 1.0)],
        minimize=(False, False, False, True),
        mapping_ref=SNAPSHOT_MAPPING_REFERENCE,
    )

    incomplete = {"status": {"approved": 0}}
    with pytest.raises(ValueError, match="cannot resolve snapshot expressions"):
        load_snapshot(snapshot_path, incomplete)


def test_snapshot_writes_are_atomic(tmp_path, monkeypatch):
    replaced = []
    original_replace = os.replace

    def recording_replace(source, destination):
        replaced.append((Path(source).name, Path(destination).name))
        return original_replace(source, destination)

    monkeypatch.setattr(archive_module.os, "replace", recording_replace)
    snapshot_path = tmp_path / "snapshots" / "generation_000000.json"
    write_snapshot(
        snapshot_path,
        [Add(MeanAmount(0), CountTotal(0))],
        [(0.8, 0.8, 0.8, 1.0)],
        minimize=(False, False, False, True),
        mapping_ref=SNAPSHOT_MAPPING_REFERENCE,
    )

    assert len(replaced) == 1
    assert replaced[0][0].startswith(".generation_000000.json.")
    assert replaced[0][0].endswith(".tmp")
    assert list(tmp_path.glob("*.tmp")) == []
    load_snapshot(snapshot_path, LABEL_MAPPING)


def test_archive_snapshot_is_mapping_free_and_round_trips(tmp_path):
    expressions = [Add(MeanAmount(0), CountTotal(0)), Mul(MaxAmount(1), CountTotal(1))]
    scores = {
        str(expressions[0]): (0.8, 0.8, 0.8, 1.0),
        str(expressions[1]): (0.9, 0.7, 0.9, 1.5),
    }
    problem = make_grammar_problem(scores)
    step = ArchiveStep(mapping=LABEL_MAPPING)
    run_archive_step(step, problem, grammar_evaluated_individuals(problem, expressions))

    document = step.archive_snapshot()
    assert "mapping" not in document
    assert document["mapping_ref"] == SNAPSHOT_MAPPING_REFERENCE
    assert document["problem"] == {
        "number_of_objectives": 4,
        "minimize": [False, False, False, True],
    }

    snapshot_path = tmp_path / "snapshots" / "generation_000000.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(document))
    snapshot = load_snapshot(snapshot_path, LABEL_MAPPING)
    assert snapshot.version == archive_module.SNAPSHOT_FORMAT_VERSION
    assert [str(expression) for expression in snapshot.expressions] == [
        str(expression) for expression in expressions
    ]
    assert snapshot.objectives == ((0.8, 0.8, 0.8, 1.0), (0.9, 0.7, 0.9, 1.5))
    assert snapshot.minimize == (False, False, False, True)


def test_archive_snapshot_requires_an_evaluated_population():
    step = ArchiveStep(mapping=LABEL_MAPPING)

    with pytest.raises(ValueError, match="has not evaluated"):
        step.archive_snapshot()


def test_archive_snapshot_matches_the_latest_auto_saved_front(tmp_path):
    archive_path = tmp_path / "archive.json"
    step = ArchiveStep(archive_path=archive_path, mapping=LABEL_MAPPING)
    winner = Mul(MaxAmount(1), CountTotal(1))
    scores = {str(winner): (0.9, 0.9, 0.9, 0.5)}
    problem = make_grammar_problem(scores)

    run_archive_step(step, problem, grammar_evaluated_individuals(problem, [winner]))

    standalone = load_archive(archive_path)
    document = step.archive_snapshot()
    assert [str(expression) for expression in standalone.expressions] == [
        str(expression) for expression in (
            decode_expression(entry["expression"])
            for entry in document["expressions"]
        )
    ]
    assert standalone.objectives == tuple(
        tuple(entry["objectives"]) for entry in document["expressions"]
    )
