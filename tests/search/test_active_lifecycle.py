from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from automatedfe.features.grammar import MeanAmount
from automatedfe.search.archive import (
    SNAPSHOT_MAPPING_REFERENCE,
    ActiveSetManager,
    ArchiveStep,
    build_snapshot_document,
)
from automatedfe.search.lifecycle import SearchLifecycleRecorder
from geneticengine.evaluation.sequential import SequentialEvaluator
from automatedfe.search.search import (
    CandidateEvaluator,
    MaterializingArchiveSearch,
    canonical_expression_key,
)
from geneticengine.problems import Fitness, MultiObjectiveProblem
from geneticengine.solutions.individual import ConcreteIndividual


@dataclass(frozen=True)
class Expression:
    name: str


def test_first_promotion_selects_sequential_winners_and_versions():
    expressions = [Expression("a"), Expression("b"), Expression("c")]
    scores = {
        "a": (0.20, 0.20, 0.20, 3.0),
        "b": (0.30, 0.30, 0.30, 2.0),
        "c": (0.40, 0.40, 0.40, 1.0),
    }
    signals = {
        "a": np.array([0.0, 1.0, 0.0, 1.0]),
        "b": np.array([0.0, 0.0, 1.0, 1.0]),
        "c": np.array([0.0, 1.0, 2.0, 1.0]),
    }
    problem = MultiObjectiveProblem(
        lambda expression: list(scores[expression.name]),
        minimize=[False, False, False, True],
    )
    archive = ActiveSetManager(
        signal_provider=lambda expression: signals[expression.name],
        use_active_set=True,
        promotion_refresh_top_n=0,
        first_promotion_top_k=2,
        promotion_min_gain=0.0,
        promotion_mean_gain=0.0,
        active_correlation_threshold=0.99,
    )
    individuals = [ConcreteIndividual(expression) for expression in expressions]
    archive.history = individuals
    archive._history_signals = [signals[expression.name] for expression in expressions]
    archive._history_keys = {
        archive._expression_key(individual) for individual in individuals
    }
    archive.admission_objectives = {
        archive._expression_key(individual): scores[individual.get_phenotype().name]
        for individual in individuals
    }
    archive._problem = problem

    class VersionedEvaluator:
        def evaluate(self, evaluation_problem, candidates):
            del evaluation_problem
            for candidate in candidates:
                version = archive.baseline_version
                gain = {
                    ("a", 0): (0.20, 0.20, 0.20),
                    ("b", 0): (0.40, 0.40, 0.40),
                    ("c", 0): (0.50, 0.50, 0.50),
                    ("a", 1): (0.20, 0.20, 0.20),
                    ("b", 1): (0.45, 0.45, 0.45),
                }[(candidate.get_phenotype().name, version)]
                candidate.set_fitness(
                    problem,
                    Fitness([*gain, 1.0], valid=True),
                )
                yield candidate

    assert archive.maybe_promote(problem, 5, evaluator=VersionedEvaluator())
    assert [individual.get_phenotype().name for individual in archive.active_individuals] == [
        "c",
        "b",
    ]
    assert archive.baseline_version == 2
    assert [
        row["baseline_version"]
        for row in archive.promotion_checks
        if row["outcome"] == "promoted"
    ] == [0, 1]


def test_incremental_promotion_appends_one_candidate_at_later_interval():
    expressions = [Expression("a"), Expression("b"), Expression("c"), Expression("d")]
    scores = {
        "a": (0.20, 0.20, 0.20, 3.0),
        "b": (0.30, 0.30, 0.30, 2.0),
        "c": (0.40, 0.40, 0.40, 1.0),
        "d": (0.10, 0.10, 0.10, 0.5),
    }
    signals = {
        "a": np.array([0.0, 1.0, 0.0, 1.0]),
        "b": np.array([0.0, 0.0, 1.0, 1.0]),
        "c": np.array([0.0, 1.0, 2.0, 1.0]),
        "d": np.array([1.0, 0.0, 1.0, 0.0]),
    }
    problem = MultiObjectiveProblem(
        lambda expression: list(scores[expression.name]),
        minimize=[False, False, False, True],
    )
    archive = ActiveSetManager(
        signal_provider=lambda expression: signals[expression.name],
        use_active_set=True,
        promotion_refresh_top_n=0,
        first_promotion_top_k=2,
        promotion_add_k=1,
        promotion_min_gain=0.0,
        promotion_mean_gain=0.0,
        active_correlation_threshold=0.99,
    )
    individuals = [ConcreteIndividual(expression) for expression in expressions]
    archive.history = individuals
    archive._history_signals = [signals[expression.name] for expression in expressions]
    archive._history_keys = {
        archive._expression_key(individual) for individual in individuals
    }
    archive.admission_objectives = {
        archive._expression_key(individual): scores[individual.get_phenotype().name]
        for individual in individuals
    }
    archive._problem = problem

    class VersionedEvaluator:
        def evaluate(self, evaluation_problem, candidates):
            del evaluation_problem
            for candidate in candidates:
                version = archive.baseline_version
                gain = {
                    ("a", 0): (0.20, 0.20, 0.20),
                    ("b", 0): (0.40, 0.40, 0.40),
                    ("c", 0): (0.50, 0.50, 0.50),
                    ("d", 0): (0.15, 0.15, 0.15),
                    ("a", 1): (0.20, 0.20, 0.20),
                    ("b", 1): (0.45, 0.45, 0.45),
                    ("d", 1): (0.15, 0.15, 0.15),
                    ("a", 2): (0.22, 0.22, 0.22),
                    ("d", 2): (0.15, 0.15, 0.15),
                }[(candidate.get_phenotype().name, version)]
                candidate.set_fitness(
                    problem,
                    Fitness([*gain, 1.0], valid=True),
                )
                yield candidate

    assert archive.maybe_promote(problem, 5, evaluator=VersionedEvaluator())
    assert [individual.get_phenotype().name for individual in archive.active_individuals] == [
        "c",
        "b",
    ]
    assert archive.baseline_version == 2

    assert archive.maybe_promote(problem, 10, evaluator=VersionedEvaluator())
    assert [individual.get_phenotype().name for individual in archive.active_individuals] == [
        "c",
        "b",
        "a",
    ]
    assert archive.baseline_version == 3
    assert [
        row["baseline_version"]
        for row in archive.promotion_checks
        if row["outcome"] == "promoted"
    ] == [0, 1, 2]


def test_promotion_refresh_top_n_zero_disables_only_the_refresh(monkeypatch):
    expressions = [Expression("a"), Expression("b")]
    scores = {
        "a": (0.20, 0.20, 0.20, 3.0),
        "b": (0.30, 0.30, 0.30, 2.0),
    }
    signals = {
        "a": np.array([0.0, 1.0, 0.0, 1.0]),
        "b": np.array([0.0, 0.0, 1.0, 1.0]),
    }
    problem = MultiObjectiveProblem(
        lambda expression: list(scores[expression.name]),
        minimize=[False, False, False, True],
    )
    archive = ActiveSetManager(
        signal_provider=lambda expression: signals[expression.name],
        use_active_set=True,
        promotion_refresh_top_n=0,
        first_promotion_top_k=1,
        promotion_min_gain=0.0,
        promotion_mean_gain=0.0,
        active_correlation_threshold=0.99,
    )
    individuals = [ConcreteIndividual(expression) for expression in expressions]
    archive.history = individuals
    archive._history_signals = [signals[expression.name] for expression in expressions]
    archive._history_keys = {
        archive._expression_key(individual) for individual in individuals
    }
    archive.admission_objectives = {
        archive._expression_key(individual): scores[individual.get_phenotype().name]
        for individual in individuals
    }
    archive._problem = problem

    refresh_calls = []
    monkeypatch.setattr(
        archive,
        "_refresh_history",
        lambda _evaluator, _problem: refresh_calls.append(1),
    )

    class Evaluating:
        def evaluate(self, evaluation_problem, candidates):
            del evaluation_problem
            for candidate in candidates:
                candidate.set_fitness(
                    problem,
                    Fitness([0.50, 0.50, 0.50, 1.0], valid=True),
                )
                yield candidate

    assert archive.maybe_promote(problem, 5, evaluator=Evaluating())
    assert refresh_calls == []
    assert len(archive.active_individuals) == 1

    refresh_calls.clear()
    archive.active_individuals = []
    archive.promotion_refresh_top_n = 1
    assert archive.maybe_promote(problem, 10, evaluator=Evaluating())
    assert refresh_calls == [1]


def test_exact_promotion_ties_preserve_stable_admission_order():
    expressions = [Expression("x"), Expression("y")]
    scores = {
        "x": (0.30, 0.30, 0.30, 2.0),
        "y": (0.30, 0.30, 0.30, 2.0),
    }
    signals = {
        "x": np.array([0.0, 1.0, 0.0, 1.0]),
        "y": np.array([0.0, 0.0, 1.0, 1.0]),
    }
    problem = MultiObjectiveProblem(
        lambda expression: list(scores[expression.name]),
        minimize=[False, False, False, True],
    )
    archive = ActiveSetManager(
        signal_provider=lambda expression: signals[expression.name],
        use_active_set=True,
        promotion_refresh_top_n=0,
        first_promotion_top_k=1,
        promotion_min_gain=0.0,
        promotion_mean_gain=0.0,
        active_correlation_threshold=0.99,
    )
    individuals = [ConcreteIndividual(expression) for expression in expressions]
    archive.history = individuals
    archive._history_signals = [signals[expression.name] for expression in expressions]
    archive._history_keys = {
        archive._expression_key(individual) for individual in individuals
    }
    archive.admission_objectives = {
        archive._expression_key(individual): scores[individual.get_phenotype().name]
        for individual in individuals
    }
    archive._problem = problem

    class TiedEvaluator:
        def evaluate(self, evaluation_problem, candidates):
            del evaluation_problem
            for candidate in candidates:
                candidate.set_fitness(
                    problem,
                    Fitness([0.50, 0.50, 0.50, 2.0], valid=True),
                )
                yield candidate

    assert archive.maybe_promote(problem, 5, evaluator=TiedEvaluator())
    assert [individual.get_phenotype().name for individual in archive.active_individuals] == [
        "x"
    ]


def test_promotion_diagnostics_contain_generation_phase_version_and_gains():
    expressions = [Expression("a"), Expression("b"), Expression("c")]
    scores = {
        "a": (0.20, 0.20, 0.20, 3.0),
        "b": (0.30, 0.30, 0.30, 2.0),
        "c": (0.40, 0.40, 0.40, 1.0),
    }
    signals = {
        "a": np.array([0.0, 1.0, 0.0, 1.0]),
        "b": np.array([0.0, 0.0, 1.0, 1.0]),
        "c": np.array([0.0, 1.0, 2.0, 1.0]),
    }
    problem = MultiObjectiveProblem(
        lambda expression: list(scores[expression.name]),
        minimize=[False, False, False, True],
    )
    archive = ActiveSetManager(
        signal_provider=lambda expression: signals[expression.name],
        use_active_set=True,
        promotion_refresh_top_n=0,
        first_promotion_top_k=1,
        promotion_min_gain=0.0,
        promotion_mean_gain=0.0,
        active_correlation_threshold=0.99,
    )
    individuals = [ConcreteIndividual(expression) for expression in expressions]
    archive.history = individuals
    archive._history_signals = [signals[expression.name] for expression in expressions]
    archive._history_keys = {
        archive._expression_key(individual) for individual in individuals
    }
    archive.admission_objectives = {
        archive._expression_key(individual): scores[individual.get_phenotype().name]
        for individual in individuals
    }
    archive._problem = problem

    class Evaluating:
        def evaluate(self, evaluation_problem, candidates):
            del evaluation_problem
            for candidate in candidates:
                candidate.set_fitness(
                    problem,
                    Fitness([0.50, 0.50, 0.50, 1.0], valid=True),
                )
                yield candidate

    assert archive.maybe_promote(problem, 5, evaluator=Evaluating())
    assert archive.promotion_checks
    for row in archive.promotion_checks:
        assert row["generation"] == 5
        assert row["phase"] in {"first_promotion", "incremental_promotion"}
        assert isinstance(row["baseline_version"], int)
        assert isinstance(row["expression"], str)
        assert isinstance(row["proxy_gains"], list)
        assert isinstance(row["current_gains"], list)
        assert isinstance(row["minimum_gain_threshold"], float)
        assert isinstance(row["mean_gain_threshold"], float)
        assert row["outcome"] in {"promoted", "checked", "rejected", "not_selected"}
        assert isinstance(row["reason"], str)


def test_active_set_manager_does_not_change_canonical_archive_membership():
    scores = {
        "a": (0.8, 0.8, 0.8, 1.0),
        "b": (0.8, 0.9, 0.7, 2.0),
        "c": (0.9, 0.9, 0.9, 3.0),
    }
    signals = {
        "a": np.array([0.0, 1.0, 2.0, 3.0]),
        "b": np.array([0.0, 2.0, 4.0, 6.0]),
        "c": np.array([3.0, 0.0, 2.0, 1.0]),
    }
    problem = MultiObjectiveProblem(
        lambda expression: list(scores[expression.name]),
        minimize=[False, False, False, True],
    )
    individuals = [ConcreteIndividual(Expression(name)) for name in scores]
    for individual in individuals:
        individual.set_fitness(problem, problem.evaluate(individual.get_phenotype()))

    archive = ArchiveStep()
    manager = ActiveSetManager(
        signal_provider=lambda expression: signals[expression.name],
        use_active_set=True,
    )
    evaluated = list(
        archive.apply(
            problem,
            SequentialEvaluator(),
            representation=None,
            random=None,
            population=iter(individuals),
            target_size=len(individuals),
            generation=0,
        )
    )
    manager.process_evaluated_population(problem, evaluated, generation=0)

    assert [item.get_phenotype().name for item in archive.archive] == ["a", "b", "c"]
    assert [item.get_phenotype().name for item in manager.history] == ["a", "c"]


def test_candidate_evaluator_invalidates_fitness_after_baseline_refresh():
    version = {"value": 0}
    problem = MultiObjectiveProblem(
        lambda _expression: [0.1, 0.1, 0.1, 1.0],
        minimize=[False, False, False, True],
    )
    candidate = ConcreteIndividual("candidate")
    evaluator = CandidateEvaluator(lambda: version["value"])

    list(evaluator.evaluate(problem, [candidate]))
    assert candidate.has_fitness(problem)
    assert candidate.metadata["evaluated_baseline_version"] == 0

    version["value"] = 1
    list(evaluator.evaluate(problem, [candidate]))
    assert candidate.metadata["evaluated_baseline_version"] == 1
    assert evaluator.number_of_evaluations() == 2


def test_lifecycle_recorder_counts_baseline_refresh_reevaluations():
    version = {"value": 0}
    recorder = SearchLifecycleRecorder(strategy="genetic")
    evaluator = CandidateEvaluator(
        lambda: version["value"],
        lifecycle=recorder,
    )
    expression = MeanAmount(0)
    problem = MultiObjectiveProblem(
        lambda _expression: [0.2, 0.2, 0.2, 1.0],
        minimize=[False, False, False, True],
    )
    candidate = ConcreteIndividual(expression)

    recorder.on_generation_started(0)
    recorder.on_candidate_generated(candidate)
    list(evaluator.evaluate(problem, [candidate]))
    recorder.on_generation_completed(
        0,
        build_snapshot_document(
            [expression],
            [(0.2, 0.2, 0.2, 1.0)],
            minimize=(False, False, False, True),
            mapping_ref=SNAPSHOT_MAPPING_REFERENCE,
        ),
    )

    version["value"] = 1
    recorder.on_generation_started(1)
    list(evaluator.evaluate(problem, [candidate]))
    recorder.on_generation_completed(
        1,
        build_snapshot_document(
            [expression],
            [(0.2, 0.2, 0.2, 1.0)],
            minimize=(False, False, False, True),
            mapping_ref=SNAPSHOT_MAPPING_REFERENCE,
        ),
    )

    assert len(recorder.candidate_rows) == 1
    assert recorder.candidate_rows[0]["Status"] == "evaluated"
    assert recorder.candidate_rows[0]["Generation"] == 0
    assert [row["Evaluated"] for row in recorder.generation_rows] == [1, 1]
    assert [row["Unique"] for row in recorder.generation_rows] == [1, 0]
    assert evaluator.number_of_evaluations() == 2

    recorder.on_search_completed(
        canonical_expression_key(expression) for expression in (expression,)
    )
    assert recorder.candidate_rows[0]["ArchiveMember"] is True


def test_promotion_boundary_is_called_once_before_generation_evaluation():
    class RecordingArchive:
        archive = []

        def __init__(self):
            self.calls = []
            self.refreshes = []

        def maybe_promote(self, problem, generation, *, evaluator):
            self.calls.append((problem, generation, evaluator))
            return generation == 5

        def reevaluate_archive(self, problem, evaluator):
            self.refreshes.append((problem, evaluator))

    class RecordingFitness:
        def __init__(self):
            self.invalidations = 0

        def invalidate_baseline_cache(self):
            self.invalidations += 1

    archive = RecordingArchive()
    fitness = RecordingFitness()
    search = object.__new__(MaterializingArchiveSearch)
    search.archive_step = archive
    search.problem = object()
    search.tracker = SimpleNamespace(evaluator=object())
    search.fitness_evaluator = fitness
    search._promotion_boundaries = set()

    assert search._promote_at_boundary(4) is False
    assert search._promote_at_boundary(5) is True
    assert search._promote_at_boundary(5) is False
    assert [call[1] for call in archive.calls] == [4, 5]
    assert archive.calls[1][2] is search.tracker.evaluator
    assert fitness.invalidations == 1
    assert archive.refreshes == [(search.problem, search.tracker.evaluator)]


def test_archive_is_reevaluated_and_rebuilt_after_baseline_change():
    version = {"value": 0}
    scores = {
        0: {
            "a": [0.9, 0.1, 0.1, 1.0],
            "b": [0.1, 0.9, 0.1, 1.0],
        },
        1: {
            "a": [0.1, 0.1, 0.1, 1.0],
            "b": [0.2, 0.2, 0.2, 1.0],
        },
    }
    problem = MultiObjectiveProblem(
        lambda expression: scores[version["value"]][expression.name],
        minimize=[False, False, False, True],
    )
    archive = ActiveSetManager()
    individuals = [ConcreteIndividual(Expression(name)) for name in ("a", "b")]
    for individual in individuals:
        individual.set_fitness(
            problem,
            Fitness(scores[0][individual.get_phenotype().name], valid=True),
        )
        individual.metadata["evaluated_baseline_version"] = 0
    archive.archive = individuals

    version["value"] = 1
    evaluator = CandidateEvaluator(lambda: version["value"])
    archive.reevaluate_archive(problem, evaluator)

    assert evaluator.number_of_evaluations() == 2
    assert [item.get_phenotype().name for item in archive.archive] == ["b"]
    assert archive.archive[0].get_fitness(problem).fitness_components == scores[1]["b"]
