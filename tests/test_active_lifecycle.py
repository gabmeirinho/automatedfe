from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from automatedfe.features.archive import FilteredArchiveStep
from automatedfe.features.search.search import CandidateEvaluator, MaterializingArchiveSearch
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
    archive = FilteredArchiveStep(
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
    archive = FilteredArchiveStep()
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
