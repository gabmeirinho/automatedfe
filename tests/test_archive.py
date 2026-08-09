from dataclasses import dataclass

import pytest
from geneticengine.evaluation.sequential import SequentialEvaluator
from geneticengine.problems import Fitness, MultiObjectiveProblem
from geneticengine.solutions.individual import ConcreteIndividual

import automatedfe.features.archive as archive_module
from automatedfe.features.archive import ArchiveStep


@dataclass(frozen=True)
class Expression:
    name: str

    def __str__(self) -> str:
        return self.name


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


def test_archive_step_merges_generations_into_one_global_front():
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

    assert archived_names(step) == ["winner"]


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
