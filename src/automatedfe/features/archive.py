"""The stateful archive step for multiobjective feature search."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator, Sequence

from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.evaluation import Evaluator
from geneticengine.problems import Problem
from geneticengine.problems.helpers import non_dominated
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import Representation
from geneticengine.solutions.individual import PhenotypicIndividual

logger = logging.getLogger(__name__)


class ArchiveStep(GeneticStep):
    """Maintain one global archive while transparently passing populations on.

    The step owns the archive state. For each complete input population it:

    1. evaluates the complete population;
    2. removes invalid and duplicate expressions;
    3. combines the candidates with the previous archive;
    4. delegates front calculation to Genetic Engine's ``non_dominated``; and
    5. stores the resulting global front.

    Every evaluated population member is yielded unchanged so this step can be
    appended to a Genetic Engine ``SequenceStep`` without changing evolution.
    """

    def __init__(self) -> None:
        self.archive: list[PhenotypicIndividual] = []

    @property
    def individuals(self) -> tuple[PhenotypicIndividual, ...]:
        """Return the live archived individuals in stable order."""

        return tuple(self.archive)

    def __len__(self) -> int:
        return len(self.archive)

    def iterate(
        self,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random: RandomSource,
        population: Iterator[PhenotypicIndividual],
        target_size: int,
        generation: int,
    ) -> Iterator[PhenotypicIndividual]:
        self._validate_problem(problem)
        evaluated = list(evaluator.evaluate(problem, population))
        current = self._valid_unique(evaluated, problem)
        candidates = self._deduplicate([*self.archive, *current])

        self.archive = list(non_dominated(iter(candidates), problem))
        yield from evaluated

    @staticmethod
    def _validate_problem(problem: Problem) -> None:
        if problem.number_of_objectives() != 4:
            raise ValueError("ArchiveStep requires four objectives")

    @classmethod
    def _valid_unique(
        cls,
        individuals: Sequence[PhenotypicIndividual],
        problem: Problem,
    ) -> list[PhenotypicIndividual]:
        valid: list[PhenotypicIndividual] = []
        seen: set[str] = set()
        for individual in individuals:
            fitness = individual.get_fitness(problem)
            key = cls._expression_key(individual)
            if not fitness.valid or len(fitness.fitness_components) != problem.number_of_objectives():
                cls._log_invalid(key, fitness.fitness_components)
                continue
            try:
                finite = all(
                    math.isfinite(float(value))
                    for value in fitness.fitness_components
                )
            except (TypeError, ValueError):
                finite = False
            if not finite:
                cls._log_invalid(key, fitness.fitness_components)
                continue
            if key in seen:
                continue
            seen.add(key)
            valid.append(individual)
        return valid

    @classmethod
    def _deduplicate(
        cls,
        individuals: Sequence[PhenotypicIndividual],
    ) -> list[PhenotypicIndividual]:
        unique: list[PhenotypicIndividual] = []
        seen: set[str] = set()
        for individual in individuals:
            key = cls._expression_key(individual)
            if key in seen:
                continue
            seen.add(key)
            unique.append(individual)
        return unique

    @staticmethod
    def _expression_key(individual: PhenotypicIndividual) -> str:
        return str(individual.get_phenotype())

    @staticmethod
    def _log_invalid(key: str, objectives: object) -> None:
        logger.warning(
            "Skipping invalid archive candidate %s with objectives=%r",
            key,
            objectives,
        )


__all__ = [
    "ArchiveStep",
]
