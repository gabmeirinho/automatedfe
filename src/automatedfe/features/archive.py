"""The stateful archive step for multiobjective feature search."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, fields
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.evaluation import Evaluator
from geneticengine.problems import Problem
from geneticengine.problems.helpers import non_dominated
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import Representation
from geneticengine.solutions.individual import PhenotypicIndividual

from .feature_schema import code_lists_from_mapping
from .grammar import NON_TERMINALS, TERMINALS, expr

logger = logging.getLogger(__name__)

FORMAT_IDENTIFIER = "automatedfe-archive"
FORMAT_VERSION = 1
OBJECTIVES_PER_ARCHIVE = 4
ARCHIVE_PROXY_OBJECTIVES = 3
DEFAULT_ARCHIVE_QUALITY_THRESHOLD = 0.001
DEFAULT_ARCHIVE_CORRELATION_THRESHOLD = 0.85

_NODE_TYPES: dict[str, type] = {
    node_type.__name__: node_type
    for node_type in (*TERMINALS, *NON_TERMINALS)
}


def validate_correlation_threshold(threshold: float) -> float:
    """Validate and normalize an absolute-correlation threshold."""

    try:
        value = float(threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("correlation threshold must be a finite number") from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("correlation threshold must be between 0 and 1")
    return value


def validate_archive_quality_threshold(threshold: float) -> float:
    """Validate the minimum per-fold proxy improvement threshold."""

    try:
        value = float(threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("archive quality threshold must be a finite number") from error
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("archive quality threshold must be finite and non-negative")
    return value


def _normalize_signal(signal: object) -> tuple[np.ndarray | None, str | None]:
    """Return a one-dimensional finite, non-constant signal or its rejection."""

    try:
        values = np.asarray(signal, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return None, "signal_invalid"
    if values.size == 0:
        return None, "signal_empty"
    if not np.all(np.isfinite(values)):
        return None, "signal_nonfinite"
    if float(np.std(values)) <= 1e-12:
        return None, "signal_constant"
    return values, None


def absolute_pearson_correlation(signal_1: object, signal_2: object) -> float:
    """Return the absolute Pearson correlation of two valid signals.

    ``nan`` is returned when either signal is empty, non-finite, constant, or
    has a different length.  Returning a value instead of raising keeps
    candidate-local signal problems inside the archive filtering path.
    """

    first, first_reason = _normalize_signal(signal_1)
    second, second_reason = _normalize_signal(signal_2)
    if (
        first is None
        or second is None
        or first_reason is not None
        or second_reason is not None
        or first.size != second.size
    ):
        return float("nan")

    first_centered = first - np.mean(first)
    second_centered = second - np.mean(second)
    denominator = float(np.linalg.norm(first_centered) * np.linalg.norm(second_centered))
    if denominator <= 1e-12:
        return float("nan")
    return min(
        1.0,
        max(0.0, abs(float(np.dot(first_centered, second_centered) / denominator))),
    )


def correlation_rejection(
    candidate: object,
    reference_signals: Sequence[object],
    threshold: float,
) -> dict[str, Any]:
    """Return deterministic diagnostics for candidate/reference correlation.

    The first reference reaching the inclusive threshold is reported.  When
    no reference reaches it, ``matched_index`` identifies the reference with
    the largest observed absolute correlation, which makes diagnostics stable
    without changing the admission decision.
    """

    threshold = validate_correlation_threshold(threshold)
    candidate_values, candidate_reason = _normalize_signal(candidate)
    if candidate_values is None:
        return {
            "rejected": True,
            "reason": candidate_reason or "signal_invalid",
            "abs_corr": float("nan"),
            "matched_index": None,
            "max_abs_corr": float("nan"),
        }

    maximum = float("nan")
    maximum_index: int | None = None
    for index, reference in enumerate(reference_signals):
        reference_values, reference_reason = _normalize_signal(reference)
        if reference_values is None:
            return {
                "rejected": True,
                "reason": reference_reason or "reference_invalid",
                "abs_corr": float("nan"),
                "matched_index": index,
                "max_abs_corr": maximum,
            }
        if reference_values.size != candidate_values.size:
            return {
                "rejected": True,
                "reason": "signal_shape_mismatch",
                "abs_corr": float("nan"),
                "matched_index": index,
                "max_abs_corr": maximum,
            }

        absolute_correlation = absolute_pearson_correlation(
            candidate_values,
            reference_values,
        )
        if not math.isfinite(absolute_correlation):
            return {
                "rejected": True,
                "reason": "correlation_invalid",
                "abs_corr": float("nan"),
                "matched_index": index,
                "max_abs_corr": maximum,
            }
        if maximum_index is None or absolute_correlation > maximum:
            maximum = absolute_correlation
            maximum_index = index
        # Centering and the norm-based implementation can be a few ulps below
        # one for a signal compared with itself.  Treat the public threshold
        # as inclusive while remaining deterministic.
        if absolute_correlation + 1e-12 >= threshold:
            return {
                "rejected": True,
                "reason": "pairwise_threshold",
                "abs_corr": absolute_correlation,
                "matched_index": index,
                "max_abs_corr": maximum,
            }

    return {
        "rejected": False,
        "reason": "",
        "abs_corr": maximum,
        "matched_index": maximum_index,
        "max_abs_corr": maximum,
    }


def is_correlated_pairwise(signal_1: object, signal_2: object, threshold: float) -> bool:
    """Return whether two signals have inclusive absolute correlation."""

    threshold = validate_correlation_threshold(threshold)
    correlation = absolute_pearson_correlation(signal_1, signal_2)
    return bool(
        math.isfinite(correlation) and correlation + 1e-12 >= threshold
    )


def encode_expression(node: expr) -> dict[str, object]:
    """Serialize a grammar expression into an allowlisted JSON structure."""

    node_type = type(node)
    if node_type not in _NODE_TYPES.values():
        raise TypeError(f"Unsupported expression node type: {node_type.__name__}")
    return {
        "type": node_type.__name__,
        "fields": {
            field.name: _encode_field(getattr(node, field.name))
            for field in fields(node)
        },
    }


def _encode_field(value: object) -> object:
    if isinstance(value, int):
        return value
    if isinstance(value, expr):
        return encode_expression(value)
    raise TypeError(f"Unsupported expression field of type {type(value).__name__}")


def decode_expression(data: object) -> expr:
    """Reconstruct a grammar expression from a serialized JSON structure."""

    if not isinstance(data, dict):
        raise TypeError(
            f"Serialized expression must be a JSON object, got {type(data).__name__}"
        )
    node_name = data.get("type")
    if not isinstance(node_name, str):
        raise TypeError("Serialized expression is missing its 'type' name")
    node_type = _NODE_TYPES.get(node_name)
    if node_type is None:
        raise ValueError(f"Unknown expression node type: {node_name!r}")
    fields_data = data.get("fields")
    if not isinstance(fields_data, dict):
        raise TypeError(
            f"Serialized {node_name} expression is missing its 'fields' object"
        )
    expected = {field.name for field in fields(node_type)}
    if set(fields_data) != expected:
        raise ValueError(
            f"{node_name} expects fields {sorted(expected)}, got {sorted(fields_data)}"
        )
    try:
        return node_type(
            **{name: _decode_field(value) for name, value in fields_data.items()}
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"Cannot reconstruct {node_name}: {error}") from error


def _decode_field(value: object) -> object:
    if isinstance(value, bool):
        raise TypeError(f"Boolean is not a valid expression field: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        return decode_expression(value)
    raise ValueError(f"Unsupported expression field value: {value!r}")


@dataclass(frozen=True)
class ArchiveSnapshot:
    """A loaded archive: reconstructed expressions and their objectives."""

    version: int
    minimize: tuple[bool, ...]
    mapping: Mapping[str, Mapping[str, int]]
    expressions: tuple[expr, ...]
    objectives: tuple[tuple[float, ...], ...]

    def __len__(self) -> int:
        return len(self.expressions)


def _resolve_mapping(
    mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None,
) -> Mapping[str, Mapping[str, int]]:
    """Resolve a mapping argument into a full label-mapping dict."""

    if mapping is None:
        from ..encoding import DEFAULT_MAPPING_OUTPUT, load_label_mapping

        return load_label_mapping(DEFAULT_MAPPING_OUTPUT)
    if isinstance(mapping, (str, PathLike)):
        from ..encoding import load_label_mapping

        return load_label_mapping(Path(mapping))
    return mapping


def _validate_mapping_compatible(
    stored_mapping: Mapping[str, Mapping[str, int]],
    provided_mapping: Mapping[str, Mapping[str, int]],
) -> None:
    try:
        stored = code_lists_from_mapping(stored_mapping)
        provided = code_lists_from_mapping(provided_mapping)
    except ValueError as error:
        raise ValueError(f"Cannot compare label mappings: {error}") from error
    if stored != provided:
        raise ValueError(
            "Archive label mapping is incompatible with the provided mapping"
        )


def _build_document(
    expressions: Sequence[expr],
    objectives: Sequence[Sequence[float]],
    *,
    minimize: Sequence[bool],
    mapping: Mapping[str, Mapping[str, int]],
) -> dict[str, object]:
    return {
        "format": FORMAT_IDENTIFIER,
        "version": FORMAT_VERSION,
        "problem": {
            "number_of_objectives": len(minimize),
            "minimize": [bool(value) for value in minimize],
        },
        "mapping": {family: dict(values) for family, values in mapping.items()},
        "expressions": [
            {
                "expression": encode_expression(expression),
                "objectives": [float(value) for value in entry_objectives],
            }
            for expression, entry_objectives in zip(expressions, objectives)
        ],
    }


def _validate_entries(data: object, n_objectives: int) -> list[dict[str, object]]:
    if not isinstance(data, list):
        raise TypeError("Archive 'expressions' must be a JSON list")
    entries: list[dict[str, object]] = []
    for index, entry in enumerate(data):
        if not isinstance(entry, dict) or set(entry) != {"expression", "objectives"}:
            raise ValueError(
                f"Archive entry {index} must contain exactly "
                "'expression' and 'objectives' keys"
            )
        objectives = entry["objectives"]
        if not isinstance(objectives, list) or len(objectives) != n_objectives:
            raise ValueError(
                f"Archive entry {index} must declare exactly "
                f"{n_objectives} objective values"
            )
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in objectives
        ):
            raise ValueError(
                f"Archive entry {index} must declare numeric objective values"
            )
        entries.append(entry)
    return entries


def _validate_document(
    data: object,
    path: Path,
) -> tuple[int, dict[str, object], dict[str, object], list[dict[str, object]]]:
    """Validate a loaded archive document; return (version, problem, mapping, entries)."""

    if not isinstance(data, dict):
        raise TypeError(f"Archive JSON must be an object: {path}")
    if data.get("format") != FORMAT_IDENTIFIER:
        raise ValueError(f"Unknown archive format: {data.get('format')!r}")
    version = data.get("version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported archive version {version!r} (expected {FORMAT_VERSION})"
        )
    problem = data.get("problem")
    if not isinstance(problem, dict):
        raise TypeError("Archive is missing its 'problem' metadata")
    n_objectives = problem.get("number_of_objectives")
    minimize = problem.get("minimize")
    if n_objectives != OBJECTIVES_PER_ARCHIVE:
        raise ValueError(
            f"Archive problem must declare {OBJECTIVES_PER_ARCHIVE} objectives"
        )
    if not isinstance(minimize, list) or len(minimize) != n_objectives:
        raise ValueError(
            f"Archive problem must declare exactly {n_objectives} "
            "minimization directions"
        )
    if not all(isinstance(value, bool) for value in minimize):
        raise ValueError("Archive minimization directions must be boolean values")
    mapping = data.get("mapping")
    if not isinstance(mapping, dict):
        raise TypeError("Archive is missing its 'mapping' metadata")
    try:
        code_lists_from_mapping(mapping)
    except ValueError as error:
        raise ValueError(f"Archive label mapping is invalid: {error}") from error
    entries = _validate_entries(data.get("expressions"), n_objectives)
    return version, problem, mapping, entries


def _atomic_write_json(path: Path, data: object) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
            json.dump(data, temp_file, indent=2, sort_keys=True)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp_name)
        raise
    return path


def load_archive(
    path: str | PathLike[str],
    *,
    mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None = None,
) -> ArchiveSnapshot:
    """Load and validate an archive JSON snapshot.

    The label mapping embedded in the archive is validated against *mapping*
    when one is provided, so categorical expressions are only reconstructed
    when their code lists are compatible. Loading never merges or resumes a
    search.
    """

    archive_path = Path(path).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive JSON file does not exist: {archive_path}")
    try:
        with open(archive_path) as archive_file:
            data = json.load(archive_file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Archive is not valid JSON: {archive_path}: {error}") from error

    version, problem, mapping_data, entries = _validate_document(data, archive_path)
    if mapping is not None:
        _validate_mapping_compatible(mapping_data, _resolve_mapping(mapping))

    expressions = tuple(decode_expression(entry["expression"]) for entry in entries)
    objectives = tuple(tuple(entry["objectives"]) for entry in entries)
    return ArchiveSnapshot(
        version=version,
        minimize=tuple(problem["minimize"]),
        mapping=mapping_data,
        expressions=expressions,
        objectives=objectives,
    )


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

    When *archive_path* is provided, an atomic JSON snapshot of the current
    front is written after every generation. *mapping* supplies the label
    mapping embedded in those snapshots; it defaults to the persisted
    preprocessing mapping.
    """

    def __init__(
        self,
        *,
        archive_path: str | PathLike[str] | None = None,
        mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None = None,
    ) -> None:
        self.archive: list[PhenotypicIndividual] = []
        self._problem: Problem | None = None
        self._mapping = mapping
        self.archive_path: Path | None = (
            Path(archive_path).resolve() if archive_path is not None else None
        )

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
        self._problem = problem
        evaluated = list(evaluator.evaluate(problem, population))
        current = self._valid_unique(evaluated, problem)
        candidates = self._deduplicate([*self.archive, *current])

        self.archive = list(non_dominated(iter(candidates), problem))
        if self.archive_path is not None:
            self.save(self.archive_path)
        yield from evaluated

    def save(
        self,
        path: str | PathLike[str] | None = None,
        *,
        mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None = None,
    ) -> Path:
        """Write an atomic JSON snapshot of the current archive front.

        Only the current strict Pareto front is saved. *mapping* defaults to
        the mapping configured for the step, then to the persisted default.
        """

        save_path = self._resolve_save_path(path)
        if self._problem is None:
            raise ValueError("Cannot save an archive that has not evaluated a population")
        resolved_mapping = mapping if mapping is not None else self._mapping
        document = _build_document(
            expressions=[individual.get_phenotype() for individual in self.archive],
            objectives=[
                tuple(
                    float(value)
                    for value in individual.get_fitness(self._problem).fitness_components
                )
                for individual in self.archive
            ],
            minimize=self._problem.minimize,
            mapping=_resolve_mapping(resolved_mapping),
        )
        return _atomic_write_json(save_path, document)

    def _resolve_save_path(self, path: str | PathLike[str] | None) -> Path:
        if path is None:
            if self.archive_path is None:
                raise ValueError(
                    "No archive path configured; pass an explicit path to save()"
                )
            return self.archive_path
        return Path(path).resolve()

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
        # Keep archive identity structural for grammar expressions.  The
        # helper's compatibility fallback preserves support for the small
        # non-grammar expression objects accepted by the historical API.
        from .search.search import canonical_expression_key

        return canonical_expression_key(individual.get_phenotype())

    @staticmethod
    def _log_invalid(key: str, objectives: object) -> None:
        logger.warning(
            "Skipping invalid archive candidate %s with objectives=%r",
            key,
            objectives,
        )


class FilteredArchiveStep(ArchiveStep):
    """Maintain the filtered history used by the GP archive variant.

    The ordinary :class:`ArchiveStep` is intentionally left unchanged for
    enumerative and random search.  This subclass adds the GP admission
    state while retaining the inherited ``archive`` front and its JSON
    serialization contract.

    For every generation, the complete evaluated population is first reduced
    to its four-objective Pareto front.  That generation front is then
    filtered in this order:

    1. minimum improvement on all three proxy folds;
    2. absolute Pearson correlation against every previously admitted history
       signal; and
    3. greedy same-generation peer clustering, retaining the best candidate in
       each correlated cluster.

    The final ``archive`` remains a global Pareto front, while ``history`` is
    unbounded, structurally unique, and ordered by admission.  Admission
    objectives are copied into ``admission_objectives`` so a later evaluator
    may refresh an individual's live fitness without changing the values that
    justified history admission.

    ``signal_provider`` receives a candidate phenotype and must return its
    full training-row signal.  For small hermetic tests, a provider accepting
    the individual itself is also supported as a compatibility fallback.
    """

    def __init__(
        self,
        *,
        archive_path: str | PathLike[str] | None = None,
        mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None = None,
        signal_provider: Callable[[object], object] | None = None,
        archive_quality_threshold: float = DEFAULT_ARCHIVE_QUALITY_THRESHOLD,
        archive_correlation_threshold: float = DEFAULT_ARCHIVE_CORRELATION_THRESHOLD,
        # These aliases keep the component convenient to use from focused
        # tests and make the terminology match both the plan and the reference.
        min_proxy_improvement: float | None = None,
        correlation_threshold: float | None = None,
        signal_function: Callable[[object], object] | None = None,
    ) -> None:
        super().__init__(archive_path=archive_path, mapping=mapping)
        if min_proxy_improvement is not None:
            archive_quality_threshold = min_proxy_improvement
        if correlation_threshold is not None:
            archive_correlation_threshold = correlation_threshold
        if signal_provider is not None and signal_function is not None:
            raise ValueError("pass only one of signal_provider or signal_function")

        self.archive_quality_threshold = validate_archive_quality_threshold(
            archive_quality_threshold
        )
        self.archive_correlation_threshold = validate_correlation_threshold(
            archive_correlation_threshold
        )
        self.signal_provider = (
            signal_provider if signal_provider is not None else signal_function
        )
        self.history: list[PhenotypicIndividual] = []
        self.admission_objectives: dict[str, tuple[float, ...]] = {}
        self._history_signals: list[np.ndarray] = []
        self._history_keys: set[str] = set()
        self.admission_diagnostics: list[dict[str, object]] = []
        # ``filter_diagnostics`` is the name used by the reference workflow;
        # keep it as the public spelling while retaining the more explicit
        # ``admission_diagnostics`` alias.
        self.filter_diagnostics = self.admission_diagnostics
        self.generation_diagnostics: list[dict[str, object]] = []

    @property
    def history_individuals(self) -> tuple[PhenotypicIndividual, ...]:
        """Return history candidates in immutable admission order."""

        return tuple(self.history)

    @property
    def history_objectives(self) -> tuple[tuple[float, ...], ...]:
        """Return immutable admission objectives aligned with ``history``."""

        return tuple(
            self.admission_objectives[self._expression_key(individual)]
            for individual in self.history
            if self._expression_key(individual) in self.admission_objectives
        )

    @property
    def history_signals(self) -> tuple[np.ndarray, ...]:
        """Return copies of the signals used for history correlation checks."""

        return tuple(signal.copy() for signal in self._history_signals)

    @property
    def history_keys(self) -> frozenset[str]:
        """Return structural identities admitted to history."""

        return frozenset(self._history_keys)

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
        self._problem = problem
        evaluated = list(evaluator.evaluate(problem, population))
        self.process_evaluated_population(problem, evaluated, generation)
        if self.archive_path is not None:
            self.save(self.archive_path)
        yield from evaluated

    def process_evaluated_population(
        self,
        problem: Problem,
        population_list: Sequence[PhenotypicIndividual],
        generation: int,
        evaluator: Evaluator | None = None,
    ) -> list[PhenotypicIndividual]:
        """Process an already evaluated generation and return its front.

        The explicit method mirrors the reference archive API and is useful
        for deterministic unit tests.  ``evaluator`` is accepted for API
        compatibility; the population is already evaluated by definition.
        """

        del evaluator
        self._validate_problem(problem)
        self._problem = problem
        evaluated = list(population_list)
        valid = self._valid_unique_with_diagnostics(evaluated, problem, generation)

        # The public archive keeps the existing global-front semantics.  The
        # history admission front is deliberately calculated from this
        # generation alone, as the GP reference admits useful candidates that
        # may later leave the global final Pareto front.
        global_candidates = self._deduplicate([*self.archive, *valid])
        self.archive = list(non_dominated(iter(global_candidates), problem))
        generation_front = list(non_dominated(iter(valid), problem))
        generation_front_keys = {
            self._expression_key(individual) for individual in generation_front
        }
        for individual in valid:
            key = self._expression_key(individual)
            if key not in generation_front_keys:
                self._record_diagnostic(
                    generation=generation,
                    individual=individual,
                    outcome="not_pareto",
                    reason="dominated_in_generation",
                )

        self._admit_generation(generation_front, problem, generation)
        self.generation_diagnostics.append(
            {
                "generation": generation,
                "population_size": len(evaluated),
                "valid_size": len(valid),
                "front_size": len(generation_front),
                "history_size": len(self.history),
                "archive_size": len(self.archive),
            }
        )
        return generation_front

    def _valid_unique_with_diagnostics(
        self,
        individuals: Sequence[PhenotypicIndividual],
        problem: Problem,
        generation: int,
    ) -> list[PhenotypicIndividual]:
        valid: list[PhenotypicIndividual] = []
        seen: set[str] = set()
        for individual in individuals:
            key = self._expression_key(individual)
            try:
                fitness = individual.get_fitness(problem)
                components = tuple(fitness.fitness_components)
                if not fitness.valid:
                    raise ValueError("fitness_invalid")
                if len(components) != OBJECTIVES_PER_ARCHIVE:
                    raise ValueError("wrong_objective_count")
                objective_values = tuple(float(value) for value in components)
                if not all(math.isfinite(value) for value in objective_values):
                    raise ValueError("objectives_nonfinite")
            except (ArithmeticError, TypeError, ValueError, AttributeError) as error:
                reason = str(error) if str(error) in {
                    "fitness_invalid",
                    "wrong_objective_count",
                    "objectives_nonfinite",
                } else "invalid_fitness"
                self._record_diagnostic(
                    generation=generation,
                    individual=individual,
                    outcome="invalid",
                    reason=reason,
                )
                self._log_invalid(key, getattr(locals().get("fitness", None), "fitness_components", ()))
                continue

            if key in seen:
                self._record_diagnostic(
                    generation=generation,
                    individual=individual,
                    outcome="duplicate",
                    reason="duplicate_generation_expression",
                    objectives=objective_values,
                )
                continue
            seen.add(key)
            valid.append(individual)
        return valid

    def _admit_generation(
        self,
        generation_front: Sequence[PhenotypicIndividual],
        problem: Problem,
        generation: int,
    ) -> None:
        candidates: list[tuple[PhenotypicIndividual, np.ndarray, str, tuple[float, ...]]] = []
        for individual in generation_front:
            key = self._expression_key(individual)
            objectives = tuple(
                float(value)
                for value in individual.get_fitness(problem).fitness_components
            )
            if key in self._history_keys:
                self._record_diagnostic(
                    generation=generation,
                    individual=individual,
                    outcome="duplicate",
                    reason="duplicate_history_expression",
                    objectives=objectives,
                )
                continue

            proxy_minimum = min(objectives[:ARCHIVE_PROXY_OBJECTIVES])
            if proxy_minimum < self.archive_quality_threshold:
                self._record_diagnostic(
                    generation=generation,
                    individual=individual,
                    outcome="rejected",
                    reason="quality_threshold",
                    objectives=objectives,
                    proxy_minimum=proxy_minimum,
                )
                continue

            signal, signal_reason = self._signal_for(individual)
            if signal is None:
                self._record_diagnostic(
                    generation=generation,
                    individual=individual,
                    outcome="invalid",
                    reason=signal_reason or "signal_invalid",
                    objectives=objectives,
                    proxy_minimum=proxy_minimum,
                )
                continue

            historical = correlation_rejection(
                signal,
                self._history_signals,
                self.archive_correlation_threshold,
            )
            if historical["rejected"]:
                self._record_diagnostic(
                    generation=generation,
                    individual=individual,
                    outcome="rejected",
                    reason=f"historical_{historical['reason']}",
                    objectives=objectives,
                    proxy_minimum=proxy_minimum,
                    correlation=historical.get("abs_corr"),
                    matched_index=historical.get("matched_index"),
                )
                continue
            candidates.append((individual, signal, key, objectives))

        clusters: list[dict[str, object]] = []
        for individual, signal, key, objectives in candidates:
            placed = False
            for cluster_index, cluster in enumerate(clusters):
                peer_check = correlation_rejection(
                    signal,
                    cluster["signals"],  # type: ignore[arg-type]
                    self.archive_correlation_threshold,
                )
                if not peer_check["rejected"]:
                    continue

                placed = True
                representative = cluster["representative"]
                representative_objectives = cluster["objectives"]
                candidate_rank = self._rank_key(objectives)
                representative_rank = self._rank_key(representative_objectives)  # type: ignore[arg-type]
                cluster["signals"].append(signal)  # type: ignore[union-attr]
                if candidate_rank > representative_rank:
                    self._record_diagnostic(
                        generation=generation,
                        individual=representative,
                        outcome="peer_replaced",
                        reason="same_generation_peer_cluster",
                        objectives=representative_objectives,  # type: ignore[arg-type]
                    )
                    cluster["representative"] = individual
                    cluster["objectives"] = objectives
                    cluster["key"] = key
                    cluster["signal"] = signal
                    cluster["peer_index"] = cluster_index
                    cluster["diagnostic"] = None
                else:
                    self._record_diagnostic(
                        generation=generation,
                        individual=individual,
                        outcome="rejected",
                        reason="same_generation_peer_cluster",
                        objectives=objectives,
                        correlation=peer_check.get("abs_corr"),
                        matched_index=peer_check.get("matched_index"),
                    )
                break

            if not placed:
                clusters.append(
                    {
                        "representative": individual,
                        "objectives": objectives,
                        "key": key,
                        "signal": signal,
                        "signals": [signal],
                        "peer_index": len(clusters),
                        "diagnostic": None,
                    }
                )

        for cluster in clusters:
            individual = cluster["representative"]
            signal = cluster["signal"]
            key = cluster["key"]
            objectives = cluster["objectives"]
            self.history.append(individual)
            self._history_keys.add(key)
            self._history_signals.append(signal.copy())
            self.admission_objectives[key] = tuple(objectives)
            self._record_diagnostic(
                generation=generation,
                individual=individual,
                outcome="admitted",
                reason="",
                objectives=objectives,
                proxy_minimum=min(objectives[:ARCHIVE_PROXY_OBJECTIVES]),
            )

    def _signal_for(
        self,
        individual: PhenotypicIndividual,
    ) -> tuple[np.ndarray | None, str | None]:
        try:
            if self.signal_provider is not None:
                phenotype = individual.get_phenotype()
                try:
                    raw_signal = self.signal_provider(phenotype)
                except (AttributeError, KeyError, TypeError):
                    raw_signal = self.signal_provider(individual)
            else:
                metadata = getattr(individual, "metadata", {})
                raw_signal = next(
                    (
                        metadata[name]
                        for name in ("signal", "signal_train", "values")
                        if name in metadata
                    ),
                    individual.get_phenotype(),
                )
        except (
            ArithmeticError,
            AttributeError,
            IndexError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            return None, f"signal_error:{type(error).__name__}"
        return _normalize_signal(raw_signal)

    def _record_diagnostic(
        self,
        *,
        generation: int,
        individual: object,
        outcome: str,
        reason: str,
        objectives: Sequence[float] = (),
        proxy_minimum: float | None = None,
        correlation: object = None,
        matched_index: object = None,
    ) -> None:
        key = self._expression_key(individual) if isinstance(individual, PhenotypicIndividual) else ""
        expression = (
            str(individual.get_phenotype())
            if isinstance(individual, PhenotypicIndividual)
            else str(individual)
        )
        record: dict[str, object] = {
            "generation": generation,
            "expression": expression,
            "key": key,
            "outcome": outcome,
            "reason": reason,
        }
        if objectives:
            record["objectives"] = [float(value) for value in objectives]
        if proxy_minimum is not None:
            record["proxy_minimum"] = float(proxy_minimum)
        if correlation is not None:
            value = float(correlation)
            record["abs_corr"] = value if math.isfinite(value) else None
        if matched_index is not None:
            record["matched_index"] = matched_index
        self.admission_diagnostics.append(record)

    @staticmethod
    def _rank_key(objectives: Sequence[float]) -> tuple[float, float]:
        return (
            float(min(objectives[:ARCHIVE_PROXY_OBJECTIVES])),
            -float(objectives[ARCHIVE_PROXY_OBJECTIVES]),
        )


# The aliases make the state machine discoverable under the two names used by
# the plan and by the reference implementation, without creating separate
# behavior or state.
GPArchiveStep = FilteredArchiveStep
FilteredHistoryArchive = FilteredArchiveStep


__all__ = [
    "ARCHIVE_PROXY_OBJECTIVES",
    "DEFAULT_ARCHIVE_CORRELATION_THRESHOLD",
    "DEFAULT_ARCHIVE_QUALITY_THRESHOLD",
    "FORMAT_IDENTIFIER",
    "FORMAT_VERSION",
    "OBJECTIVES_PER_ARCHIVE",
    "ArchiveSnapshot",
    "ArchiveStep",
    "FilteredArchiveStep",
    "FilteredHistoryArchive",
    "GPArchiveStep",
    "absolute_pearson_correlation",
    "correlation_rejection",
    "decode_expression",
    "encode_expression",
    "is_correlated_pairwise",
    "load_archive",
    "validate_archive_quality_threshold",
    "validate_correlation_threshold",
]
