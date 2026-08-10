"""The stateful archive step for multiobjective feature search."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, fields
from os import PathLike
from pathlib import Path

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

_NODE_TYPES: dict[str, type] = {
    node_type.__name__: node_type
    for node_type in (*TERMINALS, *NON_TERMINALS)
}


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


__all__ = [
    "FORMAT_IDENTIFIER",
    "FORMAT_VERSION",
    "OBJECTIVES_PER_ARCHIVE",
    "ArchiveSnapshot",
    "ArchiveStep",
    "decode_expression",
    "encode_expression",
    "load_archive",
]
