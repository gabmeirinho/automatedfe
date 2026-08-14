"""Lifecycle observation for evaluated feature searches.

The recorder is the single source of the evidence the run plots and run
bundles consume without reconstructing search history afterward:

* one candidate row per candidate that reached a final state
  (``generated`` for evaluation-free strategies, otherwise ``duplicate``,
  ``materialization_failed``, ``invalid``, or ``evaluated``);
* one generation row per processed generation with consistent counts and
  monotonic cumulative runtime; and
* one mapping-free archive snapshot per processed generation, with
  additions/removals derived from adjacent snapshots.

Observation never changes the search: candidates are recorded by structural
identity, duplicate candidates are counted and rowed but never evaluated
twice, and invalid-after-materialization candidates retain their finite
materialization duration while failed materializations are never reported as
completed evaluations.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from os import PathLike
from pathlib import Path
from time import monotonic_ns
from typing import Any, TextIO

from ..analysis.artifacts import (
    CANDIDATES_COLUMNS,
    write_candidates_csv,
)
from ..features.grammar import collect_features

GENERATED = "generated"
DUPLICATE = "duplicate"
MATERIALIZATION_FAILED = "materialization_failed"
INVALID = "invalid"
EVALUATED = "evaluated"

FINAL_STATUSES = (DUPLICATE, MATERIALIZATION_FAILED, INVALID, EVALUATED, GENERATED)


def _as_expression(individual: Any) -> Any:
    get_phenotype = getattr(individual, "get_phenotype", None)
    if callable(get_phenotype):
        return get_phenotype()
    return individual


class SearchLifecycleRecorder:
    """Observe candidate, generation, and archive lifecycle events.

    The recorder accumulates candidate rows (following
    :data:`automatedfe.analysis.artifacts.CANDIDATES_COLUMNS`), generation
    rows (following ``GENERATIONS_COLUMNS``), and one mapping-free snapshot
    document per processed generation. When *candidate_csv_path* is set, the
    candidates table is also streamed incrementally so an interrupted run
    leaves readable rows; final archive membership is applied on
    :meth:`on_search_completed`.
    """

    def __init__(
        self,
        *,
        strategy: str = "",
        candidate_csv_path: str | PathLike[str] | None = None,
    ) -> None:
        if not isinstance(strategy, str):
            raise ValueError("strategy must be a string")
        self.strategy = strategy
        self.candidate_rows: list[dict[str, object]] = []
        self.generation_rows: list[dict[str, object]] = []
        self.snapshots: dict[int, dict[str, object]] = {}
        # Rows are per candidate instance: structurally equal candidates that
        # are skipped as duplicates get their own row, while a later
        # re-evaluation of the same instance updates no row.  Every evaluated
        # candidate stays referenced by the search, so object ids are stable
        # for the lifetime of a run.
        self._row_ids: dict[int, int] = {}
        self._row_keys: list[str] = []
        self._generation_counts: dict[str, int] = self._empty_counts()
        self._current_generation: int | None = None
        self._generation_started_ns: int | None = None
        self._search_started_ns: int | None = None
        self._previous_archive_keys: frozenset[str] = frozenset()
        self._candidate_csv_path = (
            None
            if candidate_csv_path is None
            else Path(candidate_csv_path).resolve()
        )
        self._candidate_file: TextIO | None = None
        self._candidate_writer: csv.DictWriter | None = None
        if self._candidate_csv_path is not None:
            self._candidate_csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._candidate_file = self._candidate_csv_path.open(
                "w", encoding="utf-8", newline=""
            )
            self._candidate_writer = csv.DictWriter(
                self._candidate_file, fieldnames=CANDIDATES_COLUMNS
            )
            self._candidate_writer.writeheader()
            self._candidate_file.flush()

    @staticmethod
    def _empty_counts() -> dict[str, int]:
        return {
            "generated": 0,
            "duplicate": 0,
            "invalid": 0,
            "evaluated": 0,
            "materialization_failed": 0,
        }

    @property
    def snapshot_documents(self) -> tuple[tuple[int, dict[str, object]], ...]:
        """Return ``(generation, document)`` pairs in ascending order."""

        return tuple(
            (generation, dict(document))
            for generation, document in sorted(self.snapshots.items())
        )

    @property
    def archived_keys(self) -> frozenset[str]:
        """Return the structural identities of the final archive snapshot."""

        if not self.snapshots:
            return frozenset()
        final = self.snapshots[max(self.snapshots)]
        return self._snapshot_keys(final)

    def on_search_started(self) -> None:
        """Start the cumulative runtime clock at search execution time."""

        self._search_started_ns = monotonic_ns()

    def on_generation_started(self, generation: int) -> None:
        """Begin a generation; resets the per-generation counters."""

        if isinstance(generation, bool) or not isinstance(generation, int):
            raise ValueError(
                f"generation must be a non-negative integer, got {generation!r}"
            )
        if generation < 0:
            raise ValueError("generation must be a non-negative integer")
        if self._search_started_ns is None:
            self.on_search_started()
        self._current_generation = generation
        self._generation_started_ns = monotonic_ns()
        self._generation_counts = self._empty_counts()

    def on_candidate_generated(self, individual: Any) -> None:
        """Record one generated candidate (before deduplication)."""

        self._generation_counts["generated"] += 1

    def on_candidate_duplicate(self, individual: Any) -> None:
        """Record a structural duplicate that will never be evaluated."""

        self._generation_counts["duplicate"] += 1
        self._write_candidate_row(individual, status=DUPLICATE, error="")

    def on_materialization_failed(
        self,
        individual: Any,
        error: str,
    ) -> None:
        """Record a candidate that could not be materialized.

        The candidate is never evaluated and never reaches the archive, so it
        does not count as a completed evaluation.
        """

        self._generation_counts["materialization_failed"] += 1
        self._write_candidate_row(
            individual,
            status=MATERIALIZATION_FAILED,
            error=error,
        )

    def on_candidate_evaluated(
        self,
        individual: Any,
        *,
        status: str,
        objectives: Sequence[float] | None,
        duration: float | None,
        error: str,
    ) -> None:
        """Record one completed evaluation.

        ``status`` is ``evaluated`` for valid fitness or ``invalid`` for a
        completed evaluation whose fitness is invalid. An invalid candidate
        that was materialized retains its finite *duration*; a valid candidate
        carries its four objectives (the last of which is the materialization
        duration).
        """

        if status not in (EVALUATED, INVALID):
            raise ValueError(f"unknown evaluated status: {status!r}")
        self._generation_counts["evaluated"] += 1
        if status == INVALID:
            self._generation_counts["invalid"] += 1
        self._write_candidate_row(
            individual,
            status=status,
            objectives=objectives,
            duration=duration,
            error=error,
        )

    def on_generation_completed(
        self,
        generation: int,
        snapshot: Mapping[str, object],
    ) -> None:
        """Retain the generation's archive snapshot and summary row.

        *snapshot* must be a mapping-free structured snapshot document; the
        additions/removals of the summary row are derived from the adjacent
        snapshot documents.
        """

        self.snapshots[generation] = dict(snapshot)
        current_keys = self._snapshot_keys(snapshot)
        added = len(current_keys - self._previous_archive_keys)
        removed = len(self._previous_archive_keys - current_keys)
        self._previous_archive_keys = current_keys
        now = monotonic_ns()
        counts = self._generation_counts
        row: dict[str, object] = {
            "Strategy": self.strategy,
            "Generation": generation,
            "Generated": counts["generated"],
            "Unique": counts["generated"] - counts["duplicate"],
            "Duplicate": counts["duplicate"],
            "Invalid": counts["invalid"],
            "Evaluated": counts["evaluated"],
            "ArchiveSize": len(current_keys),
            "Added": added,
            "Removed": removed,
            "DurationSeconds": self._elapsed(self._generation_started_ns, now),
            "CumulativeRuntimeSeconds": self._elapsed(self._search_started_ns, now),
        }
        self.generation_rows.append(row)

    def on_search_completed(self, archive_keys: Iterable[str]) -> None:
        """Record final archive membership on every candidate row."""

        keys = frozenset(archive_keys)
        for index, key in enumerate(self._row_keys):
            self.candidate_rows[index]["ArchiveMember"] = key in keys
        if self._candidate_csv_path is not None:
            write_candidates_csv(self._candidate_csv_path, self.candidate_rows)

    def record_generated(self, expression: Any) -> None:
        """Record one evaluation-free generated expression."""

        self._write_candidate_row(expression, status=GENERATED, error="")

    def close(self) -> None:
        """Close the incremental candidate output, leaving it readable.

        The output is closed in a readable state whenever the search is
        interrupted; final archive membership is only applied by
        :meth:`on_search_completed`.
        """

        if self._candidate_file is not None:
            self._candidate_file.close()
            self._candidate_file = None
            self._candidate_writer = None

    def _write_candidate_row(
        self,
        individual: Any,
        *,
        status: str,
        objectives: Sequence[float] | None = None,
        duration: float | None = None,
        error: str,
    ) -> None:
        expression = _as_expression(individual)
        from .search import canonical_expression_key

        key = canonical_expression_key(expression)
        if id(individual) in self._row_ids:
            return
        objective_cells: tuple[object, object, object, object]
        if objectives is not None and len(objectives) == 4:
            objective_cells = tuple(objectives)  # type: ignore[assignment]
        else:
            objective_cells = ("", "", "", "")
        if duration is None:
            duration = objective_cells[3]
        row: dict[str, object] = {
            "Strategy": self.strategy,
            "CandidateIndex": len(self.candidate_rows),
            "Generation": (
                "" if self._current_generation is None else self._current_generation
            ),
            "Expression": str(expression),
            "Dependencies": ";".join(
                sorted(feature.name for feature in collect_features(expression))
            ),
            "Split1": objective_cells[0],
            "Split2": objective_cells[1],
            "Split3": objective_cells[2],
            "MaterializationTime": duration,
            "ArchiveMember": "",
            "Status": status,
            "Error": error,
        }
        self.candidate_rows.append(row)
        self._row_ids[id(individual)] = len(self.candidate_rows) - 1
        self._row_keys.append(key)
        if self._candidate_writer is not None and self._candidate_file is not None:
            self._candidate_writer.writerow(row)
            self._candidate_file.flush()

    @staticmethod
    def _elapsed(started_ns: int | None, now_ns: int) -> float:
        if started_ns is None:
            return 0.0
        return (now_ns - started_ns) * 1e-9

    @classmethod
    def _snapshot_keys(cls, snapshot: Mapping[str, object]) -> frozenset[str]:
        from ..analysis.artifacts import canonical_json_text

        entries = snapshot.get("expressions")
        if not isinstance(entries, list):
            return frozenset()
        keys = set()
        for entry in entries:
            if not isinstance(entry, Mapping) or "expression" not in entry:
                continue
            keys.add(canonical_json_text(entry["expression"]))
        return frozenset(keys)


__all__ = [
    "DUPLICATE",
    "EVALUATED",
    "FINAL_STATUSES",
    "GENERATED",
    "INVALID",
    "MATERIALIZATION_FAILED",
    "SearchLifecycleRecorder",
]
