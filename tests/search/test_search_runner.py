import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from geneticengine.solutions.individual import ConcreteIndividual

import automatedfe.search.runner as runner_module
import automatedfe.search.search as shared_search_module
from automatedfe.analysis.artifacts import (
    CANDIDATES_COLUMNS,
    load_run_manifest,
)
from automatedfe.evaluation import MaterializationError, NumericalFitnessError
from automatedfe.features.grammar import MeanAmount, TotalAmount
from automatedfe.search.archive import (
    SNAPSHOT_MAPPING_REFERENCE,
    load_archive,
    load_snapshot,
)
from automatedfe.search.lifecycle import SearchLifecycleRecorder
from automatedfe.search.runner import (
    SearchAnalysisError,
    SearchRunResult,
    SearchStrategy,
)
from automatedfe.search.runner import (
    _run_feature_search_impl as run_feature_search,
)
from automatedfe.search.runner import (
    run_feature_search as run_tracked_feature_search,
)
from automatedfe.search.search import canonical_expression_key

LABEL_MAPPING = {
    "status": {"approved": 0, "complete": 1},
    "capture_method": {"contactless": 0},
    "payment_method": {"credit": 0},
    "card_brand": {"visa": 0},
    "document_type": {"cpf": 0},
}


class _FinalEvaluator:
    def __init__(self, received):
        self.received = received

    def evaluate(
        self,
        expressions,
        *,
        search_fold_scores=None,
        include_diagnostics=True,
    ):
        del search_fold_scores, include_diagnostics
        self.received.extend(expressions)
        return SimpleNamespace(metrics={"roc_auc": 0.75})

    def evaluate_additive_ensemble(self, expressions):
        self.received.extend(expressions)
        return SimpleNamespace(
            metrics={"train_auc": 0.80, "test_auc": 0.70},
            train_predictions=None,
            test_predictions=None,
            models=(),
            expressions=tuple(expressions),
        )


def test_evaluation_free_runner_writes_common_generated_rows(tmp_path, monkeypatch):
    received = []

    class StubMaterializer:
        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(runner_module, "FeatureMaterializer", StubMaterializer)
    monkeypatch.setattr(
        runner_module,
        "_build_final_evaluator",
        lambda *_args, **_kwargs: _FinalEvaluator(received),
    )
    csv_path = tmp_path / "diagnostics.csv"

    result = run_feature_search(
        SearchStrategy.ENUMERATIVE_WITHOUT_ARCHIVE,
        candidate_count=3,
        dataset_path=tmp_path / "dataset.parquet",
        mapping=LABEL_MAPPING,
        mmap_dir=tmp_path / "mmap",
        csv_path=csv_path,
    )

    rows = list(csv.DictReader(csv_path.open(newline="")))
    assert tuple(rows[0]) == CANDIDATES_COLUMNS
    assert [row["CandidateIndex"] for row in rows] == ["0", "1", "2"]
    assert {row["Strategy"] for row in rows} == {
        "enumerative_without_archive"
    }
    assert {row["Status"] for row in rows} == {"generated"}
    assert {row["ArchiveMember"] for row in rows} == {"False"}
    assert all(
        not row[column]
        for row in rows
        for column in (
            "Generation",
            "Split1",
            "Split2",
            "Split3",
            "MaterializationTime",
            "Error",
        )
    )
    assert tuple(received) == result.expressions
    assert result.evaluated_count == 0


def test_runner_preflights_all_outputs_before_search_setup(tmp_path, monkeypatch):
    csv_path = tmp_path / "existing.csv"
    csv_path.write_text("keep me")
    setup_called = False

    def fail_if_built(*_args, **_kwargs):
        nonlocal setup_called
        setup_called = True
        raise AssertionError("search setup should not run")

    monkeypatch.setattr(
        runner_module,
        "build_unbound_enumerative_search",
        fail_if_built,
    )

    with pytest.raises(FileExistsError, match="force=True"):
        run_feature_search(
            "enumerative_without_archive",
            candidate_count=1,
            dataset_path=tmp_path / "dataset.parquet",
            mapping=LABEL_MAPPING,
            csv_path=csv_path,
        )

    assert not setup_called
    assert csv_path.read_text() == "keep me"


def test_runner_rejects_archive_output_for_evaluation_free_strategy(tmp_path):
    with pytest.raises(ValueError, match="archive_path is not supported"):
        run_feature_search(
            "enumerative_without_archive",
            candidate_count=1,
            dataset_path=tmp_path / "dataset.parquet",
            mapping=LABEL_MAPPING,
            archive_path=tmp_path / "archive.json",
        )


def test_evaluated_runner_finalizes_membership_and_saves_one_archive(
    tmp_path,
    monkeypatch,
):
    received = []

    class StubMaterializer:
        def __init__(self, *_args, **_kwargs):
            pass

    class StubFitness:
        def __init__(self, *_args, **_kwargs):
            pass

        def prepare_population(self, _expressions):
            pass

        def objective_vector(self, _expression):
            return [0.1, 0.2, 0.3, 0.01]

    monkeypatch.setattr(shared_search_module, "FeatureMaterializer", StubMaterializer)
    monkeypatch.setattr(shared_search_module, "ResidualEvaluator", StubFitness)
    monkeypatch.setattr(
        runner_module,
        "_build_final_evaluator",
        lambda *_args, **_kwargs: _FinalEvaluator(received),
    )
    csv_path = tmp_path / "diagnostics.csv"
    archive_path = tmp_path / "archive.json"

    result = run_feature_search(
        "enumerative",
        time_budget_seconds=0.001,
        dataset_path=tmp_path / "dataset.parquet",
        mapping=LABEL_MAPPING,
        mmap_dir=tmp_path / "mmap",
        csv_path=csv_path,
        archive_path=archive_path,
    )

    rows = list(csv.DictReader(csv_path.open(newline="")))
    snapshot = load_archive(archive_path, mapping=LABEL_MAPPING)
    assert rows
    assert all(row["Generation"] for row in rows)
    assert {row["Status"] for row in rows} == {"evaluated"}
    assert sum(row["ArchiveMember"] == "True" for row in rows) == len(snapshot)
    assert snapshot.expressions == result.expressions
    assert snapshot.objectives == result.objectives
    assert tuple(received) == result.expressions


def test_invalid_evaluated_rows_survive_empty_archive_failure(tmp_path, monkeypatch):
    class StubMaterializer:
        def __init__(self, *_args, **_kwargs):
            pass

    class InvalidFitness:
        def __init__(self, *_args, **_kwargs):
            pass

        def prepare_population(self, _expressions):
            pass

        def objective_vector(self, _expression):
            return [0.1, float("nan"), 0.3, 0.01]

    monkeypatch.setattr(shared_search_module, "FeatureMaterializer", StubMaterializer)
    monkeypatch.setattr(shared_search_module, "ResidualEvaluator", InvalidFitness)
    csv_path = tmp_path / "invalid.csv"

    with pytest.raises(ValueError, match="empty archive"):
        run_feature_search(
            "enumerative",
            time_budget_seconds=0.000001,
            dataset_path=tmp_path / "dataset.parquet",
            mapping=LABEL_MAPPING,
            mmap_dir=tmp_path / "mmap",
            csv_path=csv_path,
        )

    rows = list(csv.DictReader(csv_path.open(newline="")))
    assert len(rows) == 1
    assert rows[0]["Status"] == "invalid"
    assert rows[0]["Error"] == "invalid objective vector"
    assert rows[0]["ArchiveMember"] == "False"
    assert all(
        not rows[0][column]
        for column in (
            "Split1",
            "Split2",
            "Split3",
            "MaterializationTime",
        )
    )


def test_force_allows_runner_outputs_to_be_replaced(tmp_path, monkeypatch):
    received = []

    class StubMaterializer:
        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(runner_module, "FeatureMaterializer", StubMaterializer)
    monkeypatch.setattr(
        runner_module,
        "_build_final_evaluator",
        lambda *_args, **_kwargs: _FinalEvaluator(received),
    )
    csv_path = tmp_path / "diagnostics.csv"
    csv_path.write_text("old contents")

    run_feature_search(
        "enumerative_without_archive",
        candidate_count=1,
        dataset_path=tmp_path / "dataset.parquet",
        mapping=LABEL_MAPPING,
        mmap_dir=tmp_path / "mmap",
        csv_path=csv_path,
        force=True,
    )

    assert csv_path.read_text().startswith("Strategy,CandidateIndex,")


def test_active_runner_evaluates_archive_and_active_set_separately(
    tmp_path,
    monkeypatch,
):
    archive_expression = MeanAmount(0)
    active_expression = TotalAmount(0)
    received = []
    builder_arguments = {}

    class StubArchive:
        use_active_set = True
        archive = [archive_expression]
        active_individuals = [active_expression]

    class StubSearch:
        archive_step = StubArchive()
        archive = archive_step
        materializer = object()
        tracker = SimpleNamespace(
            start_time=0,
            get_number_evaluations=lambda: 0,
            recorders=[],
        )
        generated_count = 2
        invalid_count = 0
        duplicate_count = 0
        grammar_exhausted = False

        def search(self):
            return ["history-expression"]

    def build_search(*_args, **kwargs):
        builder_arguments.update(kwargs)
        return StubSearch()

    monkeypatch.setattr(runner_module, "build_search_algorithm", build_search)
    monkeypatch.setattr(
        runner_module,
        "_build_final_evaluator",
        lambda *_args, **_kwargs: _FinalEvaluator(received),
    )

    result = run_feature_search(
        "genetic",
        time_budget_seconds=0.001,
        dataset_path=tmp_path / "dataset.parquet",
        mapping=LABEL_MAPPING,
        mmap_dir=tmp_path / "mmap",
        use_active_set=True,
    )

    assert builder_arguments["use_active_set"] is True
    assert result.expressions == (archive_expression,)
    assert result.active_set_expressions == (active_expression,)
    assert received == [archive_expression, active_expression, active_expression]
    assert result.archive_final_evaluation is result.final_evaluation
    assert result.active_set_final_evaluation is not None
    assert result.active_set_final_metrics == {"roc_auc": 0.75}
    assert result.additive_metrics == {"train_auc": 0.80, "test_auc": 0.70}
    assert result.history_count == 0
    assert result.active_set_count == 1
    assert result.additive_evaluation_duration_seconds is not None


def test_runner_final_evaluation_uses_complete_canonical_archive(
    tmp_path,
    monkeypatch,
):
    archive_expressions = (MeanAmount(0), TotalAmount(0))
    received = []

    class StubArchive:
        use_active_set = False
        archive = list(archive_expressions)

    class StubSearch:
        archive_step = StubArchive()
        archive = archive_step
        materializer = object()
        tracker = SimpleNamespace(
            start_time=0,
            get_number_evaluations=lambda: 2,
            recorders=[],
        )
        generated_count = 2
        invalid_count = 0
        duplicate_count = 0
        grammar_exhausted = False

        def search(self):
            return ["stale-search-result"]

    monkeypatch.setattr(
        runner_module,
        "build_enumerative_search",
        lambda *_args, **_kwargs: StubSearch(),
    )
    monkeypatch.setattr(
        runner_module,
        "_build_final_evaluator",
        lambda *_args, **_kwargs: _FinalEvaluator(received),
    )

    result = run_feature_search(
        "enumerative",
        time_budget_seconds=0.001,
        dataset_path=tmp_path / "dataset.parquet",
        mapping=LABEL_MAPPING,
        mmap_dir=tmp_path / "mmap",
    )

    assert result.expressions == archive_expressions
    assert received == list(archive_expressions)


def test_runner_rejects_history_paths_without_active_set(tmp_path):
    with pytest.raises(ValueError, match="require use_active_set=True"):
        run_feature_search(
            "genetic",
            time_budget_seconds=0.001,
            dataset_path=tmp_path / "dataset.parquet",
            mapping=LABEL_MAPPING,
            history_path=tmp_path / "history.json",
        )
    with pytest.raises(ValueError, match="require use_active_set=True"):
        run_feature_search(
            "genetic",
            time_budget_seconds=0.001,
            dataset_path=tmp_path / "dataset.parquet",
            mapping=LABEL_MAPPING,
            active_archive_path=tmp_path / "active.json",
        )


def test_runner_preflights_history_and_active_collisions(tmp_path, monkeypatch):
    setup_called = False

    def fail_if_built(*_args, **_kwargs):
        nonlocal setup_called
        setup_called = True
        raise AssertionError("search setup should not run")

    monkeypatch.setattr(runner_module, "build_search_algorithm", fail_if_built)

    with pytest.raises(ValueError, match="must identify different files"):
        run_feature_search(
            "genetic",
            time_budget_seconds=0.001,
            dataset_path=tmp_path / "dataset.parquet",
            mapping=LABEL_MAPPING,
            use_active_set=True,
            archive_path=tmp_path / "artifact.json",
            history_path=tmp_path / "artifact.json",
        )
    path = tmp_path / "existing.json"
    path.write_text("keep me")
    with pytest.raises(FileExistsError, match="force=True"):
        run_feature_search(
            "genetic",
            time_budget_seconds=0.001,
            dataset_path=tmp_path / "dataset.parquet",
            mapping=LABEL_MAPPING,
            use_active_set=True,
            history_path=path,
        )

    assert not setup_called
    assert path.read_text() == "keep me"


def test_active_runner_persists_history_and_active_snapshot(tmp_path, monkeypatch):
    received = []
    saved = {"history": None, "active": None}

    class StubManager:
        history_individuals = ("history-expression",)

        def save_history(self, path, *, mapping=None):
            saved["history"] = (Path(path).resolve(), mapping)
            Path(path).write_text(json.dumps({"format": "history"}))

        def save_active_snapshot(self, path, *, mapping=None):
            saved["active"] = (Path(path).resolve(), mapping)
            Path(path).write_text(json.dumps({"format": "active"}))

    class StubArchive:
        use_active_set = True
        archive = [MeanAmount(0)]
        active_individuals = [TotalAmount(0)]

        def save(self, path, *, mapping=None):
            Path(path).write_text(json.dumps({"format": "archive"}))

    class StubSearch:
        archive_step = StubArchive()
        archive = archive_step
        active_set_manager = StubManager()
        materializer = object()
        tracker = SimpleNamespace(
            start_time=0,
            get_number_evaluations=lambda: 0,
            recorders=[],
        )
        generated_count = 2
        invalid_count = 0
        duplicate_count = 0
        grammar_exhausted = False

        def search(self):
            return []

    monkeypatch.setattr(runner_module, "build_search_algorithm", lambda *_a, **_k: StubSearch())
    monkeypatch.setattr(
        runner_module,
        "_build_final_evaluator",
        lambda *_args, **_kwargs: _FinalEvaluator(received),
    )

    archive_path = tmp_path / "archive.json"
    history_path = tmp_path / "history.json"
    active_path = tmp_path / "active.json"
    run_feature_search(
        "genetic",
        time_budget_seconds=0.001,
        dataset_path=tmp_path / "dataset.parquet",
        mapping=LABEL_MAPPING,
        mmap_dir=tmp_path / "mmap",
        use_active_set=True,
        archive_path=archive_path,
        history_path=history_path,
        active_archive_path=active_path,
    )

    assert json.loads(archive_path.read_text()) == {"format": "archive"}
    assert json.loads(history_path.read_text()) == {"format": "history"}
    assert json.loads(active_path.read_text()) == {"format": "active"}
    assert saved["history"][0] == history_path.resolve()
    assert saved["active"][0] == active_path.resolve()
    assert saved["history"][1] == LABEL_MAPPING
    assert saved["active"][1] == LABEL_MAPPING


def test_loose_runner_outputs_are_not_structured_runs(tmp_path, monkeypatch):
    class StubMaterializer:
        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(runner_module, "FeatureMaterializer", StubMaterializer)
    monkeypatch.setattr(
        runner_module,
        "_build_final_evaluator",
        lambda *_args, **_kwargs: _FinalEvaluator([]),
    )
    csv_path = tmp_path / "diagnostics.csv"

    run_feature_search(
        "enumerative_without_archive",
        candidate_count=1,
        dataset_path=tmp_path / "dataset.parquet",
        mapping=LABEL_MAPPING,
        mmap_dir=tmp_path / "mmap",
        csv_path=csv_path,
    )

    assert csv_path.is_file()
    assert not any(tmp_path.rglob("manifest.json"))
    assert not any(tmp_path.rglob("*.sha256"))
    with pytest.raises(ValueError, match="Not a structured run"):
        load_run_manifest(tmp_path)


class _FiniteGenerator:
    """Yield one deterministic batch of candidates and then exhaust."""

    def __init__(self, expressions):
        self.expressions = list(expressions)
        self.exhausted = False

    def generate(self, _previous, _generation):
        if self.exhausted:
            return []
        self.exhausted = True
        return [ConcreteIndividual(expression) for expression in self.expressions]


def _evaluated_runner_harness(
    tmp_path,
    monkeypatch,
    *,
    expressions,
    fitness_cls,
):
    """Run one evaluated strategy with a fixed candidate batch and stub fitness."""

    received = []

    class StubMaterializer:
        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(shared_search_module, "FeatureMaterializer", StubMaterializer)
    monkeypatch.setattr(shared_search_module, "ResidualEvaluator", fitness_cls)
    monkeypatch.setattr(
        runner_module,
        "_build_final_evaluator",
        lambda *_args, **_kwargs: _FinalEvaluator(received),
    )
    original_builder = runner_module.build_enumerative_search

    def build_with_batch(*args, **kwargs):
        search = original_builder(*args, **kwargs)
        search.candidate_generator = _FiniteGenerator(expressions)
        return search

    monkeypatch.setattr(runner_module, "build_enumerative_search", build_with_batch)
    csv_path = tmp_path / "diagnostics.csv"
    result = run_feature_search(
        "enumerative",
        time_budget_seconds=0.001,
        dataset_path=tmp_path / "dataset.parquet",
        mapping=LABEL_MAPPING,
        mmap_dir=tmp_path / "mmap",
        csv_path=csv_path,
    )
    rows = list(csv.DictReader(csv_path.open(newline="")))
    return result, rows


class _ValidFitness:
    def __init__(self, *_args, **_kwargs):
        pass

    def prepare_population(self, _expressions):
        pass

    def objective_vector(self, _expression):
        return [0.1, 0.2, 0.3, 0.01]


def test_runner_records_duplicates_without_evaluating_twice(tmp_path, monkeypatch):
    result, rows = _evaluated_runner_harness(
        tmp_path,
        monkeypatch,
        expressions=[MeanAmount(0), MeanAmount(0), MeanAmount(1)],
        fitness_cls=_ValidFitness,
    )

    assert len(rows) == 3
    assert {row["Status"] for row in rows} == {"duplicate", "evaluated"}
    assert sum(row["Status"] == "duplicate" for row in rows) == 1
    assert sum(row["Status"] == "evaluated" for row in rows) == 2
    assert all(row["ArchiveMember"] == "True" for row in rows)
    assert result.generated_count == 3
    assert result.duplicate_count == 1
    assert result.accepted_count == 2
    assert result.evaluated_count == 2

    assert len(result.lifecycle.generation_rows) == 1
    generation = result.lifecycle.generation_rows[0]
    assert generation["Generated"] == 3
    assert generation["Unique"] == 2
    assert generation["Duplicate"] == 1
    assert generation["Invalid"] == 0
    assert generation["Evaluated"] == 2
    assert generation["ArchiveSize"] == 2
    assert generation["Added"] == 2
    assert generation["DurationSeconds"] >= 0
    assert generation["CumulativeRuntimeSeconds"] >= 0


class _MaterializationFailingFitness:
    def __init__(self, *_args, **_kwargs):
        pass

    def prepare_population(self, expressions):
        if str(expressions[0]) == str(MeanAmount(1)):
            raise MaterializationError(
                f"cannot materialize {expressions[0]!s}: boom"
            )

    def objective_vector(self, _expression):
        return [0.1, 0.2, 0.3, 0.01]


def test_runner_distinguishes_materialization_failure_from_invalid(
    tmp_path,
    monkeypatch,
):
    result, rows = _evaluated_runner_harness(
        tmp_path,
        monkeypatch,
        expressions=[MeanAmount(0), MeanAmount(1)],
        fitness_cls=_MaterializationFailingFitness,
    )

    by_expression = {row["Expression"]: row for row in rows}
    failed = by_expression[str(MeanAmount(1))]
    assert failed["Status"] == "materialization_failed"
    assert "cannot materialize" in failed["Error"]
    assert failed["MaterializationTime"] == ""
    assert all(
        not failed[column] for column in ("Split1", "Split2", "Split3")
    )
    assert failed["ArchiveMember"] == "False"
    assert by_expression[str(MeanAmount(0))]["Status"] == "evaluated"
    assert by_expression[str(MeanAmount(0))]["ArchiveMember"] == "True"

    assert result.evaluated_count == 1
    assert result.invalid_count == 0
    generation = result.lifecycle.generation_rows[0]
    assert generation["Generated"] == 2
    assert generation["Unique"] == 2
    assert generation["Evaluated"] == 1
    assert generation["Invalid"] == 0
    assert generation["ArchiveSize"] == 1


class _InvalidWithDurationFitness:
    def __init__(self, *_args, **_kwargs):
        self.last_materialization_duration = None

    def prepare_population(self, _expressions):
        pass

    def objective_vector(self, expression):
        self.last_materialization_duration = 0.42
        if str(expression) == str(MeanAmount(1)):
            raise NumericalFitnessError("fold scoring exploded")
        return [0.1, 0.2, 0.3, 0.01]


def test_invalid_after_materialization_retains_finite_duration(
    tmp_path,
    monkeypatch,
):
    result, rows = _evaluated_runner_harness(
        tmp_path,
        monkeypatch,
        expressions=[MeanAmount(0), MeanAmount(1)],
        fitness_cls=_InvalidWithDurationFitness,
    )

    by_expression = {row["Expression"]: row for row in rows}
    invalid = by_expression[str(MeanAmount(1))]
    assert invalid["Status"] == "invalid"
    assert invalid["MaterializationTime"] == "0.42"
    assert "NumericalFitnessError" in invalid["Error"]
    assert all(not invalid[column] for column in ("Split1", "Split2", "Split3"))
    assert invalid["ArchiveMember"] == "False"
    assert by_expression[str(MeanAmount(0))]["Status"] == "evaluated"
    assert by_expression[str(MeanAmount(0))]["MaterializationTime"] == "0.01"

    assert result.invalid_count == 1
    generation = result.lifecycle.generation_rows[0]
    assert generation["Evaluated"] == 2
    assert generation["Invalid"] == 1


def test_runner_lifecycle_emits_generation_history_and_mapping_free_snapshots(
    tmp_path,
    monkeypatch,
):
    expressions = [MeanAmount(0), MeanAmount(1), TotalAmount(1)]
    result, rows = _evaluated_runner_harness(
        tmp_path,
        monkeypatch,
        expressions=expressions,
        fitness_cls=_ValidFitness,
    )

    lifecycle = result.lifecycle
    assert len(lifecycle.generation_rows) == 1
    assert sorted(lifecycle.snapshots) == [0]
    assert lifecycle.generation_rows[0]["Generation"] == 0
    assert lifecycle.generation_rows[0]["ArchiveSize"] == 3
    assert lifecycle.generation_rows[0]["Added"] == 3

    generation, document = lifecycle.snapshot_documents[0]
    assert generation == 0
    assert "mapping" not in document
    assert document["mapping_ref"] == SNAPSHOT_MAPPING_REFERENCE

    snapshot_path = tmp_path / "snapshots" / "generation_000000.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(document))
    snapshot = load_snapshot(snapshot_path, LABEL_MAPPING)
    assert len(snapshot) == 3
    assert [str(expression) for expression in snapshot.expressions] == [
        str(expression) for expression in result.expressions
    ]
    assert snapshot.minimize == (False, False, False, True)
    assert lifecycle.archived_keys == {
        canonical_expression_key(expression) for expression in result.expressions
    }
    assert all(row["ArchiveMember"] == "True" for row in rows)


class _ExplodingFitness:
    def __init__(self, *_args, **_kwargs):
        pass

    def prepare_population(self, _expressions):
        pass

    def objective_vector(self, expression):
        if str(expression) == str(MeanAmount(1)):
            raise RuntimeError("search crashed")
        return [0.1, 0.2, 0.3, 0.01]


def test_interrupted_lifecycle_csv_remains_readable(tmp_path, monkeypatch):
    csv_path = tmp_path / "diagnostics.csv"

    class StubMaterializer:
        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(shared_search_module, "FeatureMaterializer", StubMaterializer)
    monkeypatch.setattr(shared_search_module, "ResidualEvaluator", _ExplodingFitness)
    original_builder = runner_module.build_enumerative_search

    def build_with_batch(*args, **kwargs):
        search = original_builder(*args, **kwargs)
        search.candidate_generator = _FiniteGenerator(
            [MeanAmount(0), MeanAmount(1)]
        )
        return search

    monkeypatch.setattr(runner_module, "build_enumerative_search", build_with_batch)

    with pytest.raises(RuntimeError, match="search crashed"):
        run_feature_search(
            "enumerative",
            time_budget_seconds=0.001,
            dataset_path=tmp_path / "dataset.parquet",
            mapping=LABEL_MAPPING,
            mmap_dir=tmp_path / "mmap",
            csv_path=csv_path,
        )

    rows = list(csv.DictReader(csv_path.open(newline="")))
    assert [row["Expression"] for row in rows] == [str(MeanAmount(0))]
    assert rows[0]["Status"] == "evaluated"
    assert rows[0]["ArchiveMember"] == ""


def _tracked_inputs(tmp_path):
    dataset = tmp_path / "dataset.parquet"
    dataset.write_bytes(b"tracked dataset")
    mmap_dir = tmp_path / "mmap"
    mmap_dir.mkdir()
    (mmap_dir / "manifest.json").write_text(
        json.dumps({"rows": 0, "columns": {}}), encoding="utf-8"
    )
    return dataset, mmap_dir


def _tracked_result(writer, *, generation=True):
    lifecycle = SearchLifecycleRecorder(strategy="enumerative_without_archive")
    if generation:
        lifecycle.generation_rows.append(
            {
                "Strategy": "enumerative_without_archive",
                "Generation": 0,
                "Generated": 2,
                "Unique": 2,
                "Duplicate": 0,
                "Invalid": 0,
                "Evaluated": 0,
                "ArchiveSize": 0,
                "Added": 0,
                "DurationSeconds": 0.1,
                "CumulativeRuntimeSeconds": 0.1,
            }
        )
    writer.lifecycle = lifecycle
    return SearchRunResult(
        strategy=SearchStrategy.ENUMERATIVE_WITHOUT_ARCHIVE,
        expressions=(),
        final_evaluation=SimpleNamespace(metrics={"roc_auc": 0.75}),
        search_duration_seconds=0.1,
        final_evaluation_duration_seconds=0.2,
        generated_count=2,
        evaluated_count=0,
        invalid_count=0,
        duplicate_count=0,
        objectives=None,
        grammar_exhausted=False,
        lifecycle=lifecycle,
    )


def test_public_runner_tracks_analyzes_and_returns_mlflow_id(
    tmp_path, monkeypatch, mlflow_store
):
    dataset, mmap_dir = _tracked_inputs(tmp_path)

    def fake_impl(*_args, **kwargs):
        return _tracked_result(kwargs["_bundle_writer"])

    def fake_report(run_dir, *, feature_labels):
        assert feature_labels == "id"
        report = Path(run_dir) / "report.html"
        report.write_text("tracked report", encoding="utf-8")
        return report

    monkeypatch.setattr(runner_module, "_run_feature_search_impl", fake_impl)
    monkeypatch.setattr(runner_module, "render_run_report", fake_report)

    result = run_tracked_feature_search(
        "enumerative_without_archive",
        candidate_count=2,
        dataset_path=dataset,
        mapping=LABEL_MAPPING,
        mmap_dir=mmap_dir,
        feature_labels="id",
        tracking_store=mlflow_store,
    )

    assert result.run_id
    run = mlflow_store.get_run(result.run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags["project_state"] == "complete"
    assert run.data.tags["strategy_group"] == "unfiltered_enumeration_benchmark"
    assert run.data.params["feature_labels"] == "id"
    assert run.data.metrics["generated"] == 2
    assert {item.path for item in mlflow_store.client.list_artifacts(result.run_id)} >= {
        "manifest.json",
        "report.html",
    }


def test_analysis_failure_preserves_completed_search_bundle(
    tmp_path, monkeypatch, mlflow_store
):
    dataset, mmap_dir = _tracked_inputs(tmp_path)
    monkeypatch.setattr(
        runner_module,
        "_run_feature_search_impl",
        lambda *_args, **kwargs: _tracked_result(kwargs["_bundle_writer"]),
    )
    monkeypatch.setattr(
        runner_module,
        "render_run_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("plots failed")),
    )

    with pytest.raises(SearchAnalysisError, match="plots failed") as raised:
        run_tracked_feature_search(
            "enumerative_without_archive",
            candidate_count=2,
            dataset_path=dataset,
            mapping=LABEL_MAPPING,
            mmap_dir=mmap_dir,
            tracking_store=mlflow_store,
        )

    run_id = raised.value.run_id
    run = mlflow_store.get_run(run_id)
    assert run.info.status == "FAILED"
    assert run.data.tags["project_state"] == "analysis_failed"
    bundle = mlflow_store.download_artifact_bundle(run_id, tmp_path / "failed-analysis")
    assert bundle.state == "search_complete"
    assert not (bundle.path / "report.html").exists()


@pytest.mark.parametrize(
    ("error", "state", "mlflow_status"),
    [
        (RuntimeError("search exploded"), "search_failed", "FAILED"),
        (KeyboardInterrupt(), "interrupted", "KILLED"),
    ],
)
def test_search_failure_and_interruption_retain_partial_diagnostics(
    tmp_path, monkeypatch, mlflow_store, error, state, mlflow_status
):
    dataset, mmap_dir = _tracked_inputs(tmp_path)

    def fail(*_args, **kwargs):
        kwargs["_bundle_writer"].lifecycle = SearchLifecycleRecorder(
            strategy="enumerative_without_archive"
        )
        raise error

    monkeypatch.setattr(runner_module, "_run_feature_search_impl", fail)
    report_called = False

    def report_must_not_run(*_args, **_kwargs):
        nonlocal report_called
        report_called = True

    monkeypatch.setattr(runner_module, "render_run_report", report_must_not_run)

    with pytest.raises(type(error)):
        run_tracked_feature_search(
            "enumerative_without_archive",
            candidate_count=2,
            dataset_path=dataset,
            mapping=LABEL_MAPPING,
            mmap_dir=mmap_dir,
            tracking_store=mlflow_store,
        )

    run = mlflow_store.search_runs()[0]
    assert run.info.status == mlflow_status
    assert run.data.tags["project_state"] == state
    bundle = mlflow_store.download_artifact_bundle(
        run.info.run_id, tmp_path / f"partial-{state}"
    )
    assert bundle.state == state
    assert not (bundle.path / "report.html").exists()
    assert not report_called


def test_tracking_preflight_happens_before_input_or_search_construction(
    tmp_path, monkeypatch
):
    built = False

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("tracking unavailable")

    def writer_must_not_run(*_args, **_kwargs):
        nonlocal built
        built = True

    monkeypatch.setattr(runner_module, "MlflowRunStore", unavailable)

    with pytest.raises(RuntimeError, match="tracking unavailable"):
        run_tracked_feature_search(
            "enumerative_without_archive",
            candidate_count=1,
            dataset_path=tmp_path / "missing.parquet",
        )
    assert not built
