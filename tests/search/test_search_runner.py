import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import automatedfe.search.runner as runner_module
import automatedfe.search.search as shared_search_module
from automatedfe.analysis.artifacts import (
    CANDIDATES_COLUMNS,
    load_run_manifest,
)
from automatedfe.search.archive import load_archive
from automatedfe.search.runner import (
    DIAGNOSTIC_COLUMNS,
    SearchStrategy,
    run_feature_search,
)


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

    def evaluate(self, expressions):
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


def test_diagnostics_columns_follow_the_analysis_candidates_schema():
    assert DIAGNOSTIC_COLUMNS == CANDIDATES_COLUMNS


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
    assert tuple(rows[0]) == DIAGNOSTIC_COLUMNS
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
    assert {row["Generation"] for row in rows} == {""}
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
    archive_expression = "archive-expression"
    active_expression = "active-expression"
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
        archive = ["archive-expression"]
        active_individuals = ["active-expression"]

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
