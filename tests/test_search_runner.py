import csv
from types import SimpleNamespace

import pytest

import automatedfe.features.runner as runner_module
import automatedfe.features.search.search as shared_search_module
from automatedfe.features import (
    DIAGNOSTIC_COLUMNS,
    SearchStrategy,
    load_archive,
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
