import csv
import json

import numpy as np
import pytest
from geneticengine.evaluation.budget import EvaluationBudget

import automatedfe.search.search as shared_module
from automatedfe.features.grammar import expr
from automatedfe.search import (
    ArchiveStep,
    MaterializingGeneticProgramming,
    build_search_algorithm,
    canonical_expression_key,
    load_archive,
)
from automatedfe.search.archive import SNAPSHOT_MAPPING_REFERENCE, load_snapshot
from geneticengine.algorithms.gp.operators.combinators import ParallelStep, SequenceStep
from geneticengine.algorithms.gp.operators.selection import LexicaseSelection
from geneticengine.problems import MultiObjectiveProblem

LABEL_MAPPING = {
    "status": {"approved": 0, "complete": 1},
    "capture_method": {"contactless": 0},
    "payment_method": {"credit": 0},
    "card_brand": {"visa": 0},
    "document_type": {"cpf": 0},
}


@pytest.fixture
def archive_dataset(tmp_path, monkeypatch):
    class RecordingObjectiveEvaluator:
        def __init__(self, materializer, dataset_path, **kwargs):
            self.materializer = materializer
            self.dataset_path = dataset_path
            self.kwargs = kwargs

        def prepare_population(self, individuals):
            pass

        def objective_vector(self, individual):
            return [0.0, 0.0, 0.0, 0.0]

    monkeypatch.setattr(shared_module, "ResidualEvaluator", RecordingObjectiveEvaluator)
    return tmp_path / "dataset.parquet"


def write_mmap_fixture(path):
    """Write the minimal transaction mmap layout needed by the GP builder."""

    path.mkdir()
    arrays = {
        "merchant_id": np.array([1, 1, 1, 2], dtype=np.int64),
        "amount": np.array([10.0, 20.0, 30.0, 100.0]),
        "created_at": np.array([1, 2, 3, 1], dtype=np.int64),
        "status": np.array([0, 1, 0, 0], dtype=np.int64),
        "capture_method": np.array([0, 0, 0, 0], dtype=np.int64),
        "payment_method": np.array([0, 0, 0, 0], dtype=np.int64),
        "card_brand": np.array([0, 0, 0, 0], dtype=np.int64),
        "document_type": np.array([0, 0, 0, 0], dtype=np.int64),
    }
    columns = {}
    for name, values in arrays.items():
        filename = f"{name}.mmap"
        mapped = np.memmap(path / filename, dtype=values.dtype, mode="w+", shape=values.shape)
        mapped[:] = values
        mapped.flush()
        columns[name] = {"file": filename, "dtype": values.dtype.name}

    (path / "manifest.json").write_text(
        json.dumps({"rows": len(next(iter(arrays.values()))), "columns": columns})
    )
    return path


def test_search_uses_the_complete_grammar_and_default_depth(tmp_path, archive_dataset):
    budget = EvaluationBudget(10)
    algorithm = build_search_algorithm(
        budget,
        mapping=LABEL_MAPPING,
        population_size=10,
        seed=123,
        dataset_path=archive_dataset,
        mmap_dir=write_mmap_fixture(tmp_path / "mmap"),
    )

    assert algorithm.budget is budget
    assert algorithm.representation.decider.max_depth == 4

    algorithm.search()

    assert algorithm.tracker.get_number_evaluations() == 10


def test_search_defaults_to_population_size_50(tmp_path, archive_dataset):
    algorithm = build_search_algorithm(
        EvaluationBudget(1),
        mapping=LABEL_MAPPING,
        dataset_path=archive_dataset,
        mmap_dir=write_mmap_fixture(tmp_path / "mmap"),
    )

    assert algorithm.population_size == 50


def test_mmap_dir_is_passed_to_feature_materializer(tmp_path, monkeypatch, archive_dataset):
    captured = {}

    class RecordingMaterializer:
        def __init__(self, columns, *, output_dir=None, features_dir=None):
            captured["columns"] = columns
            captured["output_dir"] = output_dir
            captured["features_dir"] = features_dir

        def materialize_population(self, individuals):
            pass

    monkeypatch.setattr(shared_module, "FeatureMaterializer", RecordingMaterializer)
    mmap_dir = tmp_path / "mmap"
    feature_cache_dir = tmp_path / "features"

    algorithm = build_search_algorithm(
        EvaluationBudget(1),
        mapping=LABEL_MAPPING,
        mmap_dir=mmap_dir,
        feature_cache_dir=feature_cache_dir,
        dataset_path=archive_dataset,
    )

    assert captured == {
        "columns": mmap_dir,
        "output_dir": None,
        "features_dir": feature_cache_dir.resolve(),
    }
    assert isinstance(algorithm.materializer, RecordingMaterializer)


def test_tracker_records_every_evaluation_to_csv(tmp_path, archive_dataset):
    csv_path = tmp_path / "gp_search.csv"
    algorithm = build_search_algorithm(
        EvaluationBudget(10),
        mapping=LABEL_MAPPING,
        population_size=10,
        seed=123,
        csv_path=csv_path,
        dataset_path=archive_dataset,
        mmap_dir=write_mmap_fixture(tmp_path / "mmap"),
    )

    algorithm.search()

    with csv_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == algorithm.tracker.get_number_evaluations()
    assert list(rows[0]) == [
        "Generation",
        "Expression",
        "Dependencies",
        "Fitness",
        "Split1",
        "Split2",
        "Split3",
        "MaterializationTime",
    ]
    assert all(row["Expression"] and row["Dependencies"] for row in rows)


def test_multiobjective_csv_records_all_four_objectives(tmp_path, monkeypatch):
    class RecordingObjectiveEvaluator:
        def __init__(self, materializer, dataset_path, **kwargs):
            pass

        def prepare_population(self, individuals):
            pass

        def objective_vector(self, individual):
            return [0.1, 0.2, 0.3, 0.4]

    monkeypatch.setattr(shared_module, "ResidualEvaluator", RecordingObjectiveEvaluator)
    csv_path = tmp_path / "gp_search.csv"
    algorithm = build_search_algorithm(
        EvaluationBudget(4),
        mapping=LABEL_MAPPING,
        population_size=4,
        seed=123,
        csv_path=csv_path,
        dataset_path=tmp_path / "dataset.parquet",
        mmap_dir=write_mmap_fixture(tmp_path / "mmap"),
    )

    algorithm.search()

    rows = list(csv.DictReader(csv_path.open(newline="")))
    assert len(rows) == algorithm.tracker.get_number_evaluations()
    assert all(row["Split1"] == "0.1" for row in rows)
    assert all(row["Split2"] == "0.2" for row in rows)
    assert all(row["Split3"] == "0.3" for row in rows)
    assert all(row["MaterializationTime"] == "0.4" for row in rows)


def test_complete_grammar_search_materializes_a_population(tmp_path, archive_dataset):
    algorithm = build_search_algorithm(
        EvaluationBudget(4),
        mapping=LABEL_MAPPING,
        population_size=4,
        seed=123,
        csv_path=tmp_path / "complete_gp_search.csv",
        dataset_path=archive_dataset,
        mmap_dir=write_mmap_fixture(tmp_path / "mmap"),
    )

    assert algorithm.representation.decider.max_depth == 4
    algorithm.search()

    assert algorithm.tracker.get_number_evaluations() == 4
    assert len(algorithm.last_individuals) == algorithm.population_size


def test_search_tracks_the_final_generated_population(tmp_path, archive_dataset):
    algorithm = build_search_algorithm(
        EvaluationBudget(10),
        mapping=LABEL_MAPPING,
        population_size=10,
        seed=123,
        dataset_path=archive_dataset,
        mmap_dir=write_mmap_fixture(tmp_path / "mmap"),
    )

    algorithm.search()

    assert len(algorithm.last_individuals) == algorithm.population_size


def test_search_requires_a_positive_population_size(tmp_path):
    with pytest.raises(ValueError, match="population_size must be positive"):
        build_search_algorithm(
            EvaluationBudget(1),
            mapping=LABEL_MAPPING,
            population_size=0,
            mmap_dir=tmp_path / "mmap",
        )


def test_search_requires_a_dataset_for_archive_mode(tmp_path):
    with pytest.raises(ValueError, match="dataset_path is required"):
        build_search_algorithm(
            EvaluationBudget(1),
            mapping=LABEL_MAPPING,
            mmap_dir=tmp_path / "mmap",
        )


def test_search_materializes_the_complete_initial_population(tmp_path, archive_dataset):
    algorithm = build_search_algorithm(
        EvaluationBudget(10),
        mapping=LABEL_MAPPING,
        population_size=10,
        seed=123,
        csv_path=tmp_path / "gp_search.csv",
        dataset_path=archive_dataset,
        mmap_dir=write_mmap_fixture(tmp_path / "mmap"),
    )

    algorithm.search()

    assert isinstance(algorithm, MaterializingGeneticProgramming)
    assert algorithm.tracker.get_number_evaluations() == 10
    rows = list(csv.DictReader((tmp_path / "gp_search.csv").open(newline="")))
    assert {row["Generation"] for row in rows} == {"0"}


def test_search_uses_four_objective_fitness(tmp_path, archive_dataset):
    algorithm = build_search_algorithm(
        EvaluationBudget(4),
        mapping=LABEL_MAPPING,
        population_size=4,
        dataset_path=archive_dataset,
        mmap_dir=write_mmap_fixture(tmp_path / "mmap"),
    )
    best = algorithm.search()

    assert algorithm.tracker.get_number_evaluations() == 4
    assert isinstance(best[0].get_phenotype(), expr)
    assert isinstance(algorithm.problem, MultiObjectiveProblem)
    assert best[0].get_fitness(algorithm.problem).fitness_components == [0.0] * 4


def test_brier_improvement_selects_residual_evaluator(tmp_path, monkeypatch):
    captured = {}

    class RecordingResidualEvaluator:
        def __init__(self, materializer, dataset_path, **kwargs):
            captured["materializer"] = materializer
            captured["dataset_path"] = dataset_path
            captured["kwargs"] = kwargs

        def prepare_population(self, individuals):
            pass

        def __call__(self, individual):
            return 0.0

        def objective_vector(self, individual):
            return [0.0, 0.0, 0.0, 0.0]

    monkeypatch.setattr(shared_module, "ResidualEvaluator", RecordingResidualEvaluator)
    mmap_dir = write_mmap_fixture(tmp_path / "mmap")
    dataset_path = tmp_path / "dataset.parquet"

    algorithm = build_search_algorithm(
        EvaluationBudget(1),
        mapping=LABEL_MAPPING,
        mmap_dir=mmap_dir,
        dataset_path=dataset_path,
        score_metric="brier_improvement",
    )

    assert algorithm.fitness_evaluator.__class__ is RecordingResidualEvaluator
    assert captured["dataset_path"] == dataset_path
    assert captured["kwargs"] == {
        "n_splits": 3,
        "score_metric": "brier_improvement",
    }


def test_dataset_search_uses_one_archive_step_and_returns_initial_archive(
    tmp_path,
    monkeypatch,
):
    class RecordingObjectiveEvaluator:
        def __init__(self, materializer, dataset_path, **kwargs):
            self.materializer = materializer
            self.dataset_path = dataset_path
            self.kwargs = kwargs

        def prepare_population(self, individuals):
            pass

        def objective_vector(self, individual):
            return [0.5, 0.5, 0.5, 0.01]

    monkeypatch.setattr(shared_module, "ResidualEvaluator", RecordingObjectiveEvaluator)
    algorithm = build_search_algorithm(
        EvaluationBudget(4),
        mapping=LABEL_MAPPING,
        population_size=4,
        seed=123,
        mmap_dir=write_mmap_fixture(tmp_path / "mmap"),
        dataset_path=tmp_path / "dataset.parquet",
    )

    assert isinstance(algorithm.problem, MultiObjectiveProblem)
    assert algorithm.problem.minimize == [False, False, False, True]
    assert isinstance(algorithm.archive, ArchiveStep)
    assert isinstance(algorithm.step, SequenceStep)
    assert algorithm.step.steps[0] is algorithm.archive
    generation_pipeline = algorithm.step.steps[1]
    assert isinstance(generation_pipeline, ParallelStep)
    reproduction = generation_pipeline.steps[2]
    assert isinstance(reproduction, SequenceStep)
    assert isinstance(reproduction.steps[0], LexicaseSelection)
    assert reproduction.steps[0].epsilon
    assert algorithm.tracker.memory is None

    result = algorithm.search()

    assert result == algorithm.archive.archive
    assert len(result) == algorithm.population_size


def test_archive_path_persists_the_final_front(tmp_path, archive_dataset):
    archive_path = tmp_path / "archive" / "front.json"
    algorithm = build_search_algorithm(
        EvaluationBudget(4),
        mapping=LABEL_MAPPING,
        population_size=4,
        seed=123,
        archive_path=archive_path,
        dataset_path=archive_dataset,
        mmap_dir=write_mmap_fixture(tmp_path / "mmap"),
    )

    assert algorithm.archive.archive_path == archive_path.resolve()
    result = algorithm.search()

    snapshot = load_archive(archive_path)
    assert [str(expression) for expression in snapshot.expressions] == [
        str(individual.get_phenotype()) for individual in result
    ]
    assert snapshot.mapping == LABEL_MAPPING


def test_archive_path_rejects_an_existing_directory(tmp_path, archive_dataset):
    archive_directory = tmp_path / "archive"
    archive_directory.mkdir()

    with pytest.raises(ValueError, match="archive_path must identify a file"):
        build_search_algorithm(
            EvaluationBudget(1),
            mapping=LABEL_MAPPING,
            archive_path=archive_directory,
            dataset_path=archive_dataset,
            mmap_dir=write_mmap_fixture(tmp_path / "mmap"),
        )


def test_archive_loading_types_are_importable_from_public_packages():
    from automatedfe import ArchiveSnapshot as RootArchiveSnapshot
    from automatedfe import ArchiveStep as RootArchiveStep
    from automatedfe import load_archive as root_load_archive
    from automatedfe.evaluation import ArchiveSource
    from automatedfe.features import ArchiveSnapshot, ArchiveStep

    assert RootArchiveSnapshot is ArchiveSnapshot
    assert RootArchiveStep is ArchiveStep
    assert root_load_archive is load_archive
    assert ArchiveSource


def test_gp_search_emits_generation_histories_and_snapshots(tmp_path, archive_dataset):
    algorithm = build_search_algorithm(
        EvaluationBudget(20),
        mapping=LABEL_MAPPING,
        population_size=10,
        seed=123,
        dataset_path=archive_dataset,
        mmap_dir=write_mmap_fixture(tmp_path / "mmap"),
    )

    algorithm.search()

    lifecycle = algorithm.lifecycle
    assert lifecycle.generation_rows
    assert sorted(lifecycle.snapshots) == [
        row["Generation"] for row in lifecycle.generation_rows
    ]
    assert (
        sum(row["Evaluated"] for row in lifecycle.generation_rows)
        == algorithm.tracker.get_number_evaluations()
    )
    cumulative_runtime = [
        row["CumulativeRuntimeSeconds"] for row in lifecycle.generation_rows
    ]
    assert cumulative_runtime == sorted(cumulative_runtime)
    for generation, document in lifecycle.snapshot_documents:
        assert "mapping" not in document
        assert document["mapping_ref"] == SNAPSHOT_MAPPING_REFERENCE
        assert document["problem"]["number_of_objectives"] == 4
    assert lifecycle.archived_keys == {
        canonical_expression_key(individual.get_phenotype())
        for individual in algorithm.archive_step.archive
    }

    snapshot_path = tmp_path / "snapshots" / "generation_000000.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(lifecycle.snapshots[0]))
    snapshot = load_snapshot(snapshot_path, LABEL_MAPPING)
    assert len(snapshot) == len(lifecycle.snapshots[0]["expressions"])


def test_gp_active_search_records_snapshots_without_changing_membership(
    tmp_path,
    monkeypatch,
):
    class RecordingActiveObjectiveEvaluator:
        def __init__(self, materializer, dataset_path, **kwargs):
            pass

        def prepare_population(self, individuals):
            pass

        def objective_vector(self, individual):
            return [0.0, 0.0, 0.0, 0.0]

    monkeypatch.setattr(
        shared_module,
        "ResidualEvaluator",
        RecordingActiveObjectiveEvaluator,
    )
    monkeypatch.setattr(
        shared_module,
        "ActiveResidualEvaluator",
        RecordingActiveObjectiveEvaluator,
    )

    def build(mmap_dir):
        return build_search_algorithm(
            EvaluationBudget(20),
            mapping=LABEL_MAPPING,
            population_size=10,
            seed=123,
            use_active_set=True,
            dataset_path=tmp_path / "dataset.parquet",
            mmap_dir=write_mmap_fixture(mmap_dir),
        )

    first = build(tmp_path / "mmap_a")
    first.search()
    second = build(tmp_path / "mmap_b")
    second.search()

    assert first.lifecycle.generation_rows
    assert second.lifecycle.generation_rows
    assert len(first.lifecycle.snapshots) == len(first.lifecycle.generation_rows)
    assert first.lifecycle.archived_keys == second.lifecycle.archived_keys
    assert first.lifecycle.archived_keys == {
        canonical_expression_key(individual.get_phenotype())
        for individual in first.archive_step.archive
    }
    assert [
        row["Evaluated"] for row in first.lifecycle.generation_rows
    ] == [row["Evaluated"] for row in second.lifecycle.generation_rows]


def test_same_seed_produces_same_initial_population_and_results(tmp_path, archive_dataset):
    mmap_dir = write_mmap_fixture(tmp_path / "mmap")

    def build(seed, *, csv_path=None):
        return build_search_algorithm(
            EvaluationBudget(10),
            mapping=LABEL_MAPPING,
            population_size=10,
            seed=seed,
            csv_path=csv_path,
            dataset_path=archive_dataset,
            mmap_dir=mmap_dir,
        )

    first_initial = build(123)._generate_initial_individuals()
    second_initial = build(123)._generate_initial_individuals()
    assert [str(ind.get_phenotype()) for ind in first_initial] == [
        str(ind.get_phenotype()) for ind in second_initial
    ]

    first_csv = tmp_path / "first.csv"
    second_csv = tmp_path / "second.csv"
    first = build(123, csv_path=first_csv)
    second = build(123, csv_path=second_csv)
    first_best = first.search()
    second_best = second.search()

    assert [str(ind.get_phenotype()) for ind in first_best] == [
        str(ind.get_phenotype()) for ind in second_best
    ]
    assert first_csv.read_text() == second_csv.read_text()
