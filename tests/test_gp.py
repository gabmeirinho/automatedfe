import csv
import json

import numpy as np
import pytest
from geneticengine.evaluation.budget import EvaluationBudget

import automatedfe.features.gp as gp_module
from automatedfe.features import (
    MaterializingGeneticProgramming,
    build_search_algorithm,
    expr,
)

LABEL_MAPPING = {
    "status": {"approved": 0, "complete": 1},
    "capture_method": {"contactless": 0},
    "payment_method": {"credit": 0},
    "card_brand": {"visa": 0},
    "document_type": {"cpf": 0},
}


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


def test_search_uses_the_complete_grammar_and_default_depth(tmp_path):
    budget = EvaluationBudget(10)
    algorithm = build_search_algorithm(
        budget,
        mapping=LABEL_MAPPING,
        population_size=10,
        seed=123,
        mmap_dir=write_mmap_fixture(tmp_path / "mmap"),
    )

    assert algorithm.budget is budget
    assert algorithm.representation.decider.max_depth == 4

    algorithm.search()

    assert algorithm.tracker.get_number_evaluations() == 10


def test_mmap_dir_is_passed_to_feature_materializer(tmp_path, monkeypatch):
    captured = {}

    class RecordingMaterializer:
        def __init__(self, columns, *, output_dir=None, features_dir=None):
            captured["columns"] = columns
            captured["output_dir"] = output_dir
            captured["features_dir"] = features_dir

        def materialize_population(self, individuals):
            pass

    monkeypatch.setattr(gp_module, "FeatureMaterializer", RecordingMaterializer)
    mmap_dir = tmp_path / "mmap"
    feature_cache_dir = tmp_path / "features"

    algorithm = build_search_algorithm(
        EvaluationBudget(1),
        mapping=LABEL_MAPPING,
        mmap_dir=mmap_dir,
        feature_cache_dir=feature_cache_dir,
    )

    assert captured == {
        "columns": mmap_dir,
        "output_dir": None,
        "features_dir": feature_cache_dir.resolve(),
    }
    assert isinstance(algorithm.materializer, RecordingMaterializer)


def test_tracker_records_every_evaluation_to_csv(tmp_path):
    csv_path = tmp_path / "gp_search.csv"
    algorithm = build_search_algorithm(
        EvaluationBudget(10),
        mapping=LABEL_MAPPING,
        population_size=10,
        seed=123,
        csv_path=csv_path,
        mmap_dir=write_mmap_fixture(tmp_path / "mmap"),
    )

    algorithm.search()

    with csv_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == algorithm.tracker.get_number_evaluations()
    assert list(rows[0]) == ["Generation", "Expression", "Dependencies", "Fitness"]
    assert all(row["Expression"] and row["Dependencies"] for row in rows)


def test_complete_grammar_search_materializes_a_population(tmp_path):
    algorithm = build_search_algorithm(
        EvaluationBudget(4),
        mapping=LABEL_MAPPING,
        population_size=4,
        seed=123,
        csv_path=tmp_path / "complete_gp_search.csv",
        mmap_dir=write_mmap_fixture(tmp_path / "mmap"),
    )

    assert algorithm.representation.decider.max_depth == 4
    algorithm.search()

    assert algorithm.tracker.get_number_evaluations() == 4
    assert len(algorithm.last_individuals) == algorithm.population_size


def test_search_tracks_the_final_generated_population(tmp_path):
    algorithm = build_search_algorithm(
        EvaluationBudget(10),
        mapping=LABEL_MAPPING,
        population_size=10,
        seed=123,
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


def test_search_materializes_the_complete_initial_population(tmp_path):
    algorithm = build_search_algorithm(
        EvaluationBudget(10),
        mapping=LABEL_MAPPING,
        population_size=10,
        seed=123,
        csv_path=tmp_path / "gp_search.csv",
        mmap_dir=write_mmap_fixture(tmp_path / "mmap"),
    )

    algorithm.search()

    assert isinstance(algorithm, MaterializingGeneticProgramming)
    assert algorithm.tracker.get_number_evaluations() == 10
    rows = list(csv.DictReader((tmp_path / "gp_search.csv").open(newline="")))
    assert {row["Generation"] for row in rows} == {"0"}


def test_search_defaults_every_fitness_to_zero(tmp_path):
    algorithm = build_search_algorithm(
        EvaluationBudget(4),
        mapping=LABEL_MAPPING,
        population_size=4,
        mmap_dir=write_mmap_fixture(tmp_path / "mmap"),
    )
    best = algorithm.search()

    assert algorithm.tracker.get_number_evaluations() == 4
    assert isinstance(best[0].get_phenotype(), expr)
    assert best[0].get_fitness(algorithm.problem).fitness_components == [0.0]


def test_same_seed_produces_same_initial_population_and_results(tmp_path):
    mmap_dir = write_mmap_fixture(tmp_path / "mmap")

    def build(seed, *, csv_path=None):
        return build_search_algorithm(
            EvaluationBudget(10),
            mapping=LABEL_MAPPING,
            population_size=10,
            seed=seed,
            csv_path=csv_path,
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
