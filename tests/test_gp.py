import csv
import json
from typing import get_type_hints

import numpy as np
import pytest
from geneticengine.evaluation.budget import EvaluationBudget

import automatedfe.features.gp as gp_module
from automatedfe.features import (
    AMOUNT_COLUMN,
    WINDOW_CATALOG,
    AggregationFeature,
    Aggregation,
    Count,
    Feature,
    Max,
    MaterializingGeneticProgramming,
    Mean,
    Sum,
    WindowIndex,
    build_grammar,
    build_search_algorithm,
)


def write_mmap_fixture(path):
    """Write the minimal transaction mmap layout needed by the GP builder."""

    path.mkdir()
    arrays = {
        "merchant_id": np.array([1, 1, 1, 2], dtype=np.int64),
        "amount": np.array([10.0, 20.0, 30.0, 100.0]),
        "created_at": np.array([1, 2, 3, 1], dtype=np.int64),
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


def test_grammar_contains_all_kernel_aggregations():
    grammar = build_grammar()

    assert grammar.starting_symbol is AggregationFeature
    assert grammar.get_min_tree_depth() == 1
    assert grammar.alternatives[AggregationFeature] == [Count, Sum, Mean, Max]


def test_mean_has_feature_and_window_parameters():
    grammar = build_grammar()

    assert list(Mean.__annotations__) == ["feature", "window"]
    annotations = get_type_hints(Mean, include_extras=True)
    assert annotations["feature"] == Feature
    assert annotations["window"] == WindowIndex


@pytest.mark.parametrize("window_i", range(len(WINDOW_CATALOG)))
def test_mean_maps_its_feature_and_window_to_a_feature_spec(window_i):
    expression = Mean(AMOUNT_COLUMN, window_i)
    spec = expression.to_feature_spec()

    assert expression.feature == AMOUNT_COLUMN
    assert expression.selected_window == WINDOW_CATALOG[window_i]
    assert spec.aggregation is Aggregation.MEAN
    assert spec.input_column == AMOUNT_COLUMN
    assert spec.window == WINDOW_CATALOG[window_i]
    assert str(expression) == spec.name


@pytest.mark.parametrize(
    "expression, aggregation, input_column",
    [
        (Count(0), Aggregation.COUNT, None),
        (Sum(AMOUNT_COLUMN, 0), Aggregation.SUM, AMOUNT_COLUMN),
        (Mean(AMOUNT_COLUMN, 0), Aggregation.MEAN, AMOUNT_COLUMN),
        (Max(AMOUNT_COLUMN, 0), Aggregation.MAX, AMOUNT_COLUMN),
    ],
)
def test_kernel_aggregation_nodes_map_to_feature_specs(
    expression, aggregation, input_column
):
    spec = expression.to_feature_spec()

    assert spec.aggregation is aggregation
    assert spec.input_column == input_column
    assert spec.window == WINDOW_CATALOG[0]


def test_search_uses_the_given_budget_and_max_depth_one(tmp_path):
    grammar = build_grammar()
    budget = EvaluationBudget(10)
    algorithm = build_search_algorithm(
        grammar,
        budget,
        population_size=10,
        seed=123,
        mmap_dir=write_mmap_fixture(tmp_path / "mmap"),
    )

    assert algorithm.budget is budget
    assert algorithm.representation.decider.max_depth == 1

    algorithm.search()

    assert algorithm.tracker.get_number_evaluations() == 10


def test_mmap_dir_is_passed_to_feature_materializer(tmp_path, monkeypatch):
    captured = {}

    class RecordingMaterializer:
        def __init__(self, columns, *, output_dir=None):
            captured["columns"] = columns
            captured["output_dir"] = output_dir

        def materialize_population(self, individuals):
            pass

    monkeypatch.setattr(gp_module, "FeatureMaterializer", RecordingMaterializer)
    mmap_dir = tmp_path / "mmap"
    feature_output_dir = tmp_path / "features"

    algorithm = build_search_algorithm(
        build_grammar(),
        EvaluationBudget(1),
        mmap_dir=mmap_dir,
        feature_output_dir=feature_output_dir,
    )

    assert captured == {
        "columns": mmap_dir,
        "output_dir": feature_output_dir.resolve(),
    }
    assert isinstance(algorithm.materializer, RecordingMaterializer)


def test_tracker_records_every_evaluation_to_csv(tmp_path):
    csv_path = tmp_path / "gp_search.csv"
    algorithm = build_search_algorithm(
        build_grammar(),
        EvaluationBudget(10),
        population_size=10,
        seed=123,
        csv_path=csv_path,
        mmap_dir=write_mmap_fixture(tmp_path / "mmap"),
    )

    algorithm.search()

    with csv_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == algorithm.tracker.get_number_evaluations()
    assert list(rows[0]) == ["Generation", "Expression", "Feature", "Window", "Fitness"]
    assert {row["Feature"] for row in rows} <= {"", AMOUNT_COLUMN}
    assert all(
        row["Expression"].startswith(
            ("count_transactions_", "sum_amount_", "mean_amount_", "max_amount_")
        )
        for row in rows
    )


def test_search_requires_a_positive_population_size(tmp_path):
    with pytest.raises(ValueError, match="population_size must be positive"):
        build_search_algorithm(
            build_grammar(),
            EvaluationBudget(1),
            population_size=0,
            mmap_dir=tmp_path / "mmap",
        )


def test_search_materializes_the_complete_initial_population(tmp_path):
    feature_output_dir = tmp_path / "features"
    algorithm = build_search_algorithm(
        build_grammar(),
        EvaluationBudget(10),
        population_size=10,
        seed=123,
        csv_path=tmp_path / "gp_search.csv",
        mmap_dir=write_mmap_fixture(tmp_path / "mmap"),
        feature_output_dir=feature_output_dir,
    )

    algorithm.search()

    assert isinstance(algorithm, MaterializingGeneticProgramming)
    assert algorithm.tracker.get_number_evaluations() == 10
    rows = list(csv.DictReader((tmp_path / "gp_search.csv").open(newline="")))
    assert {row["Generation"] for row in rows} == {"0"}
    assert all(
        (feature_output_dir / f"{row['Expression']}.mmap").exists() for row in rows
    )


def test_feature_output_dir_writes_expected_mmaps(tmp_path):
    csv_path = tmp_path / "gp_search.csv"
    feature_output_dir = tmp_path / "features"
    algorithm = build_search_algorithm(
        build_grammar(),
        EvaluationBudget(10),
        population_size=10,
        seed=123,
        csv_path=csv_path,
        mmap_dir=write_mmap_fixture(tmp_path / "mmap"),
        feature_output_dir=feature_output_dir,
    )

    algorithm.search()

    with csv_path.open(newline="") as csv_file:
        expressions = {row["Expression"] for row in csv.DictReader(csv_file)}
    assert {path.stem for path in feature_output_dir.glob("*.mmap")} == expressions


def test_search_defaults_every_fitness_to_zero(tmp_path):
    algorithm = build_search_algorithm(
        build_grammar(),
        EvaluationBudget(4),
        population_size=4,
        mmap_dir=write_mmap_fixture(tmp_path / "mmap"),
    )
    best = algorithm.search()

    assert algorithm.tracker.get_number_evaluations() == 4
    assert isinstance(best[0].get_phenotype(), AggregationFeature)
    assert best[0].get_fitness(algorithm.problem).fitness_components == [0.0]


def test_same_seed_produces_same_initial_population_and_results(tmp_path):
    mmap_dir = write_mmap_fixture(tmp_path / "mmap")

    def build(seed, *, csv_path=None, feature_output_dir=None):
        return build_search_algorithm(
            build_grammar(),
            EvaluationBudget(10),
            population_size=10,
            seed=seed,
            csv_path=csv_path,
            mmap_dir=mmap_dir,
            feature_output_dir=feature_output_dir,
        )

    first_initial = build(123)._generate_initial_individuals()
    second_initial = build(123)._generate_initial_individuals()
    assert [str(ind.get_phenotype()) for ind in first_initial] == [
        str(ind.get_phenotype()) for ind in second_initial
    ]

    first_csv = tmp_path / "first.csv"
    second_csv = tmp_path / "second.csv"
    first = build(123, csv_path=first_csv, feature_output_dir=tmp_path / "first")
    second = build(123, csv_path=second_csv, feature_output_dir=tmp_path / "second")
    first_best = first.search()
    second_best = second.search()

    assert [str(ind.get_phenotype()) for ind in first_best] == [
        str(ind.get_phenotype()) for ind in second_best
    ]
    assert first_csv.read_text() == second_csv.read_text()
