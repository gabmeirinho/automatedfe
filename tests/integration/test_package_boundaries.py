"""Import checks for the domain package boundaries."""

from pathlib import Path

import automatedfe
import automatedfe.cli
import automatedfe.data
import automatedfe.evaluation
import automatedfe.search
import automatedfe.search.runner as search_runner


def test_domain_namespaces_import_and_preserve_public_exports():
    assert automatedfe.data.encode_transactions.__module__ == "automatedfe.data.encoding"
    assert automatedfe.data.preprocess.__module__ == "automatedfe.data.preprocessing"
    assert automatedfe.data.sort_transactions.__module__ == "automatedfe.data.sorting"
    assert automatedfe.data.materialize_transactions.__module__ == (
        "automatedfe.data.transaction_materialization"
    )

    assert automatedfe.evaluation.RandomForestFitness.__module__ == (
        "automatedfe.evaluation.fitness"
    )
    assert automatedfe.evaluation.FinalEvaluator is (
        automatedfe.evaluation.final_evaluation.FinalEvaluator
    )

    assert automatedfe.search.ArchiveSnapshot.__module__ == (
        "automatedfe.search.archive"
    )
    assert not hasattr(automatedfe, "SearchStrategy")
    assert not hasattr(automatedfe, "ArchiveSnapshot")
    assert not hasattr(automatedfe, "write_summary_json")
    assert not hasattr(automatedfe.search, "write_summary_json")


def test_data_pipeline_implementations_have_canonical_module_paths():
    assert automatedfe.data.encode_transactions.__module__ == (
        "automatedfe.data.encoding"
    )
    assert automatedfe.data.preprocess.__module__ == "automatedfe.data.preprocessing"
    assert automatedfe.data.sort_dataset.__module__ == "automatedfe.data.sorting"
    assert automatedfe.data.materialize_transactions.__module__ == (
        "automatedfe.data.transaction_materialization"
    )
    assert automatedfe.data.first_sorting_violation.__module__ == (
        "automatedfe.data.validation"
    )
    assert automatedfe.data.PROJECT_ROOT == Path(__file__).resolve().parents[2]


def test_evaluation_implementations_have_canonical_module_paths():
    assert automatedfe.evaluation.RandomForestFitness.__module__ == (
        "automatedfe.evaluation.fitness"
    )
    assert automatedfe.evaluation.FinalEvaluator.__module__ == (
        "automatedfe.evaluation.final_evaluation"
    )


def test_search_implementations_have_canonical_module_paths():
    assert automatedfe.search.ArchiveSnapshot.__module__ == (
        "automatedfe.search.archive"
    )
    assert automatedfe.search.MaterializingArchiveSearch.__module__ == (
        "automatedfe.search.search"
    )
    assert automatedfe.search.build_random_search.__module__ == (
        "automatedfe.search.random_search"
    )
    assert search_runner.SearchStrategy.__module__ == "automatedfe.search.runner"
    assert not hasattr(automatedfe.search, "SearchStrategy")
    assert not hasattr(automatedfe.features, "ArchiveSnapshot")
    assert not hasattr(automatedfe.features, "SearchStrategy")


def test_cli_namespace_has_no_import_side_effect_exports_yet():
    assert automatedfe.cli.__all__ == []
