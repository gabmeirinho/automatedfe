"""Import and compatibility checks for the domain package boundaries."""

from pathlib import Path

import automatedfe
import automatedfe.cli
import automatedfe.data
import automatedfe.evaluation
import automatedfe.search


def test_domain_namespaces_import_and_preserve_public_exports():
    assert automatedfe.data.encode_transactions is automatedfe.encode_transactions
    assert automatedfe.data.preprocess is automatedfe.preprocess
    assert automatedfe.data.sort_transactions is automatedfe.sort_transactions
    assert (
        automatedfe.data.materialize_transactions
        is automatedfe.materialize_transactions
    )

    assert (
        automatedfe.evaluation.FitnessEvaluator is automatedfe.FitnessEvaluator
    )
    assert automatedfe.evaluation.FinalEvaluator is (
        automatedfe.evaluation.final_evaluation.FinalEvaluator
    )

    assert automatedfe.search.SearchStrategy is automatedfe.SearchStrategy
    assert automatedfe.search.run_feature_search is automatedfe.run_feature_search
    assert automatedfe.search.ArchiveSnapshot is automatedfe.ArchiveSnapshot


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
    assert automatedfe.data.PROJECT_ROOT == Path(__file__).resolve().parents[1]


def test_evaluation_implementations_have_canonical_module_paths():
    assert automatedfe.evaluation.FitnessEvaluator.__module__ == (
        "automatedfe.evaluation.fitness"
    )
    assert automatedfe.evaluation.FinalEvaluator.__module__ == (
        "automatedfe.evaluation.final_evaluation"
    )


def test_cli_namespace_has_no_import_side_effect_exports_yet():
    assert automatedfe.cli.__all__ == []
