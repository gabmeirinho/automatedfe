"""Import and compatibility checks for the phase-one package boundaries."""

import automatedfe
import automatedfe.cli
import automatedfe.data
import automatedfe.evaluation
import automatedfe.search


def test_domain_namespaces_import_without_moving_implementations():
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
    assert automatedfe.evaluation.FinalEvaluator is automatedfe.features.FinalEvaluator

    assert automatedfe.search.SearchStrategy is automatedfe.SearchStrategy
    assert automatedfe.search.run_feature_search is automatedfe.run_feature_search
    assert automatedfe.search.ArchiveSnapshot is automatedfe.ArchiveSnapshot


def test_cli_namespace_has_no_import_side_effect_exports_yet():
    assert automatedfe.cli.__all__ == []
