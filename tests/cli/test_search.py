from types import SimpleNamespace

import pytest

import automatedfe.cli.search as search_cli
from automatedfe.features.grammar import MeanAmount
from automatedfe.search.runner import SearchAnalysisError, SearchStrategy


def fake_result(strategy=SearchStrategy.ENUMERATIVE_WITHOUT_ARCHIVE):
    return SimpleNamespace(
        run_id="mlflow-run-123",
        strategy=strategy,
        expressions=(MeanAmount(0), MeanAmount(1)),
        generated_count=2,
        evaluated_count=0,
        invalid_count=0,
        duplicate_count=0,
        search_duration_seconds=0.25,
        final_evaluation_duration_seconds=0.5,
        grammar_exhausted=False,
        objectives=None,
        final_metrics={"roc_auc": 0.8},
    )


def test_cli_dispatches_tracked_search_and_prints_run_id(monkeypatch, capsys):
    calls = []

    def fake_run(strategy, **kwargs):
        calls.append((strategy, kwargs))
        return fake_result(strategy)

    monkeypatch.setattr(search_cli, "run_feature_search", fake_run)

    assert (
        search_cli.main(
            [
                "--strategy",
                "enumerative_without_archive",
                "--candidate-count",
                "2",
                "--feature-labels",
                "id",
                "--tracking-uri",
                "sqlite:///isolated.db",
            ]
        )
        == 0
    )

    assert calls[0][0] is SearchStrategy.ENUMERATIVE_WITHOUT_ARCHIVE
    assert calls[0][1]["candidate_count"] == 2
    assert calls[0][1]["time_budget_seconds"] is None
    assert calls[0][1]["feature_labels"] == "id"
    assert calls[0][1]["tracking_uri"] == "sqlite:///isolated.db"
    stdout = capsys.readouterr().out
    assert "Run ID: mlflow-run-123" in stdout
    assert "Final metrics:" in stdout


def test_cli_defaults_to_residual_brier_and_expression_labels(monkeypatch):
    calls = []

    def fake_run(strategy, **kwargs):
        calls.append((strategy, kwargs))
        return fake_result(strategy)

    monkeypatch.setattr(search_cli, "run_feature_search", fake_run)
    assert search_cli.main(["--strategy", "enumerative", "--time-budget", "1"]) == 0
    assert calls[0][1]["score_metric"] == "brier_improvement"
    assert calls[0][1]["feature_labels"] == "expression"


def test_cli_forwards_active_set_configuration(monkeypatch):
    calls = []

    def fake_run(strategy, **kwargs):
        calls.append((strategy, kwargs))
        return fake_result(strategy)

    monkeypatch.setattr(search_cli, "run_feature_search", fake_run)
    assert (
        search_cli.main(
            [
                "--strategy",
                "genetic",
                "--time-budget",
                "1",
                "--use-active-set",
                "--promotion-interval",
                "7",
                "--promotion-add-k",
                "2",
            ]
        )
        == 0
    )
    assert calls[0][1]["use_active_set"] is True
    assert calls[0][1]["promotion_interval"] == 7
    assert calls[0][1]["promotion_add_k"] == 2


@pytest.mark.parametrize(
    "legacy_flag",
    ["--csv", "--archive", "--history", "--active-archive", "--summary", "--force"],
)
def test_cli_rejects_legacy_loose_output_flags(monkeypatch, legacy_flag):
    called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(search_cli, "run_feature_search", fail_if_called)
    with pytest.raises(SystemExit) as raised:
        search_cli.main(
            [
                "--strategy",
                "enumerative_without_archive",
                "--candidate-count",
                "1",
                legacy_flag,
            ]
        )
    assert raised.value.code == 2
    assert not called


def test_cli_validates_strategy_budget_before_dispatch(monkeypatch):
    called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(search_cli, "run_feature_search", fail_if_called)
    with pytest.raises(SystemExit) as raised:
        search_cli.main(["--strategy", "random", "--candidate-count", "1"])
    assert raised.value.code == 2
    assert not called


def test_cli_returns_nonzero_for_analysis_failure(monkeypatch, capsys):
    def fail(*_args, **_kwargs):
        raise SearchAnalysisError("run-analysis", ValueError("plots broke"))

    monkeypatch.setattr(search_cli, "run_feature_search", fail)
    assert (
        search_cli.main(
            [
                "--strategy",
                "enumerative_without_archive",
                "--candidate-count",
                "1",
            ]
        )
        == 1
    )
    assert "run-analysis" in capsys.readouterr().err


def test_cli_returns_nonzero_for_search_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        search_cli,
        "run_feature_search",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert (
        search_cli.main(
            [
                "--strategy",
                "enumerative_without_archive",
                "--candidate-count",
                "1",
            ]
        )
        == 1
    )
    assert "Feature search failed: boom" in capsys.readouterr().err


def test_cli_returns_130_for_keyboard_interrupt(monkeypatch, capsys):
    monkeypatch.setattr(
        search_cli,
        "run_feature_search",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    assert (
        search_cli.main(
            [
                "--strategy",
                "enumerative_without_archive",
                "--candidate-count",
                "1",
            ]
        )
        == 130
    )
    assert "interrupted" in capsys.readouterr().err


def test_cli_help_lists_all_strategies_and_tracked_options():
    help_text = search_cli.build_parser().format_help()
    assert "enumerative_without_archive" in help_text
    assert "--feature-labels" in help_text
    assert "--tracking-uri" in help_text
    assert "--summary" not in help_text
