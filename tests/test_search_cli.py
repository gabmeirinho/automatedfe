import json
from types import SimpleNamespace

import pytest

import scripts.search as search_cli
from automatedfe.features import MeanAmount, SearchStrategy


def fake_result(strategy=SearchStrategy.ENUMERATIVE_WITHOUT_ARCHIVE):
    return SimpleNamespace(
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
        final_metrics={"accuracy": 0.75, "roc_auc": 0.8},
    )


def test_cli_dispatches_evaluation_free_strategy_and_writes_summary(
    tmp_path, monkeypatch, capsys
):
    calls = []

    def fake_run(strategy, **kwargs):
        calls.append((strategy, kwargs))
        return fake_result(strategy)

    monkeypatch.setattr(search_cli, "run_feature_search", fake_run)
    csv_path = tmp_path / "diagnostics.csv"
    summary_path = tmp_path / "summary.json"

    assert search_cli.main(
        [
            "--strategy",
            "enumerative_without_archive",
            "--candidate-count",
            "2",
            "--csv",
            str(csv_path),
            "--summary-json",
            str(summary_path),
            "--dataset",
            str(tmp_path / "dataset.parquet"),
            "--mapping",
            str(tmp_path / "mapping.json"),
        ]
    ) == 0

    assert calls[0][0] is SearchStrategy.ENUMERATIVE_WITHOUT_ARCHIVE
    assert calls[0][1]["candidate_count"] == 2
    assert calls[0][1]["time_budget_seconds"] is None
    summary = json.loads(summary_path.read_text())
    assert summary["strategy"] == "enumerative_without_archive"
    assert summary["counts"] == {
        "generated": 2,
        "evaluated": 0,
        "invalid": 0,
        "duplicates": 0,
    }
    assert summary["selected_feature_count"] == 2
    assert summary["final_metrics"] == {"accuracy": 0.75, "roc_auc": 0.8}
    assert "objectives" not in summary
    assert "predictions" not in summary
    assert "model" not in summary
    assert "MeanAmount" not in summary
    stdout = capsys.readouterr().out
    assert "Counts:" in stdout
    assert "Final metrics:" in stdout
    assert str(summary_path.resolve()) in stdout


def test_cli_defaults_to_residual_brier_improvement(tmp_path, monkeypatch):
    calls = []

    def fake_run(strategy, **kwargs):
        calls.append((strategy, kwargs))
        return fake_result(strategy)

    monkeypatch.setattr(search_cli, "run_feature_search", fake_run)

    assert search_cli.main(
        [
            "--strategy",
            "enumerative",
            "--time-budget",
            "1",
            "--dataset",
            str(tmp_path / "dataset.parquet"),
            "--mapping",
            str(tmp_path / "mapping.json"),
        ]
    ) == 0

    assert calls[0][1]["score_metric"] == "brier_improvement"


def test_cli_forwards_active_set_for_genetic_search(tmp_path, monkeypatch):
    calls = []

    def fake_run(strategy, **kwargs):
        calls.append((strategy, kwargs))
        return fake_result(strategy)

    monkeypatch.setattr(search_cli, "run_feature_search", fake_run)

    assert search_cli.main(
        [
            "--strategy",
            "genetic",
            "--time-budget",
            "1",
            "--use-active-set",
            "--dataset",
            str(tmp_path / "dataset.parquet"),
            "--mapping",
            str(tmp_path / "mapping.json"),
        ]
    ) == 0

    assert calls[0][1]["use_active_set"] is True


@pytest.mark.parametrize(
    "arguments",
    [
        ["--strategy", "enumerative"],
        ["--strategy", "enumerative", "--candidate-count", "2", "--time-budget", "1"],
        [
            "--strategy",
            "enumerative_without_archive",
            "--candidate-count",
            "2",
            "--time-budget",
            "1",
        ],
        [
            "--strategy",
            "enumerative_without_archive",
            "--candidate-count",
            "2",
            "--archive",
            "archive.json",
        ],
    ],
)
def test_cli_rejects_strategy_budget_mismatches(arguments, monkeypatch):
    called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("search should not be dispatched")

    monkeypatch.setattr(search_cli, "run_feature_search", fail_if_called)
    with pytest.raises(SystemExit) as error:
        search_cli.main(arguments)
    assert error.value.code == 2
    assert not called


def test_cli_preflights_summary_before_search(tmp_path, monkeypatch):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("keep")
    called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("search should not be dispatched")

    monkeypatch.setattr(search_cli, "run_feature_search", fail_if_called)
    with pytest.raises(SystemExit) as error:
        search_cli.main(
            [
                "--strategy",
                "enumerative_without_archive",
                "--candidate-count",
                "1",
                "--summary",
                str(summary_path),
            ]
        )
    assert error.value.code == 2
    assert not called
    assert summary_path.read_text() == "keep"


def test_cli_force_replaces_summary(tmp_path, monkeypatch):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("old")
    monkeypatch.setattr(
        search_cli,
        "run_feature_search",
        lambda strategy, **_kwargs: fake_result(strategy),
    )

    search_cli.main(
        [
            "--strategy",
            "enumerative_without_archive",
            "--candidate-count",
            "1",
            "--summary",
            str(summary_path),
            "--force",
        ]
    )

    assert json.loads(summary_path.read_text())["strategy"] == (
        "enumerative_without_archive"
    )


def test_both_cli_entry_points_show_the_same_strategy_choices():
    scripts_help = search_cli.build_parser().format_help()
    assert "genetic" in scripts_help
    assert "enumerative" in scripts_help
    assert "random" in scripts_help
    assert "enumerative_without_archive" in scripts_help
