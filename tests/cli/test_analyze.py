import pytest

import automatedfe.cli.analyze as analyze_cli
from automatedfe.analysis.run_services import RunAnalysisError


def test_analyze_cli_forwards_run_and_store_options(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_analyze(run_id, **kwargs):
        calls.append((run_id, kwargs))
        return run_id

    monkeypatch.setattr(analyze_cli, "analyze_run", fake_analyze)
    artifact_root = tmp_path / "artifacts"

    assert (
        analyze_cli.main(
            [
                "--run-id",
                "run-123",
                "--feature-labels",
                "id",
                "--tracking-uri",
                "sqlite:///runs.db",
                "--artifact-root",
                str(artifact_root),
            ]
        )
        == 0
    )
    assert calls == [
        (
            "run-123",
            {
                "feature_labels": "id",
                "tracking_uri": "sqlite:///runs.db",
                "artifact_root": artifact_root,
            },
        )
    ]
    assert "Analysis complete: run-123" in capsys.readouterr().out


def test_analyze_cli_returns_nonzero_and_names_run(monkeypatch, capsys):
    monkeypatch.setattr(
        analyze_cli,
        "analyze_run",
        lambda run_id, **_kwargs: (_ for _ in ()).throw(
            RunAnalysisError(run_id, "render failed")
        ),
    )

    assert analyze_cli.main(["--run-id", "broken-run"]) == 1
    assert "broken-run" in capsys.readouterr().err


def test_analyze_cli_requires_run_id():
    with pytest.raises(SystemExit) as raised:
        analyze_cli.main([])
    assert raised.value.code == 2
