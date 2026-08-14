import pytest

import automatedfe.cli.report as report_cli
from automatedfe.analysis.run_services import RunReportError


def test_report_cli_forwards_run_and_label_mode(monkeypatch, capsys):
    calls = []

    def fake_report(run_id, **kwargs):
        calls.append((run_id, kwargs))
        return run_id

    monkeypatch.setattr(report_cli, "rerender_run_report", fake_report)

    assert (
        report_cli.main(["--run-id", "run-456", "--feature-labels", "expression"]) == 0
    )
    assert calls[0][0] == "run-456"
    assert calls[0][1]["feature_labels"] == "expression"
    assert "Report rerendered: run-456" in capsys.readouterr().out


def test_report_cli_returns_nonzero_and_names_run(monkeypatch, capsys):
    monkeypatch.setattr(
        report_cli,
        "rerender_run_report",
        lambda run_id, **_kwargs: (_ for _ in ()).throw(
            RunReportError(run_id, "corrupt artifact")
        ),
    )

    assert report_cli.main(["--run-id", "corrupt-run"]) == 1
    assert "corrupt-run" in capsys.readouterr().err


def test_report_cli_requires_run_id():
    with pytest.raises(SystemExit) as raised:
        report_cli.main([])
    assert raised.value.code == 2
