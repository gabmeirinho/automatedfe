from types import SimpleNamespace

import pytest

import automatedfe.cli.tracking_ui as tracking_ui_cli
from automatedfe.tracking import launch_tracking_ui, tracking_ui_command


def test_tracking_ui_command_uses_exact_configured_store(mlflow_store):
    command = tracking_ui_command(mlflow_store, host="localhost", port=5050)

    assert command[:4] == (
        command[0],
        "-m",
        "mlflow",
        "ui",
    )
    assert command[command.index("--backend-store-uri") + 1] == (
        mlflow_store.tracking_uri
    )
    assert command[command.index("--default-artifact-root") + 1] == (
        mlflow_store.artifact_location
    )
    assert command[command.index("--host") + 1] == "localhost"
    assert command[command.index("--port") + 1] == "5050"


def test_launch_tracking_ui_returns_process_status(mlflow_store):
    calls = []

    def fake_runner(command, *, check):
        calls.append((tuple(command), check))
        return SimpleNamespace(returncode=7)

    assert (
        launch_tracking_ui(
            tracking_store=mlflow_store,
            runner=fake_runner,
        )
        == 7
    )
    assert calls[0][0][3] == "ui"
    assert calls[0][1] is False


def test_tracking_ui_cli_forwards_options(monkeypatch, tmp_path):
    calls = []

    def fake_launch(**kwargs):
        calls.append(kwargs)
        return 9

    monkeypatch.setattr(tracking_ui_cli, "launch_tracking_ui", fake_launch)
    root = tmp_path / "artifacts"

    assert (
        tracking_ui_cli.main(
            [
                "--tracking-uri",
                "sqlite:///ui.db",
                "--artifact-root",
                str(root),
                "--host",
                "localhost",
                "--port",
                "5051",
            ]
        )
        == 9
    )
    assert calls == [
        {
            "tracking_uri": "sqlite:///ui.db",
            "artifact_root": root,
            "host": "localhost",
            "port": 5051,
        }
    ]


def test_tracking_ui_cli_rejects_invalid_port():
    with pytest.raises(SystemExit) as raised:
        tracking_ui_cli.main(["--port", "70000"])
    assert raised.value.code == 2
