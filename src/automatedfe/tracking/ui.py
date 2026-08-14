"""Launch the MLflow UI against AutomatedFE's configured run store."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Sequence
from os import PathLike

from .mlflow_store import MlflowRunStore


def tracking_ui_command(
    store: MlflowRunStore,
    *,
    host: str = "127.0.0.1",
    port: int = 5000,
) -> tuple[str, ...]:
    """Build the UI command for the store's exact backend and artifact root."""

    if not isinstance(host, str) or not host.strip():
        raise ValueError("host must be a non-empty string")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("port must be an integer between 1 and 65535")
    return (
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        store.tracking_uri,
        "--default-artifact-root",
        store.artifact_location,
        "--no-serve-artifacts",
        "--host",
        host.strip(),
        "--port",
        str(port),
    )


def launch_tracking_ui(
    *,
    tracking_uri: str | None = None,
    artifact_root: str | PathLike[str] | None = None,
    host: str = "127.0.0.1",
    port: int = 5000,
    tracking_store: MlflowRunStore | None = None,
    runner: Callable[..., subprocess.CompletedProcess[object]] = subprocess.run,
) -> int:
    """Run the blocking MLflow UI process and return its exit code."""

    if tracking_store is not None and (
        tracking_uri is not None or artifact_root is not None
    ):
        raise ValueError(
            "tracking_store cannot be combined with tracking_uri or artifact_root"
        )
    store = tracking_store or MlflowRunStore(
        tracking_uri,
        artifact_root=artifact_root,
    )
    command: Sequence[str] = tracking_ui_command(store, host=host, port=port)
    completed = runner(command, check=False)
    return int(completed.returncode)


__all__ = ["launch_tracking_ui", "tracking_ui_command"]
