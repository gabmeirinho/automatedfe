"""Launch the local MLflow tracking UI."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from automatedfe.tracking import launch_tracking_ui


def _port(value: str) -> int:
    try:
        converted = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if not 1 <= converted <= 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return converted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch MLflow for the AutomatedFE tracking repository."
    )
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI")
    parser.add_argument(
        "--artifact-root", type=Path, default=None, help="MLflow artifact root"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind")
    parser.add_argument("--port", type=_port, default=5000, help="Port to bind")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return launch_tracking_ui(
            tracking_uri=args.tracking_uri,
            artifact_root=args.artifact_root,
            host=args.host,
            port=args.port,
        )
    except KeyboardInterrupt:
        return 130
    except (OSError, TypeError, ValueError) as error:
        parser._print_message(f"Could not launch MLflow tracking UI: {error}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
