"""Rerender a completed tracked run report."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from automatedfe.analysis.run_services import RunServiceError, rerender_run_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rerender a completed MLflow run from persisted analysis tables."
    )
    parser.add_argument("--run-id", required=True, help="Immutable MLflow run ID")
    parser.add_argument(
        "--feature-labels",
        choices=("expression", "id"),
        default=None,
        help="Override the persisted report label mode",
    )
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI")
    parser.add_argument(
        "--artifact-root", type=Path, default=None, help="MLflow artifact root"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_id = rerender_run_report(
            args.run_id,
            feature_labels=args.feature_labels,
            tracking_uri=args.tracking_uri,
            artifact_root=args.artifact_root,
        )
    except KeyboardInterrupt:
        parser._print_message(f"Report rerender interrupted for run {args.run_id}\n")
        return 130
    except (RunServiceError, OSError, ValueError) as error:
        parser._print_message(f"{error}\n")
        return 1
    except Exception as error:  # noqa: BLE001 - command boundary
        parser._print_message(
            f"Report rerender failed for run {args.run_id}: {error}\n"
        )
        return 1
    print(f"Report rerendered: {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
