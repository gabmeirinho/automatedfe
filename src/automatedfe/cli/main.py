"""Unified command-line dispatcher for :mod:`automatedfe`."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib import import_module
import sys
from types import ModuleType


_COMMANDS: dict[str, tuple[str, str]] = {
    "preprocess": (
        "automatedfe.cli.preprocess",
        "Run the full preprocessing pipeline",
    ),
    "sort-transactions": (
        "automatedfe.cli.sort_transactions",
        "Sort and enrich transactions",
    ),
    "sort-dataset": (
        "automatedfe.cli.sort_dataset",
        "Sort the event dataset",
    ),
    "encode": (
        "automatedfe.cli.encode",
        "Fit or apply a label mapping",
    ),
    "materialize": (
        "automatedfe.cli.materialize",
        "Materialize transformed transactions into mmap files",
    ),
    "search": (
        "automatedfe.cli.search",
        "Run a feature-search strategy",
    ),
    "validate": (
        "automatedfe.cli.validate",
        "Run a data validation or diagnostic check",
    ),
}


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    """Build the top-level parser used by the console command."""

    parser = argparse.ArgumentParser(
        prog=prog,
        description="Unified command line for automated feature engineering.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, title="commands")
    for name, (_, help_text) in _COMMANDS.items():
        subparsers.add_parser(name, help=help_text)
    return parser


def _module_for(command: str) -> ModuleType:
    return import_module(_COMMANDS[command][0])


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch *argv* to the selected package-owned command."""

    arguments = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser(prog="automatedfe")

    if not arguments or arguments[0].startswith("-"):
        # This handles top-level help, the missing-command error, and invalid
        # top-level options with argparse's standard exit semantics.
        parser.parse_args(arguments)
        return 2

    command = arguments[0]
    if command not in _COMMANDS:
        parser.parse_args(arguments)
        return 2

    module = _module_for(command)
    return int(module.main(arguments[1:]))


__all__ = ["build_parser", "main"]
